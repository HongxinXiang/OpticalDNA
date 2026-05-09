# MIT License modified from prithivMLmods/OpticalDNA-Latest-BF16.I64
import os
import math
import re
from tqdm import tqdm
from abc import ABC
from typing import List, Optional, Tuple, Union

from addict import Dict
from PIL import Image, ImageOps, ImageDraw, ImageFont
import numpy as np

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torchvision import transforms

from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers import DeepseekV2Model, DeepseekV2ForCausalLM
from transformers import DeepseekV2Config
from transformers.models.deepseek_v2.modeling_deepseek_v2 import (
    DeepseekV2Attention, DeepseekV2MLP, DeepseekV2MoE, DeepseekV2RMSNorm, DeepseekV2DecoderLayer)
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding
from transformers import TextStreamer
from .deepencoder import build_sam_vit_b, build_clip_l, MlpProjector
from .conversation import get_conv_template
from .multi_page_fusion import MultiPageSelfAttentionFusion


torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

def load_image(image_path):

    try:
        image = Image.open(image_path)
        
        corrected_image = ImageOps.exif_transpose(image)
        
        return corrected_image
        
    except Exception as e:
        print(f"error: {e}")
        try:
            return Image.open(image_path)
        except:
            return None


def re_match(text):
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)

    # pattern1 = r'<\|ref\|>.*?<\|/ref\|>\n'
    # new_text1 = re.sub(pattern1, '', text, flags=re.DOTALL)

    mathes_image = []
    mathes_other = []
    for a_match in matches:
        if '<|ref|>image<|/ref|>' in a_match[0]:
            mathes_image.append(a_match[0])
        else:
            mathes_other.append(a_match[0])
    return matches, mathes_image, mathes_other


def extract_coordinates_and_label(ref_text, image_width, image_height):

    try:
        label_type = ref_text[1]
        cor_list = eval(ref_text[2])
    except Exception as e:
        print(e)
        return None

    return (label_type, cor_list)


def draw_bounding_boxes(image, refs, ouput_path):

    image_width, image_height = image.size
    
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)
    
    # try:
    # except IOError:
    #     try:
    #         font = ImageFont.truetype("DejaVuSans.ttf", 20) 
    #     except IOError:
    font = ImageFont.load_default()

    img_idx = 0
    
    for i, ref in enumerate(refs):
        try:
            result = extract_coordinates_and_label(ref, image_width, image_height)
            if result:
                label_type, points_list = result
                
                color = (np.random.randint(0, 200), np.random.randint(0, 200), np.random.randint(0, 255))

                color_a = color + (20, )
                for points in points_list:
                    try:
                        x1, y1, x2, y2 = points
                    except:  #NOTE: 
                        _, x1, y1, x2, y2 = points

                    # x1 = int(x1 / 999 * image_width)
                    # y1 = int(y1 / 999 * image_height)
                    #
                    # x2 = int(x2 / 999 * image_width)
                    # y2 = int(y2 / 999 * image_height)

                    if label_type == 'image':
                        try:
                            cropped = image.crop((x1, y1, x2, y2))
                            cropped.save(f"{ouput_path}/images/{img_idx}.jpg")
                        except Exception as e:
                            print(e)
                            pass
                        img_idx += 1
                        
                    try:
                        if label_type == 'title':
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                        else:
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                        text_x = x1
                        text_y = max(0, y1 - 15)
                            
                        
                        text_bbox = draw.textbbox((0, 0), label_type, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height], 
                                    fill=(255, 255, 255, 30))
                        
                        draw.text((text_x, text_y), label_type, font=font, fill=color)
                    except:
                        pass
        except:
            continue
    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw


def process_image_with_refs(image, ref_texts, output_path):

    result_image = draw_bounding_boxes(image, ref_texts, output_path)
    
    return result_image





def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    # print(f'width: {width}, height: {height}, best_ratio: {best_ratio}')
    return best_ratio


def dynamic_preprocess(image, min_num=2, max_num=9, image_size=640, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    # print(target_ratios)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # print(target_aspect_ratio)
    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images, target_aspect_ratio



def normalize_transform(mean, std):
    if mean is None and std is None:
        transform = None
    elif mean is None and std is not None:
        mean = [0.] * len(std)
        transform = transforms.Normalize(mean=mean, std=std)
    elif mean is not None and std is None:
        std = [1.] * len(mean)
        transform = transforms.Normalize(mean=mean, std=std)
    else:
        transform = transforms.Normalize(mean=mean, std=std)

    return transform



def format_messages(
        conversations: List[Dict[str, str]],
        sft_format: str = "opticaldna",
        system_prompt: str = "",
):
    """
    Applies the SFT template to conversation.

    Args:
        conversations (List[Dict]): A List of messages.
        sft_format (str, optional): The format of the SFT template to use. Defaults to "opticaldna".
        system_prompt (str, optional): The system prompt to use in the SFT template. Defaults to "".

    Returns:
        sft_prompt (str): The formatted text.
    """

    conv = get_conv_template(sft_format)
    conv.set_system_message(system_prompt)
    for message in conversations:
        conv.append_message(message["role"], message["content"].strip())
    sft_prompt = conv.get_prompt().strip()

    return sft_prompt


def text_encode(tokenizer, text: str, bos: bool = True, eos: bool = False):
    t = tokenizer.encode(text, add_special_tokens=False)
    bos_id = 0
    eos_id = 1
    if bos:
        t = [bos_id] + t
    if eos:
        t = t + [eos_id]

    return t

def load_pil_images(conversations: List[Dict[str, str]]) -> List[Image.Image]:
    """
    Load PIL images from conversations.

    Supported formats for message["images"]:

        1) ["a.png", "b.png"]
           -> [[PIL(a)], [PIL(b)]]

        2) [["a.png", "b.png"], ["c.png", "d.png"]]
           -> [[PIL(a), PIL(b)], [PIL(c), PIL(d)]]

    Args:
        conversations (List[Dict[str, str]]): the conversations with a list of messages. An example is :
            [
                {
                    "role": "User",
                    "content": "<image_placeholder>\nExtract all information from this image and convert them into markdown format.",
                    "images": ["./examples/table_datasets.png"] | [["./examples/table_datasets_1.png", "./examples/table_datasets_2.png", ...]]
                },
                {"role": "Assistant", "content": ""},
            ]

    Returns:
        pil_images : List[List[PIL.Image.Image]]
            A list of pages, each page is a list of PIL images.

    """

    pil_images: List[List[Image.Image]] = []

    for message in conversations:
        if "images" not in message:
            continue

        images = message["images"]

        # ------------------------------------------------------------
        # Case 1: ["a.png", "b.png"]
        # → treat each image as a single-page group
        # ------------------------------------------------------------
        if isinstance(images, list) and len(images) > 0 and isinstance(images[0], str):
            for image_path in images:
                pil_img = load_image(image_path).convert("RGB")
                pil_images.append([pil_img])  #  [pil_img]

        # ------------------------------------------------------------
        # Case 2: [["a.png", "b.png"], ["c.png", "d.png"]]
        # → each sub-list is a page
        # ------------------------------------------------------------
        elif isinstance(images, list) and len(images) > 0 and isinstance(images[0], list):
            for page in images:
                page_images: List[Image.Image] = []
                for image_path in page:
                    pil_img = load_image(image_path).convert("RGB")
                    page_images.append(pil_img)
                pil_images.append(page_images)

        else:
            raise TypeError(
                f"Unsupported image format: type(images)={type(images)}, value={images}"
            )

    return pil_images



class BaseTransform(ABC):

    def set_rng(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        pass

    @property
    def default_shape(self):
        raise NotImplementedError


class BasicImageTransform(BaseTransform):
    def __init__(
        self, 
        mean: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
        std: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
        normalize: bool = True
    ):
        self.mean = mean
        self.std = std
    
        transform_pipelines = [
            transforms.ToTensor()
        ]

        normalize = normalize_transform(mean, std) if normalize else nn.Identity()
        if normalize is not None:
            transform_pipelines.append(normalize)

        self.transform = transforms.Compose(transform_pipelines)
    
    def __call__(self, x):
        x = self.transform(x)
        return x

class NoEOSTextStreamer(TextStreamer):
    def on_finalized_text(self, text: str, stream_end: bool = False):

        eos_text = self.tokenizer.decode([self.tokenizer.eos_token_id], skip_special_tokens=False)
        text = text.replace(eos_text, "\n")
        print(text, flush=True, end="")


def decoder_layer_init(self, config: DeepseekV2Config, layer_idx: int):
    nn.Module.__init__(self)
    self.hidden_size = config.hidden_size

    if config.use_mla:
        self.self_attn = DeepseekV2Attention(config=config, layer_idx=layer_idx)
    else:
        config.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attn = LlamaAttention(config, layer_idx)
    self.mlp = DeepseekV2MoE(config) if layer_idx >= config.first_k_dense_replace else DeepseekV2MLP(config)

    self.input_layernorm = DeepseekV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    self.post_attention_layernorm = DeepseekV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


DeepseekV2DecoderLayer.__init__ = decoder_layer_init

class DeepseekOCRConfig(DeepseekV2Config):
    model_type = "DeepseekOCR"

class DeepseekOCRModel(DeepseekV2Model):
    config_class = DeepseekOCRConfig

    def __init__(self, config: DeepseekV2Config):
        super(DeepseekOCRModel, self).__init__(config)

        self.sam_model = build_sam_vit_b()
        self.vision_model = build_clip_l()
        # self.conv_2 = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=2, stride=2)
        n_embed = 1280
        self.projector =  MlpProjector(Dict(projector_type="linear", input_dim=2048, n_embed=n_embed))
        embed_std = 1 / torch.sqrt(torch.tensor(n_embed, dtype=torch.float32))
        self.image_newline = nn.Parameter(torch.randn(n_embed) * embed_std)
        self.view_seperator = nn.Parameter(torch.randn(n_embed) * embed_std)
        self.page_fusion_layer = MultiPageSelfAttentionFusion(
            d_model=n_embed,
            num_heads=20,
            proj_in=False,
            proj_out=False,
            reduce="mean",
            num_attn_layers=1,
        )
        # print("page_fusion_layer param num =",
        #       sum(p.numel() for p in self.page_fusion_layer.parameters()))
        # keys = [k for k in self.state_dict().keys() if "page_fusion_layer" in k]
        # print("state_dict keys:", keys[:50], " ... total =", len(keys))

        self.rotary_emb = LlamaRotaryEmbedding(config=config)

    def ensure_page_dim_patches_ori(
            self,
            patches: torch.Tensor,
            image_ori: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
         patches/image_ori:
            patches   : [n_image, n_page, P, C, H, W]
            image_ori : [n_image, n_page, C, H, W]
        """
        # patches: [n_page, P, C, H, W] -> [1, n_page, P, C, H, W]
        if patches.dim() == 5:
            assert image_ori.shape[0] == 1
            patches = patches.unsqueeze(0)
        elif patches.dim() != 6:
            raise ValueError(f"patches must be 5D or 6D, got {patches.shape}")

        n_image, n_page, P, C, H, W = patches.shape

        # image_ori: [n_image, n_page, C, H, W]
        if image_ori.dim() != 5:
            raise ValueError(f"image_ori must be 5D, got {image_ori.shape}")

        if image_ori.shape[1] != n_page:
            raise ValueError(
                f"n_page mismatch: patches n_page={n_page}, image_ori n_page={image_ori.shape[1]}"
            )

        return patches, image_ori

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        images_seq_mask: Optional[torch.FloatTensor] = None,
        images_spatial_crop: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # print("xhxhxh...", len(images), len(images[0]))


        if inputs_embeds is None:
            # inputs_embeds = self.embed_tokens(input_ids)
            inputs_embeds = self.get_input_embeddings()(input_ids)

        inputs_embeds = inputs_embeds.clone()

        sam_model = getattr(self, 'sam_model', None)
        # sam_model = self.sam_model
        vision_model = getattr(self, 'vision_model', None)



        if sam_model is not None and (input_ids.shape[1] != 1 or self.training) and torch.sum(images[0][1]).item() != 0:

            idx = 0
            
            # sam_model = torch.jit.script(sam_model)
            
            # start_time = time.time()
            for image, crop_shape in zip(images, images_spatial_crop):
                images_in_this_batch = []

                patches = image[0]  # (n_page, P, C, H, W): (6, 4, 3, 640, 640)
                image_ori = image[1]  # (n_img, n_page, C, H, W): (1, 6, 3, 1280, 1280)

                # ==========================================================
                # patches:   [n_image, n_page, P, C, H, W]
                # image_ori: [n_image, n_page, C, H, W]
                # ==========================================================
                patches, image_ori = self.ensure_page_dim_patches_ori(patches, image_ori)

                n_img_p, n_page_p, P_p, C_p, H_p, W_p = patches.shape
                n_img_o, n_page_o, C_o, H_o, W_o = image_ori.shape

                # ======================================================
                # ======================================================
                patches_f = patches.view(n_img_p * n_page_p * P_p, C_p, H_p, W_p)
                image_ori_f = image_ori.view(n_img_o * n_page_o, C_o, H_o, W_o)

                if True:
                # with torch.no_grad():
                # with torch.inference_mode():

                    if torch.sum(patches).item() != 0:
                        # P, C, H, W = patches.shape
                        crop_flag = 1
                        # local_features_1 = sam_model(patches)
                        # local_features_2 = vision_model(patches, local_features_1)
                        # ======================================================
                        # ======================================================
                        local_features_1 = sam_model(patches_f)  # (n_img_p*n_page_p*P, d, 10, 10): (24, 1024, 10, 10)
                        local_features_2 = vision_model(patches_f, local_features_1)  # (n_img_p*n_page_p*P, T+1, d): (24, 101, 1024)

                        # vit_time = time.time()
                        local_features_page = torch.cat((
                            local_features_2[:, 1:], local_features_1.flatten(2).permute(0, 2, 1)),
                            dim=-1)  # (n_img_p*n_page_p*P, T, d*2): (24, 100, 2048)
                        local_features_page = self.projector(local_features_page)  # (n_img_p*n_page_p*P, T, d): (24, 100, 1280)
                        feat_dim_local = list(local_features_page.shape[1:])  # (T, d): (100, 1280)
                        local_features_page_permute = local_features_page.view(*([n_img_p, n_page_p, P_p] + feat_dim_local)).permute(0, 2, 1, 3, 4)  # (n_img_p, P, n_page_p, T, d): (1, 4, 6, 100, 1280)
                        local_features = self.page_fusion_layer(local_features_page_permute.view(*([n_img_p * P_p, n_page_p] + feat_dim_local)))

                        # ======================================================
                        # ======================================================
                        global_features_1 = sam_model(image_ori_f)  # (n_img_o*n_page_o, d, 20, 20): (6, 1024, 20, 20)
                        global_features_2 = vision_model(image_ori_f, global_features_1)  # (n_img_o*n_page_o, T+1, d): (6, 401, 1024)
                        global_features_page = torch.cat((
                            global_features_2[:, 1:], global_features_1.flatten(2).permute(0, 2, 1)),
                            dim=-1
                        )  # (n_img_o*n_page_o, T, d*2): (6, 400, 2048)
                        global_features_page = self.projector(global_features_page)  # (n_img_o*n_page_o, T, d): (6, 400, 1280)
                        # global_features.view(*([n_img_o, n_page_o] + list(global_features.shape[1:])))  # in: (n_img_o, n_page_o, 100, 1280); out: (n_img_p, T, d): (24, 100, 1280)
                        feat_dim_global = list(global_features_page.shape[1:])
                        global_features = self.page_fusion_layer(global_features_page.view(*([n_img_o, n_page_o] + feat_dim_global))).view(*([n_img_o]+feat_dim_global))

                        print('=====================')
                        print('BASE: ', global_features.shape)
                        print('PATCHES: ', local_features.shape)
                        print('=====================')

                        _, hw, n_dim = global_features.shape
                        h = w = int(hw ** 0.5)

                        _2, hw2, n_dim2 = local_features.shape
                        h2 = w2 = int(hw2 ** 0.5)

                        width_crop_num, height_crop_num = crop_shape[0], crop_shape[1]

                        global_features = global_features.view(h, w, n_dim)

                        global_features = torch.cat(
                            [global_features, self.image_newline[None, None, :].expand(h, 1, n_dim)], dim=1
                        )

                        global_features = global_features.view(-1, n_dim)


                        local_features = local_features.view(height_crop_num, width_crop_num, h2, w2, n_dim2).permute(0, 2, 1, 3, 4).reshape(height_crop_num*h2, width_crop_num*w2, n_dim2)
                        local_features = torch.cat(
                            [local_features, self.image_newline[None, None, :].expand(height_crop_num * h2, 1, n_dim2)], dim=1
                        )
                        local_features = local_features.view(-1, n_dim2)

                        global_local_features = torch.cat([local_features, global_features, self.view_seperator[None, :]], dim=0)

                        # end_time = time.time()

                        # print('sam: ', sam_time - start_time)
                        # print('vit: ', vit_time - sam_time)
                        # print('all: ', end_time - start_time)

                        # exit()

                    else:
                        # ======================================================
                        # ======================================================
                        global_features_1 = sam_model(image_ori_f)
                        global_features_2 = vision_model(image_ori_f, global_features_1)
                        global_features_page = torch.cat((
                            global_features_2[:, 1:], global_features_1.flatten(2).permute(0, 2, 1)),
                            dim=-1
                        )  # # (n_img_o*n_page_o, T, d*2): (6, 400, 2048)
                        global_features_page = self.projector(
                            global_features_page)  # (n_img_o*n_page_o, T, d): (6, 400, 1280)
                        feat_dim_global = list(global_features_page.shape[1:])
                        global_features = self.page_fusion_layer(
                            global_features_page.view(*([n_img_o, n_page_o] + feat_dim_global))).view(
                            *([n_img_o] + feat_dim_global))

                        # print('=====================')
                        # print('BASE: ', global_features.shape)
                        # print('NO PATCHES')
                        # print('=====================')
                        _, hw, n_dim = global_features.shape
                        h = w = int(hw ** 0.5)


                        global_features = global_features.view(h, w, n_dim)

                        global_features = torch.cat(
                            [global_features, self.image_newline[None, None, :].expand(h, 1, n_dim)], dim=1
                        )

                        global_features = global_features.view(-1, n_dim)

                        global_local_features = torch.cat([global_features, self.view_seperator[None, :]], dim=0)

                    images_in_this_batch.append(global_local_features)
                

                if images_in_this_batch:
                    images_in_this_batch = torch.cat(images_in_this_batch, dim=0)
                    images_in_this_batch = images_in_this_batch.to(
                        device=inputs_embeds.device, dtype=inputs_embeds.dtype
                    )
                    mask = images_seq_mask[idx].unsqueeze(-1).to(inputs_embeds.device)   # bool [T, 1]
                    # torch.save({
                    #     'inputs_embeds': inputs_embeds.detach().cpu(),
                    #     'mask': mask.detach().cpu(),
                    #     'images_in_this_batch': images_in_this_batch.detach().cpu()
                    # }, f"{idx}.pt")
                    updated_row = inputs_embeds[idx].masked_scatter(mask, images_in_this_batch)
                    inputs_embeds[idx] = updated_row

                idx += 1
            
        return super(DeepseekOCRModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache, position_ids = position_ids,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
    

class DeepseekOCRForCausalLM(DeepseekV2ForCausalLM):

    config_class = DeepseekOCRConfig
    # supports_gradient_checkpointing = True

    def __init__(self, config):
        super(DeepseekV2ForCausalLM, self).__init__(config)
        self.model = DeepseekOCRModel(config)

        self.vocab_size = config.vocab_size

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        images_seq_mask: Optional[torch.FloatTensor] = None,
        images_spatial_crop: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict



        outputs  = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            images=images,
            images_seq_mask = images_seq_mask,
            images_spatial_crop = images_spatial_crop,
            return_dict=return_dict
            
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()  # torch.Size([8, 1381, 129280]), (b, n_seq, n_classes)

        # logits

        loss = None
        if labels is not None:  # torch.Size([8, 1126]): (b, n_seq)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )


    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # Omit tokens covered by past_key_values
        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.get_seq_length()
                max_cache_length = None
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if self.generation_config.cache_implementation == "static":
        #     # generation with static cache
        #     cache_position = kwargs.get("cache_position", None)
        #     if cache_position is None:
        #         past_length = 0
        #     else:
        #         past_length = cache_position[-1] + 1
        #     input_ids = input_ids[:, past_length:]
        #     position_ids = position_ids[:, past_length:]

        # TODO @gante we should only keep a `cache_position` in generate, and do +=1.
        # same goes for position ids. Could also help with continued generation.
        cache_position = torch.arange(past_length, past_length + position_ids.shape[-1], device=position_ids.device)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
                "images_seq_mask": kwargs.get("images_seq_mask", None),
                "images_spatial_crop": kwargs.get("images_spatial_crop", None),
            }
        )
        return model_inputs
    

    def disable_torch_init(self):
        """
        Disable the redundant torch default initialization to accelerate model creation.
        """
        import torch
        setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
        setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)



    def infer(self, tokenizer, prompt='', image_file='', output_path = '', base_size=1024, image_size=640, crop_mode=True, test_compress=False, save_results=False, eval_mode=False):
        self.disable_torch_init()

        os.makedirs(output_path, exist_ok=True)
        os.makedirs(f'{output_path}/images', exist_ok=True)

        if prompt and image_file:
            conversation = [
                {
                    "role": "<|User|>",
                    # "content": "<image>\n<|grounding|>Given the layout of the image. ",
                    "content": f'{prompt}',
                    # "content": "<image>\nFree OCR. ",
                    # "content": "<image>\nParse the figure. ",
                    # "content": "<image>\nExtract the text in the image. ",
                    "images": image_file, # [['1.png']],
                },
                {"role": "<|Assistant|>", "content": ""},
            ]
        
        elif prompt:
            conversation = [
                {
                    "role": "<|User|>",
                    # "content": "<image>\n<|grounding|>Given the layout of the image. ",
                    "content": f'{prompt}',
                    # "content": "<image>\nFree OCR. ",
                    # "content": "<image>\nParse the figure. ",
                    # "content": "<image>\nExtract the text in the image. ",
                    # "images": [f'{image_file}'],
                },
                {"role": "<|Assistant|>", "content": ""},
            ]
        else:
            assert False, f'prompt is none!'
        
        prompt = format_messages(conversations=conversation, sft_format='plain', system_prompt='')

        patch_size = 16
        downsample_ratio = 4
        images = load_pil_images(conversation)
        # images = [[item.resize((1280, 1280)) for item in images[0]]]

        valid_img_tokens = 0
        ratio = 1

        image_draw = images[0].copy()
        assert isinstance(image_draw, list)
        assert len(set([item.size for item in image_draw])) == 1, ""

        w, h = image_draw[0].size
        # print(w, h)
        ratio = 1 - ((max(w, h) - min(w, h)) / (max(w, h)))
    

        image_transform = BasicImageTransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True)
        images_seq_mask = []

        image_token = '<image>'
        image_token_id = 128815
        text_splits = prompt.split(image_token)

        images_list, images_crop_list, images_seq_mask = [], [], []
        tokenized_str = []
        images_spatial_crop = []
        for text_sep, page_list in zip(text_splits, images):

            tokenized_sep = text_encode(tokenizer, text_sep, bos=False, eos=False)
            tokenized_str += tokenized_sep
            images_seq_mask += [False] * len(tokenized_sep)

            if crop_mode:

                if page_list[0].size[0] <= 640 and page_list[0].size[1] <= 640:
                    crop_ratio = [1, 1]

                else:
                    if crop_mode:
                        # best_width, best_height = select_best_resolution(image.size, self.candidate_resolutions)
                        images_crop_raw_list, crop_ratio_list = [], []
                        for image in page_list:
                            images_crop_raw, crop_ratio = dynamic_preprocess(image)
                            images_crop_raw_list.append(images_crop_raw)
                            crop_ratio_list.append(crop_ratio)
                    else:
                        # best_width, best_height = self.image_size, self.image_size
                        crop_ratio = [1, 1]
                
                """process the global view"""
                # image = image.resize((base_size, base_size))
                global_view_list = [ImageOps.pad(image, (base_size, base_size),
                                        color=tuple(int(x * 255) for x in image_transform.mean)) for image in page_list]
                
                if base_size == 1024:
                    valid_img_tokens += int(256 * ratio)
                elif base_size == 1280:
                    valid_img_tokens += int(400 * ratio)
                # elif base_size == 640:
                #     valid_img_tokens += int(100 * ratio)
                



                images_list.append(torch.stack([image_transform(item).to(torch_dtype) for item in global_view_list]))

                # global_view_tensor = image_transform(global_view).to(torch_dtype)

                width_crop_num, height_crop_num = crop_ratio

                images_spatial_crop.append([width_crop_num, height_crop_num])
                
                
                if width_crop_num > 1 or height_crop_num > 1:
                    """process the local views"""
                    
                    for images_crop_raw in images_crop_raw_list:
                        images_crop_list.append([image_transform(item).to(torch_dtype) for item in images_crop_raw])

                if image_size == 640:
                    valid_img_tokens += len(images_crop_list[0]) * 100

                num_queries = math.ceil((image_size // patch_size) / downsample_ratio)
                num_queries_base = math.ceil((base_size // patch_size) / downsample_ratio)



                """add image tokens"""

                

                tokenized_image = ([image_token_id] * num_queries_base + [image_token_id]) * num_queries_base
                tokenized_image += [image_token_id]
                if width_crop_num > 1 or height_crop_num > 1:
                    tokenized_image += ([image_token_id] * (num_queries * width_crop_num) + [image_token_id]) * (
                                num_queries * height_crop_num)
                tokenized_str += tokenized_image
                images_seq_mask += [True] * len(tokenized_image)
                # num_image_tokens.append(len(tokenized_image))

            else:
                # best_width, best_height = self.image_size, self.image_size
                # print(image.size, (best_width, best_height)) # check the select_best_resolutions func

                """process the global view"""
                if image_size <= 640:
                    # print('directly resize')
                    for i, image in enumerate(page_list):
                        page_list[i] = image.resize((image_size, image_size))
                # else:
                global_view_list = [ImageOps.pad(image, (image_size, image_size),
                                        color=tuple(int(x * 255) for x in image_transform.mean)) for image in page_list]
                images_list.append(torch.stack([image_transform(item).to(torch_dtype) for item in global_view_list]))

                if base_size == 1024:
                    valid_img_tokens += int(256 * ratio)
                elif base_size == 1280:
                    valid_img_tokens += int(400 * ratio)
                elif base_size == 640:
                    valid_img_tokens += int(100 * 1)
                elif base_size == 512:
                    valid_img_tokens += int(64 * 1)

                width_crop_num, height_crop_num = 1, 1

                images_spatial_crop.append([width_crop_num, height_crop_num])


                """add image tokens"""
                num_queries = math.ceil((image_size // patch_size) / downsample_ratio)

                tokenized_image = ([image_token_id] * num_queries + [image_token_id]) * num_queries
                tokenized_image += [image_token_id]
                # tokenized_image += ([self.image_token_id] * (num_queries * width_crop_num) + [self.image_token_id]) * (
                #             num_queries * height_crop_num)
                tokenized_str += tokenized_image
                images_seq_mask += [True] * len(tokenized_image)
                # num_image_tokens.append(len(tokenized_image))
        

        """process the last text split"""
        tokenized_sep = text_encode(tokenizer, text_splits[-1], bos=False, eos=False)
        tokenized_str += tokenized_sep
        images_seq_mask += [False] * len(tokenized_sep)

        """add the bos tokens"""
        bos_id = 0
        tokenized_str = [bos_id] + tokenized_str 
        images_seq_mask = [False] + images_seq_mask



        input_ids = torch.LongTensor(tokenized_str)

        images_seq_mask = torch.tensor(images_seq_mask, dtype=torch.bool)

        if len(images_list) == 0:
            images_ori = torch.zeros((1, 1, 3, image_size, image_size))  # (n_img, n_page, C, H, W)
            images_spatial_crop = torch.zeros((1, 2), dtype=torch.long)
            images_crop = torch.zeros((1, 1, 3, base_size, base_size))  # (n_page, n_crop, C, H, W)

        else:
            images_ori = torch.stack(images_list, dim=0)
            images_spatial_crop = torch.tensor(images_spatial_crop, dtype=torch.long)
            if images_crop_list:
                images_crop = torch.stack([torch.stack(item, dim=0) for item in images_crop_list], 0)  # (n_page, n_crop, C, H, W),  base_size = 1280, image_size = 1280, crop_mode = True, (4, 3, 640, 640)
            else:
                images_crop = torch.zeros((images_ori.shape[1], 1, 3, base_size, base_size))  # (n_page, n_crop, C, H, W)



        if not eval_mode:
            streamer = NoEOSTextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
            with torch.autocast("cuda", dtype=torch_dtype):
                with torch.no_grad():
                    output_ids = self.generate(
                        input_ids.unsqueeze(0).cuda(),  # (1, T): (1, 2071)
                        images=[(images_crop.cuda(), images_ori.cuda())],  # ([n_page, n_crop, C, H, W], [n_img, n_page, C, H, W]): [(6, 4, 3, 640, 640), (1, 6, 3, 1280, 1280)]
                        images_seq_mask = images_seq_mask.unsqueeze(0).cuda(),  # (1, 2071)
                        images_spatial_crop = images_spatial_crop,  # (1, 2)
                        # do_sample=False,
                        # num_beams = 1,
                        temperature=0.0,
                        eos_token_id=tokenizer.eos_token_id,
                        streamer=streamer,
                        max_new_tokens=8192,
                        no_repeat_ngram_size = 20,
                        use_cache = True
                        )

        else:
            with torch.autocast("cuda", dtype=torch_dtype):
                with torch.no_grad():
                    output_ids = self.generate(
                        input_ids.unsqueeze(0).cuda(),
                        images=[(images_crop.cuda(), images_ori.cuda())],
                        images_seq_mask = images_seq_mask.unsqueeze(0).cuda(),
                        images_spatial_crop = images_spatial_crop,
                        # do_sample=False,
                        # num_beams = 1,
                        temperature=0.0,
                        eos_token_id=tokenizer.eos_token_id,
                        max_new_tokens=8192,
                        no_repeat_ngram_size = 35,
                        use_cache = True
                        )
                

        if '<image>' in conversation[0]['content'] and eval_mode:
                outputs = tokenizer.decode(output_ids[0, input_ids.unsqueeze(0).cuda().shape[1]:])
                stop_str = '<｜end▁of▁sentence｜>'
                if outputs.endswith(stop_str):
                    outputs = outputs[:-len(stop_str)]
                # re_match
                outputs = outputs.strip()

                return outputs
        
        if '<image>' in conversation[0]['content'] and test_compress:
            outputs = tokenizer.decode(output_ids[0, input_ids.unsqueeze(0).cuda().shape[1]:])
            pure_texts_outputs_token_length = len(text_encode(tokenizer, outputs, bos=False, eos=False))
            print('='*50)
            print('image size: ', (w, h))
            print('valid image tokens: ', int(valid_img_tokens))
            print('output texts tokens (valid): ', pure_texts_outputs_token_length)
            print('compression ratio: ', round(pure_texts_outputs_token_length/valid_img_tokens, 2))
            print('='*50)


        if '<image>' in conversation[0]['content'] and save_results:
            outputs = tokenizer.decode(output_ids[0, input_ids.unsqueeze(0).cuda().shape[1]:])
            stop_str = '<｜end▁of▁sentence｜>'

            print('='*15 + 'save results:' + '='*15)
            
            # # # # conv.messages[-1][-1] = outputs
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
            ret_outputs = str(outputs)

            matches_ref, matches_images, mathes_other = re_match(outputs)
            # print(matches_ref)

            # ============================================================
            # Multi-image box rendering support (minimal, backward compatible)
            # - Old behavior (single image): identical to previous version.
            # - New behavior (multiple images): route boxes to image n by the
            #   first index in each box: [n, x1, y1, x2, y2]
            # ============================================================
            def _filter_refs_for_image(refs, image_index: int):
                """Keep only boxes that belong to image_index, and strip the
                leading image index so downstream drawing code remains unchanged.
                Supports both:
                  - [x1,y1,x2,y2]  (treated as image 0)
                  - [img_idx,x1,y1,x2,y2]
                """
                filtered = []
                for ref in refs:
                    try:
                        label_type = ref[1]
                        points_list = eval(ref[2])
                    except Exception:
                        continue

                    new_points = []
                    for p in points_list:
                        # single-image legacy
                        if isinstance(p, (list, tuple)) and len(p) == 4:
                            if image_index == 0:
                                new_points.append(list(p))
                        # multi-image: [img_idx, x1, y1, x2, y2]
                        elif isinstance(p, (list, tuple)) and len(p) == 5:
                            if int(p[0]) == int(image_index):
                                new_points.append([p[1], p[2], p[3], p[4]])
                    if new_points:
                        # keep tuple structure expected by extract_coordinates_and_label
                        filtered.append((ref[0], label_type, repr(new_points)))
                return filtered

            if len(image_draw) == 1:
                # old single-image behavior (fully reproducible)
                result = process_image_with_refs(image_draw[0], matches_ref, output_path)
            else:
                # multi-image rendering: draw per image
                rendered = []
                for img_i, img in enumerate(image_draw):
                    refs_i = _filter_refs_for_image(matches_ref, img_i)
                    rendered_i = process_image_with_refs(img, refs_i, output_path)
                    rendered.append(rendered_i)
                    rendered_i.save(f"{output_path}/result_with_boxes_page{img_i}.jpg")

                # stitch vertically (top-to-bottom) to match page order
                max_w = max(im.size[0] for im in rendered) if rendered else 0
                total_h = sum(im.size[1] for im in rendered) if rendered else 0
                stitched = Image.new('RGB', (max_w, total_h), color=(255, 255, 255))
                y = 0
                for im in rendered:
                    stitched.paste(im, (0, y))
                    y += im.size[1]
                result = stitched

            for idx, a_match_image in enumerate(tqdm(matches_images, desc="image")):
                outputs = outputs.replace(a_match_image, '![](images/' + str(idx) + '.jpg)\n')
            
            for idx, a_match_other in enumerate(tqdm(mathes_other, desc="other")):
                outputs = outputs.replace(a_match_other, '').replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')


            # if 'structural formula' in conversation[0]['content']:
            #     outputs = '<smiles>' + outputs + '</smiles>'
            with open(f'{output_path}/result.mmd', 'w', encoding = 'utf-8') as afile:
                afile.write(outputs)

            if 'line_type' in outputs:
                import matplotlib.pyplot as plt
                lines = eval(outputs)['Line']['line']

                line_type = eval(outputs)['Line']['line_type']
                # print(lines)

                endpoints = eval(outputs)['Line']['line_endpoint']

                fig, ax = plt.subplots(figsize=(3,3), dpi=200)
                ax.set_xlim(-15, 15)
                ax.set_ylim(-15, 15)

                for idx, line in enumerate(lines):
                    try:
                        p0 = eval(line.split(' -- ')[0])
                        p1 = eval(line.split(' -- ')[-1])

                        if line_type[idx] == '--':
                            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth=0.8, color='k')
                        else:
                            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth = 0.8, color = 'k')

                        ax.scatter(p0[0], p0[1], s=5, color = 'k')
                        ax.scatter(p1[0], p1[1], s=5, color = 'k')
                    except:
                        pass

                for endpoint in endpoints:

                    label = endpoint.split(': ')[0]
                    (x, y) = eval(endpoint.split(': ')[1])
                    ax.annotate(label, (x, y), xytext=(1, 1), textcoords='offset points', 
                                fontsize=5, fontweight='light')
                

                plt.savefig(f'{output_path}/geo.jpg')
                plt.close()

            result.save(f"{output_path}/result_with_boxes.jpg")
            return ret_outputs
