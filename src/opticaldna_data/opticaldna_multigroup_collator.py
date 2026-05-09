from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

import torch
from PIL import Image, ImageOps
from torch.nn.utils.rnn import pad_sequence

try:
    from opticaldna.modeling_opticaldna import (
        text_encode,
        BasicImageTransform,
        dynamic_preprocess,
    )
except Exception:
    from opticaldna.modeling_opticaldna import text_encode, BasicImageTransform, dynamic_preprocess

ImageItem = Union[Image.Image, Dict[str, Any], str]
ImageGroups = List[List[ImageItem]]


@dataclass
class OpticalDNAMultiGroupDataCollator:
    tokenizer: Any
    model: Any
    image_size: int = 640
    base_size: int = 1024
    crop_mode: bool = True
    image_token_id: int = 128815
    train_on_responses_only: bool = True
    fuse_shards_in_group: bool = True

    max_text_tokens: int | None = None
    """Optional maximum number of text tokens before inserting image tokens."""

    max_total_tokens: int | None = None
    """Optional maximum final sequence length including image tokens."""

    stage: int = 0
    stages: List[Dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        self.dtype = self.model.dtype
        self.image_transform = BasicImageTransform(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            normalize=True,
        )
        self.patch_size = 16
        self.downsample_ratio = 4

        if hasattr(self.tokenizer, 'bos_token_id') and self.tokenizer.bos_token_id is not None:
            self.bos_id = self.tokenizer.bos_token_id
        else:
            self.bos_id = 0
            print(f"Warning: tokenizer has no bos_token_id, using default: {self.bos_id}")

        self._apply_stage()

    def _apply_stage(self) -> None:
        if not self.stages:
            return
        if not (0 <= int(self.stage) < len(self.stages)):
            raise ValueError(f"stage={self.stage} out of range for stages (len={len(self.stages)}).")
        overrides = self.stages[int(self.stage)] or {}
        for k, v in overrides.items():
            if not hasattr(self, k):
                raise AttributeError(f"Unknown stage override key: {k}")
            setattr(self, k, v)

    def set_stage(self, stage: int) -> None:
        self.stage = int(stage)
        self._apply_stage()

    def deserialize_image(self, image_data: ImageItem) -> Image.Image:
        if isinstance(image_data, Image.Image):
            return image_data.convert("RGB")
        if isinstance(image_data, dict) and "bytes" in image_data:
            image = Image.open(io.BytesIO(image_data["bytes"]))
            return image.convert("RGB")
        if isinstance(image_data, str):
            return Image.open(image_data).convert("RGB")
        raise ValueError(f"Unsupported image format: {type(image_data)}")

    def calculate_image_token_count(self, image: Image.Image, crop_ratio: Tuple[int, int]) -> int:
        num_queries = math.ceil((self.image_size // self.patch_size) / self.downsample_ratio)
        num_queries_base = math.ceil((self.base_size // self.patch_size) / self.downsample_ratio)

        width_crop_num, height_crop_num = crop_ratio

        if self.crop_mode:
            img_tokens = num_queries_base * num_queries_base + 1
            if width_crop_num > 1 or height_crop_num > 1:
                img_tokens += (num_queries * width_crop_num + 1) * (num_queries * height_crop_num)
        else:
            img_tokens = num_queries * num_queries + 1

        return img_tokens

    def process_image(
            self, image: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[int], Tuple[int, int]]:
        images_crop_list: List[torch.Tensor] = []

        if self.crop_mode:

            if image.size[0] <= 640 and image.size[1] <= 640:
                crop_ratio = (1, 1)
                images_crop_raw = []
            else:
                images_crop_raw, crop_ratio = dynamic_preprocess(
                    image,
                    min_num=2,
                    max_num=9,
                    image_size=self.image_size,
                    use_thumbnail=False,
                )

            global_view = ImageOps.pad(
                image,
                (self.base_size, self.base_size),
                color=tuple(int(x * 255) for x in self.image_transform.mean),
            )
            images_ori_one = self.image_transform(global_view).to(self.dtype)

            width_crop_num, height_crop_num = crop_ratio
            spatial_crop = [int(width_crop_num), int(height_crop_num)]

            if width_crop_num > 1 or height_crop_num > 1:
                for crop_img in images_crop_raw:
                    images_crop_list.append(self.image_transform(crop_img).to(self.dtype))

            num_queries = math.ceil((self.image_size // self.patch_size) / self.downsample_ratio)
            num_queries_base = math.ceil((self.base_size // self.patch_size) / self.downsample_ratio)

            tokenized_image: List[int] = ([self.image_token_id] * num_queries_base + [
                self.image_token_id]) * num_queries_base
            tokenized_image += [self.image_token_id]

            if width_crop_num > 1 or height_crop_num > 1:
                tokenized_image += ([self.image_token_id] * (num_queries * width_crop_num) + [self.image_token_id]) * (
                        num_queries * height_crop_num
                )

        else:
            crop_ratio = (1, 1)
            spatial_crop = [1, 1]

            if self.base_size <= 640:
                resized_image = image.resize((self.base_size, self.base_size), Image.LANCZOS)
                images_ori_one = self.image_transform(resized_image).to(self.dtype)
            else:
                global_view = ImageOps.pad(
                    image,
                    (self.base_size, self.base_size),
                    color=tuple(int(x * 255) for x in self.image_transform.mean),
                )
                images_ori_one = self.image_transform(global_view).to(self.dtype)

            num_queries = math.ceil((self.base_size // self.patch_size) / self.downsample_ratio)
            tokenized_image = ([self.image_token_id] * num_queries + [self.image_token_id]) * num_queries
            tokenized_image += [self.image_token_id]

        if images_crop_list:
            images_crop_one = torch.stack(images_crop_list, dim=0)
        else:
            images_crop_one = torch.empty((0, 3, self.base_size, self.base_size), dtype=self.dtype)

        return images_ori_one, images_crop_one, spatial_crop, tokenized_image, crop_ratio

    def _extract_image_groups(self, messages: List[Dict[str, Any]]) -> ImageGroups:
        groups: List[List[ImageItem]] = []
        for m in messages:
            if "images" not in m or not m["images"]:
                continue
            imgs = m["images"]

            if isinstance(imgs, list) and len(imgs) > 0 and not isinstance(imgs[0], list):
                groups.append(list(imgs))
            elif isinstance(imgs, list) and len(imgs) > 0 and isinstance(imgs[0], list):
                groups.extend([list(g) for g in imgs])
            else:
                raise ValueError(f"Unsupported images field: {type(imgs)}")
        if not groups:
            raise ValueError("No images found in sample messages.")
        return groups

    def process_single_sample(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        groups = self._extract_image_groups(messages)

        tokenized_str: List[int] = []
        images_seq_mask: List[bool] = []

        images_ori_groups: List[torch.Tensor] = []
        images_crop_groups: List[torch.Tensor] = []
        images_spatial_crop_rows: List[List[int]] = []

        prompt_token_count = -1
        assistant_started = False
        group_idx = 0

        tokenized_str.append(self.bos_id)
        images_seq_mask.append(False)

        text_token_count = 1

        def _append_text_tokens(tok: List[int]) -> None:
            nonlocal text_token_count
            if not tok:
                return
            if self.max_text_tokens is None:
                tokenized_str.extend(tok)
                images_seq_mask.extend([False] * len(tok))
                text_token_count += len(tok)
                return
            remain = int(self.max_text_tokens) - int(text_token_count)
            if remain <= 0:
                return
            if len(tok) > remain:
                tok = tok[:remain]
            tokenized_str.extend(tok)
            images_seq_mask.extend([False] * len(tok))
            text_token_count += len(tok)

        for message in messages:
            role = message["role"]
            content = message["content"]

            if role == "<|Assistant|>":
                if not assistant_started:
                    prompt_token_count = len(tokenized_str)
                    assistant_started = True

                content = f"{content.strip()} {self.tokenizer.eos_token}"

            text_splits = content.split("<image>")

            for i, text_sep in enumerate(text_splits):
                tokenized_sep = text_encode(self.tokenizer, text_sep, bos=False, eos=False)

                _append_text_tokens(tokenized_sep)

                if i < len(text_splits) - 1:
                    if group_idx >= len(groups):
                        raise ValueError("Data mismatch: Found '<image>' token but no corresponding image group.")

                    shard_items = groups[group_idx]

                    ori_pages: List[torch.Tensor] = []
                    crop_pages: List[torch.Tensor] = []
                    P_max = 0

                    tok_img_ref: List[int] | None = None
                    tok_img_lens: List[int] = []

                    for shard in shard_items:
                        pil = self.deserialize_image(shard)
                        ori_one, crop_one, spatial_crop, tok_img, _ = self.process_image(pil)

                        ori_pages.append(ori_one)
                        crop_pages.append(crop_one)
                        images_spatial_crop_rows.append(spatial_crop)

                        P_max = max(P_max, int(crop_one.shape[0]))

                        tok_img_lens.append(len(tok_img))
                        if tok_img_ref is None:
                            tok_img_ref = tok_img

                        if not self.fuse_shards_in_group:
                            tokenized_str.extend(tok_img)
                            images_seq_mask.extend([True] * len(tok_img))

                    if self.fuse_shards_in_group:
                        if tok_img_ref is None:
                            raise ValueError("No tok_img generated for this image group (unexpected).")
                        if len(set(tok_img_lens)) != 1:
                            raise ValueError(
                                f"Inconsistent image-token lengths within one <image> group: {tok_img_lens}. "
                                "This would break fusion alignment. Please ensure pages are preprocessed consistently "
                                "(same crop_mode/base_size/image_size), or disable fusion for debugging."
                            )
                        tokenized_str.extend(tok_img_ref)
                        images_seq_mask.extend([True] * len(tok_img_ref))

                    P_max = max(P_max, 1)

                    padded_crop_pages: List[torch.Tensor] = []
                    for crop_one in crop_pages:
                        p = int(crop_one.shape[0])
                        if p == 0:
                            pad = torch.zeros((P_max, 3, self.base_size, self.base_size), dtype=self.dtype)
                            padded_crop_pages.append(pad)
                            continue
                        if p < P_max:
                            pad = torch.zeros((P_max - p, 3, self.base_size, self.base_size), dtype=self.dtype)
                            padded_crop_pages.append(torch.cat([crop_one, pad], dim=0))
                        else:
                            padded_crop_pages.append(crop_one[:P_max])

                    images_crop_group = torch.stack(padded_crop_pages, dim=0)
                    images_ori_group = torch.stack(ori_pages, dim=0)

                    images_crop_groups.append(images_crop_group)
                    images_ori_groups.append(images_ori_group)

                    group_idx += 1

        if group_idx != len(groups):
            raise ValueError(f"Mismatch: {len(groups)} image groups but {group_idx} '<image>' tokens used.")

        if not assistant_started:
            print("Warning: No assistant message found in sample. Masking all tokens.")
            prompt_token_count = len(tokenized_str)

        n_image = len(images_ori_groups)
        if n_image == 0:
            raise ValueError("No image groups were processed; check <image>/images alignment.")

        n_pages = [int(t.shape[0]) for t in images_ori_groups]
        if len(set(n_pages)) != 1:
            raise ValueError(
                f"Inconsistent n_page across image groups: {n_pages}. "
                "This may break img_id/page alignment; please fix upstream or add padding with a page_mask."
            )

        images_ori = torch.stack(images_ori_groups, dim=0)

        if n_image != 1:
            raise NotImplementedError(
                f"Current expected images_crop shape is (n_page,P,C,H,W) for ONE <image>. "
                f"Got n_image={n_image}. If you truly need multiple <image>, please extend protocol to "
                f"(n_image,n_page,P,C,H,W) and update downstream accordingly."
            )
        images_crop = images_crop_groups[0]

        images_spatial_crop_tensor = torch.tensor(images_spatial_crop_rows, dtype=torch.long)

        if self.max_total_tokens is not None and len(tokenized_str) > int(self.max_total_tokens):
            max_len = int(self.max_total_tokens)

            try:
                last_img_pos = max(i for i, m in enumerate(images_seq_mask) if m)
            except ValueError:
                last_img_pos = -1

            if last_img_pos >= 0 and max_len <= last_img_pos:
                raise ValueError(
                    f"max_total_tokens={max_len} is too small: it would cut inside image tokens "
                    f"(last_img_pos={last_img_pos}). "
                    "Increase max_total_tokens or reduce the number of image tokens."
                )

            cut = max_len
            while cut > 0 and images_seq_mask[cut - 1]:
                cut -= 1

            if cut <= 1:
                raise ValueError(
                    f"Unsafe truncation: cut={cut}. "
                    "Increase max_total_tokens to preserve at least one non-image token."
                )

            tokenized_str = tokenized_str[:cut]
            images_seq_mask = images_seq_mask[:cut]

            if prompt_token_count > cut:
                prompt_token_count = cut

        return {
            "input_ids": torch.tensor(tokenized_str, dtype=torch.long),
            "images_seq_mask": torch.tensor(images_seq_mask, dtype=torch.bool),
            "images_ori": images_ori,
            "images_crop": images_crop,
            "images_spatial_crop": images_spatial_crop_tensor,
            "prompt_token_count": int(prompt_token_count),
        }

    def __call__(self, batch_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(batch_items, list) or len(batch_items) == 0:
            raise ValueError("batch_items must be a non-empty list.")

        conversations: List[Dict[str, Any]] = []
        task_names: List[Any] = []
        metadatas: List[Any] = []
        labels_raw: List[Any] = []
        masks_batch: List[Any] = []

        for i, item in enumerate(batch_items):
            if not isinstance(item, dict):
                raise TypeError(f"batch_items[{i}] must be dict, got {type(item)}")

            conv = item.get("conversation", None)
            if not isinstance(conv, dict) or "messages" not in conv:
                raise ValueError(
                    f"batch_items[{i}] missing a valid 'conversation' dict with key 'messages'. "
                    f"Got keys={list(item.keys())}"
                )

            conversations.append(conv)
            task_names.append(item.get("task_name", None))
            metadatas.append(item.get("metadata", None))
            labels_raw.append(item.get("label", None))

            mval = item.get("masks", None)
            if mval is None:
                for msg in conv.get("messages", []):
                    if isinstance(msg, dict) and "masks" in msg:
                        mval = msg["masks"]
                        break
            masks_batch.append(mval)

        batch_data = [self.process_single_sample(conv["messages"]) for conv in conversations]

        input_ids_list = [item["input_ids"] for item in batch_data]
        images_seq_mask_list = [item["images_seq_mask"] for item in batch_data]
        prompt_token_counts = [item["prompt_token_count"] for item in batch_data]

        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        images_seq_mask = pad_sequence(images_seq_mask_list, batch_first=True, padding_value=False)

        labels = input_ids.clone()

        labels[labels == self.tokenizer.pad_token_id] = -100

        labels[images_seq_mask] = -100

        if self.train_on_responses_only:
            for idx, prompt_count in enumerate(prompt_token_counts):
                if int(prompt_count) > 0:
                    labels[idx, : int(prompt_count)] = -100

        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        images_batch = [(item["images_crop"], item["images_ori"]) for item in batch_data]

        images_spatial_crop = torch.cat([item["images_spatial_crop"] for item in batch_data], dim=0)

        out: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "images": images_batch,
            "images_seq_mask": images_seq_mask,
            "images_spatial_crop": images_spatial_crop,
        }

        return out

