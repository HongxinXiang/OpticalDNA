"""Public inference and feature-extraction helpers for OpticalDNA."""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Sequence, Union

import torch
from PIL import ImageOps

from .modeling_deepseekocr import (
    BasicImageTransform,
    dynamic_preprocess,
    load_image,
    text_encode,
)
from .prompts import PromptGenerator, PromptLength

PathLike = Union[str, os.PathLike]
ImageInput = Union[PathLike, Sequence[PathLike], Sequence[Sequence[PathLike]]]


def _normalize_pages(images: ImageInput) -> List[str]:
    if isinstance(images, (str, os.PathLike)):
        pages = [os.fspath(images)]
    elif isinstance(images, Sequence) and images:
        first = images[0]
        if isinstance(first, (str, os.PathLike)):
            pages = [os.fspath(item) for item in images]  # type: ignore[arg-type]
        elif isinstance(first, Sequence):
            if len(images) != 1:
                raise ValueError(
                    "OpticalDNA public inference currently accepts one document per call. "
                    "Pass one nested page list, e.g. [[page1, page2]]."
                )
            pages = [os.fspath(item) for item in first]  # type: ignore[arg-type]
        else:
            raise TypeError(f"Unsupported image input type: {type(first)!r}")
    else:
        raise ValueError("At least one rendered DNA page is required.")

    if not pages:
        raise ValueError("At least one rendered DNA page is required.")
    return pages


def _model_device(model) -> torch.device:
    try:
        device = model.device
        if device is not None and str(device) != "meta":
            return torch.device(device)
    except Exception:
        pass
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    raise RuntimeError("Could not determine an execution device for OpticalDNA.")


def _model_dtype(model) -> torch.dtype:
    for parameter in model.parameters():
        if parameter.is_floating_point() and parameter.device.type != "meta":
            return parameter.dtype
    return torch.float32


def _pool_features(features: torch.Tensor, pooling: str, to_cpu: bool) -> torch.Tensor:
    pooling = pooling.lower()
    if pooling == "none":
        pooled = features
    elif pooling == "mean":
        pooled = features.mean(dim=0)
    elif pooling == "max":
        pooled = features.max(dim=0).values
    else:
        raise ValueError("pooling must be one of: 'none', 'mean', 'max'.")
    pooled = pooled.detach()
    return pooled.cpu() if to_cpu else pooled


def prepare_opticaldna_images(
    model,
    images: ImageInput,
    *,
    base_size: int = 640,
    image_size: int = 640,
    crop_mode: bool = False,
) -> Dict[str, object]:
    """Prepare image tensors only, without constructing any decoder prompt tokens."""
    pages = _normalize_pages(images)
    pil_pages = []
    for path in pages:
        image = load_image(path)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {path}")
        pil_pages.append(image.convert("RGB"))

    page_sizes = {image.size for image in pil_pages}
    if len(page_sizes) != 1:
        raise ValueError(
            "All pages in one OpticalDNA document must have the same image size; "
            f"got {sorted(page_sizes)}."
        )

    patch_size = 16
    downsample_ratio = 4
    transform = BasicImageTransform(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        normalize=True,
    )
    dtype = _model_dtype(model)
    device = _model_device(model)

    if crop_mode:
        crop_lists = []
        crop_ratios = []
        for page in pil_pages:
            if page.size[0] <= image_size and page.size[1] <= image_size:
                crops, ratio = [], (1, 1)
            else:
                crops, ratio = dynamic_preprocess(page, image_size=image_size)
            crop_lists.append(crops)
            crop_ratios.append(tuple(ratio))
        if len(set(crop_ratios)) != 1:
            raise ValueError(
                "All pages must produce the same crop grid for page fusion; "
                f"got {crop_ratios}."
            )
        width_crop_num, height_crop_num = crop_ratios[0]

        global_pages = [
            ImageOps.pad(
                page,
                (base_size, base_size),
                color=tuple(int(x * 255) for x in transform.mean),
            )
            for page in pil_pages
        ]
        images_ori = torch.stack([transform(page).to(dtype) for page in global_pages], dim=0)

        if width_crop_num > 1 or height_crop_num > 1:
            images_crop = torch.stack(
                [
                    torch.stack([transform(crop).to(dtype) for crop in crops], dim=0)
                    for crops in crop_lists
                ],
                dim=0,
            )
        else:
            images_crop = torch.zeros(
                (len(pil_pages), 1, 3, image_size, image_size), dtype=dtype
            )

        num_queries = math.ceil((image_size // patch_size) / downsample_ratio)
        num_queries_base = math.ceil((base_size // patch_size) / downsample_ratio)
        num_image_tokens = (num_queries_base + 1) * num_queries_base + 1
        if width_crop_num > 1 or height_crop_num > 1:
            num_image_tokens += (
                num_queries * width_crop_num + 1
            ) * (num_queries * height_crop_num)
    else:
        processed_pages = [
            page.resize((image_size, image_size)) if image_size <= 640 else page
            for page in pil_pages
        ]
        global_pages = [
            ImageOps.pad(
                page,
                (image_size, image_size),
                color=tuple(int(x * 255) for x in transform.mean),
            )
            for page in processed_pages
        ]
        images_ori = torch.stack([transform(page).to(dtype) for page in global_pages], dim=0)
        images_crop = torch.zeros(
            (len(pil_pages), 1, 3, base_size, base_size), dtype=dtype
        )
        width_crop_num, height_crop_num = 1, 1
        num_queries = math.ceil((image_size // patch_size) / downsample_ratio)
        num_image_tokens = (num_queries + 1) * num_queries + 1

    images_ori = images_ori.unsqueeze(0).to(device=device)
    images_crop = images_crop.to(device=device)
    images_spatial_crop = torch.tensor(
        [[width_crop_num, height_crop_num]], dtype=torch.long
    )

    return {
        "images": [(images_crop, images_ori)],
        "images_spatial_crop": images_spatial_crop,
        "num_image_tokens": int(num_image_tokens),
    }


def prepare_opticaldna_inputs(
    model,
    tokenizer,
    images: ImageInput,
    prompt: Optional[str] = None,
    *,
    base_size: int = 640,
    image_size: int = 640,
    crop_mode: bool = False,
) -> Dict[str, object]:
    """Prepare one document plus prompt for decoder inference."""
    if prompt is None:
        prompt = PromptGenerator().free_ocr(PromptLength.SHORT)
    if "<image>" not in prompt:
        prompt = "<image>\n" + prompt
    if prompt.count("<image>") != 1:
        raise ValueError("Exactly one <image> placeholder is supported per inference call.")

    visual_inputs = prepare_opticaldna_images(
        model,
        images,
        base_size=base_size,
        image_size=image_size,
        crop_mode=crop_mode,
    )
    device = _model_device(model)
    image_token_id = 128815
    image_tokens = [image_token_id] * int(visual_inputs["num_image_tokens"])

    text_before, text_after = prompt.split("<image>", 1)
    tokenized = text_encode(tokenizer, text_before, bos=False, eos=False)
    image_mask = [False] * len(tokenized)
    tokenized += image_tokens
    image_mask += [True] * len(image_tokens)
    tail_tokens = text_encode(tokenizer, text_after, bos=False, eos=False)
    tokenized += tail_tokens
    image_mask += [False] * len(tail_tokens)

    tokenized = [0] + tokenized
    image_mask = [False] + image_mask

    return {
        "input_ids": torch.tensor(tokenized, dtype=torch.long, device=device).unsqueeze(0),
        "images": visual_inputs["images"],
        "images_seq_mask": torch.tensor(
            image_mask, dtype=torch.bool, device=device
        ).unsqueeze(0),
        "images_spatial_crop": visual_inputs["images_spatial_crop"],
    }


def generate_from_images(
    model,
    tokenizer,
    images: ImageInput,
    prompt: Optional[str] = None,
    *,
    base_size: int = 640,
    image_size: int = 640,
    crop_mode: bool = False,
    **generation_kwargs,
) -> str:
    """Generate decoder output; default prompt is short T1 / ``Free OCR.``."""
    inputs = prepare_opticaldna_inputs(
        model,
        tokenizer,
        images,
        prompt,
        base_size=base_size,
        image_size=image_size,
        crop_mode=crop_mode,
    )
    defaults = {
        "do_sample": False,
        "max_new_tokens": 512,
        "use_cache": True,
        "eos_token_id": tokenizer.eos_token_id,
    }
    defaults.update(generation_kwargs)

    model.eval()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **defaults)

    prompt_len = inputs["input_ids"].shape[1]  # type: ignore[index]
    output = tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=False)
    stop_str = "<｜end▁of▁sentence｜>"
    if output.endswith(stop_str):
        output = output[: -len(stop_str)]
    return output.strip()


def extract_visual_features_from_images(
    model,
    images: ImageInput,
    *,
    pooling: str = "mean",
    base_size: int = 640,
    image_size: int = 640,
    crop_mode: bool = False,
    to_cpu: bool = False,
) -> torch.Tensor:
    """Extract visual tokens before they are injected into the language decoder."""
    visual_inputs = prepare_opticaldna_images(
        model,
        images,
        base_size=base_size,
        image_size=image_size,
        crop_mode=crop_mode,
    )
    if not hasattr(model, "encode_visual_features"):
        raise RuntimeError("This OpticalDNA runtime does not expose encode_visual_features().")

    model.eval()
    with torch.inference_mode():
        documents = model.encode_visual_features(
            images=visual_inputs["images"],
            images_spatial_crop=visual_inputs["images_spatial_crop"],
        )
    if len(documents) != 1:
        raise RuntimeError(f"Expected one document feature tensor, got {len(documents)}.")
    features = documents[0]
    if features.ndim != 2 or features.numel() == 0:
        raise RuntimeError(f"Invalid visual feature tensor: shape={tuple(features.shape)}")
    return _pool_features(features, pooling=pooling, to_cpu=to_cpu)


def extract_decoder_features_from_images(
    model,
    tokenizer,
    images: ImageInput,
    prompt: Optional[str] = None,
    *,
    layer: int = -1,
    pooling: str = "mean",
    image_tokens_only: bool = True,
    base_size: int = 640,
    image_size: int = 640,
    crop_mode: bool = False,
    to_cpu: bool = False,
) -> torch.Tensor:
    """Extract prompt-conditioned decoder hidden states without LM logits."""
    inputs = prepare_opticaldna_inputs(
        model,
        tokenizer,
        images,
        prompt,
        base_size=base_size,
        image_size=image_size,
        crop_mode=crop_mode,
    )

    base_model = model.get_model() if hasattr(model, "get_model") else model.model
    base_model.eval()
    with torch.inference_mode():
        outputs = base_model(
            input_ids=inputs["input_ids"],
            images=inputs["images"],
            images_seq_mask=inputs["images_seq_mask"],
            images_spatial_crop=inputs["images_spatial_crop"],
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("OpticalDNA did not return decoder hidden states.")
    try:
        features = hidden_states[layer][0]
    except IndexError as exc:
        raise ValueError(
            f"Invalid layer={layer}; model returned {len(hidden_states)} hidden-state tensors."
        ) from exc

    if image_tokens_only:
        mask = inputs["images_seq_mask"][0]  # type: ignore[index]
        features = features[mask]
        if features.numel() == 0:
            raise RuntimeError("No image-token decoder features were produced.")

    return _pool_features(features, pooling=pooling, to_cpu=to_cpu)


# Backward-compatible public helper name: "features" means visual features.
extract_features_from_images = extract_visual_features_from_images
