# -*- coding: utf-8 -*-
"""OpticalDNA model entry points.

This module provides public OpticalDNA class names while preserving compatibility
with the underlying OCR-style vision-language implementation.
"""

from __future__ import annotations

import torch

from .modeling_deepseekocr import (
    DeepseekOCRConfig as _BaseOpticalDNAConfig,
    DeepseekOCRModel as _BaseOpticalDNAModel,
    DeepseekOCRForCausalLM as _BaseOpticalDNAForCausalLM,
)
from .inference import (
    extract_decoder_features_from_images,
    extract_visual_features_from_images,
    generate_from_images,
)
from .prompts import PromptGenerator, PromptLength, TaskType


class OpticalDNAConfig(_BaseOpticalDNAConfig):
    """Configuration class for OpticalDNA."""

    model_type = "opticaldna"


class OpticalDNAModel(_BaseOpticalDNAModel):
    """OpticalDNA base model."""

    config_class = OpticalDNAConfig


class OpticalDNAForCausalLM(_BaseOpticalDNAForCausalLM):
    """OpticalDNA visual encoder + autoregressive genomic document decoder."""

    config_class = OpticalDNAConfig

    # CLIP creates this deterministic index buffer at module construction time with
    # persistent=False, so it is intentionally absent from safetensors checkpoints.
    _keys_to_ignore_on_load_missing = [
        r"model\.vision_model\.embeddings\.position_ids",
    ]
    _allowed_nonpersistent_missing_keys = {
        "model.vision_model.embeddings.position_ids",
    }

    @staticmethod
    def _is_nonpersistent_buffer(model, key: str) -> bool:
        """Return True only when ``key`` resolves to a non-persistent torch buffer."""
        parts = key.split(".")
        module = model
        try:
            for part in parts[:-1]:
                module = getattr(module, part)
        except AttributeError:
            return False
        name = parts[-1]
        return (
            name in getattr(module, "_buffers", {})
            and name in getattr(module, "_non_persistent_buffers_set", set())
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Load OpticalDNA weights strictly; partial/random fallback is not allowed."""
        return_loading_info = bool(kwargs.pop("output_loading_info", False))
        model, loading_info = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            output_loading_info=True,
            **kwargs,
        )

        loading_info = dict(loading_info)
        raw_missing = list(loading_info.get("missing_keys") or [])
        remaining_missing = []
        for key in raw_missing:
            if key in cls._allowed_nonpersistent_missing_keys:
                if not cls._is_nonpersistent_buffer(model, key):
                    del model
                    raise RuntimeError(
                        f"Allowed derived key {key!r} is not a non-persistent buffer; "
                        "refusing to relax strict checkpoint loading."
                    )
                continue
            remaining_missing.append(key)
        loading_info["missing_keys"] = remaining_missing

        problems = {
            key: value
            for key, value in loading_info.items()
            if key in {"missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"}
            and value
        }
        if problems:
            del model
            raise RuntimeError(
                "OpticalDNA checkpoint did not load exactly; refusing to continue with "
                f"partially/randomly initialized parameters. Loading info: {problems}"
            )
        return (model, loading_info) if return_loading_info else model

    @staticmethod
    def default_prompt(length: PromptLength = PromptLength.SHORT) -> str:
        """Return the default T1 Free-OCR prompt."""
        return PromptGenerator().free_ocr(length=PromptLength(length))

    @staticmethod
    def build_prompt(
        task: TaskType,
        length: PromptLength = PromptLength.SHORT,
        sample=None,
    ) -> str:
        """Build one of the released T1--T6 OpticalDNA task prompts."""
        return PromptGenerator().build(
            task=TaskType(task),
            length=PromptLength(length),
            sample=sample or {},
        )

    def encode_visual_features(self, images, images_spatial_crop):
        """Return projected/fused visual tokens before the language Decoder.

        This method intentionally lives in ``modeling_opticaldna.py`` because it
        defines OpticalDNA-specific feature semantics. It reuses the frozen model
        components created by the OCR backbone but does not execute any language
        Decoder layer.
        """
        backbone = self.get_model()
        sam_model = getattr(backbone, "sam_model", None)
        vision_model = getattr(backbone, "vision_model", None)
        projector = getattr(backbone, "projector", None)
        page_fusion_layer = getattr(backbone, "page_fusion_layer", None)

        if any(x is None for x in (sam_model, vision_model, projector, page_fusion_layer)):
            raise RuntimeError("OpticalDNA visual encoder components are not available.")
        if images is None or images_spatial_crop is None:
            raise ValueError("images and images_spatial_crop are required.")

        document_features = []
        for image, crop_shape in zip(images, images_spatial_crop):
            patches = image[0]
            image_ori = image[1]
            patches, image_ori = backbone.ensure_page_dim_patches_ori(patches, image_ori)

            n_img_p, n_page_p, n_patch, c_p, h_p, w_p = patches.shape
            n_img_o, n_page_o, c_o, h_o, w_o = image_ori.shape

            patches_flat = patches.view(n_img_p * n_page_p * n_patch, c_p, h_p, w_p)
            image_ori_flat = image_ori.view(n_img_o * n_page_o, c_o, h_o, w_o)

            has_local_crops = torch.count_nonzero(patches).item() != 0
            if has_local_crops:
                local_sam = sam_model(patches_flat)
                local_vit = vision_model(patches_flat, local_sam)
                local_per_page = torch.cat(
                    (local_vit[:, 1:], local_sam.flatten(2).permute(0, 2, 1)),
                    dim=-1,
                )
                local_per_page = projector(local_per_page)
                local_tail_shape = list(local_per_page.shape[1:])
                local_permute = local_per_page.view(
                    *([n_img_p, n_page_p, n_patch] + local_tail_shape)
                ).permute(0, 2, 1, 3, 4)
                local_features = page_fusion_layer(
                    local_permute.view(*([n_img_p * n_patch, n_page_p] + local_tail_shape))
                )
            else:
                local_features = None

            global_sam = sam_model(image_ori_flat)
            global_vit = vision_model(image_ori_flat, global_sam)
            global_per_page = torch.cat(
                (global_vit[:, 1:], global_sam.flatten(2).permute(0, 2, 1)),
                dim=-1,
            )
            global_per_page = projector(global_per_page)
            global_tail_shape = list(global_per_page.shape[1:])
            global_features = page_fusion_layer(
                global_per_page.view(*([n_img_o, n_page_o] + global_tail_shape))
            ).view(*([n_img_o] + global_tail_shape))

            # Public inference currently accepts one document per call, matching
            # the existing backbone forward path where n_img_o == 1.
            if n_img_o != 1:
                raise RuntimeError(
                    "OpticalDNA visual feature extraction currently expects one document "
                    f"per call; got n_img={n_img_o}."
                )

            _, global_hw, global_dim = global_features.shape
            global_h = global_w = int(global_hw ** 0.5)
            if global_h * global_w != global_hw:
                raise RuntimeError(
                    f"Global visual token count is not square: {global_hw}."
                )

            global_features = global_features.view(global_h, global_w, global_dim)
            global_features = torch.cat(
                [
                    global_features,
                    backbone.image_newline[None, None, :].expand(global_h, 1, global_dim),
                ],
                dim=1,
            ).view(-1, global_dim)

            if local_features is not None:
                _, local_hw, local_dim = local_features.shape
                local_h = local_w = int(local_hw ** 0.5)
                if local_h * local_w != local_hw:
                    raise RuntimeError(
                        f"Local visual token count is not square: {local_hw}."
                    )

                width_crop_num = int(crop_shape[0])
                height_crop_num = int(crop_shape[1])
                local_features = (
                    local_features.view(
                        height_crop_num,
                        width_crop_num,
                        local_h,
                        local_w,
                        local_dim,
                    )
                    .permute(0, 2, 1, 3, 4)
                    .reshape(
                        height_crop_num * local_h,
                        width_crop_num * local_w,
                        local_dim,
                    )
                )
                local_features = torch.cat(
                    [
                        local_features,
                        backbone.image_newline[None, None, :].expand(
                            height_crop_num * local_h,
                            1,
                            local_dim,
                        ),
                    ],
                    dim=1,
                ).view(-1, local_dim)
                features = torch.cat(
                    [local_features, global_features, backbone.view_seperator[None, :]],
                    dim=0,
                )
            else:
                features = torch.cat(
                    [global_features, backbone.view_seperator[None, :]],
                    dim=0,
                )

            document_features.append(features)

        return document_features

    def generate_document(self, tokenizer, images, prompt=None, **kwargs) -> str:
        """Generate Decoder output; default prompt is short T1 / ``Free OCR.``."""
        return generate_from_images(self, tokenizer, images, prompt, **kwargs)

    def extract_visual_features(self, images, **kwargs):
        """Extract OpticalDNA visual features without executing the Decoder."""
        return extract_visual_features_from_images(self, images, **kwargs)

    def extract_features(self, images, **kwargs):
        """Alias for visual feature extraction used by downstream tasks."""
        return self.extract_visual_features(images, **kwargs)

    def extract_decoder_features(self, tokenizer, images, prompt=None, **kwargs):
        """Extract prompt-conditioned Decoder hidden states."""
        return extract_decoder_features_from_images(
            self,
            tokenizer,
            images,
            prompt=prompt,
            **kwargs,
        )


class OpticalDNAProcessor:
    """Placeholder processor name for configuration compatibility."""

    pass
