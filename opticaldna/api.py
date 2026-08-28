"""High-level OpticalDNA API, intentionally similar to Evo2's usage surface."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from .checkpoint import validate_hf_model_repo
from .prompts import PromptGenerator, PromptLength, TaskType


class OpticalDNA:
    """Load a released OpticalDNA checkpoint and expose generation/features APIs."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: str = "cuda",
        dtype="auto",
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        token: Optional[str] = None,
        local_files_only: bool = False,
    ) -> None:
        try:
            from huggingface_hub import snapshot_download
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "OpticalDNA inference requires transformers and huggingface_hub. "
                "Install the project environment before loading a checkpoint."
            ) from exc

        source = Path(model_name_or_path).expanduser()
        if source.is_dir():
            local_dir = source.resolve()
        else:
            local_dir = Path(
                snapshot_download(
                    repo_id=model_name_or_path,
                    revision=revision,
                    cache_dir=cache_dir,
                    token=token,
                    local_files_only=local_files_only,
                )
            )

        validate_hf_model_repo(local_dir)

        tokenizer = AutoTokenizer.from_pretrained(
            local_dir,
            local_files_only=True,
        )

        # The high-level OpticalDNA API uses the installed/public OpticalDNA class
        # directly. This avoids Transformers local dynamic-module caching while
        # preserving strict checkpoint loading. Standard Hugging Face AutoModel
        # loading remains available for model IDs hosted on the Hub.
        from .modeling_opticaldna import OpticalDNAForCausalLM

        model, loading_info = OpticalDNAForCausalLM.from_pretrained(
            local_dir,
            local_files_only=True,
            torch_dtype=dtype,
            output_loading_info=True,
            ignore_mismatched_sizes=False,
        )

        problems = {
            key: value
            for key, value in loading_info.items()
            if key in {"missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"} and value
        }
        if problems:
            del model
            raise RuntimeError(
                "OpticalDNA checkpoint did not load exactly; refusing to continue with "
                f"partially/randomly initialized parameters. Loading info: {problems}"
            )

        if device.startswith("cuda") and not torch.cuda.is_available():
            del model
            raise RuntimeError(
                f"device={device!r} was requested but CUDA is not available. "
                "Use device='cpu' only for debugging."
            )
        model = model.to(device)
        model.eval()

        self.model_name_or_path = model_name_or_path
        self.local_dir = local_dir
        self.tokenizer = tokenizer
        self.model = model
        self.prompts = PromptGenerator()

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs) -> "OpticalDNA":
        return cls(model_name_or_path, **kwargs)

    @staticmethod
    def default_prompt(length: PromptLength | str = PromptLength.SHORT) -> str:
        return PromptGenerator().free_ocr(length=length)

    @staticmethod
    def build_prompt(
        task: TaskType | str,
        length: PromptLength | str = PromptLength.SHORT,
        sample: Optional[dict] = None,
    ) -> str:
        return PromptGenerator().build(task=task, length=length, sample=sample or {})

    def generate(self, images, prompt: Optional[str] = None, **kwargs) -> str:
        """Run decoder inference; prompt defaults to short T1 / ``Free OCR.``."""
        return self.model.generate_document(
            tokenizer=self.tokenizer,
            images=images,
            prompt=prompt,
            **kwargs,
        )

    def extract_visual_features(self, images, **kwargs):
        """Extract visual encoder/projector/page-fusion features before the decoder."""
        return self.model.extract_visual_features(images=images, **kwargs)

    def extract_features(self, images, **kwargs):
        """Alias for :meth:`extract_visual_features`."""
        return self.extract_visual_features(images, **kwargs)

    def extract_decoder_features(self, images, prompt: Optional[str] = None, **kwargs):
        """Extract prompt-conditioned decoder hidden states."""
        return self.model.extract_decoder_features(
            tokenizer=self.tokenizer,
            images=images,
            prompt=prompt,
            **kwargs,
        )
