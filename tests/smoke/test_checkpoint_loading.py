"""Smoke-test a released local or Hugging Face OpticalDNA checkpoint."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _check_feature_tensor(features, label: str) -> None:
    import torch

    if features.ndim != 1 or features.numel() == 0 or not torch.isfinite(features).all():
        raise RuntimeError(f"Invalid {label}: shape={tuple(features.shape)}")
    print(f"PASS: {label}: shape={tuple(features.shape)}")


def _run_smoke(model_id: str, device: str, image: str | None, run_generation: bool) -> None:
    from opticaldna import OpticalDNA, PromptGenerator, PromptLength

    opticaldna = OpticalDNA(model_id, device=device)
    print(f"PASS: strict checkpoint load succeeded: {opticaldna.model_name_or_path}")

    if image:
        visual_features = opticaldna.extract_features(
            image,
            pooling="mean",
            to_cpu=True,
        )
        _check_feature_tensor(
            visual_features,
            "visual feature extraction succeeded (decoder not executed)",
        )

        if run_generation:
            prompt = PromptGenerator().free_ocr(PromptLength.SHORT)
            decoder_features = opticaldna.extract_decoder_features(
                image,
                prompt=prompt,
                pooling="mean",
                to_cpu=True,
            )
            _check_feature_tensor(
                decoder_features,
                "prompt-conditioned decoder feature extraction succeeded",
            )

            text = opticaldna.generate(
                image,
                prompt=prompt,
                max_new_tokens=32,
            )
            if not isinstance(text, str):
                raise RuntimeError(f"Generation returned {type(text)!r}, expected str")
            print(f"PASS: Free OCR generation succeeded: {text[:120]!r}")


@pytest.mark.smoke
def test_released_checkpoint_from_environment() -> None:
    """Opt-in pytest smoke test; excluded from normal GitHub CI."""
    model_id = os.environ.get("OPTICALDNA_TEST_MODEL")
    if not model_id:
        pytest.skip("Set OPTICALDNA_TEST_MODEL to run the real-checkpoint smoke test.")
    _run_smoke(
        model_id=model_id,
        device=os.environ.get("OPTICALDNA_TEST_DEVICE", "cuda"),
        image=os.environ.get("OPTICALDNA_TEST_IMAGE"),
        run_generation=os.environ.get("OPTICALDNA_TEST_GENERATION", "0") == "1",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Hugging Face model ID or local model directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image", default=None, help="Optional rendered DNA page for feature smoke test")
    parser.add_argument(
        "--run-generation",
        action="store_true",
        help="Also test prompt-conditioned decoder features and Free OCR generation.",
    )
    args = parser.parse_args()
    _run_smoke(args.model, args.device, args.image, args.run_generation)


if __name__ == "__main__":
    main()
