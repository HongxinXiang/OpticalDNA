"""Prompt-builder tests for the released OpticalDNA inference API."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PROMPTS_PATH = ROOT / "opticaldna" / "prompts.py"

spec = importlib.util.spec_from_file_location("opticaldna_prompts", PROMPTS_PATH)
module = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(module)

PromptGenerator = module.PromptGenerator
PromptLength = module.PromptLength
TaskType = module.TaskType


def test_default_free_ocr_prompt() -> None:
    prompt = PromptGenerator().free_ocr()
    assert prompt == "Free OCR."


def test_t1_prompt_build_matches_free_ocr() -> None:
    generator = PromptGenerator()
    prompt = generator.build(TaskType.T1_FULL_OCR, PromptLength.SHORT, sample={})
    assert prompt == generator.free_ocr(PromptLength.SHORT)


def test_t5_prompt_requires_query_and_contains_it() -> None:
    prompt = PromptGenerator().build(
        TaskType.T5_SUBSEQ_LOCATE,
        PromptLength.MEDIUM,
        sample={"query": "ACGTACGT"},
    )
    assert "ACGTACGT" in prompt
    assert "<|grounding|>" in prompt


def test_missing_required_task_payload_raises() -> None:
    with pytest.raises(KeyError):
        PromptGenerator().build(TaskType.T3_ROI_OCR, PromptLength.SHORT, sample={})
