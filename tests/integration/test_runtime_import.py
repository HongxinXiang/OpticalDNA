"""Lightweight runtime-import test for the public OpticalDNA API."""

import pytest


@pytest.mark.integration
def test_public_runtime_imports() -> None:
    import opticaldna
    from opticaldna import (
        OpticalDNA,
        OpticalDNAConfig,
        OpticalDNAForCausalLM,
        OpticalDNAModel,
        PromptGenerator,
        PromptLength,
        TaskType,
    )

    assert opticaldna.__version__
    assert OpticalDNA.__name__ == "OpticalDNA"
    assert OpticalDNAConfig.__name__ == "OpticalDNAConfig"
    assert OpticalDNAModel.__name__ == "OpticalDNAModel"
    assert OpticalDNAForCausalLM.__name__ == "OpticalDNAForCausalLM"
    assert PromptGenerator.__name__ == "PromptGenerator"
    assert PromptLength.SHORT.value == "short"
    assert TaskType.T1_FULL_OCR.value == "t1_full_ocr"
