# -*- coding: utf-8 -*-
"""OpticalDNA model entry points.

This module provides public OpticalDNA class names while preserving compatibility
with the underlying OCR-style vision-language implementation.
"""

from .modeling_deepseekocr import (
    DeepseekOCRConfig as _BaseOpticalDNAConfig,
    DeepseekOCRModel as _BaseOpticalDNAModel,
    DeepseekOCRForCausalLM as _BaseOpticalDNAForCausalLM,
    BasicImageTransform,
    dynamic_preprocess,
    text_encode,
)


class OpticalDNAConfig(_BaseOpticalDNAConfig):
    """Configuration class for OpticalDNA."""

    model_type = "opticaldna"


class OpticalDNAModel(_BaseOpticalDNAModel):
    """OpticalDNA base model."""

    config_class = OpticalDNAConfig


class OpticalDNAForCausalLM(_BaseOpticalDNAForCausalLM):
    """OpticalDNA model with an autoregressive document decoder."""

    config_class = OpticalDNAConfig


class OpticalDNAProcessor:
    """Placeholder processor name for configuration compatibility."""

    pass
