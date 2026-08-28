"""OpticalDNA public API."""

__version__ = "0.1.0"
__all__ = [
    "OpticalDNA",
    "OpticalDNAConfig",
    "OpticalDNAModel",
    "OpticalDNAForCausalLM",
    "PromptGenerator",
    "PromptLength",
    "TaskType",
]


def __getattr__(name):
    if name == "OpticalDNA":
        from .api import OpticalDNA

        return OpticalDNA
    if name in {"OpticalDNAConfig", "OpticalDNAModel", "OpticalDNAForCausalLM"}:
        from .modeling_opticaldna import (
            OpticalDNAConfig,
            OpticalDNAModel,
            OpticalDNAForCausalLM,
        )

        return {
            "OpticalDNAConfig": OpticalDNAConfig,
            "OpticalDNAModel": OpticalDNAModel,
            "OpticalDNAForCausalLM": OpticalDNAForCausalLM,
        }[name]
    if name in {"PromptGenerator", "PromptLength", "TaskType"}:
        from .prompts import PromptGenerator, PromptLength, TaskType

        return {
            "PromptGenerator": PromptGenerator,
            "PromptLength": PromptLength,
            "TaskType": TaskType,
        }[name]
    raise AttributeError(name)
