"""Static release-contract tests that avoid importing the heavyweight model stack."""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OPTICALDNA = ROOT / "opticaldna"


def _class_methods(path: Path, class_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                child.name
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    raise AssertionError(f"Class {class_name!r} not found in {path}")


def test_huggingface_config_contract() -> None:
    config = json.loads((OPTICALDNA / "config.json").read_text(encoding="utf-8"))
    assert config["model_type"] == "opticaldna"
    assert config["architectures"] == ["OpticalDNAForCausalLM"]
    assert config["auto_map"] == {
        "AutoConfig": "modeling_opticaldna.OpticalDNAConfig",
        "AutoModel": "modeling_opticaldna.OpticalDNAForCausalLM",
        "AutoModelForCausalLM": "modeling_opticaldna.OpticalDNAForCausalLM",
    }


def test_public_high_level_api_contract() -> None:
    methods = _class_methods(OPTICALDNA / "api.py", "OpticalDNA")
    assert {"from_pretrained", "default_prompt", "build_prompt", "generate", "extract_visual_features", "extract_features", "extract_decoder_features"} <= methods


def test_transformers_model_api_contract() -> None:
    methods = _class_methods(OPTICALDNA / "modeling_opticaldna.py", "OpticalDNAForCausalLM")
    assert {"from_pretrained", "default_prompt", "build_prompt", "generate_document", "extract_visual_features", "extract_features", "extract_decoder_features"} <= methods


def test_strict_loading_guards_remain_present() -> None:
    source = (OPTICALDNA / "modeling_opticaldna.py").read_text(encoding="utf-8")
    for required in (
        "output_loading_info=True",
        "missing_keys",
        "unexpected_keys",
        "mismatched_keys",
        "error_msgs",
        "RuntimeError",
    ):
        assert required in source, f"Strict loading guard disappeared: {required}"

    assert "model.vision_model.embeddings.position_ids" in source
    assert "_is_nonpersistent_buffer" in source
    assert "_non_persistent_buffers_set" in source


def test_public_package_exports() -> None:
    tree = ast.parse((OPTICALDNA / "__init__.py").read_text(encoding="utf-8"))
    exports = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    exports = ast.literal_eval(node.value)
    assert exports is not None
    assert {
        "OpticalDNA",
        "OpticalDNAConfig",
        "OpticalDNAModel",
        "OpticalDNAForCausalLM",
        "PromptGenerator",
        "PromptLength",
        "TaskType",
    } <= set(exports)


def test_visual_feature_path_is_opticaldna_specific_and_pre_decoder() -> None:
    upstream_methods = _class_methods(OPTICALDNA / "modeling_deepseekocr.py", "DeepseekOCRModel")
    assert "encode_visual_features" not in upstream_methods

    opticaldna_methods = _class_methods(
        OPTICALDNA / "modeling_opticaldna.py", "OpticalDNAForCausalLM"
    )
    assert "encode_visual_features" in opticaldna_methods

    inference_source = (OPTICALDNA / "inference.py").read_text(encoding="utf-8")
    assert 'model.encode_visual_features(' in inference_source
    assert "def extract_decoder_features_from_images(" in inference_source
