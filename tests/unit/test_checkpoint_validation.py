"""Fast tests for OpticalDNA checkpoint/repository validation."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from opticaldna.checkpoint import (
    REQUIRED_AUTO_MAP,
    REQUIRED_PAGE_FUSION_KEYS,
    REQUIRED_RUNTIME_FILES,
    inspect_checkpoint_weights,
    validate_hf_model_repo,
)


def _write_fake_safetensors(path: Path, keys: list[str]) -> None:
    """Write the smallest safetensors-like file needed by the header validator."""
    header = {}
    offset = 0
    for key in keys:
        header[key] = {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [offset, offset + 4],
        }
        offset += 4
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + (b"\0" * offset))


def _make_valid_checkpoint(root: Path) -> None:
    main_key = "model.decoder.example.weight"
    main_name = "model-00001-of-000001.safetensors"
    fusion_name = "model-page_fusion_layer.safetensors"

    _write_fake_safetensors(root / main_name, [main_key])
    _write_fake_safetensors(root / fusion_name, sorted(REQUIRED_PAGE_FUSION_KEYS))

    weight_map = {main_key: main_name}
    weight_map.update({key: fusion_name for key in REQUIRED_PAGE_FUSION_KEYS})
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 20}, "weight_map": weight_map}),
        encoding="utf-8",
    )


def _make_valid_hf_repo(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_RUNTIME_FILES:
        if name != "config.json":
            (root / name).write_text("# test fixture\n", encoding="utf-8")
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "opticaldna",
                "auto_map": dict(REQUIRED_AUTO_MAP),
            }
        ),
        encoding="utf-8",
    )
    _make_valid_checkpoint(root)


def test_valid_sharded_checkpoint_passes(tmp_path: Path) -> None:
    _make_valid_checkpoint(tmp_path)
    report = inspect_checkpoint_weights(tmp_path)
    assert report["format"] == "sharded"
    assert report["num_tensors"] == 1 + len(REQUIRED_PAGE_FUSION_KEYS)
    assert report["total_tensor_bytes"] == 4 * (1 + len(REQUIRED_PAGE_FUSION_KEYS))


def test_missing_page_fusion_weight_fails(tmp_path: Path) -> None:
    main_name = "model-00001-of-000001.safetensors"
    main_key = "model.decoder.example.weight"
    _write_fake_safetensors(tmp_path / main_name, [main_key])
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {main_key: main_name}}), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="page-fusion weights are missing"):
        inspect_checkpoint_weights(tmp_path)


def test_index_and_shard_key_mismatch_fails(tmp_path: Path) -> None:
    _make_valid_checkpoint(tmp_path)
    fusion_path = tmp_path / "model-page_fusion_layer.safetensors"
    _write_fake_safetensors(
        fusion_path,
        sorted(REQUIRED_PAGE_FUSION_KEYS | {"model.page_fusion_layer.unexpected"}),
    )

    with pytest.raises(RuntimeError, match="Index/shard key mismatch"):
        inspect_checkpoint_weights(tmp_path)


def test_missing_referenced_shard_fails(tmp_path: Path) -> None:
    _make_valid_checkpoint(tmp_path)
    (tmp_path / "model-page_fusion_layer.safetensors").unlink()

    with pytest.raises(FileNotFoundError, match="missing shard"):
        inspect_checkpoint_weights(tmp_path)


def test_complete_hf_repo_contract_passes(tmp_path: Path) -> None:
    _make_valid_hf_repo(tmp_path)
    report = validate_hf_model_repo(tmp_path)
    assert report["model_dir"] == str(tmp_path)


def test_invalid_auto_map_fails(tmp_path: Path) -> None:
    _make_valid_hf_repo(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["auto_map"]["AutoModelForCausalLM"] = "wrong.module"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(RuntimeError, match="auto_map"):
        validate_hf_model_repo(tmp_path)


def test_missing_runtime_file_fails(tmp_path: Path) -> None:
    _make_valid_hf_repo(tmp_path)
    (tmp_path / "tokenizer.json").unlink()

    with pytest.raises(RuntimeError, match="Missing runtime files"):
        validate_hf_model_repo(tmp_path)
