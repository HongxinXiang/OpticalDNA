"""Checkpoint validation helpers for OpticalDNA releases."""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from typing import Dict, Mapping, Set, Tuple, Union

PathLike = Union[str, os.PathLike]

REQUIRED_PAGE_FUSION_KEYS = {
    "model.page_fusion_layer.attn.0.in_proj_weight",
    "model.page_fusion_layer.attn.0.in_proj_bias",
    "model.page_fusion_layer.attn.0.out_proj.weight",
    "model.page_fusion_layer.attn.0.out_proj.bias",
}

REQUIRED_RUNTIME_FILES = {
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "modeling_opticaldna.py",
    "modeling_deepseekocr.py",
    "modeling_deepseekv2.py",
    "configuration_deepseek_v2.py",
    "deepencoder.py",
    "conversation.py",
    "multi_page_fusion.py",
    "inference.py",
    "prompts.py",
}

REQUIRED_AUTO_MAP = {
    "AutoConfig": "modeling_opticaldna.OpticalDNAConfig",
    "AutoModel": "modeling_opticaldna.OpticalDNAForCausalLM",
    "AutoModelForCausalLM": "modeling_opticaldna.OpticalDNAForCausalLM",
}


def _read_safetensors_header(path: Path) -> Mapping[str, object]:
    """Read only the safetensors header; tensor payloads are never materialized."""
    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise RuntimeError(f"Invalid safetensors file (missing header length): {path}")
        header_len = struct.unpack("<Q", raw)[0]
        header_raw = f.read(header_len)
    try:
        header = json.loads(header_raw.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Invalid safetensors header: {path}") from exc
    if not isinstance(header, dict):
        raise RuntimeError(f"Invalid safetensors header object: {path}")
    return header


def safetensors_keys_and_nbytes(path: PathLike) -> Tuple[Set[str], int]:
    path = Path(path)
    header = _read_safetensors_header(path)
    keys: Set[str] = set()
    total_nbytes = 0
    for key, value in header.items():
        if key == "__metadata__":
            continue
        if not isinstance(value, dict) or "data_offsets" not in value:
            raise RuntimeError(f"Malformed tensor entry {key!r} in {path}")
        offsets = value["data_offsets"]
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise RuntimeError(f"Malformed data_offsets for {key!r} in {path}")
        start, end = int(offsets[0]), int(offsets[1])
        if end < start:
            raise RuntimeError(f"Invalid data_offsets for {key!r} in {path}")
        keys.add(key)
        total_nbytes += end - start
    return keys, total_nbytes


def inspect_checkpoint_weights(checkpoint_dir: PathLike) -> Dict[str, object]:
    """Validate the checkpoint's safetensors files and return a compact report."""
    root = Path(checkpoint_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {root}")

    index_path = root / "model.safetensors.index.json"
    single_path = root / "model.safetensors"

    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as f:
            index = json.load(f)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise RuntimeError(f"Invalid or empty weight_map: {index_path}")

        mapped_by_shard: Dict[str, Set[str]] = {}
        for key, shard_name in weight_map.items():
            if not isinstance(key, str) or not isinstance(shard_name, str):
                raise RuntimeError(f"Non-string weight_map entry in {index_path}")
            mapped_by_shard.setdefault(shard_name, set()).add(key)

        total_nbytes = 0
        actual_union: Set[str] = set()
        for shard_name, mapped_keys in sorted(mapped_by_shard.items()):
            shard_path = root / shard_name
            if not shard_path.is_file():
                raise FileNotFoundError(
                    f"Checkpoint index references a missing shard: {shard_path}"
                )
            actual_keys, shard_nbytes = safetensors_keys_and_nbytes(shard_path)
            missing_in_shard = mapped_keys - actual_keys
            extra_in_shard = actual_keys - mapped_keys
            if missing_in_shard or extra_in_shard:
                raise RuntimeError(
                    f"Index/shard key mismatch for {shard_name}: "
                    f"missing_in_shard={sorted(missing_in_shard)[:20]}, "
                    f"extra_in_shard={sorted(extra_in_shard)[:20]}"
                )
            actual_union.update(actual_keys)
            total_nbytes += shard_nbytes

        weight_keys = set(weight_map)
        if actual_union != weight_keys:
            raise RuntimeError("Checkpoint index does not exactly match shard tensor keys.")
        format_name = "sharded"
        shard_names = sorted(mapped_by_shard)
    elif single_path.exists():
        weight_keys, total_nbytes = safetensors_keys_and_nbytes(single_path)
        index = None
        format_name = "single"
        shard_names = [single_path.name]
    else:
        raise FileNotFoundError(
            f"No model.safetensors or model.safetensors.index.json found in {root}"
        )

    missing_fusion = REQUIRED_PAGE_FUSION_KEYS - weight_keys
    if missing_fusion:
        raise RuntimeError(
            "OpticalDNA page-fusion weights are missing from the checkpoint: "
            + ", ".join(sorted(missing_fusion))
            + ". Refusing to allow random initialization."
        )

    return {
        "format": format_name,
        "weight_keys": weight_keys,
        "num_tensors": len(weight_keys),
        "total_tensor_bytes": total_nbytes,
        "shards": shard_names,
        "index": index,
    }


def validate_hf_model_repo(model_dir: PathLike) -> Dict[str, object]:
    """Validate a local directory before upload or strict loading from Hugging Face."""
    root = Path(model_dir)
    missing_files = sorted(name for name in REQUIRED_RUNTIME_FILES if not (root / name).is_file())
    if missing_files:
        raise RuntimeError(
            f"Incomplete OpticalDNA Hugging Face repository at {root}. "
            f"Missing runtime files: {missing_files}"
        )

    with (root / "config.json").open("r", encoding="utf-8") as f:
        config = json.load(f)
    if config.get("model_type") != "opticaldna":
        raise RuntimeError("config.json must contain model_type='opticaldna'.")
    auto_map = config.get("auto_map", {})
    for key, expected in REQUIRED_AUTO_MAP.items():
        if auto_map.get(key) != expected:
            raise RuntimeError(
                f"config.json auto_map[{key!r}] must be {expected!r}; "
                f"got {auto_map.get(key)!r}."
            )

    report = inspect_checkpoint_weights(root)
    report["model_dir"] = str(root)
    return report
