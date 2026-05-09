
from __future__ import annotations

import re
import logging
from visualdna.utils.timing import time_block
import argparse
from pathlib import Path
from visualdna.utils.logger import redirect_stdouterr_to_tee
import os
from opticaldna_data.export_merged_every_n_steps_callback import ExportMergedEveryNStepsCallback, patch_merged_add_page_fusion_layer
import torch
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel
from transformers import Trainer, TrainingArguments
from unsloth import FastVisionModel
from unsloth import is_bf16_supported

from opticaldna_data.opticaldna_multigroup_collator import OpticalDNAMultiGroupDataCollator
from opticaldna_data.pretrain_dataset import (
    DNAPretrainConversationDataset,
    TaskSamplingConfig,
    TailTruncationConfig,
    LineSpanSamplerConfig,
    SubseqLocateSamplerConfig,
    AnnealedPromptCurriculumConfig,
)
from visualdna.render.base import BaseRenderConfig

from visualdna.utils.ddp_utils import (
    setup_distributed,
    suppress_print_if_not_main,
    init_rank0_logger,
    get_rank0_log_fn,
    save_args_json,
    barrier,
)
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
)
from transformers.trainer import _is_peft_model
import ast
from typing import Tuple
import hashlib
import json
from typing import Dict, Any, Union, Optional, List

"""
Set HF_HUB_OFFLINE=1 to force local checkpoint loading.
"""


class TaskAwareCollator:
    def __init__(self, base_collator):
        self.base_collator = base_collator

    @staticmethod
    def _get_task_name(item: Dict[str, Any]) -> str:

        for k in ("task_name", "task", "task_type", "task_id", "task_type_name"):
            v = item.get(k, None)
            if v is None:
                continue

            return str(v)
        return "unknown"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        task_names = [self._get_task_name(x) for x in features]
        batch = self.base_collator(features)
        batch["__task_name"] = task_names
        return batch


class TaskLossLoggingTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_taskloss_log_step: Optional[int] = None

    def _should_log_task_loss(self) -> bool:
        if getattr(self, "control", None) is not None and getattr(self.control, "should_log", False):
            return True

        gs = int(getattr(self.state, "global_step", 0) or 0)
        if gs == 0 and getattr(self.args, "logging_first_step", False):
            return True
        ls = int(getattr(self.args, "logging_steps", 0) or 0)
        return (ls > 0) and (gs % ls == 0)

    @torch.no_grad()
    def _compute_per_sample_ce_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        if logits.dim() != 3 or labels.dim() != 2:
            raise ValueError(f"logits/labels shape mismatch: logits={tuple(logits.shape)}, labels={tuple(labels.shape)}")


        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        bsz, seqm1 = shift_labels.shape
        vocab = shift_logits.shape[-1]


        ce = F.cross_entropy(
            shift_logits.reshape(-1, vocab).float(),
            shift_labels.reshape(-1),
            reduction="none",
            ignore_index=ignore_index,
        ).reshape(bsz, seqm1)


        mask = (shift_labels != ignore_index).to(ce.dtype)
        denom = mask.sum(dim=1).clamp_min(1.0)
        per_sample = (ce * mask).sum(dim=1) / denom
        return per_sample

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        task_names = inputs.pop("__task_name", None)

        labels_in_batch = inputs.get("labels", None)

        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.model_accepts_loss_kwargs:
            kwargs = {}
            if num_items_in_batch is not None:
                kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **kwargs}
        outputs = model(**inputs)





        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()

            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )

            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes

        try:
            if (
                task_names is not None
                and isinstance(task_names, list)
                and labels_in_batch is not None
                and self._should_log_task_loss()
            ):
                gs = int(getattr(self.state, "global_step", 0) or 0)
                if self._last_taskloss_log_step != gs:
                    if isinstance(outputs, dict):
                        logits = outputs.get("logits", None)
                    else:
                        logits = getattr(outputs, "logits", None)
                        if logits is None:
                            logits = outputs[1] if len(outputs) > 1 else None

                    if logits is not None:
                        bsz = int(logits.shape[0])
                        if len(task_names) == bsz and labels_in_batch.shape[0] == bsz:
                            per_sample_loss = self._compute_per_sample_ce_loss(
                                logits=logits.detach(),
                                labels=labels_in_batch.detach(),
                                ignore_index=-100,
                            )


                            task2vals: Dict[str, List[float]] = {}
                            for t, v in zip(task_names, per_sample_loss.tolist()):
                                task2vals.setdefault(str(t), []).append(float(v))

                            logs: Dict[str, float] = {}
                            def task_sort_key(t: str):
                                m = re.search(r"\d+", t)
                                return int(m.group()) if m else float("inf")
                            for t in sorted(task2vals.keys(), key=task_sort_key):
                                vals = task2vals[t]
                                logs[f"loss_task/{t}"] = float(sum(vals) / max(len(vals), 1))

                            self.log(logs)
                            self._last_taskloss_log_step = gs
        except Exception:
            pass

        return (loss, outputs) if return_outputs else loss


def _parse_kv_str(s: str) -> Dict[str, Any]:
    s = (s or "").strip()
    if not s:
        return {}

    out: Dict[str, Any] = {}
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if "=" not in p:
            raise ValueError(f"Bad kv item: '{p}' (expect key=value)")
        k, v = p.split("=", 1)
        k = k.strip()
        v = v.strip()

        low = v.lower()
        if low in ("true", "false"):
            out[k] = (low == "true")
            continue
        if low in ("none", "null"):
            out[k] = None
            continue

        try:
            out[k] = ast.literal_eval(v)
        except Exception:
            out[k] = v
    return out


def _cfg_get_str(cfg: Dict[str, Any], key: str, default: str = "") -> str:
    v = cfg.get(key, None)
    if v is None:
        return default
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        items = ",".join([f"{k}={repr(v[k])}" for k in sorted(v.keys())])
        return items
    raise TypeError(f"Config field '{key}' must be str or dict, got {type(v)}")


def _canon_kv_str(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    parts = [p.strip() for p in s.split(",") if p.strip()]
    kv = []
    for p in parts:
        if "=" not in p:
            raise ValueError(f"Bad kv item: '{p}' (expect key=value)")
        k, v = p.split("=", 1)
        kv.append((k.strip(), v.strip()))
    kv.sort(key=lambda x: x[0])
    return ",".join([f"{k}={v}" for k, v in kv])


def build_sampler_fingerprint(args) -> Dict[str, Any]:
    canon = {
        "task_sampling": _canon_kv_str(args.task_sampling),
        "tail_truncation": _canon_kv_str(args.tail_truncation),
        "line_span_cfg": _canon_kv_str(args.line_span_cfg),
        "subseq_locate_cfg": _canon_kv_str(args.subseq_locate_cfg),
    }
    payload = json.dumps(canon, ensure_ascii=False, sort_keys=True)
    sampler_id = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]
    return {"sampler_id": sampler_id, "sampler_payload": canon, "sampler_payload_json": payload}




def _enable_trainable_params_by_keywords(_model, keywords, *, strict: bool = True, log = None) -> None:
    log = log if log is not None else print
    if isinstance(keywords, str):
        keywords = [keywords]
    matched = []
    for name, p in _model.named_parameters():
        if any(k in name for k in keywords):
            p.requires_grad = True
            matched.append(name)
    if strict and not matched:
        raise ValueError(
            f"[TrainConfig] No parameters matched keywords={keywords}. "
            "Please check whether the requested module name appears in named_parameters()."
        )
    log(f"[TrainConfig] Enabled trainable params for keywords={keywords}. Matched {len(matched)} tensors.")
    for n in matched[:20]:
        log(f"  - {n}")


def build_sampling_configs(args) -> Tuple[
    "TaskSamplingConfig",
    "TailTruncationConfig",
    "LineSpanSamplerConfig",
    "SubseqLocateSamplerConfig",
]:
    task_sampling = TaskSamplingConfig(**_parse_kv_str(args.task_sampling))
    tail_truncation = TailTruncationConfig(**_parse_kv_str(args.tail_truncation))
    line_span_cfg = LineSpanSamplerConfig(**_parse_kv_str(args.line_span_cfg))
    subseq_locate_cfg = SubseqLocateSamplerConfig(**_parse_kv_str(args.subseq_locate_cfg))
    return task_sampling, tail_truncation, line_span_cfg, subseq_locate_cfg


def _sanity_check_one_batch(dataset, collator, tokenizer, out_dir: str = "outputs", log = None):
    log = log if log is not None else print

    from pathlib import Path
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    items = [dataset[i] for i in range(min(2, len(dataset)))]
    batch = collator(items)

    input_ids = batch["input_ids"]
    images_seq_mask = batch["images_seq_mask"]
    images = batch["images"]

    stats = {
        "batch_size": int(input_ids.shape[0]),
        "seq_len": int(input_ids.shape[1]),
        "image_token_counts": [int(images_seq_mask[i].sum().item()) for i in range(int(input_ids.shape[0]))],
        "images_0_images_crop_shape": list(images[0][0].shape) if len(images) > 0 else None,
        "images_0_images_ori_shape": list(images[0][1].shape) if len(images) > 0 else None,
    }

    try:
        decoded = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=False)
        stats["decoded_input_ids_0_head"] = decoded[:1500]
    except Exception as e:
        stats["decoded_input_ids_0_head_error"] = str(e)

    out_json = Path(out_dir) / "sanity_batch_stats.json"
    out_json.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"[SanityCheck] Wrote: {out_json}")
    log(f"[SanityCheck] image_token_counts = {stats["image_token_counts"]}")
    log(f"[SanityCheck] images[0] crop/ori shapes = {stats["images_0_images_crop_shape"]} {stats["images_0_images_ori_shape"]}")


def assert_only_lora_and_fusion_trainable(model, trainable_modules_to_save=None, log=None):
    log = log if log is not None else print
    trainable_modules_to_save = trainable_modules_to_save or []

    allow_substrings = ["lora_", "page_fusion_layer"]
    allow_substrings.extend([x for x in trainable_modules_to_save if x])

    bad = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            ok = any(s in n for s in allow_substrings)
            if not ok:
                bad.append(n)

    if bad:
        log("[TrainConfig] Unexpected trainable params (show first 50):")
        for x in bad[:50]:
            log(f"  - {x}")
        log(f"[TrainConfig] Allowed substrings = {allow_substrings}")
        raise RuntimeError("Found unexpected trainable params; freeze them to avoid DDP unused issues.")





@dataclass(frozen=True)
class LoraGroupSpec:
    prefixes: Tuple[str, ...]
    linear_suffixes: Tuple[str, ...]
    enabled_arg: str


LORA_GROUPS: Dict[str, LoraGroupSpec] = {

    "sam": LoraGroupSpec(
        prefixes=("model.sam_model.",),
        linear_suffixes=("qkv", "proj", "lin1", "lin2"),
        enabled_arg="lora_sam",
    ),

    "clip_vit": LoraGroupSpec(
        prefixes=("model.vision_model.",),
        linear_suffixes=("qkv_proj", "out_proj", "fc1", "fc2"),
        enabled_arg="lora_clip_vit",
    ),

    "decoder": LoraGroupSpec(
        prefixes=("model.layers.",),
        linear_suffixes=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
        enabled_arg="lora_decoder",
    ),

    "projector": LoraGroupSpec(
        prefixes=("model.projector.",),
        linear_suffixes=("layers",),
        enabled_arg="lora_projector",
    ),
}


def _is_allowed_prefix(name: str, prefixes: Tuple[str, ...]) -> bool:
    return any(name.startswith(p) for p in prefixes)


def _is_allowed_suffix(name: str, suffixes: Tuple[str, ...]) -> bool:
    return any(name.endswith("." + s) for s in suffixes)


def collect_lora_target_modules(
    model: nn.Module,
    enabled_groups: Sequence[str],
    log: Callable[[str], None] = print,
) -> List[str]:
    targets: List[str] = []

    for group_name in enabled_groups:
        if group_name not in LORA_GROUPS:
            raise KeyError(f"Unknown LoRA group: {group_name}. Available: {list(LORA_GROUPS.keys())}")

        spec = LORA_GROUPS[group_name]
        n_hit = 0

        for name, m in model.named_modules():
            if not _is_allowed_prefix(name, spec.prefixes):
                continue
            if not isinstance(m, nn.Linear):
                continue
            if _is_allowed_suffix(name, spec.linear_suffixes):
                targets.append(name)
                n_hit += 1

        log(f"[LoRACollect] group={group_name}, hits={n_hit}")

    targets = sorted(set(targets))
    return targets


def assert_lora_hits_by_group_strict(
    model: nn.Module,
    enabled_groups: Sequence[str],
    log: Callable[[str], None] = print,
) -> Dict[str, List[str]]:
    stats: Dict[str, List[str]] = {g: [] for g in LORA_GROUPS.keys()}

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_" not in n:
            continue

        def _param_prefix_match(param_name: str, prefixes: Tuple[str, ...]) -> bool:
            wrappers = (
                "", "base_model.", "base_model.model.", "base_model.model.model.", "base_model.model.model.model.")
            return any(param_name.startswith(w + p) for w in wrappers for p in prefixes)

        for g, spec in LORA_GROUPS.items():
            if _param_prefix_match(n, spec.prefixes):
                stats[g].append(n)
                break

    for g in enabled_groups:
        hits = len(stats[g])
        log(f"[LoRAAssert] group={g}, hits={hits}")
        if hits == 0:
            raise RuntimeError(
                f"LoRA group '{g}' enabled but got 0 lora_ parameters. "
                f"Check LORA_GROUPS['{g}'] prefixes/suffixes."
            )

    for g in stats.keys():
        if g not in enabled_groups and len(stats[g]) > 0:
            log(f"[LoRAAssert][WARN] group={g} is disabled but has hits={len(stats[g])} (possible mismatch).")

    return stats


def build_grouped_lora_targets(model: nn.Module, args: argparse.Namespace, log=print) -> Tuple[List[str], List[str]]:
    enabled_groups: List[str] = []
    for g, spec in LORA_GROUPS.items():
        if getattr(args, spec.enabled_arg, False):
            enabled_groups.append(g)

    if len(enabled_groups) == 0:
        raise ValueError("All LoRA switches are disabled: no LoRA groups selected.")

    log(f"[LoRA] enabled_groups = {enabled_groups}")

    target_modules = collect_lora_target_modules(model, enabled_groups, log=log)
    if len(target_modules) == 0:
        raise RuntimeError(
            f"Enabled LoRA groups = {enabled_groups}, but no nn.Linear modules matched. "
            "Please check prefixes/suffixes in LORA_GROUPS."
        )

    log(f"[LoRA] target_modules (n={len(target_modules)})")
    for x in target_modules[:50]:
        log(f"  - {x}")
    if len(target_modules) > 50:
        log(f"  ... (and {len(target_modules) - 50} more)")

    return enabled_groups, target_modules




def _load_config_file(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"--config not found: {path}")

    suffix = p.suffix.lower()
    text = p.read_text(encoding="utf-8")

    if suffix in [".json"]:
        return json.loads(text)

    if suffix in [".yml", ".yaml"]:
        try:
            import yaml
        except Exception as e:
            raise RuntimeError("YAML config requires PyYAML. Please `pip install pyyaml`.") from e
        return yaml.safe_load(text) or {}

    raise ValueError(f"Unsupported config file type: {suffix}. Use .json/.yml/.yaml")


def parse_args() -> argparse.Namespace:

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None, help="Path to JSON/YAML config file.")
    pre.add_argument(
        "--backend",
        type=str,
        default="nccl",
        choices=["nccl", "gloo"],
        help="Distributed backend. Use 'gloo' for debugging, 'nccl' for multi-GPU training.",
    )
    pre_args, _ = pre.parse_known_args()

    cfg: Dict[str, Any] = {}
    if pre_args.config:
        cfg = _load_config_file(pre_args.config)

    parser = argparse.ArgumentParser(
        description=(
            "Distributed training script for OpticalDNA pre-training.\n"
            "Supports single-node multi-GPU training via torch.distributed (DDP).\n"
            "All arguments can be provided via CLI or a JSON/YAML config file; "
            "CLI arguments always override values loaded from the config."
        )
    )




    g_io = parser.add_argument_group("I/O & Paths")
    g_io.add_argument("--config", type=str, default=pre_args.config,
                      help="Path to JSON/YAML config file (CLI overrides config).")
    g_io.add_argument("--model_name", type=str, default=cfg.get("model_name", "./unsloth_hf_model_new"),
                      help="Path or HuggingFace model name for the base vision-language model to load.")
    g_io.add_argument("--dataroot", type=str,
                      default=cfg.get("dataroot", "/path/to/opticaldna_data"),
                      help="Root directory containing all VisualDNA pretraining datasets (images, metadata, annotations, indexes).")
    g_io.add_argument("--dataset", type=str,
                      default=cfg.get("dataset", "hg38-2048"),
                      help="Dataset name (subdirectory under --dataroot) to use for training, e.g. 'hg38-2048'.")
    g_io.add_argument("--output_dir", type=str, default=cfg.get("output_dir", "./outputs/train_hg38_2048_ddp_rank0log"),
                      help="Output dir for logs/checkpoints/args.json (rank0 writes).")




    g_sys = parser.add_argument_group("Distributed & System")
    g_sys.add_argument("--backend", type=str, default=pre_args.backend, choices=["nccl", "gloo"],
                       help="DDP backend: 'nccl' for CUDA, 'gloo' for debug/CPU.")
    g_sys.add_argument("--seed", type=int, default=cfg.get("seed", 2025),
                       help="Global random seed shared across DDP ranks.")




    g_data = parser.add_argument_group("Data Pipeline")
    g_data.add_argument("--read_bbox", action="store_true", default=bool(cfg.get("read_bbox", False)),
                        help="Load bounding-box (bbox) annotations during training.")
    g_data.add_argument("--cache_bbox", action="store_true", default=bool(cfg.get("cache_bbox", False)),
                        help="Cache BBoxReader objects into dataset metadata (self.samples). "
                             "If enabled, bbox paths will be replaced by BBoxReader instances "
                             "during __getitem__. This may improve speed for small datasets, "
                             "but can significantly increase memory usage with multiple workers.")
    g_data.add_argument("--dataloader_num_workers", type=int, default=cfg.get("dataloader_num_workers", 16),
                        help="Workers per GPU/process (total = num_gpus × workers).")
    g_data.add_argument("--dataloader_prefetch_factor", type=int, default=1,
                        help="Prefetch factor per worker; set 1 to reduce CPU RAM peak.")
    g_data.add_argument("--dataloader_persistent_workers", action="store_true", default=False,
                        help="Keep workers alive across epochs (off by default for stability).")
    g_data.add_argument("--dataloader_pin_memory", action="store_true",
                        default=bool(cfg.get("dataloader_pin_memory", False)),
                        help="Enable pin_memory for faster H2D transfer.")

    g_data.add_argument("--task_sampling", type=str,
                        default=_cfg_get_str(cfg, "task_sampling", "p_t1=0.25,p_t2=0.2,p_t3=0.15,p_t4=0.15,p_t5=0.15,p_t6=0.10"),
                        help=("Task sampling parameters as key-value pairs. "
                              'Example: --task_sampling "p_t1=0.25,p_t2=0.2,p_t3=0.15,p_t4=0.15,p_t5=0.15,p_t6=0.10". '
                              "Unspecified fields use dataclass defaults."))
    g_data.add_argument("--tail_truncation", type=str,
                        default=_cfg_get_str(cfg, "tail_truncation", "enabled=true,base_delete_ratio=0,max_delete_ratio=0.5"),
                        help=("Tail truncation parameters as key-value pairs. "
                              'Example: --tail_truncation "enabled=true,base_delete_ratio=0,max_delete_ratio=0.5"')
                        )
    g_data.add_argument("--line_span_cfg", type=str,
                        default=_cfg_get_str(cfg, "line_span_cfg", "min_n_base=1,max_n_base=8,min_n_sample=1,max_n_sample=3,unique_lines=true"),
                        help=("Line span sampler parameters as key-value pairs. "
                              'Example: --line_span_cfg "min_n_base=1,max_n_base=8,min_n_sample=1,max_n_sample=3,unique_lines=true"')
                        )
    g_data.add_argument("--subseq_locate_cfg", type=str,
                        default=_cfg_get_str(cfg, "subseq_locate_cfg", "min_len=6,max_len=64,allow_overlap=true"),
                        help=("Subsequence localization parameters as key-value pairs. "
                              'Example: --subseq_locate_cfg "min_len=6,max_len=64,allow_overlap=true"'),
                        )




    g_model = parser.add_argument_group("Model / Input Limits")
    g_model.add_argument("--max_text_tokens", type=int, default=cfg.get("max_text_tokens", 4096),
                         help="Max text tokens per sample (truncate longer) to control memory usage.")
    g_model.add_argument("--lora_sam", action="store_true", default = bool(cfg.get("lora_sam", False)),
                         help = "Apply LoRA to SAM vision backbone.")
    g_model.add_argument("--lora_clip_vit", action="store_true", default = bool(cfg.get("lora_clip_vit", False)),
                         help = "Apply LoRA to CLIP-ViT encoder.")
    g_model.add_argument("--lora_decoder", action="store_true", default = bool(cfg.get("lora_decoder", False)),
                         help = "Apply LoRA to decoder (LLM) layers.")
    g_model.add_argument("--lora_projector", action="store_true", default=bool(cfg.get("lora_projector", False)),
                         help="Apply LoRA to vision projector (model.projector.layers).")
    g_model.add_argument("--lora_r", type=int, default=16, help="LoRA rank (r) for all LoRA modules.")
    g_model.add_argument("--trainable_modules_to_save", type=str, default="page_fusion_layer",
                         help="Comma-separated module names to save alongside LoRA weights, e.g. 'page_fusion_layer,xxx'.")



    g_train = parser.add_argument_group("Training Schedule")
    g_train.add_argument("--per_device_train_batch_size", type=int,
                         default=cfg.get("per_device_train_batch_size", 2 * 4),
                         help="Batch size per GPU (global scales with GPUs and grad accumulation).")
    g_train.add_argument("--gradient_accumulation_steps", type=int, default=cfg.get("gradient_accumulation_steps", 1),
                         help="Accumulate gradients for N steps before optimizer update.")
    g_train.add_argument("--max_steps", type=int, default=cfg.get("max_steps", None),
                         help="Total training steps; if None, derive automatically.")
    g_train.add_argument("--warmup_steps", type=int, default=cfg.get("warmup_steps", 15_000),
                         help="Warmup steps for LR schedule.")
    g_train.add_argument("--annealed_sampler_total_steps", type=int,
                         default=cfg.get("annealed_sampler_total_steps", None),
                         help="Total steps for annealed sampler (defaults to max_steps).")




    g_opt = parser.add_argument_group("Optimization")
    g_opt.add_argument("--learning_rate", type=float, default=cfg.get("learning_rate", 2e-4),
                       help="Initial learning rate.")
    g_opt.add_argument("--optim", type=str, default=cfg.get("optim", "adamw_8bit"),
                       help="Optimizer type (e.g., adamw, adamw_8bit).")
    g_opt.add_argument("--weight_decay", type=float, default=cfg.get("weight_decay", 0.001),
                       help="Weight decay coefficient.")
    g_opt.add_argument("--lr_scheduler_type", type=str, default=cfg.get("lr_scheduler_type", "linear"),
                       help="LR scheduler type.")




    g_log = parser.add_argument_group("Logging & Checkpoint")
    g_log.add_argument("--logging_steps", type=int, default=cfg.get("logging_steps", 1_00),
                       help="Log every N steps (rank0 only).")
    g_log.add_argument("--save_steps", type=int, default=cfg.get("save_steps", 1_000),
                       help="Save checkpoint every N steps (rank0 only).")
    g_log.add_argument("--save_total_limit", type=int, default=cfg.get("save_total_limit", 5),
                       help="Max checkpoints to keep (delete older).")

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:


    is_dist, local_rank, rank, world_size = setup_distributed(args.backend)

    suppress_print_if_not_main(rank)

    if rank == 0:
        redirect_stdouterr_to_tee(Path(args.output_dir) / "train.log", save_progress=False)


    logger = init_rank0_logger(output_dir=args.output_dir, name=__name__, rank=rank)
    log = get_rank0_log_fn(logger, rank)
    if rank == 0 and logger is not None:
        for h in logger.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                h.stream = sys.__stdout__

    log(f"[DDP] is_dist={is_dist}, backend={args.backend}, "
        f"local_rank={local_rank}, rank={rank}, world_size={world_size}, "
        f"cuda_available={torch.cuda.is_available()}")



    output_dir = args.output_dir

    seed = args.seed
    dataloader_num_workers = args.dataloader_num_workers
    dataloader_pin_memory = args.dataloader_pin_memory

    read_bbox = args.read_bbox
    cache_bbox = args.cache_bbox
    max_text_tokens = args.max_text_tokens
    per_device_train_batch_size = args.per_device_train_batch_size


    if args.max_steps is None:
        max_steps = 18_000_000 // per_device_train_batch_size // world_size * 2
    else:
        max_steps = args.max_steps

    if args.annealed_sampler_total_steps is None:
        annealed_sampler_total_steps = int(max_steps * 0.2)
    else:
        annealed_sampler_total_steps = args.annealed_sampler_total_steps

    gradient_accumulation_steps = args.gradient_accumulation_steps
    warmup_steps = args.warmup_steps
    learning_rate = args.learning_rate
    logging_steps = args.logging_steps
    optim = args.optim
    weight_decay = args.weight_decay
    lr_scheduler_type = args.lr_scheduler_type

    save_steps = args.save_steps
    save_total_limit = args.save_total_limit

    if rank == 0:
        argv_str = " ".join([sys.executable] + sys.argv)
        log("[Launch] argv:\n" + argv_str)

        env_keys = [
            "CUDA_VISIBLE_DEVICES",
            "LD_PRELOAD",
            "NCCL_DEBUG",
            "NCCL_IB_DISABLE",
            "NCCL_SOCKET_IFNAME",
            "NCCL_P2P_DISABLE",
            "NCCL_ASYNC_ERROR_HANDLING",
            "MASTER_ADDR",
            "MASTER_PORT",
            "RANK",
            "LOCAL_RANK",
            "WORLD_SIZE",
            "LOCAL_WORLD_SIZE",
            "TORCHELASTIC_RUN_ID",
        ]
        env_dump = {k: os.environ.get(k) for k in env_keys if os.environ.get(k) is not None}
        log("[Launch] env snapshot:\n" + json.dumps(env_dump, ensure_ascii=False, indent=2))

        args_dump: Dict[str, Any] = vars(args).copy()
        args_dump["world_size"] = world_size
        args_dump["local_rank"] = local_rank
        args_dump["rank"] = rank
        args_dump["max_steps_effective"] = max_steps
        args_dump["annealed_sampler_total_steps_effective"] = annealed_sampler_total_steps

        sampler_fp = build_sampler_fingerprint(args)
        sampler_id = sampler_fp["sampler_id"]
        log(f"[Sampler] sampler_id={sampler_id}")
        log(f"[Sampler] task_sampling={sampler_fp['sampler_payload']['task_sampling']}")
        log(f"[Sampler] tail_truncation={sampler_fp['sampler_payload']['tail_truncation']}")
        log(f"[Sampler] line_span_cfg={sampler_fp['sampler_payload']['line_span_cfg']}")
        log(f"[Sampler] subseq_locate_cfg={sampler_fp['sampler_payload']['subseq_locate_cfg']}")
        args_dump["sampler_fingerprint"] = sampler_fp
        args_dump["sampler_id"] = sampler_id
        args_dump["sampler_payload"] = sampler_fp["sampler_payload"]

        saved_path = save_args_json(args_dump, output_dir, filename="args.json")
        log(f"[Args] saved to: {saved_path}")
        log("[Args] effective hyperparameters:\n" + json.dumps(args_dump, ensure_ascii=False, indent=2))

    barrier()




    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True



    os.environ["UNSLOTH_WARN_UNINITIALIZED"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"



    model, tokenizer = FastVisionModel.from_pretrained(
        args.model_name,
        load_in_4bit=False,
        auto_model=AutoModel,
        trust_remote_code=True,
        unsloth_force_compile=True,
        use_gradient_checkpointing="unsloth",
        device_map={"": local_rank},
    )
    log("[Model] loaded FastVisionModel & tokenizer.")




    enabled_groups, target_modules = build_grouped_lora_targets(model, args, log=log)
    trainable_modules_to_save = [x.strip() for x in args.trainable_modules_to_save.split(",") if x.strip()]

    model = FastVisionModel.get_peft_model(
        model=model,
        target_modules=target_modules,
        r=args.lora_r,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        random_state=seed,
        use_rslora=False,
        loftq_config=None,
        modules_to_save=trainable_modules_to_save,
    )
    log(model)

    for train_layer in trainable_modules_to_save:
        _enable_trainable_params_by_keywords(model, train_layer, strict=True, log=log)
    assert_only_lora_and_fusion_trainable(model, trainable_modules_to_save, log)
    assert_lora_hits_by_group_strict(model, enabled_groups, log=log)

    trainable = 0
    total = 0
    _n_list = []
    for _n, _p in model.named_parameters():
        total += _p.numel()
        if _p.requires_grad:
            trainable += _p.numel()
            _n_list.append(_n)
    log(f"[trainable parameters] {_n_list}")
    log(f"[TrainConfig] Trainable params: {trainable:,} / {total:,} ({trainable /total:.4%})")

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})



    dataroot = args.dataroot
    dataset = args.dataset
    config = BaseRenderConfig(
        img_width=640,
        img_height=640,
        font_size=14,
        line_spacing=1.6,
        merge_pages=True,
        save_bbox=True,
    )
    img_root = f"{dataroot}/{dataset}/processed/{config.to_dirname()}"


    dataset_dict = {}
    for split_values in ["train"]:
        with time_block(f"[{split_values}] Init DNAPretrainConversationDataset", log):
            task_sampling, tail_truncation, line_span_cfg, subseq_locate_cfg = build_sampling_configs(args)
            dataset = DNAPretrainConversationDataset(
                split_size=(config.img_width, config.img_height),
                seed=seed,
                task_sampling=task_sampling,
                tail_truncation=tail_truncation,
                line_span_cfg=line_span_cfg,
                subseq_locate_cfg=subseq_locate_cfg,
                annealed_sampler_cfg=AnnealedPromptCurriculumConfig(
                    total_steps=annealed_sampler_total_steps,
                    seed=seed,
                    start=(0.2, 0.2, 0.6),
                    end=(0.1, 0.1, 0.8),
                ),
                split_col="split",
                split_values=split_values,
                root=img_root,
                lazy=True,
                read_bbox=read_bbox,
                cache_bbox=cache_bbox
            )
            dataset_dict[split_values] = dataset
        log(f"[{split_values}] Dataset size = {len(dataset)}")

    data_collator = OpticalDNAMultiGroupDataCollator(
        tokenizer=tokenizer,
        model=model,
        image_size=640,
        base_size=640,
        crop_mode=False,
        train_on_responses_only=True,
        fuse_shards_in_group=True,
        max_text_tokens=max_text_tokens,
    )

    data_collator = TaskAwareCollator(data_collator)



    FastVisionModel.for_training(model)

    _sanity_check_one_batch(dataset, data_collator, tokenizer, out_dir=f"{output_dir}/sanity_check_one_batch", log=log)

    callbacks = [
        ExportMergedEveryNStepsCallback(
            output_dir=output_dir,
            tokenizer=tokenizer,
            save_steps=save_steps * 5,
            top_k=save_total_limit,
            name_prefix="unsloth_finetune",
            layer_name="page_fusion_layer",
            update_latest=True,
            verbose=1,
            log=log,

        )
    ]
    trainer = TaskLossLoggingTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=dataset_dict["train"],

        callbacks=callbacks,
        args=TrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            learning_rate=learning_rate,
            logging_steps=logging_steps,
            optim=optim,
            weight_decay=weight_decay,
            lr_scheduler_type=lr_scheduler_type,
            seed=seed,
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            dataloader_num_workers=dataloader_num_workers,
            dataloader_prefetch_factor=args.dataloader_prefetch_factor,
            dataloader_persistent_workers=args.dataloader_persistent_workers,
            remove_unused_columns=False,
            output_dir=output_dir,
            report_to="tensorboard",
            logging_dir=f"{output_dir}/tb",
            logging_first_step=True,
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            ddp_find_unused_parameters=True,
            dataloader_pin_memory=dataloader_pin_memory,

        ),
    )


    gpu_stats = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    if gpu_stats is not None:
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        log(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        log(f"{start_gpu_memory} GB of memory reserved.")
    else:
        start_gpu_memory = 0.0
        max_memory = 0.0

    trainer_stats = trainer.train()

    if gpu_stats is not None:
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3) if max_memory > 0 else 0.0
        lora_percentage = round(used_memory_for_lora / max_memory * 100, 3) if max_memory > 0 else 0.0
        log(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
        log(f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.")
        log(f"Peak reserved memory = {used_memory} GB.")
        log(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
        log(f"Peak reserved memory % of max memory = {used_percentage} %.")
        log(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")



    model.save_pretrained(f"{output_dir}/lora_model")
    tokenizer.save_pretrained(f"{output_dir}/lora_model")

    if True:
        model.save_pretrained_merged(f"{output_dir}/unsloth_finetune", tokenizer)
        patch_merged_add_page_fusion_layer(model, f"{output_dir}/unsloth_finetune", layer_name="page_fusion_layer", shard_name="model-page_fusion_layer.safetensors", log=log)

    # test evaluation (It's too slow)
    # if "test" in dataset_dict.keys():
    #     test_metrics = trainer.evaluate(eval_dataset=dataset_dict["test"], metric_key_prefix="test")
    #     log(test_metrics)


if __name__ == "__main__":
    args = parse_args()
    main(args)
