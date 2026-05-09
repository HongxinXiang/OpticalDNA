import json
import os
import shutil
import time
from dataclasses import dataclass
from typing import Optional, List

from transformers import TrainerCallback
import logging


def _unwrap_model(m):
    return getattr(m, "module", m)


@dataclass
class _SavedItem:
    step: int
    path: str
    t: int


class ExportMergedEveryNStepsCallback(TrainerCallback):

    def __init__(
            self,
            output_dir: str,
            tokenizer,
            save_steps: int = 1000,
            top_k: int = 3,
            name_prefix: str = "unsloth_finetune",
            layer_name: str = "page_fusion_layer",
            subdir: str = "exports_merged",
            update_latest: bool = True,
            allow_partial: bool = True,
            copy_ignore_patterns: Optional[List[str]] = None,
            verbose: int = 0,
            log: Optional[logging.Logger] = None
    ):
        self.output_dir = output_dir
        self.tokenizer = tokenizer
        self.save_steps = int(save_steps)
        self.top_k = int(top_k)
        self.name_prefix = str(name_prefix)
        self.layer_name = str(layer_name)
        self.subdir = str(subdir)
        self.update_latest = bool(update_latest)
        self.allow_partial = bool(allow_partial)
        self.verbose = verbose
        self.log = log if log is not None else print

        self.copy_ignore_patterns = copy_ignore_patterns or []

        self._saved: List[_SavedItem] = []

    def _should_save_now(self, step: int) -> bool:
        return step > 0 and self.save_steps > 0 and (step % self.save_steps == 0)

    def _safe_rmtree(self, path: str) -> None:
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
        except Exception:
            pass

    def _copytree_atomic_replace(self, src_dir: str, dst_dir: str) -> None:
        parent = os.path.dirname(dst_dir)
        os.makedirs(parent, exist_ok=True)

        tmp_dir = f"{dst_dir}.tmp_{time.time_ns()}"
        self._safe_rmtree(tmp_dir)

        ignore = None
        if self.copy_ignore_patterns:
            ignore = shutil.ignore_patterns(*self.copy_ignore_patterns)

        shutil.copytree(src_dir, tmp_dir, dirs_exist_ok=False, ignore=ignore)

        self._safe_rmtree(dst_dir)
        os.replace(tmp_dir, dst_dir)

    def on_step_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return control

        step = int(getattr(state, "global_step", 0))
        if not self._should_save_now(step):
            return control

        model = kwargs.get("model", None)
        tokenizer = self.tokenizer
        if model is None or tokenizer is None:
            return control

        model_to_save = _unwrap_model(model)

        export_root = os.path.join(self.output_dir, self.subdir)
        os.makedirs(export_root, exist_ok=True)

        export_dir = os.path.join(export_root, f"{self.name_prefix}-step{step}")

        self._safe_rmtree(export_dir)

        try:
            if not hasattr(model_to_save, "save_pretrained_merged"):
                return control

            model_to_save.save_pretrained_merged(export_dir, tokenizer)
            try:
                patch_merged_add_page_fusion_layer(
                    model_to_save,
                    export_dir,
                    layer_name=self.layer_name,
                    verbose=self.verbose,
                    log=self.log
                )
            except Exception:
                if not self.allow_partial:
                    self._safe_rmtree(export_dir)
                    return control
        except Exception:
            self._safe_rmtree(export_dir)
            return control

        if self.update_latest:
            latest_dir = os.path.join(export_root, f"{self.name_prefix}-latest")
            try:
                self._copytree_atomic_replace(export_dir, latest_dir)
            except Exception:
                pass

        t = time.time_ns()
        self._saved.append(_SavedItem(step=step, path=export_dir, t=t))
        dedup = {}
        for item in self._saved:
            if (item.step not in dedup) or (item.t > dedup[item.step].t):
                dedup[item.step] = item
        self._saved = list(dedup.values())

        self._saved.sort(key=lambda x: (x.step, x.t), reverse=True)
        to_remove = self._saved[self.top_k:]
        self._saved = self._saved[: self.top_k]

        for item in to_remove:
            self._safe_rmtree(item.path)

        return control

    def on_train_end(self, args, state, control, **kwargs):
        if hasattr(args, "should_save") and (not args.should_save):
            return control
        if not state.is_world_process_zero:
            return control

        step = int(getattr(state, "global_step", 0))
        if self.save_steps <= 0 or step <= 0:
            return control

        last_step = self._saved[0].step if self._saved else -1
        if step != last_step:
            return self.on_step_end(args, state, control, **kwargs)

        return control


def _pick_fusion_tensor(sd: dict, suffix: str, layer_name: str = "page_fusion_layer"):
    prefer_suffix = f"{layer_name}.modules_to_save.default.{suffix}"
    original_suffix = f"{layer_name}.original_module.{suffix}"
    plain_suffix = f"{layer_name}.{suffix}"

    for k in sd.keys():
        if k.endswith(prefer_suffix):
            return k

    for k in sd.keys():
        if k.endswith(original_suffix):
            return k

    for k in sd.keys():
        if k.endswith(plain_suffix) or k == plain_suffix:
            return k

    return None


def _strip_prefix_until_layer(k: str, layer_name: str) -> str:
    token = f".{layer_name}."
    if k.startswith(f"{layer_name}."):
        return k
    if token in k:
        return layer_name + "." + k.split(token, 1)[1]
    return k


def _normalize_fusion_suffix(suffix: str) -> str:
    if suffix.startswith("modules_to_save.default."):
        return suffix[len("modules_to_save.default."):]
    if suffix.startswith("original_module."):
        return suffix[len("original_module."):]
    return suffix


def _fusion_priority_from_full_key(full_key: str, layer_name: str) -> int:
    if f"{layer_name}.modules_to_save.default." in full_key:
        return 3
    if f"{layer_name}.original_module." in full_key:
        return 2
    if f"{layer_name}." in full_key:
        return 1
    return 0


def patch_merged_add_page_fusion_layer(
        model,
        merged_dir: str,
        *,
        base_prefix: str = "model",
        layer_name: str = "page_fusion_layer",
        shard_name: str = "model-page_fusion_layer.safetensors",
        verbose: int = 0,
        log=None
):
    log = log if log is not None else print
    index_path = os.path.join(merged_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        raise RuntimeError(
            f"Not found: {index_path}. "
            "save_pretrained_merged may have exported a single model.safetensors file without an index."
        )

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    weight_map = index.get("weight_map", {})

    sd = model.state_dict()

    fusion_out = {}
    chosen_prio = {}
    hit = 0

    for k, v in sd.items():
        if layer_name not in k:
            continue
        hit += 1

        rel = _strip_prefix_until_layer(k, layer_name)
        suffix = rel.split(f"{layer_name}.", 1)[1] if rel.startswith(f"{layer_name}.") else rel

        norm_suffix = _normalize_fusion_suffix(suffix)

        dst_key = f"{base_prefix}.{layer_name}.{norm_suffix}"

        prio = _fusion_priority_from_full_key(rel, layer_name)
        if (dst_key not in fusion_out) or (prio > chosen_prio[dst_key]):
            fusion_out[dst_key] = v.detach().cpu()
            chosen_prio[dst_key] = prio

    if hit == 0:
        raise RuntimeError(
            f"Cannot find any key containing '{layer_name}' in model.state_dict(). "
            "No page_fusion_layer tensors were found."
        )

    try:
        from safetensors.torch import save_file
    except Exception as e:
        raise RuntimeError("Need safetensors. Please `pip install safetensors`.") from e

    shard_path = os.path.join(merged_dir, shard_name)
    save_file(fusion_out, shard_path)

    for k in fusion_out.keys():
        weight_map[k] = shard_name
    index["weight_map"] = weight_map

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    if verbose > 0:
        log(f"[patch_merged_add_page_fusion_layer] wrote: {shard_path}")
        log(f"[patch_merged_add_page_fusion_layer] patched index: {index_path}")
        log("[patch_merged_add_page_fusion_layer] added keys (after normalize + priority):")
        for k in sorted(fusion_out.keys()):
            log(f"  - {k}")

