from __future__ import annotations

import json
from pathlib import Path
import copy
import random
from typing import Any, Dict, List, Tuple, Union, Optional

from PIL import Image
from dataclasses import dataclass

from visualdna.render.base import BaseRenderConfig, BaseTextToImageGenerator
from visualdna.render.bbox import ImageBBoxVisualizer
from visualdna.data.reader.base_reader import AbsReader, CSVMixin
from visualdna.data.reader.bbox_mask_reader import BBoxMaskBase
from visualdna.data.reader.splitable_reader import SplitableReader
from visualdna.utils.ocr_utils import sort_boxes_and_seqs
from visualdna.render.bbox import BBoxReader

Image.MAX_IMAGE_PIXELS = None

try:
    from .conversation_builder import (
        ConversationBuilder,
        DNASampleFactory,
        PromptGenerator,
        TaskType,
        PromptLength,
        AnnealedPromptCurriculum,
        PromptCurriculumSampler
    )
except Exception:
    from .conversation_builder import (
        ConversationBuilder,
        DNASampleFactory,
        PromptGenerator,
        TaskType,
        PromptLength,
        AnnealedPromptCurriculum,
        PromptCurriculumSampler
    )


def _union_xyxy(boxes_xyxy: List[List[int]]) -> List[int]:
    x1 = min(b[0] for b in boxes_xyxy)
    y1 = min(b[1] for b in boxes_xyxy)
    x2 = max(b[2] for b in boxes_xyxy)
    y2 = max(b[3] for b in boxes_xyxy)
    return [int(x1), int(y1), int(x2), int(y2)]


def build_bbox_list_page_from_boxes5(boxes5: List[List[int]]) -> Dict[int, List[Dict[str, Any]]]:
    bbox_list_page: Dict[int, List[Dict[str, Any]]] = {}
    for b in boxes5:
        if not (isinstance(b, (list, tuple)) and len(b) == 5):
            raise ValueError(f"Invalid box5: {b}")
        img_id, x1, y1, x2, y2 = map(int, b)
        page_id = img_id + 1
        bbox_list_page.setdefault(page_id, []).append({"page_bbox": [x1, y1, x2, y2]})
    return bbox_list_page


def _sample_roi(
        rng: random.Random,
        lines: List[Dict[str, Any]],
        k: int = 2,
) -> Tuple[List[List[int]], List[str]]:
    if not lines:
        return [], []
    chosen = rng.sample(lines, k=min(k, len(lines)))
    boxes5, seqs = [], []
    for it in chosen:
        img_id = int(it["img_id"])
        x1, y1, x2, y2 = map(int, it["bbox"])
        boxes5.append([img_id, x1, y1, x2, y2])
        seqs.append(str(it["seq"]))
    return boxes5, seqs


@dataclass(frozen=True)
class TailTruncationConfig:
    enabled: bool = True
    randomize: bool = True

    base_delete_ratio: float = 0.0
    base_delete_lines: int = 0

    max_delete_ratio: float = 0.10
    max_delete_lines: int = 0

    apply_tasks: Optional[Tuple[str, ...]] = (
        "t2_full_ocr_grounding",
        "t3_roi_ocr",
        "t5_subseq_locate",
    )
    allow_force: bool = True


@dataclass
class TailTruncator:
    cfg: TailTruncationConfig

    def is_enabled_for_task(self, task_name: Optional[str], *, force: bool = False) -> bool:
        if not self.cfg.enabled:
            return False
        if force:
            return bool(self.cfg.allow_force)

        if self.cfg.apply_tasks is None:
            return True

        if task_name is None:
            return False

        return task_name in set(self.cfg.apply_tasks)

    def __call__(
            self,
            *,
            patches: List[Image.Image],
            bbox_list_page: Dict[int, List[Dict[str, Any]]],
            rng: random.Random,
            task_name: Optional[str] = None,
            force: bool = False,
    ) -> Tuple[List[Image.Image], Dict[int, List[Dict[str, Any]]], int]:
        if not self.is_enabled_for_task(task_name, force=force):
            return patches, bbox_list_page, -1

        if not bbox_list_page:
            return patches, bbox_list_page, 0

        global_boxes: List[Tuple[int, Dict[str, Any]]] = []
        for page_id in sorted(bbox_list_page.keys()):
            for b in bbox_list_page[page_id]:
                global_boxes.append((page_id, b))

        total_lines = len(global_boxes)
        if total_lines == 0:
            return patches, bbox_list_page, 0

        def _cap_by_ratio_lines(total: int, ratio: float, lines: int) -> int:
            by_ratio = int(total * max(0.0, min(1.0, ratio)))
            if lines and lines > 0:
                return min(by_ratio, int(lines))
            return by_ratio

        base_del_cap = _cap_by_ratio_lines(total_lines, self.cfg.base_delete_ratio, self.cfg.base_delete_lines)
        base_del = max(0, min(total_lines, base_del_cap))

        remain_after_base = total_lines - base_del
        if remain_after_base <= 0:
            return [], {}, 0

        extra_cap = _cap_by_ratio_lines(total_lines, self.cfg.max_delete_ratio, self.cfg.max_delete_lines)
        extra_cap = max(0, min(remain_after_base, extra_cap))

        if extra_cap > 0:
            extra_del = rng.randint(0, extra_cap) if self.cfg.randomize else extra_cap
        else:
            extra_del = 0

        delete_n = base_del + extra_del
        keep_n = total_lines - delete_n

        if keep_n <= 0:
            return [], {}, 0

        kept_global = global_boxes[:keep_n]
        tail_global = global_boxes[keep_n:]

        new_patches = [img.copy() for img in patches]

        for page_id, b in tail_global:
            pid = page_id - 1
            if pid < 0 or pid >= len(new_patches):
                continue

            if "page_bbox" not in b:
                continue

            x1, y1, x2, y2 = map(int, b["page_bbox"])
            if x2 > x1 and y2 > y1:
                new_patches[pid].paste((255, 255, 255), (x1, y1, x2, y2))

        tmp_bbox_page: Dict[int, List[Dict[str, Any]]] = {}
        for page_id, b in kept_global:
            tmp_bbox_page.setdefault(page_id, []).append(b)

        final_patches: List[Image.Image] = []
        new_bbox_list_page: Dict[int, List[Dict[str, Any]]] = {}

        new_page_id = 1
        for old_page_id in sorted(tmp_bbox_page.keys()):
            boxes = tmp_bbox_page[old_page_id]
            if not boxes:
                continue

            old_pid = old_page_id - 1
            if old_pid < 0 or old_pid >= len(new_patches):
                continue

            final_patches.append(new_patches[old_pid])
            new_bbox_list_page[new_page_id] = boxes
            new_page_id += 1

        if len(final_patches) != len(new_bbox_list_page):
            raise AssertionError(
                f"[TailTruncator] patches ({len(final_patches)}) != bbox_list_page ({len(new_bbox_list_page)})"
            )

        return final_patches, new_bbox_list_page, keep_n


@dataclass(frozen=True)
class LineSpanSamplerConfig:
    min_n_base: int = 1
    max_n_base: int = 32

    min_n_sample: int = 1
    max_n_sample: int = 8

    unique_lines: bool = True


@dataclass(frozen=True)
class LineSpanSampler:
    cfg: LineSpanSamplerConfig

    def _validate_params(
            self,
            *,
            min_n_base: int,
            max_n_base: int,
            min_n_sample: int,
            max_n_sample: int,
    ) -> None:
        if min_n_sample < 1 or max_n_sample < 1:
            raise ValueError("LineSpanSampler: min_n_sample/max_n_sample must be >= 1.")
        if min_n_base < 1 or max_n_base < 1:
            raise ValueError("LineSpanSampler: min_n_base/max_n_base must be >= 1.")
        if max_n_sample < min_n_sample:
            raise ValueError("LineSpanSampler: max_n_sample must be >= min_n_sample.")
        if max_n_base < min_n_base:
            raise ValueError("LineSpanSampler: max_n_base must be >= min_n_base.")

    def _valid_lines(self, lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        valid: List[Dict[str, Any]] = []
        for ln in lines:
            if not isinstance(ln, dict):
                continue
            if "img_id" not in ln:
                continue
            chars = ln.get("char_boxes", None)
            if not isinstance(chars, list) or len(chars) == 0:
                continue
            valid.append(ln)
        return valid

    def _sample_span_from_line(
            self,
            rng: random.Random,
            line: Dict[str, Any],
            *,
            min_n_base: int,
            max_n_base: int,
    ) -> Tuple[List[int], str]:
        chars: List[Dict[str, Any]] = line["char_boxes"]
        n = len(chars)
        if n <= 0:
            raise ValueError("LineSpanSampler: line['char_boxes'] is empty, cannot sample.")

        lo = max(1, int(min_n_base))
        hi = max(lo, int(max_n_base))
        L = rng.randint(lo, hi)
        L = min(L, n)

        st = rng.randint(0, n - L)
        span = chars[st: st + L]

        text = "".join(str(c["char"]) for c in span)
        assert len(text) == len(text.strip())
        if not text:
            raise ValueError("LineSpanSampler: sampled span text is empty.")

        span_boxes: List[List[int]] = []
        for c in span:
            bb = c.get("page_bbox", None)
            if bb is None or not (isinstance(bb, (list, tuple)) and len(bb) == 4):
                raise ValueError(f"LineSpanSampler: invalid char page_bbox={bb}")
            span_boxes.append([int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])])

        bbox = _union_xyxy(span_boxes)
        return bbox, text

    def sample(
            self,
            *,
            rng: random.Random,
            lines: List[Dict[str, Any]],
            min_n_base: Optional[int] = None,
            max_n_base: Optional[int] = None,
            min_n_sample: Optional[int] = None,
            max_n_sample: Optional[int] = None,
            unique_lines: Optional[bool] = None,
    ) -> Tuple[List[List[int]], List[str]]:
        valid = self._valid_lines(lines)
        if not valid:
            raise ValueError("LineSpanSampler: no valid lines with non-empty char_boxes.")

        _min_n_base = self.cfg.min_n_base if min_n_base is None else int(min_n_base)
        _max_n_base = self.cfg.max_n_base if max_n_base is None else int(max_n_base)
        _min_n_sample = self.cfg.min_n_sample if min_n_sample is None else int(min_n_sample)
        _max_n_sample = self.cfg.max_n_sample if max_n_sample is None else int(max_n_sample)
        _unique_lines = self.cfg.unique_lines if unique_lines is None else bool(unique_lines)

        self._validate_params(
            min_n_base=_min_n_base,
            max_n_base=_max_n_base,
            min_n_sample=_min_n_sample,
            max_n_sample=_max_n_sample,
        )

        n_sample = rng.randint(_min_n_sample, _max_n_sample)

        if _unique_lines:
            n_sample = min(n_sample, len(valid))
            chosen_indices = rng.sample(range(len(valid)), k=n_sample)
        else:
            chosen_indices = [rng.randrange(len(valid)) for _ in range(n_sample)]

        boxes5: List[List[int]] = []
        seqs: List[str] = []
        for idx in chosen_indices:
            line = valid[idx]
            bbox, text = self._sample_span_from_line(
                rng, line, min_n_base=_min_n_base, max_n_base=_max_n_base
            )
            img_id = int(line["img_id"])
            boxes5.append([img_id, bbox[0], bbox[1], bbox[2], bbox[3]])
            seqs.append(text)

        return boxes5, seqs


@dataclass(frozen=True)
class SubseqLocateSamplerConfig:
    min_len: int = 6
    max_len: int = 64

    allow_overlap: bool = True


@dataclass(frozen=True)
class AnnealedPromptCurriculumConfig:
    total_steps: int
    seed: int = 0

    start: Tuple[float, float, float] = (0.80, 0.15, 0.05)

    end: Tuple[float, float, float] = (0.30, 0.30, 0.40)

    def to_kwargs(self) -> dict:
        return {
            "total_steps": self.total_steps,
            "seed": self.seed,
            "start": self.start,
            "end": self.end,
        }


@dataclass(frozen=True)
class SubseqLocateSamplerV2:
    cfg: SubseqLocateSamplerConfig

    def _validate_params(self, *, min_len: int, max_len: int) -> None:
        if min_len < 1 or max_len < 1:
            raise ValueError("SubseqLocateSamplerV2: min_len/max_len must be >= 1.")
        if max_len < min_len:
            raise ValueError("SubseqLocateSamplerV2: max_len must be >= min_len.")

    def _find_all_occurrences(self, s: str, sub: str, *, allow_overlap: bool) -> List[int]:
        if not sub:
            return []
        out: List[int] = []
        start = 0
        step = 1 if allow_overlap else len(sub)
        while True:
            i = s.find(sub, start)
            if i < 0:
                break
            out.append(i)
            start = i + step
        return out

    def _sample_query_from_line(
            self,
            rng: random.Random,
            line: Dict[str, Any],
            *,
            min_len: int,
            max_len: int,
    ) -> str:
        seq = str(line.get("seq", ""))
        if not seq:
            raise ValueError("SubseqLocateSamplerV2: sampled line has empty seq.")

        if len(seq) <= min_len:
            return seq

        L = rng.randint(min_len, min(max_len, len(seq)))
        st = rng.randint(0, len(seq) - L)
        query = seq[st: st + L]
        if not query:
            raise ValueError("SubseqLocateSamplerV2: sampled empty query (unexpected).")
        return query

    def _span_bbox_from_line(self, line: Dict[str, Any], *, st: int, ed: int) -> List[int]:
        char_boxes = line.get("char_boxes", None)
        if not isinstance(char_boxes, list) or len(char_boxes) == 0:
            raise ValueError("SubseqLocateSamplerV2: line missing non-empty 'char_boxes'.")

        if ed > len(char_boxes):
            raise ValueError(
                f"SubseqLocateSamplerV2: seq/char_boxes mismatch: ed={ed} len(char_boxes)={len(char_boxes)}"
            )

        span = char_boxes[st:ed]
        span_boxes: List[List[int]] = []
        for c in span:
            bb = c.get("page_bbox", None)
            if bb is None or not (isinstance(bb, (list, tuple)) and len(bb) == 4):
                raise ValueError(f"SubseqLocateSamplerV2: invalid char page_bbox={bb}")
            span_boxes.append([int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])])

        return _union_xyxy(span_boxes)

    def sample(
            self,
            *,
            rng: random.Random,
            lines: List[Dict[str, Any]],
            min_len: Optional[int] = None,
            max_len: Optional[int] = None,
            allow_overlap: Optional[bool] = None,
    ) -> Tuple[str, List[List[int]]]:

        _min_len = self.cfg.min_len if min_len is None else int(min_len)
        _max_len = self.cfg.max_len if max_len is None else int(max_len)
        _allow_overlap = self.cfg.allow_overlap if allow_overlap is None else bool(allow_overlap)

        self._validate_params(min_len=_min_len, max_len=_max_len)

        if not lines:
            return "N" * _min_len, []

        src = rng.choice(lines)
        query = self._sample_query_from_line(rng, src, min_len=_min_len, max_len=_max_len)

        boxes5: List[List[int]] = []
        qlen = len(query)

        for it in lines:
            it_seq = str(it.get("seq", ""))
            if not it_seq or query not in it_seq:
                continue

            starts = self._find_all_occurrences(it_seq, query, allow_overlap=_allow_overlap)
            if not starts:
                continue

            img_id = int(it.get("img_id", 0))
            for st in starts:
                ed = st + qlen
                bbox = self._span_bbox_from_line(it, st=st, ed=ed)
                boxes5.append([img_id, bbox[0], bbox[1], bbox[2], bbox[3]])

        return query, boxes5


@dataclass(frozen=True)
class TaskSamplingConfig:
    p_t1: float = 0.25
    p_t2: float = 0.20
    p_t3: float = 0.15
    p_t4: float = 0.15
    p_t5: float = 0.15
    p_t6: float = 0.10

    def sample(self, rng: random.Random) -> str:
        ps = [
            ("t1_full_ocr", self.p_t1),
            ("t2_full_ocr_grounding", self.p_t2),
            ("t3_roi_ocr", self.p_t3),
            ("t4_mask_completion", self.p_t4),
            ("t5_subseq_locate", self.p_t5),
            ("t6_chr_classification", self.p_t6),
        ]
        s = sum(p for _, p in ps)
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"Task probs must sum to 1.0, got {s}")
        r = rng.random()
        acc = 0.0
        for name, p in ps:
            acc += p
            if r <= acc:
                return name
        return ps[-1][0]


class DNAPretrainConversationDataset(BBoxMaskBase):
    def __init__(
            self,
            split_size=(256, 256),
            seed: int = 0,
            task_sampling: Optional[TaskSamplingConfig] = None,
            prompt_length: Optional[str] = None,
            y_merge_tol: int = 6,
            tail_truncation: Optional[TailTruncationConfig] = None,
            line_span_cfg: Optional[LineSpanSamplerConfig] = None,
            subseq_locate_cfg: Optional[SubseqLocateSamplerConfig] = None,
            annealed_sampler_cfg: Optional[AnnealedPromptCurriculumConfig] = None,
            split_col: Optional[str] = None,
            split_values: Optional[str] = None,
            **kwargs
    ):
        self.split_size = split_size
        self.rng = random.Random(seed)
        self.task_sampling = task_sampling or TaskSamplingConfig()
        self.prompt_length = prompt_length
        self.y_merge_tol = y_merge_tol
        self.split_col, self.split_values = split_col, split_values
        self.tail_truncation = tail_truncation or TailTruncationConfig()
        self.tail_truncator = TailTruncator(cfg=self.tail_truncation)
        self.line_span_sampler = LineSpanSampler(
            cfg=LineSpanSamplerConfig(
                min_n_base=1,
                max_n_base=32,
                min_n_sample=1,
                max_n_sample=8,
                unique_lines=True,
            ) if line_span_cfg is None else line_span_cfg
        )
        self.subseq_locate_sampler = SubseqLocateSamplerV2(
            cfg=SubseqLocateSamplerConfig(
                min_len=6,
                max_len=64,
                allow_overlap=True,
            ) if subseq_locate_cfg is None else subseq_locate_cfg
        )
        self.annealed_sampler = AnnealedPromptCurriculum(**(AnnealedPromptCurriculumConfig(
            total_steps=100_000).to_kwargs() if annealed_sampler_cfg is None else annealed_sampler_cfg.to_kwargs()))

        self._splitter = BaseTextToImageGenerator(
            img_width=split_size[0],
            img_height=split_size[1],
            **kwargs
        ).split_vertical

        self._builder, self._factory, self._TaskType, self._PromptLength = self._init_builder()
        super().__init__(**kwargs)
        self.cur_step = 0

    def _init_builder(self):
        prompt_gen = PromptGenerator()
        factory = DNASampleFactory()

        builder = ConversationBuilder(prompt_generator=prompt_gen, fixed_sampler=None,
                                      annealed_sampler=self.annealed_sampler, validate_fields=True)
        return builder, factory, TaskType, PromptLength

    def load_index(self):
        return self._load_index_from_csv(self.csv_file, read_bbox=self.read_bbox, split_col=self.split_col,
                                         split_values=self.split_values)

    def _extend_sample_from_row(self, row, sample):
        sample["chr_name"] = row["chr_name"]

    def _sample_mask(self, pages, bbox_list_page):
        masked_images: List[Image.Image] = []
        masks: List[Any] = []
        mask_bboxes: List[Any] = []

        for idx, patch in enumerate(pages):
            page_id = idx + 1
            W, H = patch.size
            if page_id not in bbox_list_page:
                masked_images.append(patch)
                masks.append(Image.new("L", (W, H), 0))
                mask_bboxes.append([])
                continue

            bbox_list = bbox_list_page.get(page_id, []) or []
            if len(bbox_list) == 0:
                masked_images.append(patch)
                masks.append(Image.new("L", (W, H), 0))
                mask_bboxes.append([])
                continue

            result = self._apply_bbox_mask_all(patch, bbox_list, bbox_key="page_bbox")
            masked_images.append(result["masked_image"])
            masks.append(result["mask_matrix"])
            mask_bboxes.append(result["mask_bboxes"])

        return masked_images, masks, mask_bboxes

    def _sample_subseq_locate(
            self,
            rng: random.Random,
            lines: List[Dict[str, Any]],
            min_len: int = 6,
            max_len: int = 16,
    ) -> Tuple[str, List[List[int]]]:
        if not lines:
            return "N" * min_len, []
        src = rng.choice(lines)
        seq = str(src["seq"])
        if len(seq) <= min_len:
            query = seq
        else:
            L = rng.randint(min_len, min(max_len, len(seq)))
            st = rng.randint(0, len(seq) - L)
            query = seq[st: st + L]

        boxes5: List[List[int]] = []
        for it in lines:
            if query and query in str(it["seq"]):
                img_id = int(it["img_id"])
                x1, y1, x2, y2 = map(int, it["bbox"])
                boxes5.append([img_id, x1, y1, x2, y2])
        return query, boxes5

    def _build_sample_for_task(
            self,
            *,
            task_name: str,
            image_groups: List[List[Union[str, Image.Image]]],
            meta: Dict[str, Any],
            lines: List[Dict[str, Any]],
    ):
        TaskType = self._TaskType

        if not isinstance(image_groups, list) or len(image_groups) == 0:
            raise ValueError("image_groups must be a non-empty List[List[str|Image.Image]]")
        if not all(isinstance(g, list) and len(g) > 0 for g in image_groups):
            raise ValueError("Each image group must be a non-empty list.")
        assert len(image_groups) == 1
        image_field = image_groups[0]

        seq: str = str(meta.get("seq", "") or "")

        seq_lines = [str(it.get("seq", "")) for it in lines]
        if "".join(seq_lines) != seq:
            raise ValueError(
                "meta['seq'] != concat(lines[*]['seq']). "
                f"len(meta_seq)={len(seq)} len(lines_concat)={len(''.join(seq_lines))} "
                f"task={task_name}"
            )

        chr_label = meta.get("chr_name", None)
        if chr_label is None:
            chr_label = meta.get("chr_label", None)
        if chr_label is not None:
            chr_label = str(chr_label)

        if task_name == "t1_full_ocr":
            sample = self._factory.t1_full_ocr(image=image_field, seq_lines=seq_lines)
            return sample, TaskType.T1_FULL_OCR

        if task_name == "t6_chr_classification":
            if chr_label is not None and chr_label != "":
                sample = self._factory.t6_chr_classification(image=image_field, chr_label=chr_label)
                return sample, TaskType.T6_CHR_CLASSIFICATION

            sample = self._factory.t1_full_ocr(image=image_field, seq_lines=seq_lines)
            return sample, TaskType.T1_FULL_OCR

        if not lines:
            sample = self._factory.t1_full_ocr(image=image_field, seq_lines=seq_lines)
            return sample, TaskType.T1_FULL_OCR

        def _line_img_id(it: Dict[str, Any]) -> int:
            if "img_id" in it:
                return int(it["img_id"])
            if "page_index" in it:
                return int(it["page_index"]) - 1
            raise KeyError("Line item missing 'img_id' or 'page_index'.")

        def _line_bbox(it: Dict[str, Any]) -> List[int]:
            b = it.get("bbox", None)
            if b is None:
                raise KeyError("Line item missing 'bbox'.")
            x1, y1, x2, y2 = b
            return [int(x1), int(y1), int(x2), int(y2)]

        if task_name == "t2_full_ocr_grounding":
            seqs = [str(it.get("seq", "")) for it in lines]
            boxes5 = [[_line_img_id(it), *_line_bbox(it)] for it in lines]
            for b in boxes5: assert 0 <= int(b[0]) < len(image_field), (b[0], len(image_field))
            sample = self._factory.t2_full_ocr_grounding(image=image_field, seqs=seqs, boxes=boxes5)
            return sample, TaskType.T2_FULL_OCR_GROUNDING

        if task_name == "t3_roi_ocr":

            boxes5, seqs = self.line_span_sampler.sample(rng=self.rng, lines=lines)
            boxes5, seqs = sort_boxes_and_seqs(boxes5, seqs)
            for b in boxes5: assert 0 <= int(b[0]) < len(image_field), (b[0], len(image_field))
            sample = self._factory.t3_roi_ocr(image=image_field, boxes=boxes5, seqs=seqs)
            return sample, TaskType.T3_ROI_OCR

        if task_name == "t4_mask_completion":

            boxes5, pred_seqs = self.line_span_sampler.sample(rng=self.rng, lines=lines)
            boxes5, pred_seqs = sort_boxes_and_seqs(boxes5, pred_seqs)
            for b in boxes5: assert 0 <= int(b[0]) < len(image_field), (b[0], len(image_field))

            if not isinstance(image_field, list) or len(image_field) == 0:
                raise ValueError("t4_mask_completion: image_field must be a non-empty list of page images.")
            bbox_list_page = build_bbox_list_page_from_boxes5(boxes5)
            masked_pages, masks, _ = self._sample_mask(image_field, bbox_list_page)
            sample = self._factory.t4_mask_completion(image=masked_pages, boxes=boxes5, pred_seqs=pred_seqs)
            sample["masks"] = masks
            return sample, TaskType.T4_MASK_COMPLETION

        if task_name == "t5_subseq_locate":

            query, boxes5 = self.subseq_locate_sampler.sample(rng=self.rng, lines=lines)
            for b in boxes5: assert 0 <= int(b[0]) < len(image_field), (b[0], len(image_field))
            sample = self._factory.t5_subseq_locate(image=image_field, query=query, boxes=boxes5)
            return sample, TaskType.T5_SUBSEQ_LOCATE

        raise ValueError(f"Unknown task_name: {task_name}")

    def _load_meta(self, metadata):
        meta = {
            "index": metadata["index"],
            "bbox": metadata["bbox"].page_bbox,
            "chr_name": metadata["chr_name"],
            "seq": "".join(
                [metadata["bbox"].merged_bbox[i]["char"] for i in range(len(metadata["bbox"].merged_bbox))]),
        }
        return meta

    def _build_lines_from_char_boxes(
            self,
            char_boxes: Union[Dict[int, List[Dict[str, Any]]], List[Dict[str, Any]]],
            *,
            y_merge_tol: int,
            strict: bool = True,
            sort_cleaned: bool = False
    ) -> List[Dict[str, Any]]:
        cleaned: List[Tuple[int, int, int, int, int, str]] = []

        if isinstance(char_boxes, dict):
            page_items_iter = sorted(char_boxes.items(), key=lambda kv: int(kv[0]))
        elif isinstance(char_boxes, list):
            tmp: Dict[int, List[Dict[str, Any]]] = {}
            for it in char_boxes:
                if not isinstance(it, dict):
                    continue
                if "page_index" not in it:
                    continue
                tmp.setdefault(int(it["page_index"]), []).append(it)
            page_items_iter = sorted(tmp.items(), key=lambda kv: int(kv[0]))
        else:
            raise TypeError("char_boxes must be Dict[int, List[dict]] or List[dict].")

        for page_id, items in page_items_iter:

            if strict and int(page_id) < 1:
                raise ValueError(f"page_id must be 1-based int. Got page_id={page_id}")

            if not items:
                if strict:
                    raise ValueError(f"Empty char_boxes list for page_id={page_id}.")
                continue

            for j, c in enumerate(items):
                if not isinstance(c, dict):
                    if strict:
                        raise TypeError(
                            f"Invalid char box item type at page_id={page_id}, idx={j}: {type(c)}"
                        )
                    continue

                if "page_bbox" not in c or "char" not in c:
                    if strict:
                        raise KeyError(
                            f"Missing required keys in char box at page_id={page_id}, idx={j}. "
                            f"Required: 'page_bbox' and 'char'. Got keys={list(c.keys())}"
                        )
                    continue

                bb = c["page_bbox"]
                if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
                    if strict:
                        raise ValueError(
                            f"Invalid page_bbox at page_id={page_id}, idx={j}: {bb} (expected len=4)."
                        )
                    continue

                x1, y1, x2, y2 = bb
                page0 = int(page_id) - 1
                cleaned.append((page0, int(x1), int(y1), int(x2), int(y2), str(c["char"])))

        if not cleaned:
            return []

        if sort_cleaned:
            cleaned.sort(key=lambda t: (t[0], t[2], t[1]))

        lines: List[Dict[str, Any]] = []
        cur_page = cleaned[0][0]
        cur_y = cleaned[0][2]
        cur_chars: List[Tuple[int, int, int, int, str]] = []

        def _union_bbox_xyxy(bboxes_xyxy: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
            x1 = min(b[0] for b in bboxes_xyxy)
            y1 = min(b[1] for b in bboxes_xyxy)
            x2 = max(b[2] for b in bboxes_xyxy)
            y2 = max(b[3] for b in bboxes_xyxy)
            return x1, y1, x2, y2

        def flush():
            nonlocal cur_chars, cur_page
            if not cur_chars:
                return

            seq = "".join(ch for *_xyxy, ch in cur_chars)
            assert len(seq) == len(seq.strip())
            if not seq:
                cur_chars = []
                return

            bbox = _union_bbox_xyxy([(x1, y1, x2, y2) for x1, y1, x2, y2, _ in cur_chars])

            char_boxes_in_line: List[Dict[str, Any]] = []
            for i, (x1, y1, x2, y2, ch) in enumerate(cur_chars):
                char_boxes_in_line.append(
                    {
                        "char": ch,
                        "page_bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "char_index": int(i),
                    }
                )

            page0 = int(cur_page)
            page1 = page0 + 1

            lines.append(
                {
                    "img_id": page0,
                    "page_index": page1,
                    "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                    "seq": seq,
                    "char_boxes": char_boxes_in_line,
                }
            )
            cur_chars = []

        for page0, x1, y1, x2, y2, ch in cleaned:
            if page0 != cur_page:
                flush()
                cur_page = page0
                cur_y = y1

            if abs(y1 - cur_y) > y_merge_tol and cur_chars:
                flush()
                cur_y = y1

            cur_chars.append((x1, y1, x2, y2, ch))

        flush()
        return lines

    def _build_lines(self, meta: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if meta is None:
            return []

        if isinstance(meta, dict) and meta.get("line_items"):
            return meta["line_items"]

        if isinstance(meta, dict) and meta.get("bbox") is not None:
            return self._build_lines_from_char_boxes(meta["bbox"], y_merge_tol=self.y_merge_tol)

        return []

    def pre_transform(self, images: List[Image.Image], metadata: Dict[str, Any]) -> Dict[str, Any]:
        assert len(images) == 1

        patches = self._splitter(
            merged_image=images[0],
            save=False
        )
        bbox_list_page = metadata.get("bbox").get_page_bbox()
        assert isinstance(bbox_list_page, dict)
        assert len(patches) == len(bbox_list_page), (
            f"Expected non-empty patches and bbox_list_page."
            f" got len(patches)={len(patches)}, len(bbox_list_page)={len(bbox_list_page)}"
        )

        task_name = self.task_sampling.sample(self.rng)

        patches, bbox_list_page, keep_n = self.tail_truncator(
            patches=patches,
            bbox_list_page=bbox_list_page,
            rng=self.rng,
            task_name=task_name
        )
        new_metadata = copy.deepcopy(metadata)
        if keep_n > 0:
            new_metadata["bbox"].merged_bbox = new_metadata["bbox"].merged_bbox[:keep_n]
            new_metadata["bbox"].page_bbox = bbox_list_page

        meta = self._load_meta(new_metadata)
        lines = self._build_lines(meta)

        if lines:
            max_img_id = max(int(it["img_id"]) for it in lines)
            assert max_img_id < len(patches), (max_img_id, len(patches))

        sample, task_enum = self._build_sample_for_task(
            task_name=task_name,
            image_groups=[patches],
            meta=meta,
            lines=lines
        )

        prompt_len_enum = None
        if self.prompt_length is not None:
            pl = str(self.prompt_length).lower().strip()
            prompt_len_enum = getattr(self._PromptLength, pl.upper())

        conversation = self._builder(sample, task=task_enum, prompt_length=prompt_len_enum, step=self.cur_step)
        self.cur_step += 1

        return {"conversation": conversation, "task_name": task_name}





