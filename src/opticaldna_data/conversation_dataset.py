
from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from .pretrain_dataset import (
    DNAPretrainConversationDataset,
    LineSpanSamplerConfig,
    SubseqLocateSamplerConfig,
    AnnealedPromptCurriculumConfig,
)

__all__ = ["DNAConversationDataset", "DNAConversationPerPageDataset"]


class DNAConversationDataset(DNAPretrainConversationDataset):

    _VALID_TASKS = {
        "t1_full_ocr",
        "t2_full_ocr_grounding",
        "t3_roi_ocr",
        "t4_mask_completion",
        "t5_subseq_locate",
        "t6_chr_classification",
    }

    def __init__(
            self,
            *,
            task_name: str,
            split_size: Tuple[int, int] = (256, 256),
            seed: int = 0,
            prompt_length: Optional[str] = None,
            y_merge_tol: int = 6,
            line_span_cfg: Optional[LineSpanSamplerConfig] = None,
            subseq_locate_cfg: Optional[SubseqLocateSamplerConfig] = None,
            annealed_sampler_cfg: Optional[AnnealedPromptCurriculumConfig] = None,
            split_col: Optional[str] = None,
            split_values: Optional[str] = None,
            **kwargs,
    ) -> None:
        if task_name not in self._VALID_TASKS:
            raise ValueError(
                f"Unknown task_name={task_name!r}. Must be one of {sorted(self._VALID_TASKS)}"
            )
        self.task_name = task_name

        super().__init__(
            split_size=split_size,
            seed=seed,
            task_sampling=None,
            prompt_length=prompt_length,
            y_merge_tol=y_merge_tol,
            tail_truncation=None,
            line_span_cfg=line_span_cfg,
            subseq_locate_cfg=subseq_locate_cfg,
            annealed_sampler_cfg=annealed_sampler_cfg,
            split_col=split_col,
            split_values=split_values,
            **kwargs,
        )

        for attr in ("task_sampling", "tail_truncation", "tail_truncator"):
            if hasattr(self, attr):
                try:
                    delattr(self, attr)
                except Exception:
                    pass



    def set_seed(self, seed: int) -> None:
        self.rng = random.Random(int(seed))

    def set_task(self, task_name: str) -> None:
        if task_name not in self._VALID_TASKS:
            raise ValueError(
                f"Unknown task_name={task_name!r}. Must be one of {sorted(self._VALID_TASKS)}"
            )
        self.task_name = task_name

    def _extend_sample_from_row(self, row, sample):
        sample["chr_name"] = row.get("chr_name", "")
        sample["split"] = row.get("split", "")



    def pre_transform(self, images: List[Image.Image], metadata: Dict[str, Any]) -> Dict[str, Any]:
        assert len(images) == 1, (
            f"Expected at least one image, got len(images)={len(images)}"
        )
        merged = images[0]
        patches = self._splitter(merged, save=False)

        bbox_list_page = metadata.get("bbox").get_page_bbox()
        assert len(patches) == len(bbox_list_page), (
            f"Image splitter returned inconsistent patches and bounding boxes:"
            f"len(patches)={len(patches)}, len(bbox_list_page)={len(bbox_list_page)}"
        )

        task_name = self.task_name

        new_metadata = metadata

        meta = self._load_meta(new_metadata)
        lines = self._build_lines(meta)
        if lines:
            max_img_id = max(int(it["img_id"]) for it in lines)
            assert max_img_id < len(patches), (
                f"Line image ids exceed the number of pages:max_img_id={max_img_id}, n_pages={len(patches)}"
            )

        sample, task_enum = self._build_sample_for_task(
            task_name=task_name,
            image_groups=[patches],
            meta=meta,
            lines=lines,
        )

        prompt_len_enum = None
        if self.prompt_length is not None:
            pl = str(self.prompt_length).lower().strip()
            prompt_len_enum = getattr(self._PromptLength, pl.upper())

        conversation = self._builder(
            sample,
            task=task_enum,
            prompt_length=prompt_len_enum,
            step=self.cur_step,
        )
        self.cur_step += 1

        return {
            "conversation": conversation,
            "task_name": task_name,
        }






class DNAConversationPerPageDataset(DNAConversationDataset):

    def pre_transform(self, images: List[Image.Image], metadata: Dict[str, Any]) -> Dict[str, Any]:
        assert len(images) == 1, (
            f"Expected at least one image, got len(images)={len(images)}"
        )
        merged = images[0]
        patches = self._splitter(merged, save=False)

        bbox_list_page = metadata.get("bbox").get_page_bbox()
        assert len(patches) == len(bbox_list_page), (
            f"Image splitter returned inconsistent patches and bounding boxes:"
            f"len(patches)={len(patches)}, len(bbox_list_page)={len(bbox_list_page)}"
        )

        task_name = self.task_name

        full_meta = self._load_meta(metadata)
        full_lines = self._build_lines(full_meta)

        conversations: List[Any] = []
        for image_id, patch in enumerate(patches):
            page_index = image_id + 1
            page_lines = []
            for it in full_lines:
                if int(it.get("img_id", -1)) != image_id:
                    continue
                it2 = dict(it)
                it2["img_id"] = 0
                it2["page_index"] = 1
                page_lines.append(it2)

            page_meta = dict(full_meta)
            page_seq = "".join([str(x.get("seq", "")) for x in page_lines])
            page_meta["seq"] = page_seq
            new_bbox = {1: copy.deepcopy(page_meta['bbox'][page_index])}
            for page_meta_index in range(len(new_bbox[1])):
                new_bbox[1][page_meta_index]["char_index"] = page_meta_index
                new_bbox[1][page_meta_index]["page_index"] = 1
            page_meta['bbox'] = new_bbox

            sample, task_enum = self._build_sample_for_task(
                task_name=task_name,
                image_groups=[[patch]],
                meta=page_meta,
                lines=page_lines,
            )

            prompt_len_enum = None
            if self.prompt_length is not None:
                pl = str(self.prompt_length).lower().strip()
                prompt_len_enum = getattr(self._PromptLength, pl.upper())

            conv = self._builder(
                sample,
                task=task_enum,
                prompt_length=prompt_len_enum,
                step=self.cur_step,
            )
            self.cur_step += 1
            conversations.append(conv)

        return {"conversations": conversations, "task_name": task_name}


