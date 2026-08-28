"""Public prompt builders for OpticalDNA decoder-style inference."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List


class TaskType(str, Enum):
    """Prompted pretraining / inference task types used by OpticalDNA."""

    T1_FULL_OCR = "t1_full_ocr"
    T2_FULL_OCR_GROUNDING = "t2_full_ocr_grounding"
    T3_ROI_OCR = "t3_roi_ocr"
    T4_MASK_COMPLETION = "t4_mask_completion"
    T5_SUBSEQ_LOCATE = "t5_subseq_locate"
    T6_CHR_CLASSIFICATION = "t6_chr_classification"


class PromptLength(str, Enum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


@dataclass(frozen=True)
class PromptGenerator:
    """Construct OpticalDNA decoder prompts.

    This mirrors the prompt families used during OpticalDNA pretraining while
    providing a small public surface for released-checkpoint inference.
    """

    def build(
        self,
        task: TaskType | str,
        length: PromptLength | str = PromptLength.SHORT,
        sample: Dict[str, Any] | None = None,
    ) -> str:
        task = TaskType(task)
        length = PromptLength(length)
        sample = sample or {}
        if task == TaskType.T1_FULL_OCR:
            return self._t1(length)
        if task == TaskType.T2_FULL_OCR_GROUNDING:
            return self._t2(length)
        if task == TaskType.T3_ROI_OCR:
            boxes = sample.get("boxes")
            if boxes is None:
                raise KeyError("T3 requires sample={'boxes': [[img_id,x1,y1,x2,y2], ...]}")
            return self._t3(length, boxes)
        if task == TaskType.T4_MASK_COMPLETION:
            boxes = sample.get("boxes")
            if boxes is None:
                raise KeyError("T4 requires sample={'boxes': [[img_id,x1,y1,x2,y2], ...]}")
            return self._t4(length, boxes)
        if task == TaskType.T5_SUBSEQ_LOCATE:
            query = sample.get("query")
            if query is None:
                raise KeyError("T5 requires sample={'query': 'ACGT...'}")
            return self._t5(length, query)
        if task == TaskType.T6_CHR_CLASSIFICATION:
            return self._t6(length)
        raise ValueError(f"Unsupported task: {task}")

    def free_ocr(self, length: PromptLength | str = PromptLength.SHORT) -> str:
        """Convenience alias for the most common release-time decoder prompt."""
        return self._t1(PromptLength(length))

    def _t1(self, length: PromptLength) -> str:
        if length == PromptLength.SHORT:
            return "Free OCR."
        if length == PromptLength.MEDIUM:
            return "Free OCR.\nReturn only DNA sequence."
        return (
            "Free OCR.\n"
            "Return only the DNA sequence (A/C/G/T/N). "
            "Keep line breaks if present. No extra words."
        )

    def _t2(self, length: PromptLength) -> str:
        if length == PromptLength.SHORT:
            return "<|grounding|>Read all DNA text and locate each line."
        if length == PromptLength.MEDIUM:
            return (
                "<|grounding|>Read all DNA text and locate each line.\n"
                "Output per line: <|ref|>SEQ<|/ref|><|det|>[[img_id,x1,y1,x2,y2]]<|/det|>."
            )
        return (
            "<|grounding|>"
            "Read all DNA text and locate each text line or block.\n"
            "Output one line per region in reading order (top-to-bottom, left-to-right).\n"
            "For each region, output EXACTLY:\n"
            "<|ref|>SEQUENCE<|/ref|><|det|>[[img_id,x1,y1,x2,y2]]<|/det|>\n"
            "Rules:\n"
            "- det MUST be a list-of-boxes. Even one box must be written as [[...]].\n"
            "- img_id is 0-based index into the images list. If NUM_IMAGES==1, img_id MUST be 0.\n"
            "Only output these lines. No extra text."
        )

    def _t3(self, length: PromptLength, boxes: List[List[int]]) -> str:
        box_str = str(boxes)
        if length == PromptLength.SHORT:
            return f"<|grounding|>OCR DNA for boxes (in order): {box_str}"
        if length == PromptLength.MEDIUM:
            return (
                "<|grounding|>"
                "OCR DNA text for each box in the SAME order.\n"
                f"Boxes:\n{box_str}\n"
                "Output one line per box: <|ref|>SEQ<|/ref|><|det|>[[img_id,x1,y1,x2,y2]]<|/det|>"
            )
        return (
            "<|grounding|>"
            "OCR DNA text for each bounding box below, in the SAME order.\n"
            f"Boxes:\n{box_str}\n"
            "Output one line per box using EXACTLY:\n"
            "<|ref|>SEQUENCE<|/ref|><|det|>[[img_id,x1,y1,x2,y2]]<|/det|>\n"
            "Rules:\n"
            "- det MUST be list-of-boxes: [[...]] for a single box.\n"
            "- img_id is 0-based index into the images list. If NUM_IMAGES==1, img_id MUST be 0.\n"
            "No extra text."
        )

    def _t4(self, length: PromptLength, boxes: List[List[int]]) -> str:
        box_str = str(boxes)
        if length == PromptLength.SHORT:
            return f"<|grounding|>Predict masked DNA for boxes (in order): {box_str}"
        if length == PromptLength.MEDIUM:
            return (
                "<|grounding|>"
                "Predict ORIGINAL DNA for masked boxes in the SAME order.\n"
                f"Boxes:\n{box_str}\n"
                "Output: <|ref|>SEQ<|/ref|><|det|>[[img_id,x1,y1,x2,y2]]<|/det|>"
            )
        return (
            "<|grounding|>"
            "The DNA text inside each box is masked/occluded.\n"
            "Predict the ORIGINAL DNA sequence for each masked region.\n"
            f"Masked boxes:\n{box_str}\n"
            "Output in the SAME order as the boxes.\n"
            "Use only A/C/G/T/N (use N if uncertain).\n"
            "For each box, output EXACTLY one line:\n"
            "<|ref|>PREDICTED_SEQUENCE<|/ref|><|det|>[[img_id,x1,y1,x2,y2]]<|/det|>\n"
            "Rules:\n"
            "- det MUST be list-of-boxes: [[...]] for a single box.\n"
            "- img_id is 0-based index into the images list. If NUM_IMAGES==1, img_id MUST be 0.\n"
            "No extra text."
        )

    def _t5(self, length: PromptLength, query: str) -> str:
        if length == PromptLength.SHORT:
            return f"<|grounding|>Locate <|ref|>{query}<|/ref|>."
        if length == PromptLength.MEDIUM:
            return (
                "<|grounding|>"
                f"Locate <|ref|>{query}<|/ref|>.\n"
                "Output exactly one line: <|ref|>QUERY<|/ref|><|det|>[...]<|/det|> (or [] if not found)."
            )
        return (
            "<|grounding|>"
            f"Locate the DNA subsequence <|ref|>{query}<|/ref|>.\n"
            "Return ALL bounding boxes where it appears.\n"
            "Output EXACTLY one line:\n"
            f"<|ref|>{query}<|/ref|><|det|>[[img_id,x1,y1,x2,y2],[img_id,x1,y1,x2,y2]]<|/det|>\n"
            "If not found, output:\n"
            f"<|ref|>{query}<|/ref|><|det|>[]<|/det|>\n"
            "Rules:\n"
            "- Each box MUST be [img_id,x1,y1,x2,y2].\n"
            "- img_id is 0-based index into the images list. If NUM_IMAGES==1, img_id MUST be 0.\n"
            "No extra text."
        )

    def _t6(self, length: PromptLength) -> str:
        if length == PromptLength.SHORT:
            return "Which chromosome?"
        if length == PromptLength.MEDIUM:
            return "Predict chromosome label (chr1-22, chrX, chrY, or unknown)."
        return (
            "Classify this DNA sequence: which human chromosome does it belong to?\n"
            "Answer with one label only: chr1-chr22, chrX, chrY, or unknown."
        )
