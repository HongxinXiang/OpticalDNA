from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image as PILImage


class TaskType(str, Enum):
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


ImageItem = Union[str, PILImage.Image]
ImageField = Union[ImageItem, List[ImageItem]]

ImageGroups = Union[ImageItem, List[ImageItem], List[List[ImageItem]]]


def ensure_image_groups(image: ImageGroups) -> List[List[ImageItem]]:
    def _is_item(x) -> bool:
        return isinstance(x, (str, PILImage.Image))

    if _is_item(image):
        return [[image]]

    if isinstance(image, list) and len(image) > 0 and all(_is_item(x) for x in image):
        return [image]

    if isinstance(image, list) and len(image) > 0 and all(isinstance(g, list) for g in image):
        if any(len(g) == 0 for g in image):
            raise ValueError("image group contains an empty list.")
        for gi, g in enumerate(image):
            for ii, x in enumerate(g):
                if not _is_item(x):
                    raise TypeError(
                        f"image groups must be List[List[ImageItem]]. "
                        f"Found invalid type at group[{gi}][{ii}]: {type(x)}"
                    )
        return image

    raise TypeError(
        "sample['image'] must be one of: "
        "ImageItem(str|PIL.Image.Image), non-empty List[ImageItem], non-empty List[List[ImageItem]]."
    )


def ensure_images_list(image: ImageField) -> List[ImageItem]:
    if isinstance(image, (str, PILImage.Image)):
        return [image]

    if isinstance(image, list):
        if len(image) == 0:
            raise ValueError("image list is empty.")

        for i, x in enumerate(image):
            if not isinstance(x, (str, PILImage.Image)):
                raise TypeError(
                    f"image[{i}] must be str or PIL.Image.Image, got {type(x)}"
                )
        return image

    raise TypeError(
        "sample['image'] must be str, PIL.Image.Image, "
        "or a non-empty List[str | PIL.Image.Image]."
    )


def is_multi_image(image: ImageField) -> bool:
    images = ensure_images_list(image)
    return len(images) > 1


@dataclass(frozen=True)
class PromptGenerator:

    def build(self, task: TaskType, length: PromptLength, sample: Dict[str, Any]) -> str:
        if task == TaskType.T1_FULL_OCR:
            return self._t1(length)
        if task == TaskType.T2_FULL_OCR_GROUNDING:
            return self._t2(length)
        if task == TaskType.T3_ROI_OCR:
            boxes = sample.get("boxes")
            if boxes is None:
                raise KeyError("T3 requires sample['boxes']")
            return self._t3(length, boxes)
        if task == TaskType.T4_MASK_COMPLETION:
            boxes = sample.get("boxes")
            if boxes is None:
                raise KeyError("T4 requires sample['boxes']")
            return self._t4(length, boxes)
        if task == TaskType.T5_SUBSEQ_LOCATE:
            query = sample.get("query")
            if query is None:
                raise KeyError("T5 requires sample['query']")
            return self._t5(length, query)
        if task == TaskType.T6_CHR_CLASSIFICATION:
            return self._t6(length)
        raise ValueError(f"Unsupported task: {task}")

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


@dataclass
class PromptCurriculumSampler:
    p_long: float = 0.7
    p_medium: float = 0.2
    p_short: float = 0.1
    seed: int = 0
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        s = self.p_long + self.p_medium + self.p_short
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"Global probs must sum to 1.0, got {s}")
        self._rng = random.Random(self.seed)

    def sample_length(self) -> PromptLength:
        r = self._rng.random()
        if r < self.p_long:
            return PromptLength.LONG
        if r < self.p_long + self.p_medium:
            return PromptLength.MEDIUM
        return PromptLength.SHORT


@dataclass
class AnnealedPromptCurriculum:
    total_steps: int
    seed: int = 0
    start: Tuple[float, float, float] = (0.80, 0.15, 0.05)
    end: Tuple[float, float, float] = (0.30, 0.30, 0.40)
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        for name, p in [("start", self.start), ("end", self.end)]:
            s = sum(p)
            if abs(s - 1.0) > 1e-6:
                raise ValueError(f"{name} probs must sum to 1.0, got {s}")
        self._rng = random.Random(self.seed)

    def _interp_probs(self, step: int) -> Tuple[float, float, float]:
        if self.total_steps <= 0:
            raise ValueError(f"total_steps must be >0, got {self.total_steps}")
        t = min(max(step, 0), self.total_steps) / float(self.total_steps)
        pL = self.start[0] + t * (self.end[0] - self.start[0])
        pM = self.start[1] + t * (self.end[1] - self.start[1])
        pS = self.start[2] + t * (self.end[2] - self.start[2])
        s = pL + pM + pS
        return (pL / s, pM / s, pS / s)

    def sample_length(self, step: int) -> PromptLength:
        pL, pM, _pS = self._interp_probs(step)
        r = self._rng.random()
        if r < pL:
            return PromptLength.LONG
        if r < pL + pM:
            return PromptLength.MEDIUM
        return PromptLength.SHORT


def _validate_box_5_ints(box: List[int]) -> None:
    if len(box) != 5:
        raise ValueError(f"expects 5-int box [img_id,x1,y1,x2,y2], got: {box}")
    if not all(isinstance(v, int) for v in box):
        raise TypeError(f"box values must be int, got types: {[type(v) for v in box]}")


def format_grounding_lines(
        seqs: List[str],
        boxes5: List[List[int]],
        *,
        num_images: int,
) -> str:
    if len(seqs) != len(boxes5):
        raise ValueError(f"len(seqs)={len(seqs)} != len(boxes)={len(boxes5)}")
    if num_images <= 0:
        raise ValueError(f"num_images must be >0, got {num_images}")

    lines: List[str] = []
    for seq, box in zip(seqs, boxes5):
        _validate_box_5_ints(box)
        img_id, x1, y1, x2, y2 = box

        if not (0 <= img_id < num_images):
            raise ValueError(f"img_id={img_id} out of range for num_images={num_images}")

        det = f"[[{img_id},{x1},{y1},{x2},{y2}]]"
        lines.append(f"<|ref|>{seq}<|/ref|><|det|>{det}<|/det|>")
    return "\n".join(lines)


def format_locate_answer(query: str, boxes5: List[List[int]], *, num_images: int) -> str:
    if num_images <= 0:
        raise ValueError(f"num_images must be >0, got {num_images}")

    if len(boxes5) == 0:
        det = "[]"
    else:
        for b in boxes5:
            _validate_box_5_ints(b)
            if not (0 <= b[0] < num_images):
                raise ValueError(f"img_id={b[0]} out of range for num_images={num_images}")
        det = "[" + ",".join([f"[{b[0]},{b[1]},{b[2]},{b[3]},{b[4]}]" for b in boxes5]) + "]"

    return f"<|ref|>{query}<|/ref|><|det|>{det}<|/det|>"


@dataclass
class DNASampleFactory:
    image_key: str = "image"
    text_key: str = "text"
    boxes_key: str = "boxes"
    query_key: str = "query"

    def t1_full_ocr(self, image: ImageField, seq_lines: List[str]) -> Dict[str, Any]:
        text = "\n".join(seq_lines)
        return {self.image_key: image, self.text_key: text}

    def t2_full_ocr_grounding(
            self,
            image: ImageField,
            seqs: List[str],
            boxes: List[List[int]],
    ) -> Dict[str, Any]:
        images = ensure_images_list(image)
        num_images = len(images)

        boxes5 = self._normalize_boxes_to_5(boxes, num_images=num_images)
        text = format_grounding_lines(seqs, boxes5, num_images=num_images)
        return {self.image_key: image, self.text_key: text}

    def t3_roi_ocr(
            self,
            image: ImageField,
            boxes: List[List[int]],
            seqs: List[str],
    ) -> Dict[str, Any]:
        images = ensure_images_list(image)
        num_images = len(images)

        boxes5 = self._normalize_boxes_to_5(boxes, num_images=num_images)
        text = format_grounding_lines(seqs, boxes5, num_images=num_images)

        return {
            self.image_key: image,
            self.boxes_key: boxes5,
            self.text_key: text,
        }

    def t4_mask_completion(
            self,
            image: ImageField,
            boxes: List[List[int]],
            pred_seqs: List[str],
    ) -> Dict[str, Any]:
        images = ensure_images_list(image)
        num_images = len(images)

        boxes5 = self._normalize_boxes_to_5(boxes, num_images=num_images)
        text = format_grounding_lines(pred_seqs, boxes5, num_images=num_images)

        return {
            self.image_key: image,
            self.boxes_key: boxes5,
            self.text_key: text,
        }

    def t5_subseq_locate(
            self,
            image: ImageField,
            query: str,
            boxes: List[List[int]],
    ) -> Dict[str, Any]:
        images = ensure_images_list(image)
        num_images = len(images)

        boxes5 = self._normalize_boxes_to_5(boxes, num_images=num_images) if boxes else []
        text = format_locate_answer(query, boxes5, num_images=num_images)

        return {
            self.image_key: image,
            self.query_key: query,
            self.text_key: text,
        }

    def t6_chr_classification(self, image: ImageField, chr_label: str) -> Dict[str, Any]:
        return {self.image_key: image, self.text_key: chr_label}

    def _normalize_boxes_to_5(self, boxes: List[List[int]], *, num_images: int) -> List[List[int]]:
        if num_images <= 0:
            raise ValueError(f"num_images must be >0, got {num_images}")

        out: List[List[int]] = []
        for b in boxes:
            if len(b) == 4:
                x1, y1, x2, y2 = b
                candidate = [0, x1, y1, x2, y2]
            elif len(b) == 5:
                candidate = list(b)
            else:
                raise ValueError(f"box must be 4 or 5 ints, got: {b}")

            _validate_box_5_ints(candidate)

            img_id = candidate[0]
            if not (0 <= img_id < num_images):
                raise ValueError(f"img_id={img_id} out of range for num_images={num_images}")

            out.append(candidate)

        return out


@dataclass
class ConversationBuilder:
    prompt_generator: PromptGenerator
    fixed_sampler: Optional[PromptCurriculumSampler] = None
    annealed_sampler: Optional[AnnealedPromptCurriculum] = None
    validate_fields: bool = True

    image_key: str = "image"
    text_key: str = "text"

    def __call__(
            self,
            sample: Dict[str, Any],
            task: TaskType,
            *,
            step: Optional[int] = None,
            prompt_length: Optional[PromptLength] = None,
            debug_dump: bool = False,
    ) -> Dict[str, Any]:
        if self.validate_fields:
            self._validate_sample_fields(sample, task)

        images_raw = sample[self.image_key]
        images = ensure_image_groups(images_raw)

        if len(images) == 1:
            num_images = len(images[0])
        else:
            num_images = sum(len(g) for g in images)

        length = prompt_length or self._sample_prompt_length(step=step)

        instruction = self.prompt_generator.build(task=task, length=length, sample=sample)

        user_content = "<image>\n" + f"NUM_IMAGES={num_images}.\n" + instruction

        conversation = [
            {"role": "<|User|>", "content": user_content, "images": images},
            {"role": "<|Assistant|>", "content": sample[self.text_key]},
        ]
        out = {"messages": conversation}

        if debug_dump:
            self._debug_dump(sample=sample, task=task, length=length, instruction=user_content, out=out)

        return out

    def _sample_prompt_length(self, *, step: Optional[int]) -> PromptLength:
        if self.annealed_sampler is not None:
            if step is None:
                raise ValueError("annealed_sampler is set, but step=None. Pass step=global_step.")
            return self.annealed_sampler.sample_length(step)
        if self.fixed_sampler is not None:
            return self.fixed_sampler.sample_length()
        return PromptLength.LONG

    def _validate_sample_fields(self, sample: Dict[str, Any], task: TaskType) -> None:
        if self.image_key not in sample:
            raise KeyError(f"Sample missing '{self.image_key}'")
        if self.text_key not in sample:
            raise KeyError(f"Sample missing '{self.text_key}'")

        _ = ensure_image_groups(sample[self.image_key])

        if task in (TaskType.T3_ROI_OCR, TaskType.T4_MASK_COMPLETION):
            if "boxes" not in sample:
                raise KeyError(f"{task.value} requires sample['boxes']")
            if not isinstance(sample["boxes"], list):
                raise TypeError("sample['boxes'] must be a list of boxes")

        if task == TaskType.T5_SUBSEQ_LOCATE:
            if "query" not in sample:
                raise KeyError("t5_subseq_locate requires sample['query']")
            if not isinstance(sample["query"], str):
                raise TypeError("sample['query'] must be a string")

    def _debug_dump(
            self,
            sample: Dict[str, Any],
            task: TaskType,
            length: PromptLength,
            instruction: str,
            out: Dict[str, Any],
    ) -> None:
        images = ensure_image_groups(sample[self.image_key])
        total_images = sum(len(g) for g in images)
        print("=" * 100)
        print(f"[ConversationBuilder] task={task.value} | prompt_length={length.value} | num_images={total_images}")
        print("-" * 100)
        if "boxes" in sample:
            print("[Sample.boxes] (if present)")
            print(sample["boxes"])
            print("-" * 100)
        if "query" in sample:
            print("[Sample.query] (if present)")
            print(sample["query"])
            print("-" * 100)
        print("[User.content]")
        print(instruction)
        print("-" * 100)
        print("[Assistant.content]")
        print(sample[self.text_key])
        print("-" * 100)
        print("[messages preview]")
        print(out)
        print("=" * 100)
        print()






