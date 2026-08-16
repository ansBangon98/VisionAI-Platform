from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from core.models.base_detector import Detection


MAX_END_TO_END_DETECTIONS = 1000


@dataclass(frozen=True)
class PostprocessConfig:
    confidence_threshold: float = 0.4
    nms_iou_threshold: float = 0.45
    bbox_format: str = "auto"
    class_ids: frozenset[int] = field(default_factory=frozenset)


@dataclass(frozen=True)
class PostprocessContext:
    scale: float
    pad_x: int
    pad_y: int
    frame_width: int
    frame_height: int


class DetectionPostprocessor(ABC):
    output_format = "auto"

    def __init__(self, config: PostprocessConfig):
        self.config = config

    def select_output(self, outputs: Mapping[str, object]) -> object:
        if len(outputs) == 1:
            return next(iter(outputs.values()))

        scored_outputs: list[tuple[int, int, object]] = []
        for index, output in enumerate(outputs.values()):
            score = self.score_output(output)
            if score >= 0:
                scored_outputs.append((score, -index, output))

        if scored_outputs:
            return max(scored_outputs, key=lambda item: (item[0], item[1]))[2]
        return next(iter(outputs.values()))

    def score_output(self, output: object) -> int:
        try:
            return self.score_predictions(prediction_matrix(output))
        except RuntimeError:
            return -1

    @abstractmethod
    def score_predictions(self, predictions: np.ndarray) -> int:
        """Score whether a normalized output matrix matches this postprocessor."""

    @abstractmethod
    def process(
        self,
        output: object,
        context: PostprocessContext,
    ) -> list[Detection]:
        """Convert one raw backend output to image-space detections."""


def prediction_matrix(output: object) -> np.ndarray:
    predictions = np.asarray(output)
    if predictions.size == 0:
        raise RuntimeError("Detector output tensor is empty.")

    if predictions.ndim == 3:
        if predictions.shape[0] != 1:
            raise RuntimeError(
                "Detector output tensor has unsupported batch size "
                f"{predictions.shape[0]}; expected 1. Shape: {predictions.shape}."
            )
        predictions = predictions[0]
    if predictions.ndim != 2:
        raise RuntimeError(
            "Detector output tensor has unsupported rank "
            f"{predictions.ndim}; expected 2 or 3. Shape: {predictions.shape}."
        )
    return predictions


def image_space_detections(
    config: PostprocessConfig,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    context: PostprocessContext,
    *,
    apply_nms: bool,
) -> list[Detection]:
    keep = scores >= config.confidence_threshold
    if config.class_ids:
        keep = keep & np.isin(class_ids, list(config.class_ids))

    if not np.any(keep):
        return []

    boxes = boxes[keep].astype(np.float32, copy=True)
    scores = scores[keep]
    class_ids = class_ids[keep]

    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - context.pad_x) / context.scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - context.pad_y) / context.scale
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, context.frame_width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, context.frame_height)

    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    if not np.any(valid):
        return []

    boxes = boxes[valid]
    scores = scores[valid]
    class_ids = class_ids[valid]

    if apply_nms:
        selected = nms_by_class(boxes, scores, class_ids, config.nms_iou_threshold)
    else:
        selected = list(np.argsort(scores)[::-1])

    return [
        Detection(
            bbox=tuple(float(value) for value in boxes[index]),
            score=float(scores[index]),
            class_id=int(class_ids[index]),
        )
        for index in selected
    ]


def normalize_output_format(value: object) -> str:
    text = str(value or "auto").strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return normalized or "auto"


def normalize_bbox_format(value: object) -> str:
    normalized = normalize_output_format(value)
    aliases = {
        "": "auto",
        "auto": "auto",
        "center_xywh": "xywh_center",
        "cxcywh": "xywh_center",
        "x_center_y_center_width_height": "xywh_center",
        "xywh": "xywh_center",
        "xywh_center": "xywh_center",
        "xywh_centre": "xywh_center",
        "tlwh": "xywh_top_left",
        "xywh_top_left": "xywh_top_left",
        "xywh_topleft": "xywh_top_left",
        "x1y1x2y2": "xyxy",
        "xyxy": "xyxy",
        "xyxy_corner": "xyxy",
    }
    return aliases.get(normalized, normalized)


def parse_optional_bool(value: object) -> bool | None:
    if value in (None, ""):
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(value)

    text = str(value).strip().lower().replace("-", "_")
    if text in {"1", "true", "yes", "y", "on", "end2end", "end_to_end"}:
        return True
    if text in {"0", "false", "no", "n", "off", "one_to_many", "one2many"}:
        return False
    return None


def boxes_to_xyxy(boxes: np.ndarray, bbox_format: str) -> np.ndarray:
    normalized_format = normalize_bbox_format(bbox_format)
    if normalized_format == "xyxy":
        return boxes.astype(np.float32, copy=True)
    if normalized_format == "xywh_top_left":
        return xywh_top_left_to_xyxy(boxes)
    return xywh_to_xyxy(boxes)


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    converted = boxes.astype(np.float32, copy=True)
    converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return converted


def xywh_top_left_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    converted = boxes.astype(np.float32, copy=True)
    converted[:, 2] = boxes[:, 0] + boxes[:, 2]
    converted[:, 3] = boxes[:, 1] + boxes[:, 3]
    return converted


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    if boxes.size == 0:
        return []

    order = scores.argsort()[::-1]
    keep: list[int] = []

    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break

        ious = box_iou(boxes[current], boxes[order[1:]])
        order = order[1:][ious <= threshold]

    return keep


def nms_by_class(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    threshold: float,
) -> list[int]:
    selected: list[int] = []
    for class_id in np.unique(class_ids):
        class_indexes = np.where(class_ids == class_id)[0]
        kept_indexes = nms(boxes[class_indexes], scores[class_indexes], threshold)
        selected.extend(int(class_indexes[index]) for index in kept_indexes)

    return sorted(selected, key=lambda index: float(scores[index]), reverse=True)


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    box_area = max(0.0, float((box[2] - box[0]) * (box[3] - box[1])))
    boxes_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0,
        boxes[:, 3] - boxes[:, 1],
    )
    union = box_area + boxes_area - intersection
    return intersection / np.maximum(union, 1e-6)
