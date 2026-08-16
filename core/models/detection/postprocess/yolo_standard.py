from __future__ import annotations

import numpy as np

from core.models.base_detector import Detection

from .base_postprocessor import (
    MAX_END_TO_END_DETECTIONS,
    DetectionPostprocessor,
    PostprocessContext,
    boxes_to_xyxy,
    image_space_detections,
    prediction_matrix,
)


class YOLOStandardPostprocessor(DetectionPostprocessor):
    output_format = "yolo_standard"

    def score_predictions(self, predictions: np.ndarray) -> int:
        return traditional_layout_score(predictions)

    def process(
        self,
        output: object,
        context: PostprocessContext,
    ) -> list[Detection]:
        predictions = prediction_matrix(output)
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T
        if predictions.shape[1] < 5:
            raise RuntimeError(
                "Detector output tensor has too few columns. Expected at least "
                f"5 values per prediction (x, y, w, h, score/classes); "
                f"got shape {predictions.shape}."
            )

        predictions = predictions.astype(np.float32, copy=False)
        boxes = boxes_to_xyxy(predictions[:, :4], self.config.bbox_format)
        if predictions.shape[1] == 5:
            scores = predictions[:, 4]
            class_ids = np.zeros_like(scores, dtype=np.int64)
        else:
            class_scores = predictions[:, 4:]
            scores = class_scores.max(axis=1)
            class_ids = class_scores.argmax(axis=1)

        return image_space_detections(
            self.config,
            boxes,
            scores,
            class_ids,
            context,
            apply_nms=True,
        )


def traditional_layout_score(predictions: np.ndarray) -> int:
    if predictions.ndim != 2:
        return -1

    rows, columns = predictions.shape
    if rows <= 0 or columns <= 0:
        return -1
    if rows < columns:
        return 100 if rows >= 5 else -1
    if columns >= 5 and rows > MAX_END_TO_END_DETECTIONS:
        return 90
    if columns >= 5:
        return 40
    return -1

