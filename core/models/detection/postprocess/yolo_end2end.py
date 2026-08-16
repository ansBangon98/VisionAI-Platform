from __future__ import annotations

import numpy as np

from core.models.base_detector import Detection

from .base_postprocessor import (
    MAX_END_TO_END_DETECTIONS,
    DetectionPostprocessor,
    PostprocessContext,
    image_space_detections,
    prediction_matrix,
)


class YOLOEnd2EndPostprocessor(DetectionPostprocessor):
    output_format = "yolo_end2end"

    def score_predictions(self, predictions: np.ndarray) -> int:
        if not looks_like_end_to_end_shape(predictions):
            return -1
        if not looks_like_end_to_end_values(
            end_to_end_rows(predictions),
            self.config.confidence_threshold,
        ):
            return 90
        return 120

    def process(
        self,
        output: object,
        context: PostprocessContext,
    ) -> list[Detection]:
        raw_predictions = prediction_matrix(output)
        if not looks_like_end_to_end_shape(raw_predictions):
            raise RuntimeError(
                "End-to-end YOLO output must have shape [N, 6] or [6, N] "
                f"with N <= {MAX_END_TO_END_DETECTIONS}; "
                f"got shape {raw_predictions.shape}."
            )

        predictions = end_to_end_rows(raw_predictions).astype(
            np.float32,
            copy=False,
        )
        if predictions.shape[1] != 6:
            raise RuntimeError(
                "End-to-end YOLO output must contain 6 values per detection "
                "(x1, y1, x2, y2, confidence, class_id); "
                f"got shape {predictions.shape}."
            )

        boxes = predictions[:, :4]
        scores = predictions[:, 4]
        class_ids = np.rint(predictions[:, 5]).astype(np.int64)
        return image_space_detections(
            self.config,
            boxes,
            scores,
            class_ids,
            context,
            apply_nms=False,
        )


def looks_like_end_to_end_shape(predictions: np.ndarray) -> bool:
    if predictions.ndim != 2:
        return False

    rows, columns = predictions.shape
    if rows <= 0 or columns <= 0:
        return False
    if columns == 6 and rows <= MAX_END_TO_END_DETECTIONS:
        return True
    return rows == 6 and columns <= MAX_END_TO_END_DETECTIONS


def end_to_end_rows(predictions: np.ndarray) -> np.ndarray:
    if predictions.shape[1] == 6:
        return predictions
    if predictions.shape[0] == 6:
        return predictions.T
    return predictions


def looks_like_end_to_end_values(
    predictions: np.ndarray,
    confidence_threshold: float,
) -> bool:
    if predictions.ndim != 2 or predictions.shape[1] != 6:
        return False

    finite_rows = np.all(np.isfinite(predictions), axis=1)
    if not np.any(finite_rows):
        return False

    values = predictions[finite_rows]
    scores = values[:, 4]
    score_like = (scores >= 0.0) & (scores <= 1.01)
    if float(np.mean(score_like)) < 0.95:
        return False

    candidate_scores = scores >= max(1e-6, float(confidence_threshold))
    if np.any(candidate_scores):
        candidate_boxes = values[candidate_scores, :4]
        ordered_boxes = (
            (candidate_boxes[:, 2] >= candidate_boxes[:, 0])
            & (candidate_boxes[:, 3] >= candidate_boxes[:, 1])
        )
        if float(np.mean(ordered_boxes)) < 0.80:
            return False

        class_values = values[candidate_scores, 5]
    else:
        class_values = values[:, 5]

    integer_like = np.abs(class_values - np.rint(class_values)) <= 1e-3
    return bool(float(np.mean(integer_like)) >= 0.95)
