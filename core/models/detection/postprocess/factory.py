from __future__ import annotations

import numpy as np

from core.models.base_detector import Detection

from .base_postprocessor import (
    DetectionPostprocessor,
    PostprocessConfig,
    PostprocessContext,
    normalize_bbox_format,
    normalize_output_format,
    parse_optional_bool,
    prediction_matrix,
)
from .yolo_end2end import YOLOEnd2EndPostprocessor
from .yolo_standard import YOLOStandardPostprocessor


AUTO_OUTPUT_FORMATS = frozenset({"", "auto"})
END_TO_END_OUTPUT_FORMATS = frozenset(
    {
        "e2e",
        "end2end",
        "end_to_end",
        "one2one",
        "one_to_one",
        "ultralytics_yolo_e2e",
        "ultralytics_yolo_end2end",
        "ultralytics_yolo_end_to_end",
        "yolo26",
        "yolo26_end2end",
        "yolo_end2end",
        "yolo_end_to_end",
    }
)
STANDARD_OUTPUT_FORMATS = frozenset(
    {
        "one2many",
        "one_to_many",
        "raw",
        "raw_yolo",
        "standard",
        "traditional",
        "traditional_yolo",
        "ultralytics_yolo",
        "ultralytics_yolo_raw",
        "ultralytics_yolo_standard",
        "ultralytics_yolo_traditional",
        "yolo_raw",
        "yolo_standard",
        "yolo_traditional",
    }
)


def create_yolo_postprocessor(
    *,
    output_format: object = "auto",
    bbox_format: object = "auto",
    end2end: object = None,
    confidence_threshold: float = 0.4,
    nms_iou_threshold: float = 0.45,
    class_ids: set[int] | frozenset[int] | None = None,
) -> DetectionPostprocessor:
    normalized_format = normalize_output_format(output_format)
    config = PostprocessConfig(
        confidence_threshold=float(confidence_threshold),
        nms_iou_threshold=float(nms_iou_threshold),
        bbox_format=normalize_bbox_format(bbox_format),
        class_ids=frozenset(class_ids or ()),
    )
    end2end_hint = parse_optional_bool(end2end)

    if normalized_format in END_TO_END_OUTPUT_FORMATS:
        return YOLOEnd2EndPostprocessor(config)
    if normalized_format in STANDARD_OUTPUT_FORMATS:
        return YOLOStandardPostprocessor(config)
    if normalized_format in AUTO_OUTPUT_FORMATS:
        return AutoYOLOPostprocessor(config, end2end_hint=end2end_hint)

    raise RuntimeError(f"Unsupported YOLO output format: {output_format}")


class AutoYOLOPostprocessor(DetectionPostprocessor):
    output_format = "auto"

    def __init__(
        self,
        config: PostprocessConfig,
        *,
        end2end_hint: bool | None = None,
    ):
        super().__init__(config)
        self.end2end_hint = end2end_hint
        self.standard = YOLOStandardPostprocessor(config)
        self.end2end = YOLOEnd2EndPostprocessor(config)

    def score_predictions(self, predictions: np.ndarray) -> int:
        if self.end2end_hint is True:
            return self.end2end.score_predictions(predictions) + 80
        if self.end2end_hint is False:
            return self.standard.score_predictions(predictions) + 20

        return max(
            self.end2end.score_predictions(predictions),
            self.standard.score_predictions(predictions),
        )

    def process(
        self,
        output: object,
        context: PostprocessContext,
    ) -> list[Detection]:
        predictions = prediction_matrix(output)
        postprocessor = self._postprocessor_for_predictions(predictions)
        return postprocessor.process(predictions, context)

    def _postprocessor_for_predictions(
        self,
        predictions: np.ndarray,
    ) -> DetectionPostprocessor:
        if self.end2end_hint is True:
            return self.end2end
        if self.end2end_hint is False:
            return self.standard

        end2end_score = self.end2end.score_predictions(predictions)
        standard_score = self.standard.score_predictions(predictions)
        if end2end_score >= standard_score and end2end_score >= 0:
            return self.end2end
        if standard_score >= 0:
            return self.standard
        raise RuntimeError(
            f"Unable to determine YOLO output format from shape: {predictions.shape}"
        )
