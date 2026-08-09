from __future__ import annotations

from core.models.base_detector import BaseDetector, Detection
from core.models.yolo_detector import (
    YoloDetector,
    YoloPeopleDetector,
    available_detector_backends,
    create_inference_backend,
    frame_size,
    load_class_labels,
)

__all__ = [
    "BaseDetector",
    "Detection",
    "YoloDetector",
    "YoloPeopleDetector",
    "available_detector_backends",
    "create_inference_backend",
    "frame_size",
    "load_class_labels",
]
