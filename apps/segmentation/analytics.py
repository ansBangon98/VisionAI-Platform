from __future__ import annotations

import numpy as np

from core.postprocess.segmentation import class_coverage
from core.results import SegmentationResult


class Analytics:
    def update(self, result: object) -> dict[str, object]:
        if not isinstance(result, SegmentationResult):
            return {}

        coverage = class_coverage(result.mask)
        labels = result.class_labels
        active_class_ids = [
            class_id
            for class_id, fraction in coverage.items()
            if class_id != 0 and fraction > 0.0
        ]
        dominant_class_id = _dominant_class_id(result.mask)
        road_class_id = _class_id_for_label(labels, "road", fallback=2)
        fallback_class_id = _class_id_for_label(labels, "drivable_fallback", fallback=1)

        road_coverage = coverage.get(road_class_id, 0.0)
        drivable_coverage = road_coverage + coverage.get(fallback_class_id, 0.0)

        return {
            "current_objects": len(active_class_ids),
            "active_class_count": len(active_class_ids),
            "dominant_class": labels.get(dominant_class_id, f"class_{dominant_class_id}"),
            "road_coverage_pct": road_coverage * 100.0,
            "drivable_coverage_pct": drivable_coverage * 100.0,
            "frame_number": result.frame_number,
            "inference_ms": result.inference_ms,
            "fps": result.fps,
        }


def _dominant_class_id(mask: np.ndarray) -> int:
    values, counts = np.unique(mask.astype(np.int64, copy=False), return_counts=True)
    if values.size == 0:
        return 0
    return int(values[int(np.argmax(counts))])


def _class_id_for_label(
    labels: dict[int, str],
    target: str,
    *,
    fallback: int,
) -> int:
    normalized_target = target.strip().lower()
    for class_id, label in labels.items():
        if str(label).strip().lower() == normalized_target:
            return int(class_id)
    return fallback
