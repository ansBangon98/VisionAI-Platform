from __future__ import annotations

import time
from collections.abc import Mapping, Sequence

from core.results.frame_result import BoundingBox, Detection, FrameResult


def frame_result_to_legacy_objects(
    result: FrameResult,
) -> list[dict[str, object]]:
    return [_detection_to_legacy_object(detection) for detection in result.detections]


def legacy_objects_to_frame_result(
    objects: Sequence[object],
    source_id: str,
    frame_number: int,
    timestamp: float | None = None,
    fps: float | None = None,
    inference_ms: float | None = None,
) -> FrameResult:
    detections = [
        detection
        for detection in (_legacy_object_to_detection(item) for item in objects)
        if detection is not None
    ]
    return FrameResult(
        source_id=source_id,
        frame_number=frame_number,
        timestamp=time.time() if timestamp is None else float(timestamp),
        detections=detections,
        fps=fps,
        inference_ms=inference_ms,
    )


def _detection_to_legacy_object(detection: Detection) -> dict[str, object]:
    bbox = detection.bbox
    item: dict[str, object] = {
        "type": _legacy_type(detection.label, detection.class_id),
        "label": detection.label,
        "class_id": detection.class_id,
        "track_id": detection.track_id,
        "bbox": (
            int(round(bbox.x)),
            int(round(bbox.y)),
            int(round(bbox.x + bbox.width)),
            int(round(bbox.y + bbox.height)),
        ),
        "score": detection.confidence,
    }
    if detection.attributes:
        item["attributes"] = {
            name: {
                "label": attribute.label,
                "confidence": attribute.confidence,
            }
            for name, attribute in detection.attributes.items()
        }
    return item


def _legacy_object_to_detection(item: object) -> Detection | None:
    value = _object_mapping(item)
    if value is None:
        return None

    bbox_value = value.get("bbox")
    if bbox_value is None:
        return None

    try:
        x1, y1, x2, y2 = [float(part) for part in bbox_value[:4]]
    except (TypeError, ValueError):
        return None

    class_id = int(value.get("class_id", 0) or 0)
    label = str(value.get("label") or value.get("type") or f"class_{class_id}")
    score = float(value.get("score", value.get("confidence", 1.0)) or 0.0)
    track_id = value.get("track_id")
    return Detection(
        class_id=class_id,
        label=label,
        confidence=score,
        bbox=BoundingBox(
            x=x1,
            y=y1,
            width=max(0.0, x2 - x1),
            height=max(0.0, y2 - y1),
        ),
        track_id=None if track_id is None else int(track_id),
    )


def _object_mapping(item: object) -> Mapping[str, object] | None:
    if isinstance(item, Mapping):
        return item
    if hasattr(item, "__dict__"):
        return vars(item)
    return None


def _legacy_type(label: str, class_id: int) -> str:
    display_label = str(label or "").strip()
    normalized = display_label.lower()
    if normalized in {"person", "people"}:
        return "person"
    if normalized == "face":
        return "face"
    return display_label or f"class_{class_id}"
