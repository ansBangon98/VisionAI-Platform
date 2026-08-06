from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class BoundingBox:
    x: float
    y: float
    width: float
    height: float


@dataclass(slots=True)
class AttributeResult:
    label: str
    confidence: float


@dataclass(slots=True)
class Detection:
    class_id: int
    label: str
    confidence: float
    bbox: BoundingBox
    track_id: int | None = None
    attributes: dict[str, AttributeResult] = field(default_factory=dict)


@dataclass(slots=True)
class FrameResult:
    source_id: str
    frame_number: int
    timestamp: float
    detections: list[Detection]
    fps: float | None = None
    inference_ms: float | None = None
