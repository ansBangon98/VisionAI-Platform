from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np


@dataclass(slots=True)
class SegmentationResult:
    source_id: str
    frame_number: int
    original_frame: np.ndarray
    mask: np.ndarray
    visualization_frame: np.ndarray | None = None
    timestamp: float = field(default_factory=time.time)
    class_labels: dict[int, str] = field(default_factory=dict)
    inference_ms: float | None = None
    fps: float | None = None
