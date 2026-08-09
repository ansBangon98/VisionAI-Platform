from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def crop_from_bbox(frame: object, bbox: Sequence[float]) -> object:
    """Return an image crop for NumPy frames or Qt images."""

    x1, y1, x2, y2 = _clip_bbox(bbox, frame)
    if hasattr(frame, "copy") and hasattr(frame, "width") and hasattr(frame, "height"):
        return frame.copy(x1, y1, max(0, x2 - x1), max(0, y2 - y1))

    array = np.asarray(frame)
    if array.ndim < 2:
        raise RuntimeError("Frame must be image-like to crop from a bounding box.")
    return array[y1:y2, x1:x2].copy()


def _clip_bbox(bbox: Sequence[float], frame: object) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    if hasattr(frame, "width") and hasattr(frame, "height"):
        width = int(frame.width())
        height = int(frame.height())
    else:
        array = np.asarray(frame)
        height = int(array.shape[0])
        width = int(array.shape[1])

    return (
        int(round(np.clip(x1, 0, width))),
        int(round(np.clip(y1, 0, height))),
        int(round(np.clip(x2, 0, width))),
        int(round(np.clip(y2, 0, height))),
    )

