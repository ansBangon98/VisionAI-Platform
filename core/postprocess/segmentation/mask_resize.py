from __future__ import annotations

import numpy as np


def resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize a class-id mask without inventing fractional labels."""
    target_width = max(1, int(width))
    target_height = max(1, int(height))
    source = np.asarray(mask)
    if source.ndim != 2:
        raise RuntimeError("Segmentation mask must be a 2D array.")
    if source.shape == (target_height, target_width):
        return np.ascontiguousarray(source)

    try:
        import cv2
    except ImportError:
        y_indices = np.linspace(0, source.shape[0] - 1, target_height).astype(np.intp)
        x_indices = np.linspace(0, source.shape[1] - 1, target_width).astype(np.intp)
        return np.ascontiguousarray(source[y_indices[:, None], x_indices])

    resized = cv2.resize(
        source,
        (target_width, target_height),
        interpolation=cv2.INTER_NEAREST,
    )
    return np.ascontiguousarray(resized)
