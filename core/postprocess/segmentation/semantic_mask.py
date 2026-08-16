from __future__ import annotations

from collections.abc import Mapping

import numpy as np


DEFAULT_SEGMENTATION_PALETTE: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),
    1: (245, 158, 11),
    2: (34, 197, 94),
}


def semantic_mask_from_logits(
    logits: object,
    *,
    class_count: int | None = None,
) -> np.ndarray:
    scores = np.asarray(logits)
    if scores.ndim == 4:
        scores = scores[0]
    if scores.ndim == 2:
        return _compact_mask(scores)
    if scores.ndim != 3:
        raise RuntimeError(
            f"Unsupported segmentation output shape: {tuple(scores.shape)}."
        )

    class_axis = _class_axis(scores, class_count)
    mask = np.argmax(scores, axis=class_axis)
    return _compact_mask(mask)


def colorize_mask(
    mask: np.ndarray,
    palette: Mapping[int, tuple[int, int, int]] | None = None,
) -> np.ndarray:
    source = np.asarray(mask)
    if source.ndim != 2:
        raise RuntimeError("Segmentation mask must be a 2D array.")

    colors = dict(DEFAULT_SEGMENTATION_PALETTE)
    colors.update(dict(palette or {}))
    output = np.zeros((*source.shape, 3), dtype=np.uint8)

    for class_id in np.unique(source):
        output[source == class_id] = colors.get(int(class_id), _fallback_color(class_id))
    return output


def class_coverage(mask: np.ndarray) -> dict[int, float]:
    source = np.asarray(mask)
    if source.size == 0:
        return {}

    class_ids, counts = np.unique(source.astype(np.int64, copy=False), return_counts=True)
    total = float(source.size)
    return {
        int(class_id): float(count) / total
        for class_id, count in zip(class_ids, counts)
    }


def _class_axis(scores: np.ndarray, class_count: int | None) -> int:
    if class_count is not None:
        if scores.shape[0] == class_count:
            return 0
        if scores.shape[-1] == class_count:
            return -1

    if scores.shape[0] <= 512 and scores.shape[0] <= scores.shape[-1]:
        return 0
    return -1


def _compact_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.size == 0:
        return np.ascontiguousarray(mask.astype(np.uint8, copy=False))

    max_value = int(np.max(mask))
    dtype = np.uint8 if max_value <= np.iinfo(np.uint8).max else np.uint16
    return np.ascontiguousarray(mask.astype(dtype, copy=False))


def _fallback_color(class_id: object) -> tuple[int, int, int]:
    value = int(class_id)
    return (
        (37 * value + 53) % 256,
        (97 * value + 101) % 256,
        (173 * value + 29) % 256,
    )
