from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from core.postprocess.segmentation import colorize_mask


class SegmentationRenderer:
    def __init__(
        self,
        *,
        mode: str = "mask",
        overlay_enabled: bool = False,
        overlay_alpha: float = 0.45,
        palette: Mapping[int, tuple[int, int, int]] | None = None,
    ):
        self.mode = str(mode or "mask").strip().lower()
        self.overlay_enabled = bool(overlay_enabled)
        self.overlay_alpha = max(0.0, min(1.0, float(overlay_alpha)))
        self.palette = dict(palette or {})

    def render(self, original_frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        color_mask = colorize_mask(mask, self.palette)
        if self.overlay_enabled or self.mode == "overlay":
            return blend_overlay(original_frame, color_mask, self.overlay_alpha)
        return color_mask


def blend_overlay(
    original_frame: np.ndarray,
    color_mask: np.ndarray,
    alpha: float,
) -> np.ndarray:
    source = np.asarray(original_frame, dtype=np.float32)
    mask = np.asarray(color_mask, dtype=np.float32)
    if source.shape[:2] != mask.shape[:2]:
        raise RuntimeError("Overlay mask size must match the original frame size.")

    blended = source * (1.0 - alpha) + mask * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)
