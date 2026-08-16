from __future__ import annotations

from core.results import SegmentationResult


class SegmentationController:
    """Small adapter for UI code that consumes semantic segmentation frames."""

    def __init__(self):
        self.last_result: SegmentationResult | None = None

    def update(self, result: SegmentationResult) -> SegmentationResult:
        self.last_result = result
        return result
