from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    bbox: tuple[float, float, float, float]
    score: float
    class_id: int


class BaseDetector(ABC):
    """Common contract for PGIE detector implementations."""

    @abstractmethod
    def predict(self, frame: object) -> list[Detection]:
        """Run model inference and return image-space detections."""

    def model_profile(self) -> dict[str, object]:
        return {}

    def set_confidence_threshold(self, confidence_threshold: float) -> None:
        del confidence_threshold

