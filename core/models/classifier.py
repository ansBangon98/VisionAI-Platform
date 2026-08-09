from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ClassificationResult:
    label: str
    confidence: float
    class_id: int | None = None


class BaseClassifier(ABC):
    """Common contract for future classification models."""

    @abstractmethod
    def predict(self, image: object) -> ClassificationResult:
        """Run classification on an image or crop."""

