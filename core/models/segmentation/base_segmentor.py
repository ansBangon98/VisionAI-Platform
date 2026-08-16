from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseSegmentor(ABC):
    """Common contract for semantic segmentation models."""

    class_labels: dict[int, str] = {}

    @abstractmethod
    def predict(self, frame: object) -> np.ndarray:
        """Return a frame-sized semantic class-id mask."""

    @abstractmethod
    def model_profile(self) -> dict[str, object]:
        """Return user-facing model/runtime metadata."""
