from __future__ import annotations

from abc import ABC, abstractmethod


class BaseSegmentor(ABC):
    """Common contract for future segmentation models."""

    @abstractmethod
    def predict(self, frame: object) -> object:
        """Return a segmentation mask or structured segmentation result."""

