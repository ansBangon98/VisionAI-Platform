from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """Common contract for future embedding models."""

    @abstractmethod
    def predict(self, image: object) -> object:
        """Return an embedding vector for an image or crop."""

