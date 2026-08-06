from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from core.results.frame_result import FrameResult


ResultCallback = Callable[[FrameResult], None]


class BasePipeline(ABC):
    @abstractmethod
    def build(self) -> None:
        pass

    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass

    @abstractmethod
    def set_result_callback(self, callback: ResultCallback) -> None:
        pass
