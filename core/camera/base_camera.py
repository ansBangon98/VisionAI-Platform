from __future__ import annotations

from dataclasses import dataclass

from core.camera.camera_config import CameraConfig


@dataclass(frozen=True)
class BaseCamera:
    name: str
    label: str
    reconnect: bool = True

    def to_camera_config(self) -> CameraConfig:
        raise NotImplementedError

    def is_opened(self) -> bool:
        raise NotImplementedError("Use the UI camera viewer for live camera state.")

    def read(self):
        raise NotImplementedError("Use the UI camera viewer frame signal for frames.")
