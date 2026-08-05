from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.camera.base_camera import BaseCamera
from core.camera.gstreamer import CameraConfig


@dataclass(frozen=True)
class VideoFileCamera(BaseCamera):
    path: str = ""
    loop: bool = False

    def to_camera_config(self) -> CameraConfig:
        return CameraConfig(
            source="video_file",
            file_path=str(Path(self.path).expanduser()),
        )
