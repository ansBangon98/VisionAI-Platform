from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.camera.base_camera import BaseCamera
from core.camera.gstreamer import CameraConfig


@dataclass(frozen=True)
class VideoFileCamera(BaseCamera):
    path: str = ""
    loop: bool = False
    inference_fps: int = 5
    inference_width: int = 640
    inference_height: int = 0

    def to_camera_config(self) -> CameraConfig:
        return CameraConfig(
            source="video_file",
            file_path=str(Path(self.path).expanduser()),
            inference_fps=int(self.inference_fps),
            inference_width=int(self.inference_width),
            inference_height=int(self.inference_height),
        )
