from __future__ import annotations

from dataclasses import dataclass

from core.camera.base_camera import BaseCamera
from core.camera.gstreamer import CameraConfig


@dataclass(frozen=True)
class RTSPCamera(BaseCamera):
    uri: str = ""
    latency: int = 300
    protocol: str = "tcp"
    inference_fps: int = 5
    inference_width: int = 640
    inference_height: int = 0

    def to_camera_config(self) -> CameraConfig:
        return CameraConfig(
            source="rtsp",
            rtsp_uri=self.uri,
            rtsp_latency=int(self.latency),
            rtsp_transport=self.protocol,
            inference_fps=int(self.inference_fps),
            inference_width=int(self.inference_width),
            inference_height=int(self.inference_height),
        )
