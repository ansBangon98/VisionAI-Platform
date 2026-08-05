from __future__ import annotations

from dataclasses import dataclass

from core.camera.base_camera import BaseCamera
from core.camera.gstreamer import CameraConfig


@dataclass(frozen=True)
class RTSPCamera(BaseCamera):
    uri: str = ""
    latency: int = 300
    protocol: str = "tcp"

    def to_camera_config(self) -> CameraConfig:
        return CameraConfig(
            source="rtsp",
            rtsp_uri=self.uri,
            rtsp_latency=int(self.latency),
            rtsp_transport=self.protocol,
        )
