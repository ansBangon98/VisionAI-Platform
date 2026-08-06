from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CameraConfig:
    source: str = "usb"
    rtsp_uri: str = ""
    usb_device: str = "/dev/video0"
    width: int = 640
    height: int = 480
    fps: int = 30
    usb_format: str = "raw"
    rtsp_latency: int = 200
    rtsp_transport: str = "tcp"
    file_path: str = ""
    inference_fps: int = 5
    inference_width: int = 640
    inference_height: int = 0
    display_sink: str = "ximagesink"


@dataclass(frozen=True)
class CameraMetrics:
    fps: float
    latency_ms: float | None
    width: int
    height: int


__all__ = ["CameraConfig", "CameraMetrics"]
