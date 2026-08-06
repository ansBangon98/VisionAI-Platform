from __future__ import annotations

from dataclasses import dataclass

from core.camera.base_camera import BaseCamera
from core.camera.camera_config import CameraConfig


@dataclass(frozen=True)
class USBCamera(BaseCamera):
    device: int | str = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    usb_format: str = "raw"
    inference_fps: int = 5
    inference_width: int = 640
    inference_height: int = 0
    display_sink: str = "ximagesink"

    def to_camera_config(self) -> CameraConfig:
        return CameraConfig(
            source="usb",
            usb_device=self._device_path(),
            width=int(self.width),
            height=int(self.height),
            fps=int(self.fps),
            usb_format=self.usb_format,
            inference_fps=int(self.inference_fps),
            inference_width=int(self.inference_width),
            inference_height=int(self.inference_height),
            display_sink=self.display_sink,
        )

    def _device_path(self) -> str:
        if isinstance(self.device, int):
            return f"/dev/video{self.device}"

        text = str(self.device)
        if text.isdigit():
            return f"/dev/video{text}"
        return text
