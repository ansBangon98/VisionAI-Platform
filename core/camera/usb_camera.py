from __future__ import annotations

from dataclasses import dataclass

from core.camera.base_camera import BaseCamera
from core.camera.gstreamer import CameraConfig


@dataclass(frozen=True)
class USBCamera(BaseCamera):
    device: int | str = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    usb_format: str = "raw"

    def to_camera_config(self) -> CameraConfig:
        return CameraConfig(
            source="usb",
            usb_device=self._device_path(),
            width=int(self.width),
            height=int(self.height),
            fps=int(self.fps),
            usb_format=self.usb_format,
        )

    def _device_path(self) -> str:
        if isinstance(self.device, int):
            return f"/dev/video{self.device}"

        text = str(self.device)
        if text.isdigit():
            return f"/dev/video{text}"
        return text
