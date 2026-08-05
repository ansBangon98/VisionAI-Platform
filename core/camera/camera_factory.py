from __future__ import annotations

from core.camera.base_camera import BaseCamera
from core.camera.rtsp_camera import RTSPCamera
from core.camera.usb_camera import USBCamera
from core.camera.video_file import VideoFileCamera


class CameraFactory:
    @staticmethod
    def create(config: dict) -> BaseCamera:
        source_type = str(config["type"]).lower()
        name = str(config.get("name", "camera"))
        label = str(config.get("label", _title_from_key(name)))
        reconnect = bool(config.get("reconnect", True))

        if source_type == "rtsp":
            uri = str(config.get("uri", ""))
            if not uri:
                raise RuntimeError(f"RTSP camera '{name}' has no resolved URI.")
            return RTSPCamera(
                name=name,
                label=label,
                uri=uri,
                latency=int(config.get("latency", 300)),
                protocol=str(config.get("protocol", "tcp")),
                reconnect=reconnect,
            )

        if source_type == "usb":
            return USBCamera(
                name=name,
                label=label,
                device=config.get("device", 0),
                width=int(config.get("width", 1280)),
                height=int(config.get("height", 720)),
                fps=int(config.get("fps", 30)),
                usb_format=str(config.get("usb_format", "raw")),
                reconnect=reconnect,
            )

        if source_type in {"video_file", "file", "video"}:
            return VideoFileCamera(
                name=name,
                label=label,
                path=str(config.get("path", "")),
                loop=bool(config.get("loop", False)),
                reconnect=reconnect,
            )

        raise ValueError(f"Unsupported camera type: {source_type}")


def _title_from_key(key: str) -> str:
    return key.replace("_", " ").title()
