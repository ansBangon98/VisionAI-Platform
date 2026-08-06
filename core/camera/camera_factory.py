from __future__ import annotations

from core.camera.base_camera import BaseCamera
from core.camera.rtsp_camera import RTSPCamera
from core.camera.usb_camera import USBCamera
from core.camera.video_file import VideoFileCamera


class CameraFactory:
    @staticmethod
    def create(config: dict) -> BaseCamera:
        if "type" not in config:
            name = str(config.get("name", "camera"))
            raise RuntimeError(f"Camera source '{name}' is missing required field: type")

        source_type = str(config["type"]).lower()
        name = str(config.get("name", "camera"))
        label = str(config.get("label", _title_from_key(name)))
        reconnect = bool(config.get("reconnect", True))
        inference_config = _inference_config(config)
        display_sink = str(config.get("display_sink", "ximagesink"))

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
                display_sink=display_sink,
                **inference_config,
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
                display_sink=display_sink,
                **inference_config,
            )

        if source_type in {"video_file", "file", "video"}:
            return VideoFileCamera(
                name=name,
                label=label,
                path=str(config.get("path", "")),
                loop=bool(config.get("loop", False)),
                reconnect=reconnect,
                display_sink=display_sink,
                **inference_config,
            )

        raise RuntimeError(f"Unsupported camera type: {source_type}")


def _title_from_key(key: str) -> str:
    return key.replace("_", " ").title()


def _inference_config(config: dict) -> dict[str, int]:
    nested = config.get("inference", {})
    if not isinstance(nested, dict):
        nested = {}

    return {
        "inference_fps": int(
            nested.get("fps", config.get("inference_fps", 5))
        ),
        "inference_width": int(
            nested.get("width", config.get("inference_width", 640))
        ),
        "inference_height": int(
            nested.get("height", config.get("inference_height", 0))
        ),
    }
