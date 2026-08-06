"""Camera source registry and runtime adapters."""

from .base_camera import BaseCamera
from .camera_config import CameraConfig, CameraMetrics
from .camera_factory import CameraFactory
from .camera_registry import CameraRegistry, CameraSourceDefinition
from .rtsp_camera import RTSPCamera
from .usb_camera import USBCamera
from .video_file import VideoFileCamera

__all__ = [
    "BaseCamera",
    "CameraConfig",
    "CameraFactory",
    "CameraMetrics",
    "CameraRegistry",
    "CameraSourceDefinition",
    "RTSPCamera",
    "USBCamera",
    "VideoFileCamera",
]
