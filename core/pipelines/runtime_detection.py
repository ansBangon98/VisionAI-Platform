from __future__ import annotations

from functools import lru_cache


DEEPSTREAM_ELEMENTS = (
    "nvstreammux",
    "nvinfer",
    "nvtracker",
    "nvvideoconvert",
    "nvdsosd",
)


@lru_cache(maxsize=1)
def load_gstreamer():
    try:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        Gst.init(None)
        return Gst
    except (ImportError, ValueError):
        return None


def require_gstreamer():
    Gst = load_gstreamer()
    if Gst is None:
        raise RuntimeError(
            "GStreamer Python bindings are not available. Install python3-gi, "
            "gir1.2-gstreamer-1.0, and the GStreamer plugin packages."
        )
    return Gst


def has_gstreamer_element(name: str) -> bool:
    Gst = load_gstreamer()
    if Gst is None:
        return False
    return Gst.ElementFactory.find(name) is not None


def deepstream_available() -> bool:
    try:
        import pyds  # noqa: F401
    except ImportError:
        return False

    return all(has_gstreamer_element(element) for element in DEEPSTREAM_ELEMENTS)


def onnxruntime_cuda_available() -> bool:
    try:
        import onnxruntime as ort
    except ImportError:
        return False

    try:
        return "CUDAExecutionProvider" in set(ort.get_available_providers())
    except Exception:
        return False


def openvino_available() -> bool:
    try:
        import openvino  # noqa: F401
    except ImportError:
        return False
    return True


def select_runtime() -> tuple[str, str]:
    if deepstream_available():
        return "deepstream", "nvinfer"
    if onnxruntime_cuda_available():
        return "cuda", "onnxruntime_cuda"
    if openvino_available():
        return "cpu", "openvino"
    return "cpu", "onnxruntime_cpu"
