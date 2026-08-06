from __future__ import annotations

from pathlib import Path
from typing import Any

from core.camera import CameraRegistry
from core.pipelines.base_pipeline import BasePipeline, ResultCallback
from core.pipelines.deepstream.probe_manager import ProbeManager
from core.pipelines.deepstream.source_bin import (
    create_source_bin,
    source_uri_from_config,
)
from core.pipelines.runtime_detection import deepstream_available, require_gstreamer


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class DeepStreamPipeline(BasePipeline):
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.pipeline = None
        self._result_callback: ResultCallback | None = None
        self._probe_manager: ProbeManager | None = None
        self._built = False

    def build(self) -> None:
        if not deepstream_available():
            raise RuntimeError(
                "DeepStream runtime is unavailable. Install NVIDIA DeepStream, "
                "pyds, and the required GStreamer plugins."
            )

        Gst = require_gstreamer()
        deepstream_config = self._deepstream_config()
        pgie_config = _required_path(deepstream_config.get("pgie_config"), "pgie_config")
        tracker_config = _optional_path(deepstream_config.get("tracker_config"))

        source_config = self._selected_source_config()
        source_uri = source_uri_from_config(source_config)
        source_id = str(source_config.get("name") or self._source_name() or "0")

        pipeline = Gst.Pipeline.new("deepstream_analytics_pipeline")
        if pipeline is None:
            raise RuntimeError("Unable to create DeepStream GStreamer pipeline.")

        source_bin = create_source_bin(0, source_uri)
        streammux = _make_element(Gst, "nvstreammux", "streammux")
        pgie = _make_element(Gst, "nvinfer", "primary-inference")
        tracker = _make_element(Gst, "nvtracker", "tracker")
        converter = _make_element(Gst, "nvvideoconvert", "nvvideo-converter")
        osd = _make_element(Gst, "nvdsosd", "onscreendisplay")
        sink = _make_element(Gst, "fakesink", "metadata-sink")

        _set_property_if_available(
            streammux,
            "batch-size",
            int(deepstream_config.get("batch_size", 1)),
        )
        _set_property_if_available(streammux, "width", int(deepstream_config.get("width", 1280)))
        _set_property_if_available(streammux, "height", int(deepstream_config.get("height", 720)))
        _set_property_if_available(
            streammux,
            "batched-push-timeout",
            int(deepstream_config.get("batched_push_timeout", 40000)),
        )
        pgie.set_property("config-file-path", str(pgie_config))
        if tracker_config is not None:
            _set_property_if_available(tracker, "ll-config-file", str(tracker_config))
        _set_property_if_available(sink, "sync", False)

        for element in (source_bin, streammux, pgie, tracker, converter, osd, sink):
            pipeline.add(element)

        sink_pad = streammux.get_request_pad("sink_0")
        src_pad = source_bin.get_static_pad("src")
        if sink_pad is None or src_pad is None:
            raise RuntimeError("Unable to get DeepStream source or streammux pads.")
        if src_pad.link(sink_pad) != Gst.PadLinkReturn.OK:
            raise RuntimeError("Unable to link DeepStream source to streammux.")

        _link_many(streammux, pgie, tracker, converter, osd, sink)
        self._probe_manager = ProbeManager(
            self._result_callback,
            source_ids={0: source_id},
        )
        self._probe_manager.attach(tracker, "src")
        self.pipeline = pipeline
        self._built = True

    def start(self) -> None:
        if not self._built:
            self.build()
        Gst = require_gstreamer()
        state_change = self.pipeline.set_state(Gst.State.PLAYING)
        if state_change == Gst.StateChangeReturn.FAILURE:
            self.stop()
            raise RuntimeError("DeepStream pipeline failed to start.")

    def stop(self) -> None:
        if self.pipeline is None:
            return
        Gst = require_gstreamer()
        self.pipeline.set_state(Gst.State.NULL)

    def set_result_callback(self, callback: ResultCallback) -> None:
        self._result_callback = callback
        if self._probe_manager is not None:
            self._probe_manager.callback = callback

    def _deepstream_config(self) -> dict[str, Any]:
        value = self.config.get("deepstream", {})
        return value if isinstance(value, dict) else {}

    def _selected_source_config(self) -> dict[str, Any]:
        source_name = self._source_name()
        if not source_name:
            raise RuntimeError("DeepStream mode requires source.name or source.selected.")
        source_config = CameraRegistry().get(source_name)
        source_config.setdefault("name", source_name)
        return source_config

    def _source_name(self) -> str:
        source = self.config.get("source", {})
        if not isinstance(source, dict):
            return ""
        return str(source.get("name") or source.get("selected") or "")


def _make_element(Gst, factory: str, name: str):
    element = Gst.ElementFactory.make(factory, name)
    if element is None:
        raise RuntimeError(f"Missing GStreamer element: {factory}")
    return element


def _link_many(*elements: object) -> None:
    for current, next_element in zip(elements, elements[1:]):
        if not current.link(next_element):
            raise RuntimeError(
                "Could not link DeepStream elements: "
                f"{current.get_name()} -> {next_element.get_name()}"
            )


def _set_property_if_available(element: object, name: str, value: object) -> None:
    if element.find_property(name) is None:
        return
    element.set_property(name, value)


def _required_path(value: object, name: str) -> Path:
    if not value:
        raise RuntimeError(f"DeepStream config is missing required field: {name}.")
    path = _resolve_project_path(str(value))
    if not path.exists():
        raise RuntimeError(f"DeepStream {name} does not exist: {path}")
    return path


def _optional_path(value: object) -> Path | None:
    if not value:
        return None
    path = _resolve_project_path(str(value))
    if not path.exists():
        raise RuntimeError(f"DeepStream tracker_config does not exist: {path}")
    return path


def _resolve_project_path(path: str) -> Path:
    resolved = Path(path).expanduser()
    if resolved.is_absolute():
        return resolved
    return PROJECT_ROOT / resolved
