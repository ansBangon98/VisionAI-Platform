from __future__ import annotations

import os
import sys
import time

import cairo

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import argparse
import threading
from collections import deque
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

try:
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstVideo", "1.0")
    from gi.repository import Gst, GstVideo

    try:
        gi.require_version("GstRtsp", "1.0")
        from gi.repository import GstRtsp
    except (ImportError, ValueError):
        GstRtsp = None

    Gst.init(None)
    GST_IMPORT_ERROR = None
except (ImportError, ValueError) as error:
    Gst = None
    GstVideo = None
    GstRtsp = None
    GST_IMPORT_ERROR = error

from core.camera.camera_config import CameraConfig, CameraMetrics
from core.pipelines.base_pipeline import BasePipeline, ResultCallback
from core.pipelines.cpu.frame_processor import (
    frame_result_to_legacy_objects,
    legacy_objects_to_frame_result,
)
from core.results.frame_result import FrameResult


class CPUGStreamerPipeline(BasePipeline):
    """Frame-based GStreamer pipeline wrapper for CPU and CUDA backends."""

    def __init__(
        self,
        config: dict[str, Any],
        frame_processor: object | None = None,
        runtime_mode: str = "cpu",
        source_id: str | None = None,
    ):
        self.config = config
        self.frame_processor = frame_processor
        self.runtime_mode = runtime_mode
        self.source_id = source_id or _source_id_from_config(config)
        self._result_callback: ResultCallback | None = None
        self._built = False
        self._running = False
        self._frame_number = 0

    def build(self) -> None:
        self._built = True

    def start(self) -> None:
        if not self._built:
            self.build()
        self._running = True

    def stop(self) -> None:
        self._running = False

    def set_result_callback(self, callback: ResultCallback) -> None:
        self._result_callback = callback

    def process_frame_result(self, frame: object) -> FrameResult:
        if self.frame_processor is None:
            raise RuntimeError(
                "CPU GStreamer pipeline has no frame processor. Attach an "
                "application frame processor before processing frames."
            )

        processor = self.frame_processor
        process_frame_result = getattr(processor, "process_frame_result", None)
        if callable(process_frame_result):
            result = process_frame_result(frame)
        else:
            process = getattr(processor, "process", None)
            if not callable(process):
                raise RuntimeError(
                    "CPU GStreamer frame processor must provide process_frame_result() "
                    "or process()."
                )
            result = legacy_objects_to_frame_result(
                process(frame),
                source_id=self.source_id,
                frame_number=self._next_frame_number(),
            )

        if not isinstance(result, FrameResult):
            raise RuntimeError(
                "CPU GStreamer frame processor returned an unsupported result. "
                "Expected FrameResult."
            )

        self._emit_result(result)
        return result

    def process(self, frame: object) -> list[dict[str, object]]:
        return frame_result_to_legacy_objects(self.process_frame_result(frame))

    def model_profile(self) -> dict[str, object]:
        profile_getter = getattr(self.frame_processor, "model_profile", None)
        profile = profile_getter() if callable(profile_getter) else {}
        if not isinstance(profile, dict):
            profile = {}
        return {
            **profile,
            "pipeline": "GStreamer",
            "runtime_mode": self.runtime_mode,
        }

    def set_confidence_threshold(self, confidence_threshold: float) -> None:
        setter = getattr(self.frame_processor, "set_confidence_threshold", None)
        if callable(setter):
            setter(confidence_threshold)

    def _next_frame_number(self) -> int:
        self._frame_number += 1
        return self._frame_number

    def _emit_result(self, result: FrameResult) -> None:
        if self._result_callback is not None:
            self._result_callback(result)


def _source_id_from_config(config: dict[str, Any]) -> str:
    source_config = config.get("source", {})
    if isinstance(source_config, dict):
        for key in ("name", "selected", "id"):
            value = source_config.get(key)
            if value:
                return str(value)
    return "default"


class OverlayWidget(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._overlays: list[object] = []
        self._source_size: tuple[int, int] | None = None
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        if hasattr(Qt.WidgetAttribute, "WA_AlwaysStackOnTop"):
            self.setAttribute(Qt.WidgetAttribute.WA_AlwaysStackOnTop, True)
        self.setAutoFillBackground(False)
        self.setMinimumSize(1, 1)

    def set_overlays(
        self,
        overlays: list[object] | tuple[object, ...] | None,
        source_size: tuple[int, int] | None = None,
    ):
        self._overlays = list(overlays or [])
        self._source_size = source_size
        self.update()

    def clear_overlays(self):
        self._overlays = []
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self._overlays:
            return

        painter = QPainter(self)
        try:
            pen_width = max(2, min(self.width(), self.height()) // 240)
            video_rect = self._video_rect()

            for overlay in self._overlays:
                rect = _overlay_rect(overlay, self._source_size, video_rect)
                if rect is None:
                    continue

                x1, y1, x2, y2 = rect
                pen = QPen(_overlay_color(overlay))
                pen.setWidth(pen_width)
                painter.setPen(pen)
                painter.drawRect(x1, y1, x2 - x1, y2 - y1)
        finally:
            painter.end()

    def _video_rect(self) -> tuple[float, float, float, float]:
        if not self._source_size:
            return 0.0, 0.0, float(self.width()), float(self.height())

        source_width, source_height = self._source_size
        if source_width <= 0 or source_height <= 0:
            return 0.0, 0.0, float(self.width()), float(self.height())

        scale = min(self.width() / source_width, self.height() / source_height)
        video_width = source_width * scale
        video_height = source_height * scale
        x = (self.width() - video_width) / 2
        y = (self.height() - video_height) / 2
        return x, y, video_width, video_height


class CameraDisplayWidget(QWidget):
    resized = Signal(int, int)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("camera_display")
        self.setMinimumSize(1, 1)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("#camera_display { background-color: #05070a; }")

        self.video_widget = QWidget(self)
        self.video_widget.setObjectName("gstreamer_video_surface")
        self.video_widget.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
        self.video_widget.setAttribute(
            Qt.WidgetAttribute.WA_DontCreateNativeAncestors,
            True,
        )
        self.video_widget.setStyleSheet(
            "#gstreamer_video_surface { background-color: #05070a; }"
        )

        self.overlay_widget = OverlayWidget(self)

        self.message_label = QLabel(self)
        self.message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.message_label.setStyleSheet(
            "background-color: #05070a;"
            "color: #8b95a7;"
            "border: 1px solid #1f2937;"
        )
        self.show_message("No camera feed")

    def video_window_id(self) -> int:
        return int(self.video_widget.winId())

    def show_video(self):
        self.message_label.hide()
        self.video_widget.show()
        self.overlay_widget.hide()

    def show_message(self, message: str):
        self.clear_overlays()
        self.video_widget.hide()
        self.overlay_widget.hide()
        self.message_label.setText(message)
        self.message_label.show()
        self.message_label.raise_()

    def set_overlays(
        self,
        overlays: list[object] | tuple[object, ...] | None,
        source_size: tuple[int, int] | None = None,
    ):
        self.overlay_widget.set_overlays(overlays, source_size)

    def clear_overlays(self):
        self.overlay_widget.clear_overlays()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        rect = self.rect()
        self.video_widget.setGeometry(rect)
        self.overlay_widget.setGeometry(rect)
        self.message_label.setGeometry(rect)
        if self.message_label.isVisible():
            self.message_label.raise_()
        self.resized.emit(rect.width(), rect.height())


def _overlay_rect(
    overlay: object,
    source_size: tuple[int, int] | None,
    video_rect: tuple[float, float, float, float],
) -> tuple[int, int, int, int] | None:
    bbox = _overlay_value(overlay, "bbox")
    if bbox is None:
        return None

    try:
        x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    except (TypeError, ValueError):
        return None

    rect_x, rect_y, rect_width, rect_height = video_rect
    if source_size:
        source_width, source_height = source_size
    else:
        source_width = rect_width
        source_height = rect_height

    if source_width <= 0 or source_height <= 0:
        return None

    scale_x = rect_width / source_width
    scale_y = rect_height / source_height
    x1 = _clamp_int(rect_x + x1 * scale_x, 0, rect_x + rect_width)
    y1 = _clamp_int(rect_y + y1 * scale_y, 0, rect_y + rect_height)
    x2 = _clamp_int(rect_x + x2 * scale_x, 0, rect_x + rect_width)
    y2 = _clamp_int(rect_y + y2 * scale_y, 0, rect_y + rect_height)

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _overlay_color(overlay: object) -> QColor:
    item_type = _overlay_caption(overlay).lower()

    if item_type in {"person", "people"}:
        return QColor("#7DD3FC")
    if item_type == "face":
        return QColor("#86EFAC")

    palette = (
        "#FDE68A",
        "#FDA4AF",
        "#C4B5FD",
        "#67E8F9",
        "#FDBA74",
        "#BEF264",
        "#F0ABFC",
        "#93C5FD",
    )
    return QColor(palette[_overlay_class_index(overlay) % len(palette)])


def _overlay_rgba(overlay: object) -> tuple[float, float, float, float]:
    item_type = _overlay_caption(overlay).lower()

    if item_type in {"person", "people"}:
        return 0.49, 0.83, 0.99, 1.0
    if item_type == "face":
        return 0.53, 0.94, 0.67, 1.0

    palette = (
        (0.99, 0.90, 0.54, 1.0),
        (0.99, 0.64, 0.69, 1.0),
        (0.77, 0.71, 0.99, 1.0),
        (0.40, 0.91, 0.98, 1.0),
        (0.99, 0.73, 0.45, 1.0),
        (0.75, 0.95, 0.39, 1.0),
        (0.94, 0.67, 0.99, 1.0),
        (0.58, 0.77, 0.99, 1.0),
    )
    return palette[_overlay_class_index(overlay) % len(palette)]


def _overlay_caption(overlay: object) -> str:
    return str(
        _overlay_value(overlay, "label")
        or _overlay_value(overlay, "type")
        or f"class_{_overlay_class_index(overlay)}"
    ).strip()


def _overlay_class_index(overlay: object) -> int:
    try:
        return int(_overlay_value(overlay, "class_id") or 0)
    except (TypeError, ValueError):
        return 0


def _overlay_value(overlay: object, key: str):
    if isinstance(overlay, dict):
        return overlay.get(key)
    return getattr(overlay, key, None)


def _clamp_int(value: float, minimum: float, maximum: float) -> int:
    return int(round(max(minimum, min(float(value), maximum))))


def _display_sink_candidates(preferred: str) -> tuple[str, ...]:
    preferred = str(preferred or "ximagesink").strip()
    stable_defaults = ("ximagesink", "xvimagesink", "glimagesink")
    if preferred == "auto":
        return stable_defaults

    candidates = [preferred]
    candidates.extend(factory for factory in stable_defaults if factory != preferred)
    return tuple(candidates)


def _normalize_usb_pixel_format(value: str) -> str:
    normalized = str(value or "").strip().upper().replace("-", "_")
    if normalized in {"", "ANY", "AUTO", "DEFAULT"}:
        return ""

    aliases = {
        "YUYV": "YUY2",
        "RGB3": "RGB",
        "BGR3": "BGR",
        "GREY": "GRAY8",
    }
    return aliases.get(normalized, normalized)


def _implements_video_overlay(element) -> bool:
    if GstVideo is None:
        return False

    try:
        return isinstance(element, GstVideo.VideoOverlay)
    except TypeError:
        pass

    return hasattr(element, "set_window_handle")


class GStreamerCamera(QObject):
    frame_ready = Signal(QImage)
    metrics_changed = Signal(object)
    status_changed = Signal(str)
    error_occurred = Signal(str)

    def __init__(
        self,
        config: CameraConfig,
        parent: QObject | None = None,
        video_window_handle: int | None = None,
    ):
        super().__init__(parent)
        self.config = config
        self.video_window_handle = video_window_handle
        self.pipeline = None
        self.appsink = None
        self.display_sink = None
        self.cairo_overlay = None
        self.bus = None
        self._bus_sync_message_handler_id = None
        self._frame_times: deque[float] = deque(maxlen=30)
        self._overlay_lock = threading.RLock()
        self._overlay_items: list[object] = []
        self._overlay_source_size: tuple[int, int] | None = None
        self._display_frame_size: tuple[int, int] | None = None

        self.bus_timer = QTimer(self)
        self.bus_timer.setInterval(100)
        self.bus_timer.timeout.connect(self._poll_bus)

    def start(self):
        self.stop(emit_status=False)
        self._reset_metrics()
        self._ensure_gstreamer()

        if self.config.source == "rtsp":
            self.pipeline = self._build_rtsp_pipeline()
        elif self.config.source == "usb":
            self.pipeline = self._build_usb_pipeline()
        elif self.config.source == "video_file":
            self.pipeline = self._build_video_file_pipeline()
        else:
            raise RuntimeError(f"Unsupported camera source: {self.config.source}")

        self.bus = self.pipeline.get_bus()
        self._connect_bus_sync_messages()
        state_change = self.pipeline.set_state(Gst.State.PLAYING)

        if state_change == Gst.StateChangeReturn.FAILURE:
            self.stop(emit_status=False)
            raise RuntimeError("GStreamer failed to start the camera pipeline.")

        self.bus_timer.start()
        self.status_changed.emit("Camera starting...")

    def set_overlays(
        self,
        overlays: list[object] | tuple[object, ...] | None,
        source_size: tuple[int, int] | None = None,
    ):
        with self._overlay_lock:
            self._overlay_items = list(overlays or [])
            self._overlay_source_size = source_size

        self.refresh_display()

    def clear_overlays(self):
        self.set_overlays([], None)

    def refresh_display(self, width: int | None = None, height: int | None = None):
        if self.display_sink is None or not _implements_video_overlay(self.display_sink):
            return

        try:
            if width is not None and height is not None:
                self.display_sink.set_render_rectangle(0, 0, int(width), int(height))
            self.display_sink.expose()
        except (TypeError, AttributeError):
            pass

    def stop(self, emit_status: bool = True):
        self.bus_timer.stop()

        if self.pipeline is not None:
            self.pipeline.set_state(Gst.State.NULL)

        self.pipeline = None
        self.appsink = None
        self.display_sink = None
        self.cairo_overlay = None
        self._disconnect_bus_sync_messages()
        self.bus = None

        if emit_status:
            self.status_changed.emit("Camera stopped")

    def _reset_metrics(self):
        self._frame_times.clear()

    def _ensure_gstreamer(self):
        if Gst is None:
            raise RuntimeError(
                "GStreamer Python bindings are not available. Install python3-gi, "
                "gir1.2-gstreamer-1.0, and the GStreamer plugin packages."
            ) from GST_IMPORT_ERROR
        if GstVideo is None:
            raise RuntimeError(
                "GStreamer video overlay bindings are not available. Install "
                "gir1.2-gst-plugins-base-1.0."
            ) from GST_IMPORT_ERROR

    def _make_element(self, factory: str, name: str):
        element = Gst.ElementFactory.make(factory, name)
        if element is None:
            raise RuntimeError(f"Missing GStreamer element: {factory}")
        return element

    def _create_appsink(self):
        appsink = self._make_element("appsink", "infer_sink")
        sync_to_clock = self._sync_inference_to_clock()
        appsink.set_property("caps", Gst.Caps.from_string(self._inference_caps()))
        appsink.set_property("emit-signals", True)
        appsink.set_property("sync", sync_to_clock)
        appsink.set_property("max-buffers", 1)
        self._set_property_if_available(appsink, "drop", not sync_to_clock)
        if not sync_to_clock:
            self._set_property_if_available(appsink, "leaky-type", "downstream")
        appsink.connect("new-sample", self._on_new_sample)
        return appsink

    def _create_cairo_overlay(self):
        overlay = self._make_element("cairooverlay", "bbox_overlay")
        overlay.connect("draw", self._draw_cairo_overlay)
        overlay.connect("caps-changed", self._handle_overlay_caps_changed)
        return overlay

    def _handle_overlay_caps_changed(self, _overlay, caps):
        structure = caps.get_structure(0)
        width = structure.get_value("width")
        height = structure.get_value("height")
        if width and height:
            with self._overlay_lock:
                self._display_frame_size = (int(width), int(height))

    def _draw_cairo_overlay(self, _overlay, context, _timestamp, _duration):
        with self._overlay_lock:
            overlays = list(self._overlay_items)
            source_size = self._overlay_source_size
            display_size = self._display_frame_size

        if not overlays or not source_size or not display_size:
            return

        source_width, source_height = source_size
        display_width, display_height = display_size
        if source_width <= 0 or source_height <= 0:
            return

        scale_x = display_width / source_width
        scale_y = display_height / source_height
        line_width = max(2.0, min(display_width, display_height) / 240.0)
        context.set_line_width(line_width)

        for overlay in overlays:
            bbox = _overlay_value(overlay, "bbox")
            if bbox is None:
                continue

            try:
                x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
            except (TypeError, ValueError):
                continue

            x1 = max(0.0, min(x1 * scale_x, display_width))
            y1 = max(0.0, min(y1 * scale_y, display_height))
            x2 = max(0.0, min(x2 * scale_x, display_width))
            y2 = max(0.0, min(y2 * scale_y, display_height))
            if x2 <= x1 or y2 <= y1:
                continue

            red, green, blue, alpha = _overlay_rgba(overlay)
            context.set_source_rgba(red, green, blue, alpha)
            context.rectangle(x1, y1, x2 - x1, y2 - y1)
            context.stroke()

            caption = _overlay_caption(overlay)

            # Label
            context.set_source_rgba(red, green, blue, alpha)
            context.select_font_face(
                "Sans",
                cairo.FONT_SLANT_NORMAL,
                cairo.FONT_WEIGHT_BOLD
            )
            context.set_font_size(20)

            context.move_to(x1, y1 - 5)
            context.show_text(caption)

    def _inference_caps(self) -> str:
        values = [
            "video/x-raw",
            "format=RGB",
            f"framerate={max(1, int(self.config.inference_fps))}/1",
        ]
        if self.config.inference_width > 0:
            values.append(f"width={int(self.config.inference_width)}")
        if self.config.inference_height > 0:
            values.append(f"height={int(self.config.inference_height)}")
        return ",".join(values)

    def _create_video_sink_bin(self):
        sink_bin = Gst.Bin.new("analytics_video_sink_bin")
        input_queue = self._make_queue("analytics_input_queue")
        orientation_filter = self._create_orientation_filter()
        tee = self._make_element("tee", "analytics_tee")

        display_queue = self._make_queue("display_queue")
        display_convert = self._make_element("videoconvert", "display_convert")
        display_caps = self._make_element("capsfilter", "display_overlay_caps")
        display_caps.set_property("caps", Gst.Caps.from_string("video/x-raw,format=BGRx"))
        self.cairo_overlay = self._create_cairo_overlay()
        self.display_sink = self._create_display_sink()

        infer_queue = self._make_queue(
            "infer_queue",
            leaky=not self._sync_inference_to_clock(),
            max_size_buffers=1,
        )
        videorate = self._make_element("videorate", "infer_videorate")
        videoscale = self._make_element("videoscale", "infer_videoscale")
        infer_convert = self._make_element("videoconvert", "infer_convert")
        self.appsink = self._create_appsink()

        pre_tee_elements = [input_queue]
        if orientation_filter is not None:
            pre_tee_elements.append(orientation_filter)
        pre_tee_elements.append(tee)

        elements = [
            *pre_tee_elements,
            display_queue,
            display_convert,
            display_caps,
            self.cairo_overlay,
            self.display_sink,
            infer_queue,
            videorate,
            videoscale,
            infer_convert,
            self.appsink,
        ]
        for element in elements:
            sink_bin.add(element)

        self._link_many(*pre_tee_elements)
        self._link_many(
            display_queue,
            display_convert,
            display_caps,
            self.cairo_overlay,
            self.display_sink,
        )
        self._link_many(infer_queue, videorate, videoscale, infer_convert, self.appsink)

        if not tee.link(display_queue):
            raise RuntimeError("Could not link analytics tee to display queue.")
        if not tee.link(infer_queue):
            raise RuntimeError("Could not link analytics tee to inference queue.")

        sink_pad = input_queue.get_static_pad("sink")
        if sink_pad is None:
            raise RuntimeError("Could not create analytics video sink pad.")
        ghost_pad = Gst.GhostPad.new("sink", sink_pad)
        if ghost_pad is None or not sink_bin.add_pad(ghost_pad):
            raise RuntimeError("Could not add analytics video sink ghost pad.")

        self._set_video_window_handle()
        return sink_bin

    def _create_orientation_filter(self):
        element = Gst.ElementFactory.make("videoflip", "auto_orientation")
        if element is None:
            return None

        self._set_property_if_available(element, "video-direction", "auto")
        return element

    def _make_queue(
        self,
        name: str,
        leaky: bool = False,
        max_size_buffers: int = 0,
    ):
        queue = self._make_element("queue", name)
        if max_size_buffers > 0:
            self._set_property_if_available(queue, "max-size-buffers", max_size_buffers)
        if leaky:
            self._set_property_if_available(queue, "leaky", 2)
            self._set_property_if_available(queue, "max-size-time", 0)
            self._set_property_if_available(queue, "max-size-bytes", 0)
        return queue

    def _create_display_sink(self):
        factories = _display_sink_candidates(self.config.display_sink)
        sync_to_clock = self._sync_inference_to_clock()
        for factory in factories:
            sink = Gst.ElementFactory.make(factory, "display_sink")
            if sink is None:
                continue
            self._set_property_if_available(sink, "sync", sync_to_clock)
            self._set_property_if_available(sink, "force-aspect-ratio", True)
            if factory != "autovideosink":
                return sink
            if _implements_video_overlay(sink):
                return sink

        raise RuntimeError(
            "No usable GStreamer video sink found. Install GStreamer good/base "
            "plugins, for example gstreamer1.0-plugins-good."
        )

    def _link_many(self, *elements) -> None:
        for current, next_element in zip(elements, elements[1:]):
            if not current.link(next_element):
                raise RuntimeError(
                    f"Could not link GStreamer elements: "
                    f"{current.get_name()} -> {next_element.get_name()}"
                )

    def _set_video_window_handle(self):
        if self.display_sink is None or not self.video_window_handle:
            return
        if not _implements_video_overlay(self.display_sink):
            return

        try:
            if hasattr(self.display_sink, "set_window_handle"):
                self.display_sink.set_window_handle(self.video_window_handle)
            else:
                GstVideo.VideoOverlay.set_window_handle(
                    self.display_sink,
                    self.video_window_handle,
                )
        except (TypeError, AttributeError):
            return

    def _connect_bus_sync_messages(self):
        if self.bus is None or self._bus_sync_message_handler_id is not None:
            return

        try:
            self.bus.enable_sync_message_emission()
            self._bus_sync_message_handler_id = self.bus.connect(
                "sync-message::element",
                self._handle_sync_message,
            )
        except (TypeError, AttributeError):
            self._bus_sync_message_handler_id = None

    def _disconnect_bus_sync_messages(self):
        if self.bus is None or self._bus_sync_message_handler_id is None:
            return

        try:
            self.bus.disconnect(self._bus_sync_message_handler_id)
            self.bus.disable_sync_message_emission()
        except (TypeError, AttributeError):
            pass
        finally:
            self._bus_sync_message_handler_id = None

    def _handle_sync_message(self, _bus, message):
        if not self.video_window_handle or GstVideo is None:
            return
        if not GstVideo.is_video_overlay_prepare_window_handle_message(message):
            return
        if not _implements_video_overlay(message.src):
            return

        try:
            message.src.set_window_handle(self.video_window_handle)
        except (TypeError, AttributeError):
            GstVideo.VideoOverlay.set_window_handle(
                message.src,
                self.video_window_handle,
            )

    def _build_rtsp_pipeline(self):
        if not self.config.rtsp_uri:
            raise RuntimeError("RTSP URI is required.")

        playbin = self._make_element("playbin", "rtsp_pipeline")

        playbin.set_property("uri", self.config.rtsp_uri)
        playbin.set_property("video-sink", self._create_video_sink_bin())

        audio_sink = Gst.ElementFactory.make("fakesink", "audio_sink")
        if audio_sink is not None:
            playbin.set_property("audio-sink", audio_sink)

        playbin.connect("source-setup", self._configure_rtsp_source)
        return playbin

    def _build_video_file_pipeline(self):
        if not self.config.file_path:
            raise RuntimeError("Video file path is required.")

        playbin = self._make_element("playbin", "video_file_pipeline")

        if "://" in self.config.file_path:
            uri = self.config.file_path
        else:
            uri = Path(self.config.file_path).expanduser().resolve().as_uri()

        playbin.set_property("uri", uri)
        playbin.set_property("video-sink", self._create_video_sink_bin())

        audio_sink = Gst.ElementFactory.make("fakesink", "audio_sink")
        if audio_sink is not None:
            playbin.set_property("audio-sink", audio_sink)

        return playbin

    def _configure_rtsp_source(self, _playbin, source):
        self._set_property_if_available(source, "latency", self.config.rtsp_latency)

        if GstRtsp is None:
            return

        if self.config.rtsp_transport == "tcp":
            protocols = GstRtsp.RTSPLowerTrans.TCP
        else:
            protocols = GstRtsp.RTSPLowerTrans.UDP

        self._set_property_if_available(source, "protocols", protocols)

    def _set_property_if_available(self, element, name: str, value):
        if element.find_property(name) is None:
            return

        try:
            element.set_property(name, value)
        except (TypeError, ValueError):
            return

    def _sync_inference_to_clock(self) -> bool:
        return self.config.source == "video_file"

    def _build_usb_pipeline(self):
        pipeline = Gst.Pipeline.new("usb_camera_pipeline")
        source = self._make_element("v4l2src", "usb_source")
        input_caps = self._make_element("capsfilter", "usb_input_caps")
        decode_convert = self._make_element("videoconvert", "usb_decode_convert")
        video_sink = self._create_video_sink_bin()

        source.set_property("device", self.config.usb_device)
        input_caps.set_property("caps", Gst.Caps.from_string(self._usb_input_caps()))

        elements = [source, input_caps]

        if str(self.config.usb_format or "raw").lower() == "mjpeg":
            elements.append(self._make_element("jpegdec", "jpeg_decoder"))

        elements.append(decode_convert)
        elements.append(video_sink)

        for element in elements:
            pipeline.add(element)

        self._link_many(*elements)

        return pipeline

    def _usb_input_caps(self) -> str:
        usb_format = str(self.config.usb_format or "raw").lower()
        media_type = "image/jpeg" if usb_format == "mjpeg" else "video/x-raw"
        values = [
            media_type,
            f"width={self.config.width}",
            f"height={self.config.height}",
            f"framerate={self.config.fps}/1",
        ]

        pixel_format = _normalize_usb_pixel_format(self.config.usb_pixel_format)
        if media_type == "video/x-raw" and pixel_format:
            values.insert(1, f"format={pixel_format}")

        return ",".join(values)

    def _on_new_sample(self, sink):
        sample_started_at = time.perf_counter()
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR

        buffer = sample.get_buffer()
        caps = sample.get_caps()
        structure = caps.get_structure(0)
        width = structure.get_value("width")
        height = structure.get_value("height")

        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR

        try:
            bytes_per_line = map_info.size // height
            image = QImage(
                map_info.data,
                width,
                height,
                bytes_per_line,
                QImage.Format.Format_RGB888,
            ).copy()
            self.frame_ready.emit(image)
            self._emit_metrics(buffer, width, height, sample_started_at)
        finally:
            buffer.unmap(map_info)

        return Gst.FlowReturn.OK

    def _emit_metrics(self, buffer, width: int, height: int, sample_started_at: float):
        self._frame_times.append(time.perf_counter())

        fps = 0.0
        if len(self._frame_times) >= 2:
            elapsed = self._frame_times[-1] - self._frame_times[0]
            if elapsed > 0:
                fps = (len(self._frame_times) - 1) / elapsed

        latency_ms = self._buffer_latency_ms(buffer)
        if latency_ms is None:
            latency_ms = (time.perf_counter() - sample_started_at) * 1000

        self.metrics_changed.emit(
            CameraMetrics(
                fps=fps,
                latency_ms=latency_ms,
                width=int(width),
                height=int(height),
            )
        )

    def _buffer_latency_ms(self, buffer) -> float | None:
        if self.pipeline is None:
            return None

        timestamp = buffer.pts
        if timestamp == Gst.CLOCK_TIME_NONE:
            timestamp = buffer.dts

        if timestamp == Gst.CLOCK_TIME_NONE:
            return None

        clock = self.pipeline.get_clock()
        if clock is None:
            return None

        running_time = clock.get_time() - self.pipeline.get_base_time()
        return max(0.0, (running_time - timestamp) / Gst.MSECOND)

    def _poll_bus(self):
        if self.bus is None:
            return

        message_types = (
            Gst.MessageType.ERROR
            | Gst.MessageType.WARNING
            | Gst.MessageType.EOS
            | Gst.MessageType.STATE_CHANGED
        )

        bus = self.bus
        while bus is self.bus:
            message = bus.timed_pop_filtered(0, message_types)
            if message is None:
                break

            self._handle_bus_message(message)

    def _handle_bus_message(self, message):
        if message.type == Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            detail = f"{error.message}"
            if debug:
                detail = f"{detail}\n{debug}"
            self.error_occurred.emit(detail)
            self.stop(emit_status=False)
            return

        if message.type == Gst.MessageType.WARNING:
            warning, _debug = message.parse_warning()
            self.status_changed.emit(f"GStreamer warning: {warning.message}")
            return

        if message.type == Gst.MessageType.EOS:
            self.status_changed.emit("Camera stream ended")
            self.stop(emit_status=False)
            return

        if message.type == Gst.MessageType.STATE_CHANGED and message.src == self.pipeline:
            _old_state, new_state, _pending_state = message.parse_state_changed()
            if new_state == Gst.State.PLAYING:
                self.status_changed.emit("Camera playing")


class CameraViewerWidget(QWidget):
    frame_ready = Signal(QImage)
    metrics_changed = Signal(object)
    status_changed = Signal(str)
    error_occurred = Signal(str)

    def __init__(
        self,
        config: CameraConfig | None = None,
        parent: QWidget | None = None,
        auto_start: bool = False,
    ):
        super().__init__(parent)
        self.setObjectName("camera_viewer")

        self.config = config or CameraConfig()
        self.camera: GStreamerCamera | None = None
        self.camera_feed = CameraDisplayWidget()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.camera_feed)
        self.camera_feed.resized.connect(self.refresh_display)

        if auto_start:
            QTimer.singleShot(0, self.start)

    def set_config(self, config: CameraConfig):
        self.config = config

    def start_usb(
        self,
        usb_device: str = "/dev/video0",
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        usb_format: str = "raw",
        usb_pixel_format: str = "",
    ):
        self.start(
            CameraConfig(
                source="usb",
                usb_device=usb_device,
                width=width,
                height=height,
                fps=fps,
                usb_format=usb_format,
                usb_pixel_format=usb_pixel_format,
            )
        )

    def start_rtsp(
        self,
        rtsp_uri: str,
        rtsp_latency: int = 200,
        rtsp_transport: str = "tcp",
    ):
        self.start(
            CameraConfig(
                source="rtsp",
                rtsp_uri=rtsp_uri,
                rtsp_latency=rtsp_latency,
                rtsp_transport=rtsp_transport,
            )
        )

    def start(self, config: CameraConfig | None = None):
        if config is not None:
            self.config = config

        try:
            self.stop(clear_feed=False)
            self.camera_feed.show_video()
            self.camera = GStreamerCamera(
                self.config,
                self,
                video_window_handle=self.camera_feed.video_window_id(),
            )
            self.camera.frame_ready.connect(self.frame_ready.emit)
            self.camera.metrics_changed.connect(self.metrics_changed.emit)
            self.camera.status_changed.connect(self.status_changed.emit)
            self.camera.error_occurred.connect(self._handle_error)
            self.camera.start()
        except RuntimeError as error:
            self._handle_error(str(error))
            raise

    def set_overlays(
        self,
        overlays: list[object] | tuple[object, ...] | None,
        source_size: tuple[int, int] | None = None,
    ):
        self.camera_feed.set_overlays(overlays, source_size)
        if self.camera is not None:
            self.camera.set_overlays(overlays, source_size)

    def clear_overlays(self):
        self.camera_feed.clear_overlays()
        if self.camera is not None:
            self.camera.clear_overlays()

    def refresh_display(self, width: int | None = None, height: int | None = None):
        if self.camera is not None:
            self.camera.refresh_display(width, height)

    def stop(self, clear_feed: bool = True):
        if self.camera is not None:
            self.camera.clear_overlays()
            self.camera.stop()
            self.camera.deleteLater()
            self.camera = None

        if clear_feed:
            self.camera_feed.show_message("No camera feed")

    def _handle_error(self, message: str):
        self.camera_feed.show_message("Camera stopped")
        self.status_changed.emit(message)
        self.error_occurred.emit(message)


def attach_camera_viewer(
    container: QWidget,
    config: CameraConfig | None = None,
    auto_start: bool = False,
) -> CameraViewerWidget:
    viewer = CameraViewerWidget(config=config, parent=container, auto_start=auto_start)

    layout = container.layout()
    if layout is None:
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

    layout.addWidget(viewer)
    return viewer


class StandaloneCameraWindow(QMainWindow):
    def __init__(self, config: CameraConfig, auto_start: bool = True):
        super().__init__()
        self.setWindowTitle("Standalone GStreamer Camera")
        self.resize(960, 640)

        self.camera_viewer = CameraViewerWidget(config)
        self.camera_feed = self.camera_viewer.camera_feed
        self.source_selector = QComboBox()
        self.source_selector.addItem("USB camera", "usb")
        self.source_selector.addItem("RTSP stream", "rtsp")

        self.usb_format_selector = QComboBox()
        self.usb_format_selector.addItem("Raw", "raw")
        self.usb_format_selector.addItem("MJPEG", "mjpeg")
        self.usb_pixel_format_input = QLineEdit(config.usb_pixel_format)
        self.usb_pixel_format_input.setPlaceholderText("auto, YUY2, NV12")

        self.rtsp_uri_input = QLineEdit(config.rtsp_uri)
        self.rtsp_uri_input.setPlaceholderText("rtsp://username:password@host:554/stream")

        self.usb_device_input = QLineEdit(config.usb_device)

        self.width_input = self._spinbox(160, 7680, config.width)
        self.height_input = self._spinbox(120, 4320, config.height)
        self.fps_input = self._spinbox(1, 240, config.fps)
        self.latency_input = self._spinbox(0, 5000, config.rtsp_latency)

        self.transport_selector = QComboBox()
        self.transport_selector.addItem("TCP", "tcp")
        self.transport_selector.addItem("UDP", "udp")

        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.status_label = QLabel("Ready")

        self._build_layout()
        self._apply_config_to_controls(config)
        self._connect_signals()
        self._update_control_state()

        if auto_start:
            QTimer.singleShot(0, self.start_camera)

    def _spinbox(self, minimum: int, maximum: int, value: int):
        spinbox = QSpinBox()
        spinbox.setRange(minimum, maximum)
        spinbox.setValue(value)
        return spinbox

    def _build_layout(self):
        form = QFormLayout()
        form.addRow("Source", self.source_selector)
        form.addRow("RTSP URI", self.rtsp_uri_input)
        form.addRow("RTSP latency", self.latency_input)
        form.addRow("RTSP transport", self.transport_selector)
        form.addRow("USB device", self.usb_device_input)
        form.addRow("USB format", self.usb_format_selector)
        form.addRow("USB pixel format", self.usb_pixel_format_input)
        form.addRow("Width", self.width_input)
        form.addRow("Height", self.height_input)
        form.addRow("FPS", self.fps_input)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addStretch()

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.addLayout(form)
        controls_layout.addLayout(button_layout)
        controls_layout.addWidget(self.status_label)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.camera_viewer, stretch=1)
        main_layout.addWidget(controls)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def _apply_config_to_controls(self, config: CameraConfig):
        self._set_combo_data(self.source_selector, config.source)
        self._set_combo_data(self.usb_format_selector, config.usb_format)
        self._set_combo_data(self.transport_selector, config.rtsp_transport)
        self.usb_pixel_format_input.setText(config.usb_pixel_format)

    def _set_combo_data(self, combo: QComboBox, value: str):
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _connect_signals(self):
        self.source_selector.currentIndexChanged.connect(self._update_control_state)
        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)
        self.camera_viewer.status_changed.connect(self.status_label.setText)
        self.camera_viewer.error_occurred.connect(self._show_error)

    def _update_control_state(self):
        source = self.source_selector.currentData()
        is_rtsp = source == "rtsp"

        self.rtsp_uri_input.setEnabled(is_rtsp)
        self.latency_input.setEnabled(is_rtsp)
        self.transport_selector.setEnabled(is_rtsp)
        self.usb_device_input.setEnabled(not is_rtsp)
        self.usb_format_selector.setEnabled(not is_rtsp)
        self.usb_pixel_format_input.setEnabled(not is_rtsp)
        self.width_input.setEnabled(not is_rtsp)
        self.height_input.setEnabled(not is_rtsp)
        self.fps_input.setEnabled(not is_rtsp)

    def start_camera(self):
        try:
            config = self._camera_config_from_controls()
            self.camera_viewer.start(config)

            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
        except RuntimeError as error:
            self.status_label.setText(str(error))
            QMessageBox.critical(self, "Camera error", str(error))

    def stop_camera(self, clear_feed: bool = True):
        self.camera_viewer.stop(clear_feed=clear_feed)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def _camera_config_from_controls(self) -> CameraConfig:
        rtsp_uri = self.rtsp_uri_input.text().strip()
        source = self.source_selector.currentData()

        if source == "rtsp" and not rtsp_uri:
            raise RuntimeError("Enter an RTSP URI before starting the stream.")

        return CameraConfig(
            source=source,
            rtsp_uri=rtsp_uri,
            usb_device=self.usb_device_input.text().strip() or "/dev/video0",
            width=self.width_input.value(),
            height=self.height_input.value(),
            fps=self.fps_input.value(),
            usb_format=self.usb_format_selector.currentData(),
            usb_pixel_format=self.usb_pixel_format_input.text().strip(),
            rtsp_latency=self.latency_input.value(),
            rtsp_transport=self.transport_selector.currentData(),
        )

    def _show_error(self, message: str):
        self.status_label.setText(message)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def closeEvent(self, event):
        self.stop_camera()
        super().closeEvent(event)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone GStreamer camera QLabel viewer.")
    parser.add_argument("--source", choices=("usb", "rtsp"), default="usb")
    parser.add_argument("--rtsp-uri", default="")
    parser.add_argument("--rtsp-latency", type=int, default=200)
    parser.add_argument("--rtsp-transport", choices=("tcp", "udp"), default="tcp")
    parser.add_argument("--usb-device", default="/dev/video0")
    parser.add_argument("--usb-format", choices=("raw", "mjpeg"), default="raw")
    parser.add_argument("--usb-pixel-format", default="")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--no-auto-start", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(sys.argv[1:] if argv is None else argv)

    app = QApplication([sys.argv[0]])
    window = StandaloneCameraWindow(
        CameraConfig(
            source=args.source,
            rtsp_uri=args.rtsp_uri,
            usb_device=args.usb_device,
            width=args.width,
            height=args.height,
            fps=args.fps,
            usb_format=args.usb_format,
            usb_pixel_format=args.usb_pixel_format,
            rtsp_latency=args.rtsp_latency,
            rtsp_transport=args.rtsp_transport,
        ),
        auto_start=not args.no_auto_start,
    )
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
