from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
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

    try:
        gi.require_version("GstRtsp", "1.0")
        from gi.repository import Gst, GstRtsp
    except (ImportError, ValueError):
        from gi.repository import Gst

        GstRtsp = None

    Gst.init(None)
    GST_IMPORT_ERROR = None
except (ImportError, ValueError) as error:
    Gst = None
    GstRtsp = None
    GST_IMPORT_ERROR = error


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


@dataclass(frozen=True)
class CameraMetrics:
    fps: float
    latency_ms: float | None
    width: int
    height: int


class CameraFeedLabel(QLabel):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._image = QImage()
        self.setObjectName("camera_feed")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(1, 1)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(
            "#camera_feed {"
            "background-color: #05070a;"
            "color: #8b95a7;"
            "border: 1px solid #1f2937;"
            "}"
        )
        self.clear_feed("No camera feed")

    def set_image(self, image: QImage):
        self._image = image
        self._paint_image()

    def clear_feed(self, message: str):
        self._image = QImage()
        self.clear()
        self.setText(message)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._paint_image()

    def _paint_image(self):
        if self._image.isNull():
            return

        pixmap = QPixmap.fromImage(self._image).scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(pixmap)


class GStreamerCamera(QObject):
    frame_ready = Signal(QImage)
    metrics_changed = Signal(object)
    status_changed = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, config: CameraConfig, parent: QObject | None = None):
        super().__init__(parent)
        self.config = config
        self.pipeline = None
        self.appsink = None
        self.bus = None
        self._frame_times: deque[float] = deque(maxlen=30)

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
        state_change = self.pipeline.set_state(Gst.State.PLAYING)

        if state_change == Gst.StateChangeReturn.FAILURE:
            self.stop(emit_status=False)
            raise RuntimeError("GStreamer failed to start the camera pipeline.")

        self.bus_timer.start()
        self.status_changed.emit("Camera starting...")

    def stop(self, emit_status: bool = True):
        self.bus_timer.stop()

        if self.pipeline is not None:
            self.pipeline.set_state(Gst.State.NULL)

        self.pipeline = None
        self.appsink = None
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

    def _make_element(self, factory: str, name: str):
        element = Gst.ElementFactory.make(factory, name)
        if element is None:
            raise RuntimeError(f"Missing GStreamer element: {factory}")
        return element

    def _create_appsink(self):
        appsink = self._make_element("appsink", "video_sink")
        appsink.set_property("caps", Gst.Caps.from_string("video/x-raw,format=RGB"))
        appsink.set_property("emit-signals", True)
        appsink.set_property("sync", False)
        appsink.set_property("max-buffers", 1)
        appsink.set_property("drop", True)
        appsink.connect("new-sample", self._on_new_sample)
        return appsink

    def _build_rtsp_pipeline(self):
        if not self.config.rtsp_uri:
            raise RuntimeError("RTSP URI is required.")

        playbin = self._make_element("playbin", "rtsp_pipeline")
        self.appsink = self._create_appsink()

        playbin.set_property("uri", self.config.rtsp_uri)
        playbin.set_property("video-sink", self.appsink)

        audio_sink = Gst.ElementFactory.make("fakesink", "audio_sink")
        if audio_sink is not None:
            playbin.set_property("audio-sink", audio_sink)

        playbin.connect("source-setup", self._configure_rtsp_source)
        return playbin

    def _build_video_file_pipeline(self):
        if not self.config.file_path:
            raise RuntimeError("Video file path is required.")

        playbin = self._make_element("playbin", "video_file_pipeline")
        self.appsink = self._create_appsink()

        if "://" in self.config.file_path:
            uri = self.config.file_path
        else:
            uri = Path(self.config.file_path).expanduser().resolve().as_uri()

        playbin.set_property("uri", uri)
        playbin.set_property("video-sink", self.appsink)

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
        except TypeError:
            return

    def _build_usb_pipeline(self):
        pipeline = Gst.Pipeline.new("usb_camera_pipeline")
        source = self._make_element("v4l2src", "usb_source")
        input_caps = self._make_element("capsfilter", "usb_input_caps")
        convert = self._make_element("videoconvert", "video_convert")
        output_caps = self._make_element("capsfilter", "rgb_output_caps")
        self.appsink = self._create_appsink()

        source.set_property("device", self.config.usb_device)
        input_caps.set_property("caps", Gst.Caps.from_string(self._usb_input_caps()))
        output_caps.set_property("caps", Gst.Caps.from_string("video/x-raw,format=RGB"))

        elements = [source, input_caps]

        if self.config.usb_format == "mjpeg":
            elements.append(self._make_element("jpegdec", "jpeg_decoder"))

        elements.extend([convert, output_caps, self.appsink])

        for element in elements:
            pipeline.add(element)

        for current, next_element in zip(elements, elements[1:]):
            if not current.link(next_element):
                raise RuntimeError(
                    f"Could not link GStreamer elements: "
                    f"{current.get_name()} -> {next_element.get_name()}"
                )

        return pipeline

    def _usb_input_caps(self) -> str:
        media_type = "image/jpeg" if self.config.usb_format == "mjpeg" else "video/x-raw"
        return (
            f"{media_type},width={self.config.width},height={self.config.height},"
            f"framerate={self.config.fps}/1"
        )

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
        self.camera_feed = CameraFeedLabel()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.camera_feed)

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
    ):
        self.start(
            CameraConfig(
                source="usb",
                usb_device=usb_device,
                width=width,
                height=height,
                fps=fps,
                usb_format=usb_format,
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
            self.camera = GStreamerCamera(self.config, self)
            self.camera.frame_ready.connect(self.camera_feed.set_image)
            self.camera.frame_ready.connect(self.frame_ready.emit)
            self.camera.metrics_changed.connect(self.metrics_changed.emit)
            self.camera.status_changed.connect(self.status_changed.emit)
            self.camera.error_occurred.connect(self._handle_error)
            self.camera.start()
        except RuntimeError as error:
            self._handle_error(str(error))
            raise

    def stop(self, clear_feed: bool = True):
        if self.camera is not None:
            self.camera.stop()
            self.camera.deleteLater()
            self.camera = None

        if clear_feed:
            self.camera_feed.clear_feed("No camera feed")

    def _handle_error(self, message: str):
        self.camera_feed.clear_feed("Camera stopped")
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
            rtsp_latency=args.rtsp_latency,
            rtsp_transport=args.rtsp_transport,
        ),
        auto_start=not args.no_auto_start,
    )
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
