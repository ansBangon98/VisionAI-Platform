#!/usr/bin/env python3

import os
import sys

# Use Qt's X11/XCB backend because this example uses ximagesink.
# This must be set before importing PySide6.
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstVideo", "1.0")

from gi.repository import Gst, GstVideo
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class GStreamerWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("PySide6 GStreamer Demo")
        self.resize(900, 600)

        Gst.init(None)

        self.pipeline: Gst.Pipeline | None = None
        self.video_sink: Gst.Element | None = None
        self.bus: Gst.Bus | None = None

        self._build_ui()
        self._build_pipeline()

        # Poll the GStreamer bus without needing a separate GLib main loop.
        self.bus_timer = QTimer(self)
        self.bus_timer.timeout.connect(self._check_bus_messages)
        self.bus_timer.start(100)

    def _build_ui(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        self.video_widget = QWidget()
        self.video_widget.setMinimumSize(640, 360)
        self.video_widget.setStyleSheet("background-color: black;")

        # Force Qt to create a native window that GStreamer can draw into.
        self.video_widget.setAttribute(
            Qt.WidgetAttribute.WA_NativeWindow,
            True,
        )

        self.status_label = QLabel("Status: stopped")

        self.start_button = QPushButton("Start")
        self.pause_button = QPushButton("Pause")
        self.stop_button = QPushButton("Stop")

        self.start_button.clicked.connect(self.start_pipeline)
        self.pause_button.clicked.connect(self.pause_pipeline)
        self.stop_button.clicked.connect(self.stop_pipeline)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.pause_button)
        button_layout.addWidget(self.stop_button)

        main_layout.addWidget(self.video_widget, 1)
        main_layout.addWidget(self.status_label)
        main_layout.addLayout(button_layout)

    def _build_pipeline(self) -> None:
        pipeline_description = (
            "v4l2src device=/dev/video0 "
            "! videoconvert "
            "! queue "
            "! ximagesink name=video_sink sync=true"
        )

        try:
            self.pipeline = Gst.parse_launch(pipeline_description)
        except Exception as error:
            QMessageBox.critical(
                self,
                "Pipeline error",
                f"Could not create the GStreamer pipeline:\n\n{error}",
            )
            return

        self.video_sink = self.pipeline.get_by_name("video_sink")
        self.bus = self.pipeline.get_bus()

        if self.video_sink is None:
            QMessageBox.critical(
                self,
                "GStreamer error",
                "The ximagesink element could not be created.",
            )

    def _attach_video_window(self) -> bool:
        if self.video_sink is None:
            return False

        # winId() is the native X11 window handle for this QWidget.
        window_handle = int(self.video_widget.winId())

        if window_handle == 0:
            QMessageBox.warning(
                self,
                "Window error",
                "Qt did not create a native video window.",
            )
            return False

        GstVideo.VideoOverlay.set_window_handle(
            self.video_sink,
            window_handle,
        )

        return True

    def start_pipeline(self) -> None:
        if self.pipeline is None:
            return

        if not self._attach_video_window():
            return

        result = self.pipeline.set_state(Gst.State.PLAYING)

        if result == Gst.StateChangeReturn.FAILURE:
            QMessageBox.critical(
                self,
                "GStreamer error",
                "GStreamer failed to enter the PLAYING state.",
            )
            self.status_label.setText("Status: error")
            return

        self.status_label.setText("Status: playing")

    def pause_pipeline(self) -> None:
        if self.pipeline is None:
            return

        result = self.pipeline.set_state(Gst.State.PAUSED)

        if result == Gst.StateChangeReturn.FAILURE:
            self.status_label.setText("Status: pause failed")
            return

        self.status_label.setText("Status: paused")

    def stop_pipeline(self) -> None:
        if self.pipeline is None:
            return

        self.pipeline.set_state(Gst.State.NULL)
        self.status_label.setText("Status: stopped")
        self.video_widget.update()

    def _check_bus_messages(self) -> None:
        if self.bus is None:
            return

        while True:
            message = self.bus.pop()

            if message is None:
                break

            if message.type == Gst.MessageType.ERROR:
                error, debug_info = message.parse_error()

                print(f"GStreamer error: {error}", file=sys.stderr)

                if debug_info:
                    print(
                        f"GStreamer debug information: {debug_info}",
                        file=sys.stderr,
                    )

                self.stop_pipeline()

                QMessageBox.critical(
                    self,
                    "GStreamer error",
                    str(error),
                )

            elif message.type == Gst.MessageType.EOS:
                self.stop_pipeline()
                self.status_label.setText("Status: end of stream")

            elif message.type == Gst.MessageType.WARNING:
                warning, debug_info = message.parse_warning()

                print(f"GStreamer warning: {warning}", file=sys.stderr)

                if debug_info:
                    print(
                        f"GStreamer debug information: {debug_info}",
                        file=sys.stderr,
                    )

    def closeEvent(self, event) -> None:
        self.bus_timer.stop()

        if self.pipeline is not None:
            self.pipeline.set_state(Gst.State.NULL)

        event.accept()


def main() -> int:
    app = QApplication(sys.argv)

    window = GStreamerWindow()
    window.show()

    # Start only after Qt has created and displayed the native window.
    QTimer.singleShot(0, window.start_pipeline)

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
