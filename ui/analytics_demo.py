import os
import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication, QComboBox, QFrame, QLabel, QMainWindow, QMessageBox
from PySide6.QtCore import QEvent, QFile
from PySide6.QtUiTools import QUiLoader

from core.camera.gstreamer import CameraConfig, CameraMetrics, attach_camera_viewer


try:
    import assets.icons.icons_rc
except ImportError:
    pass


# HELPER FUNCTIONS

def resource_path(relative: str) -> str:
    """PyInstaller-aware resource path."""
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative)
    project_root = Path(__file__).resolve().parent.parent
    return str(project_root / relative)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Vision Analytics")
        self.setGeometry(100, 100, 862, 875)
        self.initUI()

    def initUI(self):
        ui_file = QFile(resource_path("ui/analytics_demo.ui"))
        if not ui_file.open(QFile.ReadOnly):
            raise RuntimeError(f"Cannot open UI file: {resource_path('ui/analytics_demo.ui')}")

        loader = QUiLoader()
        self.ui = loader.load(ui_file)
        ui_file.close()

        try:
            if self.ui is None:
                raise RuntimeError(loader.errorString())

            self.setCentralWidget(self.ui)
            self.performance_metrics()
            self.setup_right_aligned_header_widgets()
            self.setup_camera_feed_widget()
        except RuntimeError as error:
                    self.status_label.setText(str(error))
                    QMessageBox.critical(self, "UI error", str(error))

    def performance_metrics(self):
        self.lbl_FPS = self.ui.findChild(QLabel, "lbl_FPS")
        self.lbl_Latency = self.ui.findChild(QLabel, "lbl_Latency")
        self.lbl_Resolution = self.ui.findChild(QLabel, "lbl_Resolution")

        missing_widgets = [
            name
            for name, widget in {
                "lbl_FPS": self.lbl_FPS,
                "lbl_Latency": self.lbl_Latency,
                "lbl_Resolution": self.lbl_Resolution,
            }.items()
            if widget is None
        ]

        if missing_widgets:
            raise RuntimeError(f"Missing UI widgets: {', '.join(missing_widgets)}")

        self.lbl_Resolution.setMinimumWidth(64)
        self.lbl_Resolution.setMaximumWidth(80)
        self.reset_performance_metrics()

    def setup_camera_feed_widget(self):
        self.frm_videofeed = self.ui.findChild(QFrame, "frm_videofeed")

        missing_widgets = [
            name
            for name, widget in {
                "frm_videofeed": self.frm_videofeed,
            }.items()
            if widget is None
        ]

        if missing_widgets:
            raise RuntimeError(f"Missing UI widgets: {', '.join(missing_widgets)}")

        self.camera_viewer = attach_camera_viewer(
            self.frm_videofeed,
            # CameraConfig(source="usb", usb_device="/dev/video0"),
            CameraConfig(source="rtsp", rtsp_uri="rtsp://viewer:Viewer%40123@192.168.3.225:554"),
            auto_start=True,
        )
        self.camera_viewer.metrics_changed.connect(self.update_performance_metrics)
        self.camera_viewer.error_occurred.connect(
            lambda _message: self.reset_performance_metrics()
        )

    def start_usb_camera(self, usb_device: str = "/dev/video0"):
        if hasattr(self, "camera_viewer"):
            self.reset_performance_metrics()
            self.camera_viewer.start_usb(usb_device=usb_device)

    def start_rtsp_camera(self, rtsp_uri: str):
        if hasattr(self, "camera_viewer"):
            self.reset_performance_metrics()
            self.camera_viewer.start_rtsp(rtsp_uri)

    def stop_camera(self):
        if hasattr(self, "camera_viewer"):
            self.camera_viewer.stop()
        self.reset_performance_metrics()

    def update_performance_metrics(self, metrics: CameraMetrics):
        fps_text = "--" if metrics.fps <= 0 else f"{metrics.fps:.1f}"
        latency_text = (
            "--"
            if metrics.latency_ms is None
            else f"{round(metrics.latency_ms):.0f}ms"
        )

        self.lbl_FPS.setText(fps_text)
        self.lbl_Latency.setText(latency_text)
        self.lbl_Resolution.setText(f"{metrics.width}x{metrics.height}")

    def reset_performance_metrics(self):
        self.lbl_FPS.setText("--")
        self.lbl_Latency.setText("--")
        self.lbl_Resolution.setText("--")

    def setup_right_aligned_header_widgets(self):
        self.header_frame = self.ui.findChild(QFrame, "frame_3")
        self.cbo_selected_test = self.ui.findChild(QComboBox, "cbo_selected_test")
        self.lbl_dotlive_indication = self.ui.findChild(QLabel, "lbl_dotlive_indication")
        self.lbl_liveinference_status = self.ui.findChild(QLabel, "lbl_liveinference_status")

        missing_widgets = [
            name
            for name, widget in {
                "frame_3": self.header_frame,
                "cbo_selected_test": self.cbo_selected_test,
                "lbl_dotlive_indication": self.lbl_dotlive_indication,
                "lbl_liveinference_status": self.lbl_liveinference_status,
            }.items()
            if widget is None
        ]

        if missing_widgets:
            raise RuntimeError(f"Missing UI widgets: {', '.join(missing_widgets)}")

        self.live_status_container = self.lbl_liveinference_status.parentWidget()
        self._header_right_margin = 5
        self._header_widget_gap = 10
        self._cbo_selected_test_y = self.cbo_selected_test.y()
        self._live_status_container_y = self.live_status_container.y()

        self.header_frame.installEventFilter(self)
        self.position_header_right_widgets()

    def eventFilter(self, watched, event):
        if watched is self.header_frame and event.type() == QEvent.Type.Resize:
            self.position_header_right_widgets()
        return super().eventFilter(watched, event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.position_header_right_widgets()

    def closeEvent(self, event):
        self.stop_camera()
        super().closeEvent(event)

    def position_header_right_widgets(self):
        if not hasattr(self, "header_frame"):
            return

        status_x = (
            self.header_frame.width()
            - self._header_right_margin
            - self.live_status_container.width()
        )
        combo_x = status_x - self._header_widget_gap - self.cbo_selected_test.width()

        self.live_status_container.move(
            max(self._header_right_margin, status_x),
            self._live_status_container_y,
        )
        self.cbo_selected_test.move(
            max(self._header_right_margin, combo_x),
            self._cbo_selected_test_y,
        )


# ── Standalone entry point ─────────────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
