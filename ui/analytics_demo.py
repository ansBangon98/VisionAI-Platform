import os
import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication, QComboBox, QFrame, QLabel, QMainWindow
from PySide6.QtCore import QEvent, QFile
from PySide6.QtUiTools import QUiLoader


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

        if self.ui is None:
            raise RuntimeError(loader.errorString())

        self.setCentralWidget(self.ui)
        self.setup_right_aligned_header_widgets()

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
        self._header_right_margin = 20
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
