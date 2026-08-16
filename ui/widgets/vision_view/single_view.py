from __future__ import annotations

from PySide6.QtWidgets import QVBoxLayout, QWidget

from ui.widgets.vision_view.video_panel import VideoPanel


class SingleViewWidget(QWidget):
    def __init__(self, *, title: str = "VIDEO", parent: QWidget | None = None):
        super().__init__(parent)
        self.panel = VideoPanel(title)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.panel)

    def set_frame(self, frame: object | None) -> None:
        self.panel.set_frame(frame)

    def clear(self) -> None:
        self.panel.clear()
