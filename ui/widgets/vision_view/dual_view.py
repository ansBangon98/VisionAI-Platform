from __future__ import annotations

from PySide6.QtWidgets import QHBoxLayout, QWidget

from ui.widgets.vision_view.video_panel import VideoPanel


class DualViewWidget(QWidget):
    def __init__(
        self,
        *,
        left_title: str = "ORIGINAL VIDEO",
        right_title: str = "SEGMENTATION MASK",
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.left_panel = VideoPanel(left_title)
        self.right_panel = VideoPanel(right_title)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        layout.addWidget(self.left_panel, stretch=1)
        layout.addWidget(self.right_panel, stretch=1)

    def set_frames(self, left_frame: object | None, right_frame: object | None) -> None:
        self.left_panel.set_frame(left_frame)
        self.right_panel.set_frame(right_frame)

    def clear(self) -> None:
        self.left_panel.clear()
        self.right_panel.clear()
