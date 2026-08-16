from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget


class VideoPanel(QWidget):
    def __init__(self, title: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self.setObjectName("vision_video_panel")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(
            "#vision_video_panel { background-color: #05070a; }"
            "QLabel#vision_panel_title {"
            " background-color: #1e222b;"
            " color: #aab2c0;"
            " padding-left: 10px;"
            " font-weight: 600;"
            "}"
            "QLabel#vision_panel_image {"
            " background-color: #05070a;"
            " color: #8b95a7;"
            "}"
        )

        self.title_label = QLabel(title)
        self.title_label.setObjectName("vision_panel_title")
        self.title_label.setFixedHeight(26)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        self.image_label = QLabel("No frame")
        self.image_label.setObjectName("vision_panel_image")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(1, 1)
        self.image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.title_label)
        layout.addWidget(self.image_label, stretch=1)

    def set_frame(self, frame: object | None) -> None:
        if frame is None:
            self.clear()
            return

        image = _frame_to_qimage(frame)
        self._pixmap = QPixmap.fromImage(image)
        self._refresh_pixmap()

    def show_message(self, message: str) -> None:
        self._pixmap = None
        self.image_label.setText(message)
        self.image_label.setPixmap(QPixmap())

    def clear(self) -> None:
        self.show_message("No frame")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_pixmap()

    def _refresh_pixmap(self) -> None:
        if self._pixmap is None or self._pixmap.isNull():
            return
        scaled = self._pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setText("")
        self.image_label.setPixmap(scaled)


def _frame_to_qimage(frame: object) -> QImage:
    if isinstance(frame, QImage):
        return frame.convertToFormat(QImage.Format.Format_RGB888).copy()

    array = np.asarray(frame)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] < 3:
        raise RuntimeError("Vision panel frames must be HxW or HxWx3 arrays.")

    rgb = np.ascontiguousarray(array[:, :, :3].astype(np.uint8, copy=False))
    height, width = rgb.shape[:2]
    bytes_per_line = width * 3
    return QImage(
        rgb.data,
        width,
        height,
        bytes_per_line,
        QImage.Format.Format_RGB888,
    ).copy()
