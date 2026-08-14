from __future__ import annotations

from collections.abc import Mapping

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class EventHistoryWidget(QWidget):
    """Compact newest-first event list for dashboard activity."""

    def __init__(self, max_items: int = 10, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("eventHistoryWidget")
        self.max_items = max(1, int(max_items))
        self._rows: list[QWidget] = []

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(4, 6, 4, 4)
        self.main_layout.setSpacing(0)

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setObjectName("eventHistoryScroll")
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        self.events_container = QWidget(self.scroll_area)
        self.events_container.setObjectName("eventHistoryContainer")

        self.events_layout = QVBoxLayout(self.events_container)
        self.events_layout.setContentsMargins(0, 0, 0, 0)
        self.events_layout.setSpacing(0)
        self.events_layout.addStretch(1)

        self.scroll_area.setWidget(self.events_container)
        self.main_layout.addWidget(self.scroll_area)

        self.setStyleSheet(self._style_sheet())

    def add_event(self, event: object, elapsed: str | None = None) -> None:
        message, event_elapsed = _event_parts(event, elapsed)
        if not message:
            return

        row = self._create_event_row(message=message, elapsed=event_elapsed)
        self.events_layout.insertWidget(0, row)
        self._rows.insert(0, row)

        while len(self._rows) > self.max_items:
            oldest = self._rows.pop()
            self.events_layout.removeWidget(oldest)
            oldest.deleteLater()

    def clear(self) -> None:
        for row in self._rows:
            self.events_layout.removeWidget(row)
            row.deleteLater()
        self._rows.clear()

    def clear_events(self) -> None:
        self.clear()

    def _create_event_row(self, *, message: str, elapsed: str) -> QWidget:
        row = QFrame(self.events_container)
        row.setObjectName("eventRow")
        row.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)

        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 6, 0, 6)
        layout.setSpacing(10)

        time_label = QLabel(elapsed, row)
        time_label.setObjectName("eventTime")
        time_label.setFixedWidth(42)
        time_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        message_label = QLabel(message, row)
        message_label.setObjectName("eventMessage")
        message_label.setWordWrap(True)
        message_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )

        layout.addWidget(time_label)
        layout.addWidget(message_label, 1)
        return row

    def _style_sheet(self) -> str:
        return """
        QWidget#eventHistoryWidget,
        QWidget#eventHistoryContainer,
        QScrollArea#eventHistoryScroll {
            background: transparent;
            border: none;
        }
        QFrame#eventRow {
            background: transparent;
            border: none;
            border-bottom: 1px solid #30353d;
        }
        QLabel#eventTime {
            color: #64728a;
            font-size: 11px;
        }
        QLabel#eventMessage {
            color: #98a6bd;
            font-size: 11px;
        }
        QScrollBar:vertical {
            width: 7px;
            background: transparent;
        }
        QScrollBar::handle:vertical {
            background: #3a414c;
            border-radius: 3px;
            min-height: 20px;
        }
        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical {
            height: 0px;
        }
        """


def _event_parts(event: object, elapsed: str | None) -> tuple[str, str]:
    if isinstance(event, Mapping):
        message = str(
            event.get("message")
            or event.get("text")
            or event.get("label")
            or _event_message_from_mapping(event)
            or ""
        )
        event_elapsed = str(
            event.get("elapsed")
            or event.get("time")
            or elapsed
            or "now"
        )
        return message, event_elapsed

    message_attr = getattr(event, "message", None)
    if message_attr is not None:
        event_elapsed = getattr(event, "elapsed", None) or elapsed or "now"
        return str(message_attr), str(event_elapsed)

    return str(event), elapsed or "now"


def _event_message_from_mapping(event: Mapping[str, object]) -> str:
    event_type = _normalize_event_type(event.get("type") or event.get("event") or "")
    label = str(
        event.get("class")
        or event.get("class_name")
        or event.get("object")
        or event.get("label")
        or ""
    ).strip()
    track_id = event.get("track_id") or event.get("id")
    subject = label or "Object"
    if track_id not in (None, ""):
        subject = f"{subject} #{track_id}"

    verb = {
        "object_appeared": "appeared",
        "object_disappeared": "disappeared",
        "object_entered_roi": "entered ROI",
        "object_exited_roi": "exited ROI",
        "person_entered": "entered",
        "person_exited": "exited",
        "track_entered": "entered",
        "track_exited": "exited",
        "zone_entered": "entered zone",
        "zone_exited": "exited zone",
        "confidence_threshold_triggered": "passed confidence threshold",
        "new_class_appeared": "appeared",
    }.get(event_type)
    if not verb:
        return ""
    return f"{subject} {verb}"


def _normalize_event_type(value: object) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")
