from __future__ import annotations

from collections.abc import Mapping, Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.analytics.metrics.base_metric import (
    detection_bbox,
    detection_confidence,
    detection_label,
    format_bbox,
    format_percent,
    read_value,
    title_from_key,
)


DEFAULT_COLUMNS = ("class", "confidence", "track_id")


class DetectionResultsWidget(QWidget):
    """Current-frame detection table with config-driven columns."""

    def __init__(
        self,
        max_items: int = 10,
        parent: QWidget | None = None,
        *,
        columns: Sequence[object] | None = None,
    ):
        super().__init__(parent)
        self.setObjectName("detectionResultsWidget")
        self.max_items = max(1, int(max_items))
        self.columns = tuple(_column_specs(columns or DEFAULT_COLUMNS))

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(4, 6, 4, 4)
        self.main_layout.setSpacing(0)

        self.table = QTableWidget(self)
        self.table.setObjectName("detectionTable")
        self.table.setColumnCount(len(self.columns))
        self.table.setHorizontalHeaderLabels(
            tuple(label.upper() for _, label in self.columns)
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(False)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.table.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        header.setHighlightSections(False)

        self.main_layout.addWidget(self.table)
        self.setStyleSheet(self._style_sheet())

    def add_detection(
        self,
        detection: object = None,
        class_name: str | None = None,
        label: str | None = None,
        confidence: float | None = None,
        *,
        track_id: object = None,
    ) -> None:
        row = detection
        if class_name is not None or label is not None or confidence is not None:
            row = {
                "track_id": detection if track_id is None else track_id,
                "class": class_name or label or "",
                "label": label or class_name or "",
                "confidence": confidence,
            }
        self._insert_row(0, row)

        while self.table.rowCount() > self.max_items:
            self.table.removeRow(self.table.rowCount() - 1)

    def set_detections(self, detections: Sequence[object] | object | None) -> None:
        self.clear()
        rows = _as_sequence(detections)
        self.table.setRowCount(min(len(rows), self.max_items))

        for row_index, row in enumerate(rows[: self.max_items]):
            self._set_row(row_index, row)

    def clear(self) -> None:
        self.table.setRowCount(0)

    def clear_detections(self) -> None:
        self.clear()

    def _insert_row(self, row_index: int, detection: object) -> None:
        self.table.insertRow(row_index)
        self._set_row(row_index, detection)

    def _set_row(self, row_index: int, detection: object) -> None:
        for column_index, (key, _) in enumerate(self.columns):
            item = QTableWidgetItem(_column_value(detection, key))
            item.setTextAlignment(
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
            )
            self.table.setItem(row_index, column_index, item)

    def _style_sheet(self) -> str:
        return """
        QWidget#detectionResultsWidget {
            background: transparent;
        }
        QTableWidget#detectionTable {
            background-color: transparent;
            border: none;
            color: #98a6bd;
            gridline-color: transparent;
            font-size: 11px;
        }
        QTableWidget#detectionTable::item {
            border-bottom: 1px solid #30353d;
            padding: 4px;
        }
        QHeaderView::section {
            background-color: transparent;
            color: #66748b;
            border: none;
            border-bottom: 1px solid #30353d;
            padding: 5px;
            font-size: 10px;
            font-weight: bold;
        }
        QTableCornerButton::section {
            background-color: transparent;
            border: none;
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


def _as_sequence(detections: Sequence[object] | object | None) -> list[object]:
    if detections is None:
        return []
    if isinstance(detections, Mapping):
        return [detections]
    if isinstance(detections, Sequence) and not isinstance(detections, (str, bytes)):
        return list(detections)
    return [detections]


def _column_specs(columns: Sequence[object]) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    for column in columns:
        if isinstance(column, Mapping):
            key = _normalize_key(column.get("key") or column.get("column") or "")
            label = str(
                column.get("label") or _COLUMN_LABELS.get(key) or title_from_key(key)
            )
        else:
            key = _normalize_key(str(column))
            label = _COLUMN_LABELS.get(key) or title_from_key(key)
        if key:
            specs.append((key, label))
    return specs or _column_specs(DEFAULT_COLUMNS)


def _column_value(detection: object, key: str) -> str:
    if key in {"class", "class_name"}:
        return detection_label(detection) or "--"
    if key == "label":
        return _display_value(read_value(detection, "label", "type", "class_name"))
    if key == "confidence":
        return format_percent(detection_confidence(detection))
    if key == "score":
        return format_percent(read_value(detection, "score", "confidence"))
    if key == "track_id":
        return _track_id_text(read_value(detection, "track_id", "id"))
    if key == "class_id":
        return _display_value(read_value(detection, "class_id"))
    if key in {"bbox", "bounding_box"}:
        return format_bbox(detection_bbox(detection))

    attribute_value = _attribute_value(detection, key)
    if attribute_value not in (None, ""):
        return _display_value(attribute_value)

    return _display_value(read_value(detection, key))


def _attribute_value(detection: object, key: str) -> object:
    attributes = read_value(detection, "attributes", default={})
    if not isinstance(attributes, Mapping):
        return None

    attribute = attributes.get(key)
    if isinstance(attribute, Mapping):
        return (
            attribute.get("label")
            or attribute.get("value")
            or attribute.get("confidence")
        )
    if hasattr(attribute, "label"):
        return getattr(attribute, "label")
    return attribute


def _track_id_text(value: object) -> str:
    if value is None or value == "":
        return "--"
    return f"#{value}"


def _display_value(value: object) -> str:
    if value is None or value == "":
        return "--"
    return str(value)


def _normalize_key(value: object) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


_COLUMN_LABELS = {
    "bbox": "Bounding Box",
    "bounding_box": "Bounding Box",
    "class": "Class",
    "class_id": "Class ID",
    "class_name": "Class",
    "confidence": "Confidence",
    "track_id": "Track ID",
}
