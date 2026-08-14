from __future__ import annotations

from collections.abc import Mapping, Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from core.analytics.metrics import (
    DEFAULT_ANALYTICS_METRICS,
    metric_default_display,
    metric_label,
)
from core.analytics.metrics.registry import metric_key_and_label


class AnalyticsSummaryWidget(QWidget):
    """Config-driven dashboard widget for high-level analytics counters."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        metrics: Sequence[object] | None = None,
        class_distribution: Mapping[str, object] | bool | None = None,
    ):
        super().__init__(parent)
        self.setObjectName("analyticsSummaryWidget")

        self.metric_configs = tuple(metrics or DEFAULT_ANALYTICS_METRICS)
        self.metrics: dict[str, dict[str, object]] = {}
        self.metric_values: dict[str, QLabel] = {}
        self.distribution_rows: list[QWidget] = []
        distribution_config = _distribution_config(class_distribution)
        self.class_distribution_enabled = bool(
            distribution_config.get("enabled", False)
        )
        self.max_class_distribution = max(
            1,
            int(
                distribution_config.get(
                    "max_items",
                    distribution_config.get("max_classes", 10),
                )
                or 10
            ),
        )

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setObjectName("analyticsSummaryScroll")
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        self.content_widget = QWidget(self.scroll_area)
        self.content_widget.setObjectName("analyticsSummaryContent")
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(4, 7, 4, 5)
        self.content_layout.setSpacing(0)

        self.scroll_area.setWidget(self.content_widget)
        self.main_layout.addWidget(self.scroll_area)

        self._build_metrics(self.metric_configs)

        self.distribution_header = None
        if self.class_distribution_enabled:
            self.distribution_header = QLabel("Class Distribution", self.content_widget)
            self.distribution_header.setObjectName("summarySectionLabel")
            self.content_layout.addSpacing(6)
            self.content_layout.addWidget(self.distribution_header)

        self.setStyleSheet(self._style_sheet())
        self.reset()

    def _build_metrics(self, metrics_config: Sequence[object]) -> None:
        for metric_config in metrics_config:
            spec = _metric_spec(metric_config)
            if spec["type"] == "progress":
                value_label, progress = self._create_progress_metric(
                    title=str(spec["label"]),
                    color=str(spec["color"]),
                )
                self.metrics[str(spec["key"])] = {
                    **spec,
                    "value_label": value_label,
                    "progress": progress,
                }
                self.metric_values[str(spec["key"])] = value_label
                continue

            value_label = self._create_text_metric(str(spec["label"]))
            self.metrics[str(spec["key"])] = {
                **spec,
                "value_label": value_label,
            }
            self.metric_values[str(spec["key"])] = value_label

    def _create_progress_metric(
        self,
        *,
        title: str,
        color: str,
    ) -> tuple[QLabel, QProgressBar]:
        row = QFrame(self.content_widget)
        row.setObjectName("summaryProgressMetric")
        row.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)

        layout = QVBoxLayout(row)
        layout.setContentsMargins(0, 4, 0, 5)
        layout.setSpacing(2)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)

        label = QLabel(title, row)
        label.setObjectName("summaryMetricLabel")

        value = QLabel("--", row)
        value.setObjectName("summaryMetricValue")

        header.addWidget(label)
        header.addStretch(1)
        header.addWidget(value)

        progress = QProgressBar(row)
        progress.setRange(0, 100)
        progress.setValue(0)
        progress.setFixedHeight(5)
        progress.setTextVisible(False)
        progress.setStyleSheet(
            f"""
            QProgressBar {{
                background-color: #2b3038;
                border: none;
                border-radius: 2px;
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 2px;
            }}
            """
        )

        layout.addLayout(header)
        layout.addWidget(progress)
        self.content_layout.addWidget(row)
        return value, progress

    def _create_text_metric(self, label_text: str) -> QLabel:
        row = QFrame(self.content_widget)
        row.setObjectName("summaryTextMetric")
        row.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)

        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 5, 0, 5)
        layout.setSpacing(8)

        label = QLabel(label_text, row)
        label.setObjectName("summaryMetricLabel")

        value = QLabel("--", row)
        value.setObjectName("summaryMetricValue")

        layout.addWidget(label)
        layout.addStretch(1)
        layout.addWidget(value)
        self.content_layout.addWidget(row)
        return value

    def set_metric_values(self, values: Mapping[str, object]) -> None:
        self.update_metrics(values)

    def update_metrics(self, values: Mapping[str, object]) -> None:
        for raw_key, raw_value in values.items():
            key = str(raw_key).strip().lower().replace("-", "_").replace(" ", "_")
            metric = self.metrics.get(key)
            if metric is None:
                continue

            value_label = metric.get("value_label")
            if isinstance(value_label, QLabel):
                value_label.setText(
                    _format_value(raw_value, str(metric.get("format") or ""))
                )

            if metric.get("type") != "progress":
                continue

            progress = metric.get("progress")
            if isinstance(progress, QProgressBar):
                progress.setValue(_progress_value(raw_value, metric))

    def set_class_distribution(self, class_counts: Mapping[str, int]) -> None:
        if not self.class_distribution_enabled:
            return

        self._clear_distribution_rows()
        ranked_counts = sorted(
            (
                (str(label), int(count))
                for label, count in class_counts.items()
                if str(label).strip() and int(count) > 0
            ),
            key=lambda item: (-item[1], item[0].lower()),
        )

        for label, count in ranked_counts[: self.max_class_distribution]:
            row = self._create_distribution_row(label, count)
            self.content_layout.addWidget(row)
            self.distribution_rows.append(row)

    def update_stats(self, **stats: object) -> None:
        values: dict[str, object] = {}
        for key, value in stats.items():
            values[key] = value
            alias = _STAT_ALIASES.get(key)
            if alias is not None:
                values[alias] = value
        self.set_metric_values(values)

    def set_people_count(self, value: int) -> None:
        count = _non_negative_int(value)
        self.set_metric_values({"current_people": count, "person_count": count})

    def set_face_count(self, value: int) -> None:
        count = _non_negative_int(value)
        self.set_metric_values({"current_faces": count, "face_count": count})

    def set_entries(self, value: int) -> None:
        count = _non_negative_int(value)
        self.set_metric_values({"entries": count, "total_entries": count})

    def set_exits(self, value: int) -> None:
        count = _non_negative_int(value)
        self.set_metric_values({"exits": count, "total_exits": count})

    def set_avg_dwell_time(self, value: float | int | str | None) -> None:
        self.set_metric_values({"avg_dwell_time": _duration_text(value)})

    def reset(self) -> None:
        for metric in self.metrics.values():
            default = metric.get("default")
            value_label = metric.get("value_label")
            if isinstance(value_label, QLabel):
                value_label.setText(
                    _format_value(default, str(metric.get("format") or ""))
                )

            progress = metric.get("progress")
            if isinstance(progress, QProgressBar):
                progress.setValue(_progress_value(default, metric))
        self._clear_distribution_rows()

    def _create_distribution_row(self, label_text: str, count: int) -> QWidget:
        row = QFrame(self.content_widget)
        row.setObjectName("summaryDistributionRow")
        row.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)

        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 3, 0, 3)
        layout.setSpacing(8)

        label = QLabel(label_text, row)
        label.setObjectName("summaryMetricLabel")

        value = QLabel(f"{count:,}", row)
        value.setObjectName("summaryMetricValue")

        layout.addWidget(label)
        layout.addStretch(1)
        layout.addWidget(value)
        return row

    def _clear_distribution_rows(self) -> None:
        for row in self.distribution_rows:
            self.content_layout.removeWidget(row)
            row.hide()
            row.setParent(None)
            row.deleteLater()
        self.distribution_rows.clear()

    def _style_sheet(self) -> str:
        return """
        QWidget#analyticsSummaryWidget {
            background: transparent;
        }
        QWidget#analyticsSummaryContent,
        QScrollArea#analyticsSummaryScroll {
            background: transparent;
            border: none;
        }
        QFrame#summaryProgressMetric,
        QFrame#summaryTextMetric,
        QFrame#summaryDistributionRow {
            background: transparent;
            border: none;
        }
        QLabel#summarySectionLabel {
            color: #c4cce0;
            font-size: 10px;
            font-weight: bold;
        }
        QLabel#summaryMetricLabel {
            color: #8998b2;
            font-size: 11px;
        }
        QLabel#summaryMetricValue {
            color: #eef2fa;
            font-size: 12px;
            font-weight: bold;
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


_STAT_ALIASES = {
    "person_count": "current_people",
    "people_count": "current_people",
    "face_count": "current_faces",
    "entries": "entries",
    "exits": "exits",
    "avg_dwell_time": "avg_dwell_time",
}


def _distribution_config(
    value: Mapping[str, object] | bool | None,
) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, bool):
        return {"enabled": value}
    return {"enabled": False}


def _metric_spec(metric_config: object) -> dict[str, object]:
    key, label = metric_key_and_label(metric_config)
    if isinstance(metric_config, Mapping):
        metric_type = str(metric_config.get("type", "text")).strip().lower()
        return {
            "key": key,
            "label": str(metric_config.get("label") or label or metric_label(metric_config)),
            "type": metric_type if metric_type in {"progress", "text"} else "text",
            "min": _optional_float(metric_config.get("min"), default=0.0),
            "max": _optional_float(metric_config.get("max"), default=100.0),
            "scale": _optional_float(metric_config.get("scale"), default=1.0),
            "color": str(metric_config.get("color") or "#6697f5"),
            "format": str(metric_config.get("format") or ""),
            "default": metric_config.get(
                "default",
                0 if metric_type == "progress" else metric_default_display(metric_config),
            ),
        }

    return {
        "key": key,
        "label": label or metric_label(metric_config),
        "type": "text",
        "min": 0.0,
        "max": 100.0,
        "scale": 1.0,
        "color": "#6697f5",
        "format": "",
        "default": metric_default_display(metric_config),
    }


def _progress_value(value: object, metric: Mapping[str, object]) -> int:
    number = _optional_float(value)
    minimum = _optional_float(metric.get("min"), default=0.0) or 0.0
    maximum = _optional_float(metric.get("max"), default=100.0) or 100.0
    scale = _optional_float(metric.get("scale"), default=1.0) or 1.0
    if number is None or maximum <= minimum:
        return 0

    scaled_value = number * scale
    percent = ((scaled_value - minimum) / (maximum - minimum)) * 100.0
    return max(0, min(100, round(percent)))


def _format_value(value: object, format_type: str) -> str:
    if value is None or value == "":
        return "--"

    normalized_format = format_type.strip().lower().replace("-", "_")
    if not normalized_format:
        return _display_value(value)

    number = _optional_float(value)
    if normalized_format == "percent":
        if number is None:
            return _display_value(value)
        if 0.0 <= number <= 1.0:
            number *= 100.0
        return f"{number:.1f}%"

    if normalized_format in {"second", "seconds"}:
        if number is None:
            return _display_value(value)
        return f"{number:.1f}s"

    if normalized_format in {"millisecond", "milliseconds", "ms"}:
        if number is None:
            return _display_value(value)
        return f"{number:.1f} ms"

    if normalized_format == "fps":
        if number is None:
            return _display_value(value)
        return f"{number:.1f} FPS"

    if normalized_format in {"int", "integer", "count"}:
        try:
            return f"{int(float(str(value).replace(',', ''))):,}"
        except (TypeError, ValueError):
            return _display_value(value)

    return _display_value(value)


def _optional_float(value: object, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return default


def _display_value(value: object) -> str:
    if value is None or value == "":
        return "--"
    return str(value)


def _non_negative_int(value: object) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _duration_text(value: float | int | str | None) -> str:
    if value is None:
        return "--"
    if isinstance(value, str):
        return value if value else "--"

    try:
        seconds = max(0.0, float(value))
    except (TypeError, ValueError):
        return "--"

    if seconds < 60:
        return f"{seconds:.0f}s"

    minutes, remaining_seconds = divmod(round(seconds), 60)
    return f"{minutes}m {remaining_seconds:02d}s"
