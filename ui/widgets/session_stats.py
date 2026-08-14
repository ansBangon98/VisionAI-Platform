from __future__ import annotations

from collections.abc import Mapping, Sequence

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

from core.analytics.metrics import (
    DEFAULT_SESSION_METRICS,
    metric_default_display,
    metric_label,
)
from core.analytics.metrics.registry import metric_key_and_label


class SessionStatsWidget(QWidget):
    """Config-driven dashboard widget for live session and runtime counters."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        metrics: Sequence[object] | None = None,
    ):
        super().__init__(parent)
        self.setObjectName("sessionStatsWidget")

        self.metric_configs = tuple(metrics or DEFAULT_SESSION_METRICS)
        self.metric_values: dict[str, QLabel] = {}
        self.metric_defaults: dict[str, str] = {}

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setObjectName("sessionStatsScroll")
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        self.content_widget = QWidget(self.scroll_area)
        self.content_widget.setObjectName("sessionStatsContent")
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(4, 7, 4, 5)
        self.content_layout.setSpacing(0)

        self.scroll_area.setWidget(self.content_widget)
        self.main_layout.addWidget(self.scroll_area)

        for metric_config in self.metric_configs:
            key, _ = metric_key_and_label(metric_config)
            self.metric_values[key] = self._add_stat(metric_label(metric_config))
            self.metric_defaults[key] = metric_default_display(metric_config)

        self.setStyleSheet(self._style_sheet())
        self.reset()

    def _add_stat(self, label_text: str) -> QLabel:
        row = QFrame(self.content_widget)
        row.setObjectName("sessionStatsRow")
        row.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)

        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 5, 0, 5)
        layout.setSpacing(8)

        label = QLabel(label_text, row)
        label.setObjectName("sessionStatsLabel")

        value = QLabel("--", row)
        value.setObjectName("sessionStatsValue")

        layout.addWidget(label)
        layout.addStretch(1)
        layout.addWidget(value)
        self.content_layout.addWidget(row)
        return value

    def set_metric_values(self, values: Mapping[str, object]) -> None:
        for raw_key, raw_value in values.items():
            key = str(raw_key).strip().lower().replace("-", "_").replace(" ", "_")
            label = self.metric_values.get(key)
            if label is not None:
                label.setText("--" if raw_value in (None, "") else str(raw_value))

    def set_fps(self, fps: float | int | None) -> None:
        self.set_metric_values({"fps": "--" if fps is None else f"{float(fps):.1f}"})

    def set_inference_ms(self, latency: float | int | None) -> None:
        value = "--" if latency is None else f"{float(latency):.1f} ms"
        self.set_metric_values({"inference_ms": value})

    def set_uptime(self, uptime: str | None) -> None:
        display = uptime or "--"
        self.set_metric_values({"uptime": display, "session_duration": display})

    def set_total_tracked_ids(self, value: int | None) -> None:
        display = "--" if value is None else str(max(0, int(value)))
        self.set_metric_values({"total_tracked_ids": display})

    def set_peak_occupancy(self, value: int | None) -> None:
        display = "--" if value is None else str(max(0, int(value)))
        self.set_metric_values({"peak_occupancy": display})

    def update_stats(self, **stats: object) -> None:
        values: dict[str, object] = {}
        for key, value in stats.items():
            metric_key = _STAT_ALIASES.get(key, key)
            values[metric_key] = value
        self.set_metric_values(values)

    def reset(self) -> None:
        for key, label in self.metric_values.items():
            label.setText(self.metric_defaults.get(key, "--"))

    def _style_sheet(self) -> str:
        return """
        QWidget#sessionStatsWidget {
            background: transparent;
        }
        QWidget#sessionStatsContent,
        QScrollArea#sessionStatsScroll {
            background: transparent;
            border: none;
        }
        QFrame#sessionStatsRow {
            background: transparent;
            border: none;
            border-bottom: 1px solid #30353d;
        }
        QLabel#sessionStatsLabel {
            color: #8998b2;
            font-size: 11px;
        }
        QLabel#sessionStatsValue {
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
    "avg_processing_time": "inference_ms",
    "processing_ms": "inference_ms",
    "total_tracked_ids": "total_tracked_ids",
    "peak_occupancy": "peak_occupancy",
}
