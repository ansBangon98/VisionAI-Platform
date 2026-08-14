from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from PySide6.QtWidgets import QFrame, QLayout, QVBoxLayout, QWidget

from ui.widgets import (
    AnalyticsSummaryWidget,
    DetectionResultsWidget,
    EventHistoryWidget,
    SessionStatsWidget,
)


@dataclass(slots=True)
class DashboardView:
    """Composition wrapper for dashboard widgets mounted into loaded UI frames."""

    analytics_summary: AnalyticsSummaryWidget
    event_history: EventHistoryWidget
    detection_results: DetectionResultsWidget
    session_stats: SessionStatsWidget

    @classmethod
    def from_loaded_ui(
        cls,
        root: QWidget,
        *,
        max_event_items: int = 10,
        max_detection_items: int = 10,
        dashboard_config: Mapping[str, object] | None = None,
    ) -> "DashboardView":
        frames = _find_dashboard_frames(root)
        config = dashboard_config if isinstance(dashboard_config, Mapping) else {}
        analytics_config = _section_config(config, "analytics_summary")
        event_config = _section_config(config, "event_history")
        detection_config = _section_config(config, "detection_results")
        session_config = _section_config(config, "session_stats")

        view = cls(
            analytics_summary=AnalyticsSummaryWidget(
                metrics=_sequence_or_none(analytics_config.get("metrics")),
                class_distribution=analytics_config.get("class_distribution"),
            ),
            event_history=EventHistoryWidget(
                max_items=_section_int(event_config, "max_items", max_event_items)
            ),
            detection_results=DetectionResultsWidget(
                max_items=_section_int(
                    detection_config,
                    "max_items",
                    max_detection_items,
                ),
                columns=_sequence_or_none(detection_config.get("columns")),
            ),
            session_stats=SessionStatsWidget(
                metrics=_sequence_or_none(session_config.get("metrics"))
            ),
        )
        _mount_widget(frames["frm_analytics_summary"], view.analytics_summary)
        _mount_widget(frames["frm_event_history"], view.event_history)
        _mount_widget(frames["frm_detection_result"], view.detection_results)
        _mount_widget(frames["frm_session_stats"], view.session_stats)
        view.analytics_summary.setVisible(_section_enabled(analytics_config))
        view.event_history.setVisible(_section_enabled(event_config))
        view.detection_results.setVisible(_section_enabled(detection_config))
        view.session_stats.setVisible(_section_enabled(session_config))
        return view

    def reset(self) -> None:
        self.analytics_summary.reset()
        self.event_history.clear()
        self.detection_results.clear()
        self.session_stats.reset()


def _find_dashboard_frames(root: QWidget) -> dict[str, QFrame]:
    frame_names = (
        "frm_analytics_summary",
        "frm_event_history",
        "frm_detection_result",
        "frm_session_stats",
    )
    frames = {name: root.findChild(QFrame, name) for name in frame_names}
    missing = [name for name, frame in frames.items() if frame is None]
    if missing:
        raise RuntimeError(f"Missing dashboard frames: {', '.join(missing)}")
    return frames  # type: ignore[return-value]


def _mount_widget(frame: QFrame, widget: QWidget) -> None:
    frame.setFrameShape(QFrame.Shape.NoFrame)
    frame.setStyleSheet(
        f"QFrame#{frame.objectName()} {{ background: transparent; border: none; }}"
    )

    layout = frame.layout()
    if layout is None:
        layout = QVBoxLayout(frame)
    else:
        _clear_layout(layout)

    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    layout.addWidget(widget)


def _clear_layout(layout: QLayout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        child = item.widget()
        if child is not None:
            child.hide()
            child.setParent(None)
            child.deleteLater()
            continue

        child_layout = item.layout()
        if child_layout is not None:
            _clear_layout(child_layout)


def _section_config(
    dashboard_config: Mapping[str, object],
    section: str,
) -> dict[str, object]:
    value = dashboard_config.get(section, {})
    return dict(value) if isinstance(value, Mapping) else {}


def _section_enabled(config: Mapping[str, object]) -> bool:
    return bool(config.get("enabled", True))


def _section_int(
    config: Mapping[str, object],
    key: str,
    default: int,
) -> int:
    try:
        return max(1, int(config.get(key, default)))
    except (TypeError, ValueError):
        return max(1, int(default))


def _sequence_or_none(value: object):
    if isinstance(value, (list, tuple)):
        return value
    return None
