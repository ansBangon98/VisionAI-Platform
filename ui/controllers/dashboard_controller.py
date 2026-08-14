from __future__ import annotations

import time
from collections import Counter, deque
from collections.abc import Mapping, Sequence

from core.analytics.metrics import (
    DEFAULT_ANALYTICS_METRICS,
    DEFAULT_SESSION_METRICS,
    MetricContext,
    resolve_metric_values,
)
from core.analytics.metrics.base_metric import (
    detection_label,
    first_value,
    optional_float,
    optional_int,
    read_value,
)


class DashboardController:
    """Updates config-driven dashboard widgets from pipeline/runtime data."""

    def __init__(
        self,
        view,
        *,
        dashboard_config: Mapping[str, object] | None = None,
        app_config: Mapping[str, object] | None = None,
        model_profile: Mapping[str, object] | None = None,
        exit_debounce_frames: int = 3,
        processing_window: int = 60,
    ):
        self.view = view
        self.dashboard_config = _mapping(dashboard_config)
        self.app_config = _mapping(app_config)
        self.model_profile = _mapping(model_profile)
        self.exit_debounce_frames = max(1, int(exit_debounce_frames))
        self.processing_times_ms: deque[float] = deque(
            maxlen=max(1, int(processing_window))
        )
        self.analytics_metric_configs = _dashboard_sequence(
            self.dashboard_config,
            "analytics_summary",
            "metrics",
            DEFAULT_ANALYTICS_METRICS,
        )
        self.session_metric_configs = _dashboard_sequence(
            self.dashboard_config,
            "session_stats",
            "metrics",
            DEFAULT_SESSION_METRICS,
        )
        self.event_types = _dashboard_events(self.dashboard_config)
        self.reset()

    def set_context(
        self,
        *,
        app_config: Mapping[str, object] | None = None,
        model_profile: Mapping[str, object] | None = None,
    ) -> None:
        if app_config is not None:
            self.app_config = _mapping(app_config)
        if model_profile is not None:
            self.model_profile = _mapping(model_profile)
        self._refresh_dashboard()

    def reset(self) -> None:
        self.session_started_at = time.monotonic()
        self.entry_count = 0
        self.exit_count = 0
        self.peak_occupancy = 0
        self.total_detections = 0
        self.frames_processed = 0
        self.known_track_ids: set[int] = set()
        self.active_track_ids: set[int] = set()
        self.track_labels: dict[int, str] = {}
        self.missing_track_frames: dict[int, int] = {}
        self.active_class_labels: set[str] = set()
        self.seen_class_labels: set[str] = set()
        self.missing_class_frames: dict[str, int] = {}
        self.class_counts: Counter[str] = Counter()
        self.runtime_stats: dict[str, object] = {}
        self.current_detections: list[object] = []
        self.last_summary: dict[str, object] = {}
        self.processing_times_ms.clear()
        self.view.reset()
        self._refresh_dashboard()

    def update_frame_result(
        self,
        results: Sequence[object] | object | None,
        summary: Mapping[str, object] | object | None = None,
    ) -> None:
        detections = _result_detections(results)
        summary_values = _mapping(summary)
        self.current_detections = detections
        self.last_summary = summary_values

        self.frames_processed += 1
        reported_frames = _first_int(
            summary_values,
            "frames_processed",
            "frame_number",
        )
        if reported_frames is not None:
            self.frames_processed = max(self.frames_processed, reported_frames)

        self.total_detections += len(detections)
        current_class_counts = Counter(
            label for label in (detection_label(item) for item in detections) if label
        )
        self.class_counts.update(current_class_counts)

        self.view.detection_results.set_detections(detections)
        self._update_track_events(detections)
        self._update_class_events(current_class_counts)
        self._add_summary_events(summary_values)
        self._update_summary_counters(summary_values, detections)
        self._refresh_dashboard()

    def update_stats(self, stats: Mapping[str, object] | object | None) -> None:
        values = _mapping(stats)
        fps = _first_float(values, "fps")
        if fps is not None:
            self.runtime_stats["fps"] = fps

        inference_ms = _first_float(
            values,
            "inference_ms",
            "processing_ms",
            "latency_ms",
        )
        if inference_ms is not None:
            self.runtime_stats["inference_ms"] = inference_ms

        self._refresh_dashboard()

    def add_event(self, event: object) -> None:
        self.view.event_history.add_event(event)

    def clear_frame(self) -> None:
        self.current_detections = []
        self.last_summary = {
            "current_objects": 0,
            "current_people": 0,
            "current_faces": 0,
        }
        self.view.detection_results.clear()
        self._refresh_dashboard()

    def clear_runtime_stats(self) -> None:
        self.runtime_stats.pop("fps", None)
        self.runtime_stats.pop("inference_ms", None)
        self._refresh_dashboard()

    def _update_track_events(self, detections: Sequence[object]) -> None:
        current_track_labels = {
            track_id: detection_label(detection)
            for detection in detections
            for track_id in (_track_id(detection),)
            if track_id is not None
        }
        current_track_ids = set(current_track_labels)

        for track_id in sorted(current_track_ids - self.known_track_ids):
            label = current_track_labels.get(track_id, "")
            self.known_track_ids.add(track_id)
            self.track_labels[track_id] = label
            self.entry_count += 1
            self._emit_track_event("entered", track_id, label)

        for track_id in current_track_ids:
            self.active_track_ids.add(track_id)
            self.track_labels[track_id] = current_track_labels.get(track_id, "")
            self.missing_track_frames.pop(track_id, None)

        for track_id in sorted(self.active_track_ids - current_track_ids):
            missing_frames = self.missing_track_frames.get(track_id, 0) + 1
            if missing_frames < self.exit_debounce_frames:
                self.missing_track_frames[track_id] = missing_frames
                continue

            self.active_track_ids.remove(track_id)
            self.missing_track_frames.pop(track_id, None)
            label = self.track_labels.get(track_id, "")
            self.exit_count += 1
            self._emit_track_event("exited", track_id, label)

    def _update_class_events(self, current_class_counts: Counter[str]) -> None:
        wants_appeared = "object_appeared" in self.event_types
        wants_disappeared = "object_disappeared" in self.event_types
        wants_new_class = "new_class_appeared" in self.event_types
        if not (wants_appeared or wants_disappeared or wants_new_class):
            return

        current_labels = set(current_class_counts)
        for label in sorted(current_labels - self.active_class_labels):
            if wants_new_class and label not in self.seen_class_labels:
                self.view.event_history.add_event(f"New class appeared: {label}")
            elif wants_appeared:
                self.view.event_history.add_event(f"{label} appeared")
            self.seen_class_labels.add(label)

        for label in current_labels:
            self.active_class_labels.add(label)
            self.missing_class_frames.pop(label, None)

        for label in sorted(self.active_class_labels - current_labels):
            missing_frames = self.missing_class_frames.get(label, 0) + 1
            if missing_frames < self.exit_debounce_frames:
                self.missing_class_frames[label] = missing_frames
                continue

            self.active_class_labels.remove(label)
            self.missing_class_frames.pop(label, None)
            if wants_disappeared:
                self.view.event_history.add_event(f"{label} disappeared")

    def _add_summary_events(self, summary: Mapping[str, object]) -> None:
        events = summary.get("events")
        if events is None:
            return

        if isinstance(events, Sequence) and not isinstance(events, (str, bytes)):
            for event in events:
                self._add_summary_event(event)
            return

        self._add_summary_event(events)

    def _add_summary_event(self, event: object) -> None:
        if not self._event_allowed(event):
            return
        self.view.event_history.add_event(event)

    def _event_allowed(self, event: object) -> bool:
        event_type = _event_type(event)
        if not self.event_types or not event_type:
            return True
        return event_type in self.event_types

    def _emit_track_event(self, action: str, track_id: int, label: str) -> None:
        label_text = label or "Object"
        normalized_label = label_text.strip().lower()
        if action == "entered":
            if (
                normalized_label in {"person", "people"}
                and "person_entered" in self.event_types
            ):
                self.view.event_history.add_event(f"Person #{track_id} entered")
            elif "track_entered" in self.event_types:
                self.view.event_history.add_event(
                    f"{label_text} #{track_id} entered"
                )
            return

        if (
            normalized_label in {"person", "people"}
            and "person_exited" in self.event_types
        ):
            self.view.event_history.add_event(f"Person #{track_id} exited")
        elif "track_exited" in self.event_types:
            self.view.event_history.add_event(f"{label_text} #{track_id} exited")

    def _update_summary_counters(
        self,
        summary: Mapping[str, object],
        detections: Sequence[object],
    ) -> None:
        entries = _first_int(summary, "entries", "entry_count", "total_entries")
        if entries is not None:
            self.entry_count = entries

        exits = _first_int(summary, "exits", "exit_count", "total_exits")
        if exits is not None:
            self.exit_count = exits

        total_detections = _first_int(summary, "total_detections")
        if total_detections is not None:
            self.total_detections = total_detections

        inference_ms = _first_float(
            summary,
            "inference_ms",
            "processing_ms",
            "avg_processing_time",
        )
        if inference_ms is not None:
            self.processing_times_ms.append(inference_ms)
            self.runtime_stats["inference_ms"] = inference_ms

        fps = _first_float(summary, "fps")
        if fps is not None:
            self.runtime_stats["fps"] = fps

        current_occupancy = _current_occupancy(summary, detections)
        self.peak_occupancy = max(self.peak_occupancy, current_occupancy)
        peak_occupancy = _first_int(summary, "peak_occupancy")
        if peak_occupancy is not None:
            self.peak_occupancy = peak_occupancy

    def _refresh_dashboard(self) -> None:
        context = self._metric_context()
        analytics_values = resolve_metric_values(self.analytics_metric_configs, context)
        session_values = resolve_metric_values(self.session_metric_configs, context)

        self.view.analytics_summary.set_metric_values(
            {metric.key: metric.raw_value for metric in analytics_values}
        )
        self.view.analytics_summary.set_class_distribution(self.class_counts)
        self.view.session_stats.set_metric_values(
            {metric.key: metric.value for metric in session_values}
        )

    def _metric_context(self) -> MetricContext:
        return MetricContext(
            detections=self.current_detections,
            summary=self.last_summary,
            session=self._session_values(),
            model_profile=self.model_profile,
            app_config=self.app_config,
        )

    def _session_values(self) -> dict[str, object]:
        session = {
            "active_tracks": len(self.active_track_ids),
            "class_counts": dict(self.class_counts),
            "entry_count": self.entry_count,
            "exit_count": self.exit_count,
            "frames_processed": self.frames_processed,
            "peak_occupancy": self.peak_occupancy,
            "session_duration_seconds": time.monotonic() - self.session_started_at,
            "total_detections": self.total_detections,
            "total_tracked_ids": len(self.known_track_ids),
        }
        session.update(self.runtime_stats)
        return session


def _mapping(value: Mapping[str, object] | object | None) -> dict[str, object]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


def _result_detections(results: Sequence[object] | object | None) -> list[object]:
    if results is None:
        return []
    detections = getattr(results, "detections", None)
    if detections is not None:
        return list(detections)
    if isinstance(results, Mapping):
        return [results]
    if isinstance(results, Sequence) and not isinstance(results, (str, bytes)):
        return list(results)
    return [results]


def _first_int(values: Mapping[str, object], *keys: str) -> int | None:
    return optional_int(first_value(values, *keys))


def _first_float(values: Mapping[str, object], *keys: str) -> float | None:
    return optional_float(first_value(values, *keys))


def _track_id(item: object) -> int | None:
    value = read_value(item, "track_id", "id")
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _current_occupancy(
    summary: Mapping[str, object],
    detections: Sequence[object],
) -> int:
    current = _first_int(
        summary,
        "current_people",
        "people_count",
        "person_count",
        "current_objects",
        "objects_in_frame",
        "object_count",
    )
    if current is not None:
        return max(0, current)
    return len(detections)


def _dashboard_sequence(
    dashboard_config: Mapping[str, object],
    section: str,
    key: str,
    default: Sequence[object],
) -> tuple[object, ...]:
    section_config = dashboard_config.get(section, {})
    if not isinstance(section_config, Mapping):
        return tuple(default)

    value = section_config.get(key)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(value)
    return tuple(default)


def _dashboard_events(dashboard_config: Mapping[str, object]) -> set[str]:
    if "event_history" not in dashboard_config:
        return {"object_appeared", "object_disappeared"}

    section_config = dashboard_config.get("event_history", {})
    if not isinstance(section_config, Mapping):
        return {"object_appeared", "object_disappeared"}

    value = section_config.get("events")
    if value is None:
        return {"object_appeared", "object_disappeared"}
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return set()
    return {
        _normalize_event_type(event)
        for event in value
        if _normalize_event_type(event)
    }


def _event_type(event: object) -> str:
    if isinstance(event, Mapping):
        return _normalize_event_type(event.get("type") or event.get("event") or "")
    event_type = getattr(event, "type", None) or getattr(event, "event", None)
    return _normalize_event_type(event_type or "")


def _normalize_event_type(event: object) -> str:
    return str(event).strip().lower().replace("-", "_").replace(" ", "_")
