from __future__ import annotations

from collections.abc import Mapping

from core.analytics.metrics.base_metric import (
    MetricContext,
    MetricDefinition,
    detection_label,
    first_value,
    format_count,
    format_text,
    optional_int,
)


def _objects_in_frame(context: MetricContext) -> int:
    value = first_value(
        context.summary,
        "objects_in_frame",
        "current_objects",
        "object_count",
    )
    count = optional_int(value)
    return len(context.detections) if count is None else max(0, count)


def _unique_classes(context: MetricContext) -> int:
    return len({label for label in _current_labels(context) if label})


def _current_people(context: MetricContext) -> int:
    value = first_value(
        context.summary,
        "current_people",
        "people_count",
        "person_count",
    )
    count = optional_int(value)
    if count is not None:
        return max(0, count)
    return _count_current_labels(context, {"person", "people"})


def _current_faces(context: MetricContext) -> int:
    value = first_value(context.summary, "current_faces", "face_count", "faces")
    count = optional_int(value)
    if count is not None:
        return max(0, count)
    return _count_current_labels(context, {"face"})


def _top_detected_class(context: MetricContext) -> str:
    class_counts = context.session.get("class_counts", {})
    if not isinstance(class_counts, Mapping) or not class_counts:
        labels = _current_labels(context)
        return labels[0] if labels else "--"

    top_label, top_count = max(
        class_counts.items(),
        key=lambda item: int(item[1] or 0),
    )
    return str(top_label) if int(top_count or 0) > 0 else "--"


def _summary_count_metric(*keys: str) -> object:
    def resolver(context: MetricContext) -> int:
        value = first_value(context.summary, *keys)
        count = optional_int(value)
        return 0 if count is None else max(0, count)

    return resolver


def _session_count_metric(*keys: str) -> object:
    def resolver(context: MetricContext) -> int:
        value = first_value(context.session, *keys)
        count = optional_int(value)
        return 0 if count is None else max(0, count)

    return resolver


def _current_labels(context: MetricContext) -> list[str]:
    return [detection_label(detection) for detection in context.detections]


def _count_current_labels(context: MetricContext, labels: set[str]) -> int:
    return sum(1 for label in _current_labels(context) if label.lower() in labels)


DETECTION_METRICS = {
    "total_detections": MetricDefinition(
        key="total_detections",
        label="Total Detections",
        resolver=_session_count_metric("total_detections"),
        formatter=format_count,
        default=0,
    ),
    "objects_in_frame": MetricDefinition(
        key="objects_in_frame",
        label="Objects in Frame",
        resolver=_objects_in_frame,
        formatter=format_count,
        default=0,
    ),
    "unique_classes": MetricDefinition(
        key="unique_classes",
        label="Unique Classes",
        resolver=_unique_classes,
        formatter=format_count,
        default=0,
    ),
    "top_detected_class": MetricDefinition(
        key="top_detected_class",
        label="Top Detected Class",
        resolver=_top_detected_class,
        formatter=format_text,
        default="--",
    ),
    "top_class": MetricDefinition(
        key="top_class",
        label="Top Detected Class",
        resolver=_top_detected_class,
        formatter=format_text,
        default="--",
    ),
    "current_people": MetricDefinition(
        key="current_people",
        label="Current People",
        resolver=_current_people,
        formatter=format_count,
        default=0,
    ),
    "person_count": MetricDefinition(
        key="person_count",
        label="Person Count",
        resolver=_current_people,
        formatter=format_count,
        default=0,
    ),
    "current_faces": MetricDefinition(
        key="current_faces",
        label="Current Faces",
        resolver=_current_faces,
        formatter=format_count,
        default=0,
    ),
    "customers": MetricDefinition(
        key="customers",
        label="Customers",
        resolver=_summary_count_metric("customers", "customer_count"),
        formatter=format_count,
        default=0,
    ),
    "staff": MetricDefinition(
        key="staff",
        label="Staff",
        resolver=_summary_count_metric("staff", "staff_count"),
        formatter=format_count,
        default=0,
    ),
    "total_entries": MetricDefinition(
        key="total_entries",
        label="Total Entries",
        resolver=_session_count_metric("entry_count", "entries", "total_entries"),
        formatter=format_count,
        default=0,
    ),
    "total_exits": MetricDefinition(
        key="total_exits",
        label="Total Exits",
        resolver=_session_count_metric("exit_count", "exits", "total_exits"),
        formatter=format_count,
        default=0,
    ),
    "peak_occupancy": MetricDefinition(
        key="peak_occupancy",
        label="Peak Occupancy",
        resolver=_session_count_metric("peak_occupancy"),
        formatter=format_count,
        default=0,
    ),
}
