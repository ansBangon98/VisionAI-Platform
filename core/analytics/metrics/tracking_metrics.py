from __future__ import annotations

from core.analytics.metrics.base_metric import (
    MetricContext,
    MetricDefinition,
    first_value,
    format_count,
    optional_int,
)


def _session_count_metric(*keys: str):
    def resolver(context: MetricContext) -> int:
        value = first_value(context.session, *keys)
        count = optional_int(value)
        return 0 if count is None else max(0, count)

    return resolver


TRACKING_METRICS = {
    "entries": MetricDefinition(
        key="entries",
        label="Entries",
        resolver=_session_count_metric("entry_count", "entries"),
        formatter=format_count,
        default=0,
    ),
    "exits": MetricDefinition(
        key="exits",
        label="Exits",
        resolver=_session_count_metric("exit_count", "exits"),
        formatter=format_count,
        default=0,
    ),
    "active_tracks": MetricDefinition(
        key="active_tracks",
        label="Active Tracks",
        resolver=_session_count_metric("active_tracks"),
        formatter=format_count,
        default=0,
    ),
    "total_tracked_ids": MetricDefinition(
        key="total_tracked_ids",
        label="Tracked IDs",
        resolver=_session_count_metric("total_tracked_ids"),
        formatter=format_count,
        default=0,
    ),
}

