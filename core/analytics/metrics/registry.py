from __future__ import annotations

from collections.abc import Iterable, Mapping

from core.analytics.metrics.base_metric import (
    MetricContext,
    MetricDefinition,
    MetricValue,
    first_value,
    format_text,
    title_from_key,
)
from core.analytics.metrics.detection_metrics import DETECTION_METRICS
from core.analytics.metrics.performance_metrics import PERFORMANCE_METRICS
from core.analytics.metrics.tracking_metrics import TRACKING_METRICS


DEFAULT_ANALYTICS_METRICS = (
    "total_detections",
    "objects_in_frame",
    "unique_classes",
    "top_detected_class",
)
DEFAULT_SESSION_METRICS = (
    "fps",
    "inference_ms",
    "frames_processed",
    "session_duration",
)

METRICS: dict[str, MetricDefinition] = {
    **DETECTION_METRICS,
    **TRACKING_METRICS,
    **PERFORMANCE_METRICS,
}


def resolve_metric_values(
    metric_keys: Iterable[object],
    context: MetricContext,
) -> list[MetricValue]:
    values: list[MetricValue] = []
    for metric_config in metric_keys:
        key, label = metric_key_and_label(metric_config)
        definition = metric_definition(key, label_override=label)
        values.append(definition.resolve(context))
    return values


def metric_definition(
    key: str,
    *,
    label_override: str | None = None,
) -> MetricDefinition:
    normalized_key = normalize_metric_key(key)
    definition = METRICS.get(normalized_key)
    if definition is not None:
        if label_override and label_override != definition.label:
            return MetricDefinition(
                key=definition.key,
                label=label_override,
                resolver=definition.resolver,
                formatter=definition.formatter,
                default=definition.default,
            )
        return definition

    return MetricDefinition(
        key=normalized_key,
        label=label_override or title_from_key(normalized_key),
        resolver=lambda context: _fallback_metric_value(context, normalized_key),
        formatter=format_text,
        default="--",
    )


def metric_label(metric_config: object) -> str:
    key, label = metric_key_and_label(metric_config)
    return metric_definition(key, label_override=label).label


def metric_default_display(metric_config: object) -> str:
    key, label = metric_key_and_label(metric_config)
    return metric_definition(key, label_override=label).default_display()


def metric_key_and_label(metric_config: object) -> tuple[str, str | None]:
    if isinstance(metric_config, Mapping):
        key = str(metric_config.get("key") or metric_config.get("metric") or "")
        label = metric_config.get("label")
        return normalize_metric_key(key), str(label) if label else None

    return normalize_metric_key(str(metric_config)), None


def normalize_metric_key(key: str) -> str:
    return str(key).strip().lower().replace("-", "_").replace(" ", "_")


def _fallback_metric_value(context: MetricContext, key: str) -> object | None:
    value = first_value(context.summary, key)
    if value is not None:
        return value
    return first_value(context.session, key)
