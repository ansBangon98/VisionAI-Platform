from __future__ import annotations

from collections.abc import Mapping, Sequence

from core.analytics.metrics.base_metric import (
    MetricContext,
    MetricDefinition,
    first_value,
    format_count,
    format_duration,
    format_fps,
    format_ms,
    format_text,
    nested_mapping,
    sequence_text,
)


def _fps(context: MetricContext) -> object:
    return first_value(context.summary, "fps") or first_value(context.session, "fps")


def _inference_ms(context: MetricContext) -> object:
    return first_value(
        context.summary,
        "inference_ms",
        "processing_ms",
        "avg_processing_time",
    ) or first_value(context.session, "inference_ms")


def _frames_processed(context: MetricContext) -> object:
    return first_value(context.summary, "frames_processed") or first_value(
        context.session,
        "frames_processed",
    )


def _session_duration(context: MetricContext) -> object:
    return first_value(context.session, "session_duration_seconds", "uptime")


def _model(context: MetricContext) -> object:
    return first_value(
        context.model_profile,
        "base_model",
        "model_name",
        "name",
    ) or first_value(
        _primary_model_config(context),
        "name",
        "model",
    )


def _backend(context: MetricContext) -> object:
    return first_value(
        context.model_profile,
        "framework",
        "backend",
    ) or first_value(
        _primary_model_config(context),
        "backend",
    )


def _device(context: MetricContext) -> object:
    return first_value(context.model_profile, "device") or first_value(
        _primary_model_config(context),
        "device",
    )


def _input_size(context: MetricContext) -> str:
    value = first_value(context.model_profile, "model_input_size", "input")
    if value not in (None, ""):
        return sequence_text(value)

    value = first_value(_primary_model_config(context), "input_size", "input_shape")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) == 4:
            return f"{value[2]} x {value[3]}"
        return " x ".join(str(part) for part in value)
    return format_text(value)


def _runtime(context: MetricContext) -> object:
    runtime_config = nested_mapping(context.app_config, "runtime")
    deepstream_config = nested_mapping(context.app_config, "deepstream")
    return (
        first_value(context.summary, "runtime")
        or first_value(context.session, "runtime")
        or first_value(runtime_config, "mode")
        or ("DeepStream" if deepstream_config else None)
    )


def _precision(context: MetricContext) -> object:
    return first_value(
        context.summary,
        "precision",
    ) or first_value(_primary_model_config(context), "precision")


def _primary_model_config(context: MetricContext) -> Mapping[str, object]:
    for key in ("primary_model", "detector"):
        value = context.app_config.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


PERFORMANCE_METRICS = {
    "fps": MetricDefinition(
        key="fps",
        label="FPS",
        resolver=_fps,
        formatter=format_fps,
        default=None,
    ),
    "inference_ms": MetricDefinition(
        key="inference_ms",
        label="Inference",
        resolver=_inference_ms,
        formatter=format_ms,
        default=None,
    ),
    "frames_processed": MetricDefinition(
        key="frames_processed",
        label="Frames Processed",
        resolver=_frames_processed,
        formatter=format_count,
        default=0,
    ),
    "session_duration": MetricDefinition(
        key="session_duration",
        label="Session Duration",
        resolver=_session_duration,
        formatter=format_duration,
        default=0,
    ),
    "uptime": MetricDefinition(
        key="uptime",
        label="Uptime",
        resolver=_session_duration,
        formatter=format_duration,
        default=0,
    ),
    "model": MetricDefinition(
        key="model",
        label="Model",
        resolver=_model,
        formatter=format_text,
        default="--",
    ),
    "backend": MetricDefinition(
        key="backend",
        label="Backend",
        resolver=_backend,
        formatter=format_text,
        default="--",
    ),
    "device": MetricDefinition(
        key="device",
        label="Device",
        resolver=_device,
        formatter=format_text,
        default="--",
    ),
    "input": MetricDefinition(
        key="input",
        label="Input",
        resolver=_input_size,
        formatter=format_text,
        default="--",
    ),
    "runtime": MetricDefinition(
        key="runtime",
        label="Runtime",
        resolver=_runtime,
        formatter=format_text,
        default="--",
    ),
    "precision": MetricDefinition(
        key="precision",
        label="Precision",
        resolver=_precision,
        formatter=format_text,
        default="--",
    ),
}
