from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MetricContext:
    detections: Sequence[object]
    summary: Mapping[str, object]
    session: Mapping[str, object]
    model_profile: Mapping[str, object]
    app_config: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class MetricValue:
    key: str
    label: str
    value: str
    raw_value: object


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    key: str
    label: str
    resolver: Callable[[MetricContext], object]
    formatter: Callable[[object], str] = lambda value: format_text(value)
    default: object = "--"

    def resolve(self, context: MetricContext) -> MetricValue:
        raw_value = self.resolver(context)
        if raw_value is None:
            raw_value = self.default
        return MetricValue(
            key=self.key,
            label=self.label,
            value=self.formatter(raw_value),
            raw_value=raw_value,
        )

    def default_display(self) -> str:
        return self.formatter(self.default)


def first_value(values: Mapping[str, object], *keys: str) -> object | None:
    for key in keys:
        value = values.get(key)
        if value is not None:
            return value
    return None


def read_value(item: object, *keys: str, default: object = None) -> object:
    if isinstance(item, Mapping):
        for key in keys:
            if key in item:
                return item[key]
        return default

    for key in keys:
        if hasattr(item, key):
            return getattr(item, key)
    return default


def optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def detection_label(detection: object) -> str:
    explicit = read_value(detection, "class_name", "class", "label", "type")
    if explicit not in (None, ""):
        return str(explicit)

    class_id = read_value(detection, "class_id")
    if class_id not in (None, ""):
        return f"class_{class_id}"
    return ""


def detection_confidence(detection: object) -> float | None:
    return optional_float(read_value(detection, "confidence", "score"))


def detection_bbox(detection: object) -> tuple[float, float, float, float] | None:
    value = read_value(detection, "bbox", "bounding_box")
    if value is None:
        return None

    if isinstance(value, Mapping):
        try:
            x = float(value.get("x", value.get("left", 0.0)) or 0.0)
            y = float(value.get("y", value.get("top", 0.0)) or 0.0)
            width = float(value.get("width", value.get("w", 0.0)) or 0.0)
            height = float(value.get("height", value.get("h", 0.0)) or 0.0)
        except (TypeError, ValueError):
            return None
        return x, y, x + max(0.0, width), y + max(0.0, height)

    if all(hasattr(value, name) for name in ("x", "y", "width", "height")):
        try:
            x = float(getattr(value, "x"))
            y = float(getattr(value, "y"))
            width = float(getattr(value, "width"))
            height = float(getattr(value, "height"))
        except (TypeError, ValueError):
            return None
        return x, y, x + max(0.0, width), y + max(0.0, height)

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        try:
            x1, y1, x2, y2 = [float(part) for part in value[:4]]
        except (TypeError, ValueError):
            return None
        return x1, y1, x2, y2

    return None


def format_count(value: object) -> str:
    number = optional_int(value)
    return "0" if number is None else f"{max(0, number):,}"


def format_float(value: object, decimals: int = 1) -> str:
    number = optional_float(value)
    return "--" if number is None else f"{number:.{decimals}f}"


def format_fps(value: object) -> str:
    return format_float(value, decimals=1)


def format_ms(value: object) -> str:
    number = optional_float(value)
    return "--" if number is None else f"{number:.1f} ms"


def format_percent(value: object) -> str:
    number = optional_float(value)
    if number is None:
        return "--"
    if 0.0 <= number <= 1.0:
        number *= 100.0
    return f"{number:.0f}%"


def format_duration(value: object) -> str:
    if isinstance(value, str):
        return value if value else "--"

    seconds = optional_float(value)
    if seconds is None:
        return "--"

    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_bbox(value: object) -> str:
    if value is None:
        return "--"
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return "--"
    try:
        x1, y1, x2, y2 = [round(float(part)) for part in value[:4]]
    except (TypeError, ValueError):
        return "--"
    return f"{x1}, {y1}, {x2}, {y2}"


def format_text(value: object) -> str:
    if value is None or value == "":
        return "--"
    return str(value)


def title_from_key(key: str) -> str:
    return str(key).replace("_", " ").strip().title()


def nested_mapping(value: object, *keys: str) -> Mapping[str, object]:
    current = value
    for key in keys:
        if not isinstance(current, Mapping):
            return {}
        current = current.get(key, {})
    return current if isinstance(current, Mapping) else {}


def sequence_text(value: object) -> str:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return " x ".join(str(part) for part in value)
    return format_text(value)
