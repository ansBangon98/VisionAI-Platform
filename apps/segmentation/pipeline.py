from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from apps.segmentation.renderer import SegmentationRenderer
from core.config import load_yaml_config
from core.models.segmentation import BaseSegmentor, SegFormerSegmentor
from core.models.yolo_detector import frame_to_rgb_array
from core.results import SegmentationResult


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "apps" / "road_segmentation.yaml"
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "configs" / "settings.yaml"


class SegmentationPipeline:
    def __init__(
        self,
        segmentor: BaseSegmentor,
        renderer: SegmentationRenderer,
        *,
        source_id: str = "default",
    ):
        self.segmentor = segmentor
        self.renderer = renderer
        self.source_id = str(source_id or "default")
        self._frame_number = 0
        self._last_completed_at: float | None = None

    def model_profile(self) -> dict[str, object]:
        return self.segmentor.model_profile()

    def process_frame_result(self, frame: object) -> SegmentationResult:
        started_at = time.perf_counter()
        original_frame = frame_to_rgb_array(frame)
        mask = self.segmentor.predict(original_frame)
        visualization_frame = self.renderer.render(original_frame, mask)
        inference_ms = (time.perf_counter() - started_at) * 1000.0

        self._frame_number += 1
        fps = self._pipeline_fps()
        return SegmentationResult(
            source_id=self.source_id,
            frame_number=self._frame_number,
            original_frame=original_frame,
            mask=mask,
            visualization_frame=visualization_frame,
            class_labels=dict(self.segmentor.class_labels),
            inference_ms=inference_ms,
            fps=fps,
        )

    def process(self, frame: object) -> list[dict[str, object]]:
        self.process_frame_result(frame)
        return []

    def _pipeline_fps(self) -> float | None:
        completed_at = time.perf_counter()
        previous_completed_at = self._last_completed_at
        self._last_completed_at = completed_at
        if previous_completed_at is None:
            return None
        elapsed = completed_at - previous_completed_at
        if elapsed <= 0.0:
            return None
        return 1.0 / elapsed


def create_pipeline_from_config(config_path: str | Path) -> object:
    from core.pipelines.pipeline_factory import PipelineFactory

    config = load_config(config_path)
    runtime_mode, inference_backend = PipelineFactory.resolve_runtime(config)
    if runtime_mode == "deepstream":
        raise RuntimeError("Segmentation is implemented for CPU/CUDA frame pipelines.")

    segmentor_config = _runtime_segmentor_config(
        config,
        runtime_mode,
        inference_backend=inference_backend,
    )
    if not isinstance(segmentor_config, dict):
        raise RuntimeError(f"Invalid segmentation model config in {config_path}.")
    if not segmentor_config.get("model"):
        raise RuntimeError(
            f"Missing primary_model.model in segmentation config {config_path}."
        )

    model_path = _resolve_project_path(segmentor_config["model"])
    segmentor = SegFormerSegmentor(
        model_path=model_path,
        backend_name=segmentor_config.get("backend", "openvino"),
        device=segmentor_config.get("device", "auto"),
        providers=_as_string_sequence(
            segmentor_config.get("providers", segmentor_config.get("provider", ()))
        ),
        input_size=_segmentor_input_size(segmentor_config),
        input_shapes=_configured_input_shapes(segmentor_config),
        input_dtypes=segmentor_config.get("input_dtypes", {}),
        warmup_runs=segmentor_config.get("warmup_runs", 0),
        dynamic_dim=segmentor_config.get("dynamic_dim", 1),
        backend_options=_segmentor_backend_options(segmentor_config),
        class_labels=_configured_class_labels(segmentor_config),
        processor_config_path=_processor_config_path(segmentor_config),
    )
    frame_processor = SegmentationPipeline(
        segmentor=segmentor,
        renderer=_renderer_from_config(config),
        source_id=_source_id_from_config(config),
    )
    return PipelineFactory.create(config, frame_processor=frame_processor)


def load_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    settings_path: str | Path = DEFAULT_SETTINGS_PATH,
) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    settings = load_yaml_config(settings_path) if Path(settings_path).exists() else {}
    return _merge_runtime_settings(config, settings)


def _runtime_segmentor_config(
    config: dict[str, Any],
    runtime_mode: str,
    inference_backend: str | None = None,
) -> dict[str, Any]:
    model_config = dict(config.get("primary_model") or {})
    runtime_config = config.get(runtime_mode, {})
    if not isinstance(runtime_config, dict):
        if inference_backend:
            _apply_runtime_backend(model_config, inference_backend)
        return model_config

    if inference_backend is None and runtime_config.get("inference_backend"):
        inference_backend = str(runtime_config["inference_backend"])
    if inference_backend:
        _apply_runtime_backend(model_config, inference_backend)

    device = runtime_config.get("device")
    if device:
        model_config["device"] = device

    runtime_model_config = runtime_config.get("primary_model", {})
    if isinstance(runtime_model_config, dict):
        model_config.update(runtime_model_config)
    return model_config


def _apply_runtime_backend(
    model_config: dict[str, Any],
    inference_backend: str,
) -> None:
    normalized = inference_backend.strip().lower().replace("-", "_")
    if normalized in {"onnxruntime_cuda", "ort_cuda"}:
        model_config["backend"] = "onnxruntime"
        model_config.setdefault(
            "providers",
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        model_config.setdefault("device", "CUDA")
        return

    if normalized in {"onnxruntime_cpu", "ort_cpu"}:
        model_config["backend"] = "onnxruntime"
        model_config.setdefault("providers", ["CPUExecutionProvider"])
        model_config.setdefault("device", "CPU")
        return

    model_config["backend"] = inference_backend


def _renderer_from_config(config: Mapping[str, Any]) -> SegmentationRenderer:
    display_config = config.get("display", {})
    if not isinstance(display_config, Mapping):
        display_config = {}

    segmentation_config = display_config.get("segmentation", {})
    if not isinstance(segmentation_config, Mapping):
        segmentation_config = {}

    overlay_config = display_config.get("overlay", {})
    if not isinstance(overlay_config, Mapping):
        overlay_config = {}

    return SegmentationRenderer(
        mode=str(segmentation_config.get("mode", "mask")),
        overlay_enabled=bool(overlay_config.get("enabled", False)),
        overlay_alpha=float(overlay_config.get("alpha", 0.45)),
    )


def _segmentor_backend_options(config: dict[str, Any]) -> dict[str, Any]:
    reserved_keys = {
        "artifacts",
        "backend",
        "classes",
        "device",
        "dynamic_dim",
        "input",
        "input_dtypes",
        "input_shape",
        "input_shapes",
        "input_size",
        "labels",
        "model",
        "name",
        "names",
        "output",
        "preprocessing",
        "provider",
        "providers",
        "task",
        "version",
        "warmup_runs",
    }
    return {
        key: value
        for key, value in config.items()
        if key not in reserved_keys
    }


def _configured_input_shapes(config: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    if "input_shapes" in config:
        return _normalize_shape_map(config.get("input_shapes") or {})
    if "input_shape" in config:
        shape = _normalize_shape(config["input_shape"])
        return {"": shape} if shape else {}
    return {}


def _normalize_shape_map(
    input_shapes: dict[str, Sequence[int]] | Sequence[int] | None,
) -> dict[str, tuple[int, ...]]:
    if not input_shapes:
        return {}
    if isinstance(input_shapes, dict):
        return {
            str(name): _normalize_shape(shape)
            for name, shape in input_shapes.items()
        }
    shape = _normalize_shape(input_shapes)
    return {"": shape} if shape else {}


def _normalize_shape(shape: Sequence[int] | str | object) -> tuple[int, ...]:
    if isinstance(shape, str):
        parts = [
            part.strip()
            for part in shape.replace("x", ",").replace("X", ",").split(",")
            if part.strip()
        ]
        return tuple(int(part) for part in parts)
    if isinstance(shape, Sequence):
        return tuple(int(dim) for dim in shape)
    return ()


def _segmentor_input_size(config: Mapping[str, Any]) -> tuple[int, int] | None:
    input_size = config.get("input_size")
    if input_size:
        return _normalize_size_pair(input_size)

    input_config = config.get("input")
    if isinstance(input_config, Mapping):
        width = input_config.get("width")
        height = input_config.get("height")
        if width and height:
            return int(width), int(height)

    return None


def _normalize_size_pair(value: object) -> tuple[int, int]:
    if isinstance(value, str):
        parts = [
            int(part.strip())
            for part in value.replace("x", ",").replace("X", ",").split(",")
            if part.strip()
        ]
    elif isinstance(value, Sequence):
        parts = [int(part) for part in value]
    else:
        parts = [int(value)]

    if len(parts) == 1:
        return parts[0], parts[0]
    return parts[0], parts[1]


def _processor_config_path(config: Mapping[str, Any]) -> str:
    preprocessing = config.get("preprocessing", {})
    if isinstance(preprocessing, Mapping):
        value = preprocessing.get("processor_config")
        if value:
            return str(value)
    return ""


def _configured_class_labels(config: Mapping[str, Any]) -> dict[int, str]:
    for key in ("classes", "names", "labels"):
        labels = _normalize_class_labels(config.get(key))
        if labels:
            return labels
    return {}


def _normalize_class_labels(value: object) -> dict[int, str]:
    if isinstance(value, Mapping):
        labels: dict[int, str] = {}
        for raw_class_id, raw_label in value.items():
            try:
                class_id = int(raw_class_id)
            except (TypeError, ValueError):
                continue
            label = str(raw_label).strip()
            if label:
                labels[class_id] = label
        return labels

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return {
            index: str(label).strip()
            for index, label in enumerate(value)
            if str(label).strip()
        }
    return {}


def _merge_runtime_settings(
    app_config: dict[str, Any],
    settings_config: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(app_config)
    for key in ("runtime", "cpu", "cuda", "deepstream"):
        if key not in settings_config:
            continue
        if key not in merged:
            merged[key] = settings_config[key]
            continue
        if isinstance(settings_config[key], dict) and isinstance(merged[key], dict):
            merged[key] = _deep_merge_dict(settings_config[key], merged[key])
    return merged


def _deep_merge_dict(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _source_id_from_config(config: Mapping[str, Any]) -> str:
    source_config = config.get("source", {})
    if isinstance(source_config, Mapping):
        for key in ("name", "selected", "id"):
            value = source_config.get(key)
            if value:
                return str(value)
    return "default"


def _resolve_project_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if resolved.is_absolute():
        return resolved
    return PROJECT_ROOT / resolved


def _as_string_sequence(value: object) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    if isinstance(value, Sequence):
        return tuple(str(part).strip() for part in value if str(part).strip())
    return (str(value).strip(),)
