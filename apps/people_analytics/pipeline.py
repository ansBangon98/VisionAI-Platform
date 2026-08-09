from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from core.config import load_yaml_config
from core.models.detector import (
    Detection,
    YoloPeopleDetector,
    frame_size,
    load_class_labels,
)
from core.pipelines.cpu.frame_processor import frame_result_to_legacy_objects
from core.results.frame_result import BoundingBox, Detection as FrameDetection, FrameResult
from core.secondary import SecondaryModelManager
from core.tracking.bytetrack.byte_tracker import BYTETracker


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "apps" / "people_analytics.yaml"
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "configs" / "settings.yaml"


@dataclass(frozen=True)
class TrackResult:
    track_id: int
    bbox: tuple[int, int, int, int]


def _as_string_sequence(value: object) -> tuple[str, ...]:
    if value in (None, ""):
        return ()

    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())

    if isinstance(value, Sequence):
        return tuple(str(part).strip() for part in value if str(part).strip())

    return (str(value).strip(),)


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


def _detector_backend_options(config: dict[str, Any]) -> dict[str, Any]:
    reserved_keys = {
        "backend",
        "confidence_threshold",
        "device",
        "dynamic_dim",
        "input_dtypes",
        "input_shape",
        "input_shapes",
        "input_size",
        "model",
        "model_type",
        "name",
        "nms_iou_threshold",
        "task",
        "detect_class_ids",
        "face_class_ids",
        "people_class_ids",
        "provider",
        "providers",
        "warmup_runs",
    }
    return {
        key: value
        for key, value in config.items()
        if key not in reserved_keys
    }


class ByteTrackPeopleTracker:
    """Adapter around the local ByteTrack implementation."""

    def __init__(
        self,
        track_thresh: float = 0.5,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
        frame_rate: int = 30,
    ):
        args = SimpleNamespace(
            track_thresh=float(track_thresh),
            match_thresh=float(match_thresh),
            track_buffer=int(track_buffer),
            mot20=False,
        )
        self.tracker = BYTETracker(args, frame_rate=int(frame_rate))

    def update(
        self,
        detections: Sequence[Detection],
        frame_shape: tuple[int, int],
    ) -> list[TrackResult]:
        frame_height, frame_width = frame_shape
        if detections:
            output_results = np.asarray(
                [
                    [
                        detection.bbox[0],
                        detection.bbox[1],
                        detection.bbox[2],
                        detection.bbox[3],
                        detection.score,
                    ]
                    for detection in detections
                ],
                dtype=np.float32,
            )
        else:
            output_results = np.empty((0, 5), dtype=np.float32)

        tracks = self.tracker.update(
            output_results,
            img_info=(frame_height, frame_width),
            img_size=(frame_height, frame_width),
        )
        return [
            TrackResult(
                track_id=int(track.track_id),
                bbox=_clip_bbox(track.tlbr, frame_width, frame_height),
            )
            for track in tracks
        ]


class PeopleAnalyticsPipeline:
    def __init__(
        self,
        detector: YoloPeopleDetector,
        tracker: ByteTrackPeopleTracker,
        people_class_ids: Sequence[int] = (),
        face_class_ids: Sequence[int] = (),
        class_labels: dict[int, str] | None = None,
        secondary_models: SecondaryModelManager | None = None,
        source_id: str = "default",
    ):
        self.detector = detector
        self.tracker = tracker
        self.secondary_models = secondary_models or SecondaryModelManager(())
        self.source_id = str(source_id or "default")
        self._frame_number = 0
        people_class_id_values = [int(class_id) for class_id in people_class_ids]
        self.people_class_ids = set(people_class_id_values)
        self.person_class_id = people_class_id_values[0] if people_class_id_values else 0
        self.face_class_ids = {int(class_id) for class_id in face_class_ids}
        self.class_labels = dict(class_labels or {})

    def model_profile(self) -> dict[str, object]:
        return self.detector.model_profile()

    def set_confidence_threshold(self, confidence_threshold: float) -> None:
        self.detector.set_confidence_threshold(confidence_threshold)

    def process(self, frame: object) -> list[dict[str, object]]:
        return frame_result_to_legacy_objects(self.process_frame_result(frame))

    def process_frame_result(self, frame: object) -> FrameResult:
        started_at = time.perf_counter()
        detections = self.detector.predict(frame)
        frame_height, frame_width = frame_size(frame)
        tracked_detections = [
            detection
            for detection in detections
            if detection.class_id in self.people_class_ids
        ]
        direct_detections = [
            detection
            for detection in detections
            if detection.class_id not in self.people_class_ids
        ]
        tracks = (
            self.tracker.update(tracked_detections, (frame_height, frame_width))
            if self.people_class_ids
            else []
        )

        frame_detections: list[FrameDetection] = []
        for track in tracks:
            source_detection = _track_source_detection(track.bbox, tracked_detections)
            class_id = (
                source_detection.class_id
                if source_detection is not None
                else self.person_class_id
            )
            confidence = source_detection.score if source_detection is not None else 1.0
            frame_detections.append(
                FrameDetection(
                    class_id=class_id,
                    label=self._class_label(class_id),
                    confidence=confidence,
                    bbox=_result_bbox(track.bbox),
                    track_id=track.track_id,
                )
            )

        frame_detections.extend(
            FrameDetection(
                class_id=detection.class_id,
                label=self._class_label(detection.class_id),
                confidence=detection.score,
                bbox=_result_bbox(
                    _clip_bbox(detection.bbox, frame_width, frame_height)
                ),
                track_id=None,
            )
            for detection in direct_detections
        )

        self._frame_number += 1
        return FrameResult(
            source_id=self.source_id,
            frame_number=self._frame_number,
            timestamp=time.time(),
            detections=frame_detections,
            inference_ms=(time.perf_counter() - started_at) * 1000.0,
        )

    def _class_label(self, class_id: int) -> str:
        return self.class_labels.get(int(class_id), f"class_{class_id}")


def create_pipeline_from_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> object:
    from core.pipelines.pipeline_factory import PipelineFactory

    config = load_config(config_path)
    runtime_mode, inference_backend = PipelineFactory.resolve_runtime(config)
    if runtime_mode == "deepstream":
        return PipelineFactory.create(config)

    detector_config = _runtime_detector_config(
        config,
        runtime_mode,
        inference_backend=inference_backend,
    )
    tracker_config = config.get("tracker", {})

    if not isinstance(detector_config, dict):
        raise RuntimeError(f"Invalid detector config in {config_path}.")
    if not detector_config.get("model"):
        raise RuntimeError(
            f"Missing primary_model.model or detector.model in {config_path}."
        )
    if not isinstance(tracker_config, dict):
        raise RuntimeError(f"Invalid tracker config in {config_path}.")

    people_analytics_config = _people_analytics_config(config)
    people_class_ids = _as_int_sequence(
        detector_config.get(
            "people_class_ids",
            people_analytics_config.get("people_class_ids", ()),
        )
    )
    face_class_ids = _as_int_sequence(
        detector_config.get(
            "face_class_ids",
            people_analytics_config.get("face_class_ids", ()),
        )
    )
    detect_class_ids = _as_optional_int_sequence(detector_config.get("detect_class_ids"))

    model_path = _resolve_project_path(detector_config["model"])
    detector = YoloPeopleDetector(
        model_path=model_path,
        backend_name=detector_config.get("backend", "onnxruntime"),
        device=detector_config.get("device", "auto"),
        providers=_as_string_sequence(
            detector_config.get("providers", detector_config.get("provider", ()))
        ),
        input_shapes=_configured_input_shapes(detector_config),
        input_dtypes=detector_config.get("input_dtypes", {}),
        warmup_runs=detector_config.get("warmup_runs", 0),
        dynamic_dim=detector_config.get("dynamic_dim", 1),
        backend_options=_detector_backend_options(detector_config),
        confidence_threshold=detector_config.get("confidence_threshold", 0.4),
        nms_iou_threshold=detector_config.get("nms_iou_threshold", 0.45),
        input_size=detector_config.get("input_size", (640, 640)),
        people_class_ids=people_class_ids,
        class_ids=detect_class_ids,
    )
    tracker = ByteTrackPeopleTracker(
        track_thresh=tracker_config.get("track_thresh", 0.5),
        match_thresh=tracker_config.get("match_thresh", 0.8),
        track_buffer=tracker_config.get("track_buffer", 30),
        frame_rate=tracker_config.get("frame_rate", 30),
    )
    frame_processor = PeopleAnalyticsPipeline(
        detector=detector,
        tracker=tracker,
        people_class_ids=people_class_ids,
        face_class_ids=face_class_ids,
        class_labels=load_class_labels(model_path),
        secondary_models=SecondaryModelManager.from_config(config),
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


def _runtime_detector_config(
    config: dict[str, Any],
    runtime_mode: str,
    inference_backend: str | None = None,
) -> dict[str, Any]:
    detector_config = dict(
        config.get("primary_model")
        or config.get("detector", {})
        or {}
    )
    runtime_config = config.get(runtime_mode, {})
    if not isinstance(runtime_config, dict):
        if inference_backend:
            _apply_runtime_backend(detector_config, inference_backend)
        return detector_config

    if inference_backend is None and runtime_config.get("inference_backend"):
        inference_backend = str(runtime_config["inference_backend"])

    if inference_backend:
        _apply_runtime_backend(detector_config, str(inference_backend))

    device = runtime_config.get("device")
    if device:
        detector_config["device"] = device

    for runtime_key in ("primary_model", "detector"):
        runtime_detector_config = runtime_config.get(runtime_key, {})
        if isinstance(runtime_detector_config, dict):
            detector_config.update(runtime_detector_config)

    return detector_config


def _people_analytics_config(config: dict[str, Any]) -> dict[str, Any]:
    analytics_config = config.get("analytics", {})
    if not isinstance(analytics_config, dict):
        return {}

    people_config = analytics_config.get("people", {})
    return people_config if isinstance(people_config, dict) else {}


def _apply_runtime_backend(
    detector_config: dict[str, Any],
    inference_backend: str,
) -> None:
    normalized = inference_backend.strip().lower().replace("-", "_")
    if normalized in {"onnxruntime_cuda", "ort_cuda"}:
        detector_config["backend"] = "onnxruntime"
        detector_config.setdefault(
            "providers",
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        detector_config.setdefault("device", "CUDA")
        return

    if normalized in {"onnxruntime_cpu", "ort_cpu"}:
        detector_config["backend"] = "onnxruntime"
        detector_config.setdefault("providers", ["CPUExecutionProvider"])
        detector_config.setdefault("device", "CPU")
        return

    detector_config["backend"] = inference_backend


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


def _source_id_from_config(config: dict[str, Any]) -> str:
    source_config = config.get("source", {})
    if isinstance(source_config, dict):
        for key in ("name", "selected", "id"):
            value = source_config.get(key)
            if value:
                return str(value)
    return "default"


def _result_bbox(bbox: Sequence[float]) -> BoundingBox:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return BoundingBox(
        x=x1,
        y=y1,
        width=max(0.0, x2 - x1),
        height=max(0.0, y2 - y1),
    )


def _track_source_detection(
    track_bbox: Sequence[float],
    detections: Sequence[Detection],
) -> Detection | None:
    if not detections:
        return None

    boxes = np.asarray([detection.bbox for detection in detections], dtype=np.float32)
    ious = _box_iou(np.asarray(track_bbox, dtype=np.float32), boxes)
    if ious.size == 0:
        return None

    best_index = int(np.argmax(ious))
    if float(ious[best_index]) <= 0:
        return None
    return detections[best_index]


def _box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    box_area = max(0.0, float((box[2] - box[0]) * (box[3] - box[1])))
    boxes_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0,
        boxes[:, 3] - boxes[:, 1],
    )
    union = box_area + boxes_area - intersection
    return intersection / np.maximum(union, 1e-6)


def _clip_bbox(
    bbox: Sequence[float],
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return (
        int(round(np.clip(x1, 0, frame_width))),
        int(round(np.clip(y1, 0, frame_height))),
        int(round(np.clip(x2, 0, frame_width))),
        int(round(np.clip(y2, 0, frame_height))),
    )


def _resolve_project_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if resolved.is_absolute():
        return resolved
    return PROJECT_ROOT / resolved


def _as_int_sequence(value: object) -> tuple[int, ...]:
    if value in (None, ""):
        return ()

    if isinstance(value, str):
        return tuple(
            int(part.strip()) for part in value.split(",") if part.strip()
        )

    if isinstance(value, Sequence):
        return tuple(int(part) for part in value)

    return (int(value),)


def _as_optional_int_sequence(value: object) -> tuple[int, ...] | None:
    if value in (None, ""):
        return None
    return _as_int_sequence(value)
