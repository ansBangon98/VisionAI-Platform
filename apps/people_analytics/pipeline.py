from __future__ import annotations

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from core.backends.base_backend import BackendConfig, InferenceBackend
from core.config import load_yaml_config
from core.backends.onnxruntime_backend import OnnxRuntimeBackend
from core.backends.openvino_backend import OpenVinoBackend
from core.backends.pytorch_backend import PyTorchBackend
from core.backends.tensorrt_backend import TensorRtBackend
from core.loaders.base_loader import LoaderConfig
from core.loaders.onnx_loader import OnnxModelLoader
from core.loaders.openvino_loader import OpenVinoModelLoader
from core.loaders.pytorch_loader import PyTorchModelLoader
from core.loaders.tensorrt_loader import TensorRtModelLoader
from core.pipelines.cpu.frame_processor import frame_result_to_legacy_objects
from core.results.frame_result import BoundingBox, Detection as FrameDetection, FrameResult
from core.tracking.bytetrack.byte_tracker import BYTETracker

try:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QImage, QPainter
except ImportError:  # pragma: no cover - PySide is present in the app runtime.
    Qt = None
    QColor = None
    QImage = None
    QPainter = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "apps" / "people_analytics.yaml"
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "configs" / "settings.yaml"


@dataclass(frozen=True)
class DetectorBackendSpec:
    loader_cls: type
    backend_cls: type[InferenceBackend]
    expected_suffixes: tuple[str, ...] = ()


DETECTOR_BACKENDS: dict[str, DetectorBackendSpec] = {
    "onnx": DetectorBackendSpec(OnnxModelLoader, OnnxRuntimeBackend, (".onnx",)),
    "onnxruntime": DetectorBackendSpec(OnnxModelLoader, OnnxRuntimeBackend, (".onnx",)),
    "ort": DetectorBackendSpec(OnnxModelLoader, OnnxRuntimeBackend, (".onnx",)),
    "openvino": DetectorBackendSpec(OpenVinoModelLoader, OpenVinoBackend),
    "ov": DetectorBackendSpec(OpenVinoModelLoader, OpenVinoBackend),
    "pytorch": DetectorBackendSpec(PyTorchModelLoader, PyTorchBackend),
    "torch": DetectorBackendSpec(PyTorchModelLoader, PyTorchBackend),
    "torchscript": DetectorBackendSpec(PyTorchModelLoader, PyTorchBackend),
    "tensorrt": DetectorBackendSpec(
        TensorRtModelLoader,
        TensorRtBackend,
        (".engine", ".plan", ".trt"),
    ),
    "trt": DetectorBackendSpec(
        TensorRtModelLoader,
        TensorRtBackend,
        (".engine", ".plan", ".trt"),
    ),
}


@dataclass(frozen=True)
class Detection:
    bbox: tuple[float, float, float, float]
    score: float
    class_id: int


@dataclass(frozen=True)
class TrackResult:
    track_id: int
    bbox: tuple[int, int, int, int]


class YoloPeopleDetector:
    """Backend-flexible detector for Ultralytics YOLO people models."""

    def __init__(
        self,
        model_path: str | Path,
        backend_name: str = "onnxruntime",
        device: str = "auto",
        providers: Sequence[str] = (),
        input_shapes: dict[str, Sequence[int]] | None = None,
        input_dtypes: dict[str, str] | None = None,
        warmup_runs: int = 0,
        dynamic_dim: int = 1,
        backend_options: dict[str, Any] | None = None,
        confidence_threshold: float = 0.4,
        nms_iou_threshold: float = 0.45,
        input_size: Sequence[int] = (640, 640),
        people_class_ids: Sequence[int] = (0,),
        class_ids: Sequence[int] | None = None,
    ):
        self.model_path = Path(model_path)
        self.confidence_threshold = float(confidence_threshold)
        self.nms_iou_threshold = float(nms_iou_threshold)
        self.input_width = int(input_size[0])
        self.input_height = int(input_size[1])
        detection_class_ids = people_class_ids if class_ids is None else class_ids
        self.class_ids = {int(class_id) for class_id in detection_class_ids}

        self.backend_name = _normalize_backend_name(backend_name)
        try:
            self.backend = create_inference_backend(
                model_path=self.model_path,
                backend_name=self.backend_name,
                device=device,
                providers=providers,
                input_shapes=input_shapes or {},
                input_dtypes=input_dtypes or {},
                warmup_runs=warmup_runs,
                dynamic_dim=dynamic_dim,
                backend_options=backend_options or {},
            )
            self.backend.initialize()
        except RuntimeError as error:
            raise RuntimeError(
                "Failed to initialize people detector "
                f"backend='{self.backend_name}', model='{self.model_path}': {error}"
            ) from error
        except Exception as error:
            raise RuntimeError(
                "Failed to initialize people detector "
                f"backend='{self.backend_name}', model='{self.model_path}': {error}"
            ) from error
        if not self.backend.input_specs:
            raise RuntimeError(
                "The people detector model does not expose any inputs: "
                f"{self.model_path}"
            )
        self.input_name = self.backend.input_specs[0].name

    def model_profile(self) -> dict[str, object]:
        sidecar_metadata = _load_model_sidecar_metadata(self.model_path)
        return {
            "base_model": _profile_base_model(self.model_path, sidecar_metadata),
            "framework": _profile_framework_name(self.backend_name, self.backend),
            "device": _profile_device_name(self.backend),
            "model_input_size": f"{self.input_width}x{self.input_height}",
            "confidence_threshold": self.confidence_threshold,
            "model_path": str(self.model_path),
        }

    def set_confidence_threshold(self, confidence_threshold: float) -> None:
        self.confidence_threshold = max(0.0, min(1.0, float(confidence_threshold)))

    def predict(self, frame: object) -> list[Detection]:
        try:
            frame_height, frame_width = frame_size(frame)
            input_tensor, scale, pad_x, pad_y = self._preprocess(frame)
            outputs = self.backend.infer({self.input_name: input_tensor})
        except RuntimeError as error:
            raise RuntimeError(
                "People detector inference failed "
                f"backend='{self.backend_name}', model='{self.model_path}': {error}"
            ) from error
        except Exception as error:
            raise RuntimeError(
                "People detector inference failed "
                f"backend='{self.backend_name}', model='{self.model_path}': {error}"
            ) from error

        if not outputs:
            raise RuntimeError(
                "People detector inference returned no outputs "
                f"backend='{self.backend_name}', model='{self.model_path}'."
            )

        output = next(iter(outputs.values()))
        try:
            return self._postprocess(
                output,
                scale,
                pad_x,
                pad_y,
                frame_width,
                frame_height,
            )
        except RuntimeError as error:
            raise RuntimeError(
                "People detector postprocess failed "
                f"backend='{self.backend_name}', model='{self.model_path}': {error}"
            ) from error
        except Exception as error:
            raise RuntimeError(
                "People detector postprocess failed "
                f"backend='{self.backend_name}', model='{self.model_path}': {error}"
            ) from error

    def _preprocess(self, frame: object) -> tuple[np.ndarray, float, int, int]:
        frame_height, frame_width = frame_size(frame)
        scale = min(self.input_width / frame_width, self.input_height / frame_height)
        resized_width = max(1, int(round(frame_width * scale)))
        resized_height = max(1, int(round(frame_height * scale)))
        pad_x = (self.input_width - resized_width) // 2
        pad_y = (self.input_height - resized_height) // 2

        if QImage is not None and isinstance(frame, QImage):
            image = _letterbox_qimage(
                frame,
                self.input_width,
                self.input_height,
                resized_width,
                resized_height,
                pad_x,
                pad_y,
            )
        else:
            image = _letterbox_array(
                frame_to_rgb_array(frame),
                self.input_width,
                self.input_height,
                resized_width,
                resized_height,
                pad_x,
                pad_y,
            )

        tensor = image.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(tensor), scale, pad_x, pad_y

    def _postprocess(
        self,
        output: object,
        scale: float,
        pad_x: int,
        pad_y: int,
        frame_width: int,
        frame_height: int,
    ) -> list[Detection]:
        predictions = np.asarray(output)
        if predictions.size == 0:
            raise RuntimeError("Detector output tensor is empty.")

        if predictions.ndim == 3:
            if predictions.shape[0] != 1:
                raise RuntimeError(
                    "Detector output tensor has unsupported batch size "
                    f"{predictions.shape[0]}; expected 1. Shape: {predictions.shape}."
                )
            predictions = predictions[0]
        if predictions.ndim != 2:
            raise RuntimeError(
                "Detector output tensor has unsupported rank "
                f"{predictions.ndim}; expected 2 or 3. Shape: {predictions.shape}."
            )

        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T
        if predictions.shape[1] < 5:
            raise RuntimeError(
                "Detector output tensor has too few columns. Expected at least "
                f"5 values per prediction (x, y, w, h, score/classes); "
                f"got shape {predictions.shape}."
            )

        boxes_xywh = predictions[:, :4]
        if predictions.shape[1] == 5:
            scores = predictions[:, 4]
            class_ids = np.zeros_like(scores, dtype=np.int64)
        else:
            class_scores = predictions[:, 4:]
            scores = class_scores.max(axis=1)
            class_ids = class_scores.argmax(axis=1)

        keep = scores >= self.confidence_threshold
        if self.class_ids:
            keep = keep & np.isin(class_ids, list(self.class_ids))

        if not np.any(keep):
            return []

        boxes_xywh = boxes_xywh[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

        boxes = _xywh_to_xyxy(boxes_xywh)
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, frame_width)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, frame_height)

        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        if not np.any(valid):
            return []

        boxes = boxes[valid]
        scores = scores[valid]
        class_ids = class_ids[valid]

        selected = _nms_by_class(boxes, scores, class_ids, self.nms_iou_threshold)
        return [
            Detection(
                bbox=tuple(float(value) for value in boxes[index]),
                score=float(scores[index]),
                class_id=int(class_ids[index]),
            )
            for index in selected
        ]


def create_inference_backend(
    model_path: str | Path,
    backend_name: str,
    device: str = "auto",
    providers: Sequence[str] = (),
    input_shapes: dict[str, Sequence[int]] | None = None,
    input_dtypes: dict[str, str] | None = None,
    warmup_runs: int = 0,
    dynamic_dim: int = 1,
    backend_options: dict[str, Any] | None = None,
) -> InferenceBackend:
    normalized_backend = _normalize_backend_name(backend_name)
    spec = DETECTOR_BACKENDS[normalized_backend]
    resolved_model_path = Path(model_path)

    _validate_backend_model_path(resolved_model_path, normalized_backend, spec)

    normalized_input_shapes = _normalize_shape_map(input_shapes or {})
    options = dict(backend_options or {})
    loader = spec.loader_cls(
        LoaderConfig(
            model_path=resolved_model_path,
            device=device,
            providers=tuple(str(provider) for provider in providers),
            input_shapes=normalized_input_shapes,
            extras=options,
        )
    )
    backend_config = BackendConfig(
        input_shapes=normalized_input_shapes,
        input_dtypes=dict(input_dtypes or {}),
        dynamic_dim=int(dynamic_dim),
        warmup_runs=int(warmup_runs),
        extras=options,
    )
    return spec.backend_cls(loader, backend_config)


def available_detector_backends() -> tuple[str, ...]:
    return tuple(
        backend
        for backend in (
            "onnxruntime",
            "openvino",
            "pytorch",
            "tensorrt",
        )
        if backend in DETECTOR_BACKENDS
    )


def _normalize_backend_name(backend_name: str) -> str:
    normalized = str(backend_name or "onnxruntime").strip().lower()
    normalized = normalized.replace("-", "").replace("_", "").replace(" ", "")
    aliases = {
        "onnxruntime": "onnxruntime",
        "onnx": "onnxruntime",
        "ort": "onnxruntime",
        "openvino": "openvino",
        "ov": "openvino",
        "pytorch": "pytorch",
        "torch": "pytorch",
        "torchscript": "pytorch",
        "tensorrt": "tensorrt",
        "trt": "tensorrt",
    }

    try:
        return aliases[normalized]
    except KeyError as error:
        supported = ", ".join(available_detector_backends())
        raise RuntimeError(
            f"Unsupported detector backend '{backend_name}'. "
            f"Supported backends: {supported}."
        ) from error


def _validate_backend_model_path(
    model_path: Path,
    backend_name: str,
    spec: DetectorBackendSpec,
) -> None:
    if not spec.expected_suffixes:
        return

    suffix = model_path.suffix.lower()
    if suffix in spec.expected_suffixes:
        return

    expected = ", ".join(spec.expected_suffixes)
    if backend_name == "tensorrt" and suffix == ".onnx":
        raise RuntimeError(
            "TensorRT backend expects a serialized TensorRT engine "
            f"({expected}), not an ONNX file. Build/export a TensorRT engine "
            "first, then update detector.model to that engine path."
        )

    raise RuntimeError(
        f"Detector backend '{backend_name}' expects model suffix {expected}; "
        f"got '{model_path.name}'."
    )


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
        "nms_iou_threshold",
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
        people_class_ids: Sequence[int] = (0,),
        face_class_ids: Sequence[int] = (1,),
        class_labels: dict[int, str] | None = None,
        source_id: str = "default",
    ):
        self.detector = detector
        self.tracker = tracker
        self.source_id = str(source_id or "default")
        self._frame_number = 0
        people_class_id_values = [int(class_id) for class_id in people_class_ids]
        self.people_class_ids = set(people_class_id_values)
        self.person_class_id = people_class_id_values[0] if people_class_id_values else 0
        self.face_class_ids = {int(class_id) for class_id in face_class_ids}
        self.class_labels = class_labels or {0: "person", 1: "face"}

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
        people_detections = [
            detection
            for detection in detections
            if detection.class_id in self.people_class_ids
        ]
        face_detections = [
            detection
            for detection in detections
            if detection.class_id in self.face_class_ids
        ]
        tracks = self.tracker.update(people_detections, (frame_height, frame_width))

        frame_detections = [
            FrameDetection(
                class_id=self.person_class_id,
                label=self._class_label(self.person_class_id),
                confidence=_track_confidence(track.bbox, people_detections),
                bbox=_result_bbox(track.bbox),
                track_id=track.track_id,
            )
            for track in tracks
        ]
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
            for detection in face_detections
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
        raise RuntimeError(f"Missing detector.model in {config_path}.")
    if not isinstance(tracker_config, dict):
        raise RuntimeError(f"Invalid tracker config in {config_path}.")

    people_class_ids = _as_int_sequence(detector_config.get("people_class_ids", (0,)))
    face_class_ids = _as_int_sequence(detector_config.get("face_class_ids", (1,)))
    detect_class_ids = _as_optional_int_sequence(detector_config.get("detect_class_ids"))
    if detect_class_ids is None:
        detect_class_ids = tuple(dict.fromkeys((*people_class_ids, *face_class_ids)))

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


def _load_model_sidecar_metadata(model_path: Path) -> dict[str, Any]:
    for metadata_path in (
        model_path.with_name("metadata.yaml"),
        model_path.with_name("metadata.yml"),
        model_path.with_suffix(".yaml"),
        model_path.with_suffix(".yml"),
    ):
        if not metadata_path.exists():
            continue
        try:
            return load_yaml_config(metadata_path)
        except RuntimeError:
            return {}
    return {}


def _profile_base_model(model_path: Path, metadata: dict[str, Any]) -> str:
    for key in ("base_model", "model_name", "name", "architecture"):
        value = metadata.get(key)
        if value:
            return _prettify_model_name(str(value))

    description = str(metadata.get("description") or "")
    model_name = _extract_known_model_name(description)
    if model_name:
        return model_name

    for candidate in (model_path.stem, model_path.parent.name):
        model_name = _extract_known_model_name(candidate)
        if model_name:
            return model_name

    stem = model_path.stem
    if stem.lower() == "best":
        stem = model_path.parent.name
    return _prettify_model_name(stem)


def _extract_known_model_name(value: str) -> str:
    match = re.search(
        r"\b(yolo(?:v)?\d+[a-z0-9]*|mobilenetv\d+(?:[-_][a-z0-9]+)?)\b",
        value,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    return _prettify_model_name(match.group(1))


def _prettify_model_name(value: str) -> str:
    text = str(value).strip().replace("_", "-")
    if not text:
        return "--"

    if text.lower().startswith("yolo"):
        return "YOLO" + text[4:]
    if text.lower().startswith("mobilenet"):
        return "MobileNet" + text[9:]
    return text


def _profile_framework_name(backend_name: str, backend: InferenceBackend) -> str:
    loader = getattr(backend, "loader", None)
    runtime_name = getattr(loader, "runtime_name", "")
    if runtime_name:
        return str(runtime_name)

    return {
        "onnxruntime": "ONNX Runtime",
        "openvino": "OpenVINO",
        "pytorch": "PyTorch",
        "tensorrt": "TensorRT",
    }.get(str(backend_name), str(backend_name))


def _profile_device_name(backend: InferenceBackend) -> str:
    loader = getattr(backend, "loader", None)
    metadata = getattr(loader, "metadata", {}) or {}
    device = metadata.get("device")
    if device:
        return str(device)

    providers = getattr(loader, "providers", None) or []
    if providers:
        return ", ".join(str(provider) for provider in providers)

    config = getattr(loader, "config", None)
    configured_device = getattr(config, "device", "") if config is not None else ""
    return str(configured_device or "--")


def _runtime_detector_config(
    config: dict[str, Any],
    runtime_mode: str,
    inference_backend: str | None = None,
) -> dict[str, Any]:
    detector_config = dict(config.get("detector", {}) or {})
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

    runtime_detector_config = runtime_config.get("detector", {})
    if isinstance(runtime_detector_config, dict):
        detector_config.update(runtime_detector_config)

    return detector_config


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


def _track_confidence(
    track_bbox: Sequence[float],
    detections: Sequence[Detection],
) -> float:
    if not detections:
        return 1.0

    boxes = np.asarray([detection.bbox for detection in detections], dtype=np.float32)
    ious = _box_iou(np.asarray(track_bbox, dtype=np.float32), boxes)
    if ious.size == 0:
        return 1.0

    best_index = int(np.argmax(ious))
    if float(ious[best_index]) <= 0:
        return 1.0
    return float(detections[best_index].score)


def frame_size(frame: object) -> tuple[int, int]:
    if QImage is not None and isinstance(frame, QImage):
        return frame.height(), frame.width()

    array = np.asarray(frame)
    if array.ndim < 2:
        raise RuntimeError("Frame must be a QImage or an image-like NumPy array.")
    return int(array.shape[0]), int(array.shape[1])


def frame_to_rgb_array(frame: object) -> np.ndarray:
    if QImage is not None and isinstance(frame, QImage):
        return _qimage_to_rgb_array(frame)

    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] < 3:
        raise RuntimeError("Frame array must have shape HxWx3.")
    return np.ascontiguousarray(array[:, :, :3])


def _qimage_to_rgb_array(image: QImage) -> np.ndarray:
    rgb_image = image.convertToFormat(QImage.Format.Format_RGB888)
    width = rgb_image.width()
    height = rgb_image.height()
    bytes_per_line = rgb_image.bytesPerLine()
    buffer = rgb_image.constBits()
    array = np.frombuffer(buffer, dtype=np.uint8).reshape((height, bytes_per_line))
    return array[:, : width * 3].reshape((height, width, 3)).copy()


def _letterbox_qimage(
    image: QImage,
    target_width: int,
    target_height: int,
    resized_width: int,
    resized_height: int,
    pad_x: int,
    pad_y: int,
) -> np.ndarray:
    if QPainter is None or QColor is None or Qt is None:
        return _letterbox_array(
            _qimage_to_rgb_array(image),
            target_width,
            target_height,
            resized_width,
            resized_height,
            pad_x,
            pad_y,
        )

    source = image.convertToFormat(QImage.Format.Format_RGB888)
    resized = source.scaled(
        resized_width,
        resized_height,
        Qt.AspectRatioMode.IgnoreAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
    canvas = QImage(target_width, target_height, QImage.Format.Format_RGB888)
    canvas.fill(QColor(114, 114, 114))
    painter = QPainter(canvas)
    painter.drawImage(pad_x, pad_y, resized)
    painter.end()
    return _qimage_to_rgb_array(canvas)


def _letterbox_array(
    image: np.ndarray,
    target_width: int,
    target_height: int,
    resized_width: int,
    resized_height: int,
    pad_x: int,
    pad_y: int,
) -> np.ndarray:
    canvas = np.full((target_height, target_width, 3), 114, dtype=np.uint8)
    resized = _resize_array(image, resized_width, resized_height)
    canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
    return canvas


def _resize_array(image: np.ndarray, width: int, height: int) -> np.ndarray:
    try:
        import cv2
    except ImportError:
        y_indices = np.linspace(0, image.shape[0] - 1, height).astype(np.intp)
        x_indices = np.linspace(0, image.shape[1] - 1, width).astype(np.intp)
        return image[y_indices[:, None], x_indices]

    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    converted = boxes.astype(np.float32, copy=True)
    converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return converted


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    if boxes.size == 0:
        return []

    order = scores.argsort()[::-1]
    keep: list[int] = []

    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break

        ious = _box_iou(boxes[current], boxes[order[1:]])
        order = order[1:][ious <= threshold]

    return keep


def _nms_by_class(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    threshold: float,
) -> list[int]:
    selected: list[int] = []
    for class_id in np.unique(class_ids):
        class_indexes = np.where(class_ids == class_id)[0]
        kept_indexes = _nms(boxes[class_indexes], scores[class_indexes], threshold)
        selected.extend(int(class_indexes[index]) for index in kept_indexes)

    return sorted(selected, key=lambda index: float(scores[index]), reverse=True)


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


def load_class_labels(model_path: str | Path) -> dict[int, str]:
    labels_path = Path(model_path).with_name("labels.txt")
    if not labels_path.exists():
        return {0: "person", 1: "face"}

    labels = {}
    for index, raw_label in enumerate(labels_path.read_text(encoding="utf-8").splitlines()):
        label = raw_label.strip()
        if label:
            labels[index] = label
    return labels or {0: "person", 1: "face"}
