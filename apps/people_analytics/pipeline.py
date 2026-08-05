from __future__ import annotations

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
    ):
        self.model_path = Path(model_path)
        self.confidence_threshold = float(confidence_threshold)
        self.nms_iou_threshold = float(nms_iou_threshold)
        self.input_width = int(input_size[0])
        self.input_height = int(input_size[1])
        self.people_class_ids = {int(class_id) for class_id in people_class_ids}

        self.backend_name = _normalize_backend_name(backend_name)
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
        if not self.backend.input_specs:
            raise RuntimeError("The people detector model does not expose any inputs.")
        self.input_name = self.backend.input_specs[0].name

    def predict(self, frame: object) -> list[Detection]:
        frame_height, frame_width = frame_size(frame)
        input_tensor, scale, pad_x, pad_y = self._preprocess(frame)
        outputs = self.backend.infer({self.input_name: input_tensor})
        output = next(iter(outputs.values()))
        return self._postprocess(output, scale, pad_x, pad_y, frame_width, frame_height)

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
        if predictions.ndim == 3:
            predictions = predictions[0]
        if predictions.ndim != 2:
            return []

        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T
        if predictions.shape[1] < 6:
            return []

        boxes_xywh = predictions[:, :4]
        class_scores = predictions[:, 4:]
        scores = class_scores.max(axis=1)
        class_ids = class_scores.argmax(axis=1)

        keep = scores >= self.confidence_threshold
        if self.people_class_ids:
            keep = keep & np.isin(class_ids, list(self.people_class_ids))

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

        selected = _nms(boxes, scores, self.nms_iou_threshold)
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
    ):
        self.detector = detector
        self.tracker = tracker

    def process(self, frame: object) -> list[dict[str, object]]:
        detections = self.detector.predict(frame)
        tracks = self.tracker.update(detections, frame_size(frame))

        return [
            {
                "track_id": track.track_id,
                "bbox": track.bbox,
            }
            for track in tracks
        ]


def create_pipeline_from_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> PeopleAnalyticsPipeline:
    config = load_config(config_path)
    detector_config = config.get("detector", {})
    tracker_config = config.get("tracker", {})

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
        people_class_ids=detector_config.get("people_class_ids", (0,)),
    )
    tracker = ByteTrackPeopleTracker(
        track_thresh=tracker_config.get("track_thresh", 0.5),
        match_thresh=tracker_config.get("match_thresh", 0.8),
        track_buffer=tracker_config.get("track_buffer", 30),
        frame_rate=tracker_config.get("frame_rate", 30),
    )
    return PeopleAnalyticsPipeline(detector=detector, tracker=tracker)


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    return load_yaml_config(config_path)


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
