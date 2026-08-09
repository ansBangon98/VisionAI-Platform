from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.backends.base_backend import BackendConfig, InferenceBackend
from core.backends.onnxruntime_backend import OnnxRuntimeBackend
from core.backends.openvino_backend import OpenVinoBackend
from core.backends.pytorch_backend import PyTorchBackend
from core.backends.tensorrt_backend import TensorRtBackend
from core.config import load_yaml_config
from core.loaders.base_loader import LoaderConfig
from core.loaders.onnx_loader import OnnxModelLoader
from core.loaders.openvino_loader import OpenVinoModelLoader
from core.loaders.pytorch_loader import PyTorchModelLoader
from core.loaders.tensorrt_loader import TensorRtModelLoader
from core.models.base_detector import BaseDetector, Detection

try:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QImage, QPainter
except ImportError:  # pragma: no cover - PySide is present in the app runtime.
    Qt = None
    QColor = None
    QImage = None
    QPainter = None


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


class YoloDetector(BaseDetector):
    """Backend-flexible detector for Ultralytics YOLO detection models."""

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
        people_class_ids: Sequence[int] = (),
        class_ids: Sequence[int] | None = None,
    ):
        del people_class_ids
        self.model_path = Path(model_path)
        self.confidence_threshold = float(confidence_threshold)
        self.nms_iou_threshold = float(nms_iou_threshold)
        self.input_width = int(input_size[0])
        self.input_height = int(input_size[1])
        self.class_ids = (
            {int(class_id) for class_id in class_ids}
            if class_ids is not None
            else set()
        )

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
                "Failed to initialize YOLO detector "
                f"backend='{self.backend_name}', model='{self.model_path}': {error}"
            ) from error
        except Exception as error:
            raise RuntimeError(
                "Failed to initialize YOLO detector "
                f"backend='{self.backend_name}', model='{self.model_path}': {error}"
            ) from error
        if not self.backend.input_specs:
            raise RuntimeError(
                "The YOLO detector model does not expose any inputs: "
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
                "YOLO detector inference failed "
                f"backend='{self.backend_name}', model='{self.model_path}': {error}"
            ) from error
        except Exception as error:
            raise RuntimeError(
                "YOLO detector inference failed "
                f"backend='{self.backend_name}', model='{self.model_path}': {error}"
            ) from error

        if not outputs:
            raise RuntimeError(
                "YOLO detector inference returned no outputs "
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
                "YOLO detector postprocess failed "
                f"backend='{self.backend_name}', model='{self.model_path}': {error}"
            ) from error
        except Exception as error:
            raise RuntimeError(
                "YOLO detector postprocess failed "
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


YoloPeopleDetector = YoloDetector


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


def load_class_labels(model_path: str | Path) -> dict[int, str]:
    path = Path(model_path)

    labels = _class_labels_from_metadata(_load_model_sidecar_metadata(path))
    if labels:
        return labels

    labels = _load_labels_txt(path.with_name("labels.txt"))
    if labels:
        return labels

    labels = _class_labels_from_openvino_xml(path)
    if labels:
        return labels

    return {}


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
    if stem.lower() == "model":
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


def _load_labels_txt(labels_path: Path) -> dict[int, str]:
    if not labels_path.exists():
        return {}

    labels: dict[int, str] = {}
    next_index = 0
    for raw_label in labels_path.read_text(encoding="utf-8").splitlines():
        label = raw_label.strip()
        if not label or label.startswith("#"):
            continue

        key, separator, value = label.partition(":")
        if separator and key.strip().isdigit():
            class_id = int(key.strip())
            labels[class_id] = value.strip()
            next_index = max(next_index, class_id + 1)
            continue

        labels[next_index] = label
        next_index += 1

    return {class_id: label for class_id, label in labels.items() if label}


def _class_labels_from_metadata(metadata: Mapping[str, Any]) -> dict[int, str]:
    for key in ("classes", "names", "labels"):
        labels = _normalize_class_labels(metadata.get(key))
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

    if isinstance(value, str):
        return _labels_from_delimited_text(value, delimiter=",")

    return {}


def _class_labels_from_openvino_xml(model_path: Path) -> dict[int, str]:
    if model_path.suffix.lower() != ".xml" or not model_path.exists():
        return {}

    try:
        root = ET.parse(model_path).getroot()
    except (ET.ParseError, OSError):
        return {}

    for element in root.iter():
        if _xml_local_name(element.tag) != "labels":
            continue

        labels = _labels_from_delimited_text(element.attrib.get("value", ""))
        if labels:
            return labels

    return {}


def _labels_from_delimited_text(
    value: str,
    delimiter: str | None = None,
) -> dict[int, str]:
    text = value.strip()
    if not text:
        return {}

    parts = text.split(delimiter) if delimiter else text.split()
    return {
        index: label
        for index, label in enumerate(part.strip() for part in parts)
        if label
    }


def _xml_local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


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
