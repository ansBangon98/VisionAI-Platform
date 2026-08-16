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
from core.models.detection.postprocess import (
    PostprocessContext,
    create_yolo_postprocessor,
)
from core.models.detection.postprocess.base_postprocessor import (
    normalize_bbox_format,
    normalize_output_format,
    parse_optional_bool,
)
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
        output_format: str | None = None,
        bbox_format: str | None = None,
        end2end: bool | str | int | None = None,
    ):
        del people_class_ids
        self.model_path = Path(model_path)
        self.sidecar_metadata = _load_model_sidecar_metadata(self.model_path)
        self.confidence_threshold = float(confidence_threshold)
        self.nms_iou_threshold = float(nms_iou_threshold)
        self.input_width = int(input_size[0])
        self.input_height = int(input_size[1])
        self.output_format = normalize_output_format(
            output_format or _metadata_output_format(self.sidecar_metadata)
        )
        self.bbox_format = normalize_bbox_format(
            bbox_format or _metadata_bbox_format(self.sidecar_metadata)
        )
        self.end2end = _resolve_end2end(end2end, self.sidecar_metadata)
        self.class_ids = (
            {int(class_id) for class_id in class_ids}
            if class_ids is not None
            else set()
        )
        self.postprocessor = create_yolo_postprocessor(
            output_format=self.output_format,
            bbox_format=self.bbox_format,
            end2end=self.end2end,
            confidence_threshold=self.confidence_threshold,
            nms_iou_threshold=self.nms_iou_threshold,
            class_ids=self.class_ids,
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
        return {
            "base_model": _profile_base_model(self.model_path, self.sidecar_metadata),
            "framework": _profile_framework_name(self.backend_name, self.backend),
            "device": _profile_device_name(self.backend),
            "model_input_size": f"{self.input_width}x{self.input_height}",
            "confidence_threshold": self.confidence_threshold,
            "model_path": str(self.model_path),
        }

    def set_confidence_threshold(self, confidence_threshold: float) -> None:
        self.confidence_threshold = max(0.0, min(1.0, float(confidence_threshold)))
        self.postprocessor = create_yolo_postprocessor(
            output_format=self.output_format,
            bbox_format=self.bbox_format,
            end2end=self.end2end,
            confidence_threshold=self.confidence_threshold,
            nms_iou_threshold=self.nms_iou_threshold,
            class_ids=self.class_ids,
        )

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

        output = self._select_output(outputs)
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
        return self.postprocessor.process(
            output,
            PostprocessContext(
                scale,
                pad_x,
                pad_y,
                frame_width,
                frame_height,
            ),
        )

    def _select_output(self, outputs: Mapping[str, object]) -> object:
        return self.postprocessor.select_output(outputs)


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


def _metadata_output_format(metadata: Mapping[str, Any]) -> str:
    output = metadata.get("output")
    if isinstance(output, Mapping):
        value = output.get("format")
        if value:
            return str(value)

    for key in ("output_format", "format"):
        value = metadata.get(key)
        if value:
            return str(value)
    return ""


def _metadata_bbox_format(metadata: Mapping[str, Any]) -> str:
    output = metadata.get("output")
    if isinstance(output, Mapping):
        value = output.get("bbox_format")
        if value:
            return str(value)

    for key in ("bbox_format", "box_format"):
        value = metadata.get(key)
        if value:
            return str(value)
    return ""


def _resolve_end2end(
    configured_value: bool | str | int | None,
    metadata: Mapping[str, Any],
) -> bool | None:
    parsed = parse_optional_bool(configured_value)
    if parsed is not None:
        return parsed

    for source in (metadata, metadata.get("export")):
        if not isinstance(source, Mapping) or "end2end" not in source:
            continue
        parsed = parse_optional_bool(source.get("end2end"))
        if parsed is not None:
            return parsed

    return None


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
