from __future__ import annotations

import json
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
from core.models.segmentation.base_segmentor import BaseSegmentor
from core.models.yolo_detector import frame_size, frame_to_rgb_array
from core.postprocess.segmentation import resize_mask, semantic_mask_from_logits


@dataclass(frozen=True)
class SegmentorBackendSpec:
    loader_cls: type
    backend_cls: type[InferenceBackend]
    expected_suffixes: tuple[str, ...] = ()


SEGMENTOR_BACKENDS: dict[str, SegmentorBackendSpec] = {
    "onnxruntime": SegmentorBackendSpec(OnnxModelLoader, OnnxRuntimeBackend, (".onnx",)),
    "openvino": SegmentorBackendSpec(OpenVinoModelLoader, OpenVinoBackend, (".xml",)),
    "pytorch": SegmentorBackendSpec(PyTorchModelLoader, PyTorchBackend, (".pt", ".pth")),
    "tensorrt": SegmentorBackendSpec(
        TensorRtModelLoader,
        TensorRtBackend,
        (".engine", ".plan", ".trt"),
    ),
}


class SegFormerSegmentor(BaseSegmentor):
    """Backend-flexible semantic segmentor for exported SegFormer models."""

    def __init__(
        self,
        model_path: str | Path,
        backend_name: str = "openvino",
        device: str = "auto",
        providers: Sequence[str] = (),
        input_size: Sequence[int] | None = None,
        input_shapes: dict[str, Sequence[int]] | None = None,
        input_dtypes: dict[str, str] | None = None,
        warmup_runs: int = 0,
        dynamic_dim: int = 1,
        backend_options: dict[str, Any] | None = None,
        class_labels: Mapping[int, str] | None = None,
        processor_config_path: str | Path | None = None,
    ):
        self.model_path = Path(model_path)
        self.model_root = _model_root(self.model_path)
        self.metadata = _load_model_metadata(self.model_root)
        self.processor_config = _load_processor_config(
            self.model_root,
            processor_config_path
            or _metadata_processor_config_path(self.metadata),
        )
        self.class_labels = (
            _normalize_class_labels(class_labels)
            or _metadata_class_labels(self.metadata)
            or _load_labels_txt(self.model_root / "labels.txt")
        )
        self.input_width, self.input_height = _resolve_input_size(
            input_size,
            self.metadata,
            self.processor_config,
        )
        self.mean, self.std, self.rescale_factor = _normalization_values(
            self.processor_config,
            self.metadata,
        )
        self.do_rescale = bool(self.processor_config.get("do_rescale", True))
        self.do_normalize = bool(
            self.processor_config.get(
                "do_normalize",
                self.metadata.get("preprocessing", {}).get("normalize", True)
                if isinstance(self.metadata.get("preprocessing"), Mapping)
                else True,
            )
        )

        self.backend_name = _normalize_backend_name(backend_name)
        self.backend = create_segmentation_backend(
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
        try:
            self.backend.initialize()
        except Exception as error:
            raise RuntimeError(
                "Failed to initialize SegFormer segmentor "
                f"backend='{self.backend_name}', model='{self.model_path}': {error}"
            ) from error

        if not self.backend.input_specs:
            raise RuntimeError(
                "The segmentation model does not expose any inputs: "
                f"{self.model_path}"
            )
        self.input_name = self.backend.input_specs[0].name

    @property
    def class_count(self) -> int | None:
        return max(self.class_labels) + 1 if self.class_labels else None

    def model_profile(self) -> dict[str, object]:
        return {
            "base_model": _profile_base_model(self.model_path, self.metadata),
            "framework": _profile_framework_name(self.backend_name, self.backend),
            "device": _profile_device_name(self.backend),
            "model_input_size": f"{self.input_width}x{self.input_height}",
            "model_path": str(self.model_path),
        }

    def predict(self, frame: object) -> np.ndarray:
        try:
            frame_height, frame_width = frame_size(frame)
            tensor = self._preprocess(frame)
            outputs = self.backend.infer({self.input_name: tensor})
        except RuntimeError:
            raise
        except Exception as error:
            raise RuntimeError(
                "SegFormer inference failed "
                f"backend='{self.backend_name}', model='{self.model_path}': {error}"
            ) from error

        output = _select_segmentation_output(outputs)
        mask = semantic_mask_from_logits(output, class_count=self.class_count)
        return resize_mask(mask, frame_width, frame_height)

    def _preprocess(self, frame: object) -> np.ndarray:
        image = frame_to_rgb_array(frame)
        if image.shape[:2] != (self.input_height, self.input_width):
            image = _resize_rgb(image, self.input_width, self.input_height)

        tensor = image.astype(np.float32, copy=False)
        if self.do_rescale:
            tensor *= self.rescale_factor
        if self.do_normalize:
            tensor = (tensor - self.mean) / self.std
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(tensor, dtype=np.float32)


def create_segmentation_backend(
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
    spec = SEGMENTOR_BACKENDS[normalized_backend]
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


def available_segmentor_backends() -> tuple[str, ...]:
    return tuple(SEGMENTOR_BACKENDS)


def _select_segmentation_output(outputs: Mapping[str, object]) -> object:
    if not outputs:
        raise RuntimeError("Segmentation inference returned no outputs.")

    for name in ("logits", "output", "output_0"):
        if name in outputs:
            return outputs[name]

    for value in outputs.values():
        array = np.asarray(value)
        if array.ndim in {2, 3, 4}:
            return value

    return next(iter(outputs.values()))


def _resolve_input_size(
    input_size: Sequence[int] | None,
    metadata: Mapping[str, Any],
    processor_config: Mapping[str, Any],
) -> tuple[int, int]:
    if input_size:
        return _normalize_size_pair(input_size)

    input_config = metadata.get("input")
    if isinstance(input_config, Mapping):
        width = input_config.get("width")
        height = input_config.get("height")
        if width and height:
            return int(width), int(height)

    processor_size = processor_config.get("size")
    if isinstance(processor_size, Mapping):
        width = processor_size.get("width")
        height = processor_size.get("height")
        if width and height:
            return int(width), int(height)

    return (512, 512)


def _normalization_values(
    processor_config: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, float]:
    preprocessing = metadata.get("preprocessing", {})
    if not isinstance(preprocessing, Mapping):
        preprocessing = {}

    mean = processor_config.get("image_mean", preprocessing.get("mean", (0.485, 0.456, 0.406)))
    std = processor_config.get("image_std", preprocessing.get("std", (0.229, 0.224, 0.225)))
    rescale_factor = float(processor_config.get("rescale_factor", 1.0 / 255.0))
    return (
        np.asarray(mean, dtype=np.float32).reshape(1, 1, 3),
        np.asarray(std, dtype=np.float32).reshape(1, 1, 3),
        rescale_factor,
    )


def _resize_rgb(image: np.ndarray, width: int, height: int) -> np.ndarray:
    try:
        import cv2
    except ImportError:
        y_indices = np.linspace(0, image.shape[0] - 1, height).astype(np.intp)
        x_indices = np.linspace(0, image.shape[1] - 1, width).astype(np.intp)
        return np.ascontiguousarray(image[y_indices[:, None], x_indices])

    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def _normalize_backend_name(backend_name: str) -> str:
    normalized = str(backend_name or "openvino").strip().lower()
    normalized = normalized.replace("-", "").replace("_", "").replace(" ", "")
    aliases = {
        "onnx": "onnxruntime",
        "onnxruntime": "onnxruntime",
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
        supported = ", ".join(available_segmentor_backends())
        raise RuntimeError(
            f"Unsupported segmentation backend '{backend_name}'. "
            f"Supported backends: {supported}."
        ) from error


def _validate_backend_model_path(
    model_path: Path,
    backend_name: str,
    spec: SegmentorBackendSpec,
) -> None:
    if not spec.expected_suffixes:
        return
    suffix = model_path.suffix.lower()
    if suffix in spec.expected_suffixes:
        return
    expected = ", ".join(spec.expected_suffixes)
    raise RuntimeError(
        f"Segmentation backend '{backend_name}' expects model suffix {expected}; "
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


def _normalize_size_pair(value: Sequence[int] | str | object) -> tuple[int, int]:
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


def _model_root(model_path: Path) -> Path:
    for candidate in (model_path.parent, model_path.parent.parent):
        if (candidate / "metadata.yaml").exists() or (candidate / "labels.txt").exists():
            return candidate
    return model_path.parent


def _load_model_metadata(model_root: Path) -> dict[str, Any]:
    for metadata_path in (
        model_root / "metadata.yaml",
        model_root / "metadata.yml",
    ):
        if not metadata_path.exists():
            continue
        try:
            return load_yaml_config(metadata_path)
        except RuntimeError:
            return {}
    return {}


def _metadata_processor_config_path(metadata: Mapping[str, Any]) -> str:
    preprocessing = metadata.get("preprocessing", {})
    if isinstance(preprocessing, Mapping):
        value = preprocessing.get("processor_config")
        if value:
            return str(value)
    return ""


def _load_processor_config(
    model_root: Path,
    processor_config_path: str | Path | None,
) -> dict[str, Any]:
    if not processor_config_path:
        return {}

    path = Path(processor_config_path)
    candidates = [path] if path.is_absolute() else [model_root / path, Path.cwd() / path]
    resolved_path = next((candidate for candidate in candidates if candidate.exists()), None)
    if resolved_path is None:
        return {}

    try:
        return json.loads(resolved_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _metadata_class_labels(metadata: Mapping[str, Any]) -> dict[int, str]:
    for key in ("classes", "labels", "names"):
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
    return {}


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


def _profile_base_model(model_path: Path, metadata: Mapping[str, Any]) -> str:
    for key in ("name", "architecture", "base_model", "model_name"):
        value = metadata.get(key)
        if value:
            return _prettify_model_name(str(value))

    stem = model_path.stem
    if stem.lower() == "model":
        stem = model_path.parent.parent.name
    return _prettify_model_name(stem)


def _prettify_model_name(value: str) -> str:
    text = str(value).strip().replace("_", "-")
    if text.lower().startswith("segformer"):
        return "SegFormer" + text[9:]
    return text or "--"


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
