from __future__ import annotations

import argparse
import sys
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from typing import Any, Sequence

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional until inference/dummy input use.
    np = None

from core.loaders.base_loader import (
    LoaderConfig,
    ModelLoader,
    TensorSpec,
    build_common_arg_parser,
    format_shape,
    parse_shape_overrides,
    print_loader_summary,
    split_csv,
    split_name_value,
)


InputData = Mapping[str, Any] | Sequence[Any] | Any


@dataclass
class BackendConfig:
    input_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    input_dtypes: dict[str, str] = field(default_factory=dict)
    dynamic_dim: int = 1
    warmup_runs: int = 0
    extras: dict[str, Any] = field(default_factory=dict)


class InferenceBackend(ABC):
    backend_name = "Inference Backend"

    def __init__(self, loader: ModelLoader, config: BackendConfig | None = None):
        self.loader = loader
        self.config = config or BackendConfig()
        self.initialized = False

    @property
    def input_specs(self) -> list[TensorSpec]:
        return self.loader.input_specs

    @property
    def output_specs(self) -> list[TensorSpec]:
        return self.loader.output_specs

    def initialize(self) -> "InferenceBackend":
        if self.initialized:
            return self

        self.loader.ensure_initialized()
        self.initialize_backend()
        self.initialized = True
        self.warmup(self.config.warmup_runs)
        return self

    def initialize_backend(self) -> None:
        """Allocate execution-time resources for this backend."""

    @abstractmethod
    def infer(self, inputs: InputData) -> dict[str, Any]:
        """Run inference and return raw backend outputs keyed by output name."""

    def warmup(self, runs: int | None = None) -> None:
        runs = self.config.warmup_runs if runs is None else runs
        if runs <= 0:
            return

        dummy_inputs = self.make_dummy_inputs()
        for _ in range(runs):
            self.infer(dummy_inputs)

    def make_dummy_inputs(self) -> dict[str, Any]:
        numpy = require_numpy()
        if not self.input_specs:
            raise RuntimeError("The loaded model does not expose any inputs.")

        dummy_inputs: dict[str, Any] = {}
        for spec in self.input_specs:
            shape = self._shape_for_input(spec)
            dtype = numpy_dtype_from_name(self._dtype_for_input(spec))
            dummy_inputs[spec.name] = numpy.zeros(shape, dtype=dtype)
        return dummy_inputs

    def close(self) -> None:
        """Release backend-owned execution resources."""

    def __enter__(self) -> "InferenceBackend":
        return self.initialize()

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        self.close()

    def _normalize_input_map(self, inputs: InputData) -> dict[str, Any]:
        if isinstance(inputs, Mapping):
            missing = [spec.name for spec in self.input_specs if spec.name not in inputs]
            if missing:
                raise RuntimeError(f"Missing model inputs: {', '.join(missing)}")
            return {spec.name: inputs[spec.name] for spec in self.input_specs}

        if len(self.input_specs) == 1:
            return {self.input_specs[0].name: inputs}

        if _is_sequence_input(inputs):
            if len(inputs) != len(self.input_specs):
                raise RuntimeError(
                    f"Expected {len(self.input_specs)} input values, got {len(inputs)}."
                )
            return {
                spec.name: value for spec, value in zip(self.input_specs, inputs)
            }

        raise RuntimeError("Multiple-input models require a mapping or sequence of inputs.")

    def _shape_for_input(self, spec: TensorSpec) -> tuple[int, ...]:
        if spec.name in self.config.input_shapes:
            return self.config.input_shapes[spec.name]

        if "" in self.config.input_shapes and len(self.input_specs) == 1:
            return self.config.input_shapes[""]

        if spec.name in self.loader.config.input_shapes:
            return self.loader.config.input_shapes[spec.name]

        if "" in self.loader.config.input_shapes and len(self.input_specs) == 1:
            return self.loader.config.input_shapes[""]

        if not spec.shape:
            raise RuntimeError(
                f"Input '{spec.name}' has no static shape. Pass "
                f"--input-shape {spec.name}:1,3,640,640 for dummy inference."
            )

        return resolve_shape(spec.shape, self.config.dynamic_dim)

    def _dtype_for_input(self, spec: TensorSpec) -> str:
        if spec.name in self.config.input_dtypes:
            return self.config.input_dtypes[spec.name]

        if "" in self.config.input_dtypes and len(self.input_specs) == 1:
            return self.config.input_dtypes[""]

        default_dtype = self.config.extras.get("default_input_dtype", "float32")
        return spec.dtype if spec.dtype != "unknown" else default_dtype


def build_backend_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = build_common_arg_parser(description)
    for action in parser._actions:
        if action.dest == "input_shape":
            action.help = (
                "Shape override used for load-time reshape and dummy inference."
            )
    parser.add_argument(
        "--input-dtype",
        action="append",
        default=[],
        metavar="NAME:DTYPE",
        help="Input dtype override used by dummy inference.",
    )
    parser.add_argument(
        "--dynamic-dim",
        type=int,
        default=1,
        help="Value used when a model input dimension is dynamic.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=0,
        help="Number of warmup inference calls after backend initialization.",
    )
    parser.add_argument(
        "--run-dummy",
        action="store_true",
        help="Run one dummy inference after loading and backend initialization.",
    )
    return parser


def run_backend_cli(
    loader_cls: type[ModelLoader],
    backend_cls: type[InferenceBackend],
    argv: list[str] | None = None,
    parser: argparse.ArgumentParser | None = None,
    backend_extra_names: set[str] | None = None,
) -> int:
    parser = parser or build_backend_arg_parser(
        f"Load a {loader_cls.runtime_name} model and run {backend_cls.backend_name}."
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    loader = None
    backend = None

    try:
        loader = loader_cls(loader_config_from_backend_args(args, backend_extra_names))
        started_at = time.perf_counter()
        loader.initialize()
        load_ms = (time.perf_counter() - started_at) * 1000

        backend = backend_cls(loader, backend_config_from_args(args))
        started_at = time.perf_counter()
        backend.initialize()
        backend_ms = (time.perf_counter() - started_at) * 1000

        print_loader_summary(loader, load_ms)
        print(f"{backend.backend_name} initialized in {backend_ms:.2f} ms")

        if args.run_dummy:
            dummy_inputs = backend.make_dummy_inputs()
            started_at = time.perf_counter()
            outputs = backend.infer(dummy_inputs)
            inference_ms = (time.perf_counter() - started_at) * 1000
            print_output_summary(outputs, inference_ms)
    except Exception as error:
        print(f"{backend_cls.backend_name} failed: {error}", file=sys.stderr)
        return 1
    finally:
        if backend is not None:
            backend.close()
        if loader is not None:
            loader.close()

    return 0


def loader_config_from_backend_args(
    args: argparse.Namespace,
    backend_extra_names: set[str] | None = None,
) -> LoaderConfig:
    excluded = {
        "model_path",
        "device",
        "providers",
        "input_shape",
        "input_dtype",
        "dynamic_dim",
        "warmup_runs",
        "run_dummy",
    }
    excluded.update(backend_extra_names or set())

    extras = {
        key: value
        for key, value in vars(args).items()
        if key not in excluded
    }
    return LoaderConfig(
        model_path=args.model_path,
        device=args.device,
        providers=tuple(split_csv(args.providers)),
        input_shapes=parse_shape_overrides(args.input_shape),
        extras=extras,
    )


def backend_config_from_args(args: argparse.Namespace) -> BackendConfig:
    excluded = {
        "model_path",
        "device",
        "providers",
        "input_shape",
        "input_dtype",
        "dynamic_dim",
        "warmup_runs",
        "run_dummy",
    }
    extras = {
        key: value
        for key, value in vars(args).items()
        if key not in excluded
    }
    return BackendConfig(
        input_shapes=parse_shape_overrides(args.input_shape),
        input_dtypes=parse_dtype_overrides(args.input_dtype),
        dynamic_dim=args.dynamic_dim,
        warmup_runs=args.warmup_runs,
        extras=extras,
    )


def print_output_summary(outputs: Mapping[str, Any], inference_ms: float) -> None:
    print(f"Inference completed in {inference_ms:.2f} ms")
    for name, value in outputs.items():
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", type(value).__name__)
        print(f"  - {name}: shape={format_shape(shape)}, dtype={dtype}")


def parse_dtype_overrides(entries: Sequence[str]) -> dict[str, str]:
    dtypes: dict[str, str] = {}
    for entry in entries:
        name, value = split_name_value(entry)
        dtypes[name] = value.strip()
    return dtypes


def resolve_shape(shape: Sequence[object], dynamic_dim: int = 1) -> tuple[int, ...]:
    resolved: list[int] = []
    for dim in shape:
        if isinstance(dim, Integral):
            resolved.append(int(dim) if int(dim) > 0 else dynamic_dim)
            continue

        if dim is None:
            resolved.append(dynamic_dim)
            continue

        text = str(dim).strip()
        if text.isdigit():
            resolved.append(int(text))
        else:
            resolved.append(dynamic_dim)

    return tuple(resolved)


def numpy_dtype_from_name(dtype_name: str | None, fallback: str = "float32"):
    numpy = require_numpy()
    if not dtype_name or dtype_name == "unknown":
        return numpy.dtype(fallback)

    text = str(dtype_name).lower().strip()
    text = text.replace("tensor(", "").replace(")", "")
    text = text.replace("torch.", "").replace("type.", "")
    aliases = {
        "bool": "bool",
        "boolean": "bool",
        "double": "float64",
        "f16": "float16",
        "f32": "float32",
        "f64": "float64",
        "float": "float32",
        "fp16": "float16",
        "fp32": "float32",
        "fp64": "float64",
        "half": "float16",
        "i8": "int8",
        "i16": "int16",
        "i32": "int32",
        "i64": "int64",
        "int": "int64",
        "long": "int64",
        "u8": "uint8",
        "u16": "uint16",
        "u32": "uint32",
        "u64": "uint64",
    }
    text = aliases.get(text, text)

    try:
        return numpy.dtype(text)
    except TypeError:
        return numpy.dtype(fallback)


def require_numpy():
    if np is None:
        raise RuntimeError("NumPy is required for dummy inputs and array conversion.")
    return np


def _is_sequence_input(value: object) -> bool:
    if isinstance(value, (str, bytes, bytearray)):
        return False
    if np is not None and isinstance(value, np.ndarray):
        return False
    return isinstance(value, Sequence)


def add_project_root_to_path() -> None:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
