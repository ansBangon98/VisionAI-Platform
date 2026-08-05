from __future__ import annotations

import argparse
import re
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from typing import Any, Sequence


ShapeDim = int | str | None


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[ShapeDim, ...] = ()
    dtype: str = "unknown"
    is_input: bool = True


@dataclass
class LoaderConfig:
    model_path: str | Path
    device: str = "auto"
    providers: tuple[str, ...] = ()
    input_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)


class ModelLoader(ABC):
    runtime_name = "Model"

    def __init__(self, config: LoaderConfig):
        self.config = config
        self.model_path = Path(config.model_path).expanduser()
        self.providers: list[str] = []
        self.input_specs: list[TensorSpec] = []
        self.output_specs: list[TensorSpec] = []
        self.metadata: dict[str, Any] = {}
        self.initialized = False

    def initialize(self) -> "ModelLoader":
        if self.initialized:
            return self

        self._validate_model_path()
        self.initialize_runtime()
        self.set_providers()
        self.load_model()
        self.read_model_io()
        self.metadata = self.read_model_metadata()
        self.initialized = True
        return self

    @abstractmethod
    def initialize_runtime(self) -> None:
        """Import and configure the runtime used to load this model."""

    @abstractmethod
    def load_model(self) -> None:
        """Load the model/session/engine into the runtime."""

    @abstractmethod
    def read_model_io(self) -> None:
        """Populate input_specs and output_specs from the loaded model."""

    def read_model_metadata(self) -> dict[str, Any]:
        return {}

    def set_providers(self) -> None:
        if self.config.providers:
            self.providers = list(self.config.providers)
        elif self.config.device != "auto":
            self.providers = [self.config.device]
        else:
            self.providers = ["default"]

    def ensure_initialized(self) -> None:
        if not self.initialized:
            self.initialize()

    def close(self) -> None:
        """Release loader-owned resources if a concrete loader needs it."""

    def __enter__(self) -> "ModelLoader":
        return self.initialize()

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        self.close()

    def _validate_model_path(self) -> None:
        if not self.model_path.exists():
            raise RuntimeError(f"Model file does not exist: {self.model_path}")
        if not self.model_path.is_file():
            raise RuntimeError(f"Model path is not a file: {self.model_path}")


def build_common_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("model_path", help="Path to the model file.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Target device, for example cpu, cuda, cuda:0, gpu, AUTO, CPU, or GPU.",
    )
    parser.add_argument(
        "--provider",
        dest="providers",
        action="append",
        default=[],
        help="Runtime provider/device. Repeat or comma-separate for fallback order.",
    )
    parser.add_argument(
        "--input-shape",
        action="append",
        default=[],
        metavar="NAME:D0,D1,...",
        help="Shape override used by loaders that support reshaping at load time.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> LoaderConfig:
    extras = {
        key: value
        for key, value in vars(args).items()
        if key
        not in {
            "model_path",
            "device",
            "providers",
            "input_shape",
        }
    }
    return LoaderConfig(
        model_path=args.model_path,
        device=args.device,
        providers=tuple(split_csv(args.providers)),
        input_shapes=parse_shape_overrides(args.input_shape),
        extras=extras,
    )


def run_loader_cli(
    loader_cls: type[ModelLoader],
    argv: list[str] | None = None,
    parser: argparse.ArgumentParser | None = None,
) -> int:
    parser = parser or build_common_arg_parser(f"Load a {loader_cls.runtime_name} model.")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    loader = None

    try:
        loader = loader_cls(config_from_args(args))
        started_at = time.perf_counter()
        loader.initialize()
        load_ms = (time.perf_counter() - started_at) * 1000
        print_loader_summary(loader, load_ms)
    except Exception as error:
        print(f"{loader_cls.runtime_name} loader failed: {error}", file=sys.stderr)
        return 1
    finally:
        if loader is not None:
            loader.close()

    return 0


def print_loader_summary(loader: ModelLoader, load_ms: float | None = None) -> None:
    timing = "" if load_ms is None else f" in {load_ms:.2f} ms"
    print(f"{loader.runtime_name} loader initialized{timing}")
    print(f"Model: {loader.model_path}")
    if loader.providers:
        print(f"Providers: {', '.join(loader.providers)}")
    print("Inputs:")
    print(format_tensor_specs(loader.input_specs))
    print("Outputs:")
    print(format_tensor_specs(loader.output_specs))
    if loader.metadata:
        print("Metadata:")
        for key, value in sorted(loader.metadata.items()):
            print(f"  - {key}: {value}")


def format_tensor_specs(specs: Sequence[TensorSpec]) -> str:
    if not specs:
        return "  - none"
    return "\n".join(
        f"  - {spec.name}: shape={format_shape(spec.shape)}, dtype={spec.dtype}"
        for spec in specs
    )


def format_shape(shape: object) -> str:
    if shape is None:
        return "unknown"

    try:
        values = list(shape)  # type: ignore[arg-type]
    except TypeError:
        return str(shape)

    if not values:
        return "unknown"

    return "[" + ", ".join(_format_dim(dim) for dim in values) + "]"


def parse_shape_overrides(entries: Sequence[str]) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    for entry in entries:
        name, value = split_name_value(entry)
        dims = [part for part in re.split(r"[,xX×\s]+", value.strip()) if part]
        if not dims:
            raise ValueError(f"Invalid input shape: {entry}")
        shapes[name] = tuple(int(dim) for dim in dims)
    return shapes


def split_name_value(entry: str) -> tuple[str, str]:
    if ":" not in entry:
        return "", entry
    name, value = entry.split(":", 1)
    if not value:
        raise ValueError(f"Invalid name/value option: {entry}")
    return name.strip(), value.strip()


def split_csv(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        result.extend(part.strip() for part in value.split(",") if part.strip())
    return result


def _format_dim(dim: object) -> str:
    if dim is None:
        return "?"
    if isinstance(dim, Integral) and int(dim) <= 0:
        return "?"
    text = str(dim)
    if text in {"-1", "None", "?", ""}:
        return "?"
    return text
