from __future__ import annotations

try:
    from .base_loader import (
        ModelLoader,
        TensorSpec,
        build_common_arg_parser,
        run_loader_cli,
    )
except ImportError:  # Allows `python core/loaders/tensorrt_loader.py ...`.
    from base_loader import (  # type: ignore[no-redef]
        ModelLoader,
        TensorSpec,
        build_common_arg_parser,
        run_loader_cli,
    )


class TensorRtModelLoader(ModelLoader):
    runtime_name = "TensorRT"

    def __init__(self, config):
        super().__init__(config)
        self.trt = None
        self.logger = None
        self.runtime = None
        self.engine = None
        self.device_index = 0

    def initialize_runtime(self) -> None:
        try:
            import tensorrt as trt
        except ImportError as error:
            raise RuntimeError(
                "tensorrt is not installed. Install TensorRT Python bindings "
                "to use this loader."
            ) from error

        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

    def set_providers(self) -> None:
        device = self.config.device
        for provider in self.config.providers:
            if provider.lower().startswith("cuda"):
                device = provider
                break

        self.device_index = _device_index_from_text(device)
        self.providers = list(self.config.providers) or [
            "TensorRT",
            f"CUDA:{self.device_index}",
        ]

    def load_model(self) -> None:
        engine_bytes = self.model_path.read_bytes()
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {self.model_path}")

    def read_model_io(self) -> None:
        input_specs: list[TensorSpec] = []
        output_specs: list[TensorSpec] = []

        if self._uses_tensor_io_api():
            for index in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(index)
                mode = self.engine.get_tensor_mode(name)
                is_input = mode == self.trt.TensorIOMode.INPUT
                spec = TensorSpec(
                    name=name,
                    shape=tuple(self.engine.get_tensor_shape(name)),
                    dtype=self._dtype_name(self.engine.get_tensor_dtype(name)),
                    is_input=is_input,
                )
                (input_specs if is_input else output_specs).append(spec)
        else:
            for index in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(index)
                is_input = self.engine.binding_is_input(index)
                spec = TensorSpec(
                    name=name,
                    shape=tuple(self.engine.get_binding_shape(index)),
                    dtype=self._dtype_name(self.engine.get_binding_dtype(index)),
                    is_input=is_input,
                )
                (input_specs if is_input else output_specs).append(spec)

        self.input_specs = input_specs
        self.output_specs = output_specs

    def _uses_tensor_io_api(self) -> bool:
        return hasattr(self.engine, "num_io_tensors")

    def _dtype_name(self, dtype: object) -> str:
        text = str(dtype).lower()
        dtype_map = {
            "bool": "bool",
            "boolean": "bool",
            "float16": "float16",
            "float32": "float32",
            "float": "float32",
            "half": "float16",
            "int8": "int8",
            "int32": "int32",
            "int64": "int64",
            "uint8": "uint8",
        }
        for needle, dtype_name in dtype_map.items():
            if needle in text:
                return dtype_name
        return text

    def read_model_metadata(self) -> dict[str, object]:
        metadata = {
            "device_index": self.device_index,
            "has_implicit_batch_dimension": getattr(
                self.engine,
                "has_implicit_batch_dimension",
                None,
            ),
            "max_batch_size": getattr(self.engine, "max_batch_size", None),
            "num_layers": getattr(self.engine, "num_layers", None),
            "num_optimization_profiles": getattr(
                self.engine,
                "num_optimization_profiles",
                None,
            ),
        }
        return {key: value for key, value in metadata.items() if value is not None}


def _device_index_from_text(device: str) -> int:
    text = device.lower()
    if ":" in text:
        _, index = text.rsplit(":", 1)
        return int(index)
    return 0


def build_arg_parser():
    return build_common_arg_parser("Load and inspect a TensorRT engine.")


def main(argv: list[str] | None = None) -> int:
    return run_loader_cli(TensorRtModelLoader, argv=argv, parser=build_arg_parser())


if __name__ == "__main__":
    raise SystemExit(main())
