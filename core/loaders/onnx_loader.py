from __future__ import annotations

try:
    from .base_loader import (
        ModelLoader,
        TensorSpec,
        build_common_arg_parser,
        run_loader_cli,
    )
except ImportError:  # Allows `python core/loaders/onnx_loader.py ...`.
    from base_loader import (  # type: ignore[no-redef]
        ModelLoader,
        TensorSpec,
        build_common_arg_parser,
        run_loader_cli,
    )


class OnnxModelLoader(ModelLoader):
    runtime_name = "ONNX Runtime"

    def __init__(self, config):
        super().__init__(config)
        self.ort = None
        self.session = None

    def initialize_runtime(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError(
                "onnxruntime is not installed. Install onnxruntime or "
                "onnxruntime-gpu to use the ONNX loader."
            ) from error

        self.ort = ort

    def set_providers(self) -> None:
        available = list(self.ort.get_available_providers())
        requested = list(self.config.providers)

        if requested:
            missing = [provider for provider in requested if provider not in available]
            if missing:
                raise RuntimeError(
                    "Requested ONNX Runtime providers are not available: "
                    f"{', '.join(missing)}. Available providers: {', '.join(available)}"
                )
            self.providers = requested
            return

        device = self.config.device.lower()
        if device == "cpu":
            self.providers = self._require_available(["CPUExecutionProvider"], available)
        elif device.startswith("cuda") or device == "gpu":
            self.providers = self._provider_chain(
                "CUDAExecutionProvider",
                ["CUDAExecutionProvider", "CPUExecutionProvider"], available
            )
        elif device in {"tensorrt", "trt"}:
            self.providers = self._provider_chain(
                "TensorrtExecutionProvider",
                [
                    "TensorrtExecutionProvider",
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                ],
                available,
            )
        else:
            preferred = [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
            self.providers = [provider for provider in preferred if provider in available]
            if not self.providers:
                self.providers = available

    def load_model(self) -> None:
        self.session = self.ort.InferenceSession(
            str(self.model_path),
            providers=self.providers,
        )
        self.providers = list(self.session.get_providers())

    def read_model_io(self) -> None:
        self.input_specs = [
            self._node_arg_to_spec(node_arg, is_input=True)
            for node_arg in self.session.get_inputs()
        ]
        self.output_specs = [
            self._node_arg_to_spec(node_arg, is_input=False)
            for node_arg in self.session.get_outputs()
        ]

    def _node_arg_to_spec(self, node_arg, is_input: bool) -> TensorSpec:
        shape = tuple(node_arg.shape or ())
        dtype = _onnx_dtype_name(node_arg.type)
        return TensorSpec(
            name=node_arg.name,
            shape=shape,
            dtype=dtype,
            is_input=is_input,
        )

    def _require_available(self, providers: list[str], available: list[str]) -> list[str]:
        selected = [provider for provider in providers if provider in available]
        if selected:
            return selected
        raise RuntimeError(
            f"None of {', '.join(providers)} are available. "
            f"Available providers: {', '.join(available)}"
        )

    def _provider_chain(
        self,
        required_provider: str,
        providers: list[str],
        available: list[str],
    ) -> list[str]:
        if required_provider not in available:
            raise RuntimeError(
                f"{required_provider} is not available. "
                f"Available providers: {', '.join(available)}"
            )
        return [provider for provider in providers if provider in available]

    def read_model_metadata(self) -> dict[str, object]:
        metadata = self.session.get_modelmeta()
        values = {
            "description": metadata.description,
            "domain": metadata.domain,
            "graph_name": metadata.graph_name,
            "producer_name": metadata.producer_name,
            "version": metadata.version,
        }
        values.update(dict(metadata.custom_metadata_map or {}))
        return {key: value for key, value in values.items() if value not in ("", None)}


def _onnx_dtype_name(type_name: str | None) -> str:
    if not type_name:
        return "unknown"

    text = type_name.strip().lower()
    if text.startswith("tensor(") and text.endswith(")"):
        text = text[len("tensor(") : -1]

    dtype_map = {
        "bool": "bool",
        "double": "float64",
        "float": "float32",
        "float16": "float16",
        "int8": "int8",
        "int16": "int16",
        "int32": "int32",
        "int64": "int64",
        "uint8": "uint8",
        "uint16": "uint16",
        "uint32": "uint32",
        "uint64": "uint64",
    }
    return dtype_map.get(text, text)


def build_arg_parser():
    return build_common_arg_parser("Load and inspect an ONNX Runtime model.")


def main(argv: list[str] | None = None) -> int:
    return run_loader_cli(OnnxModelLoader, argv=argv, parser=build_arg_parser())


if __name__ == "__main__":
    raise SystemExit(main())
