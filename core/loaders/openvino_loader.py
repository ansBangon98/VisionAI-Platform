from __future__ import annotations

try:
    from .base_loader import (
        ModelLoader,
        TensorSpec,
        build_common_arg_parser,
        run_loader_cli,
    )
except ImportError:  # Allows `python core/loaders/openvino_loader.py ...`.
    from base_loader import (  # type: ignore[no-redef]
        ModelLoader,
        TensorSpec,
        build_common_arg_parser,
        run_loader_cli,
    )


class OpenVinoModelLoader(ModelLoader):
    runtime_name = "OpenVINO"

    def __init__(self, config):
        super().__init__(config)
        self.ov = None
        self.core = None
        self.model = None
        self.compiled_model = None
        self.device_name = "AUTO"
        self._input_ports = []
        self._output_ports = []

    def initialize_runtime(self) -> None:
        try:
            import openvino as ov

            self.ov = ov
            self.core = ov.Core()
        except (ImportError, AttributeError):
            try:
                from openvino.runtime import Core
            except ImportError as error:
                raise RuntimeError(
                    "openvino is not installed. Install OpenVINO to use this loader."
                ) from error

            self.core = Core()

    def set_providers(self) -> None:
        if self.config.providers:
            self.device_name = self.config.providers[0].upper()
        elif self.config.device == "auto":
            self.device_name = "AUTO"
        else:
            self.device_name = self.config.device.upper()

        self.providers = [self.device_name]

    def load_model(self) -> None:
        self.model = self.core.read_model(str(self.model_path))

        if self.config.input_shapes:
            reshape_map = {
                name: list(shape)
                for name, shape in self.config.input_shapes.items()
                if name
            }
            if "" in self.config.input_shapes and len(self.model.inputs) == 1:
                reshape_map[_port_name(self.model.inputs[0])] = list(
                    self.config.input_shapes[""]
                )
            if reshape_map:
                try:
                    self.model.reshape(reshape_map)
                except Exception as error:
                    raise RuntimeError(
                        f"Failed to reshape OpenVINO model inputs: {error}"
                    ) from error

        self.compiled_model = self.core.compile_model(
            self.model,
            self.device_name,
            self._compile_config(),
        )

    def read_model_io(self) -> None:
        self._input_ports = list(self.compiled_model.inputs)
        self._output_ports = list(self.compiled_model.outputs)
        self.input_specs = [
            self._port_to_spec(port, is_input=True) for port in self._input_ports
        ]
        self.output_specs = [
            self._port_to_spec(port, is_input=False) for port in self._output_ports
        ]

    def _port_to_spec(self, port, is_input: bool) -> TensorSpec:
        return TensorSpec(
            name=_port_name(port),
            shape=_port_shape(port),
            dtype=_openvino_dtype_name(_port_dtype(port)),
            is_input=is_input,
        )

    def read_model_metadata(self) -> dict[str, object]:
        metadata: dict[str, object] = {"device": self.device_name}
        try:
            runtime_model = self.compiled_model.get_runtime_model()
            metadata["runtime_model_name"] = runtime_model.get_friendly_name()
        except Exception:
            pass
        return metadata

    def _compile_config(self) -> dict[str, object]:
        config: dict[str, object] = {}
        extras = dict(self.config.extras or {})

        performance_hint = extras.get("performance_hint")
        if performance_hint:
            config["PERFORMANCE_HINT"] = str(performance_hint).upper()

        for option_key, openvino_key in {
            "num_streams": "NUM_STREAMS",
            "inference_num_threads": "INFERENCE_NUM_THREADS",
        }.items():
            if option_key in extras:
                config[openvino_key] = extras[option_key]

        return config


def _port_name(port) -> str:
    try:
        name = port.get_any_name()
        if name:
            return name
    except Exception:
        pass

    try:
        names = list(port.get_names())
        if names:
            return sorted(names)[0]
    except Exception:
        pass

    return str(port)


def _port_shape(port) -> tuple[int | str | None, ...]:
    try:
        partial_shape = port.get_partial_shape()
    except Exception:
        try:
            partial_shape = port.partial_shape
        except Exception:
            return ()

    try:
        return tuple(_dimension_value(dim) for dim in partial_shape)
    except TypeError:
        return ()


def _dimension_value(dim) -> int | str | None:
    try:
        if not dim.is_static:
            return None
        return int(dim.get_length())
    except Exception:
        pass

    try:
        value = int(dim)
        return value if value > 0 else None
    except Exception:
        text = str(dim)
        if text in {"?", "-1"} or ".." in text:
            return None
        return text


def _port_dtype(port) -> object:
    try:
        return port.get_element_type()
    except Exception:
        return getattr(port, "element_type", None)


def _openvino_dtype_name(dtype: object) -> str:
    if dtype is None:
        return "unknown"

    try:
        text = dtype.get_type_name().lower()
    except Exception:
        text = str(dtype).lower()

    dtype_map = {
        "boolean": "bool",
        "bf16": "bfloat16",
        "bool": "bool",
        "f16": "float16",
        "f32": "float32",
        "f64": "float64",
        "float16": "float16",
        "float32": "float32",
        "float64": "float64",
        "i8": "int8",
        "i16": "int16",
        "i32": "int32",
        "i64": "int64",
        "int8": "int8",
        "int16": "int16",
        "int32": "int32",
        "int64": "int64",
        "u8": "uint8",
        "u16": "uint16",
        "u32": "uint32",
        "u64": "uint64",
        "uint8": "uint8",
        "uint16": "uint16",
        "uint32": "uint32",
        "uint64": "uint64",
    }
    if text in dtype_map:
        return dtype_map[text]

    for needle, dtype_name in sorted(dtype_map.items(), key=lambda item: -len(item[0])):
        if needle in text:
            return dtype_name

    return text


def build_arg_parser():
    return build_common_arg_parser("Load and inspect an OpenVINO model.")


def main(argv: list[str] | None = None) -> int:
    return run_loader_cli(OpenVinoModelLoader, argv=argv, parser=build_arg_parser())


if __name__ == "__main__":
    raise SystemExit(main())
