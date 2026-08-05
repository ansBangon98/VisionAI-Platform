from __future__ import annotations

from collections.abc import Mapping

try:
    from .base_backend import (
        InferenceBackend,
        InputData,
        build_backend_arg_parser,
        run_backend_cli,
    )
except ImportError:  # Allows `python core/backends/pytorch_backend.py ...`.
    from pathlib import Path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from core.backends.base_backend import (  # type: ignore[no-redef]
        InferenceBackend,
        InputData,
        build_backend_arg_parser,
        run_backend_cli,
    )

from core.loaders.pytorch_loader import PyTorchModelLoader


class PyTorchBackend(InferenceBackend):
    backend_name = "PyTorch Backend"

    def infer(self, inputs: InputData) -> dict[str, object]:
        self.initialize()
        input_map = self._normalize_input_map(inputs)
        tensors = [
            self._to_tensor(input_map[spec.name], self._dtype_for_input(spec))
            for spec in self.input_specs
        ]

        with self.loader.torch.no_grad():
            result = self.loader.model(*tensors)

        return self._outputs_to_dict(result)

    def _to_tensor(self, value: object, dtype_name: str):
        torch = self.loader.torch
        if torch.is_tensor(value):
            tensor = value
        else:
            tensor = torch.as_tensor(value)

        dtype = self._torch_dtype(dtype_name)
        if dtype is not None and tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype)

        return tensor.to(self.loader.device)

    def _torch_dtype(self, dtype_name: str):
        torch = self.loader.torch
        text = str(dtype_name).lower().replace("torch.", "")
        dtype_map = {
            "bool": torch.bool,
            "float": torch.float32,
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
            "half": torch.float16,
            "int8": torch.int8,
            "int16": torch.int16,
            "int32": torch.int32,
            "int64": torch.int64,
            "long": torch.int64,
            "uint8": torch.uint8,
        }
        return dtype_map.get(text)

    def _outputs_to_dict(self, result: object) -> dict[str, object]:
        if isinstance(result, Mapping):
            return {
                str(name): self._to_cpu_array(value)
                for name, value in result.items()
            }

        if isinstance(result, (list, tuple)):
            return {
                f"output_{index}": self._to_cpu_array(value)
                for index, value in enumerate(result)
            }

        return {"output_0": self._to_cpu_array(result)}

    def _to_cpu_array(self, value: object) -> object:
        if self.loader.torch.is_tensor(value):
            return value.detach().cpu().numpy()
        return value


def build_arg_parser():
    parser = build_backend_arg_parser("Run inference with a loaded PyTorch model.")
    parser.add_argument(
        "--load-mode",
        choices=("auto", "torchscript", "pickle"),
        default="auto",
        help="How the PyTorch loader should load the file.",
    )
    parser.add_argument(
        "--default-input-dtype",
        default="float32",
        help="Default dtype used for PyTorch dummy inputs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    return run_backend_cli(
        PyTorchModelLoader,
        PyTorchBackend,
        argv=argv,
        parser=build_arg_parser(),
        backend_extra_names={"default_input_dtype"},
    )


if __name__ == "__main__":
    raise SystemExit(main())
