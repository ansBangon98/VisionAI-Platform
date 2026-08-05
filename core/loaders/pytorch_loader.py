from __future__ import annotations

import inspect
from collections.abc import Mapping

try:
    from .base_loader import (
        ModelLoader,
        TensorSpec,
        build_common_arg_parser,
        run_loader_cli,
    )
except ImportError:  # Allows `python core/loaders/pytorch_loader.py ...`.
    from base_loader import (  # type: ignore[no-redef]
        ModelLoader,
        TensorSpec,
        build_common_arg_parser,
        run_loader_cli,
    )


class PyTorchModelLoader(ModelLoader):
    runtime_name = "PyTorch"

    def __init__(self, config):
        super().__init__(config)
        self.torch = None
        self.device = None
        self.model = None
        self.load_mode = "auto"

    def initialize_runtime(self) -> None:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError(
                "torch is not installed. Install PyTorch to use this loader."
            ) from error

        self.torch = torch

    def set_providers(self) -> None:
        requested = self.config.providers[0] if self.config.providers else self.config.device
        device = self._select_device(requested)
        self.device = self.torch.device(device)
        self.providers = [str(self.device)]

    def load_model(self) -> None:
        mode = self.config.extras.get("load_mode", "auto")

        if mode == "torchscript":
            self.model = self._load_torchscript()
            self.load_mode = "torchscript"
        elif mode == "pickle":
            self.model = self._load_pickle()
            self.load_mode = "pickle"
        else:
            try:
                self.model = self._load_torchscript()
                self.load_mode = "torchscript"
            except Exception:
                self.model = self._load_pickle()
                self.load_mode = "pickle"

        if isinstance(self.model, Mapping):
            raise RuntimeError(
                "This file looks like a state_dict/checkpoint dictionary. "
                "Load it into its model class first or export a TorchScript model."
            )

        if hasattr(self.model, "eval"):
            self.model.eval()

        if hasattr(self.model, "to"):
            self.model.to(self.device)

    def read_model_io(self) -> None:
        input_names = self._input_names_from_model()
        if not input_names:
            input_names = [name for name in self.config.input_shapes if name]
        if not input_names:
            input_names = ["input_0"]

        self.input_specs = [
            TensorSpec(
                name=name,
                shape=self._configured_input_shape(name, len(input_names)),
                dtype="unknown",
                is_input=True,
            )
            for name in input_names
        ]

        output_names = self._output_names_from_model()
        if not output_names:
            output_names = ["output_0"]

        self.output_specs = [
            TensorSpec(name=name, dtype="unknown", is_input=False)
            for name in output_names
        ]

    def read_model_metadata(self) -> dict[str, object]:
        return {
            "device": str(self.device),
            "load_mode": self.load_mode,
            "model_type": type(self.model).__name__,
        }

    def _load_torchscript(self):
        return self.torch.jit.load(str(self.model_path), map_location=self.device)

    def _load_pickle(self):
        kwargs = {"map_location": self.device}
        try:
            if "weights_only" in inspect.signature(self.torch.load).parameters:
                kwargs["weights_only"] = False
        except (TypeError, ValueError):
            pass
        return self.torch.load(str(self.model_path), **kwargs)

    def _select_device(self, requested: str) -> str:
        requested = requested.lower()
        if requested == "auto":
            if self.torch.cuda.is_available():
                return "cuda:0"
            if hasattr(self.torch.backends, "mps") and self.torch.backends.mps.is_available():
                return "mps"
            return "cpu"

        if requested == "gpu":
            requested = "cuda:0"
        elif requested == "cuda":
            requested = "cuda:0"

        if requested.startswith("cuda") and not self.torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but PyTorch cannot access CUDA.")

        if requested == "mps":
            if not hasattr(self.torch.backends, "mps"):
                raise RuntimeError("MPS was requested, but this PyTorch build has no MPS backend.")
            if not self.torch.backends.mps.is_available():
                raise RuntimeError("MPS was requested, but it is not available.")

        return requested

    def _configured_input_shape(
        self,
        input_name: str,
        input_count: int,
    ) -> tuple[int, ...]:
        if input_name in self.config.input_shapes:
            return self.config.input_shapes[input_name]
        if "" in self.config.input_shapes and input_count == 1:
            return self.config.input_shapes[""]
        return ()

    def _input_names_from_model(self) -> list[str]:
        schema = self._forward_schema()
        if schema is None:
            return []

        names: list[str] = []
        for argument in getattr(schema, "arguments", []):
            name = getattr(argument, "name", "")
            if name and name != "self":
                names.append(name)
        return names

    def _output_names_from_model(self) -> list[str]:
        schema = self._forward_schema()
        if schema is None:
            return []

        names: list[str] = []
        for index, value in enumerate(getattr(schema, "returns", [])):
            name = getattr(value, "name", "") or f"output_{index}"
            names.append(name)
        return names

    def _forward_schema(self):
        try:
            return self.model.forward.schema
        except AttributeError:
            return None


def build_arg_parser():
    parser = build_common_arg_parser("Load and inspect a PyTorch model.")
    parser.add_argument(
        "--load-mode",
        choices=("auto", "torchscript", "pickle"),
        default="auto",
        help="How to load the PyTorch file.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    return run_loader_cli(PyTorchModelLoader, argv=argv, parser=build_arg_parser())


if __name__ == "__main__":
    raise SystemExit(main())
