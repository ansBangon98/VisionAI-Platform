from __future__ import annotations

try:
    from .base_backend import (
        InferenceBackend,
        InputData,
        build_backend_arg_parser,
        run_backend_cli,
    )
except ImportError:  # Allows `python core/backends/openvino_backend.py ...`.
    from pathlib import Path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from core.backends.base_backend import (  # type: ignore[no-redef]
        InferenceBackend,
        InputData,
        build_backend_arg_parser,
        run_backend_cli,
    )

from core.loaders.openvino_loader import OpenVinoModelLoader


class OpenVinoBackend(InferenceBackend):
    backend_name = "OpenVINO Backend"

    def infer(self, inputs: InputData) -> dict[str, object]:
        self.initialize()
        input_map = self._normalize_input_map(inputs)
        results = self.loader.compiled_model(input_map)

        outputs: dict[str, object] = {}
        for spec, port in zip(self.output_specs, self.loader._output_ports):
            try:
                outputs[spec.name] = results[port]
            except KeyError:
                outputs[spec.name] = results[spec.name]
        return outputs


def build_arg_parser():
    return build_backend_arg_parser("Run inference with a loaded OpenVINO model.")


def main(argv: list[str] | None = None) -> int:
    return run_backend_cli(
        OpenVinoModelLoader,
        OpenVinoBackend,
        argv=argv,
        parser=build_arg_parser(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
