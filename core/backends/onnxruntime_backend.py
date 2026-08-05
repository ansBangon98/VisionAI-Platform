from __future__ import annotations

try:
    from .base_backend import (
        InferenceBackend,
        InputData,
        build_backend_arg_parser,
        run_backend_cli,
    )
except ImportError:  # Allows `python core/backends/onnxruntime_backend.py ...`.
    from pathlib import Path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from core.backends.base_backend import (  # type: ignore[no-redef]
        InferenceBackend,
        InputData,
        build_backend_arg_parser,
        run_backend_cli,
    )

from core.loaders.onnx_loader import OnnxModelLoader


class OnnxRuntimeBackend(InferenceBackend):
    backend_name = "ONNX Runtime Backend"

    def infer(self, inputs: InputData) -> dict[str, object]:
        self.initialize()
        input_map = self._normalize_input_map(inputs)
        output_names = [spec.name for spec in self.output_specs]
        outputs = self.loader.session.run(output_names, input_map)
        return dict(zip(output_names, outputs))


def build_arg_parser():
    return build_backend_arg_parser("Run inference with a loaded ONNX Runtime model.")


def main(argv: list[str] | None = None) -> int:
    return run_backend_cli(
        OnnxModelLoader,
        OnnxRuntimeBackend,
        argv=argv,
        parser=build_arg_parser(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
