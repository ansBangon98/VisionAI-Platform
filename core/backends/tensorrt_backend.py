from __future__ import annotations

try:
    from .base_backend import (
        InferenceBackend,
        InputData,
        build_backend_arg_parser,
        numpy_dtype_from_name,
        require_numpy,
        run_backend_cli,
    )
except ImportError:  # Allows `python core/backends/tensorrt_backend.py ...`.
    from pathlib import Path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from core.backends.base_backend import (  # type: ignore[no-redef]
        InferenceBackend,
        InputData,
        build_backend_arg_parser,
        numpy_dtype_from_name,
        require_numpy,
        run_backend_cli,
    )

from core.loaders.tensorrt_loader import TensorRtModelLoader


class TensorRtBackend(InferenceBackend):
    backend_name = "TensorRT Backend"

    def __init__(self, loader, config=None):
        super().__init__(loader, config)
        self.context = None
        self.cuda = None
        self.cuda_context = None

    def initialize_backend(self) -> None:
        self.context = self.loader.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context.")

    def infer(self, inputs: InputData) -> dict[str, object]:
        self.initialize()
        self._ensure_cuda_driver()

        input_map = self._normalize_input_map(inputs)
        self.cuda_context.push()
        try:
            if self._uses_tensor_io_api():
                return self._infer_tensor_io(input_map)
            return self._infer_binding_io(input_map)
        finally:
            self.cuda_context.pop()

    def close(self) -> None:
        self.context = None
        if self.cuda_context is not None:
            self.cuda_context.detach()
            self.cuda_context = None

    def _infer_tensor_io(self, input_map: dict[str, object]) -> dict[str, object]:
        numpy = require_numpy()
        stream = self.cuda.Stream()
        device_allocations = []
        output_buffers: dict[str, tuple[object, object]] = {}

        for spec in self.input_specs:
            host_array = numpy.ascontiguousarray(
                input_map[spec.name],
                dtype=numpy_dtype_from_name(self._dtype_for_input(spec)),
            )
            if not self.context.set_input_shape(spec.name, tuple(host_array.shape)):
                raise RuntimeError(
                    f"TensorRT rejected input shape for '{spec.name}': "
                    f"{tuple(host_array.shape)}"
                )
            device_memory = self.cuda.mem_alloc(host_array.nbytes)
            device_allocations.append(device_memory)
            self.cuda.memcpy_htod_async(device_memory, host_array, stream)
            self.context.set_tensor_address(spec.name, int(device_memory))

        for spec in self.output_specs:
            shape = tuple(int(dim) for dim in self.context.get_tensor_shape(spec.name))
            if any(dim < 0 for dim in shape):
                raise RuntimeError(
                    f"TensorRT output '{spec.name}' still has a dynamic shape. "
                    "Provide concrete --input-shape values."
                )
            host_array = numpy.empty(shape, dtype=numpy_dtype_from_name(spec.dtype))
            device_memory = self.cuda.mem_alloc(host_array.nbytes)
            device_allocations.append(device_memory)
            self.context.set_tensor_address(spec.name, int(device_memory))
            output_buffers[spec.name] = (host_array, device_memory)

        try:
            success = self.context.execute_async_v3(stream.handle)
        except TypeError:
            success = self.context.execute_async_v3(stream_handle=stream.handle)
        if not success:
            raise RuntimeError("TensorRT execution failed.")

        outputs: dict[str, object] = {}
        for name, (host_array, device_memory) in output_buffers.items():
            self.cuda.memcpy_dtoh_async(host_array, device_memory, stream)
            outputs[name] = host_array

        stream.synchronize()
        return outputs

    def _infer_binding_io(self, input_map: dict[str, object]) -> dict[str, object]:
        numpy = require_numpy()
        stream = self.cuda.Stream()
        bindings = [0] * self.loader.engine.num_bindings
        device_allocations = []
        output_buffers: dict[str, tuple[object, object]] = {}

        for index in range(self.loader.engine.num_bindings):
            name = self.loader.engine.get_binding_name(index)
            dtype = numpy_dtype_from_name(
                self.loader._dtype_name(self.loader.engine.get_binding_dtype(index))
            )

            if self.loader.engine.binding_is_input(index):
                host_array = numpy.ascontiguousarray(input_map[name], dtype=dtype)
                if self._binding_shape_is_dynamic(index):
                    if not self.context.set_binding_shape(index, tuple(host_array.shape)):
                        raise RuntimeError(
                            f"TensorRT rejected input shape for '{name}': "
                            f"{tuple(host_array.shape)}"
                        )
                device_memory = self.cuda.mem_alloc(host_array.nbytes)
                device_allocations.append(device_memory)
                self.cuda.memcpy_htod_async(device_memory, host_array, stream)
                bindings[index] = int(device_memory)
            else:
                shape = tuple(int(dim) for dim in self.context.get_binding_shape(index))
                if any(dim < 0 for dim in shape):
                    raise RuntimeError(
                        f"TensorRT output '{name}' still has a dynamic shape. "
                        "Provide concrete --input-shape values."
                    )
                host_array = numpy.empty(shape, dtype=dtype)
                device_memory = self.cuda.mem_alloc(host_array.nbytes)
                device_allocations.append(device_memory)
                bindings[index] = int(device_memory)
                output_buffers[name] = (host_array, device_memory)

        try:
            success = self.context.execute_async_v2(bindings, stream.handle)
        except TypeError:
            success = self.context.execute_async_v2(
                bindings=bindings,
                stream_handle=stream.handle,
            )
        if not success:
            raise RuntimeError("TensorRT execution failed.")

        outputs: dict[str, object] = {}
        for name, (host_array, device_memory) in output_buffers.items():
            self.cuda.memcpy_dtoh_async(host_array, device_memory, stream)
            outputs[name] = host_array

        stream.synchronize()
        return outputs

    def _ensure_cuda_driver(self) -> None:
        if self.cuda is not None:
            return

        try:
            import pycuda.driver as cuda
        except ImportError as error:
            raise RuntimeError(
                "TensorRT inference requires pycuda. Install pycuda to use "
                "the TensorRT backend."
            ) from error

        cuda.init()
        self.cuda = cuda
        self.cuda_context = cuda.Device(self.loader.device_index).make_context()
        self.cuda_context.pop()

    def _uses_tensor_io_api(self) -> bool:
        return hasattr(self.loader.engine, "num_io_tensors")

    def _binding_shape_is_dynamic(self, index: int) -> bool:
        return any(int(dim) < 0 for dim in self.loader.engine.get_binding_shape(index))


def build_arg_parser():
    return build_backend_arg_parser("Run inference with a loaded TensorRT engine.")


def main(argv: list[str] | None = None) -> int:
    return run_backend_cli(
        TensorRtModelLoader,
        TensorRtBackend,
        argv=argv,
        parser=build_arg_parser(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
