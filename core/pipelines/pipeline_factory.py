from __future__ import annotations

from typing import Any

from core.pipelines.base_pipeline import BasePipeline
from core.pipelines.runtime_detection import deepstream_available, select_runtime


class PipelineFactory:
    @staticmethod
    def create(
        config: dict[str, Any],
        frame_processor: object | None = None,
    ) -> BasePipeline:
        mode, _backend = PipelineFactory.resolve_runtime(config)

        if mode == "deepstream":
            if not deepstream_available():
                raise RuntimeError(
                    "DeepStream mode was selected, but pyds or required "
                    "GStreamer DeepStream elements are unavailable."
                )
            try:
                from core.pipelines.deepstream.deepstream_pipeline import (
                    DeepStreamPipeline,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "DeepStream mode was selected, but its Python dependencies "
                    "are unavailable."
                ) from exc
            return DeepStreamPipeline(config)

        if mode in {"cpu", "cuda"}:
            from core.pipelines.cpu.gstreamer_pipeline import CPUGStreamerPipeline

            return CPUGStreamerPipeline(
                config,
                frame_processor=frame_processor,
                runtime_mode=mode,
            )

        raise ValueError(f"Unsupported runtime mode: {mode}")

    @staticmethod
    def resolve_mode(config: dict[str, Any]) -> str:
        mode, _backend = PipelineFactory.resolve_runtime(config)
        return mode

    @staticmethod
    def resolve_runtime(config: dict[str, Any]) -> tuple[str, str | None]:
        mode = _configured_mode(config)
        if mode == "auto":
            return select_runtime()
        return mode, _configured_inference_backend(config, mode)


def _configured_mode(config: dict[str, Any]) -> str:
    runtime = config.get("runtime", {})
    if isinstance(runtime, dict):
        raw_mode = runtime.get("mode", runtime.get("pipeline", "auto"))
    else:
        raw_mode = runtime or "auto"

    normalized = str(raw_mode or "auto").strip().lower().replace("-", "_")
    aliases = {
        "auto": "auto",
        "cpu": "cpu",
        "gstreamer": "cpu",
        "gstreamer_cpu": "cpu",
        "cuda": "cuda",
        "gpu": "cuda",
        "gstreamer_cuda": "cuda",
        "deepstream": "deepstream",
        "nvidia": "deepstream",
    }
    try:
        return aliases[normalized]
    except KeyError as error:
        supported = ", ".join(sorted(set(aliases.values())))
        raise ValueError(
            f"Unsupported runtime mode: {raw_mode}. Supported modes: {supported}."
        ) from error


def _configured_inference_backend(
    config: dict[str, Any],
    mode: str,
) -> str | None:
    runtime = config.get("runtime", {})
    if isinstance(runtime, dict) and runtime.get("inference_backend"):
        return str(runtime["inference_backend"])

    mode_config = config.get(mode, {})
    if isinstance(mode_config, dict) and mode_config.get("inference_backend"):
        return str(mode_config["inference_backend"])

    return None
