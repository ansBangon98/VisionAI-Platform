__all__ = ["CPUGStreamerPipeline"]


def __getattr__(name: str):
    if name == "CPUGStreamerPipeline":
        from .gstreamer_pipeline import CPUGStreamerPipeline

        return CPUGStreamerPipeline
    raise AttributeError(name)
