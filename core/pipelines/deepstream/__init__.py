__all__ = ["DeepStreamPipeline"]


def __getattr__(name: str):
    if name == "DeepStreamPipeline":
        from .deepstream_pipeline import DeepStreamPipeline

        return DeepStreamPipeline
    raise AttributeError(name)
