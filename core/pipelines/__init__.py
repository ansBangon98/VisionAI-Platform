from .base_pipeline import BasePipeline, ResultCallback

__all__ = ["BasePipeline", "PipelineFactory", "ResultCallback"]


def __getattr__(name: str):
    if name == "PipelineFactory":
        from .pipeline_factory import PipelineFactory

        return PipelineFactory
    raise AttributeError(name)
