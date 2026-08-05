"""People analytics application package."""

from .analytics import PeopleAnalytics
from .pipeline import (
    PeopleAnalyticsPipeline,
    available_detector_backends,
    create_pipeline_from_config,
    load_config,
)

__all__ = [
    "PeopleAnalytics",
    "PeopleAnalyticsPipeline",
    "available_detector_backends",
    "create_pipeline_from_config",
    "load_config",
]
