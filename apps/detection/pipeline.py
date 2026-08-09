from __future__ import annotations

from apps.people_analytics.pipeline import (
    PeopleAnalyticsPipeline,
    create_pipeline_from_config,
    load_config,
)


DetectionPipeline = PeopleAnalyticsPipeline

__all__ = [
    "DetectionPipeline",
    "create_pipeline_from_config",
    "load_config",
]

