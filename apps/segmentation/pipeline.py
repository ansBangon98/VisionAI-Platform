from __future__ import annotations

from pathlib import Path


class SegmentationPipeline:
    """Placeholder for future segmentation apps."""


def create_pipeline_from_config(config_path: str | Path) -> SegmentationPipeline:
    raise RuntimeError(
        f"Segmentation pipeline is not implemented yet: {config_path}"
    )

