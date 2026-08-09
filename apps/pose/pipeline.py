from __future__ import annotations

from pathlib import Path


class PosePipeline:
    """Placeholder for future pose-estimation apps."""


def create_pipeline_from_config(config_path: str | Path) -> PosePipeline:
    raise RuntimeError(
        f"Pose pipeline is not implemented yet: {config_path}"
    )

