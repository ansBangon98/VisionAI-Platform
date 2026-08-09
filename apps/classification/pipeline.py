from __future__ import annotations

from pathlib import Path


class ClassificationPipeline:
    """Placeholder for future full-frame or crop classification apps."""


def create_pipeline_from_config(config_path: str | Path) -> ClassificationPipeline:
    raise RuntimeError(
        f"Classification pipeline is not implemented yet: {config_path}"
    )

