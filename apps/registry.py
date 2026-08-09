from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.config import load_yaml_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APPS_ROOT = PROJECT_ROOT / "apps"
APP_CONFIGS_ROOT = PROJECT_ROOT / "configs" / "apps"


@dataclass(frozen=True)
class AppDefinition:
    key: str
    name: str
    pipeline_type: str
    enabled: bool
    app_dir: Path
    config_path: Path
    config: dict[str, Any]


def discover_app_configs(configs_root: str | Path = APP_CONFIGS_ROOT) -> list[AppDefinition]:
    root = Path(configs_root)
    if not root.exists():
        return []

    apps: list[AppDefinition] = []
    for config_path in sorted(path for path in root.glob("*.yaml") if path.is_file()):
        key = config_path.stem
        if key.startswith("__"):
            continue

        config = load_yaml_config(config_path)
        application_config = config.get("application", {})
        pipeline_type = _pipeline_type_from_config(application_config, key)
        name = application_config.get("name") or _title_from_key(key)
        apps.append(
            AppDefinition(
                key=key,
                name=str(name),
                pipeline_type=pipeline_type,
                enabled=bool(application_config.get("enabled", True)),
                app_dir=APPS_ROOT / pipeline_type,
                config_path=config_path,
                config=config,
            )
        )

    return apps


def _title_from_key(key: str) -> str:
    return key.replace("_", " ").title()


def _pipeline_type_from_config(
    application_config: Any,
    fallback_key: str,
) -> str:
    if not isinstance(application_config, dict):
        return fallback_key

    pipeline_type = str(application_config.get("pipeline") or fallback_key).strip()
    if not pipeline_type:
        return fallback_key
    return pipeline_type.replace("-", "_").replace(" ", "_").lower()
