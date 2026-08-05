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
        app_dir = APPS_ROOT / key
        if key.startswith("__"):
            continue

        config = load_yaml_config(config_path)
        application_config = config.get("application", {})
        name = application_config.get("name") or _title_from_key(key)
        apps.append(
            AppDefinition(
                key=key,
                name=str(name),
                enabled=bool(application_config.get("enabled", True)),
                app_dir=app_dir,
                config_path=config_path,
                config=config,
            )
        )

    return apps


def _title_from_key(key: str) -> str:
    return key.replace("_", " ").title()
