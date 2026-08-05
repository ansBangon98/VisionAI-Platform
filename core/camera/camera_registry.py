from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.config import load_yaml_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAMERAS_CONFIG = PROJECT_ROOT / "configs" / "cameras.yaml"
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"


@dataclass(frozen=True)
class CameraSourceDefinition:
    key: str
    label: str
    config: dict[str, Any]


class CameraRegistry:
    def __init__(
        self,
        config_path: str | Path = DEFAULT_CAMERAS_CONFIG,
        env_path: str | Path = DEFAULT_ENV_PATH,
    ):
        self.config_path = Path(config_path)
        self.env_path = Path(env_path)
        self._env = load_env_file(self.env_path)
        self._sources = self._load_sources()

    def list_sources(self) -> list[CameraSourceDefinition]:
        return [
            CameraSourceDefinition(
                key=key,
                label=str(config.get("label") or _title_from_key(key)),
                config=dict(config),
            )
            for key, config in sorted(self._sources.items())
        ]

    def get(self, name: str) -> dict[str, Any]:
        if name not in self._sources:
            available = ", ".join(sorted(self._sources)) or "none"
            raise RuntimeError(
                f"Camera source '{name}' is not defined. Available sources: {available}."
            )

        raw_config = dict(self._sources[name])
        raw_config["name"] = name
        return self._resolve_sensitive_values(raw_config)

    def _load_sources(self) -> dict[str, dict[str, Any]]:
        if not self.config_path.exists():
            return {}

        data = load_yaml_config(self.config_path)
        sources = data.get("sources", {})
        if not isinstance(sources, dict):
            raise RuntimeError(f"Invalid camera sources config: {self.config_path}")

        return {
            str(name): dict(config)
            for name, config in sources.items()
            if isinstance(config, dict)
        }

    def _resolve_sensitive_values(self, config: dict[str, Any]) -> dict[str, Any]:
        uri_env = config.get("uri_env")
        if uri_env and not config.get("uri"):
            uri_env_name = str(uri_env)
            if "://" in uri_env_name:
                raise RuntimeError(
                    f"Camera source '{config['name']}' has a URL in uri_env. "
                    "Use uri_env for the environment variable name only, then put "
                    "the real RTSP URL in .env."
                )

            value = os.environ.get(uri_env_name, self._env.get(uri_env_name, ""))
            if not value:
                raise RuntimeError(
                    f"Camera source '{config['name']}' requires env var {uri_env_name}. "
                    "Add it to .env or export it in the shell."
                )
            config["uri"] = value

        return config


def load_env_file(env_path: str | Path = DEFAULT_ENV_PATH) -> dict[str, str]:
    path = Path(env_path)
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        values[key.strip()] = _unquote_env_value(value.strip())

    return values


def _unquote_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _title_from_key(key: str) -> str:
    return key.replace("_", " ").title()
