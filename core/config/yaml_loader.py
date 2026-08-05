from __future__ import annotations

from pathlib import Path
from typing import Any


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    try:
        import yaml
    except ImportError:
        return _load_simple_yaml(path)

    with path.open("r", encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid config: {path}")
    return data


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key, separator, value = raw_line.strip().partition(":")
        if not separator:
            continue

        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        value = value.strip()

        if value:
            parent[key] = _parse_yaml_scalar(value)
            continue

        child: dict[str, Any] = {}
        parent[key] = child
        stack.append((indent, child))

    return root


def _parse_yaml_scalar(value: str) -> Any:
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None"}:
        return None

    if value.startswith("[") and value.endswith("]"):
        items = [item.strip() for item in value[1:-1].split(",") if item.strip()]
        return [_parse_yaml_scalar(item) for item in items]

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        return value
