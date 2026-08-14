from __future__ import annotations

from pathlib import Path
from typing import Any


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise RuntimeError(f"Config file does not exist: {path}")
    if not path.is_file():
        raise RuntimeError(f"Config path is not a file: {path}")

    try:
        import yaml
    except ImportError:
        return _load_simple_yaml(path)

    try:
        with path.open("r", encoding="utf-8") as config_file:
            data = yaml.safe_load(config_file) or {}
    except RuntimeError:
        raise
    except Exception as error:
        raise RuntimeError(f"Failed to load YAML config {path}: {error}") from error

    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid config: {path}")
    return data


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    lines: list[tuple[int, str]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        lines.append((indent, raw_line.strip()))

    if not lines:
        return {}

    parsed, _ = _parse_yaml_block(lines, 0, lines[0][0])
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Invalid config: {path}")
    return parsed


def _parse_yaml_block(
    lines: list[tuple[int, str]],
    index: int,
    indent: int,
) -> tuple[Any, int]:
    if index >= len(lines):
        return {}, index

    if lines[index][1].startswith("- "):
        return _parse_yaml_list(lines, index, indent)
    return _parse_yaml_dict(lines, index, indent)


def _parse_yaml_dict(
    lines: list[tuple[int, str]],
    index: int,
    indent: int,
) -> tuple[dict[str, Any], int]:
    data: dict[str, Any] = {}

    while index < len(lines):
        line_indent, text = lines[index]
        if line_indent < indent:
            break
        if line_indent > indent or text.startswith("- "):
            break

        key, separator, value = text.partition(":")
        if not separator:
            index += 1
            continue

        key = key.strip()
        value = value.strip()
        index += 1
        if value:
            data[key] = _parse_yaml_scalar(value)
            continue

        if index >= len(lines) or lines[index][0] <= line_indent:
            data[key] = {}
            continue

        child, index = _parse_yaml_block(lines, index, lines[index][0])
        data[key] = child

    return data, index


def _parse_yaml_list(
    lines: list[tuple[int, str]],
    index: int,
    indent: int,
) -> tuple[list[Any], int]:
    data: list[Any] = []

    while index < len(lines):
        line_indent, text = lines[index]
        if line_indent < indent:
            break
        if line_indent > indent or not text.startswith("- "):
            break

        item_text = text[2:].strip()
        index += 1
        if not item_text:
            if index < len(lines) and lines[index][0] > line_indent:
                child, index = _parse_yaml_block(lines, index, lines[index][0])
                data.append(child)
            else:
                data.append(None)
            continue

        key, separator, value = item_text.partition(":")
        if separator and key.strip():
            item: dict[str, Any] = {}
            if value.strip():
                item[key.strip()] = _parse_yaml_scalar(value.strip())
            elif index < len(lines) and lines[index][0] > line_indent:
                child, index = _parse_yaml_block(lines, index, lines[index][0])
                item[key.strip()] = child
                data.append(item)
                continue
            else:
                item[key.strip()] = {}

            if index < len(lines) and lines[index][0] > line_indent:
                child, index = _parse_yaml_block(lines, index, lines[index][0])
                if isinstance(child, dict):
                    item.update(child)
            data.append(item)
            continue

        data.append(_parse_yaml_scalar(item_text))

    return data, index


def _parse_yaml_scalar(value: str) -> Any:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]

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
