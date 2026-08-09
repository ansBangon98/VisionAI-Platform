from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SecondaryModelConfig:
    name: str
    task: str
    enabled: bool
    model_path: str
    backend: str
    operate_on_classes: tuple[int, ...]
    raw_config: dict[str, Any]

    def applies_to(self, class_id: int) -> bool:
        return not self.operate_on_classes or int(class_id) in self.operate_on_classes


class SecondaryModelManager:
    """Configuration-level manager for SGIE-style secondary models.

    The first version only normalizes config and answers applicability. Runtime
    model loading can plug into this without changing the YAML shape.
    """

    def __init__(self, configs: Sequence[SecondaryModelConfig]):
        self.configs = tuple(configs)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "SecondaryModelManager":
        secondary_models = config.get("secondary_models", {})
        if not isinstance(secondary_models, Mapping):
            return cls(())

        return cls(
            tuple(
                _secondary_model_config(name, value)
                for name, value in secondary_models.items()
                if isinstance(value, Mapping)
            )
        )

    def enabled(self) -> tuple[SecondaryModelConfig, ...]:
        return tuple(config for config in self.configs if config.enabled)

    def applicable_to(self, class_id: int) -> Iterator[SecondaryModelConfig]:
        for config in self.enabled():
            if config.applies_to(class_id):
                yield config


def _secondary_model_config(
    name: object,
    raw_config: Mapping[str, Any],
) -> SecondaryModelConfig:
    return SecondaryModelConfig(
        name=str(name),
        task=str(raw_config.get("task") or "classification"),
        enabled=bool(raw_config.get("enabled", False)),
        model_path=str(raw_config.get("model") or ""),
        backend=str(raw_config.get("backend") or "onnxruntime"),
        operate_on_classes=_as_int_tuple(raw_config.get("operate_on_classes", ())),
        raw_config=dict(raw_config),
    )


def _as_int_tuple(value: object) -> tuple[int, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        return tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if isinstance(value, Sequence):
        return tuple(int(part) for part in value)
    return (int(value),)

