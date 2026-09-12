"""Decorator-based plugin registries for hale-vlm."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from hale_vlm.registry.base import NamedRegistry
from hale_vlm.registry.variant import VariantRegistry

T = TypeVar("T")

MODELS_REGISTRY = NamedRegistry("model")
LOSSES_REGISTRY = NamedRegistry("loss")
SAMPLERS_REGISTRY = NamedRegistry("sampler")
OPTIMIZERS_REGISTRY = NamedRegistry("optimizer")
CONFIGS_REGISTRY = NamedRegistry("config")
LOGGERS_REGISTRY = NamedRegistry("logger")
TRAINERS_REGISTRY = NamedRegistry("trainer")
DATASETS_REGISTRY = NamedRegistry("dataset")
METRICS_REGISTRY = VariantRegistry("metric")

MODELS: dict[str, type] = MODELS_REGISTRY.items
VARIANTS = MODELS
LOSSES: dict[str, Callable[..., Any]] = LOSSES_REGISTRY.items
SAMPLERS: dict[str, Callable[..., Any]] = SAMPLERS_REGISTRY.items
OPTIMIZERS: dict[str, type] = OPTIMIZERS_REGISTRY.items
CONFIGS: dict[str, type] = CONFIGS_REGISTRY.items
LOGGERS: dict[str, Callable[..., Any]] = LOGGERS_REGISTRY.items
TRAINERS: dict[str, type] = TRAINERS_REGISTRY.items
DATASETS: dict[str, Callable[..., Any]] = DATASETS_REGISTRY.items
METRICS = METRICS_REGISTRY.items


def register_model(name: str) -> Callable[[type[T]], type[T]]:
    def deco(cls: type[T]) -> type[T]:
        MODELS_REGISTRY.add(name, cls)
        cls.variant_name = name  # type: ignore[attr-defined]
        return cls

    return deco


register_variant = register_model


def register_loss(name: str) -> Callable[[T], T]:
    def deco(fn: T) -> T:
        LOSSES_REGISTRY.add(name, fn)
        return fn

    return deco


def register_optimizer(name: str) -> Callable[[type[T]], type[T]]:
    def deco(cls: type[T]) -> type[T]:
        OPTIMIZERS_REGISTRY.add(name.lower(), cls)
        return cls

    return deco


def register_config(name: str) -> Callable[[type[T]], type[T]]:
    def deco(cls: type[T]) -> type[T]:
        CONFIGS_REGISTRY.add(name, cls)
        return cls

    return deco


def register_logger(name: str) -> Callable[[T], T]:
    def deco(factory: T) -> T:
        LOGGERS_REGISTRY.add(name, factory)
        return factory

    return deco


def register_trainer(name: str) -> Callable[[type[T]], type[T]]:
    def deco(cls: type[T]) -> type[T]:
        TRAINERS_REGISTRY.add(name, cls)
        return cls

    return deco


def register_dataset(name: str) -> Callable[[T], T]:
    def deco(factory: T) -> T:
        DATASETS_REGISTRY.add(name, factory)
        return factory

    return deco


def get_model(name: str) -> type:
    return MODELS_REGISTRY.get(name)


get_variant = get_model


def get_loss(name: str) -> Callable[..., Any]:
    return LOSSES_REGISTRY.get(name)


def get_optimizer(name: str) -> type:
    return OPTIMIZERS_REGISTRY.get(name.lower())


def get_config_schema(name: str = "vlm") -> type:
    return CONFIGS_REGISTRY.get(name)


def get_trainer(name: str = "default") -> type:
    return TRAINERS_REGISTRY.get(name)


def build_logger(name: str, *, cfg: Any = None, **kwargs: Any) -> Any:
    return LOGGERS_REGISTRY.get(name)(cfg=cfg, **kwargs)


def get_dataset(name: str) -> Any:
    return DATASETS_REGISTRY.get(name)()


def list_datasets(**kwargs) -> list[str]:
    del kwargs
    return sorted(DATASETS_REGISTRY.items)
