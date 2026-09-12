"""One name may map to several implementations, resolved by variant at runtime."""

from __future__ import annotations

from typing import Any

from loguru import logger

VariantKey = frozenset[str] | None


class VariantRegistry:
    """Register multiple implementations under one name, scoped by variant.

    Resolution order for ``resolve(name, variant)``:
      1. entry whose variant set contains ``variant``
      2. entry registered with ``variants=None`` (generic fallback)
      3. skip (return ``None``) when nothing matches
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.items: dict[str, list[tuple[VariantKey, type]]] = {}

    def add(
        self,
        name: str,
        cls: type,
        *,
        variants: tuple[str, ...] | None = None,
        attr: str | None = "metric_name",
    ) -> type:
        key: VariantKey = frozenset(variants) if variants is not None else None
        entries = self.items.setdefault(name, [])
        for existing, _ in entries:
            if existing == key:
                raise ValueError(f"{self.kind} {name!r} already registered for variants={variants}")
        entries.append((key, cls))
        if attr is not None:
            setattr(cls, attr, name)
        logger.debug("registered {} {} variants={}", self.kind, name, variants)
        return cls

    def resolve(self, name: str, variant: str) -> type | None:
        entries = self.items.get(name)
        if entries is None:
            logger.error("unknown {} {!r}; registered={}", self.kind, name, sorted(self.items))
            raise KeyError(f"unknown {self.kind} {name!r}; registered={sorted(self.items)}")
        specific = [
            cls for variants, cls in entries if variants is not None and variant in variants
        ]
        generic = [cls for variants, cls in entries if variants is None]
        return (specific or generic or [None])[0]

    def instantiate(self, names: list[str], variant: str) -> list[Any]:
        built: list[Any] = []
        for name in names:
            cls = self.resolve(name, variant)
            if cls is None:
                logger.info("skipping {} {} (not defined for variant {})", self.kind, name, variant)
                continue
            built.append(cls())
        return built
