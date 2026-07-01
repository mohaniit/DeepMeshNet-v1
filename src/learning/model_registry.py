"""
Model registry for DeepMeshNet-v1.

This module centralizes learning-model registration and creation.
"""

from __future__ import annotations

from typing import Any, Callable


ModelFactory = Callable[..., Any]


class ModelRegistry:
    """
    Registry for model factory functions/classes.
    """

    def __init__(self) -> None:
        self._models: dict[str, ModelFactory] = {}

    def register(
        self,
        name: str,
        factory: ModelFactory,
        overwrite: bool = False,
    ) -> None:
        if name is None:
            raise ValueError("Model name cannot be None.")

        key = self._normalize_name(name)

        if factory is None:
            raise ValueError("Model factory cannot be None.")

        if key in self._models and not overwrite:
            raise ValueError(f"Model already registered: {name}")

        self._models[key] = factory

    def create(self, name: str, **kwargs: Any) -> Any:
        key = self._normalize_name(name)

        if key not in self._models:
            raise KeyError(f"Model not registered: {name}")

        return self._models[key](**kwargs)

    def has_model(self, name: str) -> bool:
        return self._normalize_name(name) in self._models

    def names(self) -> list[str]:
        return sorted(self._models.keys())

    def get_factory(self, name: str) -> ModelFactory:
        key = self._normalize_name(name)

        if key not in self._models:
            raise KeyError(f"Model not registered: {name}")

        return self._models[key]

    def clear(self) -> None:
        self._models.clear()

    def summary(self) -> dict[str, Any]:
        return {
            "num_models": len(self._models),
            "models": self.names(),
        }

    @staticmethod
    def _normalize_name(name: str) -> str:
        if name is None:
            raise ValueError("Model name cannot be None.")

        name = str(name).strip().lower()

        if not name:
            raise ValueError("Model name cannot be empty.")

        return name


_GLOBAL_REGISTRY = ModelRegistry()


def register_model(
    name: str,
    factory: ModelFactory,
    overwrite: bool = False,
) -> None:
    """
    Register model in global registry.
    """
    _GLOBAL_REGISTRY.register(
        name=name,
        factory=factory,
        overwrite=overwrite,
    )


def create_model(name: str, **kwargs: Any) -> Any:
    """
    Create model from global registry.
    """
    return _GLOBAL_REGISTRY.create(name, **kwargs)


def has_model(name: str) -> bool:
    """
    Check whether model exists in global registry.
    """
    return _GLOBAL_REGISTRY.has_model(name)


def registered_models() -> list[str]:
    """
    Return registered model names.
    """
    return _GLOBAL_REGISTRY.names()


def get_global_registry() -> ModelRegistry:
    """
    Return global model registry.
    """
    return _GLOBAL_REGISTRY


def clear_global_registry() -> None:
    """
    Clear global registry.

    Mainly useful for tests.
    """
    _GLOBAL_REGISTRY.clear()


__all__ = [
    "ModelFactory",
    "ModelRegistry",
    "register_model",
    "create_model",
    "has_model",
    "registered_models",
    "get_global_registry",
    "clear_global_registry",
]