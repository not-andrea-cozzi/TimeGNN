from __future__ import annotations

from typing import Callable, Dict, Type

from .base import BaseModel


class ModelRegistry:
    """Registry for model classes keyed by name."""
    def __init__(self) -> None:
        self._models: Dict[str, Type[BaseModel]] = {}

    def register(self, name: str) -> Callable[[Type[BaseModel]], Type[BaseModel]]:
        """Decorator to register a model class by name."""
        def decorator(cls: Type[BaseModel]) -> Type[BaseModel]:
            self._models[name] = cls
            return cls

        return decorator

    def get(self, name: str) -> Type[BaseModel]:
        """Retrieve a registered model class by name."""
        if name not in self._models:
            raise KeyError(f"Model '{name}' is not registered.")
        return self._models[name]

    def available(self) -> Dict[str, Type[BaseModel]]:
        """Return a copy of the registered model mapping."""
        return dict(self._models)


model_registry = ModelRegistry()
