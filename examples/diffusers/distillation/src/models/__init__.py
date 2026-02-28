"""Model backend registry.

Maps model names to their (ModelLoader, strategy_factory, InferencePipeline) tuple.
Strategy is returned as a factory callable because it may need to be initialized
with model-specific components after the model is loaded.

Usage:
    loader, create_strategy, pipeline_cls = get_model_backend("wan")
"""

from __future__ import annotations

from typing import Any, Callable

from ..interfaces import InferencePipeline, ModelLoader, TrainingStrategy

BackendTuple = tuple[ModelLoader, Callable[..., TrainingStrategy], type[InferencePipeline] | None]

_REGISTRY: dict[str, Callable[[], BackendTuple]] = {}


def register_backend(name: str, factory: Callable[[], BackendTuple]) -> None:
    _REGISTRY[name] = factory


def get_model_backend(name: str) -> BackendTuple:
    if name not in _REGISTRY:
        available = ", ".join(_REGISTRY.keys()) or "(none)"
        raise ValueError(f"Unknown model backend '{name}'. Available: {available}")
    return _REGISTRY[name]()


def _register_builtin_backends() -> None:
    def _wan_backend() -> BackendTuple:
        from .wan import WanInferencePipeline, WanModelLoader, create_wan_strategy

        return WanModelLoader(), create_wan_strategy, WanInferencePipeline

    register_backend("wan", _wan_backend)

    # LTX-2 backend (requires ltx-core)
    def _ltx2_backend() -> BackendTuple:
        from .ltx2 import LTX2InferencePipeline, LTX2ModelLoader, create_ltx2_strategy

        return LTX2ModelLoader(), create_ltx2_strategy, LTX2InferencePipeline

    register_backend("ltx2", _ltx2_backend)


_register_builtin_backends()
