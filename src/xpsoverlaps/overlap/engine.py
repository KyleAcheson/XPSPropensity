"""AO overlaps computed by an electronic-structure engine's own integral library."""

from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np

from xpsoverlaps.model import EngineBasis
from xpsoverlaps.overlap.base import AOOverlapProvider

# Engine name -> function(handle_a, handle_b) returning the cross AO overlap.
EngineOverlapFunction = Callable[[Any, Any], np.ndarray]
_ENGINES: Dict[str, EngineOverlapFunction] = {}


def register_engine(name: str, cross_overlap: EngineOverlapFunction) -> None:
    """Register an engine backend, e.g. from ``xpsoverlaps.overlap.engines.<name>``.

    Engine libraries should be imported only inside the backend module, so
    users of other programs do not need them installed.
    """
    _ENGINES[name] = cross_overlap


class EngineOverlapProvider(AOOverlapProvider):
    """AO overlaps for ``EngineBasis`` representations via a registered backend."""

    basis_type = EngineBasis

    def _compute(self, basis_a: EngineBasis, basis_b: EngineBasis) -> np.ndarray:
        if basis_a.engine != basis_b.engine:
            raise ValueError(
                f"Cannot compute overlaps between engines {basis_a.engine!r} and {basis_b.engine!r}."
            )
        if basis_a.engine not in _ENGINES:
            raise NotImplementedError(f"No overlap backend registered for engine {basis_a.engine!r}.")
        return _ENGINES[basis_a.engine](basis_a.handle, basis_b.handle)
