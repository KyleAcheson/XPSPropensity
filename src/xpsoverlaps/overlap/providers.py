"""Selection of the AO overlap provider for a pair of basis representations."""

from __future__ import annotations

from typing import Dict, Type

from xpsoverlaps.model import AOBasis, EngineBasis, GaussianBasis
from xpsoverlaps.overlap.base import AOOverlapProvider
from xpsoverlaps.overlap.engine import EngineOverlapProvider
from xpsoverlaps.overlap.gaussian import GaussianOverlapProvider


# Basis representation -> provider. Chosen by representation, not by program.
_PROVIDERS: Dict[Type[AOBasis], Type[AOOverlapProvider]] = {
    GaussianBasis: GaussianOverlapProvider,
    EngineBasis: EngineOverlapProvider,
}


def get_overlap_provider(basis_a: AOBasis, basis_b: AOBasis) -> AOOverlapProvider:
    if type(basis_a) is not type(basis_b):
        raise ValueError(
            "Both calculations must use the same basis representation; got "
            f"{type(basis_a).__name__} and {type(basis_b).__name__}."
        )
    if type(basis_a) not in _PROVIDERS:
        raise NotImplementedError(f"No AO overlap provider for {type(basis_a).__name__}.")
    return _PROVIDERS[type(basis_a)]()
