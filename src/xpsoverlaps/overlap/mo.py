"""Molecular-orbital overlaps between two calculations."""

from __future__ import annotations

from typing import Optional

import numpy as np

from xpsoverlaps.model import Calculation
from xpsoverlaps.overlap.base import AOOverlapProvider
from xpsoverlaps.overlap.providers import get_overlap_provider


def mo_overlap(
    calculation_a: Calculation,
    calculation_b: Calculation,
    provider: Optional[AOOverlapProvider] = None,
) -> np.ndarray:
    """``S_MO = C_a.T @ S_AO(a, b) @ C_b``; rows/columns follow each orbital set's IDs.

    The provider defaults to the one matching the calculations' basis
    representation.
    """
    if provider is None:
        provider = get_overlap_provider(calculation_a.basis, calculation_b.basis)
    ao_overlap = provider.cross_overlap(calculation_a.basis, calculation_b.basis)
    return calculation_a.orbitals.coefficients.T @ ao_overlap @ calculation_b.orbitals.coefficients
