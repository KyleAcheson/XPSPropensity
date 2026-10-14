"""Propensity-rule approximations to Dyson norms for XPS transitions."""

from __future__ import annotations

import logging

from xpsoverlaps.mapping import map_active_orbitals
from xpsoverlaps.model import Calculation

logger = logging.getLogger(__name__)


def propensity_overlaps(initial: Calculation, final: Calculation) -> None:
    """Approximate Dyson norms from initial/final state orbital overlaps."""
    map_active_orbitals(initial, final)


def propensity_chp(initial: Calculation) -> None:
    """Approximate Dyson norms from the initial state using a core-hole potential."""
    pass
