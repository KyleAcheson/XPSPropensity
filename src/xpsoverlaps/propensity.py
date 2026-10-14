"""Propensity-rule approximations to Dyson norms for XPS transitions."""

from __future__ import annotations

import logging

from xpsoverlaps.mapping import map_active_orbitals, mapping_report
from xpsoverlaps.model import Calculation

logger = logging.getLogger(__name__)

_DASHES = "-" * 80


def propensity_overlaps(initial: Calculation, final: Calculation) -> None:
    """Approximate Dyson norms from initial/final state orbital overlaps."""
    mapping = map_active_orbitals(initial, final)
    logger.info("")
    for line in mapping_report(mapping):
        logger.info(line)
    logger.info("")
    logger.info(_DASHES)


def propensity_chp(initial: Calculation) -> None:
    """Approximate Dyson norms from the initial state using a core-hole potential."""
    pass
