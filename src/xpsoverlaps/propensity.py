"""Propensity-rule approximations to Dyson norms for XPS transitions."""

from __future__ import annotations

import logging
from typing import Optional

from xpsoverlaps.frontier import FrontierOrbitals, frontier_report, map_frontier
from xpsoverlaps.mapping import map_active_orbitals, mapping_report
from xpsoverlaps.model import Calculation

logger = logging.getLogger(__name__)

_DASHES = "-" * 80


def propensity_overlaps(
    initial: Calculation, final: Calculation, frontier: Optional[FrontierOrbitals] = None
) -> None:
    """Approximate Dyson norms from initial/final state orbital overlaps.

    ``frontier`` is the initial HOMO/LUMO; the final ones come from the mapping.
    """
    mapping = map_active_orbitals(initial, final)
    logger.info("")
    for line in mapping_report(mapping):
        logger.info(line)
    logger.info("")
    logger.info(_DASHES)
    logger.info("")
    if frontier is None:
        logger.info("HOMO/LUMO not given in the input; not assigned.")
    else:
        for line in frontier_report(frontier, map_frontier(frontier, mapping)):
            logger.info(line)
    logger.info("")
    logger.info(_DASHES)


def propensity_chp(initial: Calculation, frontier: Optional[FrontierOrbitals] = None) -> None:
    """Approximate Dyson norms from the initial state using a core-hole potential."""
    pass
