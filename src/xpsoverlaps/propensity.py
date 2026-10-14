"""Propensity-rule approximations to Dyson norms for XPS transitions."""

from __future__ import annotations

import logging
from typing import Optional

from xpsoverlaps.frontier import FrontierOrbitals, frontier_report, map_frontier
from xpsoverlaps.mapping import map_active_orbitals, mapped_ci_report, mapping_report
from xpsoverlaps.model import Calculation
from xpsoverlaps.shakedown import ShakeDownError, find_shake_downs, shake_down_report

logger = logging.getLogger(__name__)

_DASHES = "-" * 80


def propensity_overlaps(
    initial: Calculation,
    final: Calculation,
    frontier: Optional[FrontierOrbitals] = None,
    max_state: Optional[int] = None,
) -> None:
    """Approximate Dyson norms from initial/final state orbital overlaps.

    ``frontier`` is the initial HOMO/LUMO; the final ones come from the mapping.
    Shake-downs are considered among initial states 1..``max_state`` (all if None).
    """
    mapping = map_active_orbitals(initial, final)
    logger.info("")
    for line in mapping_report(mapping):
        logger.info(line)
    logger.info("")
    logger.info(_DASHES)
    logger.info("")
    for line in mapped_ci_report(initial, final, mapping):
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
    logger.info("")
    _log_shake_downs(initial, frontier, max_state)
    logger.info("")
    logger.info(_DASHES)


def _log_shake_downs(
    initial: Calculation, frontier: Optional[FrontierOrbitals], max_state: Optional[int]
) -> None:
    if frontier is None:
        logger.info("Shake-down transitions need the HOMO/LUMO; not identified.")
        return
    try:
        shake_downs = find_shake_downs(initial, frontier, max_state)
    except ShakeDownError as error:
        logger.warning("Shake-down transitions not identified: %s", error)
        return
    for line in shake_down_report(initial, frontier, shake_downs, max_state):
        logger.info(line)


def propensity_chp(initial: Calculation, frontier: Optional[FrontierOrbitals] = None) -> None:
    """Approximate Dyson norms from the initial state using a core-hole potential."""
    pass
