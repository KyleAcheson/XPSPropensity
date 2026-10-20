"""Propensity-rule approximations to Dyson norms for XPS transitions."""

from __future__ import annotations

import logging
from typing import List, Optional

from xpsoverlaps.frontier import (
    FrontierOrbitals,
    OrbitalOrder,
    OrbitalOrderError,
    orbital_order,
    orbital_order_report,
    reference_warning,
)
from xpsoverlaps.mapping import OrbitalMapping, map_active_orbitals, mapped_ci_report, mapping_report
from xpsoverlaps.model import Calculation
from xpsoverlaps.rules import compute_propensity, propensity_report
from xpsoverlaps.shakedown import ShakeDown, ShakeDownError, find_shake_downs, reference_determinant, shake_down_report

logger = logging.getLogger(__name__)

_DASHES = "-" * 80


def propensity_overlaps(
    initial: Calculation,
    final: Calculation,
    frontier: Optional[FrontierOrbitals] = None,
    max_state: Optional[int] = None,
) -> None:
    """Approximate Dyson norms from initial/final state orbital overlaps.

    The initial active orbitals are ordered by population; ``frontier`` (the
    input HOMO/LUMO) overrides the HOMO and LUMO of that order. Shake-downs are
    considered among initial states 1..``max_state`` (all if None).
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
    order = _log_orbital_order(initial, final, mapping, frontier, max_state)
    logger.info("")
    logger.info(_DASHES)
    logger.info("")
    shake_downs = _log_shake_downs(initial, order, max_state)
    logger.info("")
    logger.info(_DASHES)
    if shake_downs:
        logger.info("")
        _log_propensities(shake_downs, initial, final, mapping)
        logger.info("")
        logger.info(_DASHES)


def _log_orbital_order(
    initial: Calculation,
    final: Calculation,
    mapping: OrbitalMapping,
    frontier: Optional[FrontierOrbitals],
    max_state: Optional[int],
) -> Optional[OrbitalOrder]:
    try:
        reference = reference_determinant(initial, max_state, frontier)
        order = orbital_order(initial, reference, frontier)
    except (OrbitalOrderError, ShakeDownError) as error:
        logger.warning("Orbital order not determined: %s", error)
        return None
    warning = reference_warning(reference, initial)
    for line in warning:
        logger.warning(line)
    if warning:
        logger.info("")
    for line in orbital_order_report(order, initial, final, mapping):
        logger.info(line)
    return order


def _log_shake_downs(
    initial: Calculation, order: Optional[OrbitalOrder], max_state: Optional[int]
) -> List[ShakeDown]:
    if order is None:
        logger.info("Shake-down transitions need the orbital order; not identified.")
        return []
    try:
        shake_downs = find_shake_downs(initial, order, max_state)
    except ShakeDownError as error:
        logger.warning("Shake-down transitions not identified: %s", error)
        return []
    for line in shake_down_report(initial, order, shake_downs, max_state):
        logger.info(line)
    return shake_downs


def _log_propensities(
    shake_downs: List[ShakeDown], initial: Calculation, final: Calculation, mapping: OrbitalMapping
) -> None:
    results = [compute_propensity(sd, initial, mapping) for sd in shake_downs if sd.computable]
    for line in propensity_report(results, mapping, final):
        logger.info(line)


def propensity_chp(initial: Calculation, frontier: Optional[FrontierOrbitals] = None) -> None:
    """Approximate Dyson norms from the initial state using a core-hole potential."""
    pass
