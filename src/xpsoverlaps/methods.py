"""Propensity-rule approximations to Dyson norms for XPS transitions.

``propensity_overlaps``: one initial calculation is compared with one or more
final calculations, each holding states of a single spin (e.g. the doublet and
the quartet core-hole states, from separate calculations). Each section of the
output covers all final calculations together: the orbital mappings, the
mapped CI vectors, the orbital order, the shake-downs (initial states only),
and the propensity rules, where each channel is evaluated with the final
calculation of its spin and one summary combines them.

``propensity_chp``: the initial calculation only. The final orbitals are the
initial ones relaxed to first order in the core-hole potential of each core
(RAS1) orbital in turn; the sections are the orbital order, the shake-downs,
the core-hole overlaps, and the propensity rules per core orbital with one
summary.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Union

from xpsoverlaps.frontier import (
    FrontierOrbitals,
    OrbitalOrder,
    OrbitalOrderError,
    orbital_order,
    orbital_order_report,
    reference_warning,
)
from xpsoverlaps.integrals import CoreHoleIntegrals
from xpsoverlaps.mapping import MappedFinal, map_active_orbitals, mapped_ci_report, mapping_report
from xpsoverlaps.model import Calculation
from xpsoverlaps.overlap_sources import CHP_OVERLAP_WARNING, CoreHoleOverlaps, core_hole_overlap_report
from xpsoverlaps.propensity_rules import core_hole_report, evaluate_final, evaluate_shake_downs, propensity_report
from xpsoverlaps.shakedown_transitions import (
    ShakeDown,
    ShakeDownError,
    find_shake_downs,
    reference_determinant,
    shake_down_report,
)

logger = logging.getLogger(__name__)

_DASHES = "-" * 80


class FinalCalculationError(ValueError):
    """Raised when the final calculations cannot be told apart by spin."""


def final_spins(finals: Sequence[Calculation]) -> List[float]:
    """The spin of each final calculation: all its states must share one S, and
    no two final calculations may have the same S (each channel needs one)."""
    spins, problems = [], []
    for n, final in enumerate(finals, start=1):
        values = {state.spin for state in final.states}
        label = "Final" if len(finals) == 1 else f"Final {n}"
        if None in values or len(values) != 1:
            problems.append(f"{label} must contain states of one known spin (found S = "
                            f"{', '.join('?' if s is None else f'{s:g}' for s in values)}).")
        spins.append(values.pop() if len(values) == 1 else None)
    known = [s for s in spins if s is not None]
    for spin in sorted({s for s in known if known.count(s) > 1}):
        problems.append(f"More than one final calculation has S = {spin:g}; each spin needs exactly one.")
    if problems:
        raise FinalCalculationError(" ".join(problems))
    return spins


def propensity_overlaps(
    initial: Calculation,
    finals: Union[Calculation, Sequence[Calculation]],
    frontier: Optional[FrontierOrbitals] = None,
    max_state: Optional[int] = None,
) -> None:
    """Approximate Dyson norms from initial/final state orbital overlaps.

    ``finals`` is one final calculation or several of different spin. The
    initial active orbitals are ordered by population; ``frontier`` (the input
    HOMO/LUMO) overrides the HOMO and LUMO of that order. Shake-downs are
    considered among initial states 1..``max_state`` (all if None).
    """
    if isinstance(finals, Calculation):
        finals = [finals]
    spins = final_spins(finals)
    labels = ["Final"] if len(finals) == 1 else [f"Final {n}" for n in range(1, len(finals) + 1)]
    mapped = [
        MappedFinal(label, final, map_active_orbitals(initial, final), spin)
        for label, final, spin in zip(labels, finals, spins)
    ]

    _section([line for final in mapped for line in _titled(final, mapping_report(final.mapping))])
    _section([line for final in mapped
              for line in _titled(final, mapped_ci_report(initial, final.calculation, final.mapping))])
    logger.info("")
    order = _log_orbital_order(initial, mapped, frontier, max_state)
    logger.info("")
    logger.info(_DASHES)
    logger.info("")
    shake_downs = _log_shake_downs(initial, order, max_state)
    logger.info("")
    logger.info(_DASHES)
    if shake_downs:
        evaluations = [(final, evaluate_final(shake_downs, initial, final)) for final in mapped]
        _section(propensity_report(shake_downs, initial, evaluations))


def _titled(final: MappedFinal, lines: List[str]) -> List[str]:
    """A final calculation's block of a section, under its title."""
    return [final.title, "=" * len(final.title), "", *lines, ""]


def _section(lines: List[str]) -> None:
    logger.info("")
    for line in lines:
        logger.info(line)
    if lines and lines[-1]:
        logger.info("")
    logger.info(_DASHES)


def _log_orbital_order(
    initial: Calculation,
    finals: Sequence[MappedFinal],
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
    for line in orbital_order_report(order, initial, finals):
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


def propensity_chp(
    initial: Calculation,
    integrals: CoreHoleIntegrals,
    frontier: Optional[FrontierOrbitals] = None,
    max_state: Optional[int] = None,
) -> None:
    """Approximate Dyson norms from the initial state using a core-hole potential.

    Each core (RAS1) orbital c in turn gives the overlaps S_ab = (ab|cc) /
    (e_b - e_a) between the valence orbitals, with (pq|cc) from ``integrals``
    and e the initial orbital energies. ``frontier`` and ``max_state`` are as
    in ``propensity_overlaps``.
    """
    if initial.orbitals.energies is None:
        raise ValueError("The core-hole potential needs the initial orbital energies (CASPT2: give mcscf_file).")
    logger.info("")
    order = _log_orbital_order(initial, [], frontier, max_state)
    logger.info("")
    logger.info(_DASHES)
    logger.info("")
    shake_downs = _log_shake_downs(initial, order, max_state)
    logger.info("")
    logger.info(_DASHES)
    if order is None:
        return
    cores = [orbital_id for orbital_id, label in order.unranked if label == "core"]
    if not cores:
        logger.warning("No core (RAS1) orbitals: the core-hole potential needs at least one.")
        return

    valence = list(reversed(order.occupied)) + list(order.virtual)
    energies = dict(zip(initial.orbitals.orbital_ids, initial.orbitals.energies))
    coulomb = integrals.coulomb(cores, valence)
    overlap_sources = [CoreHoleOverlaps(coulomb[core], energies) for core in cores]
    _section(core_hole_overlap_report(overlap_sources, {o: order.label(o) for o in valence}, integrals.source))
    large = sum(len(overlap_source.large()) for overlap_source in overlap_sources)
    if large:
        logger.warning("%d core-hole overlaps |S_ab| above %s (listed under Core-hole overlaps): "
                       "first-order relaxation is unreliable for those orbital pairs.", large, CHP_OVERLAP_WARNING)
    if shake_downs:
        evaluations = [(source.core, evaluate_shake_downs(shake_downs, initial, source))
                       for source in overlap_sources]
        _section(core_hole_report(shake_downs, initial, evaluations))
