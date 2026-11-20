"""Propensity-rule approximations to Dyson norms for XPS transitions.

``propensity_overlaps``: an initial calculation is compared with one or more
final calculations, each holding states of a single spin and irrep (e.g. the
doublet and the quartet core-hole states, from separate calculations). Each
section of the output covers all final calculations together: the orbital
mappings (with the core-orbital overlap block), the mapped CI vectors, the
orbital order, the shake-downs (initial states only), and the propensity
rules, where each channel is evaluated with the final calculation of its spin
and one summary combines them.

With point-group symmetry there is one initial calculation per irrep, each
analysed on its own in a part of the output (ADR-0005). A core hole of irrep
Γ_c in an initial calculation of irrep Γ_I reaches final states of irrep
Γ_I × Γ_c only, so its rules use the final calculations of that irrep; the
rules are grouped by core-hole irrep, and a closing summary covers all initial
calculations.

``propensity_chp``: the initial calculation only. The final orbitals are the
initial ones relaxed, to first or to all orders, in the core-hole potential of
each core (RAS1) orbital in turn; the sections are the orbital order, the
shake-downs, the core-hole overlaps, and the propensity rules per core orbital
with one summary.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Literal, NamedTuple, Optional, Sequence, Tuple, Union

from xpsoverlaps.frontier import (
    FrontierOrbitals,
    core_orbitals,
    OrbitalOrder,
    OrbitalOrderError,
    orbital_order,
    orbital_order_report,
    reference_warning,
)
from xpsoverlaps.integrals import CoreHoleIntegrals
from xpsoverlaps.mapping import (
    CoreOverlapBlock,
    MappedFinal,
    OrbitalMappingError,
    map_active_orbitals,
    mapped_ci_report,
    mapping_report,
    unmatched_warning,
)
from xpsoverlaps.model import Calculation
from xpsoverlaps.overlap_sources import (
    AllOrdersCoreHoleOverlaps,
    CoreHoleOverlaps,
    FirstOrderCoreHoleOverlaps,
    core_hole_overlap_report,
)
from xpsoverlaps.propensity_rules import (
    SummaryRow,
    closing_summary,
    core_hole_closing_summary,
    core_hole_report,
    core_hole_rows,
    core_hole_routes,
    core_titles,
    evaluate_final,
    evaluate_shake_downs,
    propensity_report,
    summary_rows,
)
from xpsoverlaps.shakedown_transitions import (
    ShakeDown,
    ShakeDownError,
    find_shake_downs,
    reference_determinant,
    shake_down_report,
)

logger = logging.getLogger(__name__)

_DASHES = "-" * 80


class CalculationSetError(ValueError):
    """Raised when the Initial and Final calculations of a run do not fit together."""


class InitialBlockError(ValueError):
    """Raised when one Initial block cannot be analysed (e.g. its orbitals cannot be mapped); the run goes on."""


class InitialCalculation(NamedTuple):
    """One Initial block: a calculation (of one irrep, with symmetry), its input HOMO/LUMO and
    max_state, and for the core-hole potential its integrals (pq|cc)."""

    label: str  # "Initial" for a single block, else "Initial 1", "Initial 2", ...
    calculation: Calculation
    frontier: Optional[FrontierOrbitals] = None
    max_state: Optional[int] = None
    integrals: Optional[CoreHoleIntegrals] = None  # method chp only

    @property
    def irrep(self) -> Optional[str]:
        return self.calculation.state_irrep

    @property
    def title(self) -> str:
        """e.g. "Initial 1: A' (valence-dz-a.rasscf.h5)"."""
        return f"{self.label}: {self.irrep.upper()} ({Path(self.calculation.provenance.source_files[0]).name})"


def check_calculations(
    initials: Sequence[InitialCalculation], finals: Sequence[Calculation]
) -> List[Tuple[float, str]]:
    """The (spin, irrep) of each final calculation, after checking the run fits together.

    All calculations use one point group; each holds states of one irrep (and
    each final one of one known spin); no two Initial blocks share an irrep and
    no two final calculations share (spin, irrep): each channel and core-hole
    irrep needs exactly one (ADR-0005).
    """
    problems = []
    final_labels = ["Final"] if len(finals) == 1 else [f"Final {n}" for n in range(1, len(finals) + 1)]
    labelled = [(i.label, i.calculation) for i in initials] + list(zip(final_labels, finals))
    groups = {calculation.point_group.irreps for _, calculation in labelled}
    if len(groups) > 1:
        problems.append("All calculations must use the same point group: " + "; ".join(
            f"{label} has irreps {', '.join(calculation.point_group.irreps)}" for label, calculation in labelled) + ".")
    for label, calculation in labelled:
        if calculation.states and calculation.state_irrep is None:
            problems.append(f"{label} must contain states of one irrep.")
    symmetric = labelled[0][1].point_group.has_symmetry
    seen: Dict[str, str] = {}
    for initial in initials:
        if initial.irrep in seen:
            problems.append(f"{seen[initial.irrep]} and {initial.label} have the same irrep "
                            f"{initial.irrep.upper()}; give one Initial block per irrep.")
        seen.setdefault(initial.irrep, initial.label)

    keys: List[Tuple[float, str]] = []
    for label, final in zip(final_labels, finals):
        spins = {state.spin for state in final.states}
        if None in spins or len(spins) != 1:
            problems.append(f"{label} must contain states of one known spin (found S = "
                            f"{', '.join('?' if s is None else f'{s:g}' for s in spins)}).")
        keys.append((spins.pop() if len(spins) == 1 else None, final.state_irrep))
    known = [key for key in keys if key[0] is not None]
    for spin, irrep in sorted({key for key in known if known.count(key) > 1}):
        if symmetric:
            problems.append(f"More than one final calculation has S = {spin:g} and irrep {irrep.upper()}; "
                            "each spin and irrep needs exactly one.")
        else:
            problems.append(f"More than one final calculation has S = {spin:g}; each spin needs exactly one.")
    if problems:
        raise CalculationSetError(" ".join(problems))
    return keys


def propensity_overlaps(
    initials: Union[Calculation, Sequence[InitialCalculation]],
    finals: Union[Calculation, Sequence[Calculation]],
    frontier: Optional[FrontierOrbitals] = None,
    max_state: Optional[int] = None,
) -> List[str]:
    """Approximate Dyson norms from initial/final state orbital overlaps; return the errors of failed blocks.

    ``initials`` is one initial calculation, or one ``InitialCalculation`` per
    irrep; each is analysed on its own (ADR-0005). ``finals`` is one final
    calculation or several of different (spin, irrep). For a single initial
    calculation ``frontier`` (the input HOMO/LUMO) overrides the HOMO and LUMO
    of the population order, and shake-downs are considered among initial
    states 1..``max_state`` (all if None). An Initial block whose active
    orbitals cannot be mapped onto a final calculation it needs is reported
    and skipped; the others and the closing summary are still produced, and
    the returned list holds one error per skipped block.
    """
    if isinstance(initials, Calculation):
        initials = [InitialCalculation("Initial", initials, frontier, max_state)]
    if isinstance(finals, Calculation):
        finals = [finals]
    keys = check_calculations(initials, finals)
    labels = ["Final"] if len(finals) == 1 else [f"Final {n}" for n in range(1, len(finals) + 1)]
    symmetric = finals[0].point_group.has_symmetry
    summaries, failures = [], []
    for initial in initials:
        try:
            summaries.append((initial.irrep, _overlaps_for_initial(initial, finals, labels, [s for s, _ in keys],
                                                                   symmetric)))
        except InitialBlockError as error:
            logger.error("%s", error)
            logger.info("")
            logger.info(_DASHES)
            failures.append(str(error))
    if len(initials) > 1:
        _section(closing_summary(summaries, failures))
    return failures


def _overlaps_for_initial(
    initial: InitialCalculation, finals: Sequence[Calculation], labels: Sequence[str], spins: Sequence[float],
    symmetric: bool,
) -> List[SummaryRow]:
    """The sections of one Initial block; returns its summary rows.

    With symmetry only the final calculations some core hole reaches are used
    (irrep Γ_I × Γ_c, ADR-0005), and the rules are grouped by core-hole irrep.
    """
    calculation = initial.calculation
    cores = core_orbitals(calculation)
    routes = core_hole_routes(calculation, cores) if symmetric else None
    reached = {route.final_irrep for route in routes} if routes is not None else None
    used = [(label, final, spin) for label, final, spin in zip(labels, finals, spins)
            if reached is None or final.state_irrep in reached]
    forbidden = [f"{label} ({final.state_irrep.upper()})" for label, final in zip(labels, finals)
                 if reached is not None and final.state_irrep not in reached]
    if symmetric:
        _log_part_title(initial)
        for title in forbidden:
            logger.info("%s: symmetry-forbidden for %s (no core hole of %s reaches its irrep).",
                        title, initial.label, initial.irrep.upper())
    mapped = []
    for label, final, spin in used:
        try:
            mapped.append(MappedFinal(label, final, map_active_orbitals(calculation, final), spin))
        except OrbitalMappingError as error:
            raise InitialBlockError(f"{initial.label} ({initial.irrep.upper()}) is skipped: its active orbitals "
                                    f"cannot be mapped onto {label}. {error}") from error
    active_irreps = calculation.orbitals.irrep_by_id() if symmetric else None
    core_blocks = [CoreOverlapBlock(final.mapping, calculation, final.calculation, cores,
                                    core_orbitals(final.calculation), final.label) for final in mapped]
    _section([line for final, block in zip(mapped, core_blocks)
              for line in _titled(final, mapping_report(final.mapping, active_irreps) + [""] + block.report())])
    for final in mapped:
        if final.mapping.unmatched_initial or final.mapping.unmatched_final:
            logger.warning(
                "%s -> %s: active orbitals rotated out of the active space have no match (initial %s; final %s); "
                "shake-downs that need them are not computable (see the orbital mapping).", initial.label,
                final.label, ", ".join(final.mapping.unmatched_initial) or "-",
                ", ".join(final.mapping.unmatched_final) or "-")
    for block in core_blocks:
        for warning in block.warnings():
            logger.warning(warning)
    _section([line for final in mapped
              for line in _titled(final, mapped_ci_report(calculation, final.calculation, final.mapping))])
    logger.info("")
    order = _log_orbital_order(calculation, mapped, initial.frontier, initial.max_state)
    logger.info("")
    logger.info(_DASHES)
    logger.info("")
    shake_downs = _log_shake_downs(calculation, order, initial.max_state)
    if symmetric:
        _warn_cross_core(initial)
    logger.info("")
    logger.info(_DASHES)
    if not shake_downs:
        return []
    evaluations = [(final, evaluate_final(shake_downs, calculation, final)) for final in mapped]
    _section(propensity_report(shake_downs, calculation, evaluations, routes))
    return summary_rows(shake_downs, calculation, evaluations, routes)


def _log_part_title(initial: InitialCalculation) -> None:
    """The title of an Initial block's part of the output (with symmetry)."""
    logger.info("")
    logger.info("=" * len(initial.title))
    logger.info(initial.title)
    logger.info("=" * len(initial.title))


def _warn_cross_core(initial: InitialCalculation) -> None:
    """The approximation every Initial block of a run with symmetry makes (ADR-0005)."""
    logger.info("")
    logger.warning(
        "%s (%s): cross-core shake-downs to states of other irreps are symmetry-allowed but not included: "
        "they need core-valence overlaps or final-state CI mixing (ADR-0005).",
        initial.label, initial.irrep.upper())


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
    initials: Union[Calculation, Sequence[InitialCalculation]],
    integrals: Optional[CoreHoleIntegrals] = None,
    frontier: Optional[FrontierOrbitals] = None,
    max_state: Optional[int] = None,
    order: Literal[1, "all"] = 1,
) -> None:
    """Approximate Dyson norms from the initial state using a core-hole potential.

    Each core (RAS1) orbital c in turn relaxes the initial orbitals in -J_c,
    with (pq|cc) from the integrals and e the initial orbital energies: to
    first order (``order`` 1), S_ab = (ab|cc) / (e_b - e_a) between the RAS2
    orbitals, or to all orders (``order`` "all") by diagonalising
    diag(e) - J_c over the valence (RAS2 and RAS3) orbitals.

    ``initials`` is one initial calculation (with ``integrals``, ``frontier``
    and ``max_state`` as in ``propensity_overlaps``), or one
    ``InitialCalculation`` per irrep, each with its own integrals. With
    symmetry each core orbital's results are labelled with the final irrep
    Γ_I × Γ_c its hole reaches, and a closing summary covers all blocks.
    """
    if order not in (1, "all"):
        raise ValueError(f"The core-hole potential order must be 1 or 'all', not {order!r}.")
    if isinstance(initials, Calculation):
        initials = [InitialCalculation("Initial", initials, frontier, max_state, integrals)]
    for initial in initials:
        if initial.calculation.orbitals.energies is None:
            raise ValueError("The core-hole potential needs the initial orbital energies (CASPT2: give mcscf_file).")
        if initial.integrals is None:
            raise ValueError(f"{initial.label} has no integrals (pq|cc) for the core-hole potential.")
    check_calculations(initials, [])
    symmetric = initials[0].calculation.point_group.has_symmetry
    blocks = [_chp_for_initial(initial, order, symmetric) for initial in initials]
    if len(initials) > 1:
        _section(core_hole_closing_summary([block for block in blocks if block is not None]))


def _chp_for_initial(initial: InitialCalculation, order: Literal[1, "all"], symmetric: bool):
    """The sections of one Initial block; returns (irrep, core-hole routes, summary rows), or None
    when no rules could be evaluated."""
    calculation = initial.calculation
    if symmetric:
        _log_part_title(initial)
    logger.info("")
    orbital_order = _log_orbital_order(calculation, [], initial.frontier, initial.max_state)
    logger.info("")
    logger.info(_DASHES)
    logger.info("")
    shake_downs = _log_shake_downs(calculation, orbital_order, initial.max_state)
    if symmetric:
        _warn_cross_core(initial)
    logger.info("")
    logger.info(_DASHES)
    if orbital_order is None:
        return None
    cores = [orbital_id for orbital_id, label in orbital_order.unranked if label == "core"]
    if not cores:
        logger.warning("No core (RAS1) orbitals: the core-hole potential needs at least one.")
        return None
    routes = core_hole_routes(calculation, cores) if symmetric else None

    role_orbitals = list(reversed(orbital_order.occupied)) + list(orbital_order.virtual)  # RAS2
    energies = dict(zip(calculation.orbitals.orbital_ids, calculation.orbitals.energies))
    overlap_sources: List[CoreHoleOverlaps]
    if order == 1:
        coulomb = initial.integrals.coulomb(cores, role_orbitals)
        overlap_sources = [FirstOrderCoreHoleOverlaps(coulomb[core], energies) for core in cores]
    else:
        valence = role_orbitals + [orbital_id for orbital_id, label in orbital_order.unranked if label == "RAS3"]
        coulomb = initial.integrals.coulomb(cores, valence)
        overlap_sources = [AllOrdersCoreHoleOverlaps(coulomb[core], energies, calculation) for core in cores]
    labels = {o: orbital_order.label(o) for o in role_orbitals}
    _section(core_hole_overlap_report(overlap_sources, labels, initial.integrals.source, core_titles(cores, routes)))
    warned = sum(len(overlap_source.warnings()) for overlap_source in overlap_sources)
    if warned:
        logger.warning("%d %s", warned, overlap_sources[0].warning_summary)
    if not shake_downs:
        return None
    evaluations = [(overlap_source, evaluate_shake_downs(shake_downs, calculation, overlap_source))
                   for overlap_source in overlap_sources]
    _section(core_hole_report(shake_downs, calculation, evaluations, routes))
    return initial.irrep, routes or [], core_hole_rows(shake_downs, calculation, evaluations)
