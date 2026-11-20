"""Index mapping between the active orbitals of two calculations.

Only state-averaged orbitals are handled: one mapping per pair of
calculations. Nothing is permuted here; the mapping records which final
active orbital corresponds to each initial active orbital.

``map_active_orbitals(initial, final)`` is the entry point: it computes the
overlaps and AO compositions and calls ``match_active_orbitals``. It does no
logging; ``mapping_report(mapping)`` turns the result into text lines.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from xpsoverlaps.composition import (
    OrbitalComposition,
    composition_similarity,
    describe_orbital_pairs,
    orbital_compositions,
)
from xpsoverlaps.model import Calculation
from xpsoverlaps.overlap import get_overlap_provider, mo_overlap


class OrbitalMappingError(ValueError):
    """Raised when active orbitals cannot be mapped consistently."""


class IrrepMappingError(ValueError):
    """Raised when orbitals are matched across irreps, whose overlaps vanish: a reader or model error,
    not a property of the calculations, so it stops the run (unlike ``OrbitalMappingError``)."""


@dataclass(frozen=True)
class MappingOptions:
    min_overlap: float = 0.5  # warn if a matched |S| is below this
    close_second_ratio: float = 0.8  # warn if runner-up |S| / matched |S| is at least this
    # Orbitals whose best partner lies outside the other active space: left unmatched (with a
    # warning), or an error.
    rotated_out: Literal["unmatched", "error"] = "unmatched"

    def __post_init__(self) -> None:
        if self.rotated_out not in {"unmatched", "error"}:
            raise ValueError("rotated_out must be 'unmatched' or 'error'.")


class OrbitalMatch(NamedTuple):
    initial_id: str
    final_id: str
    overlap: float  # signed S_MO[initial, final]
    similarity: float  # AO-composition similarity of the matched pair
    runner_up_id: Optional[str]  # best other final active orbital by |S|
    runner_up_overlap: Optional[float]  # signed
    runner_up_similarity: Optional[float]
    assigned_by: str  # "overlap" or "composition"
    warnings: Tuple[str, ...] = ()  # this match's low-overlap and close-runner-up warnings


@dataclass(frozen=True, eq=False)
class OrbitalMapping:
    """Initial active orbital -> final active orbital, in initial active order.

    An active orbital whose best partner lies outside the other calculation's
    active space has rotated out of it: it has no match (``unmatched_initial``,
    ``unmatched_final``) and ``rotated_out`` says where it went.
    """

    matches: List[OrbitalMatch]
    initial_active_ids: List[str]
    final_active_ids: List[str]
    initial_compositions: Mapping[str, OrbitalComposition]
    final_compositions: Mapping[str, OrbitalComposition]
    mo_overlap: np.ndarray  # S_MO[initial MO, final MO] = <psi_initial|phi_final>, signed
    initial_orbital_ids: List[str]  # row order of mo_overlap
    final_orbital_ids: List[str]  # column order of mo_overlap
    warnings: List[str] = field(default_factory=list)
    unmatched_initial: List[str] = field(default_factory=list)
    unmatched_final: List[str] = field(default_factory=list)
    rotated_out: List[str] = field(default_factory=list)  # one message per rotated-out orbital

    def has_match(self, initial_id: str) -> bool:
        return any(match.initial_id == initial_id for match in self.matches)

    def final_id(self, initial_id: str) -> str:
        """The final orbital matched to ``initial_id``; an error for an unmatched orbital."""
        for match in self.matches:
            if match.initial_id == initial_id:
                return match.final_id
        raise OrbitalMappingError(f"{initial_id} has no match: it has rotated out of the active space.")

    def overlap(self, initial_id: str, final_id: str) -> float:
        """Signed <psi_initial|phi_final> between any two orbitals."""
        row = self.initial_orbital_ids.index(initial_id)
        return float(self.mo_overlap[row, self.final_orbital_ids.index(final_id)])

    @property
    def active_index_map(self) -> List[Optional[int]]:
        """``index_map[k]``: final active position matched to initial active position ``k`` (None if unmatched)."""
        return [self.final_active_ids.index(self.final_id(i)) if self.has_match(i) else None
                for i in self.initial_active_ids]


def match_active_orbitals(
    mo_overlap: np.ndarray,
    initial_orbital_ids: Sequence[str],
    final_orbital_ids: Sequence[str],
    initial_active_ids: Sequence[str],
    final_active_ids: Sequence[str],
    initial_compositions: Mapping[str, OrbitalComposition],
    final_compositions: Mapping[str, OrbitalComposition],
    options: Optional[MappingOptions] = None,
) -> OrbitalMapping:
    """Match each initial active orbital to a final active orbital.

    1. Every active orbital's largest |S| partner over *all* MOs of the other
       calculation must be active, otherwise it has rotated out of the active
       space: it is left unmatched, with a warning (or it is an error, see
       ``MappingOptions.rotated_out``).
    2. Within the remaining active block, each initial orbital takes its
       largest-|S| final orbital.
    3. If that is not one-to-one, orbitals competing for the same partner are
       reassigned among the remaining final orbitals by |S| x AO-composition
       similarity, highest first.
    4. Low overlaps and close runner-ups are reported as warnings.
    """
    options = options or MappingOptions()
    overlap = np.asarray(mo_overlap, dtype=float)
    if overlap.shape != (len(initial_orbital_ids), len(final_orbital_ids)):
        raise ValueError(
            f"MO overlap has shape {overlap.shape}; expected "
            f"({len(initial_orbital_ids)}, {len(final_orbital_ids)})."
        )
    if len(initial_active_ids) != len(final_active_ids):
        raise OrbitalMappingError(
            f"Initial and final active spaces differ in size "
            f"({len(initial_active_ids)} vs {len(final_active_ids)})."
        )

    initial_rows = [list(initial_orbital_ids).index(i) for i in initial_active_ids]
    final_columns = [list(final_orbital_ids).index(j) for j in final_active_ids]
    rotated_out, out_rows, out_columns = _check_active_space_rotation(
        overlap, initial_orbital_ids, final_orbital_ids, initial_rows, final_columns, options
    )
    warnings = list(rotated_out)

    active = np.abs(overlap[np.ix_(initial_rows, final_columns)])
    # Rotated-out orbitals take no part in the matching.
    active[[k for k, row in enumerate(initial_rows) if row in out_rows], :] = -1.0
    active[:, [k for k, column in enumerate(final_columns) if column in out_columns]] = -1.0
    n_active = len(initial_active_ids)
    matchable_rows = [k for k, row in enumerate(initial_rows) if row not in out_rows]
    matchable_columns = [k for k, column in enumerate(final_columns) if column not in out_columns]

    def similarity(row: int, column: int) -> float:
        return composition_similarity(
            initial_compositions[initial_active_ids[row]],
            final_compositions[final_active_ids[column]],
        )

    best = active.argmax(axis=1)
    claimants: Dict[int, List[int]] = {}
    for row in matchable_rows:
        claimants.setdefault(int(best[row]), []).append(row)
    contested_rows = [row for rows in claimants.values() if len(rows) > 1 for row in rows]

    assignment: Dict[int, int] = {
        row: int(best[row]) for row in matchable_rows if row not in contested_rows
    }
    assigned_by = {row: "overlap" for row in assignment}
    if contested_rows:
        for column, rows in claimants.items():
            if len(rows) > 1:
                names = ", ".join(initial_active_ids[row] for row in rows)
                warnings.append(
                    f"Final {final_active_ids[column]} has the largest overlap with several "
                    f"initial orbitals ({names}); these are assigned by |S| x AO-composition "
                    "similarity."
                )
        free_columns = [c for c in matchable_columns if c not in assignment.values()]
        scored = sorted(
            (
                (active[row, column] * similarity(row, column), row, column)
                for row in contested_rows
                for column in free_columns
            ),
            reverse=True,
        )
        for _, row, column in scored:
            if row not in assignment and column not in assignment.values():
                assignment[row] = column
                assigned_by[row] = "composition"

    matches = []
    for row in sorted(assignment):
        column = assignment[row]
        others = [c for c in matchable_columns if c != column]
        runner_up = max(others, key=lambda c: active[row, c]) if others else None
        match = OrbitalMatch(
            initial_id=initial_active_ids[row],
            final_id=final_active_ids[column],
            overlap=float(overlap[initial_rows[row], final_columns[column]]),
            similarity=similarity(row, column),
            runner_up_id=final_active_ids[runner_up] if runner_up is not None else None,
            runner_up_overlap=(
                float(overlap[initial_rows[row], final_columns[runner_up]])
                if runner_up is not None
                else None
            ),
            runner_up_similarity=similarity(row, runner_up) if runner_up is not None else None,
            assigned_by=assigned_by[row],
        )
        match = match._replace(warnings=tuple(_match_warnings(match, options)))
        warnings.extend(match.warnings)
        matches.append(match)

    return OrbitalMapping(
        matches=matches,
        initial_active_ids=list(initial_active_ids),
        final_active_ids=list(final_active_ids),
        initial_compositions=initial_compositions,
        final_compositions=final_compositions,
        mo_overlap=overlap,
        initial_orbital_ids=list(initial_orbital_ids),
        final_orbital_ids=list(final_orbital_ids),
        warnings=warnings,
        unmatched_initial=[initial_active_ids[row] for row in range(n_active) if row not in assignment],
        unmatched_final=[final_active_ids[c] for c in range(n_active) if c not in assignment.values()],
        rotated_out=rotated_out,
    )


def _check_active_space_rotation(
    overlap: np.ndarray,
    initial_orbital_ids: Sequence[str],
    final_orbital_ids: Sequence[str],
    initial_rows: Sequence[int],
    final_columns: Sequence[int],
    options: MappingOptions,
) -> Tuple[List[str], List[int], List[int]]:
    """Active orbitals whose best partner lies outside the other active space: a message for
    each, and their initial rows and final columns in ``overlap``."""
    magnitude = np.abs(overlap)
    problems, out_rows, out_columns = [], [], []
    for row in initial_rows:
        column = int(magnitude[row].argmax())
        if column not in final_columns:
            out_rows.append(row)
            problems.append(
                f"Initial active {initial_orbital_ids[row]} overlaps most with final "
                f"{final_orbital_ids[column]} (|S| = {magnitude[row, column]:.3f}), which is "
                "outside the final active space."
            )
    for column in final_columns:
        row = int(magnitude[:, column].argmax())
        if row not in initial_rows:
            out_columns.append(column)
            problems.append(
                f"Final active {final_orbital_ids[column]} overlaps most with initial "
                f"{initial_orbital_ids[row]} (|S| = {magnitude[row, column]:.3f}), which is "
                "outside the initial active space."
            )
    if problems and options.rotated_out == "error":
        raise OrbitalMappingError(
            "Active orbitals have rotated out of the active space:\n  "
            + "\n  ".join(problems)
        )
    return problems, out_rows, out_columns


def _match_warnings(match: OrbitalMatch, options: MappingOptions) -> List[str]:
    warnings = []
    magnitude = abs(match.overlap)
    pair = f"{match.initial_id} -> {match.final_id}"
    if magnitude < options.min_overlap:
        warnings.append(f"Low overlap for {pair}: |S| = {magnitude:.3f}.")
    if match.runner_up_overlap is not None and magnitude > 0:
        runner_up = abs(match.runner_up_overlap)
        if runner_up / magnitude >= options.close_second_ratio:
            preference = (
                " AO composition favours the runner-up."
                if match.runner_up_similarity is not None
                and match.runner_up_similarity > match.similarity
                else ""
            )
            warnings.append(
                f"Close runner-up for {pair} (|S| = {magnitude:.3f}, similarity "
                f"{match.similarity:.2f}): {match.runner_up_id} has |S| = {runner_up:.3f}, "
                f"similarity {match.runner_up_similarity:.2f}.{preference}"
            )
    return warnings


def format_mapping(mapping: OrbitalMapping, initial_labels: Optional[Mapping[str, str]] = None) -> List[str]:
    """Table of matches (warnings are in ``mapping.warnings``); ``initial_labels`` shows initial
    orbitals under other names, e.g. marked."""
    initial_labels = initial_labels or {}
    header = (
        f"{'Initial':<8} {'Final':<8} {'S':>8} {'Sim':>5}   "
        f"{'Runner-up':<10} {'|S|':>6} {'Sim':>5}   Assigned by"
    )
    lines = [header, "-" * len(header)]
    for match in mapping.matches:
        runner_up = (
            f"{match.runner_up_id:<10} {abs(match.runner_up_overlap):6.3f} "
            f"{match.runner_up_similarity:5.2f}"
            if match.runner_up_id is not None
            else f"{'-':<10} {'-':>6} {'-':>5}"
        )
        lines.append(
            f"{initial_labels.get(match.initial_id, match.initial_id):<8} {match.final_id:<8} {match.overlap:+8.3f} "
            f"{match.similarity:5.2f}   {runner_up}   {match.assigned_by}"
        )
    lines += [f"{initial_labels.get(i, i):<8} {'-':<8} {'-':>8} {'-':>5}   {'-':<10} {'-':>6} {'-':>5}   no match"
              for i in mapping.unmatched_initial]
    return lines


def map_active_orbitals(
    initial: Calculation, final: Calculation, options: Optional[MappingOptions] = None
) -> OrbitalMapping:
    """Map the initial active orbitals onto the final ones (no logging).

    Uses state-averaged orbitals (one orbital set per calculation). Nothing is
    permuted; the returned mapping only records the correspondence. With
    symmetry, overlaps between irreps vanish, so every orbital must be matched
    within its irrep; a match across irreps is an error.
    """
    provider = get_overlap_provider(initial.basis, final.basis)
    mapping = match_active_orbitals(
        mo_overlap(initial, final, provider),
        initial.orbitals.orbital_ids,
        final.orbitals.orbital_ids,
        initial.active_space.orbital_ids,
        final.active_space.orbital_ids,
        orbital_compositions(initial, provider=provider),
        orbital_compositions(final, provider=provider),
        options,
    )
    initial_irrep, final_irrep = initial.orbitals.irrep_by_id(), final.orbitals.irrep_by_id()
    crossed = [f"{m.initial_id} ({initial_irrep[m.initial_id]}) -> {m.final_id} ({final_irrep[m.final_id]})"
               for m in mapping.matches if initial_irrep[m.initial_id] != final_irrep[m.final_id]]
    if crossed:
        raise IrrepMappingError("Active orbitals were matched across irreps, whose overlaps vanish: "
                                  + ", ".join(crossed) + ".")
    return mapping


class MappedFinal(NamedTuple):
    """One final calculation, its spin, and the mapping of the initial active orbitals onto it."""

    label: str  # "Final" for a single final calculation, else "Final 1", "Final 2", ...
    calculation: Calculation
    mapping: OrbitalMapping
    spin: float  # the S shared by all its states

    @property
    def irrep(self) -> Optional[str]:
        """Irrep of the final states (each calculation holds the states of one irrep)."""
        return self.calculation.state_irrep

    @property
    def title(self) -> str:
        """e.g. "Final 1: S = 0.5 (core-doub.caspt2.h5)"; with symmetry "Final 2: S = 0.5, A" (core-dz-ap.rasscf.h5)"."""
        irrep = f", {self.irrep.upper()}" if self.calculation.point_group.has_symmetry else ""
        return f"{self.label}: S = {self.spin:g}{irrep} ({Path(self.calculation.provenance.source_files[0]).name})"


def mapping_report(mapping: OrbitalMapping, irreps: Optional[Mapping[str, str]] = None) -> List[str]:
    """Text report of a mapping: match table, warnings, and paired orbital compositions.

    ``irreps`` (initial orbital ID -> irrep), given for a calculation with
    symmetry, splits the match table into one table per irrep.
    """
    lines = ["Active orbital mapping (initial -> final), by MO overlap", ""]
    lines += unmatched_warning(mapping)
    if irreps is None:
        lines += format_mapping(mapping)
    else:
        for irrep in dict.fromkeys(irreps[i] for i in mapping.initial_active_ids):
            part = dataclasses.replace(
                mapping, matches=[match for match in mapping.matches if irreps[match.initial_id] == irrep],
                unmatched_initial=[i for i in mapping.unmatched_initial if irreps[i] == irrep])
            lines += [f"Irrep {irrep}", *format_mapping(part), ""]
        lines.pop()
    lines.append("")
    if mapping.warnings:
        lines.append("Mapping warnings:")
        lines += [f"  - {warning}" for warning in mapping.warnings]
    else:
        lines.append("No mapping warnings.")
    lines += ["", "Dominant AO contributions (AO weights, Mulliken partition) of the paired orbitals", ""]
    pairs = [
        (mapping.initial_compositions[match.initial_id], mapping.final_compositions[match.final_id])
        for match in mapping.matches
    ]
    return lines + describe_orbital_pairs(pairs)


def unmatched_warning(mapping: OrbitalMapping) -> List[str]:
    """A prominent block naming the orbitals that rotated out of the active space and so have no match
    (empty if there are none): shake-downs whose orbital roles include them cannot be computed."""
    if not mapping.unmatched_initial and not mapping.unmatched_final:
        return []
    stars = "*" * 80
    lines = [stars, "* WARNING: active orbitals without a match (rotated out of the active space)", "*"]
    lines += [f"*   {message}" for message in mapping.rotated_out]
    lines += [f"*   Unmatched initial: {', '.join(mapping.unmatched_initial) or '-'};  "
              f"unmatched final: {', '.join(mapping.unmatched_final) or '-'}.",
              "*   Shake-downs whose orbital roles i, j, k include an unmatched initial orbital are",
              "*   not computable with this final calculation.", stars, ""]
    return lines


class CoreOverlapBlock:
    """|<phi_c'|psi_c>| between the initial and final core (RAS1) orbitals, one block per irrep.

    The propensity rules take this block as the identity (S_cc = 1, S_cc' = 0):
    the final state keeps its core hole in the ionised orbital. Near-degenerate
    core orbitals of one irrep (e.g. in-plane S 2p) can rotate between the
    initial and final calculations; a matched |S_cc| below
    ``DIAGONAL_WARNING`` is warned about, as cross-core shake-downs then carry
    intensity the rules leave out (ADR-0005).
    """

    DIAGONAL_WARNING = 0.9

    def __init__(self, mapping: OrbitalMapping, initial: Calculation, final: Calculation,
                 initial_cores: Sequence[str], final_cores: Sequence[str], final_label: str) -> None:
        self.mapping = mapping
        self.final_label = final_label
        initial_irrep, final_irrep = initial.orbitals.irrep_by_id(), final.orbitals.irrep_by_id()
        self.blocks: Dict[str, Tuple[List[str], List[str]]] = {}  # irrep -> (initial cores, final cores)
        for irrep in dict.fromkeys(initial_irrep[c] for c in initial_cores):
            self.blocks[irrep] = ([c for c in initial_cores if initial_irrep[c] == irrep],
                                  [c for c in final_cores if final_irrep[c] == irrep])

    def report(self) -> List[str]:
        lines = ["Core (RAS1) orbital overlaps |<final|initial>| by irrep (the rules take them as the identity)"]
        for irrep, (initial_cores, final_cores) in self.blocks.items():
            lines += ["", f"Irrep {irrep}", f"{'Initial':<8}  " + "  ".join(f"{c:>8}" for c in final_cores)]
            for c in initial_cores:
                lines.append(f"{c:<8}  " + "  ".join(f"{abs(self.mapping.overlap(c, f)):8.3f}" for f in final_cores))
        return lines

    def warnings(self) -> List[str]:
        return [
            f"Core orbital {c} rotates in {self.final_label}: |S| with its match {self.mapping.final_id(c)} is "
            f"{abs(self.mapping.overlap(c, self.mapping.final_id(c))):.3f} < {self.DIAGONAL_WARNING}; the rules "
            "take the core block of S as the identity, so cross-core shake-downs are not included (ADR-0005)."
            for initial_cores, _ in self.blocks.values() for c in initial_cores
            if self.mapping.has_match(c) and abs(self.mapping.overlap(c, self.mapping.final_id(c))) < self.DIAGONAL_WARNING
        ]


def remap_occupations(occupations: Sequence[str], mapping: OrbitalMapping) -> List[str]:
    """Final-state occupations reordered onto the initial active orbitals.

    Position k holds the occupation of the final orbital matched to initial
    active orbital k, so initial and final strings compare position by position;
    "-" where initial orbital k has no match.
    """
    return ["-" if position is None else occupations[position] for position in mapping.active_index_map]


def open_shells_reordered(occupations: Sequence[str], mapping: OrbitalMapping) -> bool:
    """True if remapping changes the order of the singly occupied orbitals.

    ``u``/``d`` couple the open shells in orbital order, so the remapped string
    is then not the same CSF; its occupation numbers are still exact.
    """
    positions = [p for p in mapping.active_index_map if p is not None and occupations[p] in "ud"]
    return positions != sorted(positions)


def mapped_ci_report(initial: Calculation, final: Calculation, mapping: OrbitalMapping) -> List[str]:
    """Dominant CI vectors of both calculations on the initial active-orbital order."""
    n_active = len(mapping.initial_active_ids)
    width, raw_width, mapped_width = (
        max(n_active, len(header)) for header in ("CI vector", "CI vector (raw)", "CI vector (mapped)")
    )
    lines = [
        "Dominant CI vectors on the initial active-orbital order",
        "",
        "Position k refers to initial orbital k and to the final orbital matched to it:",
        f"  initial: {' '.join(mapping.initial_active_ids)}",
        f"  final:   {' '.join(mapping.final_id(i) if mapping.has_match(i) else '-' for i in mapping.initial_active_ids)}",
        "",
        f"Initial states ({initial.provenance.method})",
        f"{'State':>5}  {'Energy (Eh)':>16}  {'CI vector':<{width}}  {'Weight':>8}",
    ]
    for state in initial.states:
        occupations = _occupation_string(state.dominant_configuration)
        lines.append(f"{_state_columns(state)}  {occupations:<{width}}  {_weight(state):>8}")

    lines += [
        "",
        f"Final states ({final.provenance.method})",
        f"{'State':>5}  {'Energy (Eh)':>16}  {'CI vector (raw)':<{raw_width}}  "
        f"{'CI vector (mapped)':<{mapped_width}}  {'Weight':>8}",
    ]
    any_reordered = False
    for state in final.states:
        configuration = state.dominant_configuration
        raw, mapped, flag = "-", "-", ""
        if configuration is not None:
            raw = "".join(configuration.occupations)
            mapped = "".join(remap_occupations(configuration.occupations, mapping))
            if open_shells_reordered(configuration.occupations, mapping):
                flag, any_reordered = "  *", True
        lines.append(
            f"{_state_columns(state)}  {raw:<{raw_width}}  {mapped:<{mapped_width}}  "
            f"{_weight(state):>8}{flag}"
        )
    if any_reordered:
        lines += [
            "",
            "* The mapping changes the order of the open shells: the occupation numbers are",
            "  exact, but the u/d spin coupling of the mapped string is not the same CSF.",
        ]
    return lines


def _occupation_string(configuration) -> str:
    return "".join(configuration.occupations) if configuration is not None else "-"


def _state_columns(state) -> str:
    energy = f"{state.energy:.8f}" if state.energy is not None else "-"
    return f"{state.state_id.split(':')[-1]:>5}  {energy:>16}"


def _weight(state) -> str:
    configuration = state.dominant_configuration
    return f"{configuration.weight:.4f}" if configuration and configuration.weight is not None else "-"
