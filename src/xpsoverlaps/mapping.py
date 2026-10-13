"""Index mapping between the active orbitals of two calculations.

Only state-averaged orbitals are handled: one mapping per pair of
calculations. Nothing is permuted here; the mapping records which final
active orbital corresponds to each initial active orbital.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Mapping, NamedTuple, Optional, Sequence

import numpy as np

from xpsoverlaps.composition import OrbitalComposition, composition_similarity


class OrbitalMappingError(ValueError):
    """Raised when active orbitals cannot be mapped consistently."""


@dataclass(frozen=True)
class MappingOptions:
    min_overlap: float = 0.5  # warn if a matched |S| is below this
    close_second_ratio: float = 0.8  # warn if runner-up |S| / matched |S| is at least this
    rotated_out: Literal["error", "warning"] = "error"  # for orbitals leaving the active space

    def __post_init__(self) -> None:
        if self.rotated_out not in {"error", "warning"}:
            raise ValueError("rotated_out must be 'error' or 'warning'.")


class OrbitalMatch(NamedTuple):
    initial_id: str
    final_id: str
    overlap: float  # signed S_MO[initial, final]
    similarity: float  # AO-composition similarity of the matched pair
    runner_up_id: Optional[str]  # best other final active orbital by |S|
    runner_up_overlap: Optional[float]  # signed
    runner_up_similarity: Optional[float]
    assigned_by: str  # "overlap" or "composition"


@dataclass(frozen=True, eq=False)
class OrbitalMapping:
    """Initial active orbital -> final active orbital, in initial active order."""

    matches: List[OrbitalMatch]
    initial_active_ids: List[str]
    final_active_ids: List[str]
    warnings: List[str] = field(default_factory=list)

    def final_id(self, initial_id: str) -> str:
        return next(match.final_id for match in self.matches if match.initial_id == initial_id)

    @property
    def active_index_map(self) -> List[int]:
        """``index_map[k]``: final active position matched to initial active position ``k``."""
        return [self.final_active_ids.index(match.final_id) for match in self.matches]


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
       space (error or warning, see ``MappingOptions.rotated_out``).
    2. Within the active block, each initial orbital takes its largest-|S|
       final orbital.
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
    warnings = _check_active_space_rotation(
        overlap, initial_orbital_ids, final_orbital_ids, initial_rows, final_columns, options
    )

    active = np.abs(overlap[np.ix_(initial_rows, final_columns)])
    n_active = len(initial_active_ids)

    def similarity(row: int, column: int) -> float:
        return composition_similarity(
            initial_compositions[initial_active_ids[row]],
            final_compositions[final_active_ids[column]],
        )

    best = active.argmax(axis=1)
    claimants: Dict[int, List[int]] = {}
    for row, column in enumerate(best):
        claimants.setdefault(int(column), []).append(row)
    contested_rows = [row for rows in claimants.values() if len(rows) > 1 for row in rows]

    assignment: Dict[int, int] = {
        row: int(best[row]) for row in range(n_active) if row not in contested_rows
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
        free_columns = [c for c in range(n_active) if c not in assignment.values()]
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
    for row in range(n_active):
        column = assignment[row]
        others = [c for c in range(n_active) if c != column]
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
        warnings.extend(_match_warnings(match, options))
        matches.append(match)

    return OrbitalMapping(matches, list(initial_active_ids), list(final_active_ids), warnings)


def _check_active_space_rotation(
    overlap: np.ndarray,
    initial_orbital_ids: Sequence[str],
    final_orbital_ids: Sequence[str],
    initial_rows: Sequence[int],
    final_columns: Sequence[int],
    options: MappingOptions,
) -> List[str]:
    """Problems where an active orbital's best partner lies outside the other active space."""
    magnitude = np.abs(overlap)
    problems = []
    for row in initial_rows:
        column = int(magnitude[row].argmax())
        if column not in final_columns:
            problems.append(
                f"Initial active {initial_orbital_ids[row]} overlaps most with final "
                f"{final_orbital_ids[column]} (|S| = {magnitude[row, column]:.3f}), which is "
                "outside the final active space."
            )
    for column in final_columns:
        row = int(magnitude[:, column].argmax())
        if row not in initial_rows:
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
    return problems


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


def format_mapping(mapping: OrbitalMapping) -> List[str]:
    """Table of matches (warnings are in ``mapping.warnings``)."""
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
            f"{match.initial_id:<8} {match.final_id:<8} {match.overlap:+8.3f} "
            f"{match.similarity:5.2f}   {runner_up}   {match.assigned_by}"
        )
    return lines
