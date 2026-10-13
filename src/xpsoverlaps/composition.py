"""Atomic-orbital composition of molecular orbitals from Mulliken populations."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from xpsoverlaps.model import Calculation
from xpsoverlaps.overlap import AOOverlapProvider, get_overlap_provider

_SHELL_LETTER = re.compile(r"^\d+([A-Za-z])")


@dataclass
class AOContribution:
    label: str
    coefficient: float
    population: float  # Mulliken gross population of this AO in the MO


@dataclass
class OrbitalComposition:
    """AO make-up of one MO; populations sum to one over all AOs."""

    orbital_id: str
    contributions: List[AOContribution]  # sorted by decreasing population
    groups: Dict[str, float]  # e.g. {"BR6 p": 0.74, ...}, sorted, non-negative, sums to one

    def dominant(self, threshold: float = 0.05, max_count: int = 8) -> List[AOContribution]:
        return [c for c in self.contributions if c.population >= threshold][:max_count]


def ao_group(label: str) -> str:
    """Group an AO label by atom and angular momentum, e.g. ``"BR6 4P[m=-1]"`` -> ``"BR6 p"``.

    The number in AO labels is a contraction index, not the atomic shell, so
    groups deliberately use only the atom and the angular-momentum letter.
    """
    atom, orbital = label.split(maxsplit=1)
    match = _SHELL_LETTER.match(orbital)
    return f"{atom} {match.group(1).lower() if match else orbital}"


def orbital_compositions(
    calculation: Calculation,
    orbital_ids: Optional[Iterable[str]] = None,
    provider: Optional[AOOverlapProvider] = None,
) -> Dict[str, OrbitalComposition]:
    """Mulliken AO composition of the given MOs (default: the active orbitals).

    The gross population of AO mu in MO k is ``C[mu, k] * (S @ C)[mu, k]``.
    """
    basis, orbitals = calculation.basis, calculation.orbitals
    if orbital_ids is None:
        orbital_ids = calculation.active_space.orbital_ids
    if provider is None:
        provider = get_overlap_provider(basis, basis)
    ao_overlap = provider.cross_overlap(basis, basis)
    positions = [orbitals.orbital_ids.index(orbital_id) for orbital_id in orbital_ids]
    coefficients = orbitals.coefficients[:, positions]
    populations = coefficients * (ao_overlap @ coefficients)

    compositions = {}
    for column, orbital_id in enumerate(orbital_ids):
        contributions = sorted(
            (
                AOContribution(label, float(coefficients[row, column]), float(populations[row, column]))
                for row, label in enumerate(basis.function_labels)
            ),
            key=lambda contribution: -contribution.population,
        )
        compositions[orbital_id] = OrbitalComposition(
            orbital_id, contributions, _group_populations(contributions)
        )
    return compositions


def _group_populations(contributions: Sequence[AOContribution]) -> Dict[str, float]:
    groups: Dict[str, float] = defaultdict(float)
    for contribution in contributions:
        groups[ao_group(contribution.label)] += contribution.population
    # Mulliken group populations can be slightly negative; clip and renormalize.
    clipped = {group: max(value, 0.0) for group, value in groups.items()}
    total = sum(clipped.values()) or 1.0
    return dict(
        sorted(((group, value / total) for group, value in clipped.items()), key=lambda kv: -kv[1])
    )


def composition_similarity(a: OrbitalComposition, b: OrbitalComposition) -> float:
    """Shared AO-group character, ``sum_g min(a_g, b_g)``: 1 identical, 0 disjoint."""
    return float(sum(min(value, b.groups.get(group, 0.0)) for group, value in a.groups.items()))


def describe_orbital(
    composition: OrbitalComposition, threshold: float = 0.05, max_count: int = 8
) -> List[str]:
    """Text lines listing an MO's dominant AO contributions and its overall character."""
    lines = [composition.orbital_id]
    character = ", ".join(
        f"{group} {value:.2f}" for group, value in list(composition.groups.items())[:3]
    )
    lines.append(f"  character: {character}")
    lines.append(f"  {'AO':<18} {'coeff':>8} {'pop':>6}")
    for contribution in composition.dominant(threshold, max_count):
        lines.append(
            f"  {contribution.label:<18} {contribution.coefficient:+8.4f} {contribution.population:6.3f}"
        )
    return lines


def describe_orbital_pairs(
    pairs: Sequence[Tuple[OrbitalComposition, OrbitalComposition]],
    titles: Tuple[str, str] = ("INITIAL", "FINAL"),
    width: int = 46,
) -> List[str]:
    """Describe paired MOs side by side, one block per pair."""
    rule = "-" * (2 * width + 3)
    lines = [f"{titles[0]:<{width}} | {titles[1]}", rule]
    for left, right in pairs:
        left_lines, right_lines = describe_orbital(left), describe_orbital(right)
        rows = max(len(left_lines), len(right_lines))
        left_lines += [""] * (rows - len(left_lines))
        right_lines += [""] * (rows - len(right_lines))
        lines += [f"{a:<{width}} | {b}".rstrip() for a, b in zip(left_lines, right_lines)]
        lines.append(rule)
    return lines
