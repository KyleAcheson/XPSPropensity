"""Atomic-orbital composition of molecular orbitals (Mulliken partition of each MO)."""

from __future__ import annotations

from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from xpsoverlaps.model import Calculation
from xpsoverlaps.overlap import AOOverlapProvider, get_overlap_provider


class AOContribution(NamedTuple):
    label: str
    coefficient: float
    weight: float  # fraction of the (normalized) MO assigned to this AO


class OrbitalComposition(NamedTuple):
    """AO make-up of one MO: its largest AO contributions and its AO-group character."""

    orbital_id: str
    dominant: List[AOContribution]  # largest weights above the threshold, decreasing
    groups: Dict[str, float]  # e.g. {"BR6 p": 0.74, ...}, sorted, non-negative, sums to one


_ANGULAR_LETTERS = "spdfghiklm"


def ao_group(atom_label: str, angular_momentum: int) -> str:
    """AO group key from an atom and an angular momentum, e.g. ``("BR6", 1)`` -> ``"BR6 p"``.

    Groups pool every contraction and component of one l on one atom, which
    is meaningful for any basis set.
    """
    letter = (
        _ANGULAR_LETTERS[angular_momentum]
        if angular_momentum < len(_ANGULAR_LETTERS)
        else f"l={angular_momentum}"
    )
    return f"{atom_label} {letter}"


def _requested_ids(calculation: Calculation, orbital_ids: Optional[Iterable[str]]) -> List[str]:
    return list(calculation.active_space.orbital_ids if orbital_ids is None else orbital_ids)


def orbital_ao_weights(
    calculation: Calculation,
    orbital_ids: Optional[Iterable[str]] = None,
    provider: Optional[AOOverlapProvider] = None,
) -> np.ndarray:
    """AO weights of each MO, shape (n_AO, n_orbitals); each column sums to one.

    ``W[mu, k] = C[mu, k] * (S @ C)[mu, k]`` for the given MOs (default: the
    active orbitals): the Mulliken partition of one normalized orbital, with
    each AO-pair overlap term split equally. These are not electron
    populations; they are not weighted by occupation.
    """
    basis, orbitals = calculation.basis, calculation.orbitals
    if provider is None:
        provider = get_overlap_provider(basis, basis)
    positions = [orbitals.orbital_ids.index(i) for i in _requested_ids(calculation, orbital_ids)]
    coefficients = orbitals.coefficients[:, positions]
    return coefficients * (provider.cross_overlap(basis, basis) @ coefficients)


def orbital_compositions(
    calculation: Calculation,
    orbital_ids: Optional[Iterable[str]] = None,
    provider: Optional[AOOverlapProvider] = None,
    threshold: float = 0.05,
    max_count: int = 8,
) -> Dict[str, OrbitalComposition]:
    """AO composition of the given MOs (default: the active orbitals).

    Group character uses every AO's weight; only the ``max_count`` largest
    contributions with weight >= ``threshold`` are kept individually.
    """
    orbital_ids = _requested_ids(calculation, orbital_ids)
    labels = calculation.basis.function_labels
    weights = orbital_ao_weights(calculation, orbital_ids, provider)
    orbitals = calculation.orbitals
    coefficients = orbitals.coefficients[:, [orbitals.orbital_ids.index(i) for i in orbital_ids]]

    basis, atoms = calculation.basis, calculation.molecule.atoms
    ao_groups = [
        ao_group(atoms[center], l)
        for center, l in zip(basis.function_centers, basis.function_angular_momenta)
    ]
    group_names = list(dict.fromkeys(ao_groups))  # unique, in basis (atom) order
    group_index = {name: index for index, name in enumerate(group_names)}
    group_of_ao = np.array([group_index[name] for name in ao_groups])
    group_weights = np.zeros((len(group_names), len(orbital_ids)))
    np.add.at(group_weights, group_of_ao, weights)

    compositions = {}
    for column, orbital_id in enumerate(orbital_ids):
        column_weights = weights[:, column]
        rows = np.flatnonzero(column_weights >= threshold)
        rows = rows[np.argsort(-column_weights[rows])][:max_count]
        dominant = [
            AOContribution(labels[row], float(coefficients[row, column]), float(column_weights[row]))
            for row in rows
        ]
        compositions[orbital_id] = OrbitalComposition(
            orbital_id, dominant, _normalized_groups(group_names, group_weights[:, column])
        )
    return compositions


def _normalized_groups(names: Sequence[str], weights: np.ndarray) -> Dict[str, float]:
    # Mulliken-partitioned group weights can be slightly negative; clip and renormalize.
    clipped = np.clip(weights, 0.0, None)
    total = clipped.sum() or 1.0
    order = np.argsort(-clipped, kind="stable")
    return {names[index]: float(clipped[index] / total) for index in order}


def composition_similarity(a: OrbitalComposition, b: OrbitalComposition) -> float:
    """Shared AO-group character, ``sum_g min(a_g, b_g)``: 1 identical, 0 disjoint."""
    return float(sum(min(value, b.groups.get(group, 0.0)) for group, value in a.groups.items()))


def describe_orbital(composition: OrbitalComposition) -> List[str]:
    """Text lines listing an MO's dominant AO contributions and its overall character."""
    lines = [composition.orbital_id]
    character = ", ".join(
        f"{group} {value:.2f}" for group, value in list(composition.groups.items())[:3]
    )
    lines.append(f"  character: {character}")
    lines.append(f"  {'AO':<18} {'coeff':>8} {'weight':>6}")
    for contribution in composition.dominant:
        lines.append(
            f"  {contribution.label:<18} {contribution.coefficient:+8.4f} {contribution.weight:6.3f}"
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
