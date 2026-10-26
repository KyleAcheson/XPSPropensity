"""Symmetry of configurations and states in an Abelian point group.

A configuration's irrep is the direct product of the irreps of its singly
occupied orbitals (closed shells are totally symmetric). Every CSF of a state
has the state's symmetry, so the dominant configuration must reproduce the
irrep the program assigned to the state. ``check_state_symmetries`` checks
this; a mismatch means the active orbitals and the CI strings are not lined up.
"""

from __future__ import annotations

from typing import List, Sequence

from xpsoverlaps.model import Calculation, PointGroup


class SymmetryError(ValueError):
    """Raised when configurations and state symmetries are inconsistent."""


def configuration_irrep(occupations: Sequence[str], orbital_irreps: Sequence[str], group: PointGroup) -> str:
    """Irrep of a configuration: the product of its open-shell (u/d) orbital irreps."""
    return group.product(*(irrep for occupation, irrep in zip(occupations, orbital_irreps) if occupation in "ud"))


def check_state_symmetries(calculation: Calculation) -> None:
    """Each state's dominant configuration has the state's symmetry, else ``SymmetryError``."""
    group = calculation.point_group
    if group is None or calculation.orbitals.irreps is None:
        return
    irrep_of = dict(zip(calculation.orbitals.orbital_ids, calculation.orbitals.irreps))
    active_irreps = [irrep_of[orbital_id] for orbital_id in calculation.active_space.orbital_ids]
    problems: List[str] = []
    for state in calculation.states:
        configuration = state.dominant_configuration
        if state.symmetry is None or configuration is None:
            continue
        irrep = configuration_irrep(configuration.occupations, active_irreps, group)
        if irrep != state.symmetry:
            problems.append(
                f"{state.state_id} is {state.symmetry} but its dominant configuration "
                f"{''.join(configuration.occupations)} is {irrep}"
            )
    if problems:
        raise SymmetryError(
            "Dominant configurations do not match the state symmetries, so the CI strings are not "
            "lined up with the active orbitals: " + "; ".join(problems) + "."
        )
