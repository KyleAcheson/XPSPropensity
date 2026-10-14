"""HOMO/LUMO of the initial calculation, and their counterparts in the final one.

For now the initial HOMO/LUMO come from the input (1-based MO indices); the
final-state ones follow from the active-orbital mapping. Assigning them from
orbital populations can be added later as another way to build
``FrontierOrbitals``.
"""

from __future__ import annotations

from typing import List, NamedTuple

from xpsoverlaps.mapping import OrbitalMapping
from xpsoverlaps.model import Calculation


class FrontierOrbitalError(ValueError):
    """Raised when the given HOMO/LUMO are not valid for the calculation."""


class FrontierOrbitals(NamedTuple):
    homo: str  # orbital IDs, e.g. "mo:34"
    lumo: str


def frontier_from_indices(calculation: Calculation, homo: int, lumo: int) -> FrontierOrbitals:
    """Validate user-given 1-based MO indices and return the orbital IDs.

    Only checks what must hold for any calculation: both orbitals exist,
    differ, and are active (only active orbitals are mapped). Whether they are
    the right HOMO and LUMO is the user's responsibility; nothing here depends
    on which states the calculation contains.
    """
    orbital_ids = calculation.orbitals.orbital_ids
    active_ids = calculation.active_space.orbital_ids
    problems: List[str] = []
    for name, index in (("HOMO", homo), ("LUMO", lumo)):
        if not 1 <= index <= len(orbital_ids):
            problems.append(f"{name} index {index} is outside 1..{len(orbital_ids)}.")
        elif orbital_ids[index - 1] not in active_ids:
            problems.append(
                f"{name} {orbital_ids[index - 1]} is not an active orbital "
                f"(active: {active_ids[0]} ... {active_ids[-1]})."
            )
    if homo == lumo:
        problems.append(f"HOMO and LUMO are the same orbital ({homo}).")
    if problems:
        raise FrontierOrbitalError("Invalid HOMO/LUMO: " + " ".join(problems))
    return FrontierOrbitals(orbital_ids[homo - 1], orbital_ids[lumo - 1])


def map_frontier(frontier: FrontierOrbitals, mapping: OrbitalMapping) -> FrontierOrbitals:
    """The final-state orbitals matched to the initial HOMO and LUMO."""
    return FrontierOrbitals(mapping.final_id(frontier.homo), mapping.final_id(frontier.lumo))


def frontier_report(initial: FrontierOrbitals, final: FrontierOrbitals) -> List[str]:
    return [
        "Frontier orbitals (initial from input, final from the orbital mapping)",
        "",
        f"HOMO  initial {initial.homo:<8} -> final {final.homo}",
        f"LUMO  initial {initial.lumo:<8} -> final {final.lumo}",
    ]

