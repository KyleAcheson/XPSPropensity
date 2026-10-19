"""Order of the initial active orbitals: HOMO, HOMO-1, ... and LUMO, LUMO+1, ...

Only RAS2 orbitals get valence labels; RAS1 orbitals (the core orbitals of a
core-hole calculation) are labelled "core", and any RAS3 orbitals "RAS3",
without a rank. Which RAS2 orbitals are occupied and which are virtual comes
from the closed-shell reference state's configuration (2 or 0). Each set is ordered by
the orbitals' state-averaged populations: the occupied orbital with the lowest
population is the HOMO, the virtual with the highest is the LUMO. Molcas
active (pseudo)natural orbitals have no well-defined orbital energies, so
populations are the ordering criterion; the stored energies (diagonal Fock
elements) only break ties and are reported for comparison.

The HOMO and LUMO can be given in the input (1-based MO indices) as an
override: they are moved to the top of their set and the remaining orbitals
keep the population order. The final-state HOMO/LUMO follow from the
active-orbital mapping.
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Tuple

from xpsoverlaps.mapping import OrbitalMapping
from xpsoverlaps.model import Calculation, ElectronicState

# ActiveSpace partition labels: RAS1 holds core orbitals, RAS2 the valence ones.
_CORE_PARTITION = "1"
_VALENCE_PARTITION = "2"


class FrontierOrbitalError(ValueError):
    """Raised when the given HOMO/LUMO are not valid for the calculation."""


class OrbitalOrderError(ValueError):
    """Raised when the active orbitals cannot be ordered."""


class FrontierOrbitals(NamedTuple):
    homo: str  # orbital IDs, e.g. "mo:34"
    lumo: str


class OrbitalOrder(NamedTuple):
    """RAS2 orbitals of the reference, each set ordered from the frontier outwards."""

    occupied: Tuple[str, ...]  # HOMO, HOMO-1, HOMO-2, ...
    virtual: Tuple[str, ...]  # LUMO, LUMO+1, LUMO+2, ...
    reference_state: str  # state whose configuration splits occupied and virtual
    override: Optional[FrontierOrbitals] = None  # HOMO/LUMO given in the input
    by_population: Optional[FrontierOrbitals] = None  # HOMO/LUMO the populations alone give
    unranked: Tuple[Tuple[str, str], ...] = ()  # (orbital ID, label) of RAS1 ("core") and RAS3 orbitals

    @property
    def homo(self) -> str:
        return self.occupied[0]

    @property
    def lumo(self) -> str:
        return self.virtual[0]

    @property
    def frontier(self) -> FrontierOrbitals:
        return FrontierOrbitals(self.homo, self.lumo)

    def is_valence(self, orbital_id: str) -> bool:
        return orbital_id in self.occupied or orbital_id in self.virtual

    def level(self, orbital_id: str) -> int:
        """Position on one scale, higher = above: HOMO-n -> -n, LUMO+m -> m + 1."""
        if orbital_id in self.occupied:
            return -self.occupied.index(orbital_id)
        if orbital_id in self.virtual:
            return self.virtual.index(orbital_id) + 1
        raise OrbitalOrderError(f"{orbital_id} is not a ranked (RAS2) orbital of the reference.")

    def label(self, orbital_id: str) -> str:
        unranked = dict(self.unranked)
        if orbital_id in unranked:
            return unranked[orbital_id]
        level = self.level(orbital_id)
        if level <= 0:
            return "HOMO" if level == 0 else f"HOMO{level}"
        return "LUMO" if level == 1 else f"LUMO+{level - 1}"


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


def orbital_order(
    calculation: Calculation, reference: ElectronicState, override: Optional[FrontierOrbitals] = None
) -> OrbitalOrder:
    """Order the RAS2 orbitals by population, split by the reference configuration.

    ``override`` (the input HOMO/LUMO) is moved to the top of its set. It must
    be an occupied (HOMO) or virtual (LUMO) RAS2 orbital in the reference, or
    the order would contradict the configurations the shake-downs are
    classified from.
    """
    orbitals = calculation.orbitals
    if orbitals.occupations is None:
        raise OrbitalOrderError(
            "The initial calculation has no orbital populations to order the active orbitals by "
            "(for an OpenMolcas CASPT2 calculation, give its RASSCF HDF5 file as mcscf_file)."
        )
    population = dict(zip(orbitals.orbital_ids, orbitals.occupations))
    energy = dict(zip(orbitals.orbital_ids, orbitals.energies or [0.0] * len(orbitals.orbital_ids)))
    active = list(zip(calculation.active_space.orbital_ids, calculation.active_space.partitions,
                      reference.dominant_configuration.occupations))
    valence = [(o, n) for o, partition, n in active if partition == _VALENCE_PARTITION]
    occupied = sorted((o for o, n in valence if n == "2"), key=lambda o: (population[o], -energy[o]))
    virtual = sorted((o for o, n in valence if n == "0"), key=lambda o: (-population[o], energy[o]))
    unranked = tuple((o, "core" if partition == _CORE_PARTITION else f"RAS{partition}")
                     for o, partition, _ in active if partition != _VALENCE_PARTITION)
    if not occupied or not virtual:
        raise OrbitalOrderError(
            f"The reference state {reference.state_id} needs occupied and virtual RAS2 orbitals."
        )
    by_population = FrontierOrbitals(occupied[0], virtual[0])

    if override is not None:
        problems = []
        for name, orbital_id, members, kind in (("HOMO", override.homo, occupied, "occupied"),
                                                ("LUMO", override.lumo, virtual, "virtual")):
            if orbital_id not in members:
                problems.append(
                    f"{name} {orbital_id} is not a {kind} RAS2 orbital in the reference state {reference.state_id}."
                )
            else:
                members.remove(orbital_id)
                members.insert(0, orbital_id)
        if problems:
            raise OrbitalOrderError("Input HOMO/LUMO cannot override the population order: " + " ".join(problems))

    return OrbitalOrder(tuple(occupied), tuple(virtual), reference.state_id, override, by_population, unranked)


def map_frontier(frontier: FrontierOrbitals, mapping: OrbitalMapping) -> FrontierOrbitals:
    """The final-state orbitals matched to the initial HOMO and LUMO."""
    return FrontierOrbitals(mapping.final_id(frontier.homo), mapping.final_id(frontier.lumo))


def _energy_order_differs(order: OrbitalOrder, energy: Dict[str, float]) -> List[str]:
    """Orbitals whose place by energy (from the frontier outwards) differs from their place in ``order``."""
    by_energy = sorted(order.occupied, key=lambda o: -energy[o]) + sorted(order.virtual, key=lambda o: energy[o])
    return [o for o, e in zip(order.occupied + order.virtual, by_energy) if o != e]


def orbital_order_report(
    order: OrbitalOrder, initial: Calculation, final: Calculation, mapping: OrbitalMapping
) -> List[str]:
    """The active orbitals from the highest virtual down: population and energy of each
    initial orbital and of the final orbital it maps onto ("-" where a calculation has none)."""
    initial_population, initial_energy = _populations(initial)
    final_population, final_energy = _populations(final)
    flagged = set(_energy_order_differs(order, initial_energy)) if initial_energy else set()

    lines = [
        "Orbital order",
        "",
        f"Initial RAS2 occupied/virtual from the reference state {order.reference_state.split(':')[-1]}; "
        "ordered by state-averaged",
        "population (lowest occupied = HOMO, highest virtual = LUMO). RAS1 orbitals are core and not ranked.",
        "Final orbitals are those the initial ones map onto. Populations are state-averaged over each",
        "calculation's roots; energies are the stored diagonal Fock elements.",
    ]
    if order.override is None:
        lines.append("HOMO/LUMO from the populations (not given in the input).")
    else:
        for name, given, populated in (("HOMO", order.override.homo, order.by_population.homo),
                                       ("LUMO", order.override.lumo, order.by_population.lumo)):
            agreement = "agrees with the populations" if given == populated else f"populations give {populated}"
            lines.append(f"{name} {given} set by the input ({agreement}).")
    above = [o for o, label in order.unranked if label != "core"]  # RAS3
    core = [o for o, label in order.unranked if label == "core"]
    rows = above + list(reversed(order.virtual)) + list(order.occupied) + core
    lines += [
        "",
        f"{'':<8}  {'Initial':<32}  Final",
        f"{'Label':<8}  {'MO':<8}  {'Population':>10}  {'Energy':>10}  {'MO':<8}  {'Population':>10}  {'Energy':>10}",
    ]
    for orbital_id in rows:
        final_id = mapping.final_id(orbital_id)
        lines.append(
            f"{order.label(orbital_id):<8}  {orbital_id:<8}  "
            f"{_number(initial_population, orbital_id)}  {_number(initial_energy, orbital_id)}  {final_id:<8}  "
            f"{_number(final_population, final_id)}  {_number(final_energy, final_id)}"
            + ("  *" if orbital_id in flagged else "")
        )
    if flagged:
        lines += ["", "*: the initial energy order differs from the population order here."]
    return lines


def _populations(calculation: Calculation) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Orbital ID -> population and -> energy; empty where the calculation has none."""
    orbitals = calculation.orbitals
    return ({} if orbitals.occupations is None else dict(zip(orbitals.orbital_ids, orbitals.occupations)),
            {} if orbitals.energies is None else dict(zip(orbitals.orbital_ids, orbitals.energies)))


def _number(values: Dict[str, float], orbital_id: str) -> str:
    return f"{values[orbital_id]:10.4f}" if orbital_id in values else f"{'-':>10}"
