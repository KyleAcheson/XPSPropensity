"""Shake-down transitions between initial states, and the orbital roles i, j, k.

A shake-down goes from an initial state a, a single excitation i -> k from the
closed-shell reference, to a lower-energy initial state b reached by moving one
electron. The source of that electron decides the case
(see the propensity-rule notes):

- k -> i, b is the reference, i the HOMO ..... 3e GS
- k -> i, b is the reference, i below HOMO ... 5e GS (j = HOMO)
- k -> j, a lower virtual (same hole i) ..... 3e ES
- j -> i, j doubly occupied in a (same k) .... 5e ES

In the excited-state cases the HOMO, when it takes no part, is a doubly
occupied spectator d and is left out. In ground-state shake-down from a
HOMO-n pump it is kept as j: the closed-shell final state then depends on it
through the minors M1 and M3.

The orbital order (``OrbitalOrder``, from populations or the input HOMO/LUMO)
checks that the shake-down electron moves down: j below k in 3e ES, j above i
in 5e ES. A pair whose electron would move up, or whose roles include a core
(RAS1) or RAS3 orbital, is reported as not computable.
"""

from __future__ import annotations

from typing import List, NamedTuple, Optional, Sequence, Tuple

from xpsoverlaps.frontier import OrbitalOrder
from xpsoverlaps.model import Calculation, ElectronicState

_ELECTRONS = {"0": 0, "u": 1, "d": 1, "2": 2}


class ShakeDownError(ValueError):
    """Raised when shake-down transitions cannot be identified at all."""


class Excitation(NamedTuple):
    """Orbitals that lose (holes) or gain (particles) electrons, one entry per electron."""

    holes: Tuple[str, ...]
    particles: Tuple[str, ...]


class ShakeDown(NamedTuple):
    initial_state: str  # state a
    lower_state: str  # state b, lower in energy
    moved: Tuple[str, str]  # (from orbital, to orbital) of the shake-down electron
    case: Optional[str]  # "3e GS", "5e GS", "3e ES", "5e ES", or None if no case applies
    i: Optional[str] = None
    j: Optional[str] = None
    k: Optional[str] = None
    d: Optional[str] = None  # HOMO, when it is a doubly occupied spectator
    reason: Optional[str] = None  # why the propensity cannot be computed; None if it can

    @property
    def computable(self) -> bool:
        return self.reason is None


def excitation(occupations: Sequence[str], reference: Sequence[str], orbital_ids: Sequence[str]) -> Excitation:
    """Holes and particles of a configuration relative to ``reference``."""
    holes, particles = [], []
    for orbital_id, now, ref in zip(orbital_ids, occupations, reference):
        change = _ELECTRONS[now] - _ELECTRONS[ref]
        holes += [orbital_id] * max(-change, 0)
        particles += [orbital_id] * max(change, 0)
    return Excitation(tuple(holes), tuple(particles))


def check_max_state(calculation: Calculation, max_state: Optional[int]) -> None:
    """``max_state`` must be a state number of the calculation (1 = first state)."""
    if max_state is not None and not 1 <= max_state <= len(calculation.states):
        raise ShakeDownError(
            f"max_state {max_state} is outside 1..{len(calculation.states)}, "
            "the states of the initial calculation."
        )


def considered_states(calculation: Calculation, max_state: Optional[int] = None) -> List[ElectronicState]:
    """States 1..max_state (all if None) that have an energy and a dominant configuration."""
    check_max_state(calculation, max_state)
    return [s for s in calculation.states[:max_state] if s.energy is not None and s.dominant_configuration]


def reference_state(calculation: Calculation, max_state: Optional[int] = None) -> ElectronicState:
    """The lowest-energy considered state, which must have a closed-shell dominant configuration."""
    states = considered_states(calculation, max_state)
    if not states:
        raise ShakeDownError("No states with an energy and a dominant configuration.")
    reference = min(states, key=lambda state: state.energy)
    occupations = "".join(reference.dominant_configuration.occupations)
    if any(occupation in "ud" for occupation in occupations):
        raise ShakeDownError(
            f"The lowest state {reference.state_id} ({occupations}) is not closed-shell; "
            "shake-down transitions are defined relative to a closed-shell reference."
        )
    return reference


def find_shake_downs(
    calculation: Calculation, order: OrbitalOrder, max_state: Optional[int] = None
) -> List[ShakeDown]:
    """All pairs a -> b among states 1..max_state (b lower in energy, one electron moved).

    Pairs whose configurations differ by more than one electron are not
    shake-downs and are not listed.
    """
    orbital_ids = calculation.active_space.orbital_ids
    reference = reference_state(calculation, max_state)
    ref = reference.dominant_configuration.occupations
    states = considered_states(calculation, max_state)
    excitations = {s.state_id: excitation(s.dominant_configuration.occupations, ref, orbital_ids) for s in states}

    shake_downs = []
    for a in sorted(states, key=lambda state: state.energy):
        for b in sorted(states, key=lambda state: state.energy):
            if b.energy >= a.energy:
                continue
            move = excitation(b.dominant_configuration.occupations, a.dominant_configuration.occupations, orbital_ids)
            if len(move.holes) != 1:
                continue
            moved = (move.holes[0], move.particles[0])
            shake_downs.append(
                _classify(a.state_id, b.state_id, moved, excitations[a.state_id], excitations[b.state_id],
                          b is reference, order)
            )
    return shake_downs


def _classify(a: str, b: str, moved: Tuple[str, str], exc_a: Excitation, exc_b: Excitation,
              b_is_reference: bool, order: OrbitalOrder) -> ShakeDown:
    homo = order.homo
    source, target = moved
    if len(exc_a.holes) != 1 or (not b_is_reference and len(exc_b.holes) != 1):
        return ShakeDown(a, b, moved, None, reason="double excitation: no rule")
    i, k = exc_a.holes[0], exc_a.particles[0]
    spectator = homo if i != homo else None
    unranked = _unranked(order, i=i, k=k)
    if unranked:
        return ShakeDown(a, b, moved, None, i=i, k=k, reason=unranked)

    if b_is_reference:  # k -> i
        if i == homo:
            return ShakeDown(a, b, moved, "3e GS", i=i, k=k)
        return ShakeDown(a, b, moved, "5e GS", i=i, j=homo, k=k) # assumes i < homo and j = homo - what about i < homo-1?

    if exc_b.holes[0] == i:  # k -> j, virtual to virtual
        j = target
        reason = _unranked(order, j=j) or (
            _moves_up(order, j, "j", k, "k") if order.level(j) > order.level(k) else None)
        return ShakeDown(a, b, moved, "3e ES", i=i, j=j, k=k, d=spectator, reason=reason)

    # j -> i, the donor j was doubly occupied in a; k is a spin-coupling spectator
    j = source
    reason = _unranked(order, j=j) or (
        _moves_up(order, i, "i", j, "j") if order.level(j) < order.level(i) else None)
    return ShakeDown(a, b, moved, "5e ES", i=i, j=j, k=k, d=homo if j != homo else None, reason=reason)


def _unranked(order: OrbitalOrder, **roles: str) -> Optional[str]:
    """Why a shake-down involving a core (RAS1) or RAS3 orbital has no rule, or None."""
    for role, orbital_id in roles.items():
        if not order.is_valence(orbital_id):
            return f"{role} = {orbital_id} is a {order.label(orbital_id)} orbital: no rule"
    return None


def _moves_up(order: OrbitalOrder, upper: str, upper_role: str, lower: str, lower_role: str) -> str:
    return (f"electron would move up: {upper_role} ({order.label(upper)}) lies above "
            f"{lower_role} ({order.label(lower)})")


def shake_down_report(calculation: Calculation, order: OrbitalOrder,
                      shake_downs: Sequence[ShakeDown], max_state: Optional[int] = None) -> List[str]:
    """Initial-state classification and the table of shake-down transitions."""
    orbital_ids = calculation.active_space.orbital_ids
    reference = reference_state(calculation, max_state)
    states = considered_states(calculation, max_state)
    ref = reference.dominant_configuration.occupations
    by_id = {state.state_id: state for state in calculation.states}

    def number(state_id: str) -> str:
        return state_id.split(":")[-1]

    def config(state_id: str) -> str:
        return "".join(by_id[state_id].dominant_configuration.occupations)

    width = max(len(orbital_ids), len("Configuration"))

    lines = [
        "Shake-down transitions between initial states",
        "",
        f"Active orbitals: {orbital_ids[0]} ... {orbital_ids[-1]};  HOMO = {order.homo}, LUMO = {order.lumo}",
        f"Reference (ground) state: {number(reference.state_id)};  states considered: "
        f"{number(states[0].state_id)}..{number(states[-1].state_id)}"
        + (f" (max_state = {max_state})" if max_state is not None else ""),
        "",
        f"{'State':>5}  {'Configuration':<{width}}  Excitation from the reference",
    ]
    for state in sorted(states, key=lambda s: s.energy):
        exc = excitation(state.dominant_configuration.occupations, ref, orbital_ids)
        text = (
            "reference" if not exc.holes
            else ", ".join(f"{h} -> {p}" for h, p in zip(exc.holes, exc.particles))
            + ("  (double excitation)" if len(exc.holes) > 1 else "")
        )
        lines.append(f"{number(state.state_id):>5}  {config(state.state_id):<{width}}  {text}")

    lines += [
        "",
        f"{'a -> b':>8}  {'Config a':<{width}}  {'Config b':<{width}}  {'Electron':<15}  "
        f"{'Case':<6}  {'i':<6} {'j':<6} {'k':<6} {'d':<6}  Status",
    ]
    for sd in shake_downs:
        roles = " ".join(f"{(orbital or '-'):<6}" for orbital in (sd.i, sd.j, sd.k, sd.d))
        lines.append(
            f"{number(sd.initial_state) + ' -> ' + number(sd.lower_state):>8}  "
            f"{config(sd.initial_state):<{width}}  {config(sd.lower_state):<{width}}  "
            f"{sd.moved[0] + ' -> ' + sd.moved[1]:<15}  {(sd.case or '-'):<6}  {roles}  "
            f"{'ok' if sd.computable else 'not computable: ' + sd.reason}"
        )
    n_ok = sum(sd.computable for sd in shake_downs)
    lines += [
        "",
        f"{len(shake_downs)} transitions: {n_ok} computable, {len(shake_downs) - n_ok} not computable.",
        "d: HOMO, doubly occupied in both states and left out of the rule.",
    ]
    return lines
