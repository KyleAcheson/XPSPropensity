"""Shake-down transitions between initial states, and the orbital roles i, j, k.

A shake-down goes from an initial state a, a single excitation i -> k from the
closed-shell reference, to a lower-energy initial state b reached by moving one
electron. The source of that electron decides the case
(see the propensity-rule notes):

- k -> i, b is the reference ................ 3e GS
- k -> j, a lower virtual (same hole i) ..... 3e ES
- j -> i, j doubly occupied in a (same k) .... 5e ES

When the HOMO takes no part it is a doubly occupied spectator d and is left
out. Only HOMO and LUMO are known, so the orbital order needed for
e_i < e_j < e_k can only be checked when j or i is the LUMO or HOMO; other
pairs are reported as not computable.
"""

from __future__ import annotations

from typing import List, NamedTuple, Optional, Sequence, Tuple

from xpsoverlaps.frontier import FrontierOrbitals
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
    case: Optional[str]  # "3e GS", "3e ES", "5e ES", or None if no case applies
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


def reference_state(calculation: Calculation) -> ElectronicState:
    """The lowest-energy state, which must have a closed-shell dominant configuration."""
    states = [s for s in calculation.states if s.energy is not None and s.dominant_configuration]
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


def find_shake_downs(calculation: Calculation, frontier: FrontierOrbitals) -> List[ShakeDown]:
    """All pairs a -> b (b lower in energy, one electron moved), classified.

    Pairs whose configurations differ by more than one electron are not
    shake-downs and are not listed.
    """
    orbital_ids = calculation.active_space.orbital_ids
    reference = reference_state(calculation)
    ref = reference.dominant_configuration.occupations
    states = [s for s in calculation.states if s.energy is not None and s.dominant_configuration]
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
                          b is reference, frontier)
            )
    return shake_downs


def _classify(a: str, b: str, moved: Tuple[str, str], exc_a: Excitation, exc_b: Excitation,
              b_is_reference: bool, frontier: FrontierOrbitals) -> ShakeDown:
    homo, lumo = frontier
    source, target = moved
    if len(exc_a.holes) != 1 or (not b_is_reference and len(exc_b.holes) != 1):
        return ShakeDown(a, b, moved, None, reason="double excitation: no rule")
    i, k = exc_a.holes[0], exc_a.particles[0]
    spectator = homo if i != homo else None

    if b_is_reference:  # k -> i
        return ShakeDown(a, b, moved, "3e GS", i=i, k=k, d=spectator)

    if exc_b.holes[0] == i:  # k -> j, virtual to virtual
        j = target
        reason = None
        if k == lumo:
            reason = "electron would move up: k is the LUMO, j lies above it"
        elif j != lumo:
            reason = "orbital order unknown: j and k both lie above the LUMO"
        return ShakeDown(a, b, moved, "3e ES", i=i, j=j, k=k, d=spectator, reason=reason)

    # j -> i, the donor j was doubly occupied in a; k is a spin-coupling spectator
    j = source
    reason = None
    if i == homo:
        reason = "electron would move up: i is the HOMO, j lies below it"
    elif j != homo:
        reason = "orbital order unknown: j and i both lie below the HOMO"
    return ShakeDown(a, b, moved, "5e ES", i=i, j=j, k=k, d=homo if j != homo else None, reason=reason)


def shake_down_report(calculation: Calculation, frontier: FrontierOrbitals,
                      shake_downs: Sequence[ShakeDown]) -> List[str]:
    """Initial-state classification and the table of shake-down transitions."""
    orbital_ids = calculation.active_space.orbital_ids
    reference = reference_state(calculation)
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
        f"Orbital order: {orbital_ids[0]} ... {orbital_ids[-1]};  HOMO = {frontier.homo}, LUMO = {frontier.lumo}",
        f"Reference (ground) state: {number(reference.state_id)}",
        "",
        f"{'State':>5}  {'Configuration':<{width}}  Excitation from the reference",
    ]
    for state in sorted(calculation.states, key=lambda s: s.energy if s.energy is not None else 0.0):
        if state.dominant_configuration is None:
            continue
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
