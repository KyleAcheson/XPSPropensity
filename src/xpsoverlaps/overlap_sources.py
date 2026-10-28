"""Overlap sources: S_ab = <phi_a|psi_b> between initial orbitals, by orbital ID.

psi_b is initial orbital b and phi_a is initial orbital a after relaxation in
the core-ionised final state. A propensity method chooses where these numbers
come from by choosing the overlap source:

- ``MappedOverlaps``: the overlaps method. phi_a is the final-calculation
  orbital the mapping matches to a; absolute values.
- ``CoreHoleOverlaps``: the core-hole potential (CHP). phi_a is psi_a relaxed
  to first order in v = -J_c of one core orbital c; signed.

An overlap source knows nothing about shake-downs: ``ShakeDown.overlaps``
pairs it with a shake-down's orbital roles i, j, k.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Mapping, Sequence, Tuple

from xpsoverlaps.integrals import CoulombMatrix
from xpsoverlaps.mapping import OrbitalMapping

# |S_ab| above this: first-order relaxation in the core-hole potential is unreliable.
CHP_OVERLAP_WARNING = 0.5


class OrbitalOverlaps(ABC):
    """S_ab = <phi_a|psi_b> for initial orbital IDs a and b."""

    @abstractmethod
    def overlap(self, a: str, b: str) -> float:
        """S_ab: the FIRST orbital is relaxed (final state), the second initial."""

    @abstractmethod
    def describe(self, a: str, b: str) -> str:
        """How S_ab is obtained, for the report."""

    def describe_orbital(self, a: str) -> str:
        """Initial orbital ``a`` and, if the source has one, the final orbital standing for phi_a."""
        return a


class MappedOverlaps(OrbitalOverlaps):
    """Overlaps method: S_ab = |<phi_final(a)|psi_b>| with the final orbital matched to a."""

    def __init__(self, mapping: OrbitalMapping) -> None:
        self.mapping = mapping

    def overlap(self, a: str, b: str) -> float:
        return abs(self.mapping.overlap(b, self.mapping.final_id(a)))

    def describe(self, a: str, b: str) -> str:
        return f"|<final {self.mapping.final_id(a)} | initial {b}>|"

    def describe_orbital(self, a: str) -> str:
        return f"{a} -> final {self.mapping.final_id(a)}"


class CoreHoleOverlaps(OrbitalOverlaps):
    """CHP: S_ab = <phi_a|psi_b> = (ab|cc) / (e_b - e_a) for core orbital c, to first order.

    In the core-hole potential v = -J_c, first-order perturbation theory gives
    phi_a = psi_a + sum_q psi_q <psi_q|v|psi_a> / (e_a - e_q), so
    S_ab = -(ab|cc) / (e_a - e_b) for b != a and S_aa = 1. S is signed and
    antisymmetric; changing an orbital's sign changes S to T S T, which leaves
    every intensity unchanged.

    Parameters
    ----------
    coulomb : CoulombMatrix
        (pq|cc) of the core orbital c over the orbitals of the shake-downs.
    energies : mapping of str to float
        Initial orbital energy of each of those orbitals, by orbital ID.
    """

    def __init__(self, coulomb: CoulombMatrix, energies: Mapping[str, float]) -> None:
        self.coulomb = coulomb
        self.energies = energies

    @property
    def core(self) -> str:
        return self.coulomb.core

    def overlap(self, a: str, b: str) -> float:
        if a == b:
            return 1.0
        coupling = self.coulomb(a, b)
        return 0.0 if coupling == 0.0 else coupling / self._gap(a, b)

    def describe(self, a: str, b: str) -> str:
        if a == b:
            return f"<phi_{a}|psi_{a}> (first order)"
        c = self.core
        return (f"({a} {b}|{c} {c}) / (e_{b} - e_{a}) = "
                f"{self.coulomb(a, b):+.5f} / {self._gap(a, b):+.5f}")

    def large(self) -> List[Tuple[str, str, float]]:
        """Orbital pairs (a, b), a before b, with |S_ab| above ``CHP_OVERLAP_WARNING``."""
        ids = self.coulomb.orbital_ids
        pairs = [(a, b, self.overlap(a, b)) for n, a in enumerate(ids) for b in ids[n + 1:]]
        return [(a, b, value) for a, b, value in pairs if abs(value) > CHP_OVERLAP_WARNING]

    def _gap(self, a: str, b: str) -> float:
        gap = self.energies[b] - self.energies[a]
        if gap == 0.0:
            raise ValueError(f"Orbitals {a} and {b} are degenerate and coupled by the core-hole potential: "
                             f"S_{a}{b} is undefined.")
        return gap


def core_hole_overlap_report(
    overlap_sources: Sequence[CoreHoleOverlaps], labels: Mapping[str, str], integral_source: str
) -> List[str]:
    """For each core orbital the matrix S_ab = <phi_a|psi_b> over the valence orbitals,
    rows a (relaxed) and columns b, with the pairs above ``CHP_OVERLAP_WARNING`` listed."""
    lines = [
        "Core-hole overlaps",
        "",
        "S_ab = <phi_a|psi_b> = (ab|cc) / (e_b - e_a): initial orbital a relaxed to first order in the",
        "core-hole potential -J_c of core orbital c (rows), with initial orbital b (columns). Signed,",
        "S_ba = -S_ab, S_aa = 1. Coulomb only, so every final spin has the same overlaps.",
        f"Integrals (pq|cc): {integral_source}.",
        "Energies e: initial orbital energies (stored diagonal Fock elements).",
    ]
    for overlap_source in overlap_sources:
        ids = overlap_source.coulomb.orbital_ids
        lines += ["", f"Core hole in {overlap_source.core}",
                  f"{'':<8}  {'':<8}" + "".join(f"{orbital_id:>9}" for orbital_id in ids)]
        for a in ids:
            lines.append(f"{labels.get(a, ''):<8}  {a:<8}"
                         + "".join(f"{overlap_source.overlap(a, b):>+9.4f}" for b in ids))
        for a, b, value in overlap_source.large():
            lines.append(f"  |S_{a},{b}| = {abs(value):.4f} above {CHP_OVERLAP_WARNING}: "
                         "first-order relaxation unreliable")
    return lines
