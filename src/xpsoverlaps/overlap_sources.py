"""Overlap sources: S_ab = <phi_a|psi_b> between initial orbitals, by orbital ID.

psi_b is initial orbital b and phi_a is initial orbital a after relaxation in
the core-ionised final state. A propensity method chooses where these numbers
come from by choosing the overlap source:

- ``MappedOverlaps``: the overlaps method. phi_a is the final-calculation
  orbital the mapping matches to a; absolute values.
- ``CoreHoleOverlaps``: the core-hole potential (CHP). phi_a is psi_a relaxed
  in v = -J_c of one core orbital c; signed. ``FirstOrderCoreHoleOverlaps``
  relaxes to first order, ``AllOrdersCoreHoleOverlaps`` to all orders (the
  eigenvectors of diag(e) - J_c, assigned to initial orbitals by the mapping).

An overlap source knows nothing about shake-downs: ``ShakeDown.overlaps``
pairs it with a shake-down's orbital roles i, j, k.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Mapping, Sequence, Tuple

import numpy as np

from xpsoverlaps.composition import orbital_compositions_from_coefficients
from xpsoverlaps.integrals import CoulombMatrix
from xpsoverlaps.mapping import OrbitalMapping, format_mapping, match_active_orbitals
from xpsoverlaps.model import Calculation
from xpsoverlaps.overlap import get_overlap_provider

# |S_ab| above this: first-order relaxation in the core-hole potential is unreliable.
CHP_OVERLAP_WARNING = 0.5
# All orders: the weight of a relaxed RAS2 orbital on the RAS3 orbitals is reported above this.
CHP_RAS3_WEIGHT = 0.01
# All orders: eigenvalues of diag(e) - J_c closer than this (Eh) leave the relaxed orbitals undefined.
CHP_DEGENERACY = 1e-10


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
    """CHP: phi_a is initial orbital psi_a relaxed in the core-hole potential v = -J_c of core orbital c.

    Both orders treat the initial orbitals as eigenfunctions of diag(e) and use
    only (pq|cc); they differ in how far phi_a is relaxed. Subclasses set the
    report texts and implement ``overlap``, ``describe``, ``warnings`` and
    ``details``.

    Parameters
    ----------
    coulomb : CoulombMatrix
        (pq|cc) of the core orbital c over the orbitals it relaxes.
    energies : mapping of str to float
        Initial orbital energy of each of those orbitals, by orbital ID.
    """

    description: Tuple[str, ...]  # how S_ab is obtained, for the head of the overlap report
    propensity_notes: Tuple[str, ...]  # the same in brief, for the head of the propensity report
    warning_summary: str  # what ``warnings`` counts, for one log warning: "<count> <warning_summary>"

    def __init__(self, coulomb: CoulombMatrix, energies: Mapping[str, float]) -> None:
        self.coulomb = coulomb
        self.energies = energies

    @property
    def core(self) -> str:
        return self.coulomb.core

    def label(self, a: str) -> str:
        """Orbital ``a`` as printed in this source's reports."""
        return a

    @abstractmethod
    def warnings(self) -> List[str]:
        """What makes these overlaps unreliable, one line each."""

    @abstractmethod
    def details(self, role_orbitals: Sequence[str]) -> List[str]:
        """Report lines following the S matrix over ``role_orbitals``."""


class FirstOrderCoreHoleOverlaps(CoreHoleOverlaps):
    """CHP to first order: S_ab = <phi_a|psi_b> = (ab|cc) / (e_b - e_a).

    In the core-hole potential v = -J_c, first-order perturbation theory gives
    phi_a = psi_a + sum_q psi_q <psi_q|v|psi_a> / (e_a - e_q), so
    S_ab = -(ab|cc) / (e_a - e_b) for b != a and S_aa = 1. S is signed and
    antisymmetric; changing an orbital's sign changes S to T S T, which leaves
    every intensity unchanged.
    """

    description = (
        "S_ab = <phi_a|psi_b> = (ab|cc) / (e_b - e_a): initial orbital a relaxed to first order in the",
        "core-hole potential -J_c of core orbital c (rows), with initial orbital b (columns). Signed,",
        "S_ba = -S_ab, S_aa = 1. Coulomb only, so every final spin has the same overlaps.",
    )
    propensity_notes = (
        "S_ab = <phi_a|psi_b> = (ab|cc) / (e_b - e_a), signed (see Core-hole overlaps).",
        "Diagonal S_ii, S_jj, S_kk are 1 to first order.",
    )
    warning_summary = (f"core-hole overlaps |S_ab| above {CHP_OVERLAP_WARNING} (listed under Core-hole "
                       "overlaps): first-order relaxation is unreliable for those orbital pairs.")

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

    def warnings(self) -> List[str]:
        return [f"|S_{a},{b}| = {abs(value):.4f} above {CHP_OVERLAP_WARNING}: first-order relaxation unreliable"
                for a, b, value in self.large()]

    def details(self, role_orbitals: Sequence[str]) -> List[str]:
        return [f"  {warning}" for warning in self.warnings()]

    def _gap(self, a: str, b: str) -> float:
        gap = self.energies[b] - self.energies[a]
        if gap == 0.0:
            raise ValueError(f"Orbitals {a} and {b} are degenerate and coupled by the core-hole potential: "
                             f"S_{a}{b} is undefined.")
        return gap


class AllOrdersCoreHoleOverlaps(CoreHoleOverlaps):
    """CHP to all orders: S_ab = <phi_a|psi_b> = U_ba, U the eigenvectors of diag(e) - J_c.

    The eigenvectors of diag(e) - J_c over the valence (RAS2 and RAS3) orbitals
    are the relaxed orbitals to all orders in v = -J_c (``docs/chp_all_orders.tex``).
    They are named ``chp:1``, ``chp:2``, ... by increasing eigenvalue and assigned
    to the initial orbitals by the orbital mapping of the overlaps method
    (largest overlap, contested orbitals by AO composition, the same warnings),
    each with the phase that makes S_aa > 0. S is orthogonal, not antisymmetric.

    Parameters
    ----------
    coulomb : CoulombMatrix
        (pq|cc) of the core orbital c over the valence orbitals.
    energies : mapping of str to float
        Initial orbital energy of each valence orbital, by orbital ID.
    initial : Calculation
        The initial calculation, for the AO compositions of the initial and relaxed orbitals.
    """

    description = (
        "S_ab = <phi_a|psi_b> = U_ba, U the eigenvectors of diag(e) - J_c over the valence (RAS2, RAS3)",
        "orbitals: initial orbital a relaxed to all orders in the core-hole potential -J_c of core orbital c",
        "(rows), with initial orbital b (columns). Each eigenvector is assigned to an initial orbital by the",
        "orbital mapping (largest overlap, contested orbitals by AO composition), with the phase S_aa > 0;",
        "* marks an orbital whose assignment has a mapping warning. S is orthogonal, not antisymmetric.",
        "Coulomb only, so every final spin has the same overlaps.",
    )
    propensity_notes = (
        "S_ab = <phi_a|psi_b> = U_ba, all orders, signed (see Core-hole overlaps); * marks a warned assignment.",
        "Diagonal S_ii, S_jj, S_kk are set to 1; their actual values are reported.",
    )
    warning_summary = ("mapping warnings for relaxed orbitals (listed under Core-hole overlaps, orbitals "
                       "marked *): their assignment to an initial orbital is ambiguous.")

    def __init__(self, coulomb: CoulombMatrix, energies: Mapping[str, float], initial: Calculation) -> None:
        super().__init__(coulomb, energies)
        ids = coulomb.orbital_ids
        levels, eigenvectors = np.linalg.eigh(np.diag([energies[o] for o in ids]) - coulomb.values)
        relaxed_ids = [f"chp:{n}" for n in range(1, len(ids) + 1)]
        degenerate = np.flatnonzero(np.diff(levels) < CHP_DEGENERACY)
        if degenerate.size:
            pairs = ", ".join(f"{relaxed_ids[n]} and {relaxed_ids[n + 1]} at E = {levels[n]:+.6f}"
                              for n in degenerate)
            raise ValueError(f"diag(e) - J_c of core orbital {coulomb.core} has degenerate eigenvalues ({pairs}): "
                             "the relaxed orbitals are undefined.")
        coefficients = initial.orbitals.coefficients[:, [initial.orbitals.orbital_ids.index(o) for o in ids]]
        provider = get_overlap_provider(initial.basis, initial.basis)
        compositions = (orbital_compositions_from_coefficients(initial, coefficients, ids, provider),
                        orbital_compositions_from_coefficients(initial, coefficients @ eigenvectors,
                                                               relaxed_ids, provider))
        mapping = match_active_orbitals(eigenvectors, ids, relaxed_ids, ids, relaxed_ids, *compositions)
        # The assignment does not depend on phases: fix each eigenvector's so that S_aa > 0, then
        # record the mapping of the phase-fixed eigenvectors.
        for match in mapping.matches:
            column = relaxed_ids.index(match.final_id)
            eigenvectors[:, column] *= np.sign(eigenvectors[ids.index(match.initial_id), column])
        self.relaxed_energies = dict(zip(relaxed_ids, levels))  # eigenvalue E of each relaxed orbital
        self.mapping = match_active_orbitals(eigenvectors, ids, relaxed_ids, ids, relaxed_ids, *compositions)

    def overlap(self, a: str, b: str) -> float:
        return self.mapping.overlap(b, self.mapping.final_id(a))

    def describe(self, a: str, b: str) -> str:
        return f"<phi_{self.label(a)}|psi_{self.label(b)}> (all orders, phi_{self.label(a)} = {self.mapping.final_id(a)})"

    def describe_orbital(self, a: str) -> str:
        final_id = self.mapping.final_id(a)
        return f"{self.label(a)} -> {final_id} (E = {self.relaxed_energies[final_id]:+.4f})"

    def label(self, a: str) -> str:
        """``a``, marked ``*`` when its assignment draws a mapping warning: its own (low overlap,
        close runner-up) or, if it was contested and settled by AO composition, the contest's."""
        match = next(match for match in self.mapping.matches if match.initial_id == a)
        return f"{a}*" if match.warnings or match.assigned_by == "composition" else a

    def warnings(self) -> List[str]:
        return list(self.mapping.warnings)

    def details(self, role_orbitals: Sequence[str]) -> List[str]:
        lines = ["", "Relaxed orbitals chp:n (eigenvectors of diag(e) - J_c, by increasing E), "
                     "assigned to the initial orbitals by MO overlap", ""]
        lines += format_mapping(self.mapping, initial_labels={o: self.label(o) for o in self.coulomb.orbital_ids})
        lines += ["", f"{'Initial':<9} {'e':>9}   {'Relaxed':<8} {'E':>9}"]
        for match in self.mapping.matches:
            lines.append(f"{self.label(match.initial_id):<9} {self.energies[match.initial_id]:+9.4f}   "
                         f"{match.final_id:<8} {self.relaxed_energies[match.final_id]:+9.4f}")
        ras3 = [o for o in self.coulomb.orbital_ids if o not in role_orbitals]
        weights = {a: sum(self.overlap(a, r) ** 2 for r in ras3) for a in role_orbitals}
        if any(weight > CHP_RAS3_WEIGHT for weight in weights.values()):
            lines += ["", f"Weight of the relaxed orbitals on RAS3 orbitals, sum_r S_ar^2, above {CHP_RAS3_WEIGHT}:"]
            lines += [f"  {self.label(a)}: {weight:.4f}" for a, weight in weights.items() if weight > CHP_RAS3_WEIGHT]
        lines.append("")
        if self.mapping.warnings:
            lines.append("Mapping warnings (orbitals marked *):")
            lines += [f"  - {warning}" for warning in self.mapping.warnings]
        else:
            lines.append("No mapping warnings.")
        return lines


def core_hole_overlap_report(
    overlap_sources: Sequence[CoreHoleOverlaps], labels: Mapping[str, str], integral_source: str
) -> List[str]:
    """For each core orbital the matrix S_ab = <phi_a|psi_b> over the role (RAS2) orbitals,
    the keys of ``labels``, rows a (relaxed) and columns b, then the source's details."""
    lines = [
        "Core-hole overlaps",
        "",
        *overlap_sources[0].description,
        f"Integrals (pq|cc): {integral_source}.",
        "Energies e: initial orbital energies (stored diagonal Fock elements).",
    ]
    ids = list(labels)
    for overlap_source in overlap_sources:
        lines += ["", f"Core hole in {overlap_source.core}",
                  f"{'':<8}  {'':<8}" + "".join(f"{overlap_source.label(b):>9}" for b in ids)]
        for a in ids:
            lines.append(f"{labels[a]:<8}  {overlap_source.label(a):<8}"
                         + "".join(f"{overlap_source.overlap(a, b):>+9.4f}" for b in ids))
        lines += overlap_source.details(ids)
    return lines
