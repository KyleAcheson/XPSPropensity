"""Overlap sources: S_ab = <phi_a|psi_b> between initial orbitals, by orbital ID.

psi_b is initial orbital b and phi_a is initial orbital a after relaxation in
the core-ionised final state. A propensity method chooses where these numbers
come from by choosing the overlap source:

- ``MappedOverlaps``: the overlaps method. phi_a is the final-calculation
  orbital the mapping matches to a; absolute values.

An overlap source knows nothing about shake-downs: ``ShakeDown.overlaps``
pairs it with a shake-down's orbital roles i, j, k.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from xpsoverlaps.mapping import OrbitalMapping


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
