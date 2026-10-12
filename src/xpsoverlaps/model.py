"""Normalized, program-neutral records for electronic-structure calculations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np


@dataclass
class Molecule:
    atoms: List[str]
    coordinates: np.ndarray
    units: str = "bohr"




@dataclass
class AOBasis:
    """Atomic-orbital basis of a calculation, in MO coefficient row order.

    Subclasses are basis *representations* (not programs); each has an
    ``AOOverlapProvider`` in ``xpsoverlaps.overlap`` that computes AO overlaps.
    """

    basis_id: str
    function_labels: List[str]

    @property
    def n_functions(self) -> int:
        return len(self.function_labels)


@dataclass
class GaussianShell:
    """A contracted Gaussian shell expressed in Cartesian primitive powers.

    Each primitive is ``x^a y^b z^c exp(-alpha r^2)`` with ``a + b + c ==
    angular_momentum`` (L), centred at ``coordinates``. Conventions:

    - ``contraction_coefficients`` (n_contractions x n_exponents): each row
      is a unit-normalized contraction over radial primitives
      ``r^L exp(-alpha r^2)``, whose normalized overlap is
      ``(2 sqrt(alpha beta) / (alpha + beta)) ** (L + 1.5)``.
    - ``angular_coefficients`` (n_angular x n_powers): row k expresses the
      shell's k-th angular function in the Cartesian ``primitive_powers``
      (n_powers x 3), normalized so each resulting AO has unit norm. Program
      ordering, phase and normalization conventions are absorbed here.
    - ``basis_function_ids``: AO index of each function, ordered
      angular-major (all contractions of angular function 0, then 1, ...).
    """

    center_index: int
    coordinates: np.ndarray
    angular_momentum: int
    exponents: np.ndarray
    contraction_coefficients: np.ndarray
    angular_coefficients: np.ndarray
    primitive_powers: np.ndarray
    basis_function_ids: List[int]


@dataclass
class GaussianBasis(AOBasis):
    """AO basis given explicitly as contracted Gaussian shells."""

    shells: List[GaussianShell] = field(default_factory=list)


@dataclass
class EngineBasis(AOBasis):
    """AO basis whose integrals come from an electronic-structure engine.

    ``engine`` names the registered overlap backend (e.g. ``"pyscf"``);
    ``handle`` is whatever that backend needs, e.g. a PySCF ``Mole``. The
    reader must ensure the engine's AO order matches the MO coefficient rows.
    """

    engine: str = ""
    handle: Any = None


@dataclass
class OrbitalSet:
    """Molecular orbitals with AO rows and MO columns."""

    basis_id: str
    coefficients: np.ndarray
    orbital_ids: List[str]
    energies: List[float]
    occupations: List[float]
    source_type_labels: List[str]

    def __post_init__(self) -> None:
        if self.coefficients.ndim != 2:
            raise ValueError("Orbital coefficients must be a two-dimensional AO-by-MO matrix.")
        n_ao, n_mo = self.coefficients.shape
        if n_mo != len(self.orbital_ids):
            raise ValueError("Orbital IDs must correspond one-to-one with coefficient columns.")
        if len(self.energies) != n_mo or len(self.occupations) != n_mo:
            raise ValueError("Orbital energies and occupations must match the MO count.")
        if len(self.source_type_labels) != n_mo:
            raise ValueError("Orbital type labels must match the MO count.")
        if n_ao == 0 or n_mo == 0:
            raise ValueError("An orbital set must contain at least one AO and one MO.")


@dataclass
class ActiveSpace:
    orbital_ids: List[str]
    partitions: List[str]
    electron_count: Optional[int] = None

    def __post_init__(self) -> None:
        if len(self.orbital_ids) != len(self.partitions):
            raise ValueError("Each active orbital must have one partition label.")


@dataclass
class Configuration:
    """A CI configuration; occupations follow its ordered orbital IDs."""

    configuration_id: str
    occupations: List[str]
    coefficient: Optional[float] = None
    weight: Optional[float] = None
    orbital_ids: List[str] = field(default_factory=list)
    source_metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if any(occupation not in {"0", "u", "d", "2"} for occupation in self.occupations):
            raise ValueError("CI occupations must use only 0, u, d, or 2.")
        if self.orbital_ids and len(self.orbital_ids) != len(self.occupations):
            raise ValueError("CI occupations must match the configuration's orbital IDs.")


@dataclass
class ElectronicState:
    state_id: str
    energy: Optional[float] = None
    energy_units: str = "hartree"
    spin: Optional[float] = None
    symmetry: Optional[str] = None
    dominant_configuration: Optional[Configuration] = None


@dataclass
class Provenance:
    program: str
    method: Optional[str]
    source_files: List[str]
    notes: List[str] = field(default_factory=list)


@dataclass
class Calculation:
    """Program-neutral calculation data; contains no parsing or overlap logic."""

    provenance: Provenance
    molecule: Molecule
    basis: AOBasis
    orbitals: OrbitalSet
    active_space: ActiveSpace
    states: List[ElectronicState] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.orbitals.basis_id != self.basis.basis_id:
            raise ValueError("Orbital coefficients and basis must share a basis ID.")
        if self.orbitals.coefficients.shape[0] != self.basis.n_functions:
            raise ValueError("Orbital coefficient rows must match the ordered basis functions.")
        known_ids = set(self.orbitals.orbital_ids)
        if any(orbital_id not in known_ids for orbital_id in self.active_space.orbital_ids):
            raise ValueError("Every active orbital ID must exist in the orbital set.")
