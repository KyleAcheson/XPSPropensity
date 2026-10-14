"""Program-neutral data model for electronic-structure calculations.

Every loader converts its program's output into one ``Calculation``. All
analysis (overlaps, orbital mapping, propensities) works on this model only,
never on program files.

Structure::

    Calculation
    ├── provenance    Provenance       program, method, source files
    ├── molecule      Molecule         atom labels, coordinates
    ├── basis         AOBasis          one entry per AO (= row of the MO coefficients)
    │                 ├── GaussianBasis    explicit contracted shells (GaussianShell)
    │                 └── EngineBasis      integrals from a program's own engine
    ├── orbitals      OrbitalSet       AO x MO coefficients, one ID per MO
    ├── active_space  ActiveSpace      active orbital IDs, in CI-string order
    └── states        [ElectronicState]
                      └── dominant_configuration   Configuration (0/u/d/2 per active orbital)

How the parts refer to each other (checked when a ``Calculation`` is created):

- ``orbitals.basis_id == basis.basis_id``, and coefficient rows follow the
  basis' AO order.
- ``basis.function_centers`` are indices into ``molecule.atoms``.
- ``active_space.orbital_ids`` are IDs from ``orbitals.orbital_ids``; MOs are
  referred to by ID, not position.
- Each configuration has one occupation per active orbital, in
  ``active_space.orbital_ids`` order.

Units: lengths in bohr, energies in hartree, spin as S.

Records are frozen: build each once (``__post_init__`` validates it) and use
``dataclasses.replace`` to derive a changed copy, which is validated again.
Freezing is shallow, so do not modify the lists or arrays inside a record
either. Generated ``==`` is disabled (``eq=False``) because it fails on numpy
array fields; compare the fields you need instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass(frozen=True, eq=False)
class Molecule:
    atoms: List[str]  # unique atom labels, e.g. "C1", "BR6"
    coordinates: np.ndarray  # (n_atoms, 3)
    units: str = "bohr"


@dataclass(frozen=True, eq=False)
class AOBasis:
    """AO basis in the row order of the MO coefficients.

    Per AO: a display label, its atom (index into ``Molecule.atoms``) and its
    angular momentum l. Analysis uses the structured fields, never the labels.
    Subclasses are basis *representations*; each has an AO overlap provider
    in ``xpsoverlaps.overlap``.
    """

    basis_id: str
    function_labels: List[str]
    function_centers: List[int]
    function_angular_momenta: List[int]

    def __post_init__(self) -> None:
        n_functions = len(self.function_labels)
        if len(self.function_centers) != n_functions or len(self.function_angular_momenta) != n_functions:
            raise ValueError("AO centers and angular momenta must have one entry per AO label.")
        if any(l < 0 for l in self.function_angular_momenta):
            raise ValueError("AO angular momenta must be non-negative.")

    @property
    def n_functions(self) -> int:
        return len(self.function_labels)


@dataclass(frozen=True, eq=False)
class GaussianShell:
    """A contracted Gaussian shell expressed in Cartesian primitive powers.

    Each primitive is ``x^a y^b z^c exp(-alpha r^2)`` with ``a + b + c ==
    angular_momentum`` (L), centred at ``coordinates``.

    - ``contraction_coefficients`` (n_contractions x n_exponents): each row is
      a unit-normalized contraction over radial primitives ``r^L exp(-alpha r^2)``,
      whose normalized overlap is ``(2 sqrt(alpha beta) / (alpha + beta)) ** (L + 1.5)``.
    - ``angular_coefficients`` (n_angular x n_powers): row k writes the shell's
      k-th angular function in the Cartesian ``primitive_powers`` (n_powers x 3),
      normalized so each AO has unit norm. Program ordering, phase and
      normalization conventions are absorbed here.
    - ``basis_function_ids``: AO index of each function, angular-major
      (all contractions of angular function 0, then 1, ...).
    """

    center_index: int
    coordinates: np.ndarray
    angular_momentum: int
    exponents: np.ndarray
    contraction_coefficients: np.ndarray
    angular_coefficients: np.ndarray
    primitive_powers: np.ndarray
    basis_function_ids: List[int]


@dataclass(frozen=True, eq=False)
class GaussianBasis(AOBasis):
    """AO basis given explicitly as contracted Gaussian shells."""

    shells: List[GaussianShell]

    def __post_init__(self) -> None:
        super().__post_init__()
        covered = sorted(i for shell in self.shells for i in shell.basis_function_ids)
        if covered != list(range(self.n_functions)):
            raise ValueError("Gaussian shells must cover every AO exactly once.")


@dataclass(frozen=True, eq=False)
class EngineBasis(AOBasis):
    """AO basis whose integrals come from an electronic-structure engine.

    ``engine`` names the registered overlap backend (e.g. ``"pyscf"``) and
    ``handle`` is whatever that backend needs (e.g. a PySCF ``Mole``). The
    loader must ensure the engine's AO order matches the coefficient rows.
    """

    engine: str
    handle: Any

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.engine:
            raise ValueError("EngineBasis needs an engine name.")


@dataclass(frozen=True, eq=False)
class OrbitalSet:
    """Molecular orbitals: AO rows x MO columns, one ID per MO (e.g. ``"mo:34"``)."""

    basis_id: str
    coefficients: np.ndarray
    orbital_ids: List[str]
    energies: List[float]
    occupations: List[float]
    source_type_labels: List[str]  # the program's orbital type per MO (e.g. Molcas "I", "1", "S")

    def __post_init__(self) -> None:
        if self.coefficients.ndim != 2:
            raise ValueError("Orbital coefficients must be a two-dimensional AO-by-MO matrix.")
        n_ao, n_mo = self.coefficients.shape
        if n_ao == 0 or n_mo == 0:
            raise ValueError("An orbital set must contain at least one AO and one MO.")
        if not len(self.orbital_ids) == len(self.energies) == len(self.occupations) == len(
            self.source_type_labels
        ) == n_mo:
            raise ValueError("Orbital IDs, energies, occupations and type labels must match the MO count.")


@dataclass(frozen=True, eq=False)
class ActiveSpace:
    """Active orbitals in the order of the CI occupation strings (RAS1, RAS2, RAS3)."""

    orbital_ids: List[str]
    partitions: List[str]  # partition label per active orbital
    electron_count: Optional[int] = None

    def __post_init__(self) -> None:
        if len(self.orbital_ids) != len(self.partitions):
            raise ValueError("Each active orbital must have one partition label.")


@dataclass(frozen=True, eq=False)
class Configuration:
    """A CSF: one occupation per active orbital, in active-space order.

    ``occupations`` use ``0``, ``2``, and ``u``/``d`` for singly occupied
    orbitals; position k refers to ``active_space.orbital_ids[k]``.
    """

    configuration_id: str
    occupations: List[str]
    coefficient: Optional[float] = None
    weight: Optional[float] = None
    source_metadata: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if any(occupation not in {"0", "u", "d", "2"} for occupation in self.occupations):
            raise ValueError("CI occupations must use only 0, u, d, or 2.")


@dataclass(frozen=True, eq=False)
class ElectronicState:
    state_id: str
    energy: Optional[float] = None
    energy_units: str = "hartree"
    spin: Optional[float] = None  # S, e.g. 0.5 for a doublet
    symmetry: Optional[str] = None
    dominant_configuration: Optional[Configuration] = None


@dataclass(frozen=True, eq=False)
class Provenance:
    program: str
    method: Optional[str]
    source_files: List[str]
    notes: List[str] = field(default_factory=list)


@dataclass(frozen=True, eq=False)
class Calculation:
    """One calculation in program-neutral form; see the module docstring for the rules."""

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
        if any(not 0 <= center < len(self.molecule.atoms) for center in self.basis.function_centers):
            raise ValueError("Every AO center must index an atom of the molecule.")
        known_ids = set(self.orbitals.orbital_ids)
        if any(orbital_id not in known_ids for orbital_id in self.active_space.orbital_ids):
            raise ValueError("Every active orbital ID must exist in the orbital set.")
        n_active = len(self.active_space.orbital_ids)
        for state in self.states:
            configuration = state.dominant_configuration
            if configuration is not None and len(configuration.occupations) != n_active:
                raise ValueError(
                    f"State {state.state_id} configuration has {len(configuration.occupations)} "
                    f"occupations; the active space has {n_active} orbitals."
                )
