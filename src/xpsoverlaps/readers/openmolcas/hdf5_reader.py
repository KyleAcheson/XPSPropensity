"""Record readers for OpenMolcas HDF5 files (RASSCF and CASPT2 alike).

Each function builds one model record from an open ``MolcasHdf5`` file and
the records it depends on; call them in order:
molecule -> basis -> orbitals -> active space.

With symmetry (NSYM > 1) the AO basis is the desymmetrized one (all atoms,
from the DESYM_ datasets), MO coefficients are transformed to it, each MO keeps
the irrep of its symmetry block, and the active orbitals are put in CI-string
(level) order using L2ACT.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from xpsoverlaps.model import ActiveSpace, GaussianBasis, Molecule, OrbitalSet, PointGroup
from xpsoverlaps.overlap import get_overlap_provider
from xpsoverlaps.readers.openmolcas.basis import reconstruct_basis, square_from_symmetry_blocks
from xpsoverlaps.readers.openmolcas.hdf5_file import MolcasHdf5

_ACTIVE_TYPES = ("1", "2", "3")  # MO_TYPEINDICES of RAS1, RAS2, RAS3


def read_molecule(h5: MolcasHdf5) -> Molecule:
    labels = h5.strings("CENTER_LABELS")
    coordinates = np.asarray(h5.dataset("CENTER_COORDINATES"), dtype=float)
    if coordinates.shape != (len(labels), 3):
        raise ValueError("Center coordinates and labels have incompatible dimensions.")
    return Molecule(atoms=labels, coordinates=coordinates, units="bohr")


def read_basis(h5: MolcasHdf5, molecule: Molecule) -> GaussianBasis:
    # Rows are (center, contraction, l, m) with 1-based centers; Cartesian
    # shells store negative l, so |l| is the angular momentum.
    function_ids = np.asarray(h5.dataset("BASIS_FUNCTION_IDS"), dtype=int)
    shells, labels = reconstruct_basis(
        centers=molecule.coordinates,
        center_labels=molecule.atoms,
        basis_ids=function_ids,
        primitive_data=h5.dataset("PRIMITIVES"),
        primitive_ids=h5.dataset("PRIMITIVE_IDS"),
    )
    return GaussianBasis(
        basis_id=f"openmolcas:{h5.path}:basis",
        function_labels=list(labels),
        function_centers=[int(center) - 1 for center in function_ids[:, 0]],
        function_angular_momenta=[abs(int(l)) for l in function_ids[:, 2]],
        shells=list(shells),
    )


def read_point_group(h5: MolcasHdf5) -> PointGroup:
    """The irreps in Molcas order, which multiply as the XOR of their 0-based indices."""
    return PointGroup(irreps=h5.irrep_labels())


def read_state_irrep(h5: MolcasHdf5, group: PointGroup) -> str:
    """Irrep of the calculation's states (LSYM, 1-based): Molcas runs one irrep per calculation."""
    lsym = int(h5.attribute("LSYM", 1))
    if not 1 <= lsym <= len(group.irreps):
        raise ValueError(f"{h5.path}: LSYM = {lsym} is not an irrep of {group.irreps}.")
    return group.irreps[lsym - 1]


def check_overlap_matrix(h5: MolcasHdf5, basis: GaussianBasis, tolerance: float = 1e-8) -> float:
    """Compare our AO overlap with Molcas's AO_OVERLAP_MATRIX; return the largest deviation.

    Molcas stores the overlap over its symmetry-adapted functions, so with
    symmetry our AO overlap is transformed first: S_SO = D^T S_AO D. A
    mismatch means the basis (or D) was not reconstructed correctly.
    """
    overlap = get_overlap_provider(basis, basis).cross_overlap(basis, basis)
    desymmetrization = h5.desymmetrization_matrix()
    if desymmetrization is not None:
        overlap = desymmetrization.T @ overlap @ desymmetrization
    stored = square_from_symmetry_blocks(h5.dataset("AO_OVERLAP_MATRIX"), h5.basis_counts())
    if stored.shape != overlap.shape:
        raise ValueError(f"{h5.path}: AO_OVERLAP_MATRIX is {stored.shape}; the basis gives {overlap.shape}.")
    deviation = float(np.abs(overlap - stored).max())
    if deviation > tolerance:
        raise ValueError(
            f"{h5.path}: the AO overlap computed from the basis differs from Molcas's AO_OVERLAP_MATRIX "
            f"by up to {deviation:.2e}; the basis was not reconstructed correctly."
        )
    return deviation


def read_orbitals(h5: MolcasHdf5, basis: GaussianBasis, populations: bool = True) -> OrbitalSet:
    """MO coefficients and labels; energies and populations only if ``populations``.

    CASPT2 HDF5 files contain MO_ENERGIES and MO_OCCUPATIONS datasets but never
    fill them (they are all zero), so CASPT2 orbitals are read with
    ``populations=False`` and take them from the RASSCF file instead.
    """
    coefficients = h5.mo_coefficients()
    if coefficients.shape[0] != basis.n_functions:
        raise ValueError(
            f"MO coefficient matrix has {coefficients.shape[0]} AO rows; "
            f"the basis has {basis.n_functions} functions."
        )
    energies, occupations = read_orbital_populations(h5) if populations else (None, None)
    # MOs come in symmetry blocks of NBAS functions each, in IRREP_LABELS order.
    irreps = [label for label, count in zip(h5.irrep_labels(), h5.basis_counts()) for _ in range(count)]
    return OrbitalSet(
        basis_id=basis.basis_id,
        coefficients=coefficients,
        orbital_ids=[f"mo:{index + 1}" for index in range(coefficients.shape[1])],
        energies=energies,
        occupations=occupations,
        source_type_labels=h5.strings("MO_TYPEINDICES"),
        irreps=irreps,
    )


def read_orbital_populations(h5: MolcasHdf5) -> Tuple[List[float], List[float]]:
    """Orbital energies and state-averaged populations (MO_ENERGIES, MO_OCCUPATIONS)."""
    return (
        [float(value) for value in np.ravel(h5.dataset("MO_ENERGIES"))],
        [float(value) for value in np.ravel(h5.dataset("MO_OCCUPATIONS"))],
    )


def read_active_space(h5: MolcasHdf5, orbitals: OrbitalSet) -> ActiveSpace:
    """RAS orbitals in CI-string (level) order.

    Molcas numbers active orbitals in MO order (irrep by irrep, RAS1-2-3
    within each irrep), but the CI strings list them by level: all RAS1
    orbitals, then RAS2, then RAS3, irrep by irrep within each. L2ACT maps
    level -> active orbital and A2LEV is its inverse. Without symmetry both
    are the identity; files without them are taken in MO order.
    """
    active = [
        (orbital_id, label)
        for orbital_id, label in zip(orbitals.orbital_ids, orbitals.source_type_labels)
        if label in _ACTIVE_TYPES
    ]
    if not active:
        raise ValueError("MO_TYPEINDICES contains no RAS active orbitals.")
    n_active = len(active)
    level_to_active = h5.attribute("L2ACT")
    if level_to_active is not None:
        level_to_active = [int(value) for value in np.ravel(level_to_active)[:n_active]]
        active_to_level = [int(value) for value in np.ravel(h5.attribute("A2LEV"))[:n_active]]
        if sorted(level_to_active) != list(range(1, n_active + 1)) or any(
            level_to_active[active_to_level[a] - 1] != a + 1 for a in range(n_active)
        ):
            raise ValueError(f"{h5.path}: L2ACT and A2LEV are not inverse permutations of the {n_active} "
                             "active orbitals.")
        active = [active[a - 1] for a in level_to_active]
    partitions = [label for _, label in active]
    if partitions != sorted(partitions):
        raise ValueError(
            "Active orbitals in CI-string order are not ordered RAS1, RAS2, RAS3 "
            f"(found {''.join(partitions)}); occupation strings cannot be mapped to orbitals."
        )
    electrons = h5.attribute("NACTEL")
    return ActiveSpace(
        orbital_ids=[orbital_id for orbital_id, _ in active],
        partitions=partitions,
        electron_count=int(electrons) if electrons is not None else None,
    )


def read_state_energies(h5: MolcasHdf5, dataset: str) -> List[float]:
    """State energies from e.g. ``ROOT_ENERGIES`` (RASSCF) or ``STATE_PT2_ENERGIES`` (CASPT2)."""
    energies = [float(value) for value in np.ravel(h5.dataset(dataset))]
    expected = h5.attribute("NSTATES")
    if expected is not None and len(energies) != int(expected):
        raise ValueError(
            f"{dataset} contains {len(energies)} energies; Molcas metadata declares {expected} states."
        )
    return energies
