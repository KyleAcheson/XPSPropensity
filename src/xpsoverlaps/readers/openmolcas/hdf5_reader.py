"""Record readers for OpenMolcas HDF5 files (RASSCF and CASPT2 alike).

Each function builds one model record from an open ``MolcasHdf5`` file and
the records it depends on; call them in order:
molecule -> basis -> orbitals -> active space; the point group needs only the file.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from xpsoverlaps.model import ActiveSpace, GaussianBasis, Molecule, OrbitalSet, PointGroup
from xpsoverlaps.overlap import get_overlap_provider
from xpsoverlaps.readers.openmolcas.basis import reconstruct_basis
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


def check_ao_overlap(h5: MolcasHdf5, basis: GaussianBasis, tolerance: float = 1e-8) -> float:
    """Check our AO overlap against Molcas's AO_OVERLAP_MATRIX; return the largest deviation.

    Molcas stores the overlap over its symmetry-adapted functions, so with
    symmetry ours is transformed first: S_SO = D^T S_AO D. A deviation above
    ``tolerance`` means the basis (or D) was not reconstructed correctly.
    """
    overlap = get_overlap_provider(basis, basis).cross_overlap(basis, basis)
    desymmetrization = h5.desymmetrization()
    if desymmetrization is not None:
        overlap = desymmetrization.T @ overlap @ desymmetrization
    stored = h5.stored_ao_overlap()
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
    # MOs come in symmetry blocks of NBAS each, in irrep order. With symmetry an
    # ID gives the MO's index within its irrep and the irrep number, as in the log.
    labels, counts = h5.irrep_labels(), h5.basis_counts()
    if len(counts) == 1:
        orbital_ids = [f"mo:{index + 1}" for index in range(counts[0])]
    else:
        orbital_ids = [f"mo:{index + 1}_i{irrep + 1}" for irrep, count in enumerate(counts) for index in range(count)]
    return OrbitalSet(
        basis_id=basis.basis_id,
        coefficients=coefficients,
        orbital_ids=orbital_ids,
        energies=energies,
        occupations=occupations,
        source_type_labels=h5.strings("MO_TYPEINDICES"),
        irreps=[label for label, count in zip(labels, counts) for _ in range(count)],
    )


def read_orbital_populations(h5: MolcasHdf5) -> Tuple[List[float], List[float]]:
    """Orbital energies and state-averaged populations (MO_ENERGIES, MO_OCCUPATIONS)."""
    return (
        [float(value) for value in np.ravel(h5.dataset("MO_ENERGIES"))],
        [float(value) for value in np.ravel(h5.dataset("MO_OCCUPATIONS"))],
    )


def read_point_group(h5: MolcasHdf5) -> PointGroup:
    """The irreps in Molcas order, in which irreps i and j (0-based) multiply to irrep i XOR j."""
    irreps = h5.irrep_labels()
    return PointGroup(irreps, [[i ^ j for j in range(len(irreps))] for i in range(len(irreps))])


def read_active_space(h5: MolcasHdf5, orbitals: OrbitalSet) -> ActiveSpace:
    """RAS orbitals in CI-string (level) order, which must be RAS1, RAS2, RAS3.

    Without symmetry this is the MO order; with symmetry the levels group the
    RAS spaces first and the irreps within each (``MolcasHdf5.active_level_order``).
    """
    in_mo_order = [
        (orbital_id, label)
        for orbital_id, label in zip(orbitals.orbital_ids, orbitals.source_type_labels)
        if label in _ACTIVE_TYPES
    ]
    if not in_mo_order:
        raise ValueError("MO_TYPEINDICES contains no RAS active orbitals.")
    active = [in_mo_order[index] for index in h5.active_level_order(len(in_mo_order))]
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
