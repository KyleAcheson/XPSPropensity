"""Record readers for OpenMolcas HDF5 files (RASSCF and CASPT2 alike).

Each function builds one model record from an open ``MolcasHdf5`` file and
the records it depends on; call them in order:
molecule -> basis -> orbitals -> active space.
"""

from __future__ import annotations

from typing import List

import numpy as np

from xpsoverlaps.model import ActiveSpace, GaussianBasis, Molecule, OrbitalSet
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


def read_orbitals(h5: MolcasHdf5, basis: GaussianBasis) -> OrbitalSet:
    coefficients = h5.mo_coefficients()
    if coefficients.shape[0] != basis.n_functions:
        raise ValueError(
            f"MO coefficient matrix has {coefficients.shape[0]} AO rows; "
            f"the basis has {basis.n_functions} functions."
        )
    return OrbitalSet(
        basis_id=basis.basis_id,
        coefficients=coefficients,
        orbital_ids=[f"mo:{index + 1}" for index in range(coefficients.shape[1])],
        energies=[float(value) for value in np.ravel(h5.dataset("MO_ENERGIES"))],
        occupations=[float(value) for value in np.ravel(h5.dataset("MO_OCCUPATIONS"))],
        source_type_labels=h5.strings("MO_TYPEINDICES"),
    )


def read_active_space(h5: MolcasHdf5, orbitals: OrbitalSet) -> ActiveSpace:
    """RAS orbitals in MO order, which must be RAS1, RAS2, RAS3 like the CI strings."""
    active = [
        (orbital_id, label)
        for orbital_id, label in zip(orbitals.orbital_ids, orbitals.source_type_labels)
        if label in _ACTIVE_TYPES
    ]
    if not active:
        raise ValueError("MO_TYPEINDICES contains no RAS active orbitals.")
    partitions = [label for _, label in active]
    if partitions != sorted(partitions):
        raise ValueError(
            "Active orbitals in MO order are not ordered RAS1, RAS2, RAS3 "
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
