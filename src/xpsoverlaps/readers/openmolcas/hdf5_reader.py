"""HDF5 readers for OpenMolcas RASSCF and CASPT2 calculations."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from xpsoverlaps.model import (
    ActiveSpace,
    GaussianBasis,
    Calculation,
    ElectronicState,
    Molecule,
    OrbitalSet,
    Provenance,
)
from xpsoverlaps.readers.openmolcas.basis import (
    _decode,
    reconstruct_basis,
    square_from_symmetry_blocks,
)


def _attribute_scalar(file: h5py.File, name: str, default=None):
    if name not in file.attrs:
        return default
    value = np.asarray(file.attrs[name])
    return value.item() if value.size == 1 else value


def _read_calculation(
    source: str | Path,
    energy_dataset: str,
    method: str,
) -> Calculation:
    """Read common Molcas HDF5 fields and states from the specified energy dataset."""
    path = Path(source)
    with h5py.File(path, "r") as h5_file:
        nsym = int(_attribute_scalar(h5_file, "NSYM", 1))
        prefix = "DESYM_" if nsym > 1 else ""
        required = (
            "CENTER_COORDINATES",
            "CENTER_LABELS",
            "BASIS_FUNCTION_IDS",
            "PRIMITIVES",
            "PRIMITIVE_IDS",
            "MO_VECTORS",
            "MO_ENERGIES",
            "MO_OCCUPATIONS",
            "MO_TYPEINDICES",
            energy_dataset,
        )
        center_datasets = {"CENTER_COORDINATES", "CENTER_LABELS", "BASIS_FUNCTION_IDS"}
        missing = [
            prefix + name if name in center_datasets else name
            for name in required
            if (prefix + name if name in center_datasets else name) not in h5_file
        ]
        if missing:
            raise ValueError(f"{path} is missing required Molcas datasets: {', '.join(missing)}")
        if "NBAS" not in h5_file.attrs:
            raise ValueError(f"{path} is missing the required Molcas NBAS attribute.")

        center_coords = np.asarray(h5_file[prefix + "CENTER_COORDINATES"][:], dtype=float)
        center_labels = tuple(_decode(value) for value in h5_file[prefix + "CENTER_LABELS"][:])
        if center_coords.shape != (len(center_labels), 3):
            raise ValueError("Center coordinates and labels have incompatible dimensions.")

        basis_shells, ao_labels = reconstruct_basis(
            centers=center_coords,
            center_labels=center_labels,
            basis_ids=h5_file[prefix + "BASIS_FUNCTION_IDS"][:],
            primitive_data=h5_file["PRIMITIVES"][:],
            primitive_ids=h5_file["PRIMITIVE_IDS"][:],
        )
        basis_id = f"openmolcas:{path.resolve()}:basis"
        basis = GaussianBasis(basis_id, list(ao_labels), shells=list(basis_shells))

        nbas = np.asarray(_attribute_scalar(h5_file, "NBAS"), dtype=int).reshape(-1)
        coefficients = square_from_symmetry_blocks(h5_file["MO_VECTORS"][:], nbas).T
        energies = np.asarray(h5_file["MO_ENERGIES"][:], dtype=float).reshape(-1)
        occupations = np.asarray(h5_file["MO_OCCUPATIONS"][:], dtype=float).reshape(-1)
        raw_types = tuple(_decode(value) for value in h5_file["MO_TYPEINDICES"][:])
        if coefficients.shape[0] != basis.n_functions:
            raise ValueError(
                f"MO coefficient matrix has {coefficients.shape[0]} AO rows; "
                f"the reconstructed basis has {basis.n_functions} functions."
            )
        if coefficients.shape[1] != len(energies) or len(energies) != len(raw_types):
            raise ValueError("MO vectors, energies, and type indices have inconsistent sizes.")
        if len(occupations) != coefficients.shape[1]:
            raise ValueError("MO occupations do not match the MO vector count.")

        orbital_ids = tuple(f"mo:{index + 1}" for index in range(coefficients.shape[1]))
        orbitals = OrbitalSet(
            basis_id=basis_id,
            coefficients=np.asarray(coefficients, dtype=float),
            orbital_ids=list(orbital_ids),
            energies=[float(value) for value in energies],
            occupations=[float(value) for value in occupations],
            source_type_labels=list(raw_types),
        )
        active_indices = [
            index
            for index, label in enumerate(raw_types)
            if label in {"1", "2", "3"}
        ]
        if not active_indices:
            raise ValueError("MO_TYPEINDICES contains no RAS active orbitals.")
        # CI occupation strings list RAS1, then RAS2, then RAS3 orbitals; the
        # active orbitals taken in MO order must follow the same partition order.
        active_types = [raw_types[index] for index in active_indices]
        if active_types != sorted(active_types):
            raise ValueError(
                "Active orbitals in MO order are not ordered RAS1, RAS2, RAS3 "
                f"(found {''.join(active_types)}); occupation strings cannot be "
                "mapped to orbital IDs."
            )
        active_electrons = _attribute_scalar(h5_file, "NACTEL")
        active_space = ActiveSpace(
            orbital_ids=[orbital_ids[index] for index in active_indices],
            partitions=[raw_types[index] for index in active_indices],
            electron_count=int(active_electrons) if active_electrons is not None else None,
        )

        state_energies = np.asarray(h5_file[energy_dataset][:], dtype=float).reshape(-1)
        if "NSTATES" in h5_file.attrs:
            expected_states = int(_attribute_scalar(h5_file, "NSTATES"))
            if len(state_energies) != expected_states:
                raise ValueError(
                    f"{energy_dataset} contains {len(state_energies)} energies; "
                    f"Molcas metadata declares {expected_states} states."
                )
        states = [
            ElectronicState(
                state_id=f"{method.lower()}:{index + 1}",
                energy=float(energy),
            )
            for index, energy in enumerate(state_energies)
        ]

    return Calculation(
        provenance=Provenance(
            program="OpenMolcas",
            method=method,
            source_files=[str(path.resolve())],
        ),
        molecule=Molecule(
            atoms=list(center_labels),
            coordinates=center_coords,
            units="bohr",
        ),
        basis=basis,
        orbitals=orbitals,
        active_space=active_space,
        states=states,
    )


class RasscfHdf5Reader:
    """Read normalized calculation data and RASSCF root energies."""

    def read(self, source: str | Path) -> Calculation:
        return _read_calculation(source, "ROOT_ENERGIES", "RASSCF")


class Caspt2Hdf5Reader:
    """Read normalized calculation data and CASPT2 state energies."""

    def read(self, source: str | Path) -> Calculation:
        return _read_calculation(source, "STATE_PT2_ENERGIES", "CASPT2")
