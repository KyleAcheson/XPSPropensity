"""Access to OpenMolcas HDF5 files: raw arrays and strings, no model records.

All Molcas HDF5 format details live here, so the record readers in
``hdf5_reader`` never deal with them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import h5py
import numpy as np

from xpsoverlaps.readers.openmolcas.basis import _decode, square_from_symmetry_blocks

# Datasets Molcas stores with a DESYM_ prefix when the file uses symmetry.
_CENTER_DATASETS = {"CENTER_COORDINATES", "CENTER_LABELS", "BASIS_FUNCTION_IDS"}


class MolcasHdf5:
    """An open OpenMolcas HDF5 file; use as a context manager."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self._file = h5py.File(self.path, "r")
        self._prefix = "DESYM_" if self.symmetry_count > 1 else ""

    def __enter__(self) -> MolcasHdf5:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._file.close()

    def attribute(self, name: str, default: Any = None) -> Any:
        """A file attribute as a Python scalar or array, or ``default`` if absent."""
        if name not in self._file.attrs:
            return default
        value = np.asarray(self._file.attrs[name])
        return value.item() if value.size == 1 else value

    def dataset(self, name: str) -> np.ndarray:
        """A dataset by its plain name (the DESYM_ prefix is added when needed)."""
        key = self._prefix + name if name in _CENTER_DATASETS else name
        if key not in self._file:
            raise ValueError(f"{self.path} is missing the Molcas dataset {key}.")
        return self._file[key][()]

    def strings(self, name: str) -> List[str]:
        """A dataset of fixed-width byte strings, decoded and stripped."""
        return [_decode(value) for value in self.dataset(name)]

    @property
    def symmetry_count(self) -> int:
        """NSYM: the number of irreps of the point group the calculation was run in."""
        return int(self.attribute("NSYM", 1))

    def basis_counts(self) -> List[int]:
        """NBAS: the number of basis functions (and of MOs) in each irrep, in irrep order."""
        nbas = self.attribute("NBAS")
        if nbas is None:
            raise ValueError(f"{self.path} is missing the Molcas NBAS attribute.")
        return [int(count) for count in np.asarray(nbas, dtype=int).reshape(-1)]

    def irrep_labels(self) -> List[str]:
        """IRREP_LABELS in Molcas order, e.g. ["a1", "b2", "a2", "b1"]; ["a"] without symmetry."""
        labels = self.attribute("IRREP_LABELS")
        if labels is None:
            if self.symmetry_count == 1:
                return ["a"]
            raise ValueError(f"{self.path} is missing the Molcas IRREP_LABELS attribute.")
        decoded = [_decode(label) for label in np.atleast_1d(labels)]
        if len(decoded) != self.symmetry_count:
            raise ValueError(f"{self.path} has {len(decoded)} IRREP_LABELS for NSYM = {self.symmetry_count}.")
        return decoded

    def state_irrep_index(self) -> int:
        """0-based index of the irrep of the calculation's states (Molcas LSYM, 1-based)."""
        lsym = int(self.attribute("LSYM", 1))
        if not 1 <= lsym <= self.symmetry_count:
            raise ValueError(f"{self.path}: LSYM = {lsym} is not one of the {self.symmetry_count} irreps.")
        return lsym - 1

    def active_level_order(self, n_active: int) -> List[int]:
        """0-based active-orbital index (in MO order) of each CI-string level.

        The CI strings list the active orbitals by level: RAS1, RAS2, RAS3,
        irrep by irrep within each. In MO order they come irrep by irrep,
        RAS1-2-3 within each irrep. L2ACT maps level -> active orbital
        (1-based, zero-padded); without it the two orders are taken to agree.
        """
        l2act = self.attribute("L2ACT")
        if l2act is None:
            return list(range(n_active))
        order = [int(index) - 1 for index in np.asarray(l2act).reshape(-1)[:n_active]]
        if sorted(order) != list(range(n_active)):
            raise ValueError(f"{self.path}: L2ACT is not a permutation of the {n_active} active orbitals.")
        return order

    def desymmetrization(self) -> Optional[np.ndarray]:
        """D[AO, SO]: each symmetry-adapted function SO = sum over AOs of D[AO, SO] * AO; None without symmetry.

        Molcas writes DESYM_MATRIX with the AO index fastest (column-major).
        """
        if self.symmetry_count == 1:
            return None
        n = sum(self.basis_counts())
        values = np.asarray(self.dataset("DESYM_MATRIX"), dtype=float).reshape(-1)
        if values.size != n * n:
            raise ValueError(f"{self.path}: DESYM_MATRIX has {values.size} values for {n} basis functions.")
        return values.reshape(n, n, order="F")

    def mo_coefficients(self) -> np.ndarray:
        """MO coefficients as an AO x MO matrix, symmetry blocks unpacked, in irrep order.

        With symmetry MO_VECTORS is over symmetry-adapted functions, so it is
        transformed to the AO basis: C_AO = D C_SO.
        """
        coefficients = np.asarray(
            square_from_symmetry_blocks(self.dataset("MO_VECTORS"), self.basis_counts()).T, dtype=float
        )
        desymmetrization = self.desymmetrization()
        return coefficients if desymmetrization is None else desymmetrization @ coefficients

    def stored_ao_overlap(self) -> np.ndarray:
        """Molcas's AO_OVERLAP_MATRIX, unpacked; over symmetry-adapted functions with symmetry."""
        return np.asarray(square_from_symmetry_blocks(self.dataset("AO_OVERLAP_MATRIX"), self.basis_counts()),
                          dtype=float)
