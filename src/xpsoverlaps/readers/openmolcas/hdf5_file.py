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
        self._prefix = "DESYM_" if self.nsym > 1 else ""

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
    def nsym(self) -> int:
        return int(self.attribute("NSYM", 1))

    def basis_counts(self) -> np.ndarray:
        """NBAS: the number of (symmetry-adapted) basis functions per irrep."""
        nbas = self.attribute("NBAS")
        if nbas is None:
            raise ValueError(f"{self.path} is missing the Molcas NBAS attribute.")
        return np.asarray(nbas, dtype=int).reshape(-1)

    def irrep_labels(self) -> List[str]:
        """Irrep labels in Molcas order (one per symmetry block), e.g. ['a1', 'b2', 'a2', 'b1']."""
        labels = self.attribute("IRREP_LABELS")
        if labels is None:
            if self.nsym == 1:
                return ["a"]
            raise ValueError(f"{self.path} is missing the Molcas IRREP_LABELS attribute.")
        labels = [_decode(label) for label in np.atleast_1d(labels)]
        if len(labels) != self.nsym:
            raise ValueError(f"{self.path} has {len(labels)} IRREP_LABELS for NSYM = {self.nsym}.")
        return labels

    def desymmetrization_matrix(self) -> Optional[np.ndarray]:
        """D with D[AO, SO]: symmetry-adapted function SO = sum over AOs of D[AO, SO] * AO.

        None without symmetry. Molcas writes DESYM_MATRIX with the
        desymmetrized (AO) index fastest, i.e. in column-major order.
        """
        if self.nsym == 1:
            return None
        n = int(self.basis_counts().sum())
        values = np.asarray(self.dataset("DESYM_MATRIX"), dtype=float).reshape(-1)
        if values.size != n * n:
            raise ValueError(f"{self.path}: DESYM_MATRIX has {values.size} values for {n} basis functions.")
        return values.reshape(n, n, order="F")

    def mo_coefficients(self) -> np.ndarray:
        """MO coefficients as an AO x MO matrix, with symmetry blocks unpacked.

        With symmetry Molcas stores MO_VECTORS over symmetry-adapted functions;
        they are transformed to the (desymmetrized) AO basis, C_AO = D C_SO.
        """
        coefficients = np.asarray(square_from_symmetry_blocks(self.dataset("MO_VECTORS"), self.basis_counts()).T,
                                  dtype=float)
        desymmetrization = self.desymmetrization_matrix()
        return coefficients if desymmetrization is None else desymmetrization @ coefficients
