"""Access to OpenMolcas HDF5 files: raw arrays and strings, no model records.

All Molcas HDF5 format details live here, so the record readers in
``hdf5_reader`` never deal with them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

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
        self._prefix = "DESYM_" if int(self.attribute("NSYM", 1)) > 1 else ""

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

    def mo_coefficients(self) -> np.ndarray:
        """MO coefficients as an AO x MO matrix, with symmetry blocks unpacked."""
        nbas = self.attribute("NBAS")
        if nbas is None:
            raise ValueError(f"{self.path} is missing the Molcas NBAS attribute.")
        blocks = np.asarray(nbas, dtype=int).reshape(-1)
        return np.asarray(square_from_symmetry_blocks(self.dataset("MO_VECTORS"), blocks).T, dtype=float)
