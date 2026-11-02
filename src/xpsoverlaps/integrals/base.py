"""Abstract interface shared by every source of core-hole Coulomb integrals."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, NamedTuple, Sequence

import numpy as np


class CoulombMatrix(NamedTuple):
    """(pq|cc) for one core orbital c: the matrix of the Coulomb operator J_c of its density."""

    core: str
    orbital_ids: List[str]  # row and column order of values
    values: np.ndarray  # (n, n), symmetric

    def __call__(self, p: str, q: str) -> float:
        return float(self.values[self.orbital_ids.index(p), self.orbital_ids.index(q)])


class CoreHoleIntegrals(ABC):
    """Computes (pq|cc) for core orbitals c and pairs of orbitals p, q, by orbital ID.

    Subclasses set ``source`` (for the report) and implement ``_compute``;
    ``coulomb`` checks the result's shape and symmetry for every source.
    """

    source: str

    def coulomb(self, core_ids: Sequence[str], orbital_ids: Sequence[str]) -> Dict[str, CoulombMatrix]:
        """One ``CoulombMatrix`` per core orbital, over ``orbital_ids``."""
        core_ids, orbital_ids = list(core_ids), list(orbital_ids)
        values = np.asarray(self._compute(core_ids, orbital_ids), dtype=float)
        expected = (len(core_ids), len(orbital_ids), len(orbital_ids))
        if values.shape != expected:
            raise ValueError(f"{type(self).__name__} returned (pq|cc) of shape {values.shape}; expected {expected}.")
        if not np.allclose(values, values.transpose(0, 2, 1), rtol=0.0, atol=1e-10):
            raise ValueError(f"{type(self).__name__} returned (pq|cc) that is not symmetric in p, q.")
        return {core: CoulombMatrix(core, orbital_ids, block) for core, block in zip(core_ids, values)}

    @abstractmethod
    def _compute(self, core_ids: List[str], orbital_ids: List[str]) -> np.ndarray:
        """(pq|cc) as an array [c, p, q] in the given orders."""
