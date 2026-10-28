"""Core-hole Coulomb integrals taken from integrals a program has written (e.g. a Molcas FCIDUMP)."""

from __future__ import annotations

from typing import List

import numpy as np

from xpsoverlaps.integrals.base import CoreHoleIntegrals
from xpsoverlaps.model import ActiveIntegrals


class StoredIntegrals(CoreHoleIntegrals):
    """(pq|cc) looked up in ``ActiveIntegrals`` read from an integral file (``get_active_integrals``)."""

    def __init__(self, integrals: ActiveIntegrals) -> None:
        self.integrals = integrals
        self.source = f"integral file {integrals.source}"

    def _compute(self, core_ids: List[str], orbital_ids: List[str]) -> np.ndarray:
        known = self.integrals.orbital_ids
        missing = [o for o in core_ids + orbital_ids if o not in known]
        if missing:
            raise ValueError(f"{', '.join(dict.fromkeys(missing))} not in the integral file's orbitals.")
        cores = [known.index(c) for c in core_ids]
        orbitals = [known.index(p) for p in orbital_ids]
        eri = self.integrals.two_electron
        return np.array([eri[np.ix_(orbitals, orbitals, [c], [c])][:, :, 0, 0] for c in cores])
