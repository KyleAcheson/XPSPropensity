"""OpenMolcas FCIDUMP HDF5 files (written by RASSCF): two-electron integrals over the active orbitals.

Format details (``src/rasscf/fcidump*.F90``):

- Molcas numbers the active orbitals 1..NORB irrep by irrep and, within an
  irrep, in MO order after the frozen and inactive orbitals. RASSCF can
  permute these labels (orbital reordering, GAS): ``ORBITAL_INDEX[i]`` is the
  label of the i-th active orbital, while ``ORBITAL_ENERGIES`` stays in
  active-orbital order.
- ``TWO_EL_INT_VALUES`` are RASSCF's TUVX integrals, chemists' (ij|kl), even
  though the dataset description says <ij|kl>. Each 8-fold
  symmetry-unique integral appears once; those below the ``CUTOFF``
  attribute (1e-11) are omitted and are zero here.
- ``ORBITAL_ENERGIES`` are rounded to about five significant figures, so they
  only identify the orbitals; exact energies come from the calculation.

The file holds no MO coefficients: it is matched to a calculation by the
number of active orbitals and electrons and by the orbital energies.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from xpsoverlaps.model import ActiveIntegrals, Calculation
from xpsoverlaps.readers.loader import IntegralFileError
from xpsoverlaps.readers.openmolcas.hdf5_file import MolcasHdf5

# FCIDUMP orbital energies are rounded: agreement within this relative tolerance identifies an orbital.
ENERGY_TOLERANCE = 1e-4

# The 8 index orders of (ij|kl) that give the same real integral.
_PERMUTATIONS = ((0, 1, 2, 3), (1, 0, 2, 3), (0, 1, 3, 2), (1, 0, 3, 2),
                 (2, 3, 0, 1), (3, 2, 0, 1), (2, 3, 1, 0), (3, 2, 1, 0))


def read_fcidump(path: str | Path, calculation: Calculation) -> ActiveIntegrals:
    """The (pq|rs) integrals of a Molcas FCIDUMP HDF5 file, over ``calculation``'s active orbitals.

    Raises ``IntegralFileError`` if the file uses symmetry or does not match
    ``calculation``'s active space and orbital energies.
    """
    path = Path(path)
    with MolcasHdf5(path) as fcidump:
        norb = int(fcidump.attribute("NORB"))
        nelec = fcidump.attribute("NELEC")
        orbsym = np.atleast_1d(fcidump.attribute("ORBSYM"))
        labels = np.asarray(fcidump.dataset("ORBITAL_INDEX"), dtype=int)
        energies = np.asarray(fcidump.dataset("ORBITAL_ENERGIES"), dtype=float)
        indices = np.asarray(fcidump.dataset("TWO_EL_INT_INDEX"), dtype=int)
        values = np.asarray(fcidump.dataset("TWO_EL_INT_VALUES"), dtype=float)

    active = _active_orbitals(calculation)
    problems = []
    if np.any(orbsym != 1):
        problems.append("it was written with symmetry, which is not supported yet")
    if norb != len(active):
        problems.append(f"it has {norb} orbitals; the active space has {len(active)}")
    electrons = calculation.active_space.electron_count
    if nelec is not None and electrons is not None and int(nelec) != electrons:
        problems.append(f"it has {int(nelec)} active electrons; the calculation has {electrons}")
    if not problems and sorted(labels) != list(range(1, norb + 1)):
        problems.append(f"ORBITAL_INDEX is not a permutation of 1..{norb}")
    if not problems:
        problems += _energy_mismatches(calculation, active, energies)
    if problems:
        raise IntegralFileError(f"FCIDUMP file {path} does not match the calculation: " + "; ".join(problems) + ".")

    position = np.empty(norb + 1, dtype=int)  # FCIDUMP label -> active orbital position
    position[labels] = np.arange(norb)
    orbitals = position[indices]
    two_electron = np.zeros((norb,) * 4)
    for order in _PERMUTATIONS:
        two_electron[tuple(orbitals[:, n] for n in order)] = values
    return ActiveIntegrals(active, two_electron, str(path))


def _active_orbitals(calculation: Calculation) -> List[str]:
    """The active orbital IDs, checked to follow MO order without gaps as Molcas numbers them."""
    active = list(calculation.active_space.orbital_ids)
    positions = [calculation.orbitals.orbital_ids.index(orbital_id) for orbital_id in active]
    if positions != list(range(positions[0], positions[0] + len(positions))):
        raise IntegralFileError(
            "FCIDUMP orbitals follow the MO order, but the active orbitals are not consecutive MOs in "
            "CI-string order; they cannot be matched."
        )
    return active


def _energy_mismatches(calculation: Calculation, active: List[str], energies: np.ndarray) -> List[str]:
    if calculation.orbitals.energies is None:
        return ["the calculation has no orbital energies to check it against (CASPT2: give mcscf_file)"]
    ids = calculation.orbitals.orbital_ids
    expected = np.array([calculation.orbitals.energies[ids.index(orbital_id)] for orbital_id in active])
    wrong = ~np.isclose(energies, expected, rtol=ENERGY_TOLERANCE, atol=1e-6)
    return [f"orbital energy of {orbital_id} is {found:.6g}, the calculation has {value:.6g}"
            for orbital_id, found, value, bad in zip(active, energies, expected, wrong) if bad]
