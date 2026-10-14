"""OpenMolcas calculation loader: an HDF5 wavefunction file plus a log state file."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from xpsoverlaps.model import Calculation, ElectronicState, Provenance
from xpsoverlaps.readers.loader import CalculationLoader
from xpsoverlaps.readers.openmolcas.caspt2_log import read_caspt2_states
from xpsoverlaps.readers.openmolcas.hdf5_file import MolcasHdf5
from xpsoverlaps.readers.openmolcas.hdf5_reader import (
    read_active_space,
    read_basis,
    read_molecule,
    read_orbitals,
    read_state_energies,
)
from xpsoverlaps.readers.openmolcas.rasscf_log import read_rasscf_states

# Calculation type -> (state reader for the log, HDF5 dataset with the same states' energies).
_STATE_SOURCES: Dict[str, Tuple[Callable[[Path], List[ElectronicState]], str]] = {
    "RASSCF": (read_rasscf_states, "ROOT_ENERGIES"),
    "CASPT2": (read_caspt2_states, "STATE_PT2_ENERGIES"),
}


class OpenMolcasLoader(CalculationLoader):
    """Molecule, basis, orbitals and active space come from the HDF5 ``wf_file``;
    states (spin, symmetry, dominant configurations) from the log ``state_file``.
    """

    programs = ("openmolcas", "molcas")
    display_name = "OpenMolcas"
    calculation_types = set(_STATE_SOURCES)

    def _load(self, kind: str, wf_file: Path, state_file: Optional[Path]) -> Calculation:
        if state_file is None:
            raise ValueError("OpenMolcas calculation construction requires a state_file (log file).")
        read_states, energy_dataset = _STATE_SOURCES[kind]

        with MolcasHdf5(wf_file) as h5:
            molecule = read_molecule(h5)
            basis = read_basis(h5, molecule)
            orbitals = read_orbitals(h5, basis)
            active_space = read_active_space(h5, orbitals)
            hdf5_energies = read_state_energies(h5, energy_dataset)
        states = _check_against_hdf5_energies(read_states(state_file), hdf5_energies)

        return Calculation(
            provenance=Provenance("OpenMolcas", kind, [str(h5.path), str(state_file.resolve())]),
            molecule=molecule,
            basis=basis,
            orbitals=orbitals,
            active_space=active_space,
            states=states,
        )


def _check_against_hdf5_energies(
    states: Sequence[ElectronicState], hdf5_energies: Sequence[float], tolerance: float = 1e-6
) -> List[ElectronicState]:
    """Confirm the log and HDF5 file describe the same states; keep the HDF5 energies.

    Matching energies are the check that the two files belong to the same
    calculation. The HDF5 values are kept because the log prints fewer digits.
    """
    if len(states) != len(hdf5_energies):
        raise ValueError(
            f"The HDF5 file contains {len(hdf5_energies)} states but the log contains {len(states)}."
        )
    checked = []
    for state, energy in zip(states, hdf5_energies):
        if state.energy is not None and abs(state.energy - energy) > tolerance:
            raise ValueError(
                f"State {state.state_id} energy {state.energy:.8f} in the log does not match "
                f"{energy:.8f} in the HDF5 file; the files may not belong to the same calculation."
            )
        checked.append(dataclasses.replace(state, energy=energy))
    return checked
