"""OpenMolcas calculation loader: an HDF5 wavefunction file plus a log state file.

A CASPT2 run also takes the RASSCF HDF5 file it was built on (``mcscf_file``):
CASPT2 keeps the RASSCF orbitals but leaves MO_ENERGIES and MO_OCCUPATIONS in
its own HDF5 file empty, so the orbital energies and populations come from
the RASSCF file. Either way the loaded ``OrbitalSet`` carries them in the same
fields; without an ``mcscf_file`` a CASPT2 calculation has none (None).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from xpsoverlaps.model import ActiveIntegrals, Calculation, ElectronicState, OrbitalSet, Provenance
from xpsoverlaps.readers.loader import CalculationLoader
from xpsoverlaps.readers.openmolcas.caspt2_log import read_caspt2_states
from xpsoverlaps.readers.openmolcas.fcidump import read_fcidump
from xpsoverlaps.readers.openmolcas.hdf5_file import MolcasHdf5
from xpsoverlaps.readers.openmolcas.hdf5_reader import (
    read_active_space,
    read_basis,
    read_molecule,
    read_orbital_populations,
    read_orbitals,
    read_state_energies,
)
from xpsoverlaps.readers.openmolcas.rasscf_log import read_rasscf_states

# CASPT2 HDF5 dataset with the energies of its reference (RASSCF) states.
_REFERENCE_ENERGIES = "STATE_REFWF_ENERGIES"

# Calculation type -> (state reader for the log, HDF5 dataset with the same states' energies).
_STATE_SOURCES: Dict[str, Tuple[Callable[[Path], List[ElectronicState]], str]] = {
    "RASSCF": (read_rasscf_states, "ROOT_ENERGIES"),
    "CASPT2": (read_caspt2_states, "STATE_PT2_ENERGIES"),
}


class OpenMolcasLoader(CalculationLoader):
    """Molecule, basis, orbitals and active space come from the HDF5 ``wf_file``;
    states (spin, symmetry, dominant configurations) from the log ``state_file``;
    for CASPT2, orbital energies and populations from the RASSCF ``mcscf_file``.
    Integral files are RASSCF FCIDUMP HDF5 files.
    """

    programs = ("openmolcas", "molcas")
    display_name = "OpenMolcas"
    calculation_types = set(_STATE_SOURCES)

    def load_integrals(self, integral_file: Path, calculation: Calculation) -> ActiveIntegrals:
        return read_fcidump(integral_file, calculation)

    def _load(
        self, kind: str, wf_file: Path, state_file: Optional[Path], mcscf_file: Optional[Path]
    ) -> Calculation:
        if state_file is None:
            raise ValueError("OpenMolcas calculation construction requires a state_file (log file).")
        if mcscf_file is not None and kind != "CASPT2":
            raise ValueError(
                f"An mcscf_file is only used for CASPT2; a {kind} wf_file already contains "
                "the orbital energies and populations."
            )
        read_states, energy_dataset = _STATE_SOURCES[kind]

        with MolcasHdf5(wf_file) as h5:
            molecule = read_molecule(h5)
            basis = read_basis(h5, molecule)
            orbitals = read_orbitals(h5, basis, populations=kind == "RASSCF")
            active_space = read_active_space(h5, orbitals)
            hdf5_energies = read_state_energies(h5, energy_dataset)
            reference_energies = read_state_energies(h5, _REFERENCE_ENERGIES) if mcscf_file else []
        states = _check_against_hdf5_energies(read_states(state_file), hdf5_energies)
        source_files = [str(h5.path), str(state_file.resolve())]
        if mcscf_file is not None:
            orbitals = _with_mcscf_populations(orbitals, mcscf_file, reference_energies)
            source_files.append(str(mcscf_file.resolve()))

        return Calculation(
            provenance=Provenance("OpenMolcas", kind, source_files),
            molecule=molecule,
            basis=basis,
            orbitals=orbitals,
            active_space=active_space,
            states=states,
        )


def _with_mcscf_populations(
    orbitals: OrbitalSet, mcscf_file: Path, reference_energies: Sequence[float], tolerance: float = 1e-6
) -> OrbitalSet:
    """CASPT2 orbitals with the energies and populations of the RASSCF ``mcscf_file``.

    Every CASPT2 reference energy (STATE_REFWF_ENERGIES) must match a RASSCF
    root energy: the check that the CASPT2 run was built on this RASSCF
    wavefunction, and so on its orbitals. MS-CASPT2 may use only some of the
    roots, so each reference energy only has to match one of them.
    """
    with MolcasHdf5(mcscf_file) as mcscf:
        root_energies = read_state_energies(mcscf, "ROOT_ENERGIES")
        energies, occupations = read_orbital_populations(mcscf)
    for energy in reference_energies:
        if not any(abs(energy - root) <= tolerance for root in root_energies):
            raise ValueError(
                f"CASPT2 reference energy {energy:.8f} matches no root energy in the mcscf_file "
                f"{mcscf.path}; it may not be the RASSCF calculation the CASPT2 run was built on."
            )
    if len(occupations) != len(orbitals.orbital_ids):
        raise ValueError(
            f"The mcscf_file {mcscf.path} has {len(occupations)} orbitals; "
            f"the CASPT2 wf_file has {len(orbitals.orbital_ids)}."
        )
    return dataclasses.replace(orbitals, energies=energies, occupations=occupations)


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
