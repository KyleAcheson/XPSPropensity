"""OpenMolcas calculation loader: an HDF5 wavefunction file plus a log state file."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

from xpsoverlaps.model import Calculation
from xpsoverlaps.readers.loader import CalculationLoader, with_states
from xpsoverlaps.readers.openmolcas.caspt2_log import Caspt2LogReader
from xpsoverlaps.readers.openmolcas.hdf5_reader import Caspt2Hdf5Reader, RasscfHdf5Reader
from xpsoverlaps.readers.openmolcas.rasscf_log import RasscfLogReader

# Calculation type -> (HDF5 reader, log reader).
_READERS: Dict[str, Tuple[Callable, Callable]] = {
    "RASSCF": (RasscfHdf5Reader, RasscfLogReader),
    "CASPT2": (Caspt2Hdf5Reader, Caspt2LogReader),
}


class OpenMolcasLoader(CalculationLoader):
    """OpenMolcas writes orbitals and energies to HDF5 and state CI data to the log.

    The HDF5 ``wf_file`` supplies geometry, basis, orbitals and state energies;
    the log ``state_file`` supplies spin, symmetry and dominant configurations.
    """

    programs = ("openmolcas", "molcas")
    display_name = "OpenMolcas"
    calculation_types = set(_READERS)

    def _load(self, kind: str, wf_file: Path, state_file: Optional[Path]) -> Calculation:
        if state_file is None:
            raise ValueError("OpenMolcas calculation construction requires a state_file (log file).")
        hdf5_reader, log_reader = _READERS[kind]
        return with_states(hdf5_reader().read(wf_file), log_reader().read(state_file))
