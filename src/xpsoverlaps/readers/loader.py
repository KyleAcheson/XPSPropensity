"""Program-neutral loading of calculations from program-specific files.

``get_calculation`` dispatches to the readers for the requested program and
method; ``attach_states`` combines their normalized results internally.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Sequence, Tuple

from xpsoverlaps.model import Calculation, ElectronicState
from xpsoverlaps.readers.openmolcas import (
    Caspt2Hdf5Reader,
    Caspt2LogReader,
    RasscfHdf5Reader,
    RasscfLogReader,
)


_OPENMOLCAS_READERS: Dict[
    str,
    Tuple[Callable, Callable],
] = {
    "RASSCF": (RasscfHdf5Reader, RasscfLogReader),
    "CASPT2": (Caspt2Hdf5Reader, Caspt2LogReader),
}


def get_calculation(
    program: str,
    calculation_type: str,
    wf_file: str | Path,
    state_file: str | Path | None = None,
) -> Calculation:
    """Read and assemble a calculation using the requested program and method.

    ``wf_file`` holds the wavefunction (geometry, basis, orbitals, energies and,
    for some programs, the CI vectors). ``state_file`` is only needed by
    programs that write state CI data to a separate file.

    OpenMolcas requires both: an HDF5 ``wf_file`` and a log ``state_file``
    (state metadata and dominant CI configurations). Add program-specific
    construction in another branch without changing the returned model.
    """
    normalized_program = program.strip().casefold()
    if normalized_program in {"openmolcas", "molcas"}:
        normalized_type = calculation_type.strip().upper()
        if normalized_type not in _OPENMOLCAS_READERS:
            supported_types = ", ".join(sorted(_OPENMOLCAS_READERS))
            raise ValueError(
                f"Unsupported OpenMolcas calculation type {calculation_type!r}; "
                f"supported types are: {supported_types}."
            )
        if state_file is None:
            raise ValueError(
                "OpenMolcas calculation construction requires a state_file (log file)."
            )

        hdf5_reader_type, log_reader_type = _OPENMOLCAS_READERS[normalized_type]
        calculation = hdf5_reader_type().read(wf_file)
        states = log_reader_type().read(state_file)
        return attach_states(calculation, states)
    else:
        raise NotImplementedError(
            f"Calculation construction for program {program!r} is not implemented."
        )


def attach_states(
    calculation: Calculation,
    states: Sequence[ElectronicState],
    energy_tolerance: float = 1e-6,
) -> Calculation:
    """Attach log-derived state metadata while retaining HDF5 state energies.

    States are matched by position; when both sources supply an energy, they
    must agree within ``energy_tolerance`` (hartree).
    """
    if len(calculation.states) != len(states):
        raise ValueError(
            f"Calculation contains {len(calculation.states)} states but the log contains "
            f"{len(states)} states."
        )

    active_ids = calculation.active_space.orbital_ids
    for calculation_state, log_state in zip(calculation.states, states):
        if (
            calculation_state.energy is not None
            and log_state.energy is not None
            and abs(calculation_state.energy - log_state.energy) > energy_tolerance
        ):
            raise ValueError(
                f"State {calculation_state.state_id} energy {calculation_state.energy:.8f} "
                f"does not match log state {log_state.state_id} energy "
                f"{log_state.energy:.8f}; the HDF5 and log files may not belong to "
                "the same calculation."
            )
        calculation_state.spin = log_state.spin
        calculation_state.symmetry = log_state.symmetry
        configuration = log_state.dominant_configuration
        if configuration is None:
            continue
        if len(configuration.occupations) != len(active_ids):
            raise ValueError(
                f"State {log_state.state_id} dominant configuration has "
                f"{len(configuration.occupations)} occupations; "
                f"the active space has {len(active_ids)} orbitals."
            )
        configuration.orbital_ids = list(active_ids)
        calculation_state.dominant_configuration = configuration
    return calculation
