from pathlib import Path

import h5py
import pytest

from xpsoverlaps.readers import get_calculation


DATA = Path(__file__).parents[1] / "data" / "molcas" / "brthph" / "s2p" / "mb"


@pytest.mark.parametrize(
    "calculation_type,hdf5_path,log_path,n_states,expected_first_id",
    [
        ("RASSCF", "valence.rasscf.h5", "valence-tz.log", 10, "rasscf:1"),
        (
            "CASPT2",
            "valence_caspt2.caspt2.h5",
            "valence_caspt2.log",
            10,
            "caspt2:1",
        ),
        ("RASSCF", "core.rasscf.h5", "core-tz.log", 40, "rasscf:1"),
        ("CASPT2", "core_caspt2.caspt2.h5", "core_caspt2.log", 40, "caspt2:1"),
    ],
)
def test_get_calculation_reads_program_and_method_inputs(
    calculation_type, hdf5_path, log_path, n_states, expected_first_id
):
    hdf5_path = DATA / hdf5_path
    calculation = get_calculation(
        "OpenMolcas",
        calculation_type,
        wf_file=hdf5_path,
        state_file=DATA / log_path,
    )

    assert calculation.provenance.program == "OpenMolcas"
    assert calculation.provenance.method == calculation_type
    assert len(calculation.states) == n_states
    assert calculation.states[0].state_id == expected_first_id
    assert calculation.states[0].dominant_configuration is not None
    assert calculation.states[0].dominant_configuration.orbital_ids == (
        calculation.active_space.orbital_ids
    )
    assert len(calculation.states[0].dominant_configuration.occupations) == 12
    assert calculation.basis.n_functions == 239

    energy_dataset = "ROOT_ENERGIES" if calculation_type == "RASSCF" else "STATE_PT2_ENERGIES"
    with h5py.File(hdf5_path, "r") as h5_file:
        assert calculation.states[0].energy == pytest.approx(h5_file[energy_dataset][0])


def test_get_calculation_requires_openmolcas_state_file():
    with pytest.raises(ValueError, match="requires a state_file"):
        get_calculation("OpenMolcas", "RASSCF", wf_file=DATA / "valence.rasscf.h5")


def test_get_calculation_rejects_unsupported_programs_and_methods():
    with pytest.raises(NotImplementedError, match="program 'ORCA'"):
        get_calculation("ORCA", "RASSCF", DATA / "unused.h5", DATA / "unused.log")

    with pytest.raises(ValueError, match="Unsupported OpenMolcas calculation type"):
        get_calculation(
            "OpenMolcas",
            "CASSCF",
            DATA / "unused.h5",
            DATA / "unused.log",
        )
