from pathlib import Path

import h5py
import pytest

from xpsoverlaps.readers import CalculationLoader, get_calculation
from xpsoverlaps.readers import loader as loader_module
from xpsoverlaps.model import Calculation, Provenance
from xpsoverlaps.readers.openmolcas import (
    MolcasHdf5,
    OpenMolcasLoader,
    read_active_space,
    read_basis,
    read_molecule,
    read_orbitals,
)


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
    assert len(calculation.states[0].dominant_configuration.occupations) == len(
        calculation.active_space.orbital_ids
    )
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


# --- Loader registration ----------------------------------------------------


@pytest.fixture
def isolated_registry(monkeypatch):
    """Let tests define loaders without leaking them into the global registry."""
    monkeypatch.setattr(loader_module, "_LOADERS", dict(loader_module._LOADERS))


def test_openmolcas_loader_is_registered_under_both_names():
    assert loader_module._LOADERS["openmolcas"] is OpenMolcasLoader
    assert loader_module._LOADERS["molcas"] is OpenMolcasLoader


def test_defining_a_loader_subclass_makes_get_calculation_support_it(isolated_registry):
    class SingleFileLoader(CalculationLoader):
        """A stand-in single-file program: everything comes from wf_file."""

        programs = ("SingleFile",)
        display_name = "SingleFile"
        calculation_types = {"CASSCF"}

        def _load(self, kind, wf_file, state_file, mcscf_file):
            assert state_file is None and mcscf_file is None
            with MolcasHdf5(wf_file) as h5:
                molecule = read_molecule(h5)
                basis = read_basis(h5, molecule)
                orbitals = read_orbitals(h5, basis)
                active_space = read_active_space(h5, orbitals)
            return Calculation(
                Provenance("SingleFile", kind, [str(wf_file)]), molecule, basis, orbitals, active_space
            )

    calculation = get_calculation(" singlefile ", "casscf", DATA / "valence.rasscf.h5")

    assert calculation.basis.n_functions == 239
    with pytest.raises(ValueError, match="Unsupported SingleFile calculation type 'RASSCF'"):
        get_calculation("SingleFile", "RASSCF", DATA / "valence.rasscf.h5")


def test_abstract_intermediate_loaders_are_not_registered(isolated_registry):
    class PartialLoader(CalculationLoader):
        programs = ("partial",)

    assert "partial" not in loader_module._LOADERS


def test_duplicate_program_names_are_rejected(isolated_registry):
    with pytest.raises(ValueError, match="'Molcas' is already registered by OpenMolcasLoader"):

        class ImpostorLoader(CalculationLoader):
            programs = ("Molcas",)
            display_name = "Impostor"
            calculation_types = {"RASSCF"}

            def _load(self, kind, wf_file, state_file, mcscf_file):
                raise AssertionError


def test_unknown_program_lists_supported_programs():
    with pytest.raises(NotImplementedError, match="supported programs are: molcas, openmolcas"):
        get_calculation("ORCA", "CASSCF", DATA / "unused.h5")
