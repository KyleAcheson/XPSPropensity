import dataclasses
from pathlib import Path
import shutil

import h5py
import numpy as np
import pytest

from xpsoverlaps.model import Configuration, EngineBasis, GaussianBasis
from xpsoverlaps.readers import get_calculation
from xpsoverlaps.readers.openmolcas import (
    MolcasHdf5,
    read_active_space,
    read_basis,
    read_caspt2_states,
    read_molecule,
    read_orbitals,
    read_rasscf_states,
    read_state_energies,
)
from xpsoverlaps.readers.openmolcas.basis import square_from_symmetry_blocks


DATA = Path(__file__).parents[1] / "data" / "molcas" / "brthph" / "s2p" / "mb"
HDF5_FILES = ["valence.rasscf.h5", "core.rasscf.h5", "valence_caspt2.caspt2.h5", "core_caspt2.caspt2.h5"]
FILE_PAIRS = [
    ("RASSCF", "valence.rasscf.h5", "valence-tz.log", "ROOT_ENERGIES", 10),
    ("RASSCF", "core.rasscf.h5", "core-tz.log", "ROOT_ENERGIES", 40),
    ("CASPT2", "valence_caspt2.caspt2.h5", "valence_caspt2.log", "STATE_PT2_ENERGIES", 10),
    ("CASPT2", "core_caspt2.caspt2.h5", "core_caspt2.log", "STATE_PT2_ENERGIES", 40),
]


def _records(path):
    with MolcasHdf5(path) as h5:
        molecule = read_molecule(h5)
        basis = read_basis(h5, molecule)
        orbitals = read_orbitals(h5, basis)
        return molecule, basis, orbitals, read_active_space(h5, orbitals)


# --- HDF5 access layer and record readers ---------------------------------


def test_missing_dataset_is_reported_with_its_name():
    with MolcasHdf5(DATA / "valence.rasscf.h5") as h5:
        with pytest.raises(ValueError, match="missing the Molcas dataset NOT_A_DATASET"):
            h5.dataset("NOT_A_DATASET")


@pytest.mark.parametrize("hdf5_name", HDF5_FILES)
def test_hdf5_record_readers(hdf5_name):
    molecule, basis, orbitals, active_space = _records(DATA / hdf5_name)

    assert molecule.atoms == ["C1", "C2", "C3", "C4", "S5", "BR6", "H7", "H8", "H9"]
    assert molecule.coordinates.shape == (9, 3)
    assert molecule.units == "bohr"
    assert basis.n_functions == 239
    assert orbitals.coefficients.shape == (239, 239)
    assert orbitals.basis_id == basis.basis_id
    assert active_space.orbital_ids == [f"mo:{index}" for index in range(31, 43)]
    assert active_space.partitions == list("111222222222")

    with h5py.File(DATA / hdf5_name, "r") as h5_file:
        ao_overlap = square_from_symmetry_blocks(h5_file["AO_OVERLAP_MATRIX"][:], h5_file.attrs["NBAS"])
        assert active_space.electron_count == int(h5_file.attrs["NACTEL"])
    coefficients = orbitals.coefficients
    np.testing.assert_allclose(coefficients.T @ ao_overlap @ coefficients, np.eye(239), atol=1e-7)


def test_basis_contractions_are_normalized():
    _, basis, _, _ = _records(DATA / "valence.rasscf.h5")

    for shell in basis.shells:
        exponents = shell.exponents
        for row in shell.contraction_coefficients:
            norm = sum(row**2)
            for i in range(len(exponents)):
                for j in range(i + 1, len(exponents)):
                    overlap = (
                        2 * (exponents[i] * exponents[j]) ** 0.5 / (exponents[i] + exponents[j])
                    ) ** (1.5 + shell.angular_momentum)
                    norm += 2 * row[i] * row[j] * overlap
            assert norm == pytest.approx(1.0, abs=1e-10)


@pytest.mark.parametrize("kind,hdf5_name,log_name,dataset,n_states", FILE_PAIRS)
def test_read_state_energies(kind, hdf5_name, log_name, dataset, n_states):
    with MolcasHdf5(DATA / hdf5_name) as h5:
        energies = read_state_energies(h5, dataset)
    with h5py.File(DATA / hdf5_name, "r") as h5_file:
        expected = h5_file[dataset][:]

    assert energies == pytest.approx(expected)
    assert len(energies) == n_states


def test_active_orbitals_out_of_ras_order_are_rejected(tmp_path):
    hdf5_path = tmp_path / "reordered.rasscf.h5"
    shutil.copy(DATA / "valence.rasscf.h5", hdf5_path)
    with h5py.File(hdf5_path, "r+") as h5_file:
        types = h5_file["MO_TYPEINDICES"][:]
        # MOs 31-42 are RAS1 x3 then RAS2 x9; swap a RAS1 and a RAS2 label.
        types[30], types[33] = types[33], types[30]
        h5_file["MO_TYPEINDICES"][:] = types

    with pytest.raises(ValueError, match="not ordered RAS1, RAS2, RAS3"):
        _records(hdf5_path)


@pytest.mark.parametrize("hdf5_name", ["valence.rasscf.h5", "core.rasscf.h5"])
def test_ao_metadata_matches_molcas_basis_function_ids(hdf5_name):
    molecule, basis, _, _ = _records(DATA / hdf5_name)
    with h5py.File(DATA / hdf5_name, "r") as h5_file:
        basis_ids = h5_file["BASIS_FUNCTION_IDS"][:]

    assert basis.function_centers == [int(c) - 1 for c in basis_ids[:, 0]]
    assert basis.function_angular_momenta == [abs(int(l)) for l in basis_ids[:, 2]]
    for label, (center, contraction, l, m) in zip(basis.function_labels, basis_ids):
        atom, orbital = label.split()
        assert atom == molecule.atoms[center - 1]
        # Principal number is contraction + l; spherical labels (l > 0) carry m.
        assert orbital.startswith(f"{contraction + l}{'SPDFGH'[l]}")
        if "[m=" in orbital:
            assert orbital.endswith(f"[m={m:+d}]")


# --- Log state readers -----------------------------------------------------


@pytest.mark.parametrize("name,n_states,spin", [("core", 40, 0.5), ("valence", 10, 0.0)])
def test_caspt2_log_states(name, n_states, spin):
    states = read_caspt2_states(DATA / f"{name}_caspt2.log")

    assert [state.state_id for state in states] == [f"caspt2:{i}" for i in range(1, n_states + 1)]
    assert all(state.spin == pytest.approx(spin) for state in states)
    assert all(len(state.dominant_configuration.occupations) == 12 for state in states)


def test_caspt2_log_known_core_dominant_configuration():
    dominant = read_caspt2_states(DATA / "core_caspt2.log")[0].dominant_configuration

    assert dominant.configuration_id == "421"
    assert dominant.occupations == list("22u222222000")
    assert dominant.coefficient == pytest.approx(0.939017)
    assert dominant.weight == pytest.approx(0.881753)


def test_rasscf_log_states():
    states = read_rasscf_states(DATA / "valence-tz.log")

    assert [state.state_id for state in states] == [f"rasscf:{root}" for root in range(1, 11)]
    assert states[0].energy == pytest.approx(-3156.32326124)
    assert states[0].spin == pytest.approx(0.0)
    assert states[0].symmetry == "1"
    first, second = states[0].dominant_configuration, states[1].dominant_configuration
    assert (first.configuration_id, first.occupations) == ("1", list("222222222000"))
    assert first.coefficient == pytest.approx(-0.94966)
    assert first.weight == pytest.approx(0.90185)
    assert (second.configuration_id, second.occupations) == ("3", list("2222222u2d00"))
    assert second.weight == pytest.approx(0.53722)


# --- OpenMolcas loader: records combined into one Calculation ----------------


@pytest.mark.parametrize("kind,hdf5_name,log_name,dataset,n_states", FILE_PAIRS)
def test_loader_combines_hdf5_records_and_log_states(kind, hdf5_name, log_name, dataset, n_states):
    calculation = get_calculation("molcas", kind, DATA / hdf5_name, DATA / log_name)
    with h5py.File(DATA / hdf5_name, "r") as h5_file:
        hdf5_energies = h5_file[dataset][:]

    assert calculation.provenance.source_files == [
        str((DATA / hdf5_name).resolve()),
        str((DATA / log_name).resolve()),
    ]
    assert len(calculation.states) == n_states
    # Energies are the full-precision HDF5 values; everything else comes from the log.
    assert [state.energy for state in calculation.states] == list(hdf5_energies)
    assert all(state.spin is not None for state in calculation.states)
    assert all(len(state.dominant_configuration.occupations) == 12 for state in calculation.states)


def test_orbital_populations_are_normalised_across_calculation_types():
    rasscf = get_calculation("molcas", "RASSCF", DATA / "valence.rasscf.h5", DATA / "valence-tz.log")
    caspt2 = get_calculation(
        "molcas", "CASPT2", DATA / "valence_caspt2.caspt2.h5", DATA / "valence_caspt2.log",
        mcscf_file=DATA / "valence.rasscf.h5",
    )
    with h5py.File(DATA / "valence.rasscf.h5", "r") as h5_file:
        occupations, energies = h5_file["MO_OCCUPATIONS"][:], h5_file["MO_ENERGIES"][:]

    for calculation in (rasscf, caspt2):
        assert calculation.orbitals.occupations == pytest.approx(occupations)
        assert calculation.orbitals.energies == pytest.approx(energies)
    assert caspt2.orbitals.occupations[33] == pytest.approx(1.597, abs=1e-3)  # mo:34, the HOMO
    assert caspt2.provenance.source_files[-1] == str((DATA / "valence.rasscf.h5").resolve())


def test_caspt2_without_mcscf_file_has_no_orbital_populations():
    # The CASPT2 HDF5 MO_ENERGIES / MO_OCCUPATIONS are all zero: not populations.
    caspt2 = get_calculation("molcas", "CASPT2", DATA / "core_caspt2.caspt2.h5", DATA / "core_caspt2.log")

    assert caspt2.orbitals.energies is None and caspt2.orbitals.occupations is None


def test_mcscf_file_must_match_the_caspt2_reference_energies():
    with pytest.raises(ValueError, match="matches no root energy in the mcscf_file"):
        get_calculation(
            "molcas", "CASPT2", DATA / "valence_caspt2.caspt2.h5", DATA / "valence_caspt2.log",
            mcscf_file=DATA / "core.rasscf.h5",
        )


def test_mcscf_file_is_rejected_for_rasscf():
    with pytest.raises(ValueError, match="only used for CASPT2"):
        get_calculation(
            "molcas", "RASSCF", DATA / "valence.rasscf.h5", DATA / "valence-tz.log",
            mcscf_file=DATA / "valence.rasscf.h5",
        )


def test_loader_rejects_log_with_different_state_count():
    with pytest.raises(ValueError, match="HDF5 file contains 10 states but the log contains 40"):
        get_calculation("molcas", "CASPT2", DATA / "valence_caspt2.caspt2.h5", DATA / "core_caspt2.log")


def test_loader_rejects_log_from_a_different_calculation(tmp_path):
    hdf5_path = tmp_path / "shifted.rasscf.h5"
    shutil.copy(DATA / "valence.rasscf.h5", hdf5_path)
    with h5py.File(hdf5_path, "r+") as h5_file:
        h5_file["ROOT_ENERGIES"][:] = h5_file["ROOT_ENERGIES"][:] + 0.5

    with pytest.raises(ValueError, match="does not match .* in the HDF5 file"):
        get_calculation("molcas", "RASSCF", hdf5_path, DATA / "valence-tz.log")


# --- Model validation on a loaded calculation -----------------------------


@pytest.fixture(scope="module")
def calculation():
    return get_calculation("molcas", "RASSCF", DATA / "valence.rasscf.h5", DATA / "valence-tz.log")


def test_calculation_rejects_configuration_of_wrong_length(calculation):
    short = Configuration(configuration_id="bad", occupations=["2"])
    states = [dataclasses.replace(calculation.states[0], dominant_configuration=short)]

    with pytest.raises(ValueError, match="configuration has 1 occupations; the active space has 12"):
        dataclasses.replace(calculation, states=states + calculation.states[1:])


def test_calculation_rejects_ao_centers_outside_the_molecule(calculation):
    centers = list(calculation.basis.function_centers)
    centers[0] = len(calculation.molecule.atoms)

    with pytest.raises(ValueError, match="AO center must index an atom"):
        dataclasses.replace(
            calculation, basis=dataclasses.replace(calculation.basis, function_centers=centers)
        )


def test_model_records_are_immutable_and_replace_revalidates(calculation):
    with pytest.raises(dataclasses.FrozenInstanceError):
        calculation.states = []
    with pytest.raises(ValueError, match="share a basis ID"):
        dataclasses.replace(
            calculation, orbitals=dataclasses.replace(calculation.orbitals, basis_id="other")
        )


def test_ao_basis_metadata_is_validated():
    with pytest.raises(ValueError, match="one entry per AO label"):
        GaussianBasis("b", ["a", "b"], [0], [0, 0], shells=[])
    with pytest.raises(ValueError, match="non-negative"):
        GaussianBasis("b", ["a"], [0], [-1], shells=[])
    with pytest.raises(ValueError, match="cover every AO exactly once"):
        GaussianBasis("b", ["a"], [0], [0], shells=[])
    with pytest.raises(ValueError, match="needs an engine name"):
        EngineBasis("b", ["a"], [0], [0], engine="", handle=None)
