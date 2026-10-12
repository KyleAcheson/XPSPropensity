from pathlib import Path
import shutil

import h5py
import numpy as np
import pytest

from xpsoverlaps.readers.loader import attach_states
from xpsoverlaps.model import Configuration, ElectronicState
from xpsoverlaps.readers.openmolcas import (
    Caspt2LogReader,
    RasscfLogReader,
)
from xpsoverlaps.readers.openmolcas.hdf5_reader import (
    Caspt2Hdf5Reader,
    RasscfHdf5Reader,
)
from xpsoverlaps.readers.openmolcas.basis import square_from_symmetry_blocks


DATA = Path(__file__).parents[1] / "data" / "molcas" / "brthph" / "s2p" / "mb"


@pytest.mark.parametrize(
    "hdf5_name,expected_states",
    [("valence_caspt2.caspt2.h5", 10), ("core_caspt2.caspt2.h5", 40)],
)
def test_caspt2_hdf5_reader_uses_pt2_state_energies_and_shared_fields(
    hdf5_name, expected_states
):
    hdf5_path = DATA / hdf5_name
    calculation = Caspt2Hdf5Reader().read(hdf5_path)

    assert calculation.provenance.method == "CASPT2"
    assert calculation.provenance.source_files == [str(hdf5_path.resolve())]
    assert calculation.molecule.atoms == ["C1", "C2", "C3", "C4", "S5", "BR6", "H7", "H8", "H9"]
    assert calculation.molecule.coordinates.shape == (9, 3)
    assert calculation.molecule.units == "bohr"
    assert calculation.basis.n_functions == 239
    assert calculation.orbitals.coefficients.shape == (239, 239)
    assert len(calculation.orbitals.orbital_ids) == 239
    assert calculation.active_space.orbital_ids == [
        f"mo:{index}" for index in range(31, 43)
    ]
    assert len(calculation.states) == expected_states
    assert [state.state_id for state in calculation.states] == [
        f"caspt2:{index}" for index in range(1, expected_states + 1)
    ]

    with h5py.File(hdf5_path, "r") as h5_file:
        pt2_energies = h5_file["STATE_PT2_ENERGIES"][:]
        assert "ROOT_ENERGIES" not in h5_file
    assert [state.energy for state in calculation.states] == pytest.approx(pt2_energies)
    assert all(state.dominant_configuration is None for state in calculation.states)


@pytest.mark.parametrize("calculation_name,expected_roots", [("core", 40), ("valence", 10)])
def test_rasscf_hdf5_reader_normalizes_tz_orbitals_basis_and_active_space(
    calculation_name, expected_roots
):
    hdf5_path = DATA / f"{calculation_name}.rasscf.h5"
    calculation = RasscfHdf5Reader().read(hdf5_path)

    assert len(calculation.molecule.atoms) == 9
    assert calculation.molecule.atoms[0] == "C1"
    assert calculation.molecule.coordinates.shape == (9, 3)
    assert calculation.molecule.units == "bohr"
    assert calculation.basis.n_functions == 239
    assert len(calculation.basis.shells) > 0
    assert calculation.orbitals.coefficients.shape == (239, 239)
    assert len(calculation.orbitals.orbital_ids) == 239
    assert calculation.orbitals.basis_id == calculation.basis.basis_id
    assert len(calculation.active_space.orbital_ids) == 12
    assert len(calculation.states) == expected_roots
    assert all(state.dominant_configuration is None for state in calculation.states)

    for shell in calculation.basis.shells:
        coefficients = shell.contraction_coefficients
        exponents = shell.exponents
        for row in coefficients:
            norm = sum(row**2)
            for i in range(len(exponents)):
                for j in range(i + 1, len(exponents)):
                    overlap = (
                        2 * (exponents[i] * exponents[j]) ** 0.5 / (exponents[i] + exponents[j])
                    ) ** (1.5 + shell.angular_momentum)
                    norm += 2 * row[i] * row[j] * overlap
            assert norm == pytest.approx(1.0, abs=1e-10)

    ordered_ids = sorted(
        function_id
        for shell in calculation.basis.shells
        for function_id in shell.basis_function_ids
    )
    assert ordered_ids == list(range(calculation.basis.n_functions))

    with h5py.File(hdf5_path, "r") as h5_file:
        ao_overlap = square_from_symmetry_blocks(
            h5_file["AO_OVERLAP_MATRIX"][:], h5_file.attrs["NBAS"]
        )
        assert calculation.active_space.electron_count == int(h5_file.attrs["NACTEL"])
    mo_overlap = calculation.orbitals.coefficients.T @ ao_overlap @ calculation.orbitals.coefficients
    assert np.allclose(mo_overlap, np.eye(239), atol=1e-7)


@pytest.mark.parametrize(
    "calculation_name,expected_roots,expected_spin",
    [("core", 40, 0.5), ("valence", 10, 0.0)],
)
def test_caspt2_log_reader_preserves_printed_mixed_state_ci(
    calculation_name, expected_roots, expected_spin
):
    states = Caspt2LogReader().read(DATA / f"{calculation_name}_caspt2.log")

    assert len(states) == expected_roots
    assert states[0].state_id == "mixed:1"
    assert states[0].energy is not None
    assert states[0].spin == pytest.approx(expected_spin)
    assert all(
        state.dominant_configuration is not None
        for state in states
    )
    assert all(len(state.dominant_configuration.occupations) == 12 for state in states)


def test_caspt2_reader_preserves_known_core_dominant_configuration():
    states = Caspt2LogReader().read(DATA / "core_caspt2.log")
    dominant = states[0].dominant_configuration

    assert dominant is not None
    assert dominant.configuration_id == "421"
    assert dominant.occupations == list("22u222222000")
    assert dominant.coefficient == pytest.approx(0.939017)
    assert dominant.weight == pytest.approx(0.881753)
    assert dominant.orbital_ids == []


def test_rasscf_log_reader_reads_roots_and_dominant_ci_vectors():
    states = RasscfLogReader().read(DATA / "valence-tz.log")

    assert len(states) == 10
    assert [state.state_id for state in states] == [
        f"rasscf:{root}" for root in range(1, 11)
    ]
    assert states[0].energy == pytest.approx(-3156.32326124)
    assert states[0].spin == pytest.approx(0.0)
    assert states[0].symmetry == "1"
    assert states[0].dominant_configuration.configuration_id == "1"
    assert states[0].dominant_configuration.occupations == list("222222222000")
    assert states[0].dominant_configuration.coefficient == pytest.approx(-0.94966)
    assert states[0].dominant_configuration.weight == pytest.approx(0.90185)
    assert states[1].dominant_configuration.configuration_id == "3"
    assert states[1].dominant_configuration.occupations == list("2222222u2d00")
    assert states[1].dominant_configuration.weight == pytest.approx(0.53722)
    assert all(
        state.dominant_configuration is not None
        and len(state.dominant_configuration.occupations) == 12
        for state in states
    )


@pytest.mark.parametrize(
    "hdf5_reader,hdf5_path,log_reader,log_path",
    [
        (RasscfHdf5Reader, "valence.rasscf.h5", RasscfLogReader, "valence-tz.log"),
        (Caspt2Hdf5Reader, "valence_caspt2.caspt2.h5", Caspt2LogReader, "valence_caspt2.log"),
        (RasscfHdf5Reader, "core.rasscf.h5", RasscfLogReader, "core-tz.log"),
        (Caspt2Hdf5Reader, "core_caspt2.caspt2.h5", Caspt2LogReader, "core_caspt2.log"),
    ],
)
def test_assembly_binds_ci_occupations_to_active_orbital_ids(
    hdf5_reader, hdf5_path, log_reader, log_path
):
    calculation = hdf5_reader().read(DATA / hdf5_path)
    states = log_reader().read(DATA / log_path)

    assembled = attach_states(calculation, states)
    assert assembled.states[0].dominant_configuration.orbital_ids == calculation.active_space.orbital_ids
    assert len(assembled.states[0].dominant_configuration.orbital_ids) == 12
    assert [state.energy for state in assembled.states] == pytest.approx(
        [state.energy for state in states], abs=1e-6
    )


def test_assembly_rejects_states_from_a_different_calculation():
    calculation = RasscfHdf5Reader().read(DATA / "valence.rasscf.h5")
    states = Caspt2LogReader().read(DATA / "valence_caspt2.log")

    with pytest.raises(ValueError, match="does not match log state"):
        attach_states(calculation, states)


def test_rasscf_hdf5_reader_rejects_active_orbitals_out_of_ras_order(tmp_path):
    hdf5_path = tmp_path / "reordered.rasscf.h5"
    shutil.copy(DATA / "valence.rasscf.h5", hdf5_path)
    with h5py.File(hdf5_path, "r+") as h5_file:
        types = h5_file["MO_TYPEINDICES"][:]
        # MOs 31-42 are RAS1 x3 then RAS2 x9; swap a RAS1 and a RAS2 label.
        types[30], types[33] = types[33], types[30]
        h5_file["MO_TYPEINDICES"][:] = types

    with pytest.raises(ValueError, match="not ordered RAS1, RAS2, RAS3"):
        RasscfHdf5Reader().read(hdf5_path)


def test_assembly_rejects_configuration_with_incompatible_orbital_count():
    calculation = RasscfHdf5Reader().read(DATA / "core.rasscf.h5")
    states = list(calculation.states)
    states[0].dominant_configuration = Configuration(
        configuration_id="bad",
        occupations=["2"],
    )

    with pytest.raises(ValueError, match="occupations"):
        attach_states(calculation, states)


def test_assembly_rejects_different_state_counts():
    calculation = RasscfHdf5Reader().read(DATA / "core.rasscf.h5")

    with pytest.raises(ValueError, match="contains 40 states.*log contains 1"):
        attach_states(calculation, [ElectronicState(state_id="rasscf:1")])


@pytest.mark.parametrize("hdf5_name", ["valence.rasscf.h5", "core.rasscf.h5"])
def test_ao_labels_match_molcas_basis_function_ids(hdf5_name):
    calculation = RasscfHdf5Reader().read(DATA / hdf5_name)
    with h5py.File(DATA / hdf5_name, "r") as h5_file:
        basis_ids = h5_file["BASIS_FUNCTION_IDS"][:]
        center_labels = [label.decode().strip() for label in h5_file["CENTER_LABELS"][:]]

    l_letters = "SPDFGH"
    for label, (center, contraction, l, m) in zip(calculation.basis.function_labels, basis_ids):
        atom, orbital = label.split()
        assert atom == center_labels[center - 1]
        # Principal number is contraction + l; spherical labels (l > 0) carry m.
        assert orbital.startswith(f"{contraction + l}{l_letters[l]}")
        if "[m=" in orbital:
            assert orbital.endswith(f"[m={m:+d}]")
