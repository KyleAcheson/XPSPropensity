import dataclasses
from pathlib import Path
import re
import shutil

import h5py
import numpy as np
import pytest

from xpsoverlaps.mapping import map_active_orbitals
from xpsoverlaps.model import PointGroup
from xpsoverlaps.overlap import get_overlap_provider
from xpsoverlaps.readers import get_calculation
from xpsoverlaps.readers.openmolcas import (
    MolcasHdf5,
    check_overlap_matrix,
    read_basis,
    read_caspt2_states,
    read_molecule,
    read_rasscf_states,
)
from xpsoverlaps.symmetry import SymmetryError, check_state_symmetries, configuration_irrep


ROOT = Path(__file__).parents[1]
CS2 = ROOT / "data" / "molcas" / "cs2" / "s2p" / "tz" / "singlet" / "2B2"
BRTHPH = ROOT / "data" / "molcas" / "brthph" / "s2p" / "mb"
C2V = PointGroup(["a1", "b2", "a2", "b1"])  # Molcas order in these files
FINAL_IRREPS = ("a1", "a2", "b1", "b2")


def _final(irrep):
    return get_calculation("molcas", "caspt2", CS2 / f"core-s2p-{irrep}-2b2.caspt2.h5", CS2 / f"core-s2p-{irrep}-2b2.log",
                           mcscf_file=CS2 / f"core-s2p-{irrep}-tz-2b2.rasscf.h5")


@pytest.fixture(scope="module")
def initial():
    # C2v; both initial states are B2 excited states.
    return get_calculation("molcas", "caspt2", CS2 / "valence-s2p-2b2.caspt2.h5", CS2 / "valence-s2p-2b2.log",
                           mcscf_file=CS2 / "valence-s2p-tz-2b2.rasscf.h5")


@pytest.fixture(scope="module")
def finals():
    return {irrep: _final(irrep) for irrep in FINAL_IRREPS}


def _ao_overlap(calculation, other=None):
    other = other or calculation
    return get_overlap_provider(calculation.basis, other.basis).cross_overlap(calculation.basis, other.basis)


# --- Point group -------------------------------------------------------------


def test_c2v_products_follow_the_xor_of_molcas_irrep_indices():
    assert C2V.totally_symmetric == "a1"
    assert C2V.product("b2", "b1") == "a2"
    assert C2V.product("a2", "b1") == "b2"
    assert C2V.product("b2", "b2") == "a1"
    assert C2V.product() == "a1"  # closed shell
    assert configuration_irrep(list("2u2d0"), ["a1", "a2", "b2", "b1", "a1"], C2V) == "b2"  # a2 x b1
    with pytest.raises(ValueError, match="1, 2, 4 or 8 irreps"):
        PointGroup(["a", "b", "c"])
    with pytest.raises(ValueError, match="'e' is not an irrep"):
        C2V.product("e")


# --- AO overlap and desymmetrized MOs ----------------------------------------


@pytest.mark.parametrize("path", [
    CS2 / "valence-s2p-2b2.caspt2.h5", CS2 / "core-s2p-b1-2b2.caspt2.h5", BRTHPH / "valence.rasscf.h5",
])
def test_ao_overlap_reproduces_molcas_with_and_without_symmetry(path):
    with MolcasHdf5(path) as h5:
        basis = read_basis(h5, read_molecule(h5))
        # With symmetry Molcas stores D^T S_AO D (symmetry-adapted functions).
        assert check_overlap_matrix(h5, basis) < 1e-12


def test_desymmetrized_mos_are_orthonormal_over_the_ao_basis(initial, finals):
    for calculation in (initial, *finals.values()):
        C = calculation.orbitals.coefficients
        assert C.shape == (98, 98)
        np.testing.assert_allclose(C.T @ _ao_overlap(calculation) @ C, np.eye(98), atol=1e-10)


def _copy(tmp_path, name, edit):
    path = tmp_path / name
    shutil.copy(CS2 / name, path)
    with h5py.File(path, "r+") as h5_file:
        edit(h5_file)
    return path


def test_a_basis_that_does_not_reproduce_molcas_overlap_is_rejected(tmp_path):
    def swap_desym_columns(h5_file):
        D = h5_file["DESYM_MATRIX"][:].reshape(98, 98, order="F")
        D[:, [13, 14]] = D[:, [14, 13]]
        h5_file["DESYM_MATRIX"][:] = D.reshape(-1, order="F")

    path = _copy(tmp_path, "valence-s2p-2b2.caspt2.h5", swap_desym_columns)
    with pytest.raises(ValueError, match="differs from Molcas's AO_OVERLAP_MATRIX"):
        get_calculation("molcas", "caspt2", path, CS2 / "valence-s2p-2b2.log")


# --- Irreps of MOs and states ------------------------------------------------


def test_mos_keep_the_irrep_of_their_symmetry_block(initial):
    # NBAS = 36, 30, 14, 18 for a1, b2, a2, b1.
    irreps = dict(zip(initial.orbitals.orbital_ids, initial.orbitals.irreps))
    assert initial.point_group.irreps == ["a1", "b2", "a2", "b1"]
    assert [irreps[f"mo:{n}"] for n in (1, 36, 37, 66, 67, 80, 81, 98)] == ["a1", "a1", "b2", "b2", "a2", "a2", "b1", "b1"]


def test_states_get_the_irrep_of_their_calculation(initial, finals):
    assert [state.symmetry for state in initial.states] == ["b2", "b2"]
    for irrep, calculation in finals.items():
        assert {state.symmetry for state in calculation.states} == {irrep}


def test_without_symmetry_everything_is_the_single_irrep():
    calculation = get_calculation("molcas", "rasscf", BRTHPH / "valence.rasscf.h5", BRTHPH / "valence-tz.log")

    assert calculation.point_group.irreps == ["a"]
    assert set(calculation.orbitals.irreps) == {"a"} and {s.symmetry for s in calculation.states} == {"a"}
    assert calculation.active_space.orbital_ids == [f"mo:{n}" for n in range(31, 43)]  # L2ACT is the identity


# --- Active orbitals in CI-string (level) order ------------------------------


def test_active_orbitals_are_in_level_order_matching_the_log_header(initial):
    # The RASSCF log labels each CI-string position with its irrep (1-based).
    header = re.search(r"conf/sym\s+([\d ]+?)\s+Coeff", (CS2 / "valence-s2p-tz-2b2.log").read_text()).group(1)
    irreps = dict(zip(initial.orbitals.orbital_ids, initial.orbitals.irreps))

    assert initial.active_space.orbital_ids == [
        f"mo:{n}" for n in (5, 6, 40, 41, 67, 81, 7, 8, 9, 10, 42, 43, 44, 68, 82, 83)
    ]
    assert initial.active_space.partitions == list("1" * 6 + "2" * 10)
    level_irreps = "".join(str(C2V.irreps.index(irreps[o]) + 1) for o in initial.active_space.orbital_ids)
    assert level_irreps == header.replace(" ", "") == "1122341111222344"


def test_inconsistent_level_maps_are_rejected(tmp_path):
    def break_a2lev(h5_file):
        a2lev = h5_file.attrs["A2LEV"].copy()
        a2lev[[0, 1]] = a2lev[[1, 0]]
        h5_file.attrs["A2LEV"] = a2lev

    path = _copy(tmp_path, "valence-s2p-2b2.caspt2.h5", break_a2lev)
    with pytest.raises(ValueError, match="L2ACT and A2LEV are not inverse"):
        get_calculation("molcas", "caspt2", path, CS2 / "valence-s2p-2b2.log")


def test_files_without_level_maps_keep_the_mo_order(tmp_path):
    path = tmp_path / "valence.rasscf.h5"
    shutil.copy(BRTHPH / "valence.rasscf.h5", path)
    with h5py.File(path, "r+") as h5_file:
        del h5_file.attrs["L2ACT"], h5_file.attrs["A2LEV"]

    calculation = get_calculation("molcas", "rasscf", path, BRTHPH / "valence-tz.log")
    assert calculation.active_space.orbital_ids == [f"mo:{n}" for n in range(31, 43)]


# --- CI strings and state symmetry -------------------------------------------


def test_spaced_ci_strings_are_read_from_both_logs():
    rasscf = read_rasscf_states(CS2 / "valence-s2p-tz-2b2.log")
    caspt2 = read_caspt2_states(CS2 / "valence-s2p-2b2.log")

    # Printed as "22 22 2 2 22u0 2d0 2 20" etc. (spaces at irrep/RAS boundaries).
    assert ["".join(s.dominant_configuration.occupations) for s in rasscf] == ["22222222u02d0220", "2222222200220u2d"]
    assert ["".join(s.dominant_configuration.occupations) for s in caspt2] == ["22222222u02d0220", "2222222200220u2d"]
    assert rasscf[0].symmetry == "2"  # the log's raw irrep number; the loader turns it into "b2"


def test_every_dominant_configuration_has_its_state_symmetry(initial, finals):
    # Loading already checks this; 2 + 4 x 20 states, open shells mapped through the level order.
    for calculation in (initial, *finals.values()):
        check_state_symmetries(calculation)
    assert sum(len(c.states) for c in (initial, *finals.values())) == 82


def test_a_state_whose_configuration_has_another_symmetry_is_rejected(initial):
    states = [dataclasses.replace(initial.states[0], symmetry="a2"), *initial.states[1:]]

    with pytest.raises(SymmetryError, match="caspt2:1 is a2 but its dominant configuration 22222222u02d0220 is b2"):
        check_state_symmetries(dataclasses.replace(initial, states=states))


def test_log_and_hdf5_must_agree_on_the_state_irrep(tmp_path):
    path = _copy(tmp_path, "valence-s2p-2b2.caspt2.h5", lambda h5_file: h5_file.attrs.__setitem__("LSYM", 1))

    with pytest.raises(ValueError, match="state symmetry 2 but the HDF5 file LSYM = 1"):
        get_calculation("molcas", "caspt2", path, CS2 / "valence-s2p-2b2.log")


# --- Overlaps between symmetric calculations ---------------------------------


def test_orbitals_of_different_irreps_do_not_overlap_and_the_mapping_stays_within_irreps(initial, finals):
    final = finals["b1"]
    S = initial.orbitals.coefficients.T @ _ao_overlap(initial, final) @ final.orbitals.coefficients
    initial_irreps = np.array(initial.orbitals.irreps)
    final_irreps = np.array(final.orbitals.irreps)

    assert np.abs(S[initial_irreps[:, None] != final_irreps[None, :]]).max() < 1e-12
    mapping = map_active_orbitals(initial, final)
    irrep_i = dict(zip(initial.orbitals.orbital_ids, initial.orbitals.irreps))
    irrep_f = dict(zip(final.orbitals.orbital_ids, final.orbitals.irreps))
    assert all(irrep_i[m.initial_id] == irrep_f[m.final_id] for m in mapping.matches)
