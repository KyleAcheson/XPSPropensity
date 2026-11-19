"""Loading calculations run with point-group symmetry (ADR-0005).

The checks are known identities against the data: our AO overlap reproduces
Molcas's, desymmetrised MOs are orthonormal, and every dominant configuration
has its state's irrep (so the CI strings are lined up with the active orbitals).
"""

from pathlib import Path
import shutil

import h5py
import numpy as np
import pytest

from xpsoverlaps.model import PointGroup
from xpsoverlaps.overlap import get_overlap_provider
from xpsoverlaps.readers import get_calculation

ROOT = Path(__file__).parents[1]
THIOPHENOL = ROOT / "data" / "molcas" / "thiophenol" / "s2p" / "dz"
CS2 = ROOT / "data" / "molcas" / "cs2" / "s2p" / "tz" / "singlet" / "2B2"
BRTHPH = ROOT / "data" / "molcas" / "brthph" / "s2p" / "mb"


def _rasscf(directory, name):
    return get_calculation("molcas", "rasscf", directory / f"{name}.rasscf.h5", directory / f"{name}.log")


@pytest.fixture(scope="module")
def thiophenol_a():
    return _rasscf(THIOPHENOL, "valence-dz-a")


@pytest.fixture(scope="module")
def cs2():
    return _rasscf(CS2, "valence-s2p-tz-2b2")


def test_loaded_calculations_know_their_point_group(thiophenol_a, cs2):
    assert thiophenol_a.point_group.irreps == ("a'", 'a"')
    assert cs2.point_group.irreps == ("a1", "b2", "a2", "b1")
    assert cs2.point_group.totally_symmetric == "a1"
    assert cs2.point_group.product("b2", "b1") == "a2"
    assert cs2.point_group.product("a2", "b1") == "b2"
    assert cs2.point_group.product("b2", "b2") == "a1"
    assert cs2.point_group.product() == "a1"
    assert thiophenol_a.point_group.product('a"', 'a"') == "a'"


def test_only_abelian_point_groups_are_accepted():
    with pytest.raises(ValueError, match="1, 2, 4 or 8 irreps"):
        PointGroup(["a", "b", "c"], [[0, 1, 2], [1, 2, 0], [2, 0, 1]])
    with pytest.raises(ValueError, match="squared"):
        PointGroup(["a", "b", "c", "d"], [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]])  # C4
    with pytest.raises(ValueError, match="'e' is not an irrep"):
        PointGroup(["a"], [[0]]).product("e")


def test_mos_are_named_and_labelled_by_their_molcas_irrep(thiophenol_a, cs2):
    orbitals = thiophenol_a.orbitals  # NBAS = 97 a', 35 a"
    assert orbitals.orbital_ids[:2] == ["mo:1_i1", "mo:2_i1"]
    assert orbitals.orbital_ids[96:98] == ["mo:97_i1", "mo:1_i2"]
    assert orbitals.orbital_ids[-1] == "mo:35_i2"
    assert orbitals.irreps == ["a'"] * 97 + ['a"'] * 35
    assert [cs2.orbitals.irreps.count(irrep) for irrep in ("a1", "b2", "a2", "b1")] == [36, 30, 14, 18]
    assert cs2.orbitals.orbital_ids[66] == "mo:1_i3"  # first a2 orbital


def test_calculations_without_symmetry_keep_plain_orbital_ids():
    brthph = get_calculation("molcas", "rasscf", BRTHPH / "valence.rasscf.h5", BRTHPH / "valence-tz.log")
    assert brthph.point_group.irreps == ("a",)
    assert brthph.orbitals.orbital_ids[:3] == ["mo:1", "mo:2", "mo:3"]
    assert set(brthph.orbitals.irreps) == {"a"}


def test_active_orbitals_follow_the_ci_string_order(thiophenol_a):
    # Log header "11 2 1111 2222222": RAS1 (2 a', 1 a"), then RAS2 (4 a', 7 a").
    active = thiophenol_a.active_space
    assert active.orbital_ids == (
        ["mo:21_i1", "mo:22_i1", "mo:1_i2"]
        + [f"mo:{n}_i1" for n in range(23, 27)]
        + [f"mo:{n}_i2" for n in range(2, 9)]
    )
    assert "".join(active.partitions) == "111" + "2" * 11


def test_occupation_strings_printed_with_spaces_are_read(thiophenol_a):
    # "22 2 2200 2222000" in both the RASSCF and the CASPT2 log.
    caspt2 = get_calculation("molcas", "caspt2", THIOPHENOL / "valence-a-caspt2.caspt2.h5",
                             THIOPHENOL / "valence-a-caspt2.log", mcscf_file=THIOPHENOL / "valence-dz-a.rasscf.h5")
    for calculation in (thiophenol_a, caspt2):
        assert calculation.states[0].dominant_configuration.occupations == list("22222002222000")


def test_states_get_the_irrep_of_their_calculation(thiophenol_a, cs2):
    assert {state.symmetry for state in thiophenol_a.states} == {"a'"}
    assert {state.symmetry for state in _rasscf(THIOPHENOL, "valence-dz-ap").states} == {'a"'}
    assert {state.symmetry for state in cs2.states} == {"b2"}


@pytest.mark.parametrize("directory, name", [
    (THIOPHENOL, "valence-dz-a"), (THIOPHENOL, "valence-dz-ap"), (THIOPHENOL, "core-dz-a"),
    (THIOPHENOL, "core-dz-ap"), (CS2, "valence-s2p-tz-2b2"), (CS2, "core-s2p-a2-tz-2b2"),
])
def test_every_dominant_configuration_has_its_state_irrep(directory, name):
    calculation = _rasscf(directory, name)
    assert calculation.states
    for state in calculation.states:
        assert calculation.configuration_irrep(state.dominant_configuration) == state.symmetry


def _edited_copy(tmp_path, name, edit):
    """A copy of a thiophenol RASSCF HDF5 file with ``edit(h5py.File)`` applied."""
    path = tmp_path / f"{name}.rasscf.h5"
    shutil.copy(THIOPHENOL / f"{name}.rasscf.h5", path)
    with h5py.File(path, "r+") as h5_file:
        edit(h5_file)
    return path


def test_ci_strings_not_lined_up_with_the_active_orbitals_are_rejected(tmp_path):
    def swap_two_ras2_levels_of_different_irreps(h5_file):
        l2act = h5_file.attrs["L2ACT"].copy()
        # Last a' RAS2 level <-> first a" RAS2 level, the open shell u of state 2 (22 2 2200 u2220d0).
        l2act[[6, 7]] = l2act[[7, 6]]
        h5_file.attrs["L2ACT"] = l2act

    path = _edited_copy(tmp_path, "valence-dz-a", swap_two_ras2_levels_of_different_irreps)
    with pytest.raises(ValueError, match="not lined up with the active orbitals"):
        get_calculation("molcas", "rasscf", path, THIOPHENOL / "valence-dz-a.log")


def test_a_log_of_another_irrep_is_rejected(tmp_path):
    path = _edited_copy(tmp_path, "valence-dz-a", lambda h5_file: h5_file.attrs.modify("LSYM", 2))
    with pytest.raises(ValueError, match="state symmetry 1 but the HDF5 file has LSYM = 2"):
        get_calculation("molcas", "rasscf", path, THIOPHENOL / "valence-dz-a.log")


def test_a_basis_that_does_not_reproduce_the_molcas_ao_overlap_is_rejected(tmp_path):
    def swap_two_desymmetrization_columns(h5_file):
        n = 132
        D = h5_file["DESYM_MATRIX"][()].reshape(n, n, order="F")
        D[:, [13, 14]] = D[:, [14, 13]]
        h5_file["DESYM_MATRIX"][...] = D.reshape(-1, order="F")

    path = _edited_copy(tmp_path, "valence-dz-a", swap_two_desymmetrization_columns)
    with pytest.raises(ValueError, match="differs from Molcas's AO_OVERLAP_MATRIX"):
        get_calculation("molcas", "rasscf", path, THIOPHENOL / "valence-dz-a.log")


@pytest.mark.parametrize("name", ["thiophenol_a", "cs2"])
def test_desymmetrised_mos_are_orthonormal_over_the_ao_basis(name, request):
    calculation = request.getfixturevalue(name)
    C = calculation.orbitals.coefficients
    S = get_overlap_provider(calculation.basis, calculation.basis).cross_overlap(calculation.basis, calculation.basis)
    np.testing.assert_allclose(C.T @ S @ C, np.eye(C.shape[1]), atol=1e-10)
