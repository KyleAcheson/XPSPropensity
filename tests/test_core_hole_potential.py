import dataclasses
import logging
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from xpsoverlaps.integrals import CoreHoleIntegrals, CoulombMatrix, StoredIntegrals
from xpsoverlaps.methods import propensity_chp
from xpsoverlaps.overlap_sources import CoreHoleOverlaps
from xpsoverlaps.propensity_rules import CHANNELS
from xpsoverlaps.readers import IntegralFileError, get_active_integrals, get_calculation
from xpsoverlaps.shakedown_transitions import ShakeDownOverlaps


DATA = Path(__file__).parents[1] / "data" / "molcas" / "brthph" / "s2p" / "mb"
FCIDUMP = DATA / "valence.FciDmp.h5"
CORES = ["mo:31", "mo:32", "mo:33"]


@pytest.fixture(scope="module")
def rasscf():
    return get_calculation("molcas", "rasscf", DATA / "valence.rasscf.h5", DATA / "valence-tz.log")


@pytest.fixture(scope="module")
def caspt2():
    return get_calculation("molcas", "caspt2", DATA / "valence_caspt2.caspt2.h5", DATA / "valence_caspt2.log",
                           mcscf_file=DATA / "valence.rasscf.h5")


@pytest.fixture(scope="module")
def integrals(rasscf):
    return get_active_integrals("molcas", FCIDUMP, rasscf)


def _eri(integrals, *orbitals):
    return integrals.two_electron[tuple(integrals.orbital_ids.index(o) for o in orbitals)]


def _edited_copy(tmp_path, edit):
    path = tmp_path / "edited.FciDmp.h5"
    shutil.copy(FCIDUMP, path)
    with h5py.File(path, "r+") as file:
        edit(file)
    return path


# --- Molcas FCIDUMP reader ---------------------------------------------------


def test_fcidump_integrals_are_indexed_by_active_orbital(integrals, rasscf):
    assert integrals.orbital_ids == rasscf.active_space.orbital_ids
    # The first stored entry, (1 1|1 1) = (mo:31 mo:31|mo:31 mo:31).
    assert _eri(integrals, "mo:31", "mo:31", "mo:31", "mo:31") == pytest.approx(2.2737428264631827, abs=1e-15)
    # (34 38|cc): chemists' notation, matches exact integrals from PySCF to the RICD error.
    assert [_eri(integrals, "mo:34", "mo:38", c, c) for c in CORES] == pytest.approx(
        [0.021136, 0.022372, 0.022776], abs=1e-6)


def test_fcidump_fills_all_eight_index_orders(integrals):
    eri = integrals.two_electron
    for order in [(1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1), (3, 2, 1, 0)]:
        assert np.array_equal(eri, eri.transpose(order))


def test_fcidump_matches_a_caspt2_calculation_through_its_rasscf_energies(caspt2, integrals):
    assert np.array_equal(get_active_integrals("molcas", FCIDUMP, caspt2).two_electron, integrals.two_electron)


def test_fcidump_follows_orbital_index_when_labels_are_permuted(tmp_path, rasscf, integrals):
    labels = np.array([3, 1, 2, 12, 11, 10, 9, 8, 7, 6, 5, 4])  # label of active orbital 1, 2, ...

    def permute(file):
        file["ORBITAL_INDEX"][...] = labels
        file["TWO_EL_INT_INDEX"][...] = labels[file["TWO_EL_INT_INDEX"][()] - 1]

    permuted = get_active_integrals("molcas", _edited_copy(tmp_path, permute), rasscf)

    assert np.array_equal(permuted.two_electron, integrals.two_electron)


@pytest.mark.parametrize(
    "edit,message",
    [
        (lambda f: f.attrs.modify("NORB", 11), "it has 11 orbitals; the active space has 12"),
        (lambda f: f.attrs.modify("ORBSYM", np.array([1] * 6 + [2] * 6)), "written with symmetry"),
        (lambda f: f.attrs.modify("NELEC", 16), "it has 16 active electrons; the calculation has 18"),
        (lambda f: f["ORBITAL_INDEX"].__setitem__(0, 2), "ORBITAL_INDEX is not a permutation"),
        (lambda f: f["ORBITAL_ENERGIES"].__setitem__(3, -0.3), "orbital energy of mo:34 is -0.3"),
    ],
)
def test_fcidump_of_another_calculation_is_rejected(tmp_path, rasscf, edit, message):
    with pytest.raises(IntegralFileError, match=message):
        get_active_integrals("molcas", _edited_copy(tmp_path, edit), rasscf)


def test_fcidump_needs_orbital_energies_to_be_checked(rasscf):
    calculation = dataclasses.replace(rasscf, orbitals=dataclasses.replace(rasscf.orbitals, energies=None))

    with pytest.raises(IntegralFileError, match="no orbital energies"):
        get_active_integrals("molcas", FCIDUMP, calculation)


# --- Integrals (pq|cc) --------------------------------------------------------


def test_stored_integrals_give_one_coulomb_matrix_per_core(integrals):
    valence = ["mo:34", "mo:38", "mo:40"]

    coulomb = StoredIntegrals(integrals).coulomb(CORES, valence)

    assert list(coulomb) == CORES
    assert coulomb["mo:32"]("mo:38", "mo:34") == _eri(integrals, "mo:38", "mo:34", "mo:32", "mo:32")
    assert np.array_equal(coulomb["mo:31"].values, coulomb["mo:31"].values.T)
    with pytest.raises(ValueError, match="mo:99 not in the integral file"):
        StoredIntegrals(integrals).coulomb(CORES, ["mo:34", "mo:99"])


def test_integral_sources_are_checked_for_shape_and_symmetry():
    class Asymmetric(CoreHoleIntegrals):
        source = "test"

        def _compute(self, core_ids, orbital_ids):
            return np.arange(4.0).reshape(1, 2, 2)

    with pytest.raises(ValueError, match="not symmetric"):
        Asymmetric().coulomb(["c"], ["p", "q"])
    with pytest.raises(ValueError, match=r"shape \(1, 2, 2\); expected \(2, 2, 2\)"):
        Asymmetric().coulomb(["c", "d"], ["p", "q"])


# --- Core-hole overlaps ----------------------------------------------------------


def _chp(values, energies, core="c"):
    ids = list(energies)
    return CoreHoleOverlaps(CoulombMatrix(core, ids, np.asarray(values, dtype=float)), energies)


def test_core_hole_overlap_is_coulomb_over_energy_gap_and_antisymmetric():
    S = _chp([[0.5, 0.02], [0.02, 0.4]], {"a": -0.3, "b": 0.1})

    assert S.overlap("a", "b") == pytest.approx(0.02 / (0.1 - -0.3))
    assert S.overlap("b", "a") == pytest.approx(-S.overlap("a", "b"))
    assert S.overlap("a", "a") == 1.0
    assert S.describe("a", "b") == "(a b|c c) / (e_b - e_a) = +0.02000 / +0.40000"


def test_core_hole_overlaps_flag_large_values_and_reject_coupled_degenerate_orbitals():
    S = _chp([[0, 0.3, 0.01], [0.3, 0, 0], [0.01, 0, 0]], {"a": -0.5, "b": -0.4, "d": -0.4})

    assert [(a, b) for a, b, _ in S.large()] == [("a", "b")]  # 0.3 / 0.1
    assert S.overlap("b", "d") == 0.0  # degenerate but not coupled
    with pytest.raises(ValueError, match="b and d are degenerate and coupled"):
        _chp([[0, 0.1], [0.1, 0]], {"b": -0.4, "d": -0.4}).overlap("b", "d")


def test_core_hole_intensities_do_not_depend_on_orbital_signs():
    rng = np.random.default_rng(1)
    J = rng.normal(scale=0.05, size=(3, 3))
    J = J + J.T
    energies = {"i": -0.4, "j": -0.3, "k": 0.1}
    signs = np.diag([1.0, -1.0, -1.0])  # T: change the sign of j and k
    roles = {"i": "i", "j": "j", "k": "k"}

    for channel in CHANNELS[("5e ES", 0)] + CHANNELS[("5e ES", 1)] + CHANNELS[("5e GS", 0)]:
        plain = channel.evaluate(ShakeDownOverlaps(roles, _chp(J, energies)))
        flipped = channel.evaluate(ShakeDownOverlaps(roles, _chp(signs @ J @ signs, energies)))
        assert flipped.intensity == pytest.approx(plain.intensity)


# --- The chp method -------------------------------------------------------------


@pytest.fixture(scope="module")
def output(caspt2, integrals):
    lines, warnings = [], []

    class Collect(logging.Handler):
        def emit(self, record):
            (warnings if record.levelno == logging.WARNING else lines).append(record.getMessage())

    logger = logging.getLogger("xpsoverlaps")
    handler, level = Collect(), logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        propensity_chp(caspt2, StoredIntegrals(integrals), max_state=4)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)
    return lines, warnings


def test_chp_reports_overlaps_and_rules_per_core_orbital(output):
    lines, _ = output

    sections = [line for line in lines if line in ("Orbital order", "Core-hole overlaps",
                                                   "Propensity rules (core-hole potential)")]
    assert sections == ["Orbital order", "Core-hole overlaps", "Propensity rules (core-hole potential)"]
    assert [line for line in lines if line.startswith("Core hole in")] == [f"Core hole in {c}" for c in CORES] * 2
    assert ("  S_ij = (mo:38 mo:34|mo:31 mo:31) / (e_mo:34 - e_mo:38) = +0.02114 / +0.05785 = 0.3654"
            in lines)


def test_chp_summary_gives_each_core_and_the_mean(output):
    lines, _ = output

    summary = lines[lines.index("Summary") + 1:]
    assert summary[0].split() == ["a", "->", "b", "Case", "Channel", *CORES, "Mean"]
    row = next(line for line in summary if line.split()[:5] == ["3", "->", "2", "5e", "ES"])
    intensities = [float(value) for value in row.split()[-4:]]
    assert intensities == pytest.approx([0.165087, 0.185354, 0.195285, 0.181909], abs=1e-6)
    assert intensities[3] == pytest.approx(sum(intensities[:3]) / 3, abs=1e-6)


def test_chp_warns_once_about_large_overlaps(output):
    _, warnings = output

    assert len(warnings) == 1 and warnings[0].startswith("9 core-hole overlaps |S_ab| above 0.5")
