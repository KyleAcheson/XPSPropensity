import dataclasses
from pathlib import Path

import numpy as np
import pytest

from xpsoverlaps.frontier import FrontierOrbitals
from xpsoverlaps.mapping import map_active_orbitals
from xpsoverlaps.readers import get_calculation
from xpsoverlaps.rules import RULES, RoleOverlaps, compute_propensity, propensity_report
from xpsoverlaps.shakedown import find_shake_downs


DATA = Path(__file__).parents[1] / "data" / "molcas" / "brthph" / "s2p" / "mb"
FRONTIER = FrontierOrbitals("mo:34", "mo:40")


@pytest.fixture(scope="module")
def setup():
    initial = get_calculation(
        "molcas", "caspt2", DATA / "valence_caspt2.caspt2.h5", DATA / "valence_caspt2.log"
    )
    final = get_calculation("molcas", "caspt2", DATA / "core_caspt2.caspt2.h5", DATA / "core_caspt2.log")
    mapping = map_active_orbitals(initial, final)
    shake_downs = {
        (sd.initial_state.split(":")[1], sd.lower_state.split(":")[1]): sd
        for sd in find_shake_downs(initial, FRONTIER, max_state=4)
    }
    return initial, final, mapping, shake_downs


def _overlap_function(values):
    """S as a function of role pairs, with diagonals 1, from a dict of off-diagonal values."""
    return lambda pair: 1.0 if pair[0] == pair[1] else values[pair]


# --- The rules reproduce the identities of the propensity notes -------------


def test_three_electron_rule_matches_notes():
    S = _overlap_function({"jk": 0.30, "ik": 0.20, "ji": 0.10})

    amplitudes = RULES["3e ES"].amplitudes(S)

    assert amplitudes["Pi"] == pytest.approx(0.30)
    assert amplitudes["Omega"] == pytest.approx(0.20 * 0.10)
    v2 = np.array([[1.0, 0.20], [0.10, 0.30]])  # rows (i, j) final, columns (i, k) initial
    assert amplitudes["Pi"] - amplitudes["Omega"] == pytest.approx(np.linalg.det(v2))


def test_five_electron_rule_matches_notes():
    values = {"ij": 0.31, "ji": 0.17, "ik": 0.23, "ki": 0.11, "jk": 0.29, "kj": 0.13}
    S = _overlap_function(values)

    amplitudes = RULES["5e ES"].amplitudes(S)

    v3 = np.array([[S(a + b) for b in "ijk"] for a in "ijk"])
    m3, m4 = amplitudes["M3"], amplitudes["M4"]
    assert amplitudes["Pi"] - amplitudes["Omega"] == pytest.approx(values["ij"] * np.linalg.det(v3))
    assert amplitudes["Pi"] + amplitudes["Omega"] == pytest.approx(
        values["ij"] * np.linalg.det(v3) + 2 * m3 * m4
    )


# --- Overlaps: first index final, second initial ----------------------------


def test_role_overlap_first_index_is_the_final_orbital(setup):
    initial, final, mapping, _ = setup
    S = RoleOverlaps({"i": "mo:38", "j": "mo:34", "k": "mo:40"}, mapping)

    # S_ij = |<phi_final(i)|psi_j>|, final(i) = final mo:34 (mapped from initial mo:38)
    assert S("ij") == pytest.approx(abs(mapping.overlap("mo:34", "mo:34")))
    assert S("ji") == pytest.approx(abs(mapping.overlap("mo:38", "mo:39")))
    assert S.describe("ij") == "<final mo:34 (i) | initial mo:34 (j)>"
    assert S("ii") == 1.0
    assert S.actual("ii") == pytest.approx(0.7122, abs=1e-4)


# --- Dispatch and channels on the bundled data -------------------------------


def test_singlet_shake_downs_use_the_right_rule_and_channel(setup):
    initial, _, mapping, shake_downs = setup

    gs = compute_propensity(shake_downs[("3", "1")], initial, mapping)
    es5 = compute_propensity(shake_downs[("3", "2")], initial, mapping)
    es3 = compute_propensity(shake_downs[("4", "2")], initial, mapping)

    assert [c.name for c in gs.channels] == ["closed-shell doublet"]
    assert gs.channels[0].intensity == pytest.approx(2 * gs.overlaps["ik"] ** 2)
    assert set(es5.amplitudes) == {"M1", "M2", "M3", "M4", "Pi", "Omega"}
    assert [c.name for c in es5.channels] == ["D1"]
    assert es5.channels[0].intensity == pytest.approx(0.015099, abs=1e-6)
    assert set(es3.overlaps) == {"jk", "ik", "ji"}
    # mo:41 is in-plane (sigma), mo:34/40 out-of-plane (pi): zero by symmetry.
    assert es3.channels[0].intensity == pytest.approx(0.0, abs=1e-9)


def test_triplet_initial_state_gives_d2_and_quartet_and_dark_gs(setup):
    initial, _, mapping, shake_downs = setup
    states = [dataclasses.replace(s, spin=1.0) if s.state_id == "caspt2:3" else s for s in initial.states]
    triplet = dataclasses.replace(initial, states=states)

    es = compute_propensity(shake_downs[("3", "2")], triplet, mapping)
    gs = compute_propensity(shake_downs[("3", "1")], triplet, mapping)

    minus = (es.amplitudes["Pi"] - es.amplitudes["Omega"]) ** 2
    assert [(c.name, c.intensity) for c in es.channels] == [
        ("D2", pytest.approx(minus / 3)),
        ("Q", pytest.approx(2 * minus / 3)),
    ]
    assert gs.channels == [] and gs.note.startswith("dark")


def test_report_names_orbitals_flags_diagonals_and_summarises(setup):
    initial, final, mapping, shake_downs = setup
    results = [compute_propensity(sd, initial, mapping) for sd in shake_downs.values() if sd.computable]

    lines = propensity_report(results, mapping, final)

    assert "Final calculation spin: S = 0.5." in lines
    assert "  S_ik = |<final mo:34 (i) | initial mo:40 (k)>| = 0.0572" in lines
    assert any("S_ii = |<final mo:34 (i) | initial mo:38 (i)>| = 0.7122 (set to 1)   below 0.9" in l for l in lines)
    summary = lines[lines.index("Summary") + 2:]
    assert [line.split()[:4] for line in summary] == [
        ["2", "->", "1", "3e"], ["3", "->", "1", "3e"], ["3", "->", "2", "5e"],
        ["4", "->", "1", "3e"], ["4", "->", "2", "3e"],
    ]
