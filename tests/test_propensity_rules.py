import dataclasses
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from xpsoverlaps.frontier import orbital_order
from xpsoverlaps.mapping import MappedFinal, map_active_orbitals
from xpsoverlaps.readers import get_calculation
from xpsoverlaps.propensity_rules import (
    CHANNELS,
    Channel,
    RoleOverlaps,
    compute_propensity,
    evaluate_final,
    propensity_report,
)
from xpsoverlaps.shakedown_transitions import find_shake_downs, reference_determinant


DATA = Path(__file__).parents[1] / "data" / "molcas" / "brthph" / "s2p" / "mb"


@pytest.fixture(scope="module")
def setup():
    initial = get_calculation(
        "molcas", "caspt2", DATA / "valence_caspt2.caspt2.h5", DATA / "valence_caspt2.log",
        mcscf_file=DATA / "valence.rasscf.h5",
    )
    final = get_calculation("molcas", "caspt2", DATA / "core_caspt2.caspt2.h5", DATA / "core_caspt2.log")
    mapping = map_active_orbitals(initial, final)
    shake_downs = {
        (sd.initial_state.split(":")[1], sd.lower_state.split(":")[1]): sd
        for sd in find_shake_downs(initial, orbital_order(initial, reference_determinant(initial)), max_state=4)
    }
    return initial, final, mapping, shake_downs


def _overlap_function(values):
    """S as a function of role pairs, with diagonals 1, from a dict of off-diagonal values."""
    return lambda pair: 1.0 if pair[0] == pair[1] else values[pair]


# --- The rules reproduce the identities of the propensity notes -------------


def test_three_electron_rule_matches_notes():
    S = _overlap_function({"jk": 0.30, "ik": 0.20, "ji": 0.10})

    amplitudes = CHANNELS[("3e ES", 0)][0].evaluate(S).values

    assert amplitudes["Pi"] == pytest.approx(0.30)
    assert amplitudes["Omega"] == pytest.approx(0.20 * 0.10)
    v2 = np.array([[1.0, 0.20], [0.10, 0.30]])  # rows (i, j) final, columns (i, k) initial
    assert amplitudes["Pi"] - amplitudes["Omega"] == pytest.approx(np.linalg.det(v2))


def test_five_electron_rule_matches_notes():
    values = {"ij": 0.31, "ji": 0.17, "ik": 0.23, "ki": 0.11, "jk": 0.29, "kj": 0.13}
    S = _overlap_function(values)

    amplitudes = CHANNELS[("5e ES", 0)][0].evaluate(S).values

    v3 = np.array([[S(a + b) for b in "ijk"] for a in "ijk"])
    m3, m4 = amplitudes["M3"], amplitudes["M4"]
    assert amplitudes["Pi"] - amplitudes["Omega"] == pytest.approx(values["ij"] * np.linalg.det(v3))
    assert amplitudes["Pi"] + amplitudes["Omega"] == pytest.approx(
        values["ij"] * np.linalg.det(v3) + 2 * m3 * m4
    )


def test_m_minors_keep_the_diagonals_explicitly():
    values = {"ii": 0.9, "jj": 0.8, "kk": 0.7, "ij": 0.31, "ji": 0.17, "ik": 0.23, "ki": 0.11, "jk": 0.29, "kj": 0.13}
    v3 = np.array([[values[a + b] for b in "ijk"] for a in "ijk"])

    amplitudes = CHANNELS[("5e ES", 1)][1].evaluate(lambda pair: values[pair]).values  # Q

    minor = lambda rows, cols: np.linalg.det(v3[np.ix_(["ijk".index(r) for r in rows], ["ijk".index(c) for c in cols])])
    assert amplitudes["M1"] == pytest.approx(minor("ij", "ij"))
    assert amplitudes["M2"] == pytest.approx(minor("ik", "jk"))
    assert amplitudes["M3"] == pytest.approx(minor("ij", "jk"))
    assert amplitudes["M4"] == pytest.approx(minor("ik", "ij"))
    assert amplitudes["Pi"] - amplitudes["Omega"] == pytest.approx(values["ij"] * np.linalg.det(v3))


def test_ground_state_rules_are_closed_shell_products():
    values = {"ij": 0.31, "ji": 0.17, "ik": 0.23, "jk": 0.29}
    S = _overlap_function(values)
    m1, m3 = 1 - 0.31 * 0.17, 0.31 * 0.29 - 0.23

    (gs5,), (gs3,) = CHANNELS[("5e GS", 0)], CHANNELS[("3e GS", 0)]
    result = gs5.evaluate(S)

    assert result.values == {"M1": pytest.approx(m1), "M3": pytest.approx(m3), "A": pytest.approx(m1 * m3)}
    assert result.intensity == pytest.approx(2 * (m1 * m3) ** 2)
    assert gs3.evaluate(S).intensity == pytest.approx(2 * 0.23 ** 2)
    assert gs5.overlaps == ["ii", "jj", "ij", "ji", "jk", "ik"]
    assert gs3.overlaps == ["ii", "ik"]
    assert CHANNELS[("5e GS", 1)] == () and CHANNELS[("3e GS", 1)] == ()


def test_channel_spin_weights_follow_the_notes():
    weights = {(case, spin): [(c.name, c.final_spin, c.weight) for c in channels]
               for (case, spin), channels in CHANNELS.items() if case.endswith("ES")}

    for case in ("3e ES", "5e ES"):
        assert weights[(case, 0)] == [("D1", 0.5, 1)]
        assert weights[(case, 1)] == [("D2", 0.5, Fraction(1, 3)), ("Q", 1.5, Fraction(2, 3))]


@pytest.mark.parametrize(
    "equations,amplitude,message",
    [
        (("Pi = S_ii S_jk",), "Pi + Omega", "amplitude = Pi \\+ Omega uses Omega, not defined"),
        (("Pi = M1 M2", "M1 = S_ii S_jj"), "Pi", "Pi = M1 M2 uses M1, M2, not defined"),
        (("Pi = S_il S_jk",), "Pi", "'S_il' in 'S_il S_jk' is not an overlap S_ab of roles i, j, k"),
        (("S_ii S_jk",), "Pi", "'S_ii S_jk' is not an equation 'name = formula'"),
    ],
)
def test_channel_checks_its_formulas_when_built(equations, amplitude, message):
    with pytest.raises(ValueError, match=f"Channel X: .*{message}"):
        Channel("X", 0.5, Fraction(1), equations, amplitude)


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

    gs3 = compute_propensity(shake_downs[("2", "1")], initial, mapping)
    gs5 = compute_propensity(shake_downs[("3", "1")], initial, mapping)
    es5 = compute_propensity(shake_downs[("3", "2")], initial, mapping)
    es3 = compute_propensity(shake_downs[("4", "2")], initial, mapping)

    assert [c.name for c in gs3.channels] == ["closed-shell doublet"]
    assert gs3.channels[0].intensity == pytest.approx(2 * gs3.overlaps["ik"] ** 2)
    # HOMO-1 pump: j = HOMO, 2 |M1 M3|^2 rather than 2 |S_ik|^2
    assert gs5.roles == {"i": "mo:38", "j": "mo:34", "k": "mo:40"}
    assert gs5.channels[0].intensity == pytest.approx(2 * (gs5.values["M1"] * gs5.values["M3"]) ** 2)
    assert gs5.channels[0].intensity == pytest.approx(0.006179, abs=1e-6)
    assert set(es5.values) == {"M1", "M2", "M3", "M4", "Pi", "Omega"}
    assert [c.name for c in es5.channels] == ["D1"]
    assert es5.channels[0].intensity == pytest.approx(0.015099, abs=1e-6)
    assert set(es3.overlaps) == {"ii", "jk", "ik", "ji"}
    # mo:41 is in-plane (sigma), mo:34/40 out-of-plane (pi): zero by symmetry.
    assert es3.channels[0].intensity == pytest.approx(0.0, abs=1e-9)


def test_triplet_initial_state_gives_d2_and_quartet_and_dark_gs(setup):
    initial, _, mapping, shake_downs = setup
    states = [dataclasses.replace(s, spin=1.0) if s.state_id == "caspt2:3" else s for s in initial.states]
    triplet = dataclasses.replace(initial, states=states)

    es = compute_propensity(shake_downs[("3", "2")], triplet, mapping)
    gs = compute_propensity(shake_downs[("3", "1")], triplet, mapping)  # 5e GS

    minus = (es.values["Pi"] - es.values["Omega"]) ** 2
    assert [(c.name, c.intensity) for c in es.channels] == [
        ("D2", pytest.approx(minus / 3)),
        ("Q", pytest.approx(2 * minus / 3)),
    ]
    assert gs.channels == [] and gs.note.startswith("dark")


def _report(initial, finals, shake_downs):
    return propensity_report(shake_downs, initial, [(f, evaluate_final(shake_downs, initial, f)) for f in finals])


def test_channels_are_evaluated_only_with_the_final_calculation_of_their_spin(setup):
    initial, _, mapping, shake_downs = setup
    states = [dataclasses.replace(s, spin=1.0) if s.state_id == "caspt2:3" else s for s in initial.states]
    triplet = dataclasses.replace(initial, states=states)

    doublet = compute_propensity(shake_downs[("3", "2")], triplet, mapping, final_spin=0.5)
    quartet = compute_propensity(shake_downs[("3", "2")], triplet, mapping, final_spin=1.5)

    assert [c.name for c in doublet.channels] == ["D2"] and [c.name for c in quartet.channels] == ["Q"]
    assert quartet.channels[0].intensity == pytest.approx(2 * doublet.channels[0].intensity)  # same mapping here


def test_summary_notes_a_channel_without_a_final_calculation_of_its_spin(setup):
    initial, final, mapping, shake_downs = setup
    states = [dataclasses.replace(s, spin=1.0) if s.state_id == "caspt2:3" else s for s in initial.states]
    triplet = dataclasses.replace(initial, states=states)

    lines = _report(triplet, [MappedFinal("Final", final, mapping, 0.5)], [shake_downs[("3", "2")]])

    summary = lines[lines.index("Summary") + 2:]
    assert summary[0].split()[:6] == ["3", "->", "2", "5e", "ES", "Final"] and summary[0].split()[6] == "D2"
    assert summary[1].endswith("Q                               -   no S = 1.5 final calculation")


def test_report_writes_each_rule_out_in_full(setup):
    initial, final, mapping, shake_downs = setup
    selected = [shake_downs[pair] for pair in (("2", "1"), ("3", "1"), ("3", "2"))]

    lines = _report(initial, [MappedFinal("Final", final, mapping, 0.5)], selected)

    assert "  A = S_ii S_ik = (1)(0.0059) = +0.00592" in lines
    assert "  closed-shell doublet: 2 |A|^2 = 2 |S_ii S_ik|^2 = 2 |(+0.00592)|^2 = 0.000070" in lines
    assert "  M1 = S_ii S_jj - S_ij S_ji = (1)(1) - (0.1263)(0.1267) = +0.98399" in lines
    assert "  M3 = S_ij S_jk - S_ik S_jj = (0.1263)(0.0059) - (0.0572)(1) = -0.05649" in lines
    assert "  A = M1 M3 = (+0.98399)(-0.05649) = -0.05558" in lines
    assert "  closed-shell doublet: 2 |A|^2 = 2 |M1 M3|^2 = 2 |(-0.05558)|^2 = 0.006179" in lines
    assert "  M2 = S_ij S_kk - S_ik S_kj = (0.1263)(1) - (0.0572)(0.0223) = +0.12505" in lines
    assert "  Pi = M1 M2 = (+0.98399)(+0.12505) = +0.12305" in lines
    assert ("  D1: |Pi + Omega|^2 = |M1 M2 + M3 M4|^2 = |(+0.12305) + (-0.00017)|^2 = 0.015099"
            in lines)


def test_report_names_orbitals_flags_diagonals_and_summarises(setup):
    initial, final, mapping, shake_downs = setup

    lines = _report(initial, [MappedFinal("Final", final, mapping, 0.5)], list(shake_downs.values()))

    assert "Final: S = 0.5 (core_caspt2.caspt2.h5)" in lines
    assert "  S_ik = |<final mo:34 (i) | initial mo:40 (k)>| = 0.0572" in lines
    assert any("S_ii = |<final mo:34 (i) | initial mo:38 (i)>| = 0.7122 (set to 1)   below 0.9" in l for l in lines)
    summary = lines[lines.index("Summary") + 2:]
    assert [line.split()[:4] for line in summary] == [
        ["2", "->", "1", "3e"], ["3", "->", "1", "5e"], ["3", "->", "2", "5e"],
        ["4", "->", "1", "3e"], ["4", "->", "2", "3e"],
    ]
