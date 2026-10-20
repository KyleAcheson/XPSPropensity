import dataclasses
from pathlib import Path

import pytest

from xpsoverlaps.frontier import FrontierOrbitals, OrbitalOrderError, orbital_order
from xpsoverlaps.readers import get_calculation
from xpsoverlaps import shakedown
from xpsoverlaps.shakedown import (
    Excitation,
    ShakeDownError,
    check_max_state,
    excitation,
    find_shake_downs,
    reference_determinant,
    shake_down_report,
)


DATA = Path(__file__).parents[1] / "data" / "molcas" / "brthph" / "s2p" / "mb"
# (a, b): (case, i, j, k, d, status), worked out by hand from the dominant configurations
# and the population order HOMO mo:34, HOMO-1 38, HOMO-2 37, HOMO-3 36; LUMO 40, LUMO+1 41.
EXPECTED = {
    ("2", "1"): ("3e GS", "mo:34", None, "mo:40", None, "ok"),
    ("3", "1"): ("5e GS", "mo:38", "mo:34", "mo:40", None, "ok"),
    ("3", "2"): ("5e ES", "mo:38", "mo:34", "mo:40", None, "ok"),
    ("4", "1"): ("3e GS", "mo:34", None, "mo:41", None, "ok"),
    ("4", "2"): ("3e ES", "mo:34", "mo:40", "mo:41", None, "ok"),
    ("5", "1"): ("5e GS", "mo:38", "mo:34", "mo:41", None, "ok"),
    ("5", "3"): ("3e ES", "mo:38", "mo:40", "mo:41", "mo:34", "ok"),
    ("5", "4"): ("5e ES", "mo:38", "mo:34", "mo:41", None, "ok"),
    ("6", "1"): ("5e GS", "mo:37", "mo:34", "mo:41", None, "ok"),
    ("6", "4"): ("5e ES", "mo:37", "mo:34", "mo:41", None, "ok"),
    ("6", "5"): ("5e ES", "mo:37", "mo:38", "mo:41", "mo:34", "ok"),
    ("7", "2"): (None, None, None, None, None, "double excitation"),
    ("8", "1"): ("5e GS", "mo:37", "mo:34", "mo:40", None, "ok"),
    ("8", "2"): ("5e ES", "mo:37", "mo:34", "mo:40", None, "ok"),
    ("8", "3"): ("5e ES", "mo:37", "mo:38", "mo:40", "mo:34", "ok"),
    ("8", "6"): ("3e ES", "mo:37", "mo:41", "mo:40", "mo:34", "electron would move up: j (LUMO+1) lies above k (LUMO)"),
    ("9", "1"): ("5e GS", "mo:36", "mo:34", "mo:40", None, "ok"),
    ("9", "2"): ("5e ES", "mo:36", "mo:34", "mo:40", None, "ok"),
    ("9", "3"): ("5e ES", "mo:36", "mo:38", "mo:40", "mo:34", "ok"),
    ("9", "8"): ("5e ES", "mo:36", "mo:37", "mo:40", "mo:34", "ok"),
    ("10", "1"): ("5e GS", "mo:36", "mo:34", "mo:41", None, "ok"),
    ("10", "4"): ("5e ES", "mo:36", "mo:34", "mo:41", None, "ok"),
    ("10", "5"): ("5e ES", "mo:36", "mo:38", "mo:41", "mo:34", "ok"),
    ("10", "6"): ("5e ES", "mo:36", "mo:37", "mo:41", "mo:34", "ok"),
    ("10", "9"): ("3e ES", "mo:36", "mo:40", "mo:41", "mo:34", "ok"),
}


@pytest.fixture(scope="module")
def initial():
    return get_calculation(
        "molcas", "caspt2", DATA / "valence_caspt2.caspt2.h5", DATA / "valence_caspt2.log",
        mcscf_file=DATA / "valence.rasscf.h5",
    )


@pytest.fixture(scope="module")
def order(initial):
    return orbital_order(initial, reference_determinant(initial))


def test_excitation_counts_holes_and_particles_per_electron():
    ids = ["a", "b", "c", "d"]
    assert excitation(list("2200"), list("2200"), ids) == Excitation((), ())
    assert excitation(list("2u0d"), list("2200"), ids) == Excitation(("b",), ("d",))
    assert excitation(list("2020"), list("2200"), ids) == Excitation(("b", "b"), ("c", "c"))


def test_reference_is_the_lowest_closed_shell_state(initial):
    assert reference_determinant(initial).state_id == "caspt2:1"


def test_open_shell_lowest_state_needs_populations_for_a_reference():
    # The core-hole states are all open-shell: the reference must come from populations.
    no_populations = get_calculation("molcas", "caspt2", DATA / "core_caspt2.caspt2.h5", DATA / "core_caspt2.log")
    with pytest.raises(OrbitalOrderError, match="no orbital populations to build a reference determinant"):
        reference_determinant(no_populations)

    final = get_calculation("molcas", "caspt2", DATA / "core_caspt2.caspt2.h5", DATA / "core_caspt2.log",
                            mcscf_file=DATA / "core.rasscf.h5")
    with pytest.raises(OrbitalOrderError, match="17 active electrons cannot form a closed-shell"):
        reference_determinant(final)


def test_without_the_ground_state_the_reference_comes_from_populations(initial):
    # As for a file of triplet excited states: no closed-shell state to take the reference from.
    excited_only = dataclasses.replace(initial, states=initial.states[1:])
    reference = reference_determinant(excited_only)
    order = orbital_order(excited_only, reference)

    assert reference.state_id is None and "".join(reference.occupations) == "222222222000"
    # max_state counts the states in the file: 3 here means states 2, 3 and 4. No GS shake-downs.
    shake_downs = find_shake_downs(excited_only, order, max_state=3)
    assert [(sd.initial_state, sd.lower_state, sd.case) for sd in shake_downs] == [
        ("caspt2:3", "caspt2:2", "5e ES"), ("caspt2:4", "caspt2:2", "3e ES"),
    ]
    lines = shake_down_report(excited_only, order, shake_downs, max_state=3)
    assert "Reference determinant: 222222222000 (from populations);  states considered: 2..4 (max_state = 3)" in lines


def test_shake_down_transitions_on_valence_states(initial, order):
    shake_downs = find_shake_downs(initial, order)

    found = {}
    for sd in shake_downs:
        key = (sd.initial_state.split(":")[1], sd.lower_state.split(":")[1])
        status = "ok" if sd.computable else sd.reason
        found[key] = (sd.case, sd.i, sd.j, sd.k, sd.d, status)

    assert set(found) == set(EXPECTED)
    for key, (case, i, j, k, d, status) in EXPECTED.items():
        assert found[key][:5] == (case, i, j, k, d), key
        assert found[key][5].startswith(status), key


def test_moved_electron_is_recorded(initial, order):
    by_pair = {(sd.initial_state, sd.lower_state): sd for sd in find_shake_downs(initial, order)}

    assert by_pair[("caspt2:2", "caspt2:1")].moved == ("mo:40", "mo:34")  # k -> i
    assert by_pair[("caspt2:4", "caspt2:2")].moved == ("mo:41", "mo:40")  # k -> j
    assert by_pair[("caspt2:3", "caspt2:2")].moved == ("mo:34", "mo:38")  # j -> i


def test_report_lists_states_transitions_and_summary(initial, order):
    lines = shake_down_report(initial, order, find_shake_downs(initial, order))

    assert "Active orbitals: mo:31 ... mo:42;  HOMO = mo:34, LUMO = mo:40" in lines
    assert any(line.split()[:2] == ["7", "222022222200"] and "double excitation" in line for line in lines)
    row = next(line for line in lines if line.lstrip().startswith("3 -> 2"))
    assert row.split()[3:12] == ["2222222u2d00", "222u22222d00", "mo:34", "->", "mo:38", "5e", "ES", "mo:38", "mo:34"]
    assert "25 transitions: 23 computable, 2 not computable." in lines


def test_max_state_limits_the_states_considered(initial, order):
    pairs = [
        (sd.initial_state.split(":")[1], sd.lower_state.split(":")[1])
        for sd in find_shake_downs(initial, order, max_state=4)
    ]

    # 4 -> 3 is not a shake-down: it would move two electrons.
    assert pairs == [("2", "1"), ("3", "1"), ("3", "2"), ("4", "1"), ("4", "2")]
    assert find_shake_downs(initial, order, max_state=1) == []
    lines = shake_down_report(initial, order, find_shake_downs(initial, order, 4), max_state=4)
    assert "Reference (ground) state: 1;  states considered: 1..4 (max_state = 4)" in lines
    assert [line.split()[0] for line in lines[6:10]] == ["1", "2", "3", "4"]


@pytest.mark.parametrize("max_state", [0, 11])
def test_max_state_outside_the_states_is_rejected(initial, max_state):
    with pytest.raises(ShakeDownError, match=f"max_state {max_state} is outside 1..10"):
        check_max_state(initial, max_state)


def test_input_override_changes_which_pairs_move_down(initial):
    # Overriding HOMO = mo:38 puts mo:34 below it: 3 -> 2 (mo:34 -> mo:38) now moves up.
    order = orbital_order(initial, reference_determinant(initial), FrontierOrbitals("mo:38", "mo:40"))
    by_pair = {(sd.initial_state, sd.lower_state): sd for sd in find_shake_downs(initial, order, max_state=4)}

    assert by_pair[("caspt2:3", "caspt2:2")].reason == "electron would move up: i (HOMO) lies above j (HOMO-1)"
    assert by_pair[("caspt2:3", "caspt2:1")].case == "3e GS"  # i = mo:38 is now the HOMO


def test_shake_down_on_a_core_orbital_has_no_rule(order):
    # A pump from the RAS1 (core) orbital mo:31 into mo:40, falling back.
    sd = shakedown._classify("a", "b", ("mo:40", "mo:31"), Excitation(("mo:31",), ("mo:40",)),
                             Excitation((), ()), True, order)

    assert sd.case is None and sd.reason == "i = mo:31 is a core orbital: no rule"
