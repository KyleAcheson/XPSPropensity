from pathlib import Path

import pytest

from xpsoverlaps.frontier import FrontierOrbitals
from xpsoverlaps.readers import get_calculation
from xpsoverlaps.shakedown import (
    Excitation,
    ShakeDownError,
    excitation,
    find_shake_downs,
    reference_state,
    shake_down_report,
)


DATA = Path(__file__).parents[1] / "data" / "molcas" / "brthph" / "s2p" / "mb"
FRONTIER = FrontierOrbitals("mo:34", "mo:40")

# (a, b): (case, i, j, k, d, status), worked out by hand from the dominant configurations.
EXPECTED = {
    ("2", "1"): ("3e GS", "mo:34", None, "mo:40", None, "ok"),
    ("3", "1"): ("3e GS", "mo:38", None, "mo:40", "mo:34", "ok"),
    ("3", "2"): ("5e ES", "mo:38", "mo:34", "mo:40", None, "ok"),
    ("4", "1"): ("3e GS", "mo:34", None, "mo:41", None, "ok"),
    ("4", "2"): ("3e ES", "mo:34", "mo:40", "mo:41", None, "ok"),
    ("5", "1"): ("3e GS", "mo:38", None, "mo:41", "mo:34", "ok"),
    ("5", "3"): ("3e ES", "mo:38", "mo:40", "mo:41", "mo:34", "ok"),
    ("5", "4"): ("5e ES", "mo:38", "mo:34", "mo:41", None, "ok"),
    ("6", "1"): ("3e GS", "mo:37", None, "mo:41", "mo:34", "ok"),
    ("6", "4"): ("5e ES", "mo:37", "mo:34", "mo:41", None, "ok"),
    ("6", "5"): ("5e ES", "mo:37", "mo:38", "mo:41", "mo:34", "orbital order unknown"),
    ("7", "2"): (None, None, None, None, None, "double excitation"),
    ("8", "1"): ("3e GS", "mo:37", None, "mo:40", "mo:34", "ok"),
    ("8", "2"): ("5e ES", "mo:37", "mo:34", "mo:40", None, "ok"),
    ("8", "3"): ("5e ES", "mo:37", "mo:38", "mo:40", "mo:34", "orbital order unknown"),
    ("8", "6"): ("3e ES", "mo:37", "mo:41", "mo:40", "mo:34", "electron would move up"),
    ("9", "1"): ("3e GS", "mo:36", None, "mo:40", "mo:34", "ok"),
    ("9", "2"): ("5e ES", "mo:36", "mo:34", "mo:40", None, "ok"),
    ("9", "3"): ("5e ES", "mo:36", "mo:38", "mo:40", "mo:34", "orbital order unknown"),
    ("9", "8"): ("5e ES", "mo:36", "mo:37", "mo:40", "mo:34", "orbital order unknown"),
    ("10", "1"): ("3e GS", "mo:36", None, "mo:41", "mo:34", "ok"),
    ("10", "4"): ("5e ES", "mo:36", "mo:34", "mo:41", None, "ok"),
    ("10", "5"): ("5e ES", "mo:36", "mo:38", "mo:41", "mo:34", "orbital order unknown"),
    ("10", "6"): ("5e ES", "mo:36", "mo:37", "mo:41", "mo:34", "orbital order unknown"),
    ("10", "9"): ("3e ES", "mo:36", "mo:40", "mo:41", "mo:34", "ok"),
}


@pytest.fixture(scope="module")
def initial():
    return get_calculation(
        "molcas", "caspt2", DATA / "valence_caspt2.caspt2.h5", DATA / "valence_caspt2.log"
    )


def test_excitation_counts_holes_and_particles_per_electron():
    ids = ["a", "b", "c", "d"]
    assert excitation(list("2200"), list("2200"), ids) == Excitation((), ())
    assert excitation(list("2u0d"), list("2200"), ids) == Excitation(("b",), ("d",))
    assert excitation(list("2020"), list("2200"), ids) == Excitation(("b", "b"), ("c", "c"))


def test_reference_is_the_lowest_closed_shell_state(initial):
    assert reference_state(initial).state_id == "caspt2:1"


def test_open_shell_reference_is_rejected():
    final = get_calculation("molcas", "caspt2", DATA / "core_caspt2.caspt2.h5", DATA / "core_caspt2.log")

    with pytest.raises(ShakeDownError, match="lowest state caspt2:1 .* is not closed-shell"):
        reference_state(final)


def test_shake_down_transitions_on_valence_states(initial):
    shake_downs = find_shake_downs(initial, FRONTIER)

    found = {}
    for sd in shake_downs:
        key = (sd.initial_state.split(":")[1], sd.lower_state.split(":")[1])
        status = "ok" if sd.computable else sd.reason
        found[key] = (sd.case, sd.i, sd.j, sd.k, sd.d, status)

    assert set(found) == set(EXPECTED)
    for key, (case, i, j, k, d, status) in EXPECTED.items():
        assert found[key][:5] == (case, i, j, k, d), key
        assert found[key][5].startswith(status), key


def test_moved_electron_is_recorded(initial):
    by_pair = {(sd.initial_state, sd.lower_state): sd for sd in find_shake_downs(initial, FRONTIER)}

    assert by_pair[("caspt2:2", "caspt2:1")].moved == ("mo:40", "mo:34")  # k -> i
    assert by_pair[("caspt2:4", "caspt2:2")].moved == ("mo:41", "mo:40")  # k -> j
    assert by_pair[("caspt2:3", "caspt2:2")].moved == ("mo:34", "mo:38")  # j -> i


def test_report_lists_states_transitions_and_summary(initial):
    lines = shake_down_report(initial, FRONTIER, find_shake_downs(initial, FRONTIER))

    assert "Orbital order: mo:31 ... mo:42;  HOMO = mo:34, LUMO = mo:40" in lines
    assert any(line.split()[:2] == ["7", "222022222200"] and "double excitation" in line for line in lines)
    row = next(line for line in lines if line.lstrip().startswith("3 -> 2"))
    assert row.split()[3:12] == ["2222222u2d00", "222u22222d00", "mo:34", "->", "mo:38", "5e", "ES", "mo:38", "mo:34"]
    assert "25 transitions: 17 computable, 8 not computable." in lines

