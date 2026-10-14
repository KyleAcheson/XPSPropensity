from pathlib import Path

import pytest

from xpsoverlaps.frontier import (
    FrontierOrbitalError,
    FrontierOrbitals,
    frontier_from_indices,
    frontier_report,
    map_frontier,
)
from xpsoverlaps.mapping import map_active_orbitals
from xpsoverlaps.readers import get_calculation


DATA = Path(__file__).parents[1] / "data" / "molcas" / "brthph" / "s2p" / "mb"


@pytest.fixture(scope="module")
def initial():
    return get_calculation(
        "molcas", "caspt2", DATA / "valence_caspt2.caspt2.h5", DATA / "valence_caspt2.log"
    )


def test_valid_indices_give_orbital_ids(initial):
    assert frontier_from_indices(initial, 34, 40) == FrontierOrbitals("mo:34", "mo:40")


@pytest.mark.parametrize(
    "homo,lumo,message",
    [
        (34, 300, "LUMO index 300 is outside 1..239"),
        (20, 40, "HOMO mo:20 is not an active orbital"),
        (34, 34, "HOMO and LUMO are the same orbital"),
    ],
)
def test_invalid_indices_are_rejected(initial, homo, lumo, message):
    with pytest.raises(FrontierOrbitalError, match=message):
        frontier_from_indices(initial, homo, lumo)


def test_final_frontier_follows_the_orbital_mapping(initial):
    final = get_calculation("molcas", "caspt2", DATA / "core_caspt2.caspt2.h5", DATA / "core_caspt2.log")
    mapping = map_active_orbitals(initial, final)
    frontier = frontier_from_indices(initial, 34, 40)

    mapped = map_frontier(frontier, mapping)

    # Initial mo:34 matches final mo:39 (see the mapping tests); mo:40 maps to itself.
    assert mapped == FrontierOrbitals("mo:39", "mo:40")
    assert frontier_report(frontier, mapped)[2:] == [
        "HOMO  initial mo:34    -> final mo:39",
        "LUMO  initial mo:40    -> final mo:40",
    ]


def test_occupations_are_not_checked(initial):
    # mo:39 is occupied in the ground state; choosing sensible orbitals is the user's job.
    assert frontier_from_indices(initial, 34, 39) == FrontierOrbitals("mo:34", "mo:39")
