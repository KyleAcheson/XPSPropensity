from pathlib import Path

import pytest

from xpsoverlaps.frontier import (
    FrontierOrbitalError,
    FrontierOrbitals,
    OrbitalOrderError,
    frontier_from_indices,
    map_frontier,
    orbital_order,
    orbital_order_report,
)
from xpsoverlaps.mapping import map_active_orbitals
from xpsoverlaps.readers import get_calculation
from xpsoverlaps.shakedown import reference_state


DATA = Path(__file__).parents[1] / "data" / "molcas" / "brthph" / "s2p" / "mb"


@pytest.fixture(scope="module")
def initial():
    return get_calculation(
        "molcas", "caspt2", DATA / "valence_caspt2.caspt2.h5", DATA / "valence_caspt2.log",
        mcscf_file=DATA / "valence.rasscf.h5",
    )


@pytest.fixture(scope="module")
def mapping(initial):
    final = get_calculation("molcas", "caspt2", DATA / "core_caspt2.caspt2.h5", DATA / "core_caspt2.log")
    return map_active_orbitals(initial, final)


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


def test_final_frontier_follows_the_orbital_mapping(initial, mapping):
    frontier = frontier_from_indices(initial, 34, 40)

    # Initial mo:34 matches final mo:39 (see the mapping tests); mo:40 maps to itself.
    assert map_frontier(frontier, mapping) == FrontierOrbitals("mo:39", "mo:40")


# --- Orbital order from populations ------------------------------------------


def test_orbitals_are_ordered_by_population_within_occupied_and_virtual(initial):
    order = orbital_order(initial, reference_state(initial))

    # Reference 222222222000; RAS2 is mo:34-42: mo:34-39 occupied, mo:40-42 virtual.
    # Lowest occupied population first (mo:34 1.597, mo:38 1.742, ...), highest virtual first.
    assert order.occupied == ("mo:34", "mo:38", "mo:37", "mo:36", "mo:39", "mo:35")
    assert order.virtual == ("mo:40", "mo:41", "mo:42")
    assert order.frontier == order.by_population == FrontierOrbitals("mo:34", "mo:40")
    assert [order.label(o) for o in ("mo:38", "mo:34", "mo:40", "mo:42")] == ["HOMO-1", "HOMO", "LUMO", "LUMO+2"]
    assert order.level("mo:37") < order.level("mo:38") < order.level("mo:34") < order.level("mo:40")


def test_ras1_orbitals_are_core_and_not_ranked(initial):
    order = orbital_order(initial, reference_state(initial))

    # mo:31-33 (RAS1) are the S 2p orbitals.
    assert order.unranked == (("mo:31", "core"), ("mo:32", "core"), ("mo:33", "core"))
    assert order.label("mo:31") == "core" and not order.is_valence("mo:31")
    with pytest.raises(OrbitalOrderError, match="mo:31 is not a ranked"):
        order.level("mo:31")


def test_input_homo_lumo_override_the_population_order(initial):
    order = orbital_order(initial, reference_state(initial), FrontierOrbitals("mo:37", "mo:41"))

    # The override moves to the top of its set; the rest keep the population order.
    assert order.occupied[:4] == ("mo:37", "mo:34", "mo:38", "mo:36")
    assert order.virtual == ("mo:41", "mo:40", "mo:42")
    assert order.by_population == FrontierOrbitals("mo:34", "mo:40")


def test_override_must_match_the_reference_occupations(initial):
    with pytest.raises(OrbitalOrderError, match="LUMO mo:39 is not a virtual RAS2 orbital in the reference state caspt2:1"):
        orbital_order(initial, reference_state(initial), FrontierOrbitals("mo:34", "mo:39"))
    with pytest.raises(OrbitalOrderError, match="HOMO mo:33 is not an? occupied RAS2 orbital"):
        orbital_order(initial, reference_state(initial), FrontierOrbitals("mo:33", "mo:40"))


def test_order_needs_orbital_populations():
    no_populations = get_calculation(
        "molcas", "caspt2", DATA / "valence_caspt2.caspt2.h5", DATA / "valence_caspt2.log"
    )

    with pytest.raises(OrbitalOrderError, match="give its RASSCF HDF5 file as mcscf_file"):
        orbital_order(no_populations, reference_state(no_populations))


def test_order_report_lists_labels_populations_energies_and_final_orbitals(initial, mapping):
    populations = orbital_order_report(orbital_order(initial, reference_state(initial)), initial, mapping)
    overridden = orbital_order_report(
        orbital_order(initial, reference_state(initial), FrontierOrbitals("mo:38", "mo:40")), initial, mapping
    )

    assert "HOMO/LUMO from the populations (not given in the input)." in populations
    assert "LUMO      mo:40         0.5547      0.0556  mo:40" in populations
    assert "HOMO      mo:34         1.5967     -0.2721  mo:39" in populations
    assert not any(line.endswith("*") for line in populations)  # energy order agrees here
    assert populations[-3:] == [
        "core      mo:31         2.0000     -6.7656  mo:31",
        "core      mo:32         2.0000     -7.5674  mo:32",
        "core      mo:33         2.0000     -6.6816  mo:33",
    ]
    assert "HOMO mo:38 set by the input (populations give mo:34)." in overridden
    assert "LUMO mo:40 set by the input (agrees with the populations)." in overridden
    # mo:38 above mo:34 contradicts their energies, so both rows are flagged.
    assert "HOMO      mo:38         1.7416     -0.3299  mo:34  *" in overridden
    assert "HOMO-1    mo:34         1.5967     -0.2721  mo:39  *" in overridden


def test_occupations_are_not_checked(initial):
    # mo:39 is occupied in the ground state; choosing sensible orbitals is the user's job.
    assert frontier_from_indices(initial, 34, 39) == FrontierOrbitals("mo:34", "mo:39")
