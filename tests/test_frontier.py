import dataclasses
import logging
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
    reference_warning,
)
from xpsoverlaps.mapping import MappedFinal, map_active_orbitals
from xpsoverlaps.readers import get_calculation
from xpsoverlaps.shakedown_transitions import reference_determinant


DATA = Path(__file__).parents[1] / "data" / "molcas" / "brthph" / "s2p" / "mb"


@pytest.fixture(scope="module")
def initial():
    return get_calculation(
        "molcas", "caspt2", DATA / "valence_caspt2.caspt2.h5", DATA / "valence_caspt2.log",
        mcscf_file=DATA / "valence.rasscf.h5",
    )


@pytest.fixture(scope="module")
def final():
    return get_calculation(
        "molcas", "caspt2", DATA / "core_caspt2.caspt2.h5", DATA / "core_caspt2.log",
        mcscf_file=DATA / "core.rasscf.h5",
    )


@pytest.fixture(scope="module")
def mapping(initial, final):
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
    order = orbital_order(initial, reference_determinant(initial))

    # Reference 222222222000; RAS2 is mo:34-42: mo:34-39 occupied, mo:40-42 virtual.
    # Lowest occupied population first (mo:34 1.597, mo:38 1.742, ...), highest virtual first.
    assert order.occupied == ("mo:34", "mo:38", "mo:37", "mo:36", "mo:39", "mo:35")
    assert order.virtual == ("mo:40", "mo:41", "mo:42")
    assert order.frontier == order.by_population == FrontierOrbitals("mo:34", "mo:40")
    assert [order.label(o) for o in ("mo:38", "mo:34", "mo:40", "mo:42")] == ["HOMO-1", "HOMO", "LUMO", "LUMO+2"]
    assert order.level("mo:37") < order.level("mo:38") < order.level("mo:34") < order.level("mo:40")


def test_ras1_orbitals_are_core_and_not_ranked(initial):
    order = orbital_order(initial, reference_determinant(initial))

    # mo:31-33 (RAS1) are the S 2p orbitals.
    assert order.unranked == (("mo:31", "core"), ("mo:32", "core"), ("mo:33", "core"))
    assert order.label("mo:31") == "core" and not order.is_valence("mo:31")
    with pytest.raises(OrbitalOrderError, match="mo:31 is not a ranked"):
        order.level("mo:31")


def test_input_homo_lumo_override_the_population_order(initial):
    order = orbital_order(initial, reference_determinant(initial), FrontierOrbitals("mo:37", "mo:41"))

    # The override moves to the top of its set; the rest keep the population order.
    assert order.occupied[:4] == ("mo:37", "mo:34", "mo:38", "mo:36")
    assert order.virtual == ("mo:41", "mo:40", "mo:42")
    assert order.by_population == FrontierOrbitals("mo:34", "mo:40")


def test_override_must_match_the_reference_occupations(initial):
    with pytest.raises(OrbitalOrderError, match="LUMO mo:39 is not a virtual RAS2 orbital in the reference state caspt2:1"):
        orbital_order(initial, reference_determinant(initial), FrontierOrbitals("mo:34", "mo:39"))
    with pytest.raises(OrbitalOrderError, match="HOMO mo:33 is not an? occupied RAS2 orbital"):
        orbital_order(initial, reference_determinant(initial), FrontierOrbitals("mo:33", "mo:40"))


def test_order_needs_orbital_populations():
    no_populations = get_calculation(
        "molcas", "caspt2", DATA / "valence_caspt2.caspt2.h5", DATA / "valence_caspt2.log"
    )

    with pytest.raises(OrbitalOrderError, match="give its RASSCF HDF5 file as mcscf_file"):
        orbital_order(no_populations, reference_determinant(no_populations))


def test_order_report_lists_initial_and_final_populations_and_energies(initial, final, mapping):
    finals = [MappedFinal("Final", final, mapping, 0.5)]
    order = orbital_order(initial, reference_determinant(initial))
    populations = orbital_order_report(order, initial, finals)
    overridden = orbital_order_report(
        orbital_order(initial, reference_determinant(initial), FrontierOrbitals("mo:38", "mo:40")), initial, finals
    )

    assert "HOMO/LUMO from the populations (not given in the input)." in populations
    initial_rows = populations[populations.index("Initial") + 2:]
    assert "LUMO      mo:40         0.5547      0.0556" in initial_rows
    assert "HOMO      mo:34         1.5967     -0.2721" in initial_rows
    assert not any(line.endswith("*") for line in populations)  # energy order agrees here
    # The final table: the final orbital each initial one maps onto (mo:34 -> mo:39).
    final_rows = populations[populations.index("Final: S = 0.5 (core_caspt2.caspt2.h5)") + 2:]
    assert "LUMO      mo:40     mo:40         0.5970     -0.1705" in final_rows
    assert "HOMO      mo:34     mo:39         1.4850     -0.4661" in final_rows
    # One S 2p hole, state-averaged over the final roots: about 1.5 in mo:31 and mo:33.
    assert final_rows[-3:] == [
        "core      mo:31     mo:31         1.4998     -7.4863",
        "core      mo:32     mo:32         2.0000     -8.8514",
        "core      mo:33     mo:33         1.5002     -7.4873",
    ]
    assert "HOMO mo:38 set by the input (populations give mo:34)." in overridden
    assert "LUMO mo:40 set by the input (agrees with the populations)." in overridden
    # mo:38 above mo:34 contradicts their initial energies, so both rows are flagged.
    assert "HOMO      mo:38         1.7416     -0.3299  *" in overridden
    assert "HOMO-1    mo:34         1.5967     -0.2721  *" in overridden


def test_order_report_has_one_final_table_per_final_calculation(initial, final, mapping):
    no_populations = get_calculation("molcas", "caspt2", DATA / "core_caspt2.caspt2.h5", DATA / "core_caspt2.log")
    finals = [MappedFinal("Final 1", final, mapping, 0.5), MappedFinal("Final 2", no_populations, mapping, 1.5)]

    lines = orbital_order_report(orbital_order(initial, reference_determinant(initial)), initial, finals)

    first = lines.index("Final 1: S = 0.5 (core_caspt2.caspt2.h5)")
    second = lines.index("Final 2: S = 1.5 (core_caspt2.caspt2.h5)")
    assert "HOMO      mo:34     mo:39         1.4850     -0.4661" in lines[first:second]
    assert "HOMO      mo:34     mo:39              -           -" in lines[second:]  # no populations


def test_occupations_are_not_checked(initial):
    # mo:39 is occupied in the ground state; choosing sensible orbitals is the user's job.
    assert frontier_from_indices(initial, 34, 39) == FrontierOrbitals("mo:34", "mo:39")


# --- Reference determinant built from populations ----------------------------


def _excited_only(calculation, swap_frontier_populations=False):
    """The calculation without its ground state, as for a file of triplet excited states;
    optionally with the HOMO (mo:34) and LUMO (mo:40) populations made nearly equal and swapped."""
    orbitals = calculation.orbitals
    occupations = list(orbitals.occupations)
    if swap_frontier_populations:
        occupations[33], occupations[39] = 0.98, 1.02
    return dataclasses.replace(
        calculation, states=calculation.states[1:],
        orbitals=dataclasses.replace(orbitals, occupations=occupations),
    )


def test_population_reference_fills_the_highest_populations(initial):
    reference = reference_determinant(_excited_only(initial))

    # 18 active electrons doubly occupy the 9 most populated orbitals, mo:31-39.
    assert "".join(reference.occupations) == "222222222000"
    assert reference.boundary == ("mo:34", "mo:40")
    assert reference.gap == pytest.approx(1.5967 - 0.5547, abs=1e-4)
    assert not reference.ambiguous and reference_warning(reference, initial) == []


def test_close_populations_give_a_big_warning(initial):
    excited = _excited_only(initial, swap_frontier_populations=True)
    reference = reference_determinant(excited)

    # mo:40 (1.02) now outranks mo:34 (0.98): the reference is swapped and flagged.
    assert "".join(reference.occupations) == "222022222200"  # mo:34 empty, mo:40 doubly occupied
    assert reference.ambiguous
    warning = reference_warning(reference, excited)
    assert warning[1] == "* WARNING: AMBIGUOUS REFERENCE DETERMINANT"
    assert "* orbital have close populations: mo:40 1.0200 (occupied) and" in warning
    assert "* (1-based MO indices) in the Initial input block to fix the reference." in warning


def test_input_homo_lumo_pin_an_ambiguous_population_reference(initial):
    excited = _excited_only(initial, swap_frontier_populations=True)
    reference = reference_determinant(excited, override=FrontierOrbitals("mo:34", "mo:40"))

    assert "".join(reference.occupations) == "222222222000"
    assert reference.ambiguous and reference_warning(reference, excited) == []
    order = orbital_order(excited, reference, FrontierOrbitals("mo:34", "mo:40"))
    assert order.frontier == FrontierOrbitals("mo:34", "mo:40")


def test_ambiguous_reference_warning_is_logged_before_the_order(initial, final, caplog):
    from xpsoverlaps.methods import propensity_overlaps

    with caplog.at_level(logging.INFO, logger="xpsoverlaps"):
        propensity_overlaps(_excited_only(initial, swap_frontier_populations=True), final, max_state=3)

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert "* WARNING: AMBIGUOUS REFERENCE DETERMINANT" in warnings
    messages = [r.getMessage() for r in caplog.records]
    assert messages.index("* WARNING: AMBIGUOUS REFERENCE DETERMINANT") < messages.index("Orbital order")
