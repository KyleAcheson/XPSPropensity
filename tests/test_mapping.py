import dataclasses
from pathlib import Path

import numpy as np
import pytest

from xpsoverlaps.composition import (
    OrbitalComposition,
    ao_group,
    composition_similarity,
    describe_orbital_pairs,
    orbital_ao_weights,
    orbital_compositions,
)
from xpsoverlaps.mapping import (
    MappingOptions,
    OrbitalMappingError,
    map_active_orbitals,
    match_active_orbitals,
)
from xpsoverlaps.readers import get_calculation


DATA = Path(__file__).parents[1] / "data" / "molcas" / "brthph" / "s2p" / "mb"


@pytest.fixture(scope="module")
def valence_and_core():
    initial = get_calculation(
        "molcas", "caspt2", DATA / "valence_caspt2.caspt2.h5", DATA / "valence_caspt2.log"
    )
    final = get_calculation(
        "molcas", "caspt2", DATA / "core_caspt2.caspt2.h5", DATA / "core_caspt2.log"
    )
    return initial, final


# --- AO composition -------------------------------------------------------


@pytest.mark.parametrize(
    "atom,l,group",
    [("BR6", 1, "BR6 p"), ("C1", 0, "C1 s"), ("S5", 2, "S5 d"), ("H7", 4, "H7 g"), ("X1", 12, "X1 l=12")],
)
def test_ao_group_uses_atom_and_angular_momentum(atom, l, group):
    assert ao_group(atom, l) == group


def test_ao_groups_come_from_structured_metadata_not_label_text(valence_and_core):
    initial, _ = valence_and_core
    relabelled = dataclasses.replace(
        initial,
        basis=dataclasses.replace(
            initial.basis, function_labels=[f"ao{i}" for i in range(initial.basis.n_functions)]
        ),
    )

    original = orbital_compositions(initial)
    unlabelled = orbital_compositions(relabelled)

    for orbital_id, composition in original.items():
        assert unlabelled[orbital_id].groups == pytest.approx(composition.groups)
    assert next(iter(original["mo:33"].groups)) == "S5 p"


def test_orbital_ao_weights_sum_to_one_for_each_active_orbital(valence_and_core):
    initial, _ = valence_and_core

    weights = orbital_ao_weights(initial)

    assert weights.shape == (239, 12)
    np.testing.assert_allclose(weights.sum(axis=0), 1.0, atol=1e-8)


def test_compositions_keep_only_dominant_contributions_but_group_every_ao(valence_and_core):
    initial, _ = valence_and_core

    compositions = orbital_compositions(initial, threshold=0.05, max_count=3)

    assert list(compositions) == initial.active_space.orbital_ids
    for composition in compositions.values():
        weights = [c.weight for c in composition.dominant]
        assert 0 < len(weights) <= 3
        assert all(w >= 0.05 for w in weights)
        assert weights == sorted(weights, reverse=True)
        assert sum(composition.groups.values()) == pytest.approx(1.0)
        group_values = list(composition.groups.values())
        assert group_values == sorted(group_values, reverse=True)


def test_sulfur_2p_core_orbital_is_described_as_sulfur_p(valence_and_core):
    initial, _ = valence_and_core

    composition = orbital_compositions(initial, ["mo:33"])["mo:33"]

    assert next(iter(composition.groups)) == "S5 p"
    assert composition.dominant[0].label == "S5 2P[m=+0]"
    lines = describe_orbital_pairs([(composition, composition)])
    assert any("S5 2P[m=+0]" in line and "|" in line for line in lines)


def _composition(orbital_id, **groups):
    return OrbitalComposition(orbital_id, [], groups)


def test_composition_similarity_is_shared_group_fraction():
    a = _composition("a", **{"BR6 p": 0.7, "C1 p": 0.3})

    assert composition_similarity(a, a) == pytest.approx(1.0)
    assert composition_similarity(a, _composition("b", **{"S5 p": 1.0})) == 0.0
    assert composition_similarity(a, _composition("c", **{"BR6 p": 0.5, "S5 p": 0.5})) == (
        pytest.approx(0.5)
    )


# --- Mapping on the bundled valence -> core calculation --------------------


def test_valence_to_core_active_mapping(valence_and_core):
    initial, final = valence_and_core

    mapping = map_active_orbitals(initial, final)

    expected = {"mo:34": "mo:39", "mo:38": "mo:34", "mo:39": "mo:38"}
    for match in mapping.matches:
        assert match.final_id == expected.get(match.initial_id, match.initial_id)
        assert match.assigned_by == "overlap"
    assert mapping.active_index_map == [0, 1, 2, 8, 4, 5, 6, 3, 7, 9, 10, 11]
    assert len(mapping.warnings) == 1
    assert mapping.warnings[0].startswith("Close runner-up for mo:36 -> mo:36")


def test_mapping_does_not_modify_calculations(valence_and_core):
    initial, final = valence_and_core
    coefficients = final.orbitals.coefficients.copy()
    active_ids = list(final.active_space.orbital_ids)

    map_active_orbitals(initial, final)

    np.testing.assert_array_equal(final.orbitals.coefficients, coefficients)
    assert final.active_space.orbital_ids == active_ids


# --- Mapping edge cases on small synthetic overlaps ------------------------


IDS = ["i1", "a1", "a2", "a3", "s1"]  # inactive, three active, secondary
ACTIVE = ["a1", "a2", "a3"]


def _match(overlap, initial_groups=None, final_groups=None, **options):
    initial_groups = initial_groups or {i: {"X p": 1.0} for i in ACTIVE}
    final_groups = final_groups or {i: {"X p": 1.0} for i in ACTIVE}
    return match_active_orbitals(
        np.asarray(overlap, dtype=float),
        IDS,
        IDS,
        ACTIVE,
        ACTIVE,
        {i: _composition(i, **g) for i, g in initial_groups.items()},
        {i: _composition(i, **g) for i, g in final_groups.items()},
        MappingOptions(**options),
    )


def _overlap(active_block):
    overlap = np.eye(5)
    overlap[1:4, 1:4] = active_block
    return overlap


def test_one_to_one_argmax_is_used_directly():
    mapping = _match(_overlap([[0.1, 0.9, 0.1], [0.95, 0.1, 0.1], [0.1, 0.1, 0.97]]))

    assert [m.final_id for m in mapping.matches] == ["a2", "a1", "a3"]
    assert mapping.active_index_map == [1, 0, 2]
    assert mapping.warnings == []


def test_argmax_collision_is_resolved_by_ao_composition():
    # a1 and a2 both overlap most with final a1; composition says a2 is final a2.
    overlap = _overlap([[0.80, 0.50, 0.0], [0.78, 0.60, 0.0], [0.0, 0.0, 0.99]])
    initial_groups = {"a1": {"BR6 p": 1.0}, "a2": {"S5 p": 1.0}, "a3": {"C1 p": 1.0}}
    final_groups = {"a1": {"BR6 p": 1.0}, "a2": {"S5 p": 1.0}, "a3": {"C1 p": 1.0}}

    mapping = _match(overlap, initial_groups, final_groups)

    assert [m.final_id for m in mapping.matches] == ["a1", "a2", "a3"]
    assert [m.assigned_by for m in mapping.matches] == ["composition", "composition", "overlap"]
    assert any("largest overlap with several initial orbitals (a1, a2)" in w for w in mapping.warnings)


def test_collision_with_opposite_composition_follows_composition():
    overlap = _overlap([[0.80, 0.50, 0.0], [0.78, 0.60, 0.0], [0.0, 0.0, 0.99]])
    initial_groups = {"a1": {"S5 p": 1.0}, "a2": {"BR6 p": 1.0}, "a3": {"C1 p": 1.0}}
    final_groups = {"a1": {"BR6 p": 1.0}, "a2": {"S5 p": 1.0}, "a3": {"C1 p": 1.0}}

    mapping = _match(overlap, initial_groups, final_groups)

    assert [m.final_id for m in mapping.matches] == ["a2", "a1", "a3"]


def test_initial_orbital_rotating_out_of_active_space_is_an_error():
    overlap = _overlap(np.eye(3) * 0.4)
    overlap[1, 4] = 0.9  # initial a1 overlaps most with final secondary s1

    with pytest.raises(OrbitalMappingError, match="Initial active a1 overlaps most with final s1"):
        _match(overlap)


def test_final_orbital_rotating_into_active_space_is_detected():
    overlap = _overlap(np.eye(3) * 0.6)
    overlap[0, 0] = 0.5
    overlap[0, 3] = 0.9  # final a3 overlaps most with initial inactive i1

    with pytest.raises(OrbitalMappingError, match="Final active a3 overlaps most with initial i1"):
        _match(overlap)


def test_rotation_out_can_be_downgraded_to_a_warning():
    overlap = _overlap(np.eye(3) * 0.4)
    overlap[1, 4] = 0.9

    mapping = _match(overlap, rotated_out="warning")

    assert any("outside the final active space" in w for w in mapping.warnings)
    assert [m.final_id for m in mapping.matches] == ACTIVE


def test_low_overlap_and_close_runner_up_warnings():
    overlap = _overlap([[0.45, 0.0, 0.0], [0.0, 0.70, 0.65], [0.0, 0.10, 0.99]])

    mapping = _match(overlap)

    assert any(w.startswith("Low overlap for a1 -> a1") for w in mapping.warnings)
    assert any(w.startswith("Close runner-up for a2 -> a2") for w in mapping.warnings)
    assert not any("a3 ->" in w for w in mapping.warnings)


def test_mapping_options_and_active_sizes_are_validated():
    with pytest.raises(ValueError, match="rotated_out"):
        MappingOptions(rotated_out="ignore")
    with pytest.raises(OrbitalMappingError, match="differ in size"):
        match_active_orbitals(np.eye(5), IDS, IDS, ACTIVE, ["a1", "a2"], {}, {})
