from pathlib import Path

import pytest
import yaml

from xpsoverlaps.inputs import InputError, load_input


ROOT = Path(__file__).parents[1]
DATA = ROOT / "data" / "molcas" / "brthph" / "s2p" / "mb"


def _valid_input():
    return {
        "Calculator": {"program": "molcas", "method": "overlaps"},
        "Initial": {
            "type": "caspt2",
            "wf_file": str(DATA / "valence_caspt2.caspt2.h5"),
            "state_file": str(DATA / "valence_caspt2.log"),
            "mcscf_file": str(DATA / "valence.rasscf.h5"),
        },
        "Final": {
            "type": "caspt2",
            "wf_file": str(DATA / "core_caspt2.caspt2.h5"),
            "state_file": str(DATA / "core_caspt2.log"),
            "mcscf_file": str(DATA / "core.rasscf.h5"),
        },
    }


def _write(tmp_path, data):
    path = tmp_path / "input.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_example_input_resolves_paths_relative_to_input_file():
    data = load_input(ROOT / "inputs" / "input.yaml")

    assert data["Initial"]["wf_file"] == DATA / "valence_caspt2.caspt2.h5"
    assert data["Initial"]["mcscf_file"] == DATA / "valence.rasscf.h5"
    # A single Final block is returned as a one-element list.
    assert [final["state_file"] for final in data["Final"]] == [DATA / "core_caspt2.log"]
    assert data["Final"][0]["mcscf_file"] == DATA / "core.rasscf.h5"


def test_chp_method_does_not_require_final_block(tmp_path):
    data = _valid_input()
    data["Calculator"]["method"] = "chp"
    del data["Final"]

    assert "Final" not in load_input(_write(tmp_path, data))


def test_choice_values_are_case_insensitive_but_paths_are_not(tmp_path):
    data = _valid_input()
    data["Calculator"] = {"program": "Molcas", "method": "OVERLAPS"}
    data["Initial"]["type"] = "CASPT2"
    data["Final"]["type"] = " Caspt2 "

    loaded = load_input(_write(tmp_path, data))

    assert loaded["Calculator"] == {"program": "molcas", "method": "overlaps"}
    assert loaded["Initial"]["type"] == loaded["Final"][0]["type"] == "caspt2"
    assert loaded["Initial"]["wf_file"] == DATA / "valence_caspt2.caspt2.h5"


def _mutate_missing_final(data):
    del data["Final"]


def _mutate_type_mismatch(data):
    data["Final"]["type"] = "rasscf"
    data["Final"]["wf_file"] = str(DATA / "core.rasscf.h5")
    del data["Final"]["mcscf_file"]


def _mutate_unknown_type(data):
    data["Initial"]["type"] = "mrci"


def _mutate_unknown_program(data):
    data["Calculator"]["program"] = "orca"


def test_homo_and_lumo_are_optional_but_accepted_together(tmp_path):
    data = _valid_input()
    assert "homo" not in load_input(_write(tmp_path, data))["Initial"]

    data["Initial"].update(homo=34, lumo=40)
    loaded = load_input(_write(tmp_path, data))["Initial"]
    assert (loaded["homo"], loaded["lumo"]) == (34, 40)


def test_rasscf_input_needs_no_mcscf_file(tmp_path):
    data = _valid_input()
    for block, prefix in (("Initial", "valence"), ("Final", "core")):
        del data[block]["mcscf_file"]
        data[block].update(type="rasscf", wf_file=str(DATA / f"{prefix}.rasscf.h5"),
                           state_file=str(DATA / f"{prefix}-tz.log"))

    loaded = load_input(_write(tmp_path, data))
    assert "mcscf_file" not in loaded["Initial"] and "mcscf_file" not in loaded["Final"][0]


TRIPLET = DATA / "triplet"


def _two_finals():
    data = _valid_input()
    data["Final"] = [
        {"type": "caspt2", "wf_file": str(TRIPLET / "core-doub.caspt2.h5"),
         "state_file": str(TRIPLET / "core-doub.log"), "mcscf_file": str(TRIPLET / "core-doub-tz.rasscf.h5")},
        {"type": "CASPT2", "wf_file": str(TRIPLET / "core-quart.caspt2.h5"),
         "state_file": str(TRIPLET / "core-quart.log"), "mcscf_file": str(TRIPLET / "core-qaurt-tz.rasscf.h5")},
    ]
    return data


def test_final_accepts_a_list_of_blocks(tmp_path):
    loaded = load_input(_write(tmp_path, _two_finals()))

    assert [final["wf_file"].name for final in loaded["Final"]] == ["core-doub.caspt2.h5", "core-quart.caspt2.h5"]
    assert loaded["Final"][1]["type"] == "caspt2"  # choices are lowercased in list items too


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda final: final[1].pop("mcscf_file"), "Final[2]: 'mcscf_file' is a required property"),
        (lambda final: final[1].pop("state_file"), "Final[2]: 'state_file' is a required property"),
        (lambda final: final[0].update(homo=34, lumo=40), "Final[1]: Unevaluated properties are not allowed"),
        (lambda final: final[1].update(wf_file="missing.h5"), "Final[2].wf_file: file not found"),
        (lambda final: final[1].update(type="rasscf"), "Final[2].type: 'rasscf' must match Initial.type 'caspt2'"),
        (lambda final: final.clear(), "Final: [] should be non-empty"),
    ],
)
def test_errors_in_a_list_of_final_blocks_name_the_block(tmp_path, mutate, message):
    data = _two_finals()
    mutate(data["Final"])

    with pytest.raises(InputError) as excinfo:
        load_input(_write(tmp_path, data))
    assert message in str(excinfo.value)


def _mutate_homo_without_lumo(data):
    data["Initial"]["homo"] = 34


def _mutate_homo_in_final(data):
    data["Final"].update(homo=39, lumo=40)


def _mutate_homo_not_integer(data):
    data["Initial"].update(homo="34", lumo=40)


def _mutate_homo_zero(data):
    data["Initial"].update(homo=0, lumo=40)


def _mutate_max_state_zero(data):
    data["Initial"]["max_state"] = 0


def _mutate_max_state_in_final(data):
    data["Final"]["max_state"] = 4


def _mutate_unknown_key(data):
    data["Initial"]["wf_fiel"] = "typo.h5"


def _mutate_missing_wf_file(data):
    del data["Final"]["wf_file"]


def _mutate_missing_molcas_state_file(data):
    del data["Initial"]["state_file"]


def _mutate_missing_file(data):
    data["Final"]["wf_file"] = "does_not_exist.h5"


def _mutate_missing_mcscf_file_for_caspt2(data):
    del data["Initial"]["mcscf_file"]


def _mutate_mcscf_file_for_rasscf(data):
    for block, prefix in (("Initial", "valence"), ("Final", "core")):
        data[block].update(type="rasscf", wf_file=str(DATA / f"{prefix}.rasscf.h5"),
                           state_file=str(DATA / f"{prefix}-tz.log"))


def _mutate_missing_final_mcscf_file_for_caspt2(data):
    del data["Final"]["mcscf_file"]


def _mutate_missing_mcscf_file(data):
    data["Initial"]["mcscf_file"] = "does_not_exist.h5"


@pytest.mark.parametrize(
    "mutate,message",
    [
        (_mutate_missing_final, "'Final' is a required property"),
        (_mutate_type_mismatch, "Final.type: 'rasscf' must match Initial.type 'caspt2'"),
        (_mutate_unknown_type, "Initial.type: 'mrci' is not one of"),
        (_mutate_unknown_program, "Calculator.program: 'orca' is not one of"),
        (_mutate_unknown_key, "Unevaluated properties are not allowed ('wf_fiel' was unexpected)"),
        (_mutate_homo_without_lumo, "Initial: 'lumo' is a dependency of 'homo'"),
        (_mutate_homo_in_final, "Final: Unevaluated properties are not allowed ('homo', 'lumo' were unexpected)"),
        (_mutate_homo_not_integer, "Initial.homo: '34' is not of type 'integer'"),
        (_mutate_homo_zero, "Initial.homo: 0 is less than the minimum of 1"),
        (_mutate_max_state_zero, "Initial.max_state: 0 is less than the minimum of 1"),
        (_mutate_max_state_in_final, "Final: Unevaluated properties are not allowed ('max_state' was unexpected)"),
        (_mutate_missing_wf_file, "Final: 'wf_file' is a required property"),
        (_mutate_missing_molcas_state_file, "Initial: 'state_file' is a required property"),
        (_mutate_missing_file, "Final.wf_file: file not found"),
        (_mutate_missing_mcscf_file_for_caspt2, "Initial: 'mcscf_file' is a required property"),
        (_mutate_mcscf_file_for_rasscf, "Initial.mcscf_file: only used for type 'caspt2'"),
        (_mutate_mcscf_file_for_rasscf, "Final.mcscf_file: only used for type 'caspt2'"),
        (_mutate_missing_final_mcscf_file_for_caspt2, "Final: 'mcscf_file' is a required property"),
        (_mutate_missing_mcscf_file, "Initial.mcscf_file: file not found"),
    ],
)
def test_invalid_inputs_are_rejected(tmp_path, mutate, message):
    data = _valid_input()
    mutate(data)

    with pytest.raises(InputError) as excinfo:
        load_input(_write(tmp_path, data))
    assert message in str(excinfo.value)
