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
        },
        "Final": {
            "type": "caspt2",
            "wf_file": str(DATA / "core_caspt2.caspt2.h5"),
            "state_file": str(DATA / "core_caspt2.log"),
        },
    }


def _write(tmp_path, data):
    path = tmp_path / "input.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_example_input_resolves_paths_relative_to_input_file():
    data = load_input(ROOT / "inputs" / "input.yaml")

    assert data["Initial"]["wf_file"] == DATA / "valence_caspt2.caspt2.h5"
    assert data["Final"]["state_file"] == DATA / "core_caspt2.log"


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
    assert loaded["Initial"]["type"] == loaded["Final"]["type"] == "caspt2"
    assert loaded["Initial"]["wf_file"] == DATA / "valence_caspt2.caspt2.h5"


def _mutate_missing_final(data):
    del data["Final"]


def _mutate_type_mismatch(data):
    data["Final"]["type"] = "rasscf"


def _mutate_unknown_type(data):
    data["Initial"]["type"] = "mrci"


def _mutate_unknown_program(data):
    data["Calculator"]["program"] = "orca"


def _mutate_unknown_key(data):
    data["Initial"]["wf_fiel"] = "typo.h5"


def _mutate_missing_wf_file(data):
    del data["Final"]["wf_file"]


def _mutate_missing_molcas_state_file(data):
    del data["Initial"]["state_file"]


def _mutate_missing_file(data):
    data["Final"]["wf_file"] = "does_not_exist.h5"


@pytest.mark.parametrize(
    "mutate,message",
    [
        (_mutate_missing_final, "'Final' is a required property"),
        (_mutate_type_mismatch, "Final.type: 'rasscf' must match Initial.type 'caspt2'"),
        (_mutate_unknown_type, "Initial.type: 'mrci' is not one of"),
        (_mutate_unknown_program, "Calculator.program: 'orca' is not one of"),
        (_mutate_unknown_key, "Additional properties are not allowed"),
        (_mutate_missing_wf_file, "Final: 'wf_file' is a required property"),
        (_mutate_missing_molcas_state_file, "Initial: 'state_file' is a required property"),
        (_mutate_missing_file, "Final.wf_file: file not found"),
    ],
)
def test_invalid_inputs_are_rejected(tmp_path, mutate, message):
    data = _valid_input()
    mutate(data)

    with pytest.raises(InputError) as excinfo:
        load_input(_write(tmp_path, data))
    assert message in str(excinfo.value)
