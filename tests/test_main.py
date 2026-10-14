import logging
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from xpsoverlaps import main as main_module
from xpsoverlaps import propensity

ROOT = Path(__file__).parents[1]
PACKAGE_DIR = ROOT / "src" / "xpsoverlaps"
INPUT = ROOT / "inputs" / "input.yaml"


def _record_calls(monkeypatch):
    calls = {}
    monkeypatch.setattr(
        propensity,
        "propensity_overlaps",
        lambda initial, final, frontier=None: calls.update(overlaps=(initial, final, frontier)),
    )
    monkeypatch.setattr(
        propensity, "propensity_chp", lambda initial, frontier=None: calls.update(chp=(initial, frontier))
    )
    return calls


def _chp_input(tmp_path):
    run_input = yaml.safe_load(INPUT.read_text(encoding="utf-8"))
    run_input["Calculator"]["method"] = "chp"
    for block in ("Initial", "Final"):
        for key in ("wf_file", "state_file"):
            run_input[block][key] = str((INPUT.parent / run_input[block][key]).resolve())
    input_file = tmp_path / "chp.yaml"
    input_file.write_text(yaml.safe_dump(run_input), encoding="utf-8")
    return input_file


def test_main_runs_directly_from_package_directory(tmp_path):
    output = tmp_path / "run.out"
    result = subprocess.run(
        [sys.executable, str(PACKAGE_DIR / "main.py"), "-i", str(INPUT), "-o", str(output)],
        cwd=PACKAGE_DIR,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert output.is_file()


def test_main_overlaps_loads_initial_and_final_calculations(monkeypatch, tmp_path):
    calls = _record_calls(monkeypatch)

    assert main_module.main(["-i", str(INPUT), "-o", str(tmp_path / "run.out")]) == 0

    assert set(calls) == {"overlaps"}
    initial, final, frontier = calls["overlaps"]
    assert len(initial.states) == 10
    assert len(final.states) == 40
    assert (frontier.homo, frontier.lumo) == ("mo:34", "mo:40")


def test_main_chp_loads_only_initial_calculation(monkeypatch, tmp_path):
    calls = _record_calls(monkeypatch)

    assert main_module.main(["-i", str(_chp_input(tmp_path))]) == 0

    assert set(calls) == {"chp"}
    initial, frontier = calls["chp"]
    assert len(initial.states) == 10
    assert (frontier.homo, frontier.lumo) == ("mo:34", "mo:40")


def test_output_file_dumps_input_and_summarizes_states(tmp_path):
    output = tmp_path / "run.out"

    main_module.main(["-i", str(INPUT), "-o", str(output)])

    text = output.read_text(encoding="utf-8")
    assert text.index("XPSoverlaps Calculation") < text.index("Input:")
    assert "Input:\nCalculator:\n  program: molcas\n  method: overlaps" in text
    assert str(ROOT / "data" / "molcas" / "brthph" / "s2p" / "mb" / "core_caspt2.log") in text
    assert "NOT yet mapped onto a common" in text
    assert "Initial states (CASPT2): 10 states" in text
    assert "Final states (CASPT2): 40 states" in text
    assert "CI vector orbital order: mo:31 ... mo:42" in text
    assert "State       Energy (Eh)  Spin (S)  Dominant CI vector    Weight" in text
    assert "    1    -3157.05052010       0.0  222222222000          0.9052" in text
    assert "caspt2:" not in text
    assert "Dominant CI vectors on the initial active-orbital order" in text
    assert "22u22222du00     22ud22222u00" in text
    assert "HOMO  initial mo:34    -> final mo:39" in text
    assert "LUMO  initial mo:40    -> final mo:40" in text
    assert "Shake-down transitions between initial states" in text
    assert "25 transitions: 17 computable, 8 not computable." in text


def test_uncaught_exception_is_logged_with_traceback(monkeypatch, tmp_path):
    def failing_overlaps(initial, final, frontier=None):
        raise RuntimeError("orbital mapping failed")

    monkeypatch.setattr(propensity, "propensity_overlaps", failing_overlaps)
    output = tmp_path / "run.out"

    with pytest.raises(RuntimeError):
        main_module.main(["-i", str(INPUT), "-o", str(output)])

    text = output.read_text(encoding="utf-8")
    assert "XPSoverlaps terminated with an error." in text
    assert "Traceback" in text and "RuntimeError: orbital mapping failed" in text


def test_output_defaults_to_input_name_with_out_suffix(monkeypatch, tmp_path):
    _record_calls(monkeypatch)
    input_file = _chp_input(tmp_path)

    main_module.main(["-i", str(input_file)])

    assert (tmp_path / "chp.out").is_file()


def test_propensity_module_logs_to_output_file(monkeypatch, tmp_path):
    def fake_overlaps(initial, final, frontier=None):
        propensity.logger.info("message from propensity module")

    monkeypatch.setattr(propensity, "propensity_overlaps", fake_overlaps)
    output = tmp_path / "run.out"

    main_module.main(["-i", str(INPUT), "-o", str(output)])

    assert "message from propensity module" in output.read_text(encoding="utf-8")
    assert not any(
        isinstance(handler, logging.FileHandler)
        for handler in logging.getLogger("xpsoverlaps").handlers
    )


def test_main_reports_invalid_input(tmp_path):
    input_file = tmp_path / "input.yaml"
    input_file.write_text("Calculator: {program: molcas, method: overlaps}\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(PACKAGE_DIR / "main.py"), "-i", str(input_file)],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "'Initial' is a required property" in result.stderr
    output_text = (tmp_path / "input.out").read_text(encoding="utf-8")
    assert "Invalid input file" in output_text
    assert "'Initial' is a required property" in output_text


def test_main_requires_input_option(capsys):
    with pytest.raises(SystemExit):
        main_module.main([])
    assert "the following arguments are required: -i/--input" in capsys.readouterr().err


def test_main_rejects_invalid_homo_lumo_before_running(monkeypatch, tmp_path):
    calls = _record_calls(monkeypatch)
    input_file = _chp_input(tmp_path)
    run_input = yaml.safe_load(input_file.read_text(encoding="utf-8"))
    run_input["Initial"]["lumo"] = 20  # an inactive orbital
    input_file.write_text(yaml.safe_dump(run_input), encoding="utf-8")

    assert main_module.main(["-i", str(input_file)]) == 1

    assert calls == {}
    assert "LUMO mo:20 is not an active orbital" in (tmp_path / "chp.out").read_text(encoding="utf-8")

