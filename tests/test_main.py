import logging
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from xpsoverlaps import main as main_module
from xpsoverlaps import methods

ROOT = Path(__file__).parents[1]
PACKAGE_DIR = ROOT / "src" / "xpsoverlaps"
INPUT = ROOT / "inputs" / "input.yaml"
FCIDUMP = ROOT / "data" / "molcas" / "brthph" / "s2p" / "mb" / "valence.FciDmp.h5"


def _record_calls(monkeypatch):
    calls = {}
    monkeypatch.setattr(
        methods,
        "propensity_overlaps",
        lambda initial, finals, frontier=None, max_state=None: calls.update(
            overlaps=(initial, finals, frontier)
        ),
    )
    monkeypatch.setattr(
        methods,
        "propensity_chp",
        lambda initial, integrals, frontier=None, max_state=None: calls.update(chp=(initial, frontier, integrals)),
    )
    return calls


def _chp_input(tmp_path, fcidump=FCIDUMP):
    run_input = yaml.safe_load(INPUT.read_text(encoding="utf-8"))
    run_input["Calculator"]["method"] = "chp"
    del run_input["Final"]
    for key in ("wf_file", "state_file", "mcscf_file"):
        run_input["Initial"][key] = str((INPUT.parent / run_input["Initial"][key]).resolve())
    run_input["Initial"]["fcidump_file"] = str(fcidump)
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
    initial, finals, frontier = calls["overlaps"]
    assert len(initial.states) == 10
    assert [len(final.states) for final in finals] == [40]
    assert (frontier.homo, frontier.lumo) == ("mo:34", "mo:40")


def test_main_chp_loads_only_initial_calculation(monkeypatch, tmp_path):
    calls = _record_calls(monkeypatch)

    assert main_module.main(["-i", str(_chp_input(tmp_path))]) == 0

    assert set(calls) == {"chp"}
    initial, frontier, integrals = calls["chp"]
    assert len(initial.states) == 10
    assert (frontier.homo, frontier.lumo) == ("mo:34", "mo:40")
    assert integrals.integrals.orbital_ids == initial.active_space.orbital_ids


def test_main_rejects_an_fcidump_file_of_another_calculation(monkeypatch, tmp_path, capsys):
    calls = _record_calls(monkeypatch)
    triplet = ROOT / "data" / "molcas" / "brthph" / "s2p" / "mb" / "triplet"
    input_file = _chp_input(tmp_path)
    run_input = yaml.safe_load(input_file.read_text(encoding="utf-8"))
    run_input["Initial"].update(wf_file=str(triplet / "valence-trip.caspt2.h5"),
                                state_file=str(triplet / "valence-trip.log"),
                                mcscf_file=str(triplet / "valence-trip-tz.rasscf.h5"))
    run_input["Initial"].pop("homo"), run_input["Initial"].pop("lumo")
    input_file.write_text(yaml.safe_dump(run_input), encoding="utf-8")

    assert main_module.main(["-i", str(input_file)]) == 1

    assert calls == {}
    assert "does not match the calculation: orbital energy of mo:31" in capsys.readouterr().err


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
    assert "HOMO      mo:34         1.5967     -0.2721\n" in text
    assert "HOMO      mo:34     mo:39         1.4850     -0.4661\n" in text
    assert "HOMO mo:34 set by the input (agrees with the populations)." in text
    assert "Shake-down transitions between initial states" in text
    assert "states considered: 1..4 (max_state = 4)" in text
    assert "5 transitions: 5 computable, 0 not computable." in text
    assert "Propensity rules" in text
    assert "  3 -> 2  5e ES   Final     D1                       0.015099" in text


def test_uncaught_exception_is_logged_with_traceback(monkeypatch, tmp_path):
    def failing_overlaps(initial, final, frontier=None, max_state=None):
        raise RuntimeError("orbital mapping failed")

    monkeypatch.setattr(methods, "propensity_overlaps", failing_overlaps)
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


def test_methods_module_logs_to_output_file(monkeypatch, tmp_path):
    def fake_overlaps(initial, final, frontier=None, max_state=None):
        methods.logger.info("message from methods module")

    monkeypatch.setattr(methods, "propensity_overlaps", fake_overlaps)
    output = tmp_path / "run.out"

    main_module.main(["-i", str(INPUT), "-o", str(output)])

    assert "message from methods module" in output.read_text(encoding="utf-8")
    assert not any(
        isinstance(handler, logging.FileHandler)
        for handler in logging.getLogger("xpsoverlaps").handlers
    )


def test_warnings_go_to_the_terminal_as_well_as_the_output_file(monkeypatch, tmp_path, capsys):
    def warning_overlaps(initial, final, frontier=None, max_state=None):
        methods.logger.info("info only in the file")
        methods.logger.warning("* WARNING: AMBIGUOUS REFERENCE DETERMINANT")

    monkeypatch.setattr(methods, "propensity_overlaps", warning_overlaps)
    output = tmp_path / "run.out"

    main_module.main(["-i", str(INPUT), "-o", str(output)])

    stderr = capsys.readouterr().err
    assert "* WARNING: AMBIGUOUS REFERENCE DETERMINANT" in stderr
    assert "info only in the file" not in stderr
    assert "* WARNING: AMBIGUOUS REFERENCE DETERMINANT" in output.read_text(encoding="utf-8")
    assert logging.getLogger("xpsoverlaps").handlers == [] or all(
        isinstance(h, logging.NullHandler) for h in logging.getLogger("xpsoverlaps").handlers
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


def test_main_orders_orbitals_by_population_without_homo_lumo(tmp_path):
    run_input = yaml.safe_load(INPUT.read_text(encoding="utf-8"))
    for key in ("homo", "lumo"):
        del run_input["Initial"][key]
    for block in ("Initial", "Final"):
        for key in ("wf_file", "state_file", "mcscf_file"):
            if key in run_input[block]:
                run_input[block][key] = str((INPUT.parent / run_input[block][key]).resolve())
    input_file = tmp_path / "no_frontier.yaml"
    input_file.write_text(yaml.safe_dump(run_input), encoding="utf-8")

    assert main_module.main(["-i", str(input_file), "-o", str(tmp_path / "run.out")]) == 0

    text = (tmp_path / "run.out").read_text(encoding="utf-8")
    assert "HOMO/LUMO from the populations (not given in the input)." in text
    assert "  3 -> 2  5e ES   Final     D1                       0.015099" in text


def test_main_rejects_invalid_homo_lumo_before_running(monkeypatch, tmp_path):
    calls = _record_calls(monkeypatch)
    input_file = _chp_input(tmp_path)
    run_input = yaml.safe_load(input_file.read_text(encoding="utf-8"))
    run_input["Initial"]["lumo"] = 20  # an inactive orbital
    input_file.write_text(yaml.safe_dump(run_input), encoding="utf-8")

    assert main_module.main(["-i", str(input_file)]) == 1

    assert calls == {}
    assert "LUMO mo:20 is not an active orbital" in (tmp_path / "chp.out").read_text(encoding="utf-8")


def test_main_rejects_max_state_beyond_the_initial_states(monkeypatch, tmp_path):
    calls = _record_calls(monkeypatch)
    input_file = _chp_input(tmp_path)
    run_input = yaml.safe_load(input_file.read_text(encoding="utf-8"))
    run_input["Initial"]["max_state"] = 11  # the initial calculation has 10 states
    input_file.write_text(yaml.safe_dump(run_input), encoding="utf-8")

    assert main_module.main(["-i", str(input_file)]) == 1

    assert calls == {}
    assert "max_state 11 is outside 1..10" in (tmp_path / "chp.out").read_text(encoding="utf-8")
