import logging
from pathlib import Path

import pytest
import yaml

from xpsoverlaps import main as main_module
from xpsoverlaps.methods import FinalCalculationError, final_spins, propensity_overlaps
from xpsoverlaps.readers import get_calculation


ROOT = Path(__file__).parents[1]
TRIPLET = ROOT / "data" / "molcas" / "brthph" / "s2p" / "mb" / "triplet"
INPUT = ROOT / "examples" / "brthph" / "triplet" / "input_triplet.yaml"


def _load(name, mcscf):
    return get_calculation("molcas", "caspt2", TRIPLET / f"{name}.caspt2.h5", TRIPLET / f"{name}.log",
                           mcscf_file=TRIPLET / f"{mcscf}.rasscf.h5")


@pytest.fixture(scope="module")
def calculations():
    # Triplet initial states (no closed-shell ground state), doublet and quartet core-hole states.
    return (_load("valence-trip", "valence-trip-tz"), _load("core-doub", "core-doub-tz"),
            _load("core-quart", "core-qaurt-tz"))


@pytest.fixture(scope="module")
def output(calculations):
    initial, doublet, quartet = calculations
    lines = []

    class Collect(logging.Handler):
        def emit(self, record):
            lines.append(record.getMessage())

    logger = logging.getLogger("xpsoverlaps")
    handler, level = Collect(), logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        propensity_overlaps(initial, [doublet, quartet])
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)
    return lines


def test_each_final_calculation_has_one_spin_and_spins_differ(calculations):
    initial, doublet, quartet = calculations

    assert final_spins([doublet, quartet]) == [0.5, 1.5]
    with pytest.raises(FinalCalculationError, match="More than one final calculation has S = 0.5"):
        final_spins([doublet, doublet])


def test_every_section_covers_both_final_calculations_in_turn(output):
    doublet, quartet = "Final 1: S = 0.5 (core-doub.caspt2.h5)", "Final 2: S = 1.5 (core-quart.caspt2.h5)"
    titles = [line for line in output if line in (doublet, quartet)]

    # Mapping, mapped CI, orbital order, propensity rules: doublet then quartet in each.
    assert titles == [doublet, quartet] * 4
    sections = [output.index(heading) for heading in (
        "Dominant CI vectors on the initial active-orbital order", "Orbital order",
        "Shake-down transitions between initial states", "Propensity rules", "Summary")]
    assert sections == sorted(sections)
    assert output.count("Shake-down transitions between initial states") == 1  # initial states only


def test_triplet_channels_use_the_final_calculation_of_their_spin(output):
    start = output.index("Summary") + 2
    summary = output[start:output.index("", start)]
    rows = {tuple(line.split()[:3]) + (line.split()[7],): line.split() for line in summary}

    # 4 -> 1 (3e ES, k -> j = LUMO+2 -> LUMO): D2 from the doublets, Q from the quartets.
    assert rows[("4", "->", "1", "D2")][5:] == ["Final", "1", "D2", "0.001115"]
    assert rows[("4", "->", "1", "Q")][5:] == ["Final", "2", "Q", "0.000602"]
    # Separate final orbitals: Q is not simply twice D2 (the 1/3 : 2/3 spin weights).
    assert float(rows[("2", "->", "1", "Q")][8]) > 2 * float(rows[("2", "->", "1", "D2")][8])
    assert len(summary) == 14  # 7 shake-downs x (D2, Q)


def test_the_reference_comes_from_populations_for_triplet_states(output):
    assert any(line.startswith("No closed-shell state among the initial states: reference determinant "
                               "222222222000") for line in output)
    assert "Reference determinant: 222222222000 (from populations);  states considered: 1..6" in output


def test_main_runs_a_list_of_final_blocks(tmp_path):
    run = tmp_path / "run.out"

    assert main_module.main(["-i", str(INPUT), "-o", str(run)]) == 0

    text = run.read_text(encoding="utf-8")
    assert "Final 1 states (CASPT2): 40 states" in text and "Final 2 states (CASPT2): 40 states" in text
    assert "  4 -> 1  3e ES   Final 2   Q                        0.000602" in text


def test_main_rejects_two_final_calculations_of_the_same_spin(tmp_path, capsys):
    run_input = yaml.safe_load(INPUT.read_text(encoding="utf-8"))
    run_input["Final"] = [run_input["Final"][0], run_input["Final"][0]]
    for block in [run_input["Initial"], *run_input["Final"]]:
        for key in ("wf_file", "state_file", "mcscf_file"):
            block[key] = str((INPUT.parent / block[key]).resolve())
    input_file = tmp_path / "two_doublets.yaml"
    input_file.write_text(yaml.safe_dump(run_input), encoding="utf-8")

    assert main_module.main(["-i", str(input_file)]) == 1
    assert "More than one final calculation has S = 0.5" in capsys.readouterr().err
