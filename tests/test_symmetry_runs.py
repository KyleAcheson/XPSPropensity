"""Overlaps-method runs with point-group symmetry, through the program (ADR-0005).

Each Initial block (one irrep) is analysed on its own; for each core-hole irrep
Γ_c among its RAS1 orbitals the rules use the Final block of irrep Γ_I × Γ_c.
"""

from pathlib import Path
import re

import pytest
import yaml

from xpsoverlaps import main as main_module

ROOT = Path(__file__).parents[1]
THIOPHENOL = ROOT / "data" / "molcas" / "thiophenol" / "s2p" / "dz"
CS2 = ROOT / "data" / "molcas" / "cs2" / "s2p" / "tz" / "singlet" / "2B2"
BRTHPH = ROOT / "data" / "molcas" / "brthph" / "s2p" / "mb"


def _rasscf(directory, name):
    return {"type": "rasscf", "wf_file": str(directory / f"{name}.rasscf.h5"),
            "state_file": str(directory / f"{name}.log")}


def _caspt2(directory, name, mcscf):
    return {"type": "caspt2", "wf_file": str(directory / f"{name}.caspt2.h5"),
            "state_file": str(directory / f"{name}.log"), "mcscf_file": str(directory / f"{mcscf}.rasscf.h5")}


def _run(tmp_path, initial, final, name="run"):
    """Run the overlaps method; return (exit code, output text)."""
    input_file = tmp_path / f"{name}.yaml"
    input_file.write_text(yaml.safe_dump({"Calculator": {"program": "molcas", "method": "overlaps"},
                                          "Initial": initial, "Final": final}), encoding="utf-8")
    output = tmp_path / f"{name}.out"
    code = main_module.main(["-i", str(input_file), "-o", str(output)])
    return code, output.read_text(encoding="utf-8")


def _part(text, title):
    """The output of one Initial block: from its title to the next block's title or the closing summary."""
    start = text.index(title)
    rest = text[start + len(title):]
    end = re.search(r"\nInitial \d+: |\nSummary over all initial", rest)
    return text[start:start + len(title) + (end.start() if end else len(rest))]


@pytest.fixture(scope="module")
def thiophenol_rasscf(tmp_path_factory):
    """The a' block with both finals (the a" block cannot be mapped until valence-dz-ap is rerun, see #17)."""
    code, text = _run(tmp_path_factory.mktemp("thiophenol"), _rasscf(THIOPHENOL, "valence-dz-a"),
                      [_rasscf(THIOPHENOL, "core-dz-a"), _rasscf(THIOPHENOL, "core-dz-ap")])
    assert code == 0
    return text


A1 = "Initial: A' (valence-dz-a.rasscf.h5)"
F1 = "Final 1: S = 0.5, A' (core-dz-a.rasscf.h5)"
F2 = 'Final 2: S = 0.5, A" (core-dz-ap.rasscf.h5)'


def test_each_core_hole_irrep_uses_the_final_calculation_of_irrep_initial_times_core(thiophenol_rasscf):
    # a' initial: in-plane S 2p (a') -> A' final, out-of-plane S 2p (a") -> A" final.
    rules = thiophenol_rasscf.split("Propensity rules", 1)[1].split("\nSummary", 1)[0]
    for heading, final in (("Core hole a' (mo:21_i1, mo:22_i1) -> final A'", F1),
                           ("Core hole a\" (mo:1_i2) -> final A\"", F2)):
        after = rules[rules.index(heading):]
        assert min((after.index(f), f) for f in (F1, F2) if f in after)[1] == final
    summary = thiophenol_rasscf.split("\nSummary\n", 1)[1].split("\n\n", 1)[0].splitlines()[1:]
    assert {tuple(row.split()[5:9]) for row in summary} == {("a'", "A'", "Final", "1"), ('a"', 'A"', "Final", "2")}


def test_a_symmetric_initial_block_warns_about_cross_core_shake_downs(thiophenol_rasscf):
    assert thiophenol_rasscf.count(
        "Initial (A'): cross-core shake-downs to states of other irreps are symmetry-allowed but not included") == 1


def test_the_core_orbital_overlap_block_is_reported_per_irrep(thiophenol_rasscf):
    block = thiophenol_rasscf.split(F1, 1)[1].split("Core (RAS1) orbital overlaps", 1)[1].split(F2, 1)[0]
    # In-plane S 2p (a') rotate by ~17 degrees between the initial and final calculations; 2p_z (a") does not.
    assert "Irrep a'\nInitial   mo:21_i1  mo:22_i1\nmo:21_i1     0.959     0.282" in block
    assert 'Irrep a"\nInitial    mo:1_i2\nmo:1_i2      1.000' in block


def test_the_example_runs_every_initial_block_and_reports_the_one_that_cannot_be_mapped(tmp_path, capsys):
    # valence-dz-ap holds the S 3s instead of the S-H sigma in its a' RAS2 (to be rerun, see #17): its
    # block is skipped with the mapping error; the a' block and the closing summary are still produced.
    output = tmp_path / "example.out"
    assert main_module.main(["-i", str(ROOT / "examples" / "thiophenol" / "singlet" / "input.yaml"),
                             "-o", str(output)]) == 1
    text = output.read_text(encoding="utf-8")
    first, second = "Initial 1: A' (valence-a-caspt2.caspt2.h5)", 'Initial 2: A" (valence-ap-caspt2.caspt2.h5)'
    assert "Propensity rules" in _part(text, first)
    skipped = 'Initial 2 (A") is skipped: its active orbitals cannot be mapped onto Final.'
    assert skipped in _part(text, second) and "Propensity rules" not in _part(text, second)
    summary = text.split("Summary over all initial calculations", 1)[1]
    assert {row.split()[0] for row in summary.split("Not evaluated:")[0].strip().splitlines()[2:]} == {"A'"}
    assert skipped in summary.split("Not evaluated:")[1]
    assert skipped in capsys.readouterr().err


def test_a_missing_final_irrep_is_reported(tmp_path):
    # CASPT2 with only the A" final (the A' final is being rerun, see #17).
    code, text = _run(tmp_path, [_caspt2(THIOPHENOL, "valence-a-caspt2", "valence-dz-a")],
                      _caspt2(THIOPHENOL, "core-ap-caspt2", "core-dz-ap"))
    assert code == 0
    rules = text.split("Propensity rules", 1)[1]
    assert "Core hole a' (mo:21_i1, mo:22_i1) -> final A'" in rules
    assert "No A' final calculation." in rules
    assert "no S = 0.5, A' final calculation" in rules.split("Summary", 1)[1]


def test_cs2_maps_its_initial_block_onto_every_final_irrep_and_flags_rotated_core_orbitals(tmp_path):
    # B2 initial states; RAS1 holds a1, b2, a2 and b1 core orbitals, so every final irrep B2 x c is reached.
    finals = [_caspt2(CS2, f"core-s2p-{irrep}-2b2", f"core-s2p-{irrep}-tz-2b2") for irrep in ("a1", "a2", "b1", "b2")]
    code, text = _run(tmp_path, _caspt2(CS2, "valence-s2p-2b2", "valence-s2p-tz-2b2"), finals)
    assert code == 0
    for n, irrep in enumerate(("A1", "A2", "B1", "B2"), start=1):
        assert f"Final {n}: S = 0.5, {irrep} (core-s2p-{irrep.lower()}-2b2.caspt2.h5)" in text
    assert "symmetry-forbidden" not in text
    # The two S atoms' 2p combinations of one irrep mix between initial and final (core-hole localisation).
    assert "Core orbital mo:5_i1 rotates in Final 2: |S| with its match mo:5_i1 is 0.768 < 0.9" in text
    # Its two B2 states differ by two electrons: no shake-down.
    assert "0 transitions: 0 computable, 0 not computable." in text


@pytest.mark.parametrize("initial, final, message", [
    ([_rasscf(THIOPHENOL, "valence-dz-a"), _rasscf(THIOPHENOL, "valence-dz-a")], _rasscf(THIOPHENOL, "core-dz-a"),
     "Initial 1 and Initial 2 have the same irrep A'"),
    (_rasscf(THIOPHENOL, "valence-dz-a"), [_rasscf(THIOPHENOL, "core-dz-a"), _rasscf(THIOPHENOL, "core-dz-a")],
     "More than one final calculation has S = 0.5 and irrep A'"),
    (_rasscf(THIOPHENOL, "valence-dz-a"), _rasscf(BRTHPH, "core") | {"state_file": str(BRTHPH / "core-tz.log")},
     "All calculations must use the same point group"),
])
def test_inconsistent_blocks_are_rejected(tmp_path, capsys, initial, final, message):
    code, _ = _run(tmp_path, initial, final)
    assert code == 1
    assert message in capsys.readouterr().err


def test_homo_and_lumo_of_a_symmetric_calculation_name_an_orbital_of_an_irrep(tmp_path, capsys):
    initial = _rasscf(THIOPHENOL, "valence-dz-a") | {"homo": "2_i2", "lumo": "6_i2"}
    code, text = _run(tmp_path, initial, _rasscf(THIOPHENOL, "core-dz-a"))
    assert code == 0
    assert "HOMO mo:2_i2 set by the input (agrees with the populations)." in text

    code, _ = _run(tmp_path, _rasscf(THIOPHENOL, "valence-dz-a") | {"homo": 2, "lumo": 6}, _rasscf(THIOPHENOL, "core-dz-a"))
    assert code == 1
    assert "give the HOMO as <index>_i<irrep>, e.g. 2_i2" in capsys.readouterr().err


# --- First-order core-hole potential -------------------------------------------


@pytest.fixture(scope="module")
def thiophenol_chp(tmp_path_factory):
    output = tmp_path_factory.mktemp("chp") / "chp.out"
    code = main_module.main(["-i", str(ROOT / "examples" / "thiophenol" / "singlet" / "input_chp.yaml"),
                             "-o", str(output)])
    assert code == 0
    return output.read_text(encoding="utf-8")


C1 = "Initial 1: A' (valence-a-caspt2.caspt2.h5)"
C2 = 'Initial 2: A" (valence-ap-caspt2.caspt2.h5)'


def test_chp_labels_each_core_hole_with_the_final_irrep_it_reaches(thiophenol_chp):
    # Every core orbital of every Initial block: thiophenol A'->A', A'->A", A"->A", A"->A'.
    expected = {C1: ("mo:21_i1 (a') -> final A'", "mo:22_i1 (a') -> final A'", 'mo:1_i2 (a") -> final A"'),
                C2: ("mo:21_i1 (a') -> final A\"", "mo:22_i1 (a') -> final A\"", "mo:1_i2 (a\") -> final A'")}
    for title, cores in expected.items():
        part = _part(thiophenol_chp, title)
        overlaps, rules = part.split("Core-hole overlaps", 1)[1].split("Propensity rules (core-hole potential)", 1)
        for core in cores:
            assert f"Core hole in {core}" in overlaps and f"Core hole in {core}" in rules
        assert "Final irreps: " + ", ".join(core.replace("final ", "") for core in cores) in rules


def test_chp_runs_each_initial_block_and_warns_about_cross_core_shake_downs(thiophenol_chp):
    for title, label in ((C1, "Initial 1 (A')"), (C2, 'Initial 2 (A")')):
        assert _part(thiophenol_chp, title).count(f"{label}: cross-core shake-downs") == 1
    # The a" block has no closed-shell state: no GS shake-down.
    a2 = _part(thiophenol_chp, C2).split("Shake-down transitions between initial states", 1)[1].split("----", 1)[0]
    assert "(from populations)" in a2 and " GS " not in a2


def test_chp_closing_summary_covers_every_initial_block_and_core(thiophenol_chp):
    summary = thiophenol_chp.split("Summary over all initial calculations (core-hole potential)\n\n", 1)[1]
    rows = [line.split() for line in summary.split("\n\n", 1)[0].splitlines()[1:]]
    routes = {(row[0], row[-3], row[-2]) for row in rows}
    assert routes == {("A'", "a'", "A'"), ("A'", 'a"', 'A"'), ('A"', "a'", 'A"'), ('A"', 'a"', "A'")}


def test_chp_rejects_two_initial_blocks_of_the_same_irrep(tmp_path, capsys):
    block = _caspt2(THIOPHENOL, "valence-a-caspt2", "valence-dz-a") | {
        "fcidump_file": str(THIOPHENOL / "valence-dz-a.FciDmp.h5")}
    input_file = tmp_path / "chp.yaml"
    input_file.write_text(yaml.safe_dump({"Calculator": {"program": "molcas", "method": "chp"},
                                          "Initial": [block, block]}), encoding="utf-8")
    assert main_module.main(["-i", str(input_file), "-o", str(tmp_path / "chp.out")]) == 1
    assert "Initial 1 and Initial 2 have the same irrep A'" in capsys.readouterr().err
