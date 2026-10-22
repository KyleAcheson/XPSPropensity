"""Command-line entry point: run XPSoverlaps from a YAML input file."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import List, Optional

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xpsoverlaps import methods
from xpsoverlaps.frontier import FrontierOrbitalError, frontier_from_indices
from xpsoverlaps.shakedown_transitions import ShakeDownError, check_max_state
from xpsoverlaps.inputs import CalculationInput, InputError, RunInput, load_input
from xpsoverlaps.model import Calculation
from xpsoverlaps.readers import get_calculation

# Parent of every module logger (logging.getLogger(__name__)) in the package.
logger = logging.getLogger("xpsoverlaps")


def _load_block(program: str, block: CalculationInput) -> Calculation:
    return get_calculation(
        program, block["type"], block["wf_file"], block.get("state_file"), block.get("mcscf_file")
    )


_WIDTH = 80
_DASHES = "-" * _WIDTH


def _log_header() -> None:
    logger.info("=" * _WIDTH)
    logger.info("XPSoverlaps Calculation".center(_WIDTH).rstrip())
    logger.info("=" * _WIDTH)
    logger.info("")


def _log_input(run_input: RunInput) -> None:
    """Dump the validated input, with file paths resolved."""
    def printable_block(fields):
        return {key: str(value) if isinstance(value, Path) else value for key, value in fields.items()}

    printable = {
        block: [printable_block(item) for item in fields] if isinstance(fields, list) else printable_block(fields)
        for block, fields in run_input.items()
    }
    logger.info("Input:")
    logger.info("%s", yaml.safe_dump(printable, sort_keys=False).rstrip())
    logger.info("")
    logger.info(_DASHES)


def _log_unmapped_ci_note() -> None:
    stars = "*" * _WIDTH
    logger.info("")
    logger.info(stars)
    logger.info("* NOTE: The CI vectors below are printed exactly as read, each in its own")
    logger.info("* calculation's active-orbital order. They are NOT yet mapped onto a common")
    logger.info("* orbital ordering, so Initial and Final occupation strings cannot be")
    logger.info("* compared position by position.")
    logger.info(stars)


def _log_states(block: str, calculation: Calculation) -> None:
    """Summarize the states as read, in the calculation's own orbital order."""
    active_ids = calculation.active_space.orbital_ids
    ci_width = max(len(active_ids), len("Dominant CI vector"))
    columns = f"{'State':>5}  {'Energy (Eh)':>16}  {'Spin (S)':>8}  {'Dominant CI vector':<{ci_width}}  {'Weight':>8}"

    logger.info("")
    logger.info(
        "%s states (%s): %d states", block, calculation.provenance.method, len(calculation.states)
    )
    logger.info("CI vector orbital order: %s ... %s", active_ids[0], active_ids[-1])
    logger.info("")
    logger.info(columns)
    logger.info("-" * len(columns))
    for state in calculation.states:
        configuration = state.dominant_configuration
        energy = f"{state.energy:.8f}" if state.energy is not None else "-"
        spin = f"{state.spin:.1f}" if state.spin is not None else "-"
        occupations = "".join(configuration.occupations) if configuration else "-"
        weight = (
            f"{configuration.weight:.4f}"
            if configuration and configuration.weight is not None
            else "-"
        )
        logger.info(
            f"{state.state_id.split(':')[-1]:>5}  {energy:>16}  {spin:>8}  "
            f"{occupations:<{ci_width}}  {weight:>8}"
        )
    logger.info("")
    logger.info(_DASHES)


def _run(input_file: Path) -> int:
    _log_header()
    try:
        run_input = load_input(input_file)
    except InputError as error:
        logger.error("%s", error)
        print(error, file=sys.stderr)
        return 1
    _log_input(run_input)
    _log_unmapped_ci_note()

    program = run_input["Calculator"]["program"]
    initial = _load_block(program, run_input["Initial"])
    _log_states("Initial", initial)

    frontier = None
    max_state = run_input["Initial"].get("max_state")
    try:
        if "homo" in run_input["Initial"]:  # the schema guarantees lumo is given too
            frontier = frontier_from_indices(
                initial, run_input["Initial"]["homo"], run_input["Initial"]["lumo"]
            )
        check_max_state(initial, max_state)
    except (FrontierOrbitalError, ShakeDownError) as error:
        logger.error("%s", error)
        print(error, file=sys.stderr)
        return 1

    if run_input["Calculator"]["method"] == "overlaps":
        blocks = run_input["Final"]
        finals = [_load_block(program, block) for block in blocks]
        for n, final in enumerate(finals, start=1):
            _log_states("Final" if len(finals) == 1 else f"Final {n}", final)
        try:
            methods.final_spins(finals)
        except methods.FinalCalculationError as error:
            logger.error("%s", error)
            print(error, file=sys.stderr)
            return 1
        methods.propensity_overlaps(initial, finals, frontier, max_state)
    else:
        methods.propensity_chp(initial, frontier)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i", "--input", dest="input_file", metavar="INPUT", type=Path, required=True,
        help="YAML input file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output file (default: the input file name with a .out suffix)",
    )
    args = parser.parse_args(argv)

    handler = logging.FileHandler(
        args.output or args.input_file.with_suffix(".out"), mode="w", encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    # Warnings (e.g. an ambiguous reference determinant) also go to the terminal;
    # errors are printed there by _run or raised.
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter("%(message)s"))
    console.addFilter(lambda record: record.levelno == logging.WARNING)
    logger.addHandler(handler)
    logger.addHandler(console)
    logger.setLevel(logging.INFO)
    try:
        return _run(args.input_file)
    except Exception:
        # Record any uncaught error (readers, methods, ...) with its traceback.
        logger.exception("XPSoverlaps terminated with an error.")
        raise
    finally:
        for attached in (handler, console):
            logger.removeHandler(attached)
            attached.close()


if __name__ == "__main__":
    sys.exit(main())
