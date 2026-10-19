"""Loading and validation of YAML input files against ``input_schema.yaml``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, TypedDict

import jsonschema
import yaml


SCHEMA_PATH = Path(__file__).with_name("input_schema.yaml")
_FILE_KEYS = ("wf_file", "state_file", "mcscf_file")


# Shape of a validated input file (see input_schema.yaml). Choice values are
# lower case and file paths are absolute. Optional keys live in the
# ``total=False`` subclasses (Python 3.9 has no ``NotRequired``).
class CalculatorInput(TypedDict):
    program: str
    method: Literal["overlaps", "chp"]


class _CalculationInputRequired(TypedDict):
    type: str
    wf_file: Path


class CalculationInput(_CalculationInputRequired, total=False):
    state_file: Path  # only for programs that write states to a separate file
    mcscf_file: Path  # MCSCF file of a correlated run (Molcas CASPT2: its RASSCF HDF5 file)
    homo: int  # Initial only; 1-based MO indices, given together with lumo
    lumo: int
    max_state: int  # Initial only; highest initial state used for shake-downs


class _RunInputRequired(TypedDict):
    Calculator: CalculatorInput
    Initial: CalculationInput


class RunInput(_RunInputRequired, total=False):
    Final: CalculationInput  # required by the schema when method is "overlaps"


class InputError(ValueError):
    """Raised when an input file violates the schema or a cross-field rule."""

    def __init__(self, source: Path, problems: List[str]) -> None:
        self.problems = problems
        details = "\n".join(f"  - {problem}" for problem in problems)
        super().__init__(f"Invalid input file {source}:\n{details}")


def load_schema() -> Dict[str, Any]:
    with SCHEMA_PATH.open(encoding="utf-8") as schema_file:
        return yaml.safe_load(schema_file)


def _location(path: Any) -> str:
    return ".".join(str(part) for part in path) or "<root>"


def _resolve(schema: Dict[str, Any], root: Dict[str, Any]) -> Dict[str, Any]:
    """Follow a local ``$ref`` such as ``'#/$defs/calculation'``."""
    reference = schema.get("$ref")
    if reference is None:
        return schema
    for part in reference.lstrip("#/").split("/"):
        root = root[part]
    return root


def _lowercase_choices(data: Any, schema: Dict[str, Any], root: Dict[str, Any]) -> None:
    """Lowercase values of properties restricted to a string ``enum``.

    This makes choice fields (program, method, type, ...) case-insensitive;
    the schema's enum values must therefore be written in lowercase. Free-form
    strings such as file paths are left unchanged.
    """
    if not isinstance(data, dict):
        return
    for key, property_schema in _resolve(schema, root).get("properties", {}).items():
        if key not in data:
            continue
        property_schema = _resolve(property_schema, root)
        choices = property_schema.get("enum")
        if isinstance(data[key], str) and choices and all(isinstance(c, str) for c in choices):
            data[key] = data[key].strip().lower()
        else:
            _lowercase_choices(data[key], property_schema, root)


def load_input(source: str | Path) -> RunInput:
    """Read and validate an input file.

    Relative file paths (``wf_file``, ``state_file``, ``mcscf_file``) are
    resolved against the input file's directory and returned as absolute
    ``Path`` objects. All problems are reported together in a single
    ``InputError``.
    """
    path = Path(source).resolve()
    with path.open(encoding="utf-8") as input_file:
        data = yaml.safe_load(input_file)
    if data is None:
        raise InputError(path, ["file is empty"])

    schema = load_schema()
    _lowercase_choices(data, schema, schema)
    validator = jsonschema.Draft202012Validator(schema)
    problems = [
        f"{_location(error.absolute_path)}: {error.message}"
        for error in sorted(validator.iter_errors(data), key=lambda error: list(error.absolute_path))
    ]
    if problems:
        raise InputError(path, problems)

    # Rules JSON Schema cannot express (comparing two values, filesystem checks)
    # or would report without naming the key.
    blocks = ["Initial"]
    if data["Calculator"]["method"] == "overlaps":
        blocks.append("Final")
        if data["Initial"]["type"] != data["Final"]["type"]:
            problems.append(
                f"Final.type: {data['Final']['type']!r} must match "
                f"Initial.type {data['Initial']['type']!r}"
            )

    for block in blocks:
        if "mcscf_file" in data[block] and data[block]["type"] == "rasscf":
            problems.append(
                f"{block}.mcscf_file: only used for type 'caspt2'; a rasscf wf_file already "
                "contains the orbital energies and populations"
            )

    for block in blocks:
        for key in _FILE_KEYS:
            if key not in data[block]:
                continue
            file_path = (path.parent / data[block][key]).resolve()
            if not file_path.is_file():
                problems.append(f"{block}.{key}: file not found: {file_path}")
            data[block][key] = file_path

    if problems:
        raise InputError(path, problems)
    return data
