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
    # Required by the schema when method is "overlaps". Always a list: a single
    # block in the file is returned as a one-element list.
    Final: List[CalculationInput]


class InputError(ValueError):
    """Raised when an input file violates the schema or a cross-field rule."""

    def __init__(self, source: Path, problems: List[str]) -> None:
        self.problems = problems
        details = "\n".join(f"  - {problem}" for problem in problems)
        super().__init__(f"Invalid input file {source}:\n{details}")


def load_schema() -> Dict[str, Any]:
    with SCHEMA_PATH.open(encoding="utf-8") as schema_file:
        return yaml.safe_load(schema_file)


def _final_label(index: int, single: bool) -> str:
    """How a Final block is named in messages: "Final" if it is the only one, else "Final[n]" (1-based)."""
    return "Final" if single else f"Final[{index + 1}]"


def _location(path: Any, single_final: bool) -> str:
    parts = list(path)
    if len(parts) >= 2 and parts[0] == "Final" and isinstance(parts[1], int):
        parts = [_final_label(parts[1], single_final)] + parts[2:]
    return ".".join(str(part) for part in parts) or "<root>"


def _chain(schema: Dict[str, Any], root: Dict[str, Any]) -> List[Dict[str, Any]]:
    """A schema followed by the schemas its local ``$ref`` chain points to."""
    chain = [schema]
    while "$ref" in chain[-1]:
        target = root
        for part in chain[-1]["$ref"].lstrip("#/").split("/"):
            target = target[part]
        chain.append(target)
    return chain


def _lowercase_choices(data: Any, schema: Dict[str, Any], root: Dict[str, Any]) -> None:
    """Lowercase values of properties restricted to a string ``enum``.

    This makes choice fields (program, method, type, ...) case-insensitive;
    the schema's enum values must therefore be written in lowercase. Free-form
    strings such as file paths are left unchanged.
    """
    chain = _chain(schema, root)
    if isinstance(data, list):
        items = next((s["items"] for s in chain if "items" in s), None)
        for element in data if items is not None else []:
            _lowercase_choices(element, items, root)
        return
    if not isinstance(data, dict):
        return
    properties: Dict[str, Any] = {}
    for linked in chain:
        for key, property_schema in linked.get("properties", {}).items():
            properties.setdefault(key, property_schema)
    for key, property_schema in properties.items():
        if key not in data:
            continue
        choices = next((s["enum"] for s in _chain(property_schema, root) if "enum" in s), None)
        if isinstance(data[key], str) and choices and all(isinstance(c, str) for c in choices):
            data[key] = data[key].strip().lower()
        else:
            _lowercase_choices(data[key], property_schema, root)


def load_input(source: str | Path) -> RunInput:
    """Read and validate an input file.

    Relative file paths (``wf_file``, ``state_file``, ``mcscf_file``) are
    resolved against the input file's directory and returned as absolute
    ``Path`` objects. ``Final`` is returned as a list of blocks, also when the
    file gives a single block. All problems are reported together in a single
    ``InputError``.
    """
    path = Path(source).resolve()
    with path.open(encoding="utf-8") as input_file:
        data = yaml.safe_load(input_file)
    if data is None:
        raise InputError(path, ["file is empty"])

    single_final = isinstance(data, dict) and isinstance(data.get("Final"), dict)
    if single_final:
        data["Final"] = [data["Final"]]

    schema = load_schema()
    _lowercase_choices(data, schema, schema)
    validator = jsonschema.Draft202012Validator(schema)
    problems = [
        f"{_location(error.absolute_path, single_final)}: {error.message}"
        for error in sorted(validator.iter_errors(data), key=lambda error: [str(p) for p in error.absolute_path])
    ]
    if problems:
        raise InputError(path, problems)

    # Rules JSON Schema cannot express (comparing two values, filesystem checks)
    # or would report without naming the key.
    blocks = [("Initial", data["Initial"])]
    if data["Calculator"]["method"] == "overlaps":
        for index, final in enumerate(data["Final"]):
            label = _final_label(index, single_final)
            blocks.append((label, final))
            if data["Initial"]["type"] != final["type"]:
                problems.append(
                    f"{label}.type: {final['type']!r} must match Initial.type {data['Initial']['type']!r}"
                )

    for label, block in blocks:
        if "mcscf_file" in block and block["type"] == "rasscf":
            problems.append(
                f"{label}.mcscf_file: only used for type 'caspt2'; a rasscf wf_file already "
                "contains the orbital energies and populations"
            )

    for label, block in blocks:
        for key in _FILE_KEYS:
            if key not in block:
                continue
            file_path = (path.parent / block[key]).resolve()
            if not file_path.is_file():
                problems.append(f"{label}.{key}: file not found: {file_path}")
            block[key] = file_path

    if problems:
        raise InputError(path, problems)
    return data
