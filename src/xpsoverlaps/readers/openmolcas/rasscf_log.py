"""Reader for state CI configurations printed in OpenMolcas RASSCF logs."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, List, Optional

from xpsoverlaps.model import Configuration, ElectronicState


_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?"
_ROOT_ENERGY = re.compile(
    rf"::\s*RASSCF root number\s+(\d+)\s+Total energy:\s*({_NUMBER})",
    re.IGNORECASE,
)
_CI_ROOT_BLOCK = re.compile(
    r"printout of CI-coefficients larger than\s+\S+\s+for root\s+(\d+)"
    r"(.*?)(?=printout of CI-coefficients larger than|"
    r"Final state energy\(ies\):|\Z)",
    re.IGNORECASE | re.DOTALL,
)
# With symmetry Molcas prints a space at every irrep/RAS boundary of the
# occupation string ("22 22 2 2 22u0 2d0 2 20"); the spaces are removed.
_OCCUPATION = r"(?:[20ud]+ )*[20ud]+"
_CI_CONFIGURATION = re.compile(
    rf"^\s*(\d+)\s+({_OCCUPATION})\s+({_NUMBER})\s+({_NUMBER})\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _as_float(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


def _first_float(pattern: str, text: str) -> Optional[float]:
    match = re.search(pattern, text, re.IGNORECASE)
    return _as_float(match.group(1)) if match else None


def read_rasscf_states(source: str | Path) -> List[ElectronicState]:
    """RASSCF roots from a Molcas log: energies and each root's dominant printed CSF."""
    path = Path(source)
    text = path.read_text(encoding="utf-8", errors="replace")
    energies: Dict[int, float] = {
        int(root): _as_float(energy) for root, energy in _ROOT_ENERGY.findall(text)
    }
    spin = _first_float(r"Spin quantum number\s+(" + _NUMBER + r")", text)
    symmetry_match = re.search(r"State symmetry\s+(\S+)", text, re.IGNORECASE)
    symmetry = symmetry_match.group(1) if symmetry_match else None
    dominant_configurations: Dict[int, Configuration] = {}

    for root_match in _CI_ROOT_BLOCK.finditer(text):
        root = int(root_match.group(1))
        configurations = []
        for match in _CI_CONFIGURATION.finditer(root_match.group(2)):
            configuration_id, occupation, coefficient, weight = match.groups()
            configurations.append(
                Configuration(
                    configuration_id=configuration_id,
                    occupations=list(occupation.replace(" ", "").lower()),
                    coefficient=_as_float(coefficient),
                    weight=_as_float(weight),
                )
            )
        if configurations:
            dominant_configurations[root] = max(
                configurations,
                key=lambda configuration: (
                    configuration.weight
                    if configuration.weight is not None
                    else float("-inf")
                ),
            )

    root_numbers = sorted(set(energies) | set(dominant_configurations))
    if not root_numbers:
        raise ValueError(f"No RASSCF root energies or CI configurations found in {path}.")

    return [
        ElectronicState(
            state_id=f"rasscf:{root}",
            energy=energies.get(root),
            spin=spin,
            symmetry=symmetry,
            dominant_configuration=dominant_configurations.get(root),
        )
        for root in root_numbers
    ]
