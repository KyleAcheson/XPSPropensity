"""Reader for OpenMolcas CASPT2/RASSCF text output and mixed-state CI tables."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, List, Optional

from xpsoverlaps.model import Configuration, ElectronicState


_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?"
_MIXED_STATE = re.compile(
    r"The CI coefficients for the MIXED state nr\.\s*(\d+)(.*?)(?="
    r"The CI coefficients for the MIXED state nr\.|\Z)",
    re.DOTALL,
)
_CONFIGURATION = re.compile(
    rf"^\s*(\d+)\s+\((.*?)\)\s+([20ud]+)\s+({_NUMBER})\s+({_NUMBER})\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_CASPT2_ENERGY = re.compile(
    rf"::\s*MS-CASPT2 Root\s+(\d+)\s+Total energy:\s*({_NUMBER})",
    re.IGNORECASE,
)
_RASSCF_ENERGY = re.compile(
    rf"::\s*RASSCF root number\s+(\d+)\s+Total energy:\s*({_NUMBER})",
    re.IGNORECASE,
)


def _as_float(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


def _first_float(pattern: str, text: str) -> Optional[float]:
    match = re.search(pattern, text, re.IGNORECASE)
    return _as_float(match.group(1)) if match else None


def read_caspt2_states(source: str | Path) -> List[ElectronicState]:
    """MS-CASPT2 states from a Molcas log: energies and each state's dominant mixed-state CSF."""
    path = Path(source)
    text = path.read_text(encoding="utf-8", errors="replace")
    caspt2_energies = {
        int(root): _as_float(energy) for root, energy in _CASPT2_ENERGY.findall(text)
    }
    rasscf_energies = {
        int(root): _as_float(energy) for root, energy in _RASSCF_ENERGY.findall(text)
    }
    energies = caspt2_energies or rasscf_energies
    mixed_blocks = list(_MIXED_STATE.finditer(text))
    spin = _first_float(r"Spin quantum number\s+(" + _NUMBER + r")", text)
    symmetry_match = re.search(r"State symmetry\s+(\S+)", text, re.IGNORECASE)
    symmetry = symmetry_match.group(1) if symmetry_match else None
    dominant_configurations: Dict[int, Configuration] = {}
    for state_match in mixed_blocks:
        state_number = int(state_match.group(1))
        configurations = []
        for match in _CONFIGURATION.finditer(state_match.group(2)):
            conf_id, sgu_info, occupation, coefficient, weight = match.groups()
            configurations.append(
                Configuration(
                    configuration_id=conf_id,
                    occupations=list(occupation.lower()),
                    coefficient=_as_float(coefficient),
                    weight=_as_float(weight),
                    source_metadata={"sguga_info": sgu_info.strip()},
                )
            )
        if configurations:
            dominant_configurations[state_number] = max(
                configurations, key=lambda configuration: configuration.weight or 0
            )

    state_numbers = sorted(set(energies) | set(dominant_configurations))
    if not state_numbers:
        raise ValueError(f"No RASSCF roots or mixed-state CI data found in {path}.")
    states = []
    for state_number in state_numbers:
        states.append(
            ElectronicState(
                state_id=(
                    f"caspt2:{state_number}"
                    if state_number in dominant_configurations or caspt2_energies
                    else f"rasscf:{state_number}"
                ),
                energy=energies.get(state_number),
                spin=spin,
                symmetry=symmetry,
                dominant_configuration=dominant_configurations.get(state_number),
            )
        )
    return states
