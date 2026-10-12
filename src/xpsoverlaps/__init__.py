"""Program-neutral electronic-structure overlap and shake-down analysis."""

import logging

from xpsoverlaps.model import (
    ActiveSpace,
    BasisSet,
    BasisShell,
    Calculation,
    Configuration,
    ElectronicState,
    Molecule,
    OrbitalSet,
    Provenance,
)
from xpsoverlaps.readers import get_calculation

__all__ = [
    "ActiveSpace",
    "BasisSet",
    "BasisShell",
    "Calculation",
    "Configuration",
    "ElectronicState",
    "Molecule",
    "OrbitalSet",
    "Provenance",
    "get_calculation",
]

# Library modules log via logging.getLogger(__name__); the CLI (main.py)
# attaches the output-file handler. Without it, nothing is emitted.
logging.getLogger(__name__).addHandler(logging.NullHandler())
