"""Readers translate program-specific files into normalized model records."""

from xpsoverlaps.readers.loader import CalculationLoader, IntegralFileError, get_active_integrals, get_calculation

# Importing each program's package defines (and so registers) its CalculationLoader.
# Supporting a new program needs only its import here.
from xpsoverlaps.readers import openmolcas  # noqa: F401  (registers OpenMolcasLoader)

__all__ = ["CalculationLoader", "IntegralFileError", "get_active_integrals", "get_calculation"]
