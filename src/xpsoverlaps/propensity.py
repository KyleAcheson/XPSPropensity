"""Propensity-rule approximations to Dyson norms for XPS transitions."""

from __future__ import annotations

import logging
from typing import Optional

from xpsoverlaps.composition import describe_orbital_pairs, orbital_compositions
from xpsoverlaps.mapping import MappingOptions, OrbitalMapping, format_mapping, match_active_orbitals
from xpsoverlaps.model import Calculation
from xpsoverlaps.overlap import get_overlap_provider, mo_overlap

logger = logging.getLogger(__name__)

_DASHES = "-" * 80


def map_active_orbitals(
    initial: Calculation, final: Calculation, options: Optional[MappingOptions] = None
) -> OrbitalMapping:
    """Map the initial active orbitals onto the final ones and log the result.

    Uses state-averaged orbitals (one orbital set per calculation). Nothing is
    permuted; the returned mapping only records the correspondence.
    """
    provider = get_overlap_provider(initial.basis, final.basis)
    overlap = mo_overlap(initial, final, provider)
    initial_compositions = orbital_compositions(initial, provider=provider)
    final_compositions = orbital_compositions(final, provider=provider)

    mapping = match_active_orbitals(
        overlap,
        initial.orbitals.orbital_ids,
        final.orbitals.orbital_ids,
        initial.active_space.orbital_ids,
        final.active_space.orbital_ids,
        initial_compositions,
        final_compositions,
        options,
    )

    logger.info("")
    logger.info("Active orbital mapping (initial -> final), by MO overlap")
    logger.info("")
    for line in format_mapping(mapping):
        logger.info(line)
    logger.info("")
    if mapping.warnings:
        logger.info("Mapping warnings:")
        for warning in mapping.warnings:
            logger.warning("  - %s", warning)
    else:
        logger.info("No mapping warnings.")
    logger.info("")
    logger.info("Dominant AO contributions (Mulliken populations) of the paired orbitals")
    logger.info("")
    pairs = [
        (initial_compositions[match.initial_id], final_compositions[match.final_id])
        for match in mapping.matches
    ]
    for line in describe_orbital_pairs(pairs):
        logger.info(line)
    logger.info("")
    logger.info(_DASHES)
    return mapping


def propensity_overlaps(initial: Calculation, final: Calculation) -> None:
    """Approximate Dyson norms from initial/final state orbital overlaps."""
    map_active_orbitals(initial, final)


def propensity_chp(initial: Calculation) -> None:
    """Approximate Dyson norms from the initial state using a core-hole potential."""
    pass
