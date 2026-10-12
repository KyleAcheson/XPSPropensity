"""Abstract interface shared by every AO overlap provider."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Type

import numpy as np

from xpsoverlaps.model import AOBasis


class AOOverlapProvider(ABC):
    """Computes ``S[i, j] = <a_i | b_j>`` in each basis' AO (MO coefficient row) order.

    Subclasses set ``basis_type`` and implement ``_compute``; ``cross_overlap``
    validates the inputs and the result shape for every provider.
    """

    basis_type: ClassVar[Type[AOBasis]]

    def cross_overlap(self, basis_a: AOBasis, basis_b: AOBasis) -> np.ndarray:
        for basis in (basis_a, basis_b):
            if not isinstance(basis, self.basis_type):
                raise TypeError(
                    f"{type(self).__name__} requires {self.basis_type.__name__}, "
                    f"got {type(basis).__name__}."
                )
        overlap = np.asarray(self._compute(basis_a, basis_b), dtype=float)
        expected = (basis_a.n_functions, basis_b.n_functions)
        if overlap.shape != expected:
            raise ValueError(
                f"{type(self).__name__} returned an AO overlap of shape {overlap.shape}; "
                f"expected {expected}."
            )
        return overlap

    @abstractmethod
    def _compute(self, basis_a: AOBasis, basis_b: AOBasis) -> np.ndarray:
        """Return the cross AO overlap; inputs are already of ``basis_type``."""
