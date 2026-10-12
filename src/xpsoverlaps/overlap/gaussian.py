"""AO overlaps for bases given explicitly as contracted Gaussian shells."""

from __future__ import annotations

import numpy as np

from xpsoverlaps.model import GaussianBasis, GaussianShell
from xpsoverlaps.overlap.base import AOOverlapProvider


def shell_overlap(shell_a: GaussianShell, shell_b: GaussianShell) -> np.ndarray:
    """Overlap block between the contracted functions of two shells.

    Rows and columns follow each shell's ``basis_function_ids`` order
    (angular component outer, radial contraction inner).
    """
    a, b = shell_a.coordinates, shell_b.coordinates
    distance2 = float(np.dot(a - b, a - b))
    alpha, beta = np.meshgrid(shell_a.exponents, shell_b.exponents, indexing="ij", sparse=True)
    product = alpha * beta
    total = alpha + beta

    # Normalized primitive s-type overlap, scaled for the shell angular momenta.
    primitive = np.exp(-distance2 * product / total) if distance2 > 0.0 else np.ones_like(product)
    primitive = primitive * np.power(4 * product / total**2, 0.75)
    l_a, l_b = shell_a.angular_momentum, shell_b.angular_momentum
    primitive = primitive * np.power(2 * alpha, l_a / 2) * np.power(2 * beta, l_b / 2)

    # Expand to every pair of Cartesian primitive powers.
    powers_a, powers_b = shell_a.primitive_powers, shell_b.primitive_powers
    integrals = np.tile(primitive, (len(powers_a), len(powers_b), 1, 1))

    # Cartesian factors per axis from the Obara-Saika-style recurrence.
    a_axes = a[:, np.newaxis, np.newaxis]
    b_axes = b[:, np.newaxis, np.newaxis]
    centre = (alpha * a_axes + beta * b_axes) / total
    max_l = l_a + l_b
    for axis in range(3):
        factors = {(0, 0): 1.0}
        if max_l > 0:
            factors[(1, 0)] = centre[axis] - a_axes[axis]
            factors[(0, 1)] = centre[axis] - b_axes[axis]
            for ax in range(2, max_l + 1):
                factors[(ax, 0)] = (
                    factors[(1, 0)] * factors[(ax - 1, 0)]
                    + (ax - 1) / (2 * total) * factors[(ax - 2, 0)]
                )
                for bx in range(1, min(ax, l_b) + 1):
                    factors[(ax - bx, bx)] = (
                        factors[(ax - bx + 1, bx - 1)]
                        + (a_axes[axis] - b_axes[axis]) * factors[(ax - bx, bx - 1)]
                    )
        for i, power_a in enumerate(powers_a):
            for j, power_b in enumerate(powers_b):
                integrals[i, j] *= factors[(power_a[axis], power_b[axis])]

    # Contract primitives and Cartesian powers into the shell's basis functions.
    block = np.einsum(
        "ik,ma,jl,nb,abkl->minj",
        shell_a.contraction_coefficients,
        shell_a.angular_coefficients,
        shell_b.contraction_coefficients,
        shell_b.angular_coefficients,
        integrals,
    )
    return block.reshape(len(shell_a.basis_function_ids), len(shell_b.basis_function_ids))


def basis_overlap(basis_a: GaussianBasis, basis_b: GaussianBasis) -> np.ndarray:
    """AO overlap matrix ``S[i, j] = <a_i | b_j>`` in each basis' function order.

    The two bases may differ (e.g. different geometries or basis sets);
    passing the same basis twice gives its ordinary AO overlap matrix.
    """
    overlap = np.zeros((basis_a.n_functions, basis_b.n_functions))
    for shell_a in basis_a.shells:
        rows = shell_a.basis_function_ids
        for shell_b in basis_b.shells:
            overlap[np.ix_(rows, shell_b.basis_function_ids)] = shell_overlap(shell_a, shell_b)
    return overlap


class GaussianOverlapProvider(AOOverlapProvider):
    """Program-independent AO overlaps for ``GaussianBasis`` representations."""

    basis_type = GaussianBasis

    def _compute(self, basis_a: GaussianBasis, basis_b: GaussianBasis) -> np.ndarray:
        return basis_overlap(basis_a, basis_b)
