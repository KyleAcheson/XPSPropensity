"""Reconstruction of OpenMolcas Gaussian basis shells from HDF5 fields."""

from __future__ import annotations

from fractions import Fraction
import itertools
import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from xpsoverlaps.model import BasisShell


def _binomial(n: int, k: int) -> Fraction:
    if k < 0 or k > n:
        return Fraction(0, 1)
    return Fraction(math.factorial(n), math.factorial(k) * math.factorial(n - k))


def _cartesian_spherical_coefficient(
    angular_momentum: int, magnetic_number: int, lx: int, ly: int, lz: int
) -> Fraction:
    absolute_m = abs(magnetic_number)
    j_value = lx + ly - absolute_m
    if lx + ly + lz != angular_momentum or min(lx, ly, lz) < 0:
        raise ValueError("Invalid Cartesian angular powers.")
    if absolute_m > angular_momentum or j_value % 2:
        return Fraction(0, 1)
    j_value //= 2

    spherical_sum = Fraction(0, 1)
    for i in range((angular_momentum - absolute_m) // 2 + 1):
        term = _binomial(angular_momentum, i) * _binomial(i, j_value)
        term *= Fraction(
            math.factorial(2 * angular_momentum - 2 * i),
            math.factorial(angular_momentum - absolute_m - 2 * i),
        )
        spherical_sum += term * (-1) ** i
    if not spherical_sum:
        return Fraction(0, 1)

    complex_sum = 0j
    for k in range(j_value + 1):
        complex_sum += (
            _binomial(j_value, k)
            * _binomial(absolute_m, lx - 2 * k)
            * 1j ** (absolute_m - lx + 2 * k)
        )
    angular_factor = int(complex_sum.real if magnetic_number >= 0 else complex_sum.imag)
    if angular_factor == 0:
        return Fraction(0, 1)
    signed_square = spherical_sum * angular_factor
    signed_square = abs(signed_square) * signed_square

    multiplicity = 1 if magnetic_number == 0 else 2
    normalization = Fraction(math.factorial(angular_momentum - absolute_m),
                             math.factorial(angular_momentum + absolute_m))
    normalization *= Fraction(multiplicity, math.factorial(angular_momentum))
    normalization *= Fraction(1, math.factorial(2 * angular_momentum))
    return signed_square * normalization


def _spherical_to_cartesian(l_max: int) -> List[Dict[int, List[Tuple[float, Tuple[int, int, int]]]]]:
    result = []
    for angular_momentum in range(l_max + 1):
        magnetic_components = {}
        for magnetic_number in range(-angular_momentum, angular_momentum + 1):
            components = []
            for lx in range(angular_momentum + 1):
                for ly in range(angular_momentum - lx + 1):
                    lz = angular_momentum - lx - ly
                    coefficient = float(
                        _cartesian_spherical_coefficient(
                            angular_momentum, magnetic_number, lx, ly, lz
                        )
                    )
                    if coefficient:
                        components.append(
                            (math.copysign(math.sqrt(abs(coefficient)), coefficient), (lx, ly, lz))
                        )
            magnetic_components[magnetic_number] = components
        result.append(magnetic_components)
    return result


def _r2_terms(power: int) -> Mapping[Tuple[int, int, int], int]:
    if power == 0:
        return {(0, 0, 0): 1}
    terms: Dict[Tuple[int, int, int], int] = {}
    choices = ((2, 0, 0), (0, 2, 0), (0, 0, 2))
    for selection in itertools.product(choices, repeat=power):
        powers = tuple(sum(component[axis] for component in selection) for axis in range(3))
        terms[powers] = terms.get(powers, 0) + 1
    return terms


def _label_contaminants(
    basis_ids: np.ndarray, center_labels: Sequence[object]
) -> Dict[int, int]:
    contaminants: Dict[int, int] = {}
    for index, basis_id in enumerate(basis_ids):
        if basis_id[2] < 0 or basis_id[1] != 1:
            continue
        if any(np.array_equal(basis_id, previous) for previous in basis_ids[:index]):
            contaminants[index] = 0

    equivalent_centers = []
    for index, label in enumerate(center_labels):
        text = _decode(label)
        key = text.split(":", 1)[0]
        equivalent_centers.append(
            next(
                (previous + 1 for previous in range(index + 1)
                 if _decode(center_labels[previous]).split(":", 1)[0] == key),
                index + 1,
            )
        )

    for index in tuple(contaminants):
        basis_id = basis_ids[index]
        for parent_index in range(index - 1, -1, -1):
            parent_id = basis_ids[parent_index]
            if parent_index in contaminants or parent_id[2] < 0:
                continue
            if (
                basis_id[0] == parent_id[0]
                and basis_id[1] == parent_id[1]
                and parent_id[2] > basis_id[2]
                and (parent_id[2] - basis_id[2]) % 2 == 0
            ):
                contaminants[index] = (parent_id[2] - basis_id[2]) // 2
                break
        if contaminants[index] <= 0:
            raise ValueError(f"Could not identify contaminant basis function {index}.")

        for later_index in range(index + 1, len(basis_ids)):
            later_id = basis_ids[later_index]
            if (
                equivalent_centers[int(later_id[0]) - 1]
                != equivalent_centers[int(basis_id[0]) - 1]
                or later_id[1] < basis_id[1]
                or later_id[2] != basis_id[2]
                or later_id[3] != basis_id[3]
            ):
                break
            if later_index not in contaminants and later_id[2] >= 0:
                contaminants[later_index] = contaminants[index]
    return contaminants


def _normalize_radial_coefficients(exponents: np.ndarray, coefficients: np.ndarray, l: int) -> None:
    integrals = np.sum(coefficients**2, axis=1)
    for i in range(len(exponents)):
        for j in range(i + 1, len(exponents)):
            primitive_overlap = (
                2 * math.sqrt(exponents[i] * exponents[j]) / (exponents[i] + exponents[j])
            ) ** (1.5 + l)
            integrals += 2 * coefficients[:, i] * coefficients[:, j] * primitive_overlap
    if np.any(integrals <= 0):
        raise ValueError("Molcas basis contains a non-normalizable contraction.")
    coefficients /= np.sqrt(integrals[:, np.newaxis])


def _decode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("ascii").strip()
    return str(value).strip()


def reconstruct_basis(
    *,
    centers: np.ndarray,
    center_labels: Sequence[object],
    basis_ids: np.ndarray,
    primitive_data: np.ndarray,
    primitive_ids: np.ndarray,
) -> Tuple[Tuple[BasisShell, ...], Tuple[str, ...]]:
    """Normalize Molcas shell records and preserve its AO function ordering."""
    raw_ids = np.asarray(basis_ids, dtype=np.int64)
    raw_primitives = np.asarray(primitive_data, dtype=float)
    raw_primitive_ids = np.asarray(primitive_ids, dtype=np.int64)
    center_array = np.asarray(centers, dtype=float)
    if raw_ids.ndim != 2 or raw_ids.shape[1] != 4:
        raise ValueError("BASIS_FUNCTION_IDS must have four integer columns.")
    if raw_primitives.ndim != 2 or raw_primitives.shape[1] != 2:
        raise ValueError("PRIMITIVES must contain exponent/coefficient pairs.")
    if raw_primitive_ids.shape != (len(raw_primitives), 3):
        raise ValueError("PRIMITIVE_IDS must have three columns and match PRIMITIVES.")
    if center_array.shape != (len(center_labels), 3):
        raise ValueError("Center coordinates and labels have incompatible dimensions.")

    contaminants = _label_contaminants(raw_ids, center_labels)
    shell_list: List[dict] = []
    shell_index = {}
    for index, basis_id in enumerate(raw_ids):
        center = int(basis_id[0])
        if center < 1 or center > len(center_array):
            raise ValueError(f"Basis function {index} references invalid center {center}.")
        angular_momentum = abs(int(basis_id[2]))
        r2_power = contaminants.get(index, 0)
        angular_momentum += 2 * r2_power
        label = (center, int(basis_id[1]), angular_momentum, r2_power)
        magnetic_number = int(basis_id[3])
        if label in shell_index:
            shell = shell_list[shell_index[label]]
            if magnetic_number in shell["magnetic_numbers"]:
                raise ValueError("Duplicate angular component in Molcas basis shell.")
            shell["magnetic_numbers"].append(magnetic_number)
            shell["basis_function_ids"].append(index)
        else:
            shell_index[label] = len(shell_list)
            shell_list.append(
                {
                    "center": center,
                    "center_label": _decode(center_labels[center - 1]),
                    "coordinates": center_array[center - 1].copy(),
                    "l": angular_momentum,
                    "cartesian": int(basis_id[2]) < 0,
                    "r2": r2_power,
                    "magnetic_numbers": [magnetic_number],
                    "basis_function_ids": [index],
                    "exponents": [],
                    "coefficient_rows": [[]],
                }
            )

    for primitive_id, primitive in zip(raw_primitive_ids, raw_primitives):
        center, angular_momentum, shell_number = (int(value) for value in primitive_id)
        for r2_power in range(angular_momentum // 2, -1, -1):
            key = (center, shell_number, angular_momentum, r2_power)
            if key in shell_index:
                shell = shell_list[shell_index[key]]
                shell["exponents"].append(float(primitive[0]))
                shell["coefficient_rows"][0].append(float(primitive[1]))

    for shell in shell_list:
        n_magnetic = len(shell["magnetic_numbers"])
        l = shell["l"]
        r2 = shell["r2"]
        expected_spherical = 2 * (l - 2 * r2) + 1
        expected_cartesian = (l + 1) * (l + 2) // 2
        if (r2 and shell["cartesian"]) or not (
            (n_magnetic == expected_spherical and not shell["cartesian"])
            or (n_magnetic == expected_cartesian and shell["cartesian"])
        ):
            raise ValueError("Inconsistent angular components in Molcas basis shell.")

    for i, shell in enumerate(shell_list):
        if shell["basis_function_ids"] is None:
            continue
        for later in shell_list[i + 1 :]:
            if (
                later["basis_function_ids"] is not None
                and np.array_equal(shell["coordinates"], later["coordinates"])
                and all(shell[key] == later[key] for key in ("l", "cartesian", "r2", "magnetic_numbers", "exponents"))
            ):
                shell["coefficient_rows"].append(later["coefficient_rows"][0])
                shell["basis_function_ids"].extend(later["basis_function_ids"])
                later["basis_function_ids"] = None

    if not shell_list:
        raise ValueError("Molcas basis contains no shells.")
    l_max = max(shell["l"] for shell in shell_list)
    spherical_coefficients = _spherical_to_cartesian(l_max)

    for shell in shell_list:
        if shell["basis_function_ids"] is None:
            continue
        magnetic_numbers = shell["magnetic_numbers"]
        coefficients = np.asarray(shell["coefficient_rows"], dtype=float)
        exponents = np.asarray(shell["exponents"], dtype=float)
        if not len(exponents) or coefficients.shape[1] != len(exponents):
            raise ValueError("Could not associate Molcas primitives with a basis shell.")
        ids = np.asarray(shell["basis_function_ids"], dtype=np.int64)
        shell["basis_function_ids"] = ids.reshape(
            len(coefficients), len(magnetic_numbers)
        ).flatten(order="F").tolist()
        _normalize_radial_coefficients(exponents, coefficients, shell["l"])

        l, r2 = shell["l"], shell["r2"]
        powers = [
            (lx, ly, l - lx - ly)
            for lx in range(l, -1, -1)
            for ly in range(l - lx, -1, -1)
        ]
        angular_coefficients = np.zeros((len(magnetic_numbers), len(powers)))
        if shell["cartesian"]:
            for row, magnetic_number in enumerate(magnetic_numbers):
                angular_coefficients[row, magnetic_number + l] = math.sqrt(2**l)
        else:
            for row, magnetic_number in enumerate(magnetic_numbers):
                for coefficient, base_powers in spherical_coefficients[l - 2 * r2][magnetic_number]:
                    for extra_powers, multiplicity in _r2_terms(r2).items():
                        combined = tuple(base_powers[i] + extra_powers[i] for i in range(3))
                        position = (combined[1] + combined[2]) * (combined[1] + combined[2] + 1) // 2 + combined[2]
                        if powers[position] != combined:
                            raise ValueError("Internal Cartesian power ordering mismatch.")
                        angular_coefficients[row, position] += coefficient * multiplicity
            if r2:
                normalization = math.prod((2 * order + 1) / 2 for order in range(l - 2 * r2 + 1, l + 1))
                angular_coefficients /= math.sqrt(normalization)

        shell.update(
            exponents=exponents,
            coefficient_rows=coefficients,
            angular_coefficients=angular_coefficients,
            primitive_powers=np.asarray(powers, dtype=np.int64),
        )

    ao_labels: List[Optional[str]] = [None] * len(raw_ids)
    angular_labels = {
        (2, 0, 0): "Dx2", (0, 2, 0): "Dy2", (0, 0, 2): "Dz2",
        (1, 1, 0): "Dxy", (1, 0, 1): "Dxz", (0, 1, 1): "Dyz",
    }
    l_labels = {0: "S", 1: "P", 2: "D", 3: "F", 4: "G", 5: "H"}
    final_shells = []
    for shell in shell_list:
        ids = shell["basis_function_ids"]
        if ids is None:
            continue
        l = shell["l"]
        for radial in range(shell["coefficient_rows"].shape[0]):
            principal = radial + l + 1
            for angular in range(len(shell["angular_coefficients"])):
                if l == 0:
                    orbital = f"{principal}S"
                elif l == 1:
                    if shell["cartesian"]:
                        powers = shell["primitive_powers"][angular]
                        axis = "x" if powers[0] else ("y" if powers[1] else "z")
                        orbital = f"{principal}P{axis}"
                    else:
                        orbital = f"{principal}P[m={shell['magnetic_numbers'][angular]:+d}]"
                elif l == 2 and shell["cartesian"]:
                    powers = shell["primitive_powers"][angular]
                    orbital = f"{principal}{angular_labels.get(tuple(int(x) for x in powers), 'D')}"
                else:
                    orbital = (
                        f"{principal}{l_labels.get(l, '?')}"
                        f"[m={shell['magnetic_numbers'][angular]:+d}]"
                    )
                # Shell function IDs are angular-major (see the order="F" flatten above).
                n_radial = shell["coefficient_rows"].shape[0]
                ao_labels[ids[angular * n_radial + radial]] = (
                    f"{shell['center_label']} {orbital}"
                )
        final_shells.append(
            BasisShell(
                center_index=shell["center"] - 1,
                center_label=shell["center_label"],
                coordinates=shell["coordinates"],
                angular_momentum=l,
                cartesian=shell["cartesian"],
                r2_power=shell["r2"],
                exponents=shell["exponents"],
                contraction_coefficients=shell["coefficient_rows"],
                angular_coefficients=shell["angular_coefficients"],
                primitive_powers=shell["primitive_powers"],
                magnetic_numbers=list(shell["magnetic_numbers"]),
                basis_function_ids=list(ids),
            )
        )

    if any(label is None for label in ao_labels):
        raise ValueError("Molcas shell reconstruction did not cover every AO index.")
    function_ids = sorted(
        function_id
        for shell in final_shells
        for function_id in shell.basis_function_ids
    )
    if function_ids != list(range(len(raw_ids))):
        raise ValueError("Molcas basis shell records do not cover each AO index exactly once.")
    return tuple(final_shells), tuple(label for label in ao_labels if label is not None)


def square_from_symmetry_blocks(data: np.ndarray, block_sizes: Sequence[int]) -> np.ndarray:
    """Expand Molcas' flattened, symmetry-block-diagonal matrix representation."""
    values = np.asarray(data)
    blocks = tuple(int(size) for size in np.asarray(block_sizes).reshape(-1))
    total = sum(blocks)
    if not blocks or any(size <= 0 for size in blocks):
        raise ValueError("NBAS must contain positive symmetry block sizes.")
    if len(blocks) == 1:
        if values.size != total * total:
            raise ValueError(f"Expected {total * total} matrix values; found {values.size}.")
        return values.reshape(total, total)
    if values.size != sum(size * size for size in blocks):
        raise ValueError("Flattened matrix size does not match NBAS symmetry blocks.")
    matrix = np.zeros((total, total), dtype=values.dtype)
    source_offset = target_offset = 0
    for size in blocks:
        matrix[target_offset : target_offset + size, target_offset : target_offset + size] = (
            values[source_offset : source_offset + size * size].reshape(size, size)
        )
        source_offset += size * size
        target_offset += size
    return matrix
