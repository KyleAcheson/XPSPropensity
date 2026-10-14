from pathlib import Path

import h5py
import numpy as np
import pytest

from xpsoverlaps.model import EngineBasis, GaussianBasis, GaussianShell
from xpsoverlaps.overlap import (
    AOOverlapProvider,
    EngineOverlapProvider,
    GaussianOverlapProvider,
    basis_overlap,
    get_overlap_provider,
    mo_overlap,
    register_engine,
)
from xpsoverlaps.overlap import engine as engine_module
from xpsoverlaps.readers.openmolcas.basis import square_from_symmetry_blocks
from xpsoverlaps.readers import get_calculation


DATA = Path(__file__).parents[1] / "data" / "molcas" / "brthph" / "s2p" / "mb"
# (calculation type, HDF5 file, matching log)
HDF5_FILES = [
    ("RASSCF", "valence.rasscf.h5", "valence-tz.log"),
    ("RASSCF", "core.rasscf.h5", "core-tz.log"),
    ("CASPT2", "valence_caspt2.caspt2.h5", "valence_caspt2.log"),
    ("CASPT2", "core_caspt2.caspt2.h5", "core_caspt2.log"),
]


def _molcas_ao_overlap(path):
    with h5py.File(path, "r") as h5_file:
        return square_from_symmetry_blocks(h5_file["AO_OVERLAP_MATRIX"][:], h5_file.attrs["NBAS"])


@pytest.mark.parametrize("kind,hdf5_name,log_name", HDF5_FILES)
def test_basis_overlap_matches_molcas_precomputed_ao_overlap(kind, hdf5_name, log_name):
    calculation = get_calculation("molcas", kind, DATA / hdf5_name, DATA / log_name)

    overlap = basis_overlap(calculation.basis, calculation.basis)

    assert overlap.shape == (239, 239)
    np.testing.assert_allclose(overlap, _molcas_ao_overlap(DATA / hdf5_name), rtol=0, atol=1e-10)


@pytest.mark.parametrize("kind,hdf5_name,log_name", HDF5_FILES)
def test_molecular_orbitals_are_orthonormal_with_selected_provider(kind, hdf5_name, log_name):
    calculation = get_calculation("molcas", kind, DATA / hdf5_name, DATA / log_name)

    assert isinstance(
        get_overlap_provider(calculation.basis, calculation.basis), GaussianOverlapProvider
    )
    np.testing.assert_allclose(
        mo_overlap(calculation, calculation),
        np.eye(len(calculation.orbitals.orbital_ids)),
        rtol=0,
        atol=1e-7,
    )


def test_cross_basis_overlap_between_calculations():
    valence = get_calculation("molcas", "RASSCF", DATA / "valence.rasscf.h5", DATA / "valence-tz.log")
    core = get_calculation("molcas", "RASSCF", DATA / "core.rasscf.h5", DATA / "core-tz.log")

    valence_core = basis_overlap(valence.basis, core.basis)

    np.testing.assert_allclose(basis_overlap(core.basis, valence.basis), valence_core.T, atol=1e-12)
    # Same geometry and basis set, so the cross overlap is the ordinary AO overlap.
    np.testing.assert_allclose(valence_core, _molcas_ao_overlap(DATA / "core.rasscf.h5"), atol=1e-10)


def test_mo_overlap_between_calculations_uses_both_coefficient_sets():
    valence = get_calculation("molcas", "RASSCF", DATA / "valence.rasscf.h5", DATA / "valence-tz.log")
    core = get_calculation("molcas", "RASSCF", DATA / "core.rasscf.h5", DATA / "core-tz.log")

    expected = (
        valence.orbitals.coefficients.T
        @ basis_overlap(valence.basis, core.basis)
        @ core.orbitals.coefficients
    )

    np.testing.assert_allclose(mo_overlap(valence, core), expected, atol=1e-12)


def _s_function(exponent, center, function_id=0):
    return GaussianShell(
        center_index=0,
        coordinates=np.asarray(center, dtype=float),
        angular_momentum=0,
        exponents=np.array([exponent]),
        contraction_coefficients=np.array([[1.0]]),
        angular_coefficients=np.array([[1.0]]),
        primitive_powers=np.array([[0, 0, 0]]),
        basis_function_ids=[function_id],
    )


def _s_basis(basis_id, exponent, center):
    """A basis of one normalized s function on atom 0."""
    return GaussianBasis(
        basis_id,
        [f"{basis_id} 1S"],
        function_centers=[0],
        function_angular_momenta=[0],
        shells=[_s_function(exponent, center)],
    )


def test_shell_overlap_of_normalized_s_primitives_matches_analytic_result():
    alpha, beta, separation = 0.8, 1.7, 1.3
    basis_a = _s_basis("a", alpha, [0.0, 0.0, 0.0])
    basis_b = _s_basis("b", beta, [0.0, 0.0, separation])

    expected = (2 * np.sqrt(alpha * beta) / (alpha + beta)) ** 1.5 * np.exp(
        -alpha * beta / (alpha + beta) * separation**2
    )

    assert basis_overlap(basis_a, basis_b)[0, 0] == pytest.approx(expected, rel=1e-12)
    assert basis_overlap(basis_a, basis_a)[0, 0] == pytest.approx(1.0, rel=1e-12)


@pytest.fixture
def fake_engine(monkeypatch):
    """Register a stand-in engine whose handles are precomputed AO matrices."""
    monkeypatch.setattr(engine_module, "_ENGINES", {})
    register_engine("fake", lambda handle_a, handle_b: handle_a @ handle_b.T)


def test_engine_bases_use_registered_engine_backend(fake_engine):
    handle_a, handle_b = np.eye(2), np.array([[1.0, 0.5], [0.0, 1.0]])
    basis_a = EngineBasis("a", ["1", "2"], [0, 0], [0, 0], engine="fake", handle=handle_a)
    basis_b = EngineBasis("b", ["1", "2"], [0, 0], [0, 0], engine="fake", handle=handle_b)

    provider = get_overlap_provider(basis_a, basis_b)

    assert isinstance(provider, EngineOverlapProvider)
    np.testing.assert_allclose(provider.cross_overlap(basis_a, basis_b), handle_a @ handle_b.T)


def test_engine_provider_rejects_unregistered_and_mismatched_engines(fake_engine):
    fake = EngineBasis("a", ["1"], [0], [0], engine="fake", handle=np.eye(1))
    other = EngineBasis("b", ["1"], [0], [0], engine="other", handle=np.eye(1))

    with pytest.raises(NotImplementedError, match="No overlap backend registered for engine 'other'"):
        EngineOverlapProvider().cross_overlap(other, other)
    with pytest.raises(ValueError, match="between engines 'fake' and 'other'"):
        EngineOverlapProvider().cross_overlap(fake, other)


def test_provider_selection_rejects_mixed_basis_representations():
    gaussian = _s_basis("a", 1.0, [0.0, 0.0, 0.0])
    engine = EngineBasis("b", ["b 1S"], [0], [0], engine="fake", handle=None)

    with pytest.raises(ValueError, match="same basis representation"):
        get_overlap_provider(gaussian, engine)
    with pytest.raises(TypeError, match="requires GaussianBasis"):
        GaussianOverlapProvider().cross_overlap(gaussian, engine)


def test_provider_without_compute_cannot_be_instantiated():
    class IncompleteProvider(AOOverlapProvider):
        basis_type = GaussianBasis

    with pytest.raises(TypeError, match="abstract"):
        IncompleteProvider()


def test_base_class_validates_overlap_shape_for_every_provider(fake_engine):
    register_engine("fake", lambda handle_a, handle_b: np.eye(3))
    basis = EngineBasis("a", ["1", "2"], [0, 0], [0, 0], engine="fake", handle=None)

    with pytest.raises(ValueError, match=r"returned an AO overlap of shape \(3, 3\); expected \(2, 2\)"):
        EngineOverlapProvider().cross_overlap(basis, basis)


def test_base_class_validates_basis_type_for_every_provider():
    gaussian = _s_basis("a", 1.0, [0.0, 0.0, 0.0])

    with pytest.raises(TypeError, match="EngineOverlapProvider requires EngineBasis, got GaussianBasis"):
        EngineOverlapProvider().cross_overlap(gaussian, gaussian)
