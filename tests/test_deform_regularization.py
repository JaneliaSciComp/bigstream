import time

import numpy as np
import pytest
import SimpleITK as sitk

from bigstream.deform_regularization import (
    bspline_coefficients,
    c4_violations,
    chun_fessler_penalty,
    coefficient_bounds,
    is_orientation_preserving,
    project_bspline_transform,
    project_to_c4,
    set_bspline_coefficients,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

GRID = (7, 6, 5)          # coefficient grid, zyx
KNOT_SPACING = np.array([8.0, 4.0, 2.0])   # zyx, deliberately anisotropic


def _random_coefficients(scale, grid=GRID, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-scale, scale, size=(3,) + grid)


def _bspline_transform(mesh_size=(4, 3, 2), size=(64, 48, 32),
                       spacing=(1.0, 2.0, 4.0)):
    """A 3D BSplineTransform over a domain with anisotropic spacing (xyz)."""
    image = sitk.Image(list(size), sitk.sitkFloat32)
    image.SetSpacing(list(spacing))
    return sitk.BSplineTransformInitializer(
        image1=image, transformDomainMeshSize=list(mesh_size), order=3,
    )


def _jacobian_stats(transform, size, spacing, origin=None, direction=None,
                    supersample=1, interior_only=False):
    """
    min |J| and fraction of non-positive |J| over a (possibly finer) grid.

    `interior_only` crops one knot spacing from every side. ITK's
    BSplineTransform evaluates to zero displacement outside the region where
    the spline is fully supported, which puts a step discontinuity in the
    sampled field at the domain boundary and therefore spurious non-positive
    Jacobians there. That artifact is independent of the coefficients, so the
    C4 guarantee - which is about the spline itself - has to be checked inside
    the supported region.
    """
    size = np.asarray(size)
    spacing = np.asarray(spacing)
    if supersample > 1:
        size = size * supersample
        spacing = spacing / supersample
    origin = [0.0] * len(size) if origin is None else origin
    direction = np.eye(len(size)).ravel() if direction is None else direction
    field = sitk.TransformToDisplacementField(
        transform, sitk.sitkVectorFloat64,
        [int(s) for s in size], list(origin), [float(s) for s in spacing],
        [float(d) for d in direction],
    )
    jacobian = sitk.GetArrayFromImage(
        sitk.DisplacementFieldJacobianDeterminant(field))
    if interior_only:
        # knot spacing (xyz) -> voxels of this sampling grid -> zyx
        knot_spacing = np.array(transform.GetCoefficientImages()[0].GetSpacing())
        margin = np.ceil(knot_spacing / spacing[::-1]).astype(int)[::-1]
        jacobian = jacobian[tuple(slice(m, -m) for m in margin)]
    return float(jacobian.min()), float((jacobian <= 0).mean())


# ---------------------------------------------------------------------------
# 1. feasibility
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('scale', [0.5, 5.0, 50.0])
def test_projection_reaches_feasibility(scale):
    coefficients = _random_coefficients(scale)
    projected, info = project_to_c4(coefficients, KNOT_SPACING, k=0.32)

    assert info['converged']
    assert info['n_violating_after'] == 0
    assert c4_violations(projected, KNOT_SPACING, k=0.32)['n_violating'] == 0


def test_projection_reaches_feasibility_anisotropic_k():
    coefficients = _random_coefficients(20.0)
    k = [0.5, 0.25, 0.24]
    projected, info = project_to_c4(coefficients, KNOT_SPACING, k=k)

    assert info['converged']
    assert c4_violations(projected, KNOT_SPACING, k=k)['n_violating'] == 0


# ---------------------------------------------------------------------------
# 2. idempotence / dead zone
# ---------------------------------------------------------------------------

def test_projection_is_a_noop_on_compliant_coefficients():
    # tiny coefficients cannot violate bounds of order knot_spacing * k
    coefficients = _random_coefficients(0.01)
    assert c4_violations(coefficients, KNOT_SPACING)['n_violating'] == 0

    projected, info = project_to_c4(coefficients, KNOT_SPACING)

    assert info['n_violating_before'] == 0
    assert info['max_coefficient_shift'] == 0.0
    assert info['sweeps_run'] == 0
    np.testing.assert_array_equal(projected, coefficients)


def test_projection_is_idempotent():
    coefficients = _random_coefficients(10.0)
    once, _ = project_to_c4(coefficients, KNOT_SPACING)
    twice, info = project_to_c4(once, KNOT_SPACING)

    assert info['max_coefficient_shift'] == 0.0
    np.testing.assert_allclose(twice, once)


# ---------------------------------------------------------------------------
# 3. minimality: a single violating pair gets the closed-form projection
# ---------------------------------------------------------------------------

def test_single_violating_pair_matches_closed_form():
    # a (1, 1, 2) grid has exactly one adjacent pair, along x, so the
    # projection cannot be perturbed by neighbouring constraints
    coefficients = np.zeros((3, 1, 1, 2))
    lo, hi = coefficient_bounds(KNOT_SPACING, k=0.32)
    excess = 3.0
    coefficients[0, 0, 0, 1] = hi[0, 2] + excess

    projected, info = project_to_c4(coefficients, KNOT_SPACING, k=0.32)

    # both ends move toward each other by excess/2
    assert projected[0, 0, 0, 0] == pytest.approx(excess / 2)
    assert projected[0, 0, 0, 1] == pytest.approx(hi[0, 2] + excess / 2)
    # the difference lands exactly on the bound, not inside it
    assert (projected[0, 0, 0, 1] - projected[0, 0, 0, 0]) == pytest.approx(hi[0, 2])
    assert info['max_coefficient_shift'] == pytest.approx(excess / 2)
    assert info['sweeps_run'] == 2   # one to project, one to confirm no motion
    # the other two components were compliant and are untouched
    np.testing.assert_array_equal(projected[1:], np.zeros((2, 1, 1, 2)))


def test_violating_pair_below_the_lower_bound():
    coefficients = np.zeros((3, 1, 1, 2))
    lo, hi = coefficient_bounds(KNOT_SPACING, k=0.32)
    excess = 2.0
    coefficients[0, 0, 0, 1] = lo[0, 2] - excess

    projected, _ = project_to_c4(coefficients, KNOT_SPACING, k=0.32)

    assert (projected[0, 0, 0, 1] - projected[0, 0, 0, 0]) == pytest.approx(lo[0, 2])


# ---------------------------------------------------------------------------
# 4. bound convention: k indexes the component, m the difference direction
# ---------------------------------------------------------------------------

def test_bounds_use_component_k_and_difference_axis_spacing():
    k = [0.5, 0.25, 0.2]
    K = [1.0, 0.25, 0.2]
    lo, hi = coefficient_bounds(KNOT_SPACING, k=k, K=K)

    # off diagonal: component q=0 (z) differenced along r=2 (x)
    #   -> k of the component (0.5), spacing of the difference axis (2.0)
    assert hi[0, 2] == pytest.approx(0.5 * 2.0)
    assert lo[0, 2] == pytest.approx(-0.5 * 2.0)
    # off diagonal: component q=2 (x) differenced along r=0 (z)
    assert hi[2, 0] == pytest.approx(0.2 * 8.0)
    # diagonal uses K for the upper bound only
    assert hi[0, 0] == pytest.approx(1.0 * 8.0)
    assert lo[0, 0] == pytest.approx(-0.5 * 8.0)


def test_sum_k_must_be_less_than_one():
    with pytest.raises(ValueError, match='sum'):
        coefficient_bounds(KNOT_SPACING, k=[0.4, 0.4, 0.4])


def test_k_may_be_scalar_or_per_axis():
    scalar_lo, scalar_hi = coefficient_bounds(KNOT_SPACING, k=0.3)
    listed_lo, listed_hi = coefficient_bounds(KNOT_SPACING, k=[0.3, 0.3, 0.3])
    np.testing.assert_allclose(scalar_lo, listed_lo)
    np.testing.assert_allclose(scalar_hi, listed_hi)


# ---------------------------------------------------------------------------
# 5. end to end: a folding transform stops folding after projection
# ---------------------------------------------------------------------------

def test_projection_removes_folding_on_a_finer_grid():
    size, spacing = (32, 32, 32), (1.0, 1.0, 1.0)
    transform = _bspline_transform(mesh_size=(4, 4, 4), size=size, spacing=spacing)

    # violent random coefficients: guaranteed folding
    rng = np.random.default_rng(7)
    n = len(transform.GetParameters())
    transform.SetParameters([float(v) for v in rng.uniform(-12, 12, size=n)])

    before_min, before_frac = _jacobian_stats(
        transform, size, spacing, interior_only=True)
    assert before_frac > 0, 'test setup failed to produce folding'
    assert before_min < 0

    k = 0.32
    info = project_bspline_transform(transform, k=k)
    assert info['converged']

    # sample finer than the image: the C4 guarantee is continuous domain, not
    # just at grid points, so a coarse check could miss a fold between voxels
    after_min, after_frac = _jacobian_stats(
        transform, size, spacing, supersample=3, interior_only=True)
    assert after_frac == 0.0
    assert after_min > 0.0
    assert after_min >= info['min_jacobian_bound'] - 1e-6


def test_projection_preserves_a_valid_transform_exactly():
    size, spacing = (32, 32, 32), (1.0, 1.0, 1.0)
    transform = _bspline_transform(mesh_size=(4, 4, 4), size=size, spacing=spacing)
    rng = np.random.default_rng(3)
    n = len(transform.GetParameters())
    # small enough to satisfy C4 at k=0.32 with 8.75 unit knot spacing
    transform.SetParameters([float(v) for v in rng.uniform(-0.5, 0.5, size=n)])
    before = np.asarray(transform.GetParameters())

    info = project_bspline_transform(transform, k=0.32)

    assert info['n_violating_before'] == 0
    np.testing.assert_array_equal(np.asarray(transform.GetParameters()), before)


# ---------------------------------------------------------------------------
# 6. penalty
# ---------------------------------------------------------------------------

def test_penalty_is_zero_exactly_on_feasible_coefficients():
    coefficients = _random_coefficients(0.01)
    assert chun_fessler_penalty(coefficients, KNOT_SPACING) == 0.0

    projected, _ = project_to_c4(_random_coefficients(10.0), KNOT_SPACING)
    assert chun_fessler_penalty(projected, KNOT_SPACING) == pytest.approx(0.0)


def test_penalty_is_quadratic_in_the_excess():
    lo, hi = coefficient_bounds(KNOT_SPACING, k=0.32)
    # isolated single pair, as above, so only one term contributes
    coefficients = np.zeros((3, 1, 1, 2))

    coefficients[0, 0, 0, 1] = hi[0, 2] + 1.0
    single = chun_fessler_penalty(coefficients, KNOT_SPACING, k=0.32)
    assert single == pytest.approx(0.5 * 1.0 ** 2)

    coefficients[0, 0, 0, 1] = hi[0, 2] + 2.0
    double = chun_fessler_penalty(coefficients, KNOT_SPACING, k=0.32)

    assert double == pytest.approx(4.0 * single)


# ---------------------------------------------------------------------------
# 7. SimpleITK parameter layout
# ---------------------------------------------------------------------------

def test_parameter_layout_round_trips_against_coefficient_images():
    transform = _bspline_transform()
    n = len(transform.GetParameters())
    transform.SetParameters([float(v) for v in np.arange(n)])

    coefficients, knot_spacing = bspline_coefficients(transform)

    images = transform.GetCoefficientImages()
    # component axis is reversed to zyx, so component 0 is the last image
    np.testing.assert_array_equal(
        coefficients[0], sitk.GetArrayFromImage(images[2]))
    np.testing.assert_array_equal(
        coefficients[2], sitk.GetArrayFromImage(images[0]))
    # knot spacing is the coefficient image spacing, reversed to zyx
    np.testing.assert_allclose(
        knot_spacing, np.array(images[0].GetSpacing())[::-1])
    # and it matches physical dimensions / mesh size
    np.testing.assert_allclose(
        knot_spacing,
        (np.array(transform.GetTransformDomainPhysicalDimensions())
         / np.array(transform.GetTransformDomainMeshSize()))[::-1],
    )


def test_set_coefficients_round_trips():
    transform = _bspline_transform()
    coefficients, _ = bspline_coefficients(transform)
    coefficients = coefficients + np.arange(coefficients.size).reshape(
        coefficients.shape)

    set_bspline_coefficients(transform, coefficients)
    read_back, _ = bspline_coefficients(transform)

    np.testing.assert_allclose(read_back, coefficients)


# ---------------------------------------------------------------------------
# affine helper
# ---------------------------------------------------------------------------

def test_is_orientation_preserving():
    assert is_orientation_preserving(np.eye(4))
    flip = np.eye(4)
    flip[0, 0] = -1.0
    assert not is_orientation_preserving(flip)
    with pytest.raises(ValueError):
        is_orientation_preserving(np.ones((3, 4)))


# ---------------------------------------------------------------------------
# 10. cost
# ---------------------------------------------------------------------------

def test_projection_is_cheap_on_a_realistic_mesh():
    # ~16^3 coefficient grid, the size a 128^3 block produces at 50um control
    # point spacing and 5um voxels
    coefficients = _random_coefficients(10.0, grid=(16, 16, 16), seed=11)
    knot_spacing = np.array([50.0, 50.0, 50.0])

    start = time.perf_counter()
    _, info = project_to_c4(coefficients, knot_spacing, k=0.32)
    elapsed = time.perf_counter() - start

    assert info['converged']
    # generous bound; the point is to catch an implementation that stops being
    # vectorized, not to benchmark the machine
    assert elapsed < 0.5, f'projection took {elapsed:.3f}s'
