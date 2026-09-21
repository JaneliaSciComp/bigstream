"""
Tests for the displacement amplitude bound (`max_displacement`).

C4 bounds the *derivatives* of the deformation, which stops a single block's
field from folding. It says nothing about amplitude, so a smooth but very large
displacement is fully compliant - and while such a block does not fold on its
own, it folds where `distributed_align` blends it against a neighbour that
fitted something different. These tests cover the bound that fixes that, and
the blend-level failure it is there to prevent.
"""
import numpy as np
import pytest
import SimpleITK as sitk

import bigstream.transform as bst
from bigstream.align_constraints import (
    blend_safe_displacement_bound,
    c4_violations,
    project_to_c4,
    validate_deform_regularization_params,
)
from bigstream.distributed_align import _get_transform_weights


NDIM = 3
GRID = (NDIM, 8, 8, 8)
KNOT = np.array([10.0, 10.0, 10.0])


def _runaway_coefficients(seed=0, amplitude=200.0, noise=1.0):
    """A smooth, huge coefficient grid - the runaway block fit."""
    rng = np.random.default_rng(seed)
    ramp = np.linspace(-amplitude, amplitude, GRID[1])
    c = np.empty(GRID, dtype=np.float64)
    for q in range(NDIM):
        c[q] = ramp.reshape(
            tuple(-1 if a == q else 1 for a in range(NDIM)))
    c += rng.uniform(-noise, noise, size=GRID)
    return c


def _compliant_coefficients(k=0.2):
    """A grid that satisfies both C4 and any generous displacement bound.

    Adjacent differences must stay under `knot_spacing * k` (= 2.0 here), so
    this uses a gentle ramp plus small noise rather than the runaway grid.
    """
    step = 0.5
    ramp = np.arange(GRID[1], dtype=np.float64) * step
    ramp -= ramp.mean()
    c = np.empty(GRID, dtype=np.float64)
    for q in range(NDIM):
        c[q] = ramp.reshape(tuple(-1 if a == q else 1 for a in range(NDIM)))
    c += np.random.default_rng(5).uniform(-0.05, 0.05, size=GRID)
    assert c4_violations(c, KNOT, k=k)['n_violating'] == 0
    return c


# ---------------------------------------------------------------------------
# 1. the bound holds on the rendered field, not just the coefficients
# ---------------------------------------------------------------------------

def test_bound_holds_on_the_rendered_displacement():
    """
    Clamping coefficients bounds the rendered displacement per component.

    The cubic bspline basis is non-negative and a partition of unity, so the
    displacement at any point is a convex combination of nearby coefficients.
    Rendered on a grid finer than the control points, because the guarantee is
    a continuous-domain one.
    """
    U = 25.0
    shape = (48, 48, 48)
    spacing = np.array([1.0, 1.0, 1.0])

    image = sitk.Image([s for s in shape], sitk.sitkUInt8)
    image.SetSpacing(tuple(float(s) for s in spacing))
    transform = sitk.BSplineTransformInitializer(image, [4] * NDIM, order=3)

    rng = np.random.default_rng(11)
    params = np.array(transform.GetParameters())
    params += rng.uniform(-300.0, 300.0, size=params.shape)
    transform.SetParameters(params.tolist())

    from bigstream.align_constraints import project_bspline_transform
    info = project_bspline_transform(transform, k=0.2, max_displacement=U)
    assert info['n_clamped'] > 0, 'test setup no longer exceeds the bound'
    assert info['max_coefficient_after'] <= U + 1e-9

    field = bst.bspline_to_displacement_field(
        transform, shape, spacing=spacing, origin=np.zeros(NDIM),
    )
    assert np.abs(field).max() <= U + 1e-4, (
        f'rendered displacement {np.abs(field).max()} exceeds the bound {U}'
    )


def test_bound_and_c4_hold_simultaneously():
    """
    POCS over both convex sets lands in their intersection.

    Alternating between the C4 slabs and the displacement box converges more
    slowly than C4 alone (~150 sweeps here vs ~15), which is why
    `validate_deform_regularization_params` raises the default sweep budget
    when `max_displacement` is set.
    """
    U = 30.0
    c = _runaway_coefficients()
    projected, info = project_to_c4(
        c, KNOT, k=0.2, max_displacement=U, max_sweeps=300, tol=1e-9)

    assert np.abs(projected).max() <= U + 1e-9
    assert c4_violations(projected, KNOT, k=0.2)['n_violating'] == 0
    assert info['converged']


def test_combined_projection_gets_a_larger_default_sweep_budget():
    """
    max_displacement is deformable_align's own parameter, not a
    bspline_constraints key - so it is deformable_align's job to pick 100 or
    300 based on whether its own max_displacement is set, and pass that
    through as default_sweeps.
    """
    assert validate_deform_regularization_params(
        {'k': 0.2}, NDIM, default_sweeps=100)['max_sweeps'] == 100
    assert validate_deform_regularization_params(
        {'k': 0.2}, NDIM, default_sweeps=300)['max_sweeps'] == 300
    # an explicit value always wins
    assert validate_deform_regularization_params(
        {'k': 0.2, 'max_sweeps': 42}, NDIM, default_sweeps=300)['max_sweeps'] == 42


def test_anisotropic_bound_is_per_component():
    U = [10.0, 20.0, 40.0]          # zyx
    projected, _ = project_to_c4(
        _runaway_coefficients(), KNOT, k=0.2, max_displacement=U)
    for q, limit in enumerate(U):
        assert np.abs(projected[q]).max() <= limit + 1e-9


# ---------------------------------------------------------------------------
# 2. dead zone / no-op behaviour
# ---------------------------------------------------------------------------

def test_compliant_grid_is_untouched():
    """A field satisfying both constraints is returned byte-identical."""
    c = _compliant_coefficients()
    projected, info = project_to_c4(c, KNOT, k=0.2, max_displacement=100.0)
    assert info['n_clamped'] == 0
    assert info['max_coefficient_shift'] == 0.0
    assert info['sweeps_run'] == 0
    np.testing.assert_array_equal(projected, c)


def test_none_is_a_noop_against_the_c4_only_projection():
    """max_displacement=None reproduces the C4-only result exactly."""
    c = _runaway_coefficients()
    with_none, info_none = project_to_c4(c, KNOT, k=0.2, max_displacement=None)
    baseline, info_base = project_to_c4(c, KNOT, k=0.2)
    np.testing.assert_array_equal(with_none, baseline)
    assert info_none['n_clamped'] == 0
    assert info_none['sweeps_run'] == info_base['sweeps_run']


# ---------------------------------------------------------------------------
# 3. the safe ceiling
# ---------------------------------------------------------------------------

def test_blend_safe_bound_matches_the_hand_worked_example():
    """
    U <= (1 - sum(k) - m) * L / (2*ndim), with L = (2*overlap - 1) * spacing.

    The `ndim` divisor matters: Lemma 2 sums the effective derivative bound
    over every row of the jacobian, so each row picks up its own 2U/L blend
    term - see `test_bound_excludes_the_measured_folding_threshold`.
    """
    overlaps = np.array([102, 102, 102])
    spacing = np.array([2.181, 1.294, 1.295])
    k, m = 0.2, 0.1
    L = (2 * 102 - 1) * 1.294               # tightest axis
    expected = (1 - 3 * k - m) * L / (2 * NDIM)
    got = blend_safe_displacement_bound(overlaps, spacing, k, min_jacobian=m)
    assert got == pytest.approx(expected, rel=1e-9)


def test_bound_excludes_the_measured_folding_threshold():
    """
    The bound must sit below where blending actually starts to fold.

    Measured by bisection on the assembled field for this lattice (one runaway
    block, unit spacing, L=25): folding begins at a per-component displacement
    of ~9.04. A bound that omits the `ndim` divisor returns 12.50 and would
    permit it; the correct bound returns 4.17.
    """
    overlaps = np.array([13, 13, 13])
    spacing = np.ones(NDIM)
    # constant per-block fields have no intra-block derivative, so k -> 0 and
    # min_jacobian -> 0 isolates the blend term
    got = blend_safe_displacement_bound(
        overlaps, spacing, 1e-12, min_jacobian=0.0)
    measured_threshold = 9.04
    assert got < measured_threshold, (
        f'bound {got} permits displacements that were measured to fold at '
        f'{measured_threshold}'
    )
    assert got == pytest.approx((2 * 13 - 1) / (2 * NDIM), rel=1e-6)


def test_blend_safe_bound_reports_no_headroom_for_the_old_default():
    """k=0.32 leaves min|J| >= 0.04, which blending immediately consumes."""
    assert blend_safe_displacement_bound(
        [102, 102, 102], [2.181, 1.294, 1.295], 0.32, min_jacobian=0.1) == 0.0


def test_blend_safe_bound_is_infinite_without_overlap():
    assert blend_safe_displacement_bound(
        [0, 0, 0], [1.0, 1.0, 1.0], 0.2) == float('inf')


def test_max_displacement_is_not_a_bspline_constraints_key():
    """
    max_displacement moved out to be deformable_align's own parameter -
    nesting it under bspline_constraints (the old control_point_constraint
    shape) must fail loudly rather than silently do nothing.
    """
    with pytest.raises(ValueError, match='unknown bspline_constraints keys'):
        validate_deform_regularization_params(
            {'k': 0.2, 'max_displacement': 40.0}, NDIM)

    cfg = validate_deform_regularization_params({'k': 0.2}, NDIM)
    assert set(cfg) == {'k', 'K', 'max_sweeps', 'tol'}


# ---------------------------------------------------------------------------
# 4. the blend-level regression - the failure every single-block test missed
# ---------------------------------------------------------------------------

def _assemble(block_fields, blocksize, overlaps, nblocks, kept, dims):
    """Stitch per-block fields exactly as _compute_block_transform does."""
    out = np.zeros(tuple(dims) + (NDIM,), dtype=np.float64)
    offsets = np.array(list(np.ndindex(*(3,) * NDIM))) - 1
    for bi in sorted(kept):
        start = np.maximum(0, blocksize * np.array(bi) - overlaps)
        stop = np.minimum(dims, blocksize * np.array(bi) + blocksize + overlaps)
        coords = tuple(slice(a, b) for a, b in zip(start, stop))
        shape = tuple(b - a for a, b in zip(start, stop))
        nbrs = {tuple(int(x) for x in o): tuple(np.array(bi) + o) in kept
                for o in offsets}
        w = _get_transform_weights(bi, (blocksize,) * NDIM, tuple(overlaps),
                                   nbrs, tuple(nblocks), True)
        w = w[tuple(slice(0, s) for s in shape)]
        out[coords] += block_fields(bi, shape) * w[..., None]
    return out


def _folded_fraction(field):
    disp = sitk.GetImageFromArray(field[..., ::-1].astype(np.float64),
                                  isVector=True)
    jac = sitk.GetArrayFromImage(
        sitk.DisplacementFieldJacobianDeterminant(disp))
    return float((jac <= 0).mean()), float(jac.min())


@pytest.mark.parametrize('amplitude,expect_folding', [(60.0, True), (8.0, False)])
def test_blending_folds_only_when_the_amplitude_exceeds_the_ramp(
        amplitude, expect_folding):
    """
    A runaway block folds the *stitched* field even though it never folds alone.

    One block is given a large constant displacement while its neighbours have
    none. Each field is individually smooth and fold free; the fold is created
    entirely by grad(w) * (u_A - u_B) in the blend. Bounding the amplitude to
    within the ramp length removes it - which is the whole point of
    `max_displacement`.
    """
    blocksize, nb = 32, np.array([5, 5, 5])
    overlaps = np.array([13, 13, 13])          # overlap_factor 0.4
    dims = nb * blocksize
    kept = set(map(tuple, np.argwhere(np.ones(nb))))
    hot = (2, 2, 2)

    def fields(bi, shape):
        f = np.zeros(shape + (NDIM,), dtype=np.float64)
        if bi == hot:
            f[...] = amplitude / np.sqrt(NDIM)
        return f

    assembled = _assemble(fields, blocksize, overlaps, nb, kept, dims)
    folded, min_jac = _folded_fraction(assembled)

    # each block's own field is constant, so it cannot fold in isolation
    if expect_folding:
        assert folded > 0, (
            f'expected blend induced folding at amplitude {amplitude}, '
            f'min|J|={min_jac}'
        )
    else:
        assert folded == 0.0, (
            f'amplitude {amplitude} within the ramp should not fold, '
            f'got {100*folded:.3f}% min|J|={min_jac}'
        )


# --------------------------------------------------------------------------
# the bound under a non-linear blending ramp
# --------------------------------------------------------------------------


def test_linear_ramp_leaves_the_bound_exactly_unchanged():
    """
    The regression guard for making the bound ramp-aware: dividing by a gain
    of 1.0 must be a no-op, so every existing config keeps the ceiling it
    had. `==`, not `approx` - this is arithmetic, not a measurement.
    """
    overlaps, spacing, k = np.array([102, 102, 102]), np.array([2.181, 1.294, 1.295]), 0.2
    baseline = blend_safe_displacement_bound(overlaps, spacing, k)
    assert blend_safe_displacement_bound(
        overlaps, spacing, k, blend_ramp='linear') == baseline


def test_cosine_ramp_tightens_the_bound_by_pi_over_two():
    """
    A cosine ramp is `pi/2` steeper at its steepest point, and the bound is
    derived from exactly that peak gradient, so the ceiling drops by the
    same factor. This is the cost the shape is chosen against, not a
    conservative fudge.
    """
    overlaps, spacing, k = np.array([64, 64, 64]), np.ones(NDIM), 0.1
    linear = blend_safe_displacement_bound(overlaps, spacing, k)
    cosine = blend_safe_displacement_bound(overlaps, spacing, k,
                                           blend_ramp='cosine')
    assert linear == pytest.approx(12.700, abs=1e-3)
    assert cosine == pytest.approx(8.085, abs=1e-3)
    assert linear / cosine == pytest.approx(np.pi / 2, rel=1e-9)


def test_cosine_bound_closed_form():
    """U <= (1 - sum(k) - m) * L / (2*ndim*g), with g = pi/2."""
    overlaps = np.array([102, 102, 102])
    spacing = np.array([2.181, 1.294, 1.295])
    k, m = 0.2, 0.1
    L = (2 * 102 - 1) * 1.294               # tightest axis
    expected = (1 - 3 * k - m) * L / (2 * NDIM * (np.pi / 2))
    got = blend_safe_displacement_bound(overlaps, spacing, k, min_jacobian=m,
                                        blend_ramp='cosine')
    assert got == pytest.approx(expected, rel=1e-9)


def test_degenerate_bounds_ignore_the_ramp():
    """No headroom stays 0 and no overlap stays inf - the gain cannot rescue
    either, and must not turn `inf` into a nan."""
    assert blend_safe_displacement_bound(
        [102, 102, 102], [2.181, 1.294, 1.295], 0.32, min_jacobian=0.1,
        blend_ramp='cosine') == 0.0
    assert blend_safe_displacement_bound(
        [0, 0, 0], [1.0, 1.0, 1.0], 0.2, blend_ramp='cosine') == float('inf')


def test_unknown_ramp_is_refused_by_the_bound():
    with pytest.raises(ValueError, match='unsupported blend_ramp'):
        blend_safe_displacement_bound([64] * NDIM, np.ones(NDIM), 0.1,
                                      blend_ramp='hann')
