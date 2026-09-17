import logging
import re

import numpy as np
import pytest
import SimpleITK as sitk

from bigstream.align import deformable_align, alignment_pipeline
from bigstream.align_constraints import project_bspline_transform
import bigstream.utility as ut
import bigstream.transform as bst


# ---------------------------------------------------------------------------
# helpers (mirrors test_demons_align helpers)
# ---------------------------------------------------------------------------

def _make_sphere(shape, radius_fraction=0.3):
    c = np.array(shape) / 2.0
    coords = np.mgrid[tuple(slice(0, s) for s in shape)]
    r = np.sqrt(sum((coords[i] - c[i]) ** 2 for i in range(len(shape))))
    return (r < radius_fraction * min(shape)).astype(np.float32)


def _random_smooth_volume(shape, noise_scale=0.1, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    base = _make_sphere(shape, radius_fraction=0.35)
    noise = rng.uniform(0, noise_scale, size=shape).astype(np.float32)
    return base + noise


def _apply_translation_sitk(image_np, spacing, shift_voxels):
    """Translate image_np by shift_voxels (ZYX), return numpy array."""
    sitk_img = ut.numpy_to_sitk(image_np, spacing)
    sitk_img = sitk.Cast(sitk_img, sitk.sitkFloat32)
    ndim = image_np.ndim
    tx = sitk.TranslationTransform(ndim)
    # SITK offset is in physical XYZ; shift_voxels is ZYX, spacing is ZYX
    physical_shift = (shift_voxels * np.asarray(spacing))[::-1]
    tx.SetOffset(physical_shift.tolist())
    resampled = sitk.Resample(
        sitk_img, sitk_img, tx, sitk.sitkLinear, 0.0, sitk.sitkFloat32,
    )
    return sitk.GetArrayFromImage(resampled).astype(np.float32)


def _apply_field(fix_np, mov_np, spacing, field):
    """Warp mov_np by field, return warped array and SSD with fix_np."""
    sitk_mov = sitk.Cast(ut.numpy_to_sitk(mov_np, spacing), sitk.sitkFloat32)
    sitk_fix = sitk.Cast(ut.numpy_to_sitk(fix_np, spacing), sitk.sitkFloat32)
    tx = bst.field_to_displacement_field_transform(field, spacing)
    warped = sitk.Resample(sitk_mov, sitk_fix, tx, sitk.sitkLinear, 0.0)
    warped_np = sitk.GetArrayFromImage(warped).astype(np.float32)
    ssd = float(np.mean((fix_np - warped_np) ** 2))
    return warped_np, ssd


SHAPE = (32, 32, 32)
SPACING = np.array([1.0, 1.0, 1.0])

# BSpline parameters used across tests:
# control_point_spacing=8.0 → 4 control points per axis at finest level
# control_point_levels=[2, 1] → optimize at 16-vox then 8-vox CP spacing
CTRL_SPACING = 8.0
CTRL_LEVELS = [2, 1]
IRM_KWARGS = dict(metric='MS', shrink_factors=(2, 1), smooth_sigmas=(1.0, 0.0))


# ---------------------------------------------------------------------------
# 1. Identity test
# ---------------------------------------------------------------------------

def test_identity_returns_near_zero_field():
    """fix == mov should produce a near-zero displacement field."""
    fix = _random_smooth_volume(SHAPE)
    params, field = deformable_align(
        fix, fix.copy(), SPACING, SPACING,
        CTRL_SPACING, CTRL_LEVELS, **IRM_KWARGS,
    )
    assert field.shape == SHAPE + (3,)
    assert np.max(np.abs(field)) < 1.0, (
        f"Identity registration produced non-trivial field: max={np.max(np.abs(field)):.3f}"
    )


# ---------------------------------------------------------------------------
# 2. Shifted volume test
# ---------------------------------------------------------------------------

def test_shifted_volume_recovery():
    """Applying the returned field to shifted mov should reduce alignment error."""
    fix = _random_smooth_volume(SHAPE)
    shift = np.array([3.0, 5.0, 2.0])  # ZYX voxels
    mov = _apply_translation_sitk(fix, SPACING, shift)

    initial_ssd = float(np.mean((fix - mov) ** 2))

    params, field = deformable_align(
        fix, mov, SPACING, SPACING,
        CTRL_SPACING, CTRL_LEVELS,
        metric='MS', shrink_factors=(4, 2, 1), smooth_sigmas=(2.0, 1.0, 0.0),
    )
    assert field.shape == SHAPE + (3,)

    _, final_ssd = _apply_field(fix, mov, SPACING, field)
    assert final_ssd < 0.5 * initial_ssd, (
        f"Alignment did not improve: initial_ssd={initial_ssd:.4f}, final_ssd={final_ssd:.4f}"
    )


# ---------------------------------------------------------------------------
# 3. BSpline-distorted volume test
# ---------------------------------------------------------------------------

def test_bspline_distorted_recovery():
    """deformable_align should substantially reduce residual from a BSpline deformation."""
    fix = _random_smooth_volume(SHAPE)

    sitk_fix = sitk.Cast(ut.numpy_to_sitk(fix, SPACING), sitk.sitkFloat32)
    bspline_tx = sitk.BSplineTransformInitializer(sitk_fix, [2] * 3, order=3)
    params_init = np.array(bspline_tx.GetParameters())
    rng = np.random.default_rng(7)
    params_init += rng.uniform(-3.0, 3.0, size=params_init.shape)
    bspline_tx.SetParameters(params_init.tolist())
    sitk_mov = sitk.Resample(
        sitk_fix, sitk_fix, bspline_tx, sitk.sitkLinear, 0.0, sitk.sitkFloat32,
    )
    mov = sitk.GetArrayFromImage(sitk_mov).astype(np.float32)

    initial_ssd = float(np.mean((fix - mov) ** 2))
    assert initial_ssd > 1e-4, "deformation too small to test recovery"

    params, field = deformable_align(
        fix, mov, SPACING, SPACING,
        CTRL_SPACING, CTRL_LEVELS, **IRM_KWARGS,
    )
    assert field.shape == SHAPE + (3,)

    _, final_ssd = _apply_field(fix, mov, SPACING, field)
    assert final_ssd < 0.5 * initial_ssd, (
        f"BSpline distortion recovery failed: initial={initial_ssd:.4f}, final={final_ssd:.4f}"
    )


# ---------------------------------------------------------------------------
# 4. Mask test
# ---------------------------------------------------------------------------

def test_mask_runs_and_returns_correct_shape():
    """With a fix_mask, deformable_align should complete and return fix.shape + (ndim,)."""
    fix = _random_smooth_volume(SHAPE)
    shift = np.array([3.0, 3.0, 3.0])
    mov = _apply_translation_sitk(fix, SPACING, shift)

    mask = np.zeros(SHAPE, dtype=np.uint8)
    mask[:, :, SHAPE[2] // 2:] = 1  # right half is foreground

    params, field = deformable_align(
        fix, mov, SPACING, SPACING,
        CTRL_SPACING, CTRL_LEVELS,
        fix_mask=mask, **IRM_KWARGS,
    )
    assert field.shape == SHAPE + (3,)
    assert params.ndim == 1


# ---------------------------------------------------------------------------
# 5. static_transform_list test
# ---------------------------------------------------------------------------

def test_static_transform_list_pre_aligns():
    """static_transform_list sets an initial affine; the BSpline corrects the residual.

    A partial (75%) pre-alignment is used to avoid the edge case where a perfect
    pre-alignment sets IRM's initial metric to 0 and final_metric_check rejects any
    subsequent BSpline result as a degradation.

    The returned BSpline field is the residual only. The composed (affine + BSpline)
    must be applied to the original mov to measure total alignment quality.
    """
    fix = _random_smooth_volume(SHAPE)
    shift = np.array([4.0, 4.0, 4.0])  # ZYX voxels
    mov = _apply_translation_sitk(fix, SPACING, shift)

    initial_ssd = float(np.mean((fix - mov) ** 2))

    # partial affine: 75% of the true shift, leaving ~1 voxel for BSpline to fix
    # bigstream affines map fixed→moving with negative translation
    affine = np.eye(4)
    affine[:3, 3] = -shift[::-1] * 0.75  # XYZ, negative to map fixed→moving

    params, field = deformable_align(
        fix, mov, SPACING, SPACING,
        CTRL_SPACING, CTRL_LEVELS,
        static_transform_list=[affine], **IRM_KWARGS,
    )
    assert field.shape == SHAPE + (3,)

    # compose affine + BSpline correction into a single total field, then apply
    total_field = bst.compose_transforms(affine, field, SPACING, SPACING)
    _, final_ssd = _apply_field(fix, mov, SPACING, total_field)
    assert final_ssd < 0.5 * initial_ssd, (
        f"Static transform pre-alignment failed: initial={initial_ssd:.4f}, final={final_ssd:.4f}"
    )


# ---------------------------------------------------------------------------
# 6. alignment_spacing round-trip test
# ---------------------------------------------------------------------------

def test_alignment_spacing_roundtrip():
    """Returned field shape must equal fix.shape + (ndim,) for any alignment_spacing."""
    fix = _random_smooth_volume(SHAPE)
    mov = fix.copy()

    for spacing in [None, 2.0, 4.0]:
        params, field = deformable_align(
            fix, mov, SPACING, SPACING,
            CTRL_SPACING, [1],
            alignment_spacing=spacing,
            metric='MS', shrink_factors=(1,), smooth_sigmas=(0.0,),
        )
        assert field.shape == SHAPE + (3,), (
            f"alignment_spacing={spacing}: expected {SHAPE + (3,)}, got {field.shape}"
        )


# ---------------------------------------------------------------------------
# 7. Pipeline composition test
# ---------------------------------------------------------------------------

def test_pipeline_with_deform_step():
    """alignment_pipeline with a 'deform' step should return a displacement field."""
    fix = _random_smooth_volume(SHAPE)
    mov = fix.copy()

    result = alignment_pipeline(
        fix, mov, SPACING, SPACING,
        steps=[
            ('affine', dict(
                metric='MS',
                shrink_factors=(2, 1),
                smooth_sigmas=(1.0, 0.0),
            )),
            ('deform', dict(
                control_point_spacing=CTRL_SPACING,
                control_point_levels=[1],
                metric='MS',
                shrink_factors=(1,),
                smooth_sigmas=(0.0,),
            )),
        ],
        return_format='flatten',
    )
    assert isinstance(result, np.ndarray)
    assert result.shape == SHAPE + (3,)


def test_pipeline_default_case_deform():
    """alignment_pipeline default case with 'deform' in steps returns zero field."""
    result = alignment_pipeline(
        None, np.zeros(SHAPE, dtype=np.float32), SPACING, SPACING,
        steps=[('deform', {
            'control_point_spacing': CTRL_SPACING,
            'control_point_levels': [1],
        })],
        return_format='flatten',
    )
    assert isinstance(result, np.ndarray)
    assert result.shape == SHAPE + (3,)
    assert np.all(result == 0.0)


# ---------------------------------------------------------------------------
# 8. Anisotropic control_point_spacing test
# ---------------------------------------------------------------------------

def test_anisotropic_control_point_spacing():
    """control_point_spacing accepts a scalar, a short array (last value
    repeated to fill remaining axes), or a full per-axis (zyx) array."""
    fix = _random_smooth_volume(SHAPE)
    mov = fix.copy()

    for cps in (8.0, [16.0, 8.0], [16.0, 8.0, 4.0]):
        params, field = deformable_align(
            fix, mov, SPACING, SPACING,
            cps, [1],
            metric='MS', shrink_factors=(1,), smooth_sigmas=(0.0,),
        )
        assert field.shape == SHAPE + (3,), (
            f"control_point_spacing={cps}: expected {SHAPE + (3,)}, got {field.shape}"
        )


def test_control_point_spacing_too_many_elements_truncates():
    """An array longer than the image dimensionality is truncated to the
    leading `ndim` (zyx) elements rather than rejected."""
    fix = _random_smooth_volume(SHAPE)
    mov = fix.copy()

    params, field = deformable_align(
        fix, mov, SPACING, SPACING,
        [16.0, 8.0, 4.0, 2.0], [1],
        metric='MS', shrink_factors=(1,), smooth_sigmas=(0.0,),
    )
    assert field.shape == SHAPE + (3,), (
        f"expected {SHAPE + (3,)}, got {field.shape}"
    )


# ---------------------------------------------------------------------------
# 9. Return-tuple consistency test
# ---------------------------------------------------------------------------

def test_return_tuple_consistency():
    """params is a 1d array; field has the right shape; applying field reduces SSD."""
    fix = _random_smooth_volume(SHAPE)
    shift = np.array([3.0, 3.0, 3.0])
    mov = _apply_translation_sitk(fix, SPACING, shift)

    initial_ssd = float(np.mean((fix - mov) ** 2))

    params, field = deformable_align(
        fix, mov, SPACING, SPACING,
        CTRL_SPACING, CTRL_LEVELS, **IRM_KWARGS,
    )

    # params is the flattened bspline parameterization (fixed + free control points)
    assert params.ndim == 1
    assert params.size > 0
    # field is the dense displacement field on the original grid
    assert field.shape == SHAPE + (3,)
    assert field.dtype == np.float32

    # verify the field is functional: applying it should reduce alignment error
    _, final_ssd = _apply_field(fix, mov, SPACING, field)
    assert final_ssd < 0.5 * initial_ssd, (
        f"Field does not reduce SSD: initial={initial_ssd:.4f}, final={final_ssd:.4f}"
    )


# ---------------------------------------------------------------------------
# 10. Local invertibility constraint (bspline_constraints) and max_displacement
# ---------------------------------------------------------------------------

def _interior_jacobian(field, spacing, margin=4):
    """min |J| and non-positive fraction, excluding a boundary margin.

    The boundary is excluded because the bspline evaluates to zero
    displacement outside its supported region, which puts a step in the
    sampled field at the domain edge regardless of the coefficients.
    """
    disp = ut.numpy_to_sitk(field[..., ::-1].astype(np.float64), spacing, vector=True)
    jacobian = sitk.GetArrayFromImage(sitk.DisplacementFieldJacobianDeterminant(disp))
    trim = slice(margin, -margin if margin else None)
    core = jacobian[trim, trim, trim]
    return float(core.min()), float((core <= 0).mean())


def _folding_pair():
    """A fix/mov pair whose unconstrained registration is known to fold."""
    fix = _random_smooth_volume(SHAPE)
    sitk_fix = sitk.Cast(ut.numpy_to_sitk(fix, SPACING), sitk.sitkFloat32)
    bspline_tx = sitk.BSplineTransformInitializer(sitk_fix, [3] * 3, order=3)
    params = np.array(bspline_tx.GetParameters())
    params += np.random.default_rng(7).uniform(-6.0, 6.0, size=params.shape)
    bspline_tx.SetParameters(params.tolist())
    mov = sitk.GetArrayFromImage(sitk.Resample(
        sitk_fix, sitk_fix, bspline_tx, sitk.sitkLinear, 0.0, sitk.sitkFloat32,
    )).astype(np.float32)
    return fix, mov


# fine control points + an aggressive optimizer, to provoke folding
FOLDING_KWARGS = dict(
    metric='MS', shrink_factors=(1,), smooth_sigmas=(0.0,),
    optimizer_args={'learningRate': 2.0, 'minStep': 0.0,
                    'numberOfIterations': 200},
)


def test_constraint_disabled_by_default():
    """Omitting the constraint and passing None give identical results."""
    fix, mov = _folding_pair()

    _, omitted = deformable_align(
        fix, mov, SPACING, SPACING, 4.0, [1], **FOLDING_KWARGS,
    )
    _, explicit_none = deformable_align(
        fix, mov, SPACING, SPACING, 4.0, [1],
        bspline_constraints=None, **FOLDING_KWARGS,
    )
    np.testing.assert_array_equal(omitted, explicit_none)


def test_constraint_removes_folding():
    """The unconstrained fit folds; the constrained fit does not."""
    fix, mov = _folding_pair()

    _, unconstrained = deformable_align(
        fix, mov, SPACING, SPACING, 4.0, [1], **FOLDING_KWARGS,
    )
    unconstrained_min, unconstrained_frac = _interior_jacobian(unconstrained, SPACING)
    assert unconstrained_frac > 0, 'test setup no longer produces folding'
    assert unconstrained_min < 0

    _, constrained = deformable_align(
        fix, mov, SPACING, SPACING, 4.0, [1],
        bspline_constraints={'k': 0.32}, **FOLDING_KWARGS,
    )
    constrained_min, constrained_frac = _interior_jacobian(constrained, SPACING)
    assert constrained_frac == 0.0
    assert constrained_min > 0.0
    # and it is still a real deformation, not a collapse to identity
    assert np.max(np.abs(constrained)) > 0.5


def test_bspline_render_extends_past_the_transform_domain():
    """
    Voxels outside the bspline domain get the nearest in-domain displacement.

    A bspline evaluates to zero outside its domain, so rendering it over a
    larger grid used to put a step in the field there. The domain below covers
    voxel centres 0..28 of a 32 voxel axis, leaving 3 voxels past the far face.
    """
    domain = sitk.Image([8] * 3, sitk.sitkUInt8)
    domain.SetSpacing((4.0, 4.0, 4.0))
    transform = sitk.BSplineTransformInitializer(domain, [2] * 3, order=3)
    transform.SetTransformDomainOrigin((0.0, 0.0, 0.0))
    transform.SetTransformDomainPhysicalDimensions((28.0, 28.0, 28.0))
    params = np.array(transform.GetParameters())
    params += np.random.default_rng(3).uniform(-2.0, 2.0, size=params.shape)
    transform.SetParameters(params.tolist())

    field = bst.bspline_to_displacement_field(
        transform, SHAPE, spacing=SPACING, origin=np.zeros(3),
    )

    # the 3 voxels past the domain replicate the last in-domain slice
    for axis in range(3):
        last_inside = np.take(field, [28], axis=axis)
        outside = np.take(field, [29, 30, 31], axis=axis)
        np.testing.assert_allclose(
            outside, np.repeat(last_inside, 3, axis=axis), rtol=0, atol=0)
    # and the replicated region is not trivially zero
    assert np.abs(np.take(field, [28], axis=0)).max() > 0

    _, folded_fraction = _interior_jacobian(field, SPACING, margin=0)
    assert folded_fraction == 0.0


def test_voxel_centre_bbox_domain_does_not_leave_a_zero_face():
    """
    A domain equal to the voxel centre bounding box must still render fold free.

    `SetInitialTransformAsBSpline` discards the initializer's domain and
    rebuilds it from the registration's fixed image as the voxel centre
    bounding box, `(size - 1) * spacing`. The outermost voxel centres then land
    exactly ON the domain edge, where ITK does not consider them inside the
    valid region and returns zero displacement - a full amplitude step in the
    field, and a fold sheet one voxel thick on a block face.

    Regression for a production block (2026-09-14) whose logs showed
    `disp magnitude min=0.0`, `smoothness dy: max jump=10.40` against
    `p99 jump=0.076`, and 80483 folded voxels that the diagnostics dismissed as
    a boundary artifact.
    """
    shape = (48, 48, 48)
    spacing = np.array([1.0905166, 0.64713602, 0.64754511])

    image = sitk.Image([s for s in shape], sitk.sitkUInt8)
    image.SetSpacing(tuple(float(s) for s in spacing[::-1]))
    transform = sitk.BSplineTransformInitializer(image, [3] * 3, order=3)
    # the domain the registration rebuilds, rather than the initializer's
    transform.SetTransformDomainOrigin((0.0, 0.0, 0.0))
    transform.SetTransformDomainPhysicalDimensions(
        tuple(float(v) for v in ((np.array(shape) - 1) * spacing)[::-1]))
    rng = np.random.default_rng(0)
    transform.SetParameters(
        rng.uniform(-16.0, 16.0, size=len(transform.GetParameters())).tolist())
    # make the coefficients fold free in their own right, so anything this
    # test catches is the boundary behaviour and not a C4 violation
    project_bspline_transform(transform, k=0.2)

    field = bst.bspline_to_displacement_field(
        transform, shape, spacing=spacing, origin=None)

    magnitude = np.linalg.norm(field, axis=-1)
    assert not np.any(magnitude == 0), (
        f'{int((magnitude == 0).sum())} voxels render to exactly zero '
        'displacement - the bspline evaluated outside its valid region'
    )
    # no single voxel step may dwarf the rest of the field
    for axis in range(3):
        jumps = np.linalg.norm(np.diff(field, axis=axis), axis=-1)
        assert jumps.max() < 10 * np.percentile(jumps, 99), (
            f'axis {axis}: max jump {jumps.max():.4f} against p99 '
            f'{np.percentile(jumps, 99):.4f} - a discontinuity, not a gradient'
        )
    full_min, full_frac = _interior_jacobian(field, spacing, margin=0)
    assert full_frac == 0.0, f'{100 * full_frac:.4f}% folded, min|J|={full_min}'


def test_alignment_spacing_does_not_fold_at_the_domain_boundary():
    """
    A constrained fit must not fold anywhere, including at the block faces.

    The bspline domain is built from the original fixed image grid rather than
    from the skip sampled image the registration runs on. If it were built
    from the skip sampled image the two would cover different physical
    extents, the outermost rendered voxels would fall outside the domain where
    a bspline evaluates to zero displacement, and the resulting step would
    report folding there no matter how well constrained the coefficients are.
    So this asserts over the whole field, with no boundary margin excluded.
    """
    fix, mov = _folding_pair()

    for alignment_spacing in (None, 2.0, 4.0):
        _, field = deformable_align(
            fix, mov, SPACING, SPACING, 4.0, [1],
            alignment_spacing=alignment_spacing,
            bspline_constraints={'k': 0.32}, **FOLDING_KWARGS,
        )
        full_min, full_frac = _interior_jacobian(field, SPACING, margin=0)
        assert full_frac == 0.0, (
            f'alignment_spacing={alignment_spacing}: {100 * full_frac:.4f}% of '
            f'voxels fold, min|J|={full_min:.4f}'
        )
        assert full_min > 0.0


def test_constraint_rejects_bad_configuration():
    fix, mov = _folding_pair()

    with pytest.raises(ValueError, match='sum'):
        deformable_align(
            fix, mov, SPACING, SPACING, 4.0, [1],
            bspline_constraints={'k': [0.4, 0.4, 0.4]}, **FOLDING_KWARGS,
        )
    with pytest.raises(ValueError, match='unknown'):
        deformable_align(
            fix, mov, SPACING, SPACING, 4.0, [1],
            bspline_constraints={'kk': 0.3}, **FOLDING_KWARGS,
        )
    # max_displacement is deformable_align's own parameter now, not a
    # bspline_constraints key - nesting it there must fail loudly
    with pytest.raises(ValueError, match='unknown bspline_constraints keys'):
        deformable_align(
            fix, mov, SPACING, SPACING, 4.0, [1],
            bspline_constraints={'max_displacement': 16}, **FOLDING_KWARGS,
        )


def test_max_displacement_bounds_the_rendered_field_independently():
    """
    max_displacement is independent of bspline_constraints: it must still
    project (and bound the field) even when bspline_constraints is left
    disabled, using the default k rather than being silently skipped.
    """
    fix, mov = _folding_pair()
    U = 3.0

    _, field = deformable_align(
        fix, mov, SPACING, SPACING, 4.0, [1],
        max_displacement=U, **FOLDING_KWARGS,
    )
    assert np.abs(field).max() <= U + 1e-6, (
        f'rendered displacement {np.abs(field).max()} exceeds bound {U}'
    )
    # and it is still a real deformation, not a collapse to identity
    assert np.max(np.abs(field)) > 0.0


def test_max_displacement_combines_with_bspline_constraints():
    """Both knobs at once still bound the field and keep it fold free."""
    fix, mov = _folding_pair()
    U = 4.0

    _, field = deformable_align(
        fix, mov, SPACING, SPACING, 4.0, [1],
        bspline_constraints={'k': 0.2}, max_displacement=U, **FOLDING_KWARGS,
    )
    assert np.abs(field).max() <= U + 1e-6
    min_jac, folded_frac = _interior_jacobian(field, SPACING)
    assert folded_frac == 0.0
    assert min_jac > 0.0


def test_constrained_metric_is_reevaluated_after_projection(caplog):
    """
    The metric reported after projection must reflect the projected transform.

    SimpleITK's MetricEvaluate reports the transform state captured during
    Execute and ignores coefficients written afterwards, so deformable_align
    has to re-set the initial transform before re-evaluating. Without that the
    'constraint adjusted metric' is stale and final_metric_check scores a
    transform that is not the one returned.
    """
    fix, mov = _folding_pair()

    with caplog.at_level(logging.INFO, logger='bigstream.align'):
        deformable_align(
            fix, mov, SPACING, SPACING, 4.0, [1],
            bspline_constraints={'k': 0.32}, **FOLDING_KWARGS,
        )

    lines = [r.message for r in caplog.records
             if 'final optimization metric' in r.message]
    assert lines, 'no constraint metric line was logged'
    optimization = float(re.search(
        r'final optimization metric: (-?[\d.eE+-]+)', lines[-1]).group(1))
    adjusted = float(re.search(
        r'constraint adjusted metric: (-?[\d.eE+-]+)', lines[-1]).group(1))
    assert optimization != adjusted, (
        'metric was not re-evaluated after projection: the projection moved '
        'coefficients but the reported metric is unchanged'
    )
