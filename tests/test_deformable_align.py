import numpy as np
import SimpleITK as sitk

from bigstream.align import deformable_align, alignment_pipeline
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
# 8. Return-tuple consistency test
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
