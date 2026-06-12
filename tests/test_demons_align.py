import numpy as np
import SimpleITK as sitk

from bigstream.align import demons_align, alignment_pipeline
import bigstream.utility as ut


# ---------------------------------------------------------------------------
# helpers
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
    # field is ZYX-component convention; field_to_displacement_field_transform
    # expects this format
    import bigstream.transform as bst
    tx = bst.field_to_displacement_field_transform(field, spacing)
    warped = sitk.Resample(sitk_mov, sitk_fix, tx, sitk.sitkLinear, 0.0)
    warped_np = sitk.GetArrayFromImage(warped).astype(np.float32)
    ssd = float(np.mean((fix_np - warped_np) ** 2))
    return warped_np, ssd


SHAPE = (32, 32, 32)
SPACING = np.array([1.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
# 1. Identity test
# ---------------------------------------------------------------------------

def test_identity_returns_near_zero_field():
    """fix == mov should produce a near-zero displacement field."""
    fix = _random_smooth_volume(SHAPE)
    _, field = demons_align(
        fix, fix.copy(), SPACING, SPACING,
        iterations=[50, 30, 20],
        smooth_sigmas=[2.0, 1.0, 0.5],
        shrink_factors=[4, 2, 1],
    )
    assert field.shape == SHAPE + (3,)
    assert np.max(np.abs(field)) < 1.0, (
        f"Identity registration produced non-trivial field: max={np.max(np.abs(field)):.3f}"
    )


# ---------------------------------------------------------------------------
# 2. Shifted volume test
# ---------------------------------------------------------------------------

def test_shifted_volume_recovery():
    """Applying the demons field to the shifted mov should reduce alignment error."""
    fix = _random_smooth_volume(SHAPE)
    shift = np.array([3.0, 5.0, 2.0])  # ZYX voxels
    mov = _apply_translation_sitk(fix, SPACING, shift)

    # initial SSD before alignment
    initial_ssd = float(np.mean((fix - mov) ** 2))

    _, field = demons_align(
        fix, mov, SPACING, SPACING,
        iterations=[100, 50, 30],
        smooth_sigmas=[2.0, 1.0, 0.5],
        shrink_factors=[4, 2, 1],
    )
    assert field.shape == SHAPE + (3,)

    _, final_ssd = _apply_field(fix, mov, SPACING, field)
    # alignment must improve SSD by at least 50%
    assert final_ssd < 0.5 * initial_ssd, (
        f"Alignment did not improve: initial_ssd={initial_ssd:.4f}, final_ssd={final_ssd:.4f}"
    )


# ---------------------------------------------------------------------------
# 3. Smooth BSpline-distorted volume test
# ---------------------------------------------------------------------------

def test_bspline_distorted_recovery():
    """demons_align should substantially reduce residual from a smooth deformation."""
    fix = _random_smooth_volume(SHAPE)

    sitk_fix = sitk.Cast(ut.numpy_to_sitk(fix, SPACING), sitk.sitkFloat32)
    mesh_size = [2] * 3
    bspline_tx = sitk.BSplineTransformInitializer(sitk_fix, mesh_size, order=3)
    params = np.array(bspline_tx.GetParameters())
    rng = np.random.default_rng(7)
    # use 3-voxel-scale perturbations to ensure the initial SSD is non-trivial
    params += rng.uniform(-3.0, 3.0, size=params.shape)
    bspline_tx.SetParameters(params.tolist())
    sitk_mov = sitk.Resample(
        sitk_fix, sitk_fix, bspline_tx, sitk.sitkLinear, 0.0, sitk.sitkFloat32,
    )
    mov = sitk.GetArrayFromImage(sitk_mov).astype(np.float32)

    initial_ssd = float(np.mean((fix - mov) ** 2))
    assert initial_ssd > 1e-4, "deformation too small to test recovery"

    _, field = demons_align(
        fix, mov, SPACING, SPACING,
        iterations=[80, 50, 30],
        smooth_sigmas=[2.0, 1.0, 0.5],
        shrink_factors=[4, 2, 1],
    )
    assert field.shape == SHAPE + (3,)

    _, final_ssd = _apply_field(fix, mov, SPACING, field)
    assert final_ssd < 0.5 * initial_ssd, (
        f"Bspline distortion recovery failed: initial={initial_ssd:.4f}, final={final_ssd:.4f}"
    )


# ---------------------------------------------------------------------------
# 4. Mask test
# ---------------------------------------------------------------------------

def test_mask_runs_and_returns_correct_shape():
    """With a fix_mask, demons_align should complete and return fix.shape + (ndim,).

    Note: Demons has no metric-mask interface. Masking is approximated by zeroing
    fix/mov outside the mask. This does NOT prevent the filter from computing
    displacements in masked regions; it only biases the forces toward zero there.
    """
    fix = _random_smooth_volume(SHAPE)
    shift = np.array([3.0, 3.0, 3.0])
    mov = _apply_translation_sitk(fix, SPACING, shift)

    mask = np.zeros(SHAPE, dtype=np.uint8)
    mask[:, :, SHAPE[2] // 2:] = 1  # right half is foreground

    _, field = demons_align(
        fix, mov, SPACING, SPACING,
        fix_mask=mask,
        iterations=[50, 30, 20],
        smooth_sigmas=[2.0, 1.0, 0.5],
        shrink_factors=[4, 2, 1],
    )
    assert field.shape == SHAPE + (3,)
    # masked-in (right) region should have non-trivial displacement
    right_field = field[:, :, SHAPE[2] // 2:]
    assert np.max(np.abs(right_field)) > 0.5, "Masked-in region should have meaningful displacement"


def test_mask_with_different_sampling_runs():
    """Masks can have the same physical domain with a different sampling."""
    fix = _random_smooth_volume(SHAPE)
    shift = np.array([2.0, 2.0, 2.0])
    mov = _apply_translation_sitk(fix, SPACING, shift)

    mask_shape = tuple(s // 2 for s in SHAPE)
    fix_mask = np.zeros(mask_shape, dtype=np.uint8)
    mov_mask = np.zeros(mask_shape, dtype=np.uint8)
    fix_mask[:, :, mask_shape[2] // 2:] = 1
    mov_mask[:, :, mask_shape[2] // 2:] = 1

    _, field = demons_align(
        fix, mov, SPACING, SPACING,
        fix_mask=fix_mask,
        mov_mask=mov_mask,
        iterations=[10],
        smooth_sigmas=[1.0],
        shrink_factors=[1],
    )

    assert field.shape == SHAPE + (3,)


# ---------------------------------------------------------------------------
# 5. static_transform_list test
# ---------------------------------------------------------------------------

def test_static_transform_list_pre_aligns_field():
    """static_transform_list pre-warps mov before Demons; returned field is the
    residual ONLY (not the total displacement). The residual should be smaller
    than the field demons produces without any pre-alignment."""
    fix = _random_smooth_volume(SHAPE)
    shift = np.array([4.0, 4.0, 4.0])  # ZYX voxels
    mov = _apply_translation_sitk(fix, SPACING, shift)

    demons_kwargs = dict(
        iterations=[30, 20, 10],
        smooth_sigmas=[2.0, 1.0, 0.5],
        shrink_factors=[4, 2, 1],
    )

    # bigstream affines map fixed→moving with negative translation
    # (affine_align returns ~-4 for a +4 voxel shift)
    affine = np.eye(4)
    affine[:3, 3] = -shift[::-1]  # XYZ, negative to map fixed→moving

    _, field_with_static = demons_align(
        fix, mov, SPACING, SPACING,
        static_transform_list=[affine],
        **demons_kwargs,
    )
    _, field_no_static = demons_align(fix, mov,
                                      SPACING, SPACING,
                                      **demons_kwargs)

    assert field_with_static.shape == SHAPE + (3,)

    # pre-alignment reduces the residual demons needs to find
    mean_with = np.mean(np.abs(field_with_static))
    mean_without = np.mean(np.abs(field_no_static))
    assert mean_with < mean_without, (
        f"static_transform_list pre-alignment should reduce residual: "
        f"mean_with={mean_with:.3f}, mean_without={mean_without:.3f}"
    )


# ---------------------------------------------------------------------------
# 6. alignment_spacing round-trip test
# ---------------------------------------------------------------------------

def test_alignment_spacing_roundtrip():
    """Returned field shape must equal fix.shape + (ndim,) for any alignment_spacing."""
    fix = _random_smooth_volume(SHAPE)
    mov = fix.copy()

    for spacing in [None, 2.0, 4.0]:
        _, field = demons_align(
            fix, mov, SPACING, SPACING,
            alignment_spacing=spacing,
            iterations=[10],
            smooth_sigmas=[1.0],
            shrink_factors=[1],
        )
        assert field.shape == SHAPE + (3,), (
            f"alignment_spacing={spacing}: expected {SHAPE + (3,)}, got {field.shape}"
        )


# ---------------------------------------------------------------------------
# 7. Pipeline composition test
# ---------------------------------------------------------------------------

def test_pipeline_with_demons_step():
    """alignment_pipeline with a 'demons' step should return a displacement field."""
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
            ('demons', dict(
                iterations=[30, 20],
                smooth_sigmas=[1.0, 0.5],
                shrink_factors=[2, 1],
            )),
        ],
        return_format='flatten',
    )
    assert isinstance(result, np.ndarray)
    assert result.shape == SHAPE + (3,)


def test_pipeline_default_case_demons():
    """alignment_pipeline default case with 'demons' in steps returns zero field."""
    result = alignment_pipeline(
        None, np.zeros(SHAPE, dtype=np.float32), SPACING, SPACING,
        steps=[('demons', {
            'iterations': [10],
            'smooth_sigmas': [1.0],
            'shrink_factors': [1],
        })],
        return_format='flatten',
    )
    assert isinstance(result, np.ndarray)
    assert result.shape == SHAPE + (3,)
    assert np.all(result == 0.0)


# ---------------------------------------------------------------------------
# 8. Diffeomorphic invertibility test
# ---------------------------------------------------------------------------

def test_diffeomorphic_positive_jacobian():
    """Jacobian determinant of the diffeomorphic field should be positive in the interior."""
    fix = _random_smooth_volume(SHAPE)
    shift = np.array([2.0, 2.0, 2.0])
    mov = _apply_translation_sitk(fix, SPACING, shift)

    _, field = demons_align(
        fix, mov, SPACING, SPACING,
        variant='diffeomorphic',
        iterations=[50, 30],
        smooth_sigmas=[2.0, 1.0],
        shrink_factors=[2, 1],
    )
    # field: (Z, Y, X, 3), components in ZYX order (bigstream convention)
    # approximate Jacobian determinant via 1 + divergence of displacement
    dz = np.gradient(field[..., 0], axis=0)
    dy = np.gradient(field[..., 1], axis=1)
    dx = np.gradient(field[..., 2], axis=2)
    jacobian_approx = 1.0 + dz + dy + dx
    interior = jacobian_approx[2:-2, 2:-2, 2:-2]
    assert np.min(interior) > 0.0, (
        f"Jacobian has non-positive values in interior: min={np.min(interior):.4f}"
    )
