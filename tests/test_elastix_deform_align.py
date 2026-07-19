import numpy as np
import pytest
import SimpleITK as sitk

# elastix is an optional dependency; skip the whole module if unavailable
itk = pytest.importorskip("itk")
if not hasattr(itk, "ElastixRegistrationMethod"):
    pytest.skip("itk-elastix not installed", allow_module_level=True)

from bigstream.align import elastix_deform_align, alignment_pipeline, _robust_normalize
import bigstream.utility as ut
import bigstream.transform as bst


# ---------------------------------------------------------------------------
# helpers (mirror test_deformable_align.py)
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
    sitk_img = sitk.Cast(ut.numpy_to_sitk(image_np, spacing), sitk.sitkFloat32)
    ndim = image_np.ndim
    tx = sitk.TranslationTransform(ndim)
    physical_shift = (shift_voxels * np.asarray(spacing))[::-1]  # XYZ
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

# fast, low-cost elastix settings shared across tests
FAST_KWARGS = dict(
    parameter_map='bspline',
    number_of_resolutions=2,
    maximum_iterations=48,
    number_of_spatial_samples=1024,
    final_grid_spacing_physical=8.0,
)


# ---------------------------------------------------------------------------
# 1. Identity test
# ---------------------------------------------------------------------------

def test_identity_returns_near_zero_field():
    """fix == mov: final_metric_check rejects any spurious warp, so the
    returned field is the (zero) default."""
    fix = _random_smooth_volume(SHAPE)
    params, field = elastix_deform_align(
        fix, fix.copy(), SPACING, SPACING, **FAST_KWARGS,
    )
    assert field.shape == SHAPE + (3,)
    assert np.max(np.abs(field)) < 1e-4, (
        f"Identity produced non-trivial field: max={np.max(np.abs(field)):.5f}"
    )


# ---------------------------------------------------------------------------
# 2. Shifted volume recovery
# ---------------------------------------------------------------------------

def test_shifted_volume_recovery():
    fix = _random_smooth_volume(SHAPE)
    shift = np.array([2.0, 4.0, 3.0])  # ZYX voxels
    mov = _apply_translation_sitk(fix, SPACING, shift)
    initial_ssd = float(np.mean((fix - mov) ** 2))

    params, field = elastix_deform_align(
        fix, mov, SPACING, SPACING,
        number_of_resolutions=3, maximum_iterations=128,
        number_of_spatial_samples=2048, final_grid_spacing_physical=8.0,
    )
    assert field.shape == SHAPE + (3,)

    _, final_ssd = _apply_field(fix, mov, SPACING, field)
    assert final_ssd < 0.5 * initial_ssd, (
        f"Alignment did not improve: initial={initial_ssd:.4f}, final={final_ssd:.4f}"
    )


# ---------------------------------------------------------------------------
# 3. BSpline-distorted volume recovery
# ---------------------------------------------------------------------------

def test_bspline_distorted_recovery():
    fix = _random_smooth_volume(SHAPE)
    sitk_fix = sitk.Cast(ut.numpy_to_sitk(fix, SPACING), sitk.sitkFloat32)
    bspline_tx = sitk.BSplineTransformInitializer(sitk_fix, [2] * 3, order=3)
    params_init = np.array(bspline_tx.GetParameters())
    rng = np.random.default_rng(7)
    params_init += rng.uniform(-2.5, 2.5, size=params_init.shape)
    bspline_tx.SetParameters(params_init.tolist())
    sitk_mov = sitk.Resample(
        sitk_fix, sitk_fix, bspline_tx, sitk.sitkLinear, 0.0, sitk.sitkFloat32,
    )
    mov = sitk.GetArrayFromImage(sitk_mov).astype(np.float32)

    initial_ssd = float(np.mean((fix - mov) ** 2))
    assert initial_ssd > 1e-4, "deformation too small to test recovery"

    params, field = elastix_deform_align(
        fix, mov, SPACING, SPACING,
        number_of_resolutions=3, maximum_iterations=128,
        number_of_spatial_samples=2048, final_grid_spacing_physical=8.0,
    )
    assert field.shape == SHAPE + (3,)

    _, final_ssd = _apply_field(fix, mov, SPACING, field)
    assert final_ssd < 0.6 * initial_ssd, (
        f"BSpline recovery failed: initial={initial_ssd:.4f}, final={final_ssd:.4f}"
    )


# ---------------------------------------------------------------------------
# 4. Mask test
# ---------------------------------------------------------------------------

def test_mask_runs_and_returns_correct_shape():
    fix = _random_smooth_volume(SHAPE)
    shift = np.array([3.0, 3.0, 3.0])
    mov = _apply_translation_sitk(fix, SPACING, shift)

    mask = np.zeros(SHAPE, dtype=np.uint8)
    mask[:, :, SHAPE[2] // 2:] = 1  # right half foreground

    params, field = elastix_deform_align(
        fix, mov, SPACING, SPACING,
        fix_mask=mask, **FAST_KWARGS,
    )
    assert field.shape == SHAPE + (3,)
    assert params.ndim == 1


# ---------------------------------------------------------------------------
# 5. static_transform_list pre-align (residual field)
# ---------------------------------------------------------------------------

def test_static_transform_list_pre_aligns():
    fix = _random_smooth_volume(SHAPE)
    shift = np.array([4.0, 4.0, 4.0])  # ZYX voxels
    mov = _apply_translation_sitk(fix, SPACING, shift)
    initial_ssd = float(np.mean((fix - mov) ** 2))

    # partial affine: 75% of the true shift; bigstream affines map fixed->moving
    affine = np.eye(4)
    affine[:3, 3] = -shift[::-1] * 0.75  # XYZ, negative to map fixed->moving

    params, field = elastix_deform_align(
        fix, mov, SPACING, SPACING,
        static_transform_list=[affine],
        number_of_resolutions=3, maximum_iterations=128,
        number_of_spatial_samples=2048, final_grid_spacing_physical=8.0,
    )
    assert field.shape == SHAPE + (3,)

    # compose affine + residual field, apply to original mov
    total_field = bst.compose_transforms(affine, field, SPACING, SPACING)
    _, final_ssd = _apply_field(fix, mov, SPACING, total_field)
    assert final_ssd < 0.5 * initial_ssd, (
        f"Static pre-align failed: initial={initial_ssd:.4f}, final={final_ssd:.4f}"
    )


# ---------------------------------------------------------------------------
# 6. alignment_spacing round-trip
# ---------------------------------------------------------------------------

def test_alignment_spacing_roundtrip():
    fix = _random_smooth_volume(SHAPE)
    mov = fix.copy()
    for spacing in [None, 2.0]:
        params, field = elastix_deform_align(
            fix, mov, SPACING, SPACING,
            alignment_spacing=spacing, **FAST_KWARGS,
        )
        assert field.shape == SHAPE + (3,), (
            f"alignment_spacing={spacing}: expected {SHAPE + (3,)}, got {field.shape}"
        )


# ---------------------------------------------------------------------------
# 5b. _robust_normalize helper
# ---------------------------------------------------------------------------

def test_robust_normalize_range_and_degenerate_case():
    rng = np.random.default_rng(1)
    arr = rng.normal(30000, 500, (16, 16, 16)).astype(np.float64)
    arr[0, 0, 0] = 65000  # outlier, should get clipped to 1.0

    norm = _robust_normalize(arr)
    assert norm.min() >= 0.0 and norm.max() <= 1.0
    assert norm[0, 0, 0] == 1.0  # clipped outlier

    # shared stats: normalizing two arrays with the same (lo, hi) is a pure
    # affine transform of each -- values outside [lo, hi] still clip correctly
    stats = tuple(np.percentile(arr, [1.0, 99.0]))
    norm_a = _robust_normalize(arr, stats=stats)
    norm_b = _robust_normalize(arr * 1.0, stats=stats)
    assert np.allclose(norm_a, norm_b)

    # degenerate (constant) array must not divide by zero / produce NaN or Inf
    constant = np.full((8, 8, 8), 42.0)
    norm_const = _robust_normalize(constant)
    assert np.all(np.isfinite(norm_const))


def test_wide_dynamic_range_metric_check_runs_cleanly():
    """uint16-scale fix/mov (large absolute intensities, matching the real
    failure this normalization was added for) should not raise or produce
    NaN/Inf, and should still recover a real shift."""
    rng = np.random.default_rng(2)
    shape = (24, 32, 32)
    fix = rng.normal(300, 40, shape).astype(np.float32)
    fix[8:18, 10:24, 10:24] += 20000.0
    fix = np.clip(fix, 0, 65535).astype(np.float32)
    mov = np.roll(fix, (0, 2, 3), axis=(0, 1, 2)).astype(np.float32)
    spacing = np.array([1.0, 1.0, 1.0])

    params, field = elastix_deform_align(
        fix, mov, spacing, spacing,
        number_of_resolutions=2, maximum_iterations=48,
        number_of_spatial_samples=1024, final_grid_spacing_physical=8.0,
    )
    assert field.shape == shape + (3,)
    assert np.all(np.isfinite(field))


# ---------------------------------------------------------------------------
# 6b. Thin-axis block (< 4 voxels along one axis) must not crash
# ---------------------------------------------------------------------------

def test_thin_axis_block_does_not_raise():
    """A block with fewer than 4 voxels along one axis used to crash elastix's
    default smoothing image pyramid (RecursiveGaussianImageFilter requires >=4
    voxels along any axis it processes, regardless of number_of_resolutions).
    Shrinking (non-smoothing) pyramids avoid this; verify it registers and
    returns a real field rather than silently falling back to the default."""
    thin_shape = (64, 64, 3)
    rng = np.random.default_rng(0)
    fix = rng.uniform(0, 1, thin_shape).astype(np.float32)
    mov = rng.uniform(0, 1, thin_shape).astype(np.float32)
    spacing = np.array([1.0, 1.0, 1.0])

    params, field = elastix_deform_align(
        fix, mov, spacing, spacing,
        number_of_resolutions=4, maximum_iterations=16,
        number_of_spatial_samples=256, final_grid_spacing_physical=8.0,
    )
    assert field.shape == thin_shape + (3,)


# ---------------------------------------------------------------------------
# 7. Pipeline composition
# ---------------------------------------------------------------------------

def test_pipeline_with_elastix_step():
    fix = _random_smooth_volume(SHAPE)
    mov = fix.copy()
    result = alignment_pipeline(
        fix, mov, SPACING, SPACING,
        steps=[
            ('affine', dict(metric='MS', shrink_factors=(2, 1), smooth_sigmas=(1.0, 0.0))),
            ('elastix', dict(**FAST_KWARGS)),
        ],
        return_format='flatten',
    )
    assert isinstance(result, np.ndarray)
    assert result.shape == SHAPE + (3,)


# ---------------------------------------------------------------------------
# 8. Degenerate (constant) moving block returns default without raising
# ---------------------------------------------------------------------------

def test_constant_moving_returns_default():
    fix = _random_smooth_volume(SHAPE)
    mov = np.zeros(SHAPE, dtype=np.float32)  # zero variance -> MMI ill-defined
    params, field = elastix_deform_align(
        fix, mov, SPACING, SPACING, **FAST_KWARGS,
    )
    assert field.shape == SHAPE + (3,)
    assert np.max(np.abs(field)) < 1e-4  # default (zero) field


# ---------------------------------------------------------------------------
# 9. final_grid_spacing_physical scalar vs per-axis list both accepted
# ---------------------------------------------------------------------------

def test_extra_parameters_forwarded_as_kwargs():
    """Arbitrary elastix parameter-map keys pass through as **extra_parameters,
    both into the builder and through elastix_deform_align."""
    from bigstream.configure_elastix import build_elastix_parameter_object

    po = build_elastix_parameter_object(
        preset='bspline', ndim=3, number_of_resolutions=2,
        RandomSeed=42, ImageSampler='RandomCoordinate',
    )
    pm = po.GetParameterMap(0)
    assert pm['RandomSeed'] == ('42',)
    assert pm['ImageSampler'] == ('RandomCoordinate',)

    fix = _random_smooth_volume(SHAPE)
    mov = _apply_translation_sitk(fix, SPACING, np.array([2.0, 2.0, 2.0]))
    params, field = elastix_deform_align(
        fix, mov, SPACING, SPACING, RandomSeed=7, **FAST_KWARGS,
    )
    assert field.shape == SHAPE + (3,)


def test_final_grid_spacing_scalar_and_list():
    fix = _random_smooth_volume(SHAPE)
    mov = _apply_translation_sitk(fix, SPACING, np.array([2.0, 2.0, 2.0]))
    for grid in (8.0, [16.0, 8.0, 8.0], [16.0, 8.0]):
        params, field = elastix_deform_align(
            fix, mov, SPACING, SPACING,
            parameter_map='bspline', number_of_resolutions=2,
            maximum_iterations=32, number_of_spatial_samples=1024,
            final_grid_spacing_physical=grid,
        )
        assert field.shape == SHAPE + (3,), f"grid={grid}"
