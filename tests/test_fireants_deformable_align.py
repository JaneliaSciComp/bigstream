import sys

import numpy as np
import pytest
import SimpleITK as sitk
from scipy import ndimage

torch = pytest.importorskip('torch')
pytest.importorskip('fireants')

import fireants.registration.greedy as fa_greedy

import bigstream.transform as bst
from bigstream.align import alignment_pipeline
from bigstream.contrib.fireants_align import (
    fireants_deformable_align,
    _resolve_device,
    _INTENSITY_RANGE,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

SHAPE = (32, 32, 32)
SPACING = np.array([1.0, 1.0, 1.0])
# FireANTs never downsamples a level below 32 voxels, so on these volumes
# every level runs at full resolution; two short levels keep the suite fast
FAST = dict(scales=(2, 1), iterations=(60, 30), device='cpu')
INTERIOR = (slice(8, -8),) * 3


def _smooth_volume(shape=SHAPE, sigma=2.0, seed=42):
    rng = np.random.default_rng(seed)
    vol = ndimage.gaussian_filter(rng.random(shape), sigma)
    vol = (vol - vol.min()) / (vol.max() - vol.min())
    return vol.astype(np.float32)


def _shift(image, shift_voxels):
    """out[i] = image[i - shift]: aligning out to image needs d = +shift"""
    return ndimage.shift(image, shift_voxels, order=1, mode='nearest').astype(np.float32)


def _translation_matrix(shift_physical):
    """bigstream affine (fix coordinates -> mov coordinates) for a pure shift"""
    matrix = np.eye(4)
    matrix[:3, -1] = shift_physical
    return matrix


@pytest.fixture
def captured_registrations(monkeypatch):
    """Keep every GreedyRegistration fireants_deformable_align builds, to inspect after"""
    captured = []

    class _Capturing(fa_greedy.GreedyRegistration):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            captured.append(self)

    monkeypatch.setattr(fa_greedy, 'GreedyRegistration', _Capturing)
    return captured


def _assert_translation(field, expected_physical, spacing, tol_voxels=0.5):
    recovered = np.median(field[INTERIOR].reshape(-1, 3), axis=0)
    tol = tol_voxels * np.asarray(spacing)
    assert np.all(np.abs(recovered - expected_physical) < tol), (
        f'recovered {recovered}, expected {expected_physical}'
    )


# ---------------------------------------------------------------------------
# 1. Identity
# ---------------------------------------------------------------------------

def _identity_field(device):
    fix = _smooth_volume()
    _, field = fireants_deformable_align(
        fix, fix.copy(), SPACING, SPACING,
        **{**FAST, 'device': device}, final_metric_check=False,
    )
    return field


def test_identity_returns_near_zero_field():
    field = _identity_field('cpu')
    assert field.shape == SHAPE + (3,)
    assert field.dtype == np.float32
    assert np.max(np.abs(field)) < 0.1


# ---------------------------------------------------------------------------
# 2. Translation recovery
# ---------------------------------------------------------------------------

SHIFT = np.array([2.0, 1.0, 3.0])


def _translation_field(device, method='greedy', loss='cc'):
    fix = _smooth_volume()
    mov = _shift(fix, SHIFT)
    params, field = fireants_deformable_align(
        fix, mov, SPACING, SPACING,
        **{**FAST, 'device': device}, method=method, loss=loss,
    )
    return params, field


def test_translation_recovery():
    params, field = _translation_field('cpu')
    assert field.shape == SHAPE + (3,)
    np.testing.assert_array_equal(params, field.ravel())
    _assert_translation(field, SHIFT * SPACING, SPACING)


# ---------------------------------------------------------------------------
# 3. Conversion matches FireANTs' own save_as_ants_transforms
# ---------------------------------------------------------------------------

def test_conversion_matches_save_as_ants_transforms(captured_registrations, tmp_path):
    fix = _smooth_volume()
    mov = _shift(fix, SHIFT)
    _, field = fireants_deformable_align(
        fix, mov, SPACING, SPACING, **FAST, final_metric_check=False,
    )
    reg = captured_registrations[-1]
    path = str(tmp_path / 'warp.nii.gz')
    reg.save_as_ants_transforms(path)
    ants_field = sitk.GetArrayFromImage(sitk.ReadImage(path))[..., ::-1]
    np.testing.assert_allclose(field, ants_field, atol=1e-4)


# ---------------------------------------------------------------------------
# 4. Units and sign: bigstream's own warp reproduces FireANTs' evaluate
# ---------------------------------------------------------------------------

def test_apply_transform_matches_evaluate(captured_registrations):
    fix = _smooth_volume()
    mov = _shift(fix, SHIFT)
    _, field = fireants_deformable_align(
        fix, mov, SPACING, SPACING, **FAST,
        clip_percentiles=None, final_metric_check=False,
    )
    reg = captured_registrations[-1]
    with torch.no_grad():
        evaluated = reg.evaluate(reg.fixed_images, reg.moving_images)[0, 0].numpy()
    # FireANTs sees mov mapped linearly onto [0, _INTENSITY_RANGE]
    evaluated = mov.min() + evaluated * (mov.max() - mov.min()) / _INTENSITY_RANGE
    warped = bst.apply_transform(fix, mov, SPACING, SPACING, transform_list=[field])
    np.testing.assert_allclose(warped[INTERIOR], evaluated[INTERIOR], atol=1e-3)


# ---------------------------------------------------------------------------
# 5. Anisotropic spacing
# ---------------------------------------------------------------------------

ANISO_SPACING = np.array([2.0, 0.5, 0.5])
ANISO_SHIFT = np.array([1.0, 2.0, 3.0])   # voxels


def _anisotropic_field(device):
    fix = _smooth_volume()
    mov = _shift(fix, ANISO_SHIFT)
    _, field = fireants_deformable_align(
        fix, mov, ANISO_SPACING, ANISO_SPACING, **{**FAST, 'device': device},
    )
    return field


def test_anisotropic_spacing_physical_units():
    field = _anisotropic_field('cpu')
    _assert_translation(field, ANISO_SHIFT * ANISO_SPACING, ANISO_SPACING)


# ---------------------------------------------------------------------------
# 6. alignment_spacing round trip
# ---------------------------------------------------------------------------

def test_alignment_spacing_returns_original_grid():
    fix = _smooth_volume((48, 40, 36))
    mov = _shift(fix, SHIFT)
    _, field = fireants_deformable_align(
        fix, mov, SPACING, SPACING, **FAST, alignment_spacing=2.0,
    )
    assert field.shape == fix.shape + (3,)


@pytest.mark.parametrize('shape, recovered', [
    ((31, 40, 40), 0.85),   # one axis just short of FireANTs' 32-voxel minimum
    ((17, 17, 17), 0.6),    # every axis short: less image, a weaker fit
])
def test_small_fixed_grid_is_padded(shape, recovered):
    # FireANTs fails on fixed grids under 32 voxels per axis; the step pads
    # them with zeros instead of returning the zero default
    fix = _smooth_volume(shape)
    mov = _shift(fix, np.ones(3))
    _, field = fireants_deformable_align(
        fix, mov, SPACING, SPACING, scales=(4, 2, 1), iterations=(30, 20, 10),
        device='cpu', final_metric_check=False,
    )
    assert field.shape == shape + (3,)
    interior = tuple(slice(4, -4) for _ in shape)
    median = np.median(field[interior].reshape(-1, 3), axis=0)
    assert np.all(median > recovered) and np.all(median < 1.15), median


def test_alignment_spacing_extrapolates_trailing_voxels():
    # z spacing 1 skip-sampled to 4 keeps z indices 0..132 of 136: voxel 135
    # lies 0.75 of a working voxel past the grid and must get the edge
    # displacement (without extrapolation it would get 0). y and x keep their
    # 36 voxels, so the working grid (34, 36, 36) needs no padding.
    spacing = np.array([1.0, 4.0, 4.0])
    fix = _smooth_volume((136, 36, 36), sigma=4.0)
    mov = _shift(fix, np.array([4.0, 0.25, 0.25]))
    _, field = fireants_deformable_align(
        fix, mov, spacing, spacing, **FAST, alignment_spacing=4.0,
        final_metric_check=False,
    )
    assert np.abs(field).max() > 0.5   # a real fit, not the zero default
    np.testing.assert_allclose(field[135], field[132], atol=1e-5)


@pytest.mark.parametrize('final_metric_check', [True, False])
def test_constant_fixed_block_returns_zero_field(final_metric_check):
    mov = _smooth_volume()
    fix = np.zeros(SHAPE, dtype=np.float32)
    _, field = fireants_deformable_align(
        fix, mov, SPACING, SPACING, **FAST, final_metric_check=final_metric_check,
    )
    assert not field.any()


# ---------------------------------------------------------------------------
# 7. static_transform_list
# ---------------------------------------------------------------------------

def test_static_transform_list_gives_residual():
    fix = _smooth_volume()
    mov = _shift(fix, SHIFT)
    pre_shift = np.array([3.0, 2.0, 1.0])
    pre_translated = _shift(mov, pre_shift)

    _, reference = fireants_deformable_align(fix, mov, SPACING, SPACING, **FAST)
    _, residual = fireants_deformable_align(
        fix, pre_translated, SPACING, SPACING, **FAST,
        static_transform_list=[_translation_matrix(pre_shift * SPACING)],
    )
    np.testing.assert_allclose(
        np.median(residual[INTERIOR].reshape(-1, 3), axis=0),
        np.median(reference[INTERIOR].reshape(-1, 3), axis=0),
        atol=0.5,
    )


# ---------------------------------------------------------------------------
# 8. Masks
# ---------------------------------------------------------------------------

def _ball(shape=SHAPE, radius=12):
    grid = np.indices(shape) - (np.array(shape) / 2)[:, None, None, None]
    return (np.sqrt((grid ** 2).sum(axis=0)) < radius).astype(np.uint8)


@pytest.mark.parametrize('which', ['both', 'fix_only', 'mov_only'])
def test_masks(which):
    fix = _smooth_volume()
    mov = _shift(fix, SHIFT)
    masks = {
        'both': dict(fix_mask=_ball(), mov_mask=_ball()),
        'fix_only': dict(fix_mask=_ball()),
        'mov_only': dict(mov_mask=_ball()),
    }[which]
    _, field = fireants_deformable_align(fix, mov, SPACING, SPACING, **FAST, **masks)
    assert field.shape == SHAPE + (3,)
    _assert_translation(field, SHIFT * SPACING, SPACING)


# ---------------------------------------------------------------------------
# 9. Both methods
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('method', ['greedy', 'syn'])
def test_methods_recover_translation(method):
    _, field = _translation_field('cpu', method=method)
    _assert_translation(field, SHIFT * SPACING, SPACING)


def test_invalid_arguments_raise():
    fix = _smooth_volume()
    with pytest.raises(ValueError, match='equal length'):
        fireants_deformable_align(fix, fix, SPACING, SPACING, scales=(2, 1), iterations=(10,))
    with pytest.raises(ValueError, match='decreasing'):
        fireants_deformable_align(fix, fix, SPACING, SPACING, scales=(1, 2), iterations=(10, 10))


# ---------------------------------------------------------------------------
# 10. final_metric_check
# ---------------------------------------------------------------------------

def test_final_metric_check_returns_default(monkeypatch):
    def _bad_evaluate(self, fixed_images, moving_images, shape=None):
        return torch.rand_like(fixed_images())

    monkeypatch.setattr(fa_greedy.GreedyRegistration, 'evaluate', _bad_evaluate)
    fix = _smooth_volume()
    mov = _shift(fix, SHIFT)
    sentinel = object()
    result = fireants_deformable_align(fix, mov, SPACING, SPACING, **FAST, default=sentinel)
    assert result is sentinel


# ---------------------------------------------------------------------------
# 11. Failure paths
# ---------------------------------------------------------------------------

def test_exception_in_optimize_returns_default(monkeypatch):
    def _fail(self):
        raise RuntimeError('out of memory')

    monkeypatch.setattr(fa_greedy.GreedyRegistration, 'optimize', _fail)
    fix = _smooth_volume()
    _, field = fireants_deformable_align(fix, fix.copy(), SPACING, SPACING, **FAST)
    assert field.shape == SHAPE + (3,)
    assert not field.any()


def test_not_implemented_error_is_raised(monkeypatch):
    def _missing_op(self):
        raise NotImplementedError('op missing on this device')

    monkeypatch.setattr(fa_greedy.GreedyRegistration, 'optimize', _missing_op)
    fix = _smooth_volume()
    with pytest.raises(NotImplementedError):
        fireants_deformable_align(fix, fix.copy(), SPACING, SPACING, **FAST)


# ---------------------------------------------------------------------------
# 12. Devices
# ---------------------------------------------------------------------------

ACCELERATORS = [
    pytest.param('mps', marks=pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason='MPS not available')),
    pytest.param('cuda', marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason='CUDA not available')),
]


def test_auto_device_resolves_to_available():
    available = {'cpu'}
    if torch.cuda.is_available():
        available.add('cuda')
    if torch.backends.mps.is_available():
        available.add('mps')
    assert _resolve_device('auto') in available
    assert _resolve_device('cpu') == 'cpu'


@pytest.mark.parametrize('device', ACCELERATORS)
def test_accelerator_matches_cpu(device):
    # two optimizations on different devices accumulate float32 rounding
    # differently over ~90 iterations, so agreement is checked well below a
    # voxel rather than at machine precision
    tol = 0.05
    np.testing.assert_allclose(_identity_field(device), _identity_field('cpu'), atol=tol)
    np.testing.assert_allclose(
        _translation_field(device)[1][INTERIOR],
        _translation_field('cpu')[1][INTERIOR],
        atol=tol,
    )
    np.testing.assert_allclose(
        _anisotropic_field(device)[INTERIOR],
        _anisotropic_field('cpu')[INTERIOR],
        atol=tol * ANISO_SPACING.max(),
    )


@pytest.mark.parametrize('device', ACCELERATORS)
def test_accelerator_mi_loss(device):
    _, field = _translation_field(device, loss='mi')
    _assert_translation(field, SHIFT * SPACING, SPACING)


# ---------------------------------------------------------------------------
# 13. Pipeline
# ---------------------------------------------------------------------------

def test_alignment_pipeline_with_fireants_step():
    fix = _smooth_volume()
    mov = _shift(fix, SHIFT)
    result = alignment_pipeline(
        fix, mov, SPACING, SPACING,
        steps=[
            ('affine', dict(
                metric='MS',
                shrink_factors=(2, 1),
                smooth_sigmas=(1.0, 0.0),
            )),
            ('fireants_deform', FAST),
        ],
        return_format='flatten',
    )
    assert result.shape == SHAPE + (3,)
    _assert_translation(result, SHIFT * SPACING, SPACING)


def test_alignment_pipeline_default_with_fireants_step():
    result = alignment_pipeline(
        None, np.zeros(SHAPE, dtype=np.float32), SPACING, SPACING,
        steps=[('fireants_deform', {})],
    )
    assert result.shape == SHAPE + (3,)
    assert not result.any()


def test_alignment_pipeline_without_fireants_installed(monkeypatch):
    # a None entry in sys.modules makes `import fireants` raise ImportError
    for name in [m for m in sys.modules if m == 'fireants' or m.startswith('fireants.')]:
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setitem(sys.modules, 'fireants', None)

    fix = _smooth_volume()
    result = alignment_pipeline(
        fix, fix.copy(), SPACING, SPACING,
        steps=[('affine', dict(
            metric='MS',
            shrink_factors=(2, 1),
            smooth_sigmas=(1.0, 0.0),
        ))],
    )
    assert result.shape == (4, 4)
    with pytest.raises(ImportError):
        alignment_pipeline(
            fix, fix.copy(), SPACING, SPACING, steps=[('fireants_deform', FAST)],
        )
