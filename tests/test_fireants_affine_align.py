import itertools
import sys

import numpy as np
import pytest
from scipy import ndimage

torch = pytest.importorskip('torch')
pytest.importorskip('fireants')

import fireants.registration.affine as fa_affine

import bigstream.transform as bst
from bigstream.align import alignment_pipeline
from bigstream.contrib.fireants_align import fireants_affine_align, _INTENSITY_RANGE


# ---------------------------------------------------------------------------
# helpers
#
# The fixed image is the centre of a larger volume and the moving image is
# that whole volume, transformed. So under the true transform every fixed
# voxel samples inside the moving image, as a block with its halo does in a
# blockwise run. As there, the fixed origin is 0 and the moving origin is
# shifted (distributed_align passes only mov_origin). When mov is cropped to fix's field of view instead,
# FireANTs samples zeros past mov's edge and the fitted affine is biased
# toward shrinking.
#
# Results are compared by the displacement they produce over the fixed
# interior, not entry by entry: FireANTs optimizes around the image centre,
# so a small error in scale shows up as a large translation in an
# origin-based matrix while moving no voxel much.
# ---------------------------------------------------------------------------

FIX_SHAPE = (48, 48, 48)
PAD = 16
SPACING = np.array([1.0, 1.0, 1.0])
# FireANTs never downsamples a level below 32 voxels, so 48^3 gives a real
# (if small) pyramid
FAST = dict(scales=(2, 1), iterations=(200, 100), device='cpu')
SHIFT = np.array([2.0, 1.0, 3.0])   # voxels, zyx

KNOWN_AFFINE = np.array([
    [1.06, 0.03, 0.00, 1.5],
    [0.00, 0.97, 0.04, -1.0],
    [0.02, 0.00, 1.02, 2.0],
    [0.00, 0.00, 0.00, 1.0],
])


def _smooth_volume(shape, sigma=2.0, seed=42):
    rng = np.random.default_rng(seed)
    vol = ndimage.gaussian_filter(rng.random(shape), sigma)
    vol = (vol - vol.min()) / (vol.max() - vol.min())
    return vol.astype(np.float32)


def _translation_matrix(shift_physical):
    matrix = np.eye(4)
    matrix[:3, -1] = shift_physical
    return matrix


def _pair(truth, spacing=SPACING, fix_shape=FIX_SHAPE, pad=PAD, sigma=2.0):
    """
    fix (the centre of a larger volume, origin 0) and mov (the whole volume,
    origin -pad * spacing) with mov(truth x) = fix(x), so aligning mov to fix
    gives truth. Returns fix, mov and mov's origin.
    """
    big = _smooth_volume(tuple(s + 2 * pad for s in fix_shape), sigma=sigma)
    mov_origin = -pad * np.asarray(spacing, dtype=float)
    mov = bst.apply_transform(
        big, big, spacing, spacing, transform_list=[bst.invert_affine(truth)],
        fix_origin=mov_origin, mov_origin=mov_origin,
    ).astype(np.float32)
    fix = big[pad:-pad, pad:-pad, pad:-pad]
    return fix, mov, mov_origin


def _align(fix, mov, spacing, mov_origin, **kwargs):
    return fireants_affine_align(
        fix, mov, spacing, spacing, mov_origin=mov_origin,
        **{**FAST, **kwargs},
    )


def _displacement_error(affine, expected, fix_shape=FIX_SHAPE, spacing=SPACING, margin=5):
    """Largest |affine x - expected x| over the fixed interior, in voxels"""
    corners = itertools.product(*[(margin, s - 1 - margin) for s in fix_shape])
    points = np.array(list(corners)) * spacing
    points = np.c_[points, np.ones(len(points))]
    diff = (points @ (affine - expected).T)[:, :3]
    return np.max(np.abs(diff) / spacing)


@pytest.fixture
def captured_registrations(monkeypatch):
    """Keep every AffineRegistration fireants_affine_align builds"""
    captured = []

    class _Capturing(fa_affine.AffineRegistration):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            captured.append(self)

    monkeypatch.setattr(fa_affine, 'AffineRegistration', _Capturing)
    return captured


# ---------------------------------------------------------------------------
# identity, translation, a general affine
# ---------------------------------------------------------------------------

def _identity_affine(device):
    fix, mov, mov_origin = _pair(np.eye(4))
    return _align(fix, mov, SPACING, mov_origin, device=device, final_metric_check=False)


def test_identity_returns_near_identity():
    affine = _identity_affine('cpu')
    assert affine.shape == (4, 4)
    assert _displacement_error(affine, np.eye(4)) < 0.1


def _translation_affine(device, method='affine', loss='cc'):
    truth = _translation_matrix(SHIFT * SPACING)
    fix, mov, mov_origin = _pair(truth)
    return _align(fix, mov, SPACING, mov_origin, device=device, method=method, loss=loss)


@pytest.mark.parametrize('method', ['affine', 'rigid'])
def test_translation_recovery(method):
    affine = _translation_affine('cpu', method=method)
    assert _displacement_error(affine, _translation_matrix(SHIFT * SPACING)) < 0.25


def test_general_affine_recovery():
    # different scale per axis, shear and translation: a wrong axis order,
    # transpose or direction in the matrix conversion cannot pass this
    fix, mov, mov_origin = _pair(KNOWN_AFFINE)
    affine = _align(fix, mov, SPACING, mov_origin)
    assert _displacement_error(affine, KNOWN_AFFINE) < 0.5


def test_rigid_returns_rotation():
    linear = _translation_affine('cpu', method='rigid')[:3, :3]
    np.testing.assert_allclose(linear @ linear.T, np.eye(3), atol=1e-4)
    assert np.linalg.det(linear) == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# units and sign: bigstream's own warp reproduces FireANTs' evaluate
# ---------------------------------------------------------------------------

def test_apply_transform_matches_evaluate(captured_registrations):
    fix, mov, mov_origin = _pair(KNOWN_AFFINE)
    affine = _align(fix, mov, SPACING, mov_origin, clip_percentiles=None, final_metric_check=False)
    reg = captured_registrations[-1]
    with torch.no_grad():
        evaluated = reg.evaluate(reg.fixed_images, reg.moving_images)[0, 0].numpy()
    # FireANTs sees mov mapped linearly onto [0, _INTENSITY_RANGE]
    evaluated = mov.min() + evaluated * (mov.max() - mov.min()) / _INTENSITY_RANGE
    warped = bst.apply_transform(
        fix, mov, SPACING, SPACING, transform_list=[affine],
        mov_origin=mov_origin,
    )
    interior = (slice(5, -5),) * 3
    np.testing.assert_allclose(warped[interior], evaluated[interior], atol=1e-3)


# ---------------------------------------------------------------------------
# anisotropic spacing and alignment_spacing
# ---------------------------------------------------------------------------

ANISO_SPACING = np.array([2.0, 0.5, 0.5])
ANISO_SHIFT = np.array([1.0, 2.0, 3.0])   # voxels


def _anisotropic_affine(device):
    truth = _translation_matrix(ANISO_SHIFT * ANISO_SPACING)
    fix, mov, mov_origin = _pair(truth, spacing=ANISO_SPACING)
    return _align(fix, mov, ANISO_SPACING, mov_origin, device=device)


def test_anisotropic_spacing_physical_units():
    error = _displacement_error(
        _anisotropic_affine('cpu'), _translation_matrix(ANISO_SHIFT * ANISO_SPACING),
        spacing=ANISO_SPACING,
    )
    assert error < 0.25


def test_alignment_spacing():
    shape, pad = (96, 96, 96), 32
    truth = _translation_matrix(2 * SHIFT * SPACING)
    # smoother than the other tests: features have to survive the skip-sampling
    fix, mov, mov_origin = _pair(truth, fix_shape=shape, pad=pad, sigma=4.0)
    affine = _align(fix, mov, SPACING, mov_origin, alignment_spacing=2.0)
    # the fit runs on a grid twice as coarse: allow half a working voxel
    error = _displacement_error(affine, truth, fix_shape=shape)
    assert error < 1.0


# ---------------------------------------------------------------------------
# static_transform_list: the result is the residual, applied before it
# ---------------------------------------------------------------------------

def _rotation_about_fix_centre(degrees):
    """Rotation in the y-x plane about the centre of the fixed volume"""
    a = np.deg2rad(degrees)
    rotation = np.eye(4)
    rotation[1:3, 1:3] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    centre = _translation_matrix(np.array(FIX_SHAPE) / 2 * SPACING)
    return centre @ rotation @ bst.invert_affine(centre)


def test_static_transform_list_gives_residual():
    # mov(static residual x) = fix(x): with truth = static KNOWN_AFFINE the
    # residual is KNOWN_AFFINE. Composing in the other order would give
    # static KNOWN_AFFINE static^-1 instead, which the 20 degree rotation
    # puts well apart from it
    static = _rotation_about_fix_centre(20)
    fix, mov, mov_origin = _pair(static @ KNOWN_AFFINE)
    wrong_order = static @ KNOWN_AFFINE @ bst.invert_affine(static)
    assert _displacement_error(wrong_order, KNOWN_AFFINE) > 1.5
    residual = _align(fix, mov, SPACING, mov_origin, static_transform_list=[static])
    # looser than the direct fit: the pre-warp crops mov to fix's field of
    # view (see the fireants_affine_align notes)
    assert _displacement_error(residual, KNOWN_AFFINE) < 0.75


# ---------------------------------------------------------------------------
# masks
# ---------------------------------------------------------------------------

def _ball(shape=FIX_SHAPE, radius=18):
    grid = np.indices(shape) - (np.array(shape) / 2)[:, None, None, None]
    return (np.sqrt((grid ** 2).sum(axis=0)) < radius).astype(np.uint8)


@pytest.mark.parametrize('which', ['both', 'fix_only', 'mov_only'])
def test_masks(which):
    truth = _translation_matrix(SHIFT * SPACING)
    fix, mov, mov_origin = _pair(truth)
    masks = {
        'both': dict(fix_mask=_ball(), mov_mask=_ball(mov.shape, radius=30)),
        'fix_only': dict(fix_mask=_ball()),
        'mov_only': dict(mov_mask=_ball(mov.shape, radius=30)),
    }[which]
    affine = _align(fix, mov, SPACING, mov_origin, **masks)
    assert _displacement_error(affine, truth) < 0.25


# ---------------------------------------------------------------------------
# argument checks, final_metric_check and failure paths
# ---------------------------------------------------------------------------

def test_invalid_arguments_raise():
    fix = _smooth_volume(FIX_SHAPE)
    with pytest.raises(ValueError, match='equal length'):
        fireants_affine_align(fix, fix, SPACING, SPACING, scales=(2, 1), iterations=(10,))
    with pytest.raises(ValueError, match='decreasing'):
        fireants_affine_align(fix, fix, SPACING, SPACING, scales=(1, 2), iterations=(10, 10))
    with pytest.raises(ValueError, match="'affine' or 'rigid'"):
        fireants_affine_align(fix, fix, SPACING, SPACING, method='similarity')


@pytest.mark.parametrize('final_metric_check', [True, False])
def test_constant_fixed_block_returns_identity(final_metric_check):
    # e.g. a background block: FireANTs would still move it
    _, mov, mov_origin = _pair(_translation_matrix(SHIFT * SPACING))
    fix = np.zeros(FIX_SHAPE, dtype=np.float32)
    affine = _align(fix, mov, SPACING, mov_origin, final_metric_check=final_metric_check)
    np.testing.assert_array_equal(affine, np.eye(4))


def test_nan_in_moving_image_returns_default():
    fix, mov, mov_origin = _pair(_translation_matrix(SHIFT * SPACING))
    # where fix samples it (mov's own corner would lie outside fix's view)
    mov[tuple(np.array(mov.shape) // 2)] = np.nan
    sentinel = object()
    assert _align(fix, mov, SPACING, mov_origin, clip_percentiles=None, default=sentinel) is sentinel


def test_different_intensity_ranges():
    # each image is rescaled on its own for cc, so a [0, 1] fix next to a
    # uint16-range mov is not squashed back into FireANTs' degenerate regime
    truth = _translation_matrix(SHIFT * SPACING)
    fix, mov, mov_origin = _pair(truth)
    affine = _align(fix, mov * 65535.0, SPACING, mov_origin)
    assert _displacement_error(affine, truth) < 0.25


def test_final_metric_check_returns_default(monkeypatch):
    def _bad_evaluate(self, fixed_images, moving_images, shape=None):
        return torch.rand_like(fixed_images())

    monkeypatch.setattr(fa_affine.AffineRegistration, 'evaluate', _bad_evaluate)
    fix, mov, mov_origin = _pair(_translation_matrix(SHIFT * SPACING))
    sentinel = object()
    assert _align(fix, mov, SPACING, mov_origin, default=sentinel) is sentinel


def test_exception_in_optimize_returns_identity(monkeypatch):
    def _fail(self):
        raise RuntimeError('out of memory')

    monkeypatch.setattr(fa_affine.AffineRegistration, 'optimize', _fail)
    fix = _smooth_volume(FIX_SHAPE)
    affine = fireants_affine_align(fix, fix.copy(), SPACING, SPACING, **FAST)
    np.testing.assert_array_equal(affine, np.eye(4))


def test_not_implemented_error_is_raised(monkeypatch):
    def _missing_op(self):
        raise NotImplementedError('op missing on this device')

    monkeypatch.setattr(fa_affine.AffineRegistration, 'optimize', _missing_op)
    fix = _smooth_volume(FIX_SHAPE)
    with pytest.raises(NotImplementedError):
        fireants_affine_align(fix, fix.copy(), SPACING, SPACING, **FAST)


# ---------------------------------------------------------------------------
# devices
# ---------------------------------------------------------------------------

ACCELERATORS = [
    pytest.param('mps', marks=pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason='MPS not available')),
    pytest.param('cuda', marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason='CUDA not available')),
]


@pytest.mark.parametrize('device', ACCELERATORS)
def test_accelerator_recovers_like_cpu(device):
    # each device lands within a fraction of a voxel of the truth, but not at
    # the same point: Adam's fixed step keeps the parameters moving by about
    # optimizer_lr per iteration, and float32 rounding differs per device, so
    # the runs stop at different places. Each is checked against the truth.
    assert _displacement_error(_identity_affine(device), np.eye(4)) < 0.1
    assert _displacement_error(
        _translation_affine(device), _translation_matrix(SHIFT * SPACING)) < 0.25
    error = _displacement_error(
        _anisotropic_affine(device), _translation_matrix(ANISO_SHIFT * ANISO_SPACING),
        spacing=ANISO_SPACING,
    )
    assert error < 0.25


@pytest.mark.parametrize('device', ACCELERATORS)
def test_accelerator_mi_loss(device):
    affine = _translation_affine(device, loss='mi')
    assert _displacement_error(affine, _translation_matrix(SHIFT * SPACING)) < 0.5


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------

def test_alignment_pipeline_fireants_affine_step():
    truth = _translation_matrix(SHIFT * SPACING)
    fix, mov, mov_origin = _pair(truth)
    result = alignment_pipeline(
        fix, mov, SPACING, SPACING,
        steps=[('fireants_affine', FAST)],
        mov_origin=mov_origin,
    )
    assert result.shape == (4, 4)
    assert _displacement_error(result, truth) < 0.25


def test_alignment_pipeline_fireants_affine_then_deform():
    fix, mov, mov_origin = _pair(KNOWN_AFFINE)
    result = alignment_pipeline(
        fix, mov, SPACING, SPACING,
        steps=[
            ('fireants_affine', FAST),
            ('fireants_deform', dict(scales=(2, 1), iterations=(60, 30), device='cpu')),
        ],
        mov_origin=mov_origin,
    )
    assert result.shape == FIX_SHAPE + (3,)
    warped = bst.apply_transform(
        fix, mov, SPACING, SPACING, transform_list=[result],
        mov_origin=mov_origin,
    )
    interior = (slice(5, -5),) * 3
    ncc = np.corrcoef(fix[interior].ravel(), warped[interior].ravel())[0, 1]
    assert ncc > 0.99


def test_alignment_pipeline_default_with_fireants_affine_step():
    result = alignment_pipeline(
        None, np.zeros(FIX_SHAPE, dtype=np.float32), SPACING, SPACING,
        steps=[('fireants_affine', {})],
    )
    np.testing.assert_array_equal(result, np.eye(4))


def test_alignment_pipeline_without_fireants_installed(monkeypatch):
    # a None entry in sys.modules makes `import fireants` raise ImportError
    for name in [m for m in sys.modules if m == 'fireants' or m.startswith('fireants.')]:
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setitem(sys.modules, 'fireants', None)
    fix = _smooth_volume(FIX_SHAPE)
    with pytest.raises(ImportError):
        alignment_pipeline(
            fix, fix.copy(), SPACING, SPACING, steps=[('fireants_affine', FAST)],
        )
