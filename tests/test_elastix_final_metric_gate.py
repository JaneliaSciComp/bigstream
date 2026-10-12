import numpy as np
import pytest
import SimpleITK as sitk

from bigstream.contrib.elastix_align import (
    FINAL_METRICS,
    MIN_VALID_VOXELS_FOR_CHECK,
    _alignment_score,
    _final_metric_check,
    elastix_affine_align,
    elastix_deformable_align,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

SHAPE = (32, 40, 48)      # zyx
SHIFT = 3                 # voxels along x (the last array axis)


def _blobs(seed=0):
    """Smooth random volume in [0, 1]."""
    rng = np.random.default_rng(seed)
    img = sitk.GetImageFromArray(rng.random(SHAPE).astype(np.float32))
    img = sitk.SmoothingRecursiveGaussian(img, 1.5)
    arr = sitk.GetArrayFromImage(img)
    arr -= arr.min()
    return arr / arr.max()


def _pair(inverted):
    """
    fix, and a mov that is fix shifted by SHIFT voxels along x, optionally
    contrast-inverted (nuclei bright in one channel, dark in the other).
    """
    f = _blobs()
    m = np.zeros_like(f)
    m[..., SHIFT:] = (1.0 - f if inverted else f)[..., :-SHIFT]
    return sitk.GetImageFromArray(f), sitk.GetImageFromArray(m)


def _translation(dx):
    return sitk.TranslationTransform(3, (float(dx), 0.0, 0.0))


def _gate(fix, mov, after, method, fix_mask=None, tolerance=0.0):
    return _final_metric_check(
        fix, mov, after, sitk.sitkLinear, fix_mask,
        method, tolerance, context='test', label='affine',
    )


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('method', ['mi', 'abs_ncc'])
@pytest.mark.parametrize('inverted', [False, True])
def test_gate_keeps_correct_alignment_for_any_contrast(method, inverted):
    fix, mov = _pair(inverted)
    # mov(x) = fix(x - SHIFT), so fix point x lands on mov point x + SHIFT
    assert _gate(fix, mov, _translation(SHIFT), method)


@pytest.mark.parametrize('method', ['mi', 'abs_ncc'])
@pytest.mark.parametrize('inverted', [False, True])
def test_gate_rejects_worse_alignment(method, inverted):
    fix, mov = _pair(inverted)
    assert not _gate(fix, mov, _translation(-SHIFT), method)


def test_ssd_rejects_correct_alignment_of_inverted_contrast():
    # documents why SSD is not the default: with inverted contrast the
    # correct alignment has the larger squared difference
    fix, mov = _pair(inverted=True)
    assert not _gate(fix, mov, _translation(SHIFT), 'ssd')


def test_ssd_keeps_correct_alignment_of_matching_contrast():
    fix, mov = _pair(inverted=False)
    assert _gate(fix, mov, _translation(SHIFT), 'ssd')


def test_gate_tolerance_allows_small_loss():
    fix, mov = _pair(inverted=False)
    worse = _translation(-SHIFT)
    assert not _gate(fix, mov, worse, 'mi', tolerance=0.0)
    assert _gate(fix, mov, worse, 'mi', tolerance=10.0)


def test_gate_refuses_to_judge_an_empty_mask():
    fix, mov = _pair(inverted=False)
    empty = sitk.GetImageFromArray(np.zeros(SHAPE, dtype=np.uint8))
    assert not _gate(fix, mov, _translation(SHIFT), 'mi', fix_mask=empty)


def test_gate_only_scores_voxels_in_the_mask():
    fix, mov = _pair(inverted=True)
    # a mask smaller than the evaluation floor cannot be judged ...
    tiny = np.zeros(SHAPE, dtype=np.uint8)
    tiny[:2, :2, :2] = 1
    assert tiny.sum() < MIN_VALID_VOXELS_FOR_CHECK
    assert not _gate(
        fix, mov, _translation(SHIFT), 'mi',
        fix_mask=sitk.GetImageFromArray(tiny),
    )
    # ... while a large one is scored normally
    big = np.ones(SHAPE, dtype=np.uint8)
    assert _gate(
        fix, mov, _translation(SHIFT), 'mi',
        fix_mask=sitk.GetImageFromArray(big),
    )


def test_gate_ignores_voxels_the_moving_image_does_not_cover():
    # the last SHIFT columns of fix map outside mov after the correct
    # translation; they must not make the result look worse
    fix, mov = _pair(inverted=False)
    assert _gate(fix, mov, _translation(SHIFT), 'mi')


# ---------------------------------------------------------------------------
# the scores
# ---------------------------------------------------------------------------

def test_mi_is_larger_for_dependent_than_independent_vectors():
    rng = np.random.default_rng(1)
    a = rng.random(20000).astype(np.float32)
    dependent = _alignment_score(a, 1.0 - a, 'mi')
    independent = _alignment_score(a, rng.random(20000).astype(np.float32), 'mi')
    assert dependent > independent
    assert independent >= 0


def test_abs_ncc_is_sign_agnostic():
    rng = np.random.default_rng(2)
    a = rng.random(5000).astype(np.float32)
    assert _alignment_score(a, a, 'abs_ncc') == pytest.approx(1.0)
    assert _alignment_score(a, 1.0 - a, 'abs_ncc') == pytest.approx(1.0)


def test_abs_ncc_of_constant_vector_is_zero():
    a = np.random.default_rng(3).random(100).astype(np.float32)
    assert _alignment_score(a, np.full(100, 0.5, np.float32), 'abs_ncc') == 0.0


def test_unknown_metric_is_rejected():
    a = np.zeros(10, np.float32)
    with pytest.raises(ValueError):
        _alignment_score(a, a, 'nope')


@pytest.mark.parametrize(
    'align', [elastix_affine_align, elastix_deformable_align],
)
def test_unknown_final_metric_fails_before_any_work(align):
    z = np.zeros((4, 4, 4), np.float32)
    with pytest.raises(ValueError, match='final_metric'):
        align(z, z, (1, 1, 1), (1, 1, 1), final_metric='nope')


def test_final_metrics_has_mi_first():
    assert FINAL_METRICS[0] == 'mi'
