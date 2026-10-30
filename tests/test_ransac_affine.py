"""
The ransac behind `feature_point_ransac_affine_align`: OpenCV by default,
skimage when `ransac_method='skimage'`.
"""

import numpy as np
import pytest

from bigstream.align import (_is_affine_valid, _ransac_affine,
                             feature_point_ransac_affine_align)
from skimage.transform import AffineTransform


@pytest.fixture
def matches():
    """Matches from a known affine, with noise and 20% gross outliers."""
    rng = np.random.default_rng(0)
    fix = (rng.random((200, 3)) * [400, 1000, 1000]).astype(np.float32)
    affine = np.array([[1.02, 0.05, 0.00, 5.0],
                       [0.01, 0.98, 0.02, -3.0],
                       [0.00, 0.03, 1.01, 7.0]])
    mov = fix @ affine[:, :3].T + affine[:, 3] + rng.normal(0, 0.5, (200, 3))
    mov[:40] += rng.normal(0, 100, (40, 3))
    return fix, mov.astype(np.float32), affine


@pytest.mark.parametrize('method', ['opencv', 'skimage'])
def test_both_methods_recover_the_affine(matches, method):
    fix, mov, affine = matches
    kwargs = {'max_trials': 2000, 'rng': 0} if method == 'skimage' else {}
    found = _ransac_affine(fix, mov, 2.0, 0.999, 0.25, method, **kwargs)
    assert found.shape == (3, 4)
    np.testing.assert_allclose(found[:, :3], affine[:, :3], atol=0.01)
    np.testing.assert_allclose(found[:, 3], affine[:, 3], atol=1.0)


def test_skimage_takes_its_other_arguments_from_kwargs(matches):
    fix, mov, _ = matches
    calls = []

    def is_model_valid(model, *data):
        calls.append(model)
        return True

    _ransac_affine(fix, mov, 2.0, 0.999, 0.25, 'skimage', max_trials=10,
                   rng=0, is_model_valid=is_model_valid)
    # one check per trial, plus one for the final refit on the inliers
    assert len(calls) == 11


@pytest.mark.filterwarnings('ignore:No inliers found')
def test_skimage_with_no_valid_model_returns_none(matches):
    fix, mov, _ = matches
    found = _ransac_affine(fix, mov, 2.0, 0.999, 0.25, 'skimage', max_trials=10,
                           rng=0, is_model_valid=lambda *args: False)
    assert found is None


def test_unknown_method_is_refused(matches):
    fix, mov, _ = matches
    with pytest.raises(ValueError, match='ransac_method'):
        _ransac_affine(fix, mov, 2.0, 0.999, 0.25, 'ransac++')


@pytest.mark.parametrize('ndim', [2, 3])
def test_affine_check_bounds_the_scale_of_every_axis(ndim):
    def check(scales):
        model = AffineTransform(matrix=np.diag([*scales, 1.0]))
        return _is_affine_valid(model, min_scale=0.75, max_scale=1.25)

    assert check([1.0] * ndim)
    assert check([0.8] + [1.2] * (ndim - 1))
    assert not check([1.3] + [1.0] * (ndim - 1))
    assert not check([1.0] * (ndim - 1) + [0.7])


def test_affine_check_refuses_a_degenerate_or_non_finite_model():
    flat = np.eye(4)
    flat[2, 2] = 0
    assert not _is_affine_valid(AffineTransform(matrix=flat))
    nan = np.eye(4)
    nan[0, 3] = np.nan
    assert not _is_affine_valid(AffineTransform(matrix=nan))


def test_skimage_rejects_models_outside_the_diagonal_constraint(matches):
    fix, _, _ = matches
    # a 1.5x scale on z: valid with a 0.6 constraint, not with 0.25
    mov = fix * [1.5, 1.0, 1.0]
    kwargs = {'max_trials': 200, 'rng': 0}
    assert _ransac_affine(fix, mov, 2.0, 0.999, 0.6, 'skimage', **kwargs) is not None
    with pytest.warns(UserWarning):
        assert _ransac_affine(fix, mov, 2.0, 0.999, 0.25, 'skimage', **kwargs) is None


@pytest.mark.parametrize('diagonal_constraint', [0, -0.1])
def test_non_positive_diagonal_constraint_is_refused(diagonal_constraint):
    image = np.zeros((8, 8, 8), dtype=np.float32)
    with pytest.raises(ValueError, match='diagonal_constraint must be positive'):
        feature_point_ransac_affine_align(
            image, image, np.ones(3), np.ones(3), blob_sizes=[2, 4],
            diagonal_constraint=diagonal_constraint,
            safeguard_exceptions=False,
        )


def _shear_3d(angle):
    """x axis tilted towards y by `angle` radians; unit column norms."""
    linear = np.eye(3)
    linear[:, 2] = [0, np.sin(angle), np.cos(angle)]
    matrix = np.eye(4)
    matrix[:3, :3] = linear
    return AffineTransform(matrix=matrix)


def test_affine_check_bounds_the_shear_in_3d():
    model = _shear_3d(0.3)
    assert _is_affine_valid(model)
    assert _is_affine_valid(model, max_shear=0.35)
    assert not _is_affine_valid(model, max_shear=0.25)


def test_affine_check_shear_matches_skimage_in_2d():
    model = AffineTransform(shear=0.3, rotation=0.5, scale=(1.1, 0.9))
    assert np.isclose(model.shear, 0.3)
    assert _is_affine_valid(model, max_shear=0.31)
    assert not _is_affine_valid(model, max_shear=0.29)


def test_skimage_rejects_models_above_max_shear(matches):
    fix, _, _ = matches
    mov = fix @ _shear_3d(0.3).params[:3, :3].T
    kwargs = {'max_trials': 200, 'rng': 0}
    assert _ransac_affine(fix, mov, 2.0, 0.999, 0.25, 'skimage',
                          max_shear=0.35, **kwargs) is not None
    with pytest.warns(UserWarning):
        assert _ransac_affine(fix, mov, 2.0, 0.999, 0.25, 'skimage',
                              max_shear=0.25, **kwargs) is None


@pytest.mark.parametrize('max_shear', [0, -0.1])
def test_non_positive_max_shear_is_refused(max_shear):
    image = np.zeros((8, 8, 8), dtype=np.float32)
    with pytest.raises(ValueError, match='max_shear must be positive'):
        feature_point_ransac_affine_align(
            image, image, np.ones(3), np.ones(3), blob_sizes=[2, 4],
            max_shear=max_shear, safeguard_exceptions=False,
        )
