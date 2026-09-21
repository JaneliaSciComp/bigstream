"""
Tests for the configurable overlap-add blending ramp.

Two things have to hold for the ramp to be safe to make configurable at all:

  * `linear` is **bit-identical** to the `np.pad(..., mode='linear_ramp')`
    the pipeline called before there was a choice. Every field computed by
    every existing config has to come out unchanged, not merely close - the
    weights multiply a displacement field that is then composed across
    passes, and "close" is not a property that survives that.
  * every offered shape is a **partition of unity**: adjacent blocks' weights
    sum to exactly 1 where they overlap. A ramp that fails this silently
    rescales the stitched field along every seam.
"""
import numpy as np
import pytest

from bigstream.blend_ramp import (BLEND_RAMPS, COSINE, DEFAULT_BLEND_RAMP,
                                  LINEAR, blend_ramp_weights, parse_blend_ramp,
                                  ramp_gradient_gain, ramp_profile)


# cores and pads chosen to be awkward: asymmetric faces, a 1-voxel core, an
# axis with no ramp at all (a zero-halo axis, which must be left alone rather
# than turned into an empty slice)
GEOMETRIES = [
    ((5,), [(3, 3)]),
    ((4, 6), [(5, 5), (3, 3)]),
    ((4, 6), [(5, 2), (0, 0)]),
    ((3, 4, 5), [(7, 7), (1, 1), (0, 0)]),
    ((1, 9, 2), [(4, 4), (6, 6), (3, 3)]),
    ((10, 1), [(0, 0), (9, 9)]),
]


@pytest.mark.parametrize('core,pad', GEOMETRIES)
def test_linear_is_bit_identical_to_np_pad(core, pad):
    """
    The regression guard: nothing else in this change matters if this fails.

    Note this is `array_equal`, not `allclose`. The obvious separable
    rewrite - an outer product of per-axis 1-D profiles - is *not* bit
    identical; it lands one ulp off in the corners, because it computes
    `(arange(n)/n) * edge` where numpy computes `arange(n) * (edge/n)`.
    `blend_ramp_weights` pads axis by axis for exactly that reason.
    """
    expected = np.pad(np.ones(core, dtype=np.float64), pad,
                      mode='linear_ramp')
    assert np.array_equal(blend_ramp_weights(core, pad, LINEAR), expected)


def test_unset_ramp_is_linear():
    core, pad = (4, 6), [(5, 5), (3, 3)]
    assert DEFAULT_BLEND_RAMP == LINEAR
    assert np.array_equal(blend_ramp_weights(core, pad, None),
                          blend_ramp_weights(core, pad, LINEAR))


@pytest.mark.parametrize('ramp', BLEND_RAMPS)
@pytest.mark.parametrize('core,pad', GEOMETRIES)
def test_weights_span_zero_to_one_and_keep_the_core(core, pad, ramp):
    weights = blend_ramp_weights(core, pad, ramp)
    assert weights.shape == tuple(c + lo + hi for c, (lo, hi) in zip(core, pad))
    assert weights.max() == pytest.approx(1.0)
    assert weights.min() >= 0.0
    # the core itself is untouched by any ramp
    core_region = tuple(slice(lo, lo + c) for c, (lo, _) in zip(core, pad))
    np.testing.assert_array_equal(weights[core_region], 1.0)


@pytest.mark.parametrize('ramp', BLEND_RAMPS)
def test_one_dimensional_ramp_partitions_unity(ramp):
    """
    `w(t) + w(1-t) = 1` on the unit ramp - a block fading out is exactly the
    complement of its neighbour fading in. Separability carries it to N-D:
    the product of per-axis partitions of unity is a partition of unity, so
    this 1-D statement is the whole content of the property. The assembled
    N-D lattice is checked over real block geometry in
    `test_blockwise_alignment.test_blending_weights_are_a_partition_of_unity`.
    """
    t = np.linspace(0.0, 1.0, 257)
    w = ramp_profile(ramp)(t)
    np.testing.assert_allclose(w + w[::-1], 1.0, rtol=0, atol=1e-15)


def test_cosine_has_a_vanishing_derivative_at_both_ramp_ends():
    """
    The point of offering cosine at all: the linear ramp has a kink where it
    meets the block core (slope 1/L on one side, 0 on the other), and cosine
    does not. The price is paid mid-ramp - see the gain tests.
    """
    n = 512
    linear = np.diff(blend_ramp_weights((1,), [(n, n)], LINEAR)[:n])
    cosine = np.diff(blend_ramp_weights((1,), [(n, n)], COSINE)[:n])
    # the linear ramp climbs at its full slope right up to the core, so the
    # meeting point is a genuine discontinuity in the derivative
    assert linear[0] / linear.max() == pytest.approx(1.0)
    assert linear[-1] / linear.max() == pytest.approx(1.0)
    # the cosine ramp arrives flat at both ends
    assert cosine[0] / cosine.max() < 0.02
    assert cosine[-1] / cosine.max() < 0.02
    # and pays for it mid-ramp, by exactly the pi/2 that divides the bound
    assert cosine.max() / linear.max() == pytest.approx(np.pi / 2, rel=1e-3)


def test_ramp_gradient_gain_values():
    """
    `gain = max |dw/dt|` over the unit ramp. This is the number that divides
    the fold-safe displacement ceiling, so both values are pinned.
    """
    assert ramp_gradient_gain(LINEAR) == 1.0
    assert ramp_gradient_gain(None) == 1.0
    assert ramp_gradient_gain(COSINE) == pytest.approx(np.pi / 2, abs=1e-9)


def test_ramp_gradient_gain_matches_the_profile_it_bounds():
    """
    The gain is measured from the same callable that lays down the weights,
    so a future profile cannot be added with a stale hand-written gain.
    """
    for ramp in BLEND_RAMPS:
        t = np.linspace(0.0, 1.0, 50001)
        measured = np.abs(np.gradient(ramp_profile(ramp)(t),
                                      t[1] - t[0])).max()
        assert ramp_gradient_gain(ramp) == pytest.approx(measured, rel=1e-6)


@pytest.mark.parametrize('name,expected', [
    (None, LINEAR), ('linear', LINEAR), ('cosine', COSINE),
    ('COSINE', COSINE), (' Cosine ', COSINE),
])
def test_parse_blend_ramp_normalizes(name, expected):
    assert parse_blend_ramp(name) == expected


@pytest.mark.parametrize('name', ['hann', 'tukey', 'liner', '', 'none'])
def test_unknown_ramp_is_refused(name):
    """
    Not silently linear: a typo would blend with one ramp while the fold
    bound was computed for another, and nothing downstream would notice.
    """
    with pytest.raises(ValueError, match='unsupported blend_ramp'):
        parse_blend_ramp(name)
    with pytest.raises(ValueError, match='unsupported blend_ramp'):
        blend_ramp_weights((4,), [(2, 2)], name)
    with pytest.raises(ValueError, match='unsupported blend_ramp'):
        ramp_gradient_gain(name)
