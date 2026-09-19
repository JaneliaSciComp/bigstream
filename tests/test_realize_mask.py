"""
Tests for `realize_mask`, focused on the `background` threshold.

`background` names an intensity below which a voxel is background. The
non-background voxels are *intersected* with whatever the other parameters
select, so it can only ever narrow the foreground - never widen it, and
never replace it.

The failure mode these guard against is a silent empty mask: an alignment
handed a mask of all zeros does not raise, it just has nothing to match on.
"""

import numpy as np

from bigstream.align import realize_mask


BACKGROUND = 100


def _gradient(shape=(10, 10, 10), low=0.0, high=255.0):
    """A ramp, so an intensity band selects a real subset rather than all
    or nothing (a bimodal image makes a percentile band degenerate)."""
    return np.linspace(low, high, int(np.prod(shape)),
                       dtype=np.float32).reshape(shape)


def test_background_alone_masks_the_background_voxels():
    """Only a background level given - it still has to produce a mask.

    The early return for "nothing was asked for" must not swallow this.
    """
    image = _gradient()
    mask = realize_mask(image, None, background=BACKGROUND)
    assert mask is not None
    assert np.array_equal(mask, (image > BACKGROUND).astype(np.uint8))


def test_background_intersects_the_percentile_band():
    """The whole point: it narrows the band, it does not replace or empty it.

    Regression test - this used to compare the already-binarized band mask
    against the background *intensity*, which is false everywhere for any
    background >= 1, so the mask came back empty.
    """
    image = _gradient()
    band = realize_mask(image, None, mask_percentile=(10, 90))
    combined = realize_mask(image, None, mask_percentile=(10, 90),
                            background=BACKGROUND)

    expected = ((band > 0) & (image > BACKGROUND)).astype(np.uint8)
    assert np.array_equal(combined, expected)
    # strictly narrower than the band, and not empty - the two ways this
    # can silently go wrong
    assert 0 < combined.sum() < band.sum()


def test_background_intersects_an_explicit_mask():
    image = _gradient()
    explicit = np.zeros(image.shape, dtype=np.uint8)
    explicit[:5] = 1

    combined = realize_mask(image, explicit, background=BACKGROUND)

    expected = (explicit & (image > BACKGROUND)).astype(np.uint8)
    assert np.array_equal(combined, expected)
    assert combined.sum() > 0


def test_background_intersects_the_roi():
    image = _gradient()
    roi = (slice(0, 5), slice(None), slice(None))

    combined = realize_mask(image, None, background=BACKGROUND, roi=roi)

    roi_box = np.zeros(image.shape, dtype=np.uint8)
    roi_box[roi] = 1
    assert np.array_equal(combined,
                          (roi_box & (image > BACKGROUND)).astype(np.uint8))


def test_no_background_leaves_the_other_filters_untouched():
    """The parameter is opt-in; omitting it changes nothing."""
    image = _gradient()
    explicit = np.ones(image.shape, dtype=np.uint8)

    assert realize_mask(image, None) is None
    assert np.array_equal(
        realize_mask(image, None, mask_percentile=(10, 90), background=None),
        realize_mask(image, None, mask_percentile=(10, 90)),
    )
    assert np.array_equal(
        realize_mask(image, explicit, background=None),
        realize_mask(image, explicit),
    )


def test_background_below_every_voxel_keeps_the_whole_foreground():
    """A threshold nothing falls under is a no-op, not an empty mask."""
    image = _gradient(low=10.0, high=255.0)
    band = realize_mask(image, None, mask_percentile=(10, 90))
    combined = realize_mask(image, None, mask_percentile=(10, 90),
                            background=0)
    assert np.array_equal(combined, band)
