"""
The weight ramp shape used to blend overlapping blocks.

`distributed_align` stitches per-block displacement fields by overlap-add:
each block's field is multiplied by a weight array `w` that is 1 on the block
core and falls to 0 across the halo, and adjacent blocks' weights sum to
exactly 1 everywhere they overlap (a *partition of unity*). Anything less and
the stitched field is silently scaled up or down along every seam.

The shape of that fall-off is what this module selects. Writing `t` for the
normalized position across one face's ramp - `t = 0` at the outermost voxel,
`t -> 1` at the core - the two available profiles are

    linear      w(t) = t
    cosine      w(t) = (1 - cos(pi*t)) / 2        ("raised cosine", "Hann")

Both satisfy `w(t) + w(1-t) = 1`, which is the 1-D partition of unity; the
weights are built as a separable product over the axes, and a product of
per-axis partitions of unity is a partition of unity in N-D.

The two are not interchangeable for fold safety. The blend contributes a
`grad(w) * (u_A - u_B)` term to the jacobian of the stitched field, so what
bounds the safe displacement is the *steepest* part of the ramp, not its
length. The relevant number is the **ramp gradient gain**

    gain = max |dw/dt|      over the unit ramp

which is 1 for the linear profile and exactly `pi/2` for the cosine one. It
divides the displacement ceiling in `blend_safe_displacement_bound`, so a
cosine ramp costs 36% of the amplitude budget at the same halo.

The trade, stated plainly: cosine is smoother *at the ramp ends* - its
derivative vanishes where the ramp meets the core, removing a real kink the
linear ramp has there - and steeper *in the middle*. For the worst-case fold
guarantee it is strictly worse. Which one a given dataset wants is an
empirical question; `linear` stays the default and reproduces every field
computed before this was configurable, bit for bit.
"""

import numpy as np


LINEAR = 'linear'
COSINE = 'cosine'

#: what an unset `blend_ramp` means, everywhere
DEFAULT_BLEND_RAMP = LINEAR

BLEND_RAMPS = (LINEAR, COSINE)


def parse_blend_ramp(value):
    """
    Normalize a configured ramp name. `None` means the default.

    Raises on an unknown name rather than falling back to linear - a typo
    in the config would otherwise change the fold bound silently.
    """
    if value is None:
        return DEFAULT_BLEND_RAMP
    name = str(value).strip().lower()
    if name not in BLEND_RAMPS:
        raise ValueError(
            f'unsupported blend_ramp {value!r}; expected one of '
            f'{list(BLEND_RAMPS)}'
        )
    return name


def ramp_profile(blend_ramp=None):
    """
    `w(t)` for the named ramp, as a callable on a `t` array in `[0, 1]`.

    `t = 0` is the outermost voxel of a face (weight 0) and `t = 1` the core
    (weight 1). This is the single definition of the ramp: both the weights
    and the gain that bounds them are derived from it, so the bound cannot
    drift from the ramp it is bounding.
    """
    name = parse_blend_ramp(blend_ramp)
    if name == LINEAR:
        return lambda t: np.asarray(t, dtype=np.float64)
    return lambda t: 0.5 * (1.0 - np.cos(np.pi * np.asarray(t,
                                                            dtype=np.float64)))


def ramp_gradient_gain(blend_ramp=None, samples=200001):
    """
    `max |dw/dt|` over the unit ramp - how much steeper than linear it gets.

    1.0 for `linear`, `pi/2` for `cosine`. Measured numerically from the same
    callable `ramp_profile` hands to the weight builder, so a new profile
    cannot acquire a stale gain. The sampling error is below 1e-9 at the
    default `samples`; the cosine value matches analytic `pi/2` to 9 decimals.
    """
    name = parse_blend_ramp(blend_ramp)
    if name == LINEAR:
        # exact, and the value every pre-existing bound was computed with
        return 1.0
    t = np.linspace(0.0, 1.0, samples)
    values = ramp_profile(name)(t)
    return float(np.abs(np.gradient(values, t[1] - t[0])).max())


def _ramp_face(width, edge, profile, reverse):
    """
    One padded face: `profile(t)` scaled by the `edge` slab it grows out of.

    `edge` is the adjacent hyperplane of the array being padded, so the ramp
    inherits whatever the already-padded axes left there - that is what makes
    the result a separable product without ever forming one explicitly.

    The `linear` case is written as the `np.linspace` numpy's own
    `linear_ramp` uses (`arange(n) * (edge/n)`, not `(arange(n)/n) * edge`),
    because the two differ by one ulp in the corners and this path has to
    reproduce `np.pad` bit for bit.
    """
    if profile is None:
        values = np.linspace(0.0, edge, num=width, endpoint=False, axis=0)
    else:
        t = (np.arange(width) / width).reshape((width,) + (1,) * edge.ndim)
        values = edge * profile(t)
    return values[::-1] if reverse else values


def blend_ramp_weights(core, pad, blend_ramp=None):
    """
    A weight array that is 1 over `core` and ramps to 0 across `pad`.

    core : tuple of int
        Shape of the full-weight interior.

    pad : sequence of (int, int)
        Ramp width in voxels on the low and high side of each axis.

    Equivalent to `np.pad(np.ones(core), pad, mode='linear_ramp')` for
    `linear` - bit-identical, which is the regression guard for every config
    that does not set `blend_ramp`.
    """
    name = parse_blend_ramp(blend_ramp)
    # `None` selects the linspace form that matches np.pad exactly
    profile = None if name == LINEAR else ramp_profile(name)

    weights = np.ones(tuple(core), dtype=np.float64)
    for axis, (low, high) in enumerate(pad):
        if not low and not high:
            continue
        weights = np.moveaxis(weights, axis, 0)
        parts = [weights]
        if low:
            parts.insert(0, _ramp_face(low, weights[0], profile, False))
        if high:
            parts.append(_ramp_face(high, weights[-1], profile, True))
        weights = np.moveaxis(np.concatenate(parts, axis=0), 0, axis)
    return weights
