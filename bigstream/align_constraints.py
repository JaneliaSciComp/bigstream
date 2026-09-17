"""
Constraints that keep per-block align/deform steps blend-safe.

The bulk of this module implements the sufficient condition for local
invertibility of B-spline parameterized deformations from

    S. Y. Chun and J. A. Fessler, "A simple regularizer for B-spline nonrigid
    image registration that encourages local invertibility", IEEE J. Sel.
    Topics Signal Processing, 3(1), 2009.

The paper adds a piecewise-quadratic penalty R(c) to the similarity metric.
SimpleITK's ImageRegistrationMethod has a single metric slot and no mechanism
for adding a penalty term, and editing the transform from inside a
registration callback does not survive (ITKv4 optimizers overwrite the
transform from an internal parameter cache on the next step). So instead of
penalizing the constraint we enforce it directly, by projecting the
coefficient grid onto the constraint set C4 after optimization. That is a
strictly stronger guarantee than the penalty and costs a few microseconds.

The condition, on differences of adjacent B-spline coefficients:

    -m_r*k_q <= dc^q/dr <= +m_r*k_q      (r != q, off diagonal)
    -m_q*k_q <= dc^q/dq <= +m_q*K_q      (r == q, diagonal)

where `q` indexes the displacement component (the Jacobian row), `r` the
difference direction, `m_r` the knot spacing along `r`, and `sum(k) < 1`.
Satisfying it guarantees

    1 - sum(k) <= |J| <= (1+Kz)(1+Ky)(1+Kx) + ... > 0

everywhere on the continuous domain, not just at grid points.

The C4 condition above operates on a B-spline coefficient grid, so it applies
to the `deform` step only - an affine transform has no coefficient grid, and
is invertible iff its determinant is nonzero (see
`is_orientation_preserving`). An affine's only fold risk is amplitude, when
`distributed_align` blends it against a neighbour that disagrees; see
`bound_affine_displacement` for the bound that constrains that.

All arrays use bigstream's zyx convention: `knot_spacing`, `k` and `K` are
zyx, and coefficient arrays are indexed [component_zyx][z][y][x].
"""

import logging
from itertools import product

import numpy as np
import SimpleITK as sitk


logger = logging.getLogger(__name__)


# sum(k) must be < 1; 0.32 per axis leaves min|J| >= 0.04
DEFAULT_K = 0.32


def _as_per_axis(value, ndim, name, allow_zero=False):
    """Broadcast a scalar or validate a per-axis sequence to length ndim."""
    arr = np.atleast_1d(np.asarray(value, dtype=np.float64))
    if arr.size == 1:
        arr = np.full(ndim, arr[0], dtype=np.float64)
    if arr.size != ndim:
        raise ValueError(
            f'{name} must be a scalar or have {ndim} values (zyx), got {value}'
        )
    if allow_zero:
        if np.any(arr < 0):
            raise ValueError(f'{name} must be non-negative, got {value}')
    elif np.any(arr <= 0):
        raise ValueError(f'{name} must be positive, got {value}')
    return arr


def coefficient_bounds(knot_spacing, k=DEFAULT_K, K=None):
    """
    C4 bounds on differences of adjacent B-spline coefficients.

    Parameters
    ----------
    knot_spacing : 1d array
        Physical knot (control point) spacing, zyx. This is the spacing of the
        coefficient images, which changes between multi-resolution levels -
        always read it from the transform rather than from a config value.

    k : float or 1d array (default: 0.32)
        Per-component shrinkage allowance, zyx. `sum(k) < 1` is required and
        guarantees `min|J| >= 1 - sum(k) > 0`.

    K : float or 1d array (default: None)
        Per-component expansion allowance for the diagonal term, zyx.
        Defaults to `k` (the symmetric case, Kim et al.). Larger values permit
        acute local expansion without permitting collapse.

    Returns
    -------
    lo, hi : 2d arrays, shape (ndim, ndim), indexed [q, r]
        Lower and upper bounds on `c^q[..., i+1, ...] - c^q[..., i, ...]`
        where the difference is taken along axis `r`.
    """
    knot_spacing = np.asarray(knot_spacing, dtype=np.float64)
    ndim = knot_spacing.size
    k = _as_per_axis(k, ndim, 'k')
    K = k if K is None else _as_per_axis(K, ndim, 'K')

    if k.sum() >= 1.0:
        raise ValueError(
            f'sum(k) must be < 1 to guarantee local invertibility, got '
            f'k={k.tolist()} (sum={k.sum()}); min|J| would be {1 - k.sum()}'
        )

    # lo[q, r] = -m_r * k_q ; hi[q, r] = m_r * (K_q if q == r else k_q)
    lo = -np.outer(k, knot_spacing)
    hi = np.outer(k, knot_spacing)
    diagonal = np.arange(ndim)
    hi[diagonal, diagonal] = K * knot_spacing
    return lo, hi


def _pair_slices(ndim, axis, parity, n):
    """Index tuples selecting the (i, i+1) pairs along `axis` for one parity."""
    first = [slice(None)] * ndim
    second = [slice(None)] * ndim
    first[axis] = slice(parity, n - 1, 2)
    second[axis] = slice(parity + 1, n, 2)
    return tuple(first), tuple(second)


def _excess(differences, lo, hi):
    """Signed-magnitude constraint violation of each difference (>= 0)."""
    return np.maximum(0.0, differences - hi) + np.maximum(0.0, lo - differences)


def _sweep(c, lo, hi, scales):
    """
    One cyclic pass over every constraint slab, projecting pair by pair.

    Each (component, difference axis) is visited in two half passes, even and
    odd indexed pairs, because an interior coefficient belongs to two pairs
    and must not be updated twice in one vectorized operation.

    Returns (max absolute correction, max relative correction) applied.
    """
    ndim = c.shape[0]
    grid_shape = c.shape[1:]
    max_absolute = 0.0
    max_relative = 0.0

    for q in range(ndim):
        for r in range(ndim):
            n = grid_shape[r]
            if n < 2:
                continue
            for parity in (0, 1):
                first, second = _pair_slices(ndim, r, parity, n)
                # basic slicing => views, so += writes through to c
                a = c[q][first]
                b = c[q][second]
                if a.size == 0:
                    continue
                d = b - a
                shift = 0.5 * (np.maximum(0.0, d - hi[q, r])
                               - np.maximum(0.0, lo[q, r] - d))
                largest = float(np.abs(shift).max())
                if largest > 0.0:
                    a += shift
                    b -= shift
                    max_absolute = max(max_absolute, largest)
                    if scales[q, r] > 0:
                        max_relative = max(max_relative, largest / scales[q, r])

    return max_absolute, max_relative


def _scan(coefficients, lo, hi, rtol=0.0):
    """
    Accumulate violation statistics over every adjacent coefficient pair.

    `rtol` is the relative excess below which a pair counts as satisfied; it
    exists so the reported violation count stays consistent with the
    convergence tolerance a projection was run at, instead of reporting
    floating point dust as violations.

    Returns (n_pairs, n_violating, max_relative_excess, penalty, per_axis)
    where per_axis maps the difference axis to its violating pair count.
    """
    ndim = coefficients.shape[0]
    n_pairs = 0
    n_violating = 0
    max_relative = 0.0
    penalty = 0.0
    per_axis = {r: 0 for r in range(ndim)}

    for q in range(ndim):
        for r in range(ndim):
            if coefficients.shape[1 + r] < 2:
                continue
            differences = np.diff(coefficients[q], axis=r)
            excess = _excess(differences, lo[q, r], hi[q, r])
            # normalize by the bound magnitude so the tolerance is scale free
            scale = max(abs(lo[q, r]), abs(hi[q, r]))
            violating = int(np.count_nonzero(excess > scale * rtol))
            n_pairs += differences.size
            n_violating += violating
            per_axis[r] += violating
            if scale > 0 and excess.size:
                max_relative = max(max_relative, float(excess.max()) / scale)
            penalty += 0.5 * float(np.sum(excess ** 2))

    return n_pairs, n_violating, max_relative, penalty, per_axis


def c4_violations(coefficients, knot_spacing, k=DEFAULT_K, K=None, rtol=1e-9):
    """
    Report how far a coefficient grid is from satisfying C4, without changing it.

    Parameters
    ----------
    coefficients : nd-array
        Shape (ndim, *grid_shape), component axis first, zyx throughout.

    knot_spacing : 1d array
        Physical knot spacing, zyx.

    k, K : see `coefficient_bounds`

    rtol : float (default: 1e-9)
        Relative excess below which a pair counts as satisfied.

    Returns
    -------
    dict with n_pairs, n_violating, max_relative_excess, penalty,
    violating_pairs_per_axis, and min_jacobian_bound (the guaranteed lower
    bound on |J| once the constraint holds).
    """
    lo, hi = coefficient_bounds(knot_spacing, k, K)
    n_pairs, n_violating, max_relative, penalty, per_axis = _scan(
        np.asarray(coefficients, dtype=np.float64), lo, hi, rtol=rtol,
    )
    ndim = np.asarray(knot_spacing).size
    return {
        'n_pairs': n_pairs,
        'n_violating': n_violating,
        'max_relative_excess': max_relative,
        'penalty': penalty,
        'violating_pairs_per_axis': per_axis,
        'min_jacobian_bound': 1.0 - _as_per_axis(k, ndim, 'k').sum(),
    }


def chun_fessler_penalty(coefficients, knot_spacing, k=DEFAULT_K, K=None):
    """
    R(c), eq. (10) of the paper: the piecewise-quadratic hinge with a dead zone.

    Diagnostics only - nothing here optimizes it. Zero exactly when the
    coefficients satisfy C4.
    """
    lo, hi = coefficient_bounds(knot_spacing, k, K)
    return _scan(np.asarray(coefficients, dtype=np.float64), lo, hi)[3]


def blend_safe_displacement_bound(block_overlaps, spacing, k,
                                  min_jacobian=0.1):
    """
    Largest per-component displacement that blockwise blending cannot fold.

    `distributed_align` stitches per-block fields as a weighted sum with
    linear-ramp weights. The derivative of that sum is

        d/dr [w*u_A + (1-w)*u_B] = w*u_A' + (1-w)*u_B' + dw/dr * (u_A - u_B)

    and the last term is bounded by nothing in C4 - two individually compliant
    blocks that disagree strongly still fold where they are blended. Over a
    linear ramp of physical length L the weight gradient is 1/L, so with
    per-component displacements bounded by U the blend raises the effective
    derivative bound for every component from k_q to k_q + 2U/L. Chun &
    Fessler's Lemma 2 sums that over all ndim rows of the jacobian:

        min|J| >= 1 - sum_q (k_q + 2U/L) = 1 - sum(k) - 2*ndim*U/L

    so

        U <= (1 - sum(k) - min_jacobian) * L / (2*ndim)

    The `ndim` factor is easy to drop - every row picks up its own blend term,
    not just one. Omitting it yields a bound that permits folding: measured on
    a 5^3 block lattice with L=25 and one runaway block, folding begins at a
    per-component U of 9.04, which `L/2 = 12.50` would have allowed and
    `L/(2*ndim) = 4.17` correctly excludes. The bound is worst case (it assumes
    every component maximally opposed and the weight gradient aligned with it),
    so the true threshold sits above it - by ~2.2x in that measurement.

    Parameters
    ----------
    block_overlaps : 1d array
        Per-axis block overlap in voxels (zyx), as used to build the blending
        weights.

    spacing : 1d array
        Physical voxel spacing (zyx).

    k : float or 1d array (default: DEFAULT_K)
        The C4 allowance the deform step is configured with.

    min_jacobian : float (default: 0.1)
        Jacobian determinant to keep in reserve for the blend.

    Returns
    -------
    float
        The bound, as a scalar over the tightest axis. `inf` when there is no
        overlap to blend across, `0.0` when `k` leaves no headroom at all (in
        which case no finite displacement is safe and `k` must be reduced).
    """
    overlaps = np.atleast_1d(np.asarray(block_overlaps, dtype=np.float64))
    voxel = np.atleast_1d(np.asarray(spacing, dtype=np.float64))
    ndim = overlaps.size
    headroom = 1.0 - _as_per_axis(k, ndim, 'k').sum() - float(min_jacobian)
    if headroom <= 0:
        return 0.0
    # the blending weights ramp linearly over 2*overlap - 1 voxels per axis
    ramp = np.maximum(2.0 * overlaps - 1.0, 0.0) * voxel
    if not np.any(ramp > 0):
        return float('inf')
    return float(np.min(ramp[ramp > 0]) * headroom / (2.0 * ndim))


def project_to_c4(coefficients, knot_spacing, k=DEFAULT_K, K=None,
                  max_displacement=None, max_sweeps=100, tol=1e-9):
    """
    Project a coefficient grid onto C4 (and optionally a displacement bound).

    Each C4 constraint is a slab `{c : lo <= c[i+1] - c[i] <= hi}`, which is
    convex, and the intersection is non-empty (any spatially constant grid is
    feasible), so cyclic projection converges to a feasible point. Pairs that
    already satisfy their bound are untouched - this is a dead-zone operator,
    not a smoother, so a compliant grid is returned unchanged.

    `max_displacement` adds a second convex set, the box
    `{c : |c^q| <= max_displacement[q]}`, whose projection is an elementwise
    clip. Because the cubic b-spline basis is non-negative and a partition of
    unity, the rendered displacement is a convex combination of nearby
    coefficients, so bounding the coefficients bounds the displacement itself
    over the whole continuous domain:

        |d_q(r)| <= max|c^q| <= max_displacement[q]

    (per component - the vector magnitude bound is `sqrt(ndim)` times that).
    This is what C4 alone cannot give: C4 bounds derivatives, so a smooth but
    huge displacement satisfies it while still folding once blended against a
    neighbouring block. See `blend_safe_displacement_bound`.

    Parameters
    ----------
    coefficients : nd-array
        Shape (ndim, *grid_shape), component axis first, zyx throughout.
        Not modified; a projected copy is returned.

    knot_spacing : 1d array
        Physical knot spacing, zyx.

    k, K : see `coefficient_bounds`

    max_displacement : float, 1d array or None (default: None)
        Per-component displacement bound in physical units, zyx. None disables
        the bound and leaves behavior identical to the C4-only projection.

    max_sweeps : int (default: 100)
        Maximum number of cyclic sweeps.

    tol : float (default: 1e-9)
        Convergence tolerance on the largest relative correction applied in a
        sweep. A sweep that moves nothing means every constraint was already
        satisfied when it was visited, i.e. the grid is feasible.

    Returns
    -------
    (projected, info) : nd-array and dict
    """
    c = np.array(coefficients, dtype=np.float64, copy=True)
    ndim = c.shape[0]
    grid_shape = c.shape[1:]
    if len(grid_shape) != ndim:
        raise ValueError(
            f'coefficients must have shape (ndim, *grid_shape) with ndim grid '
            f'axes, got {coefficients.shape}'
        )
    lo, hi = coefficient_bounds(knot_spacing, k, K)
    scales = np.maximum(np.abs(lo), np.abs(hi))

    bound = (None if max_displacement is None
             else _as_per_axis(max_displacement, ndim, 'max_displacement'))
    max_coefficient_before = float(np.abs(c).max()) if c.size else 0.0
    n_clamped = (0 if bound is None
                 else int(np.count_nonzero(
                     np.abs(c) > bound.reshape((ndim,) + (1,) * ndim))))

    def clamp(grid):
        """Projection onto the displacement box; a no-op when unbounded."""
        if bound is None:
            return 0.0
        limits = bound.reshape((ndim,) + (1,) * ndim)
        clipped = np.clip(grid, -limits, limits)
        shift = float(np.abs(clipped - grid).max()) if grid.size else 0.0
        grid[...] = clipped
        return shift

    before = _scan(c, lo, hi, rtol=tol)
    sweeps_run = 0
    max_shift = clamp(c)

    # dead zone: a grid that satisfies every constraint is returned untouched
    if before[1] == 0 and n_clamped == 0:
        after, converged = before, True
    else:
        after, converged = None, False
        for _ in range(max_sweeps):
            absolute_shift, relative_shift = _sweep(c, lo, hi, scales)
            # re-project onto the box each sweep: the C4 sweep can push a
            # coefficient back outside it, and vice versa. POCS over the two
            # convex sets converges to a point in their intersection.
            clamp_shift = clamp(c)
            sweeps_run += 1
            max_shift = max(max_shift, absolute_shift, clamp_shift)
            # a pair's residual excess is twice the correction it still needs,
            # so a sweep that corrects almost nothing is a candidate for
            # feasibility - but confirm it rather than infer it, since
            # projecting one slab can push a neighbouring one back out
            if relative_shift <= 0.5 * tol and clamp_shift == 0.0:
                after = _scan(c, lo, hi, rtol=tol)
                if after[1] == 0:
                    converged = True
                    break
        if after is None or not converged:
            after = _scan(c, lo, hi, rtol=tol)
            converged = after[1] == 0
    if not converged:
        logger.warning(
            f'C4 projection did not converge in {sweeps_run} sweeps: '
            f'{after[1]}/{after[0]} coefficient pairs still violate the '
            f'constraint (max relative excess {after[2]})'
        )

    info = {
        'sweeps_run': sweeps_run,
        'converged': bool(converged),
        'n_pairs': after[0],
        'n_violating_before': before[1],
        'n_violating_after': after[1],
        'max_relative_excess_before': before[2],
        'max_relative_excess_after': after[2],
        'penalty_before': before[3],
        'penalty_after': after[3],
        'max_coefficient_shift': max_shift,
        'min_jacobian_bound': 1.0 - _as_per_axis(k, ndim, 'k').sum(),
        'n_clamped': n_clamped,
        'max_coefficient_before': max_coefficient_before,
        'max_coefficient_after': float(np.abs(c).max()) if c.size else 0.0,
        'max_displacement_bound': (None if bound is None else bound.tolist()),
    }
    return c, info


def bspline_coefficients(transform):
    """
    Extract a sitk.BSplineTransform's coefficients as a zyx numpy array.

    Returns
    -------
    (coefficients, knot_spacing) : nd-array of shape (ndim, *grid_shape) with
    the component axis and all grid axes in zyx order, and the physical knot
    spacing (zyx).
    """
    coefficient_images = transform.GetCoefficientImages()
    ndim = len(coefficient_images)
    grid_size_xyz = coefficient_images[0].GetSize()
    knot_spacing = np.array(
        coefficient_images[0].GetSpacing(), dtype=np.float64)[::-1]

    flat = np.asarray(transform.GetParameters(), dtype=np.float64)
    expected = ndim * int(np.prod(grid_size_xyz))
    if flat.size != expected:
        raise ValueError(
            f'unexpected b-spline parameter count {flat.size}, expected '
            f'{expected} for a {grid_size_xyz} coefficient grid in {ndim}D'
        )
    # parameters are per-component blocks, x fastest
    coefficients = flat.reshape(ndim, *grid_size_xyz[::-1])
    # cheap guard against a SimpleITK layout change silently corrupting results
    reference = sitk.GetArrayFromImage(coefficient_images[0])
    if not np.allclose(coefficients[0], reference):
        raise RuntimeError(
            'b-spline parameter vector does not match the coefficient images; '
            'the assumed parameter layout (per-component blocks, x fastest) is '
            'wrong for this SimpleITK version'
        )
    # component axis xyz -> zyx
    return coefficients[::-1].copy(), knot_spacing


def set_bspline_coefficients(transform, coefficients):
    """Write a zyx coefficient array back into a sitk.BSplineTransform."""
    flat = np.asarray(coefficients, dtype=np.float64)[::-1].reshape(-1)
    transform.SetParameters([float(v) for v in flat])
    return transform


def project_bspline_transform(transform, k=DEFAULT_K, K=None,
                              max_displacement=None,
                              max_sweeps=100, tol=1e-9, context=''):
    """
    Project a sitk.BSplineTransform's coefficients onto C4, in place.

    The knot spacing is read from the transform's coefficient images, so this
    is correct at any multi-resolution level without being told the control
    point spacing.

    `max_displacement` additionally bounds the rendered displacement per
    component - see `project_to_c4`.

    Returns the `info` dict from `project_to_c4`, with `knot_spacing` added.
    A compliant transform is left byte-identical.
    """
    coefficients, knot_spacing = bspline_coefficients(transform)
    projected, info = project_to_c4(
        coefficients, knot_spacing, k=k, K=K,
        max_displacement=max_displacement, max_sweeps=max_sweeps, tol=tol,
    )
    info['knot_spacing'] = knot_spacing.tolist()
    changed = info['n_violating_before'] > 0 or info['n_clamped'] > 0
    if changed:
        set_bspline_coefficients(transform, projected)
        logger.info((
            f'{context} C4 projection: '
            f"{info['n_violating_before']}/{info['n_pairs']} coefficient pairs "
            f"violated the constraint, {info['n_clamped']} coefficients "
            f"exceeded the displacement bound "
            f"{info['max_displacement_bound']}, projected in "
            f"{info['sweeps_run']} sweeps "
            f"(max shift {info['max_coefficient_shift']:.4g}, "
            f"max |c| {info['max_coefficient_before']:.4g} -> "
            f"{info['max_coefficient_after']:.4g}, "
            f"knot spacing {info['knot_spacing']}, "
            f"guaranteed min|J| >= {info['min_jacobian_bound']:.4g})"
        ))
    else:
        logger.info((
            f'{context} C4 projection: no violations over '
            f"{info['n_pairs']} coefficient pairs, transform unchanged "
            f"(max |c| {info['max_coefficient_before']:.4g}, "
            f"guaranteed min|J| >= {info['min_jacobian_bound']:.4g})"
        ))
    return info


_CONFIG_KEYS = ('k', 'K', 'max_sweeps', 'tol')


def validate_deform_regularization_params(config, ndim, default_sweeps=300):
    """
    Validate and normalize a `bspline_constraints` configuration.

    Called before any expensive work so a bad configuration fails fast rather
    than after an optimization has run.

    Parameters
    ----------
    config : None or dict
        See `bigstream.align.deformable_align` for the accepted keys. A falsy
        value (None, {}, False) disables the constraint. Note that
        `max_displacement` is a sibling parameter of `deformable_align`, not
        a key of this dict - it shares no state with C4 and is validated
        separately by `validate_affine_displacement_bound`.

    ndim : int
        Image dimensionality, used to broadcast scalar `k`/`K`.

    default_sweeps : int (default: 300)
        `max_sweeps` to use when the config does not set one explicitly. The
        caller decides this - `deformable_align` picks 300 when its own
        `max_displacement` is set (POCS against both constraint sets
        converges slower than C4 alone) and 100 otherwise.

    Returns
    -------
    None if disabled, else a dict with normalized k, K, max_sweeps, tol.
    """
    if not config:
        return None
    if not isinstance(config, dict):
        raise ValueError(
            f'bspline_constraints must be a dict or None, got {config!r}'
        )
    unknown = set(config) - set(_CONFIG_KEYS)
    if unknown:
        raise ValueError(
            f'unknown bspline_constraints keys {sorted(unknown)}, '
            f'supported keys are {list(_CONFIG_KEYS)}. max_displacement is '
            "no longer one of them - it's deformable_align's own parameter."
        )

    k = _as_per_axis(
        DEFAULT_K if config.get('k') is None else config['k'], ndim, 'k')
    K = k if config.get('K') is None else _as_per_axis(config['K'], ndim, 'K')
    # validates sum(k) < 1; the knot spacing is irrelevant to that check
    coefficient_bounds(np.ones(ndim), k, K)
    max_sweeps = int(config.get('max_sweeps') or default_sweeps)
    tol = float(config['tol']) if config.get('tol') is not None else 1e-9
    if max_sweeps < 1:
        raise ValueError(f'max_sweeps must be >= 1, got {max_sweeps}')
    if tol <= 0:
        raise ValueError(f'tol must be positive, got {tol}')

    return {'k': k, 'K': K, 'max_sweeps': max_sweeps, 'tol': tol}


def is_orientation_preserving(affine_matrix):
    """
    True if a homogeneous affine matrix preserves orientation (det > 0).

    Not a regularizer and not used by the affine alignment steps: this exists
    so `deformable_align` can check the static transforms it composes with.
    The C4 guarantee applies to the b-spline warp; the composite is
    orientation preserving only if the transforms it is composed with are too.
    """
    matrix = np.asarray(affine_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f'expected a square affine matrix, got {matrix.shape}')
    return bool(np.linalg.det(matrix[:-1, :-1]) > 0)


def validate_affine_displacement_bound(max_displacement, ndim):
    """
    Validate and normalize a per-block affine `max_displacement` bound.

    An affine transform has no coefficient grid, so the C4 condition above
    does not apply to it (see the module docstring and
    `is_orientation_preserving`). The only fold risk it carries is amplitude:
    `distributed_align` blends per-block transforms together, and an affine
    that disagrees strongly with its neighbour folds on blend regardless of
    how "smooth" it is - see `blend_safe_displacement_bound` for the same
    reasoning applied to the deform step. This bound exists to cap that
    amplitude directly, the same way the deform step's `max_displacement`
    caps its coefficients.

    Parameters
    ----------
    max_displacement : None, float, or 1d array (zyx)
        None (the default) disables the bound and leaves `affine_align`
        behavior unchanged. Otherwise the per-component displacement bound,
        in the same physical units as the block's voxel spacing.

    ndim : int
        Image dimensionality, used to broadcast a scalar bound.

    Returns
    -------
    None if disabled, else a 1d array of length `ndim`.
    """
    if max_displacement is None:
        return None
    return _as_per_axis(max_displacement, ndim, 'max_displacement')


def bound_affine_displacement(matrix, extent, max_displacement, context=''):
    """
    Scale an affine matrix's deviation from identity so the displacement it
    induces over a block of given physical extent stays within
    `max_displacement`, per component.

    The displacement an affine `matrix` induces relative to identity is
    `d(x) = (L - I)x + t`, affine in `x`, where `L` is the linear part and
    `t` the translation. Being affine, its extrema over an axis-aligned box
    `[0, extent]` are attained at the box's corners, so the per-component
    worst case is found by evaluating there directly - no bound needs to be
    conservative here, unlike the coefficient-grid case in `project_to_c4`.

    If every component is already within bound, `matrix` is returned
    unchanged (a dead-zone operator, same philosophy as `project_to_c4`).
    Otherwise every component that violates its bound forces a single scalar
    `alpha = min(1, max_displacement / max|d|)` (the tightest across
    components), and the whole deviation from identity - both `L - I` and
    `t` - is scaled by `alpha`. Scaling uniformly rather than clipping
    individual matrix entries keeps the affine's relative shape (its
    rotation/shear/anisotropy pattern) intact, just dialed back toward
    identity, which is what "clamp into validity" should mean for a
    transform that has no natural per-entry bound of its own.

    Parameters
    ----------
    matrix : 2d array, (ndim+1, ndim+1)
        Homogeneous affine matrix, zyx order.

    extent : 1d array (zyx)
        Physical size of the block this affine applies to
        (`fix.shape * fix_spacing`), i.e. the box `[0, extent]` displacement
        is bounded over.

    max_displacement : float or 1d array (zyx)
        Per-component displacement bound, physical units. Use
        `validate_affine_displacement_bound` first to normalize/validate a
        user-supplied value.

    context : str (default: '')
        Prefix for log messages.

    Returns
    -------
    (matrix, info) : 2d array and dict
        `info` carries `max_displacement_before`, `max_displacement_bound`,
        `alpha`, `clamped`, and (only when clamped) `max_displacement_after`
        and `orientation_preserved`.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f'expected a square affine matrix, got {matrix.shape}')
    ndim = matrix.shape[0] - 1
    bound = _as_per_axis(max_displacement, ndim, 'max_displacement')
    extent = np.atleast_1d(np.asarray(extent, dtype=np.float64))
    if extent.size != ndim:
        raise ValueError(f'extent must have {ndim} values (zyx), got {extent}')

    linear = matrix[:ndim, :ndim]
    translation = matrix[:ndim, ndim]
    deviation = linear - np.eye(ndim)

    # d(x) = deviation @ x + translation is affine in x, so its extrema over
    # the box [0, extent] are attained at the box's corners
    corners = np.array(list(product(*[(0.0, e) for e in extent])))
    displacements_at_corners = corners @ deviation.T + translation
    max_abs = np.abs(displacements_at_corners).max(axis=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        ratios = np.where(max_abs > 0, bound / max_abs, np.inf)
    alpha = float(min(1.0, ratios.min()))

    info = {
        'max_displacement_before': max_abs.tolist(),
        'max_displacement_bound': bound.tolist(),
        'alpha': alpha,
        'clamped': alpha < 1.0,
    }

    if alpha >= 1.0:
        logger.info((
            f'{context} Affine displacement bound: '
            f'max |d|={max_abs.tolist()} within bound {bound.tolist()} '
            f'over extent {extent.tolist()}, transform unchanged'
        ))
        return matrix, info

    new_matrix = matrix.copy()
    new_matrix[:ndim, :ndim] = np.eye(ndim) + alpha * deviation
    new_matrix[:ndim, ndim] = alpha * translation
    info['max_displacement_after'] = (alpha * max_abs).tolist()
    info['orientation_preserved'] = is_orientation_preserving(new_matrix)
    if not info['orientation_preserved']:
        logger.error((
            f'{context} Affine displacement bound: scaling the deviation '
            f'from identity by alpha={alpha:.4g} produced a non '
            f'orientation-preserving affine (unexpected) - '
            f'det={np.linalg.det(new_matrix[:ndim, :ndim])}'
        ))
    logger.info((
        f'{context} Affine displacement bound: '
        f'max |d|={max_abs.tolist()} exceeded bound {bound.tolist()} '
        f'over extent {extent.tolist()}, scaled deviation from identity by '
        f'alpha={alpha:.4g} (max |d| after={info["max_displacement_after"]})'
    ))
    return new_matrix, info
