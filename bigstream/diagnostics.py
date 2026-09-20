import logging
import numpy as np
import SimpleITK as sitk

import bigstream.utility as ut


logger = logging.getLogger(__name__)


def deform_field_diagnostics(field, spacing, context='', level=None):
    """
    Compute diagnostics for a displacement vector field: jacobian determinant
    (folding), field sanity (NaN/Inf), displacement magnitude statistics, and
    a crude discontinuity check in voxel index space.

    This **returns** the statistics rather than only logging them, so it can
    run on a dask worker and have the caller report the result. Everything
    returned is a scalar, so the dict is a few hundred bytes however large
    the field was - cheap to ship back through the scheduler. Pass `level`
    to also log on the spot; `log_deform_field_diagnostics` renders the same
    lines from the returned dict.

    Note that this always computes. The jacobian determinant is one of the
    largest allocations in a block's alignment, so a caller that only wants
    the numbers for a log line nobody will read should not call at all -
    see the `logger.isEnabledFor` guard on the per-block call site.

    Parameters
    ----------
    field : nd-array
        The displacement vector field in zyx order, with the last axis holding
        the displacement components in zyx order.

    spacing : 1d array
        The physical voxel spacing of the field (zyx order).

    context : str (default: '')
        A prefix prepended to the log messages, when `level` asks for any.

    level : int (default: None)
        When given, log the statistics at this level before returning them.
        Below INFO the fold reports are demoted with everything else, which
        is what a per-block call wants: a block's field is not the field
        that ends up on disk - its overlap regions only reach their final
        values once every neighbour has added its weighted share - so a fold
        seen there is not yet a fold in the result. The run on the assembled
        field is the one that decides.

    Returns
    -------
    stats : dict of scalars
        See `log_deform_field_diagnostics` for what each key means.
    """
    # build a sitk displacement field image (xyz vector order) from the
    # numpy field (zyx order, components in zyx)
    # float32 rather than float64: the jacobian determinant comes out
    # identical, and this is a diagnostic-only check that was otherwise one
    # of the largest single allocations in a block's alignment
    disp = ut.numpy_to_sitk(
        np.ascontiguousarray(field[..., ::-1], dtype=np.float32),
        spacing, vector=True,
    )
    # Jacobian determinant: det(I + grad(u)); values <= 0 indicate folding
    jac = sitk.DisplacementFieldJacobianDeterminant(disp)
    del disp
    jac_arr = sitk.GetArrayViewFromImage(jac)
    n_folded = int(np.count_nonzero(jac_arr <= 0))
    stats = {
        'size': int(jac_arr.size),
        'jacobian_min': float(jac_arr.min()),
        'jacobian_max': float(jac_arr.max()),
        'jacobian_mean': float(jac_arr.mean()),
        'n_folded': n_folded,
        'n_face_folded': 0,
        'n_interior_folded': 0,
    }

    if n_folded:
        # Locate the folding, but do not explain it away. Folding at a block
        # face used to be reported as "likely a transform domain boundary
        # artifact" and ignored - which is exactly how a real one voxel fold
        # sheet survived several production runs (a bspline rendered outside
        # its valid region returns zero displacement, putting a full amplitude
        # step at the face). `bspline_to_displacement_field` now extends the
        # field past the domain instead, so face folding is NOT expected and
        # is reported as an error wherever it carries a real displacement.
        folded = np.argwhere(jac_arr <= 0)
        border_distance = np.minimum(
            folded, np.array(jac_arr.shape) - 1 - folded).min(axis=1)
        n_interior = int(np.count_nonzero(border_distance > 2))
        stats['n_interior_folded'] = n_interior
        stats['n_face_folded'] = n_folded - n_interior

    del jac_arr, jac

    # field sanity and displacement magnitude statistics. The overwhelmingly
    # common case is an all-finite field, so establish that with a single
    # temporary and only pay for the NaN/Inf breakdown when there is actually
    # something to break down.
    u = field
    if np.isfinite(u).all():
        has_nan = has_inf = False
    else:
        has_nan = bool(np.isnan(u).any())
        has_inf = bool(np.isinf(u).any())
    mag = np.linalg.norm(u, axis=-1)
    stats.update({
        'has_nan': has_nan,
        'has_inf': has_inf,
        'magnitude_min': float(mag.min()),
        'magnitude_max': float(mag.max()),
        'magnitude_mean': float(mag.mean()),
        'magnitude_p99': float(np.percentile(mag, 99)),
    })
    del mag

    # crude discontinuity check in voxel index space, one axis at a time: the
    # diff of a vector field is nearly as large as the field itself, so
    # holding dx, dy and dz at once tripled the footprint of this check.
    smoothness = {}
    for name, axis in (("dx", 2), ("dy", 1), ("dz", 0)):
        g = np.linalg.norm(np.diff(u, axis=axis), axis=-1)
        smoothness[name] = {'max_jump': float(g.max()),
                            'p99_jump': float(np.percentile(g, 99))}
        del g
    stats['smoothness'] = smoothness

    if level is not None:
        log_deform_field_diagnostics(stats, context=context, level=level)
    return stats


def log_deform_field_diagnostics(stats, context='', level=logging.INFO):
    """
    Render the lines `deform_field_diagnostics` used to emit itself.

    Split out so the numbers can be computed on a worker and reported by
    whoever gathers them, without the formatting drifting between the two.
    """
    if not logger.isEnabledFor(level):
        return
    fold_level = logging.ERROR if level >= logging.INFO else level

    size = stats['size']
    n_folded = stats['n_folded']
    logger.log(level, (
        f'{context} Deform align jacobian determinant: '
        f'min={stats["jacobian_min"]}, max={stats["jacobian_max"]}, '
        f'mean={stats["jacobian_mean"]}, '
        f'folded voxels (det<=0)={n_folded} '
        f'({100.0 * n_folded / size:.4f}%)'
    ))

    if n_folded:
        n_interior = stats['n_interior_folded']
        n_face = stats['n_face_folded']
        logger.log(level, (
            f'{context} Deform align folding location: '
            f'{n_face} within 2 voxels of a face, '
            f'{n_interior} in the interior '
            f'({100.0 * n_interior / size:.4f}% of the block)'
        ))
        if n_interior:
            logger.log(fold_level, (
                f'{context} Deform align has {n_interior} interior folded '
                'voxels - the deformation is not locally invertible'
            ))
        if n_face:
            logger.log(fold_level, (
                f'{context} Deform align has {n_face} folded voxels at a block '
                'face. These blend into neighbouring blocks, so they are not '
                'self correcting; check the field smoothness maxima below for '
                'a step at the domain boundary'
            ))

    logger.log(level, (
        f'{context} Deform align field stats: '
        f'has NaN={stats["has_nan"]}, has Inf={stats["has_inf"]}, '
        f'disp magnitude min={stats["magnitude_min"]}, '
        f'max={stats["magnitude_max"]}, '
        f'mean={stats["magnitude_mean"]}, p99={stats["magnitude_p99"]}'
    ))

    for name, jumps in stats['smoothness'].items():
        logger.log(level, (
            f'{context} Deform align field smoothness {name}: '
            f'max jump={jumps["max_jump"]}, p99 jump={jumps["p99_jump"]}'
        ))


def dice_score(a, b, background=0):
    a, b = a > background, b > background
    a_and_b = np.logical_and(a, b).sum()
    a_sum = a.sum()
    b_sum = b.sum()
    logger.debug(f'a and b: {a_and_b}, as: {a_sum}, bs: {b_sum}')
    return 2. * a_and_b / max(a_sum + b_sum, 1)
