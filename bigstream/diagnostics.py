import logging
import numpy as np
import SimpleITK as sitk

import bigstream.utility as ut


logger = logging.getLogger(__name__)


def deform_field_diagnostics(field, spacing, context=''):
    """
    Log diagnostics for a displacement vector field: jacobian determinant
    (folding), field sanity (NaN/Inf), displacement magnitude statistics, and
    a crude discontinuity check in voxel index space.

    Parameters
    ----------
    field : nd-array
        The displacement vector field in zyx order, with the last axis holding
        the displacement components in zyx order.

    spacing : 1d array
        The physical voxel spacing of the field (zyx order).

    context : str (default: '')
        A prefix prepended to all log messages.
    """

    # build a sitk displacement field image (xyz vector order) from the
    # numpy field (zyx order, components in zyx)
    # float32 rather than float64: the jacobian determinant comes out
    # identical, and this is a logging-only check that was otherwise one of
    # the largest single allocations in a block's alignment
    disp = ut.numpy_to_sitk(
        np.ascontiguousarray(field[..., ::-1], dtype=np.float32),
        spacing, vector=True,
    )
    # Jacobian determinant: det(I + grad(u)); values <= 0 indicate folding
    jac = sitk.DisplacementFieldJacobianDeterminant(disp)
    del disp
    jac_arr = sitk.GetArrayViewFromImage(jac)
    n_folded = int(np.count_nonzero(jac_arr <= 0))
    logger.info((
        f'{context} Deform align jacobian determinant: '
        f'min={jac_arr.min()}, max={jac_arr.max()}, '
        f'mean={jac_arr.mean()}, '
        f'folded voxels (det<=0)={n_folded} '
        f'({100.0 * n_folded / jac_arr.size:.4f}%)'
    ))

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
        n_face = n_folded - n_interior
        logger.info((
            f'{context} Deform align folding location: '
            f'{n_face} within 2 voxels of a face, '
            f'{n_interior} in the interior '
            f'({100.0 * n_interior / jac_arr.size:.4f}% of the block)'
        ))
        if n_interior:
            logger.error((
                f'{context} Deform align has {n_interior} interior folded '
                'voxels - the deformation is not locally invertible'
            ))
        if n_face:
            logger.error((
                f'{context} Deform align has {n_face} folded voxels at a block '
                'face. These blend into neighbouring blocks, so they are not '
                'self correcting; check the field smoothness maxima below for '
                'a step at the domain boundary'
            ))

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
    logger.info((
        f'{context} Deform align field stats: '
        f'has NaN={has_nan}, has Inf={has_inf}, '
        f'disp magnitude min={mag.min()}, max={mag.max()}, '
        f'mean={mag.mean()}, p99={np.percentile(mag, 99)}'
    ))
    del mag

    # crude discontinuity check in voxel index space, one axis at a time: the
    # diff of a vector field is nearly as large as the field itself, so
    # holding dx, dy and dz at once tripled the footprint of this check.
    for name, axis in (("dx", 2), ("dy", 1), ("dz", 0)):
        g = np.linalg.norm(np.diff(u, axis=axis), axis=-1)
        logger.info((
            f'{context} Deform align field smoothness {name}: '
            f'max jump={g.max()}, p99 jump={np.percentile(g, 99)}'
        ))
        del g


def dice_score(a, b, background=0):
    a, b = a > background, b > background
    a_and_b = np.logical_and(a, b).sum()
    a_sum = a.sum()
    b_sum = b.sum()
    logger.debug(f'a and b: {a_and_b}, as: {a_sum}, bs: {b_sum}')
    return 2. * a_and_b / max(a_sum + b_sum, 1)
