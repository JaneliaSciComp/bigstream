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
    disp = ut.numpy_to_sitk(
        field[..., ::-1].astype(np.float64),
        spacing, vector=True,
    )
    # Jacobian determinant: det(I + grad(u)); values <= 0 indicate folding
    jac = sitk.DisplacementFieldJacobianDeterminant(disp)
    jac_arr = sitk.GetArrayFromImage(jac)
    n_folded = int(np.count_nonzero(jac_arr <= 0))
    logger.info((
        f'{context} Deform align jacobian determinant: '
        f'min={jac_arr.min()}, max={jac_arr.max()}, '
        f'mean={jac_arr.mean()}, '
        f'folded voxels (det<=0)={n_folded} '
        f'({100.0 * n_folded / jac_arr.size:.4f}%)'
    ))

    if n_folded:
        # Distinguish real folding from the boundary artifact: a bspline
        # evaluates to zero displacement outside the region it supports, so
        # sampling the field anywhere past that region puts a step in it and
        # reports folding there no matter what the coefficients are. This
        # happens whenever the rendered grid is larger than the transform
        # domain, e.g. when alignment_spacing skip samples the images the
        # bspline domain was built from. Folding hugging the block faces is
        # that artifact; folding in the interior is a real fold.
        folded = np.argwhere(jac_arr <= 0)
        border_distance = np.minimum(
            folded, np.array(jac_arr.shape) - 1 - folded).min(axis=1)
        n_interior = int(np.count_nonzero(border_distance > 2))
        logger.info((
            f'{context} Deform align folding location: '
            f'{n_folded - n_interior} within 2 voxels of a face '
            f'(likely a transform domain boundary artifact), '
            f'{n_interior} in the interior '
            f'({100.0 * n_interior / jac_arr.size:.4f}% genuinely folded)'
        ))

    # field sanity and displacement magnitude statistics
    u = field
    mag = np.linalg.norm(u, axis=-1)
    logger.info((
        f'{context} Deform align field stats: '
        f'has NaN={bool(np.isnan(u).any())}, has Inf={bool(np.isinf(u).any())}, '
        f'disp magnitude min={mag.min()}, max={mag.max()}, '
        f'mean={mag.mean()}, p99={np.percentile(mag, 99)}'
    ))

    # crude discontinuity check in voxel index space
    gx = np.linalg.norm(np.diff(u, axis=2), axis=-1)
    gy = np.linalg.norm(np.diff(u, axis=1), axis=-1)
    gz = np.linalg.norm(np.diff(u, axis=0), axis=-1)
    for name, g in [("dx", gx), ("dy", gy), ("dz", gz)]:
        logger.info((
            f'{context} Deform align field smoothness {name}: '
            f'max jump={g.max()}, p99 jump={np.percentile(g, 99)}'
        ))


def dice_score(a, b, background=0):
    a, b = a > background, b > background
    a_and_b = np.logical_and(a, b).sum()
    a_sum = a.sum()
    b_sum = b.sum()
    logger.debug(f'a and b: {a_and_b}, as: {a_sum}, bs: {b_sum}')
    return 2. * a_and_b / max(a_sum + b_sum, 1)
