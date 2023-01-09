import numpy as np
import SimpleITK as sitk
from bigstream.configure_irm import configure_irm
import bigstream.utility as ut


def patch_mutual_information(
    fix,
    mov,
    spacing,
    radius,
    stride,
    percentile_cutoff=0,
    fix_mask=None,
    mov_mask=None,
    return_metric_image=False,
    **kwargs,
):
    """
    Local mutual information metric between two images
    MI is computed patch-wise across both images and the mean over all
    patches is returned

    Parameters
    ----------
    fix : nd-array
        fixed image
    mov : nd-array
        moving image
    spacing : 1d-array
        The voxel spacing of the two images (must be the same)
    radius : scalar float
        Neighborhood half-width in physical units
    stride : scalar int
        Spacing between neighborhood centers
    percentile_cutoff : scalar float (default: 0)
        local MI scores below this value are ignored in final mean computation
    fix_mask : binary nd-array (default: None)
        mask over fixed data (only data in foreground is considered)
    mov_mask : binary nd-array (default: None)
        mask over moving data (only data in foreground is considered)
    return_metric_image : bool (default: False)
        Return an image with local MIs
    **kwargs : any additional arguments
        Passed to bigstream.configure_irm.configure_irm
        Use these arguments to parameterize the metric

    Returns
    -------
    score : scalar float
        The local MI averaged over all patches
    metric_image : nd-array
        Optional output only returned if return_metric_image == True
        The local MIs rendered in an image
    """

    # create sitk versions of data
    fix_sitk = ut.numpy_to_sitk(fix.transpose(2, 1, 0), spacing[::-1])
    fix_sitk = sitk.Cast(fix_sitk, sitk.sitkFloat32)
    mov_sitk = ut.numpy_to_sitk(mov.transpose(2, 1, 0), spacing[::-1])
    mov_sitk = sitk.Cast(mov_sitk, sitk.sitkFloat32)

    # determine patch sample centers
    samples = np.zeros_like(fix)
    radius = np.round(radius / spacing).astype(np.uint16)
    stride = np.round(stride / spacing).astype(np.uint16)
    samples[tuple(slice(r, -r, s) for r, s in zip(radius, stride))] = 1
    if fix_mask is not None: samples = samples * fix_mask
    if mov_mask is not None: samples = samples * mov_mask
    samples = np.column_stack(np.nonzero(samples))

    # create irm and containers for results
    irm = configure_irm(**kwargs)
    if return_metric_image:
        metric_image = np.zeros(fix.shape, dtype=np.float32)
    scores = []

    # score all blocks
    for sample in samples:
        # get patches
        patch = tuple(slice(s-r, s+r+1) for s, r in zip(sample, radius))
        f = fix_sitk[patch]
        m = mov_sitk[patch]
        # evaluate metric
        try:
            scores.append( irm.MetricEvaluate(f, m) )
        except Exception as e:
            scores.append( 0 )
        # update metric image
        if return_metric_image:
            metric_image[patch] = scores[-1]

    # threshold scores
    scores = np.array(scores)
    if percentile_cutoff > 0:
        cutoff = np.percentile(-scores, percentile_cutoff)
        scores = scores[-scores > cutoff]

    # return results
    if return_metric_image:
        return np.mean(scores), metric_image
    else:
        return np.mean(scores)

