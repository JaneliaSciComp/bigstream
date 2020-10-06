import numpy as np
import greedypy.greedypy_registration_method as gprm
from bigstream import distributed
import dask.array as da


def deformable_align(
    fixed, moving,
    fixed_vox, moving_vox,
    cc_radius,
    gradient_smoothing=[3., 0. 1., 2.],
    field_smoothing=[0.5, 0., 1., 6.],
    iterations=[200, 100],
    shrink_factors=[2, 1],
    smooth_sigmas=[1, 0],
    step=5.0,
):
    """
    """

    register = gprm.greedypy_registration_method(
        fixed, fixed_vox,
        moving, moving_vox,
        iterations,
        shrink_factors,
        smooth_sigmas,
        radius=cc_radius,
        gradient_abcd=gradien_smoothing,
        field_abcd=field_smoothing,
    )

    register.mask_values(0)
    register.optimize()
    return register.get_warp()


def deformable_align_distributed(
    fixed, moving,
    fixed_vox, moving_vox,
    cc_radius,
    gradient_smoothing,
    field_smoothing,
    iterations,
    shrink_factors,
    smooth_sigmas,
    step,
    blocksize=[256,]*3,
    cluster_extra=["-P multifish"],
):
    """
    """

    # get number of blocks required
    block_grid = np.ceil(np.array(fixed.shape) / blocksize)
    nblocks = np.prod(block_grid)

    # distributed computations done in cluster context
    # TODO: generalize w.r.t. workstations and cluster managers
    with distributed.distributedState() as ds:

        # set up the cluster
        ds.initializeLSFCluster(job_extra=cluster_extra)
        ds.initializeClient()
        ds.scaleCluster(njobs=nblocks)

        # wrap images as dask arrays
        fixed_da = da.from_array(fixed, chunks=blocksize)
        moving_da = da.from_array(moving, chunks=blocksize)

        # deform all chunks
        my_deformable_align = lambda x, y: deformable_align(
            x, y, fixed_vox, moving_vox, cc_radius, gradient_smoothing,
            field_smoothing, iterations, shrink_factors, smooth_sigmas, step,
        )
        deforms = da.map_overlap(
