import numpy as np
from bigstream import dog_spots
from bigstream import cv2_ransac
from bigstream import distributed
import dask.array as da


def dog_ransac_affine(
    fixed, moving,
    fixed_vox, moving_vox,
    cc_radius,
    nspots,
    match_threshold,
    align_threshold,
):
    """
    """

    # get spots
    fixed_spots = dog_spots.dog_filter_3d(fixed)
    moving_spots = dog_spots.dog_filter_3d(moving)

    # need a practical number of spots
    spot_limit = 100000
    if fixed_spots.shape[0] > spot_limit:
        sort_idx = np.argsort(fixed_spots[:, 3])[::-1]
        fixed_spots = fixed_spots[sort_idx, :][:spot_limit]
    if moving_spots.shape[0] > spot_limit:
        sort_idx = np.argsort(moving_spots[:, 3])[::-1]
        moving_spots = moving_spots[sort_idx, :][:spot_limit]


    # filter overlapping blobs
    fixed_spots = dog_spots.prune_neighbors(fixed_spots)
    moving_spots = dog_spots.prune_neighbors(moving_spots)

    # sort
    sort_idx = np.argsort(fixed_spots[:, 3])[::-1]
    fixed_spots = fixed_spots[sort_idx, :][:nspots]
    sort_idx = np.argsort(moving_spots[:, 3])[::-1]
    moving_spots = moving_spots[sort_idx, :][:nspots]

    # get contexts
    fixed_spots = dog_spots.get_all_context(fixed, fixed_spots, cc_radius)
    moving_spots = dog_spots.get_all_context(moving, moving_spots, cc_radius)

    # get point correspondences
    correlations = cv2_ransac.pairwise_correlation(fixed_spots, moving_spots)
    fixed_spots, moving_spots = cv2_ransac.match_points(
        fixed_spots, moving_spots, correlations, match_threshold,
    )

    # align
    return cv2_ransac.ransac_align_points(fixed_spots, moving_spots, align_threshold)


def interpolate_affines(affines):
    """
    """

    # TODO: replace identity matrices
    return affines


def dog_ransac_affine_distributed(
    fixed, moving,
    fixed_vox, moving_vox,
    cc_radius,
    nspots,
    match_threshold,
    align_threshold,
    blocksize=[256,]*3,
    cluster_extra=["-P multifish",]
):
    """
    """

    # get number of blocks required
    block_grid = np.ceil(np.array(fixed.shape) / blocksize).astype(int)
    nblocks = np.prod(block_grid)

    # distributed computations done in cluster context
    # TODO: generalize w.r.t. workstations and cluster managers
    with distributed.distributedState() as ds:

        # set up the cluster
        ds.initializeLSFCluster(job_extra=cluster_extra)
        ds.initializeClient()
        ds.scaleCluster(njobs=nblocks)

        # wrap images as dask arrays
        fixed_da = da.from_array(fixed, chunks=blocksize).transpose(2,1,0)
        moving_da = da.from_array(moving, chunks=blocksize).transpose(2,1,0)

        # wrap affine function
        def my_dog_ransac_affine(x, y):
            affine = dog_ransac_affine(
                x, y, fixed_vox, moving_vox,
                cc_radius, nspots, match_threshold, align_threshold,
            )
            return affine.reshape((1,1,1,3,4))

        # affine align all chunks
        affines = da.map_overlap(
            my_dog_ransac_affine, fixed_da, moving_da,
            depth=tuple([int(round(x/8)) for x in blocksize]),
            boundary='reflect',
            trim=False,
            align_arrays=False,
            dtype=np.float64,
            new_axis=[3, 4],
            chunks=[1, 1, 1, 3, 4],
        ).compute()

    return interpolate_affines(affines)

