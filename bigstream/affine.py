import numpy as np
from bigstream import features
from bigstream import ransac
import dask.array as da
from ClusterWrap.clusters import janelia_lsf_cluster

WORKER_BUFFER = 8


def ransac_affine(
    fixed, moving,
    fixed_vox, moving_vox,
    min_radius,
    max_radius,
    match_threshold,
    cc_radius=12,
    nspots=5000,
    align_threshold=2.0,
    num_sigma_max=10,
):
    """
    """

    # get spots
    fixed_spots = features.blob_detection(
        fixed, min_radius, max_radius,
        num_sigma=max(max_radius-min_radius, num_sigma_max),
        threshold=0, exclude_border=cc_radius,
    )
    moving_spots = features.blob_detection(
        moving, min_radius, max_radius,
        num_sigma=max(max_radius-min_radius, num_sigma_max),
        threshold=0, exclude_border=cc_radius,
    )

    # sort
    sort_idx = np.argsort(fixed_spots[:, 3])[::-1]
    fixed_spots = fixed_spots[sort_idx, :3][:nspots]
    sort_idx = np.argsort(moving_spots[:, 3])[::-1]
    moving_spots = moving_spots[sort_idx, :3][:nspots]

    # convert to physical units
    fixed_spots = fixed_spots * fixed_vox
    moving_spots = moving_spots * moving_vox

    # get contexts
    fixed_spots = features.get_spot_context(
        fixed, fixed_spots, fixed_vox, cc_radius,
    )
    moving_spots = features.get_spot_context(
        moving, moving_spots, moving_vox, cc_radius,
    )

    # get point correspondences
    correlations = features.pairwise_correlation(
        fixed_spots, moving_spots,
    )
    fixed_spots, moving_spots = features.match_points(
        fixed_spots, moving_spots,
        correlations, match_threshold,
    )

    # align
    return ransac.ransac_align_points(
        fixed_spots, moving_spots, align_threshold,
    )


def interpolate_affines(affines):
    """
    """

    # TODO: replace identity matrices
    return affines


def tiled_ransac_affine(
    fixed, moving,
    fixed_vox, moving_vox,
    min_radius,
    max_radius,
    match_threshold,
    blocksize,
    cluster_kwargs={},
    **kwargs,
):
    """
    """

    # get number of blocks required
    block_grid = np.ceil(np.array(fixed.shape) / blocksize).astype(int)
    nblocks = np.prod(block_grid)
    overlap = [int(round(x/8)) for x in blocksize]

    # distributed computations done in cluster context
    with janelia_lsf_cluster(**cluster_kwargs) as cluster:
        cluster.scale_cluster(nblocks + WORKER_BUFFER)

        # wrap images as dask arrays
        fixed_da = da.from_array(fixed, chunks=blocksize)
        moving_da = da.from_array(moving, chunks=blocksize)

        # wrap affine function
        def wrapped_ransac_affine(x, y):
            affine = ransac_affine(
                x, y, fixed_vox, moving_vox,
                min_radius, max_radius, match_threshold,
                **kwargs,
            )
            return affine.reshape((1,1,1,3,4))

        # affine align all chunks
        affines = da.map_overlap(
            wrapped_ransac_affine, fixed_da, moving_da,
            depth=tuple(overlap),
            boundary='reflect',
            trim=False,
            align_arrays=False,
            dtype=np.float64,
            new_axis=[3, 4],
            chunks=[1, 1, 1, 3, 4],
        ).compute()

    # improve on identity affines
    affines = interpolate_affines(affines)

    # adjust affines for block origins
    for i in range(nblocks):
        x, y, z = np.unravel_index(i, block_grid)
        origin = (np.array(blocksize) * [x, y, z] - overlap) * fixed_vox
        tl, tr, a = np.eye(4), np.eye(4), np.eye(4)
        tl[:3, -1], tr[:3, -1] = origin, -origin
        a[:3, :] = affines[x, y, z]
        affines[x, y, z] = np.matmul(tl, np.matmul(a, tr))[:3, :]

    return affines

