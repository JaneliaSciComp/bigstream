import numpy as np
from bigstream import features
from bigstream import ransac
import dask.array as da


def ransac_affine(
    fix, mov,
    fix_spacing, mov_spacing,
    min_radius,
    max_radius,
    match_threshold,
    cc_radius=12,
    nspots=5000,
    align_threshold=2.0,
    num_sigma_max=15,
    verbose=True,
    fix_spots=None,
    mov_spots=None,
    default=np.eye(4),
    **kwargs,
):
    """
    """

    if verbose:
        print('Getting key points')

    # get spots
    if fix_spots is None:
        fix_spots = features.blob_detection(
            fix, min_radius, max_radius,
            num_sigma=min(max_radius-min_radius, num_sigma_max),
            threshold=0, exclude_border=cc_radius,
        )
        if fix_spots.shape[0] < 50:
            print('Fewer than 50 spots found in fixed image, returning default')
            return default
        if verbose:
            ns = fix_spots.shape[0]
            print(f'FIXED image: found {ns} key points')

    if mov_spots is None:
        mov_spots = features.blob_detection(
            mov, min_radius, max_radius,
            num_sigma=min(max_radius-min_radius, num_sigma_max),
            threshold=0, exclude_border=cc_radius,
        )
        if mov_spots.shape[0] < 50:
            print('Fewer than 50 spots found in moving image, returning default')
            return default
        if verbose:
            ns = mov_spots.shape[0]
            print(f'MOVING image: found {ns} key points')

    # sort
    sort_idx = np.argsort(fix_spots[:, 3])[::-1]
    fix_spots = fix_spots[sort_idx, :3][:nspots]
    sort_idx = np.argsort(mov_spots[:, 3])[::-1]
    mov_spots = mov_spots[sort_idx, :3][:nspots]

    # convert to physical units
    fix_spots = fix_spots * fix_spacing
    mov_spots = mov_spots * mov_spacing

    # get contexts
    fix_spots = features.get_spot_context(
        fix, fix_spots, fix_spacing, cc_radius,
    )
    mov_spots = features.get_spot_context(
        mov, mov_spots, mov_spacing, cc_radius,
    )

    # get point correspondences
    correlations = features.pairwise_correlation(
        fix_spots, mov_spots,
    )
    fix_spots, mov_spots = features.match_points(
        fix_spots, mov_spots,
        correlations, match_threshold,
    )
    if verbose:
        ns = fix_spots.shape[0]
        print(f'MATCHED points: found {ns} matched points')

    # align
    return ransac.ransac_align_points(
        fix_spots, mov_spots, align_threshold, **kwargs,
    )


def prepare_piecewise_ransac_affine(
    fix, mov,
    fix_spacing, mov_spacing,
    min_radius,
    max_radius,
    match_threshold,
    blocksize,
    **kwargs,
):
    """
    """

    # get number of blocks required
    block_grid = np.ceil(np.array(fix.shape) / blocksize).astype(int)
    nblocks = np.prod(block_grid)
    overlap = [int(round(x/8)) for x in blocksize]

    # wrap images as dask arrays
    fix_da = da.from_array(fix, chunks=blocksize)
    mov_da = da.from_array(mov, chunks=blocksize)

    # wrap affine function
    def wrapped_ransac_affine(x, y, block_info=None):

        # compute affine
        affine = ransac_affine(
            x, y, fix_spacing, mov_spacing,
            min_radius, max_radius, match_threshold,
            **kwargs,
        )

        # adjust for block origin
        idx = np.array(block_info[0]['chunk-location'])
        origin = (idx * blocksize - overlap) * fix_spacing
        tl, tr = np.eye(4), np.eye(4)
        tl[:3, -1], tr[:3, -1] = origin, -origin
        affine = np.matmul(tl, np.matmul(affine, tr))

        # return with block index axes
        return affine.reshape((1,1,1,4,4))

    # affine align all chunks
    return da.map_overlap(
        wrapped_ransac_affine, fix_da, mov_da,
        depth=tuple(overlap),
        boundary='reflect',
        trim=False,
        align_arrays=False,
        dtype=np.float64,
        new_axis=[3, 4],
        chunks=[1, 1, 1, 4, 4],
    )


def interpolate_affines(affines):
    """
    """

    # get block grid
    block_grid = affines.shape[:3]

    # construct an all identities matrix for comparison
    all_identities = np.empty_like(affines)
    for i in range(np.prod(block_grid)):
        idx = np.unravel_index(i, block_grid)
        all_identities[idx] = np.eye(4)

    # if affines are all identity, just return
    if np.all(affines == all_identities):
        return affines

    # process continues until there are no identity matrices left
    new_affines = np.copy(affines)
    identities = True
    while identities:
        identities = False

        # loop over all affine matrices
        for i in range(np.prod(block_grid)):
            idx = np.unravel_index(i, block_grid)

            # if an identity matrix is found
            if np.all(new_affines[idx] == np.eye(4)):
                identities = True
                trans, denom = np.array([0, 0, 0]), 0

                # average translations from 6 connected neighborhood
                for ax in range(3):
                    if idx[ax] > 0:
                        neighbor = tuple(
                            x-1 if j == ax else x for j, x in enumerate(idx)
                        )
                        neighbor_trans = new_affines[neighbor][:3, -1]
                        if not np.all(neighbor_trans == 0):
                            trans = trans + neighbor_trans
                            denom += 1
                    if idx[ax] < block_grid[ax]-1:
                        neighbor = tuple(
                            x+1 if j == ax else x for j, x in enumerate(idx)
                        )
                        neighbor_trans = new_affines[neighbor][:3, -1]
                        if not np.all(neighbor_trans == 0):
                            trans = trans + neighbor_trans
                            denom += 1

                # normalize then update matrix
                if denom > 0: trans /= denom
                new_affines[idx][:3, -1] = trans

    return new_affines

