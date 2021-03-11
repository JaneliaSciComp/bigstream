import numpy as np
import greedypy.greedypy_registration_method as gprm
from bigstream import stitch
from bigstream import transform
import dask.array as da
import zarr
from numcodecs import Blosc
from itertools import product
from ClusterWrap.clusters import janelia_lsf_cluster


def deformable_align(
    fixed, moving,
    fixed_vox, moving_vox,
    cc_radius,
    gradient_smoothing=[3., 0., 1., 2.],
    field_smoothing=[0.5, 0., 1., 6.],
    iterations=[200, 100],
    shrink_factors=[2, 1],
    smooth_sigmas=[4, 2],
    step=1.0,
):
    """
    """

    fixed = fixed.astype(np.float32)
    moving = moving.astype(np.float32)

    register = gprm.greedypy_registration_method(
        fixed, fixed_vox,
        moving, moving_vox,
        iterations,
        shrink_factors,
        smooth_sigmas,
        radius=cc_radius,
        gradient_abcd=gradient_smoothing,
        field_abcd=field_smoothing,
    )

    register.mask_values(0)
    register.optimize()
    warp = register.get_warp()
    return warp.reshape((1,1,1)+warp.shape)


def deformable_align_distributed(
    fixed, moving,
    fixed_vox, moving_vox,
    write_path,
    cc_radius,
    gradient_smoothing,
    field_smoothing,
    iterations,
    shrink_factors,
    smooth_sigmas,
    step,
    blocksize=[256,]*3,
    transpose=False,
    cluster_kwargs={},
):
    """
    """

    # get number of blocks required
    block_grid = np.ceil(np.array(fixed.shape) / blocksize)
    nblocks = np.prod(block_grid)

    # we need bigger workers for deformable alignments
    cluster_kwargs["cores"] = 4

    # distributed computations done in cluster context
    with janelia_lsf_cluster(**cluster_kwargs) as cluster:
        cluster.scale_cluster(nblocks)

        # wrap images as dask arrays
        fixed_da = da.from_array(fixed)
        moving_da = da.from_array(moving)

        # in case xyz convention is flipped for input file
        if transpose:
            fixed_da = fixed_da.transpose(2,1,0)

        # pad the ends to fill in the last blocks
        orig_sh = fixed_da.shape
        pads = [(0, y - x % y) if x % y != 0 else (0, 0) for x, y in zip(orig_sh, blocksize)]
        fixed_da = da.pad(fixed_da, pads)
        moving_da = da.pad(moving_da, pads)
        fixed_da = fixed_da.rechunk(tuple(blocksize))
        moving_da = moving_da.rechunk(tuple(blocksize))

        # wrap deformable function to simplify passing parameters
        def my_deformable_align(x, y):
            return deformable_align(
                x, y, fixed_vox, moving_vox,
                cc_radius, gradient_smoothing, field_smoothing,
                iterations, shrink_factors, smooth_sigmas, step,
            )

        # deform all chunks
        overlaps = tuple([int(round(x/8)) for x in blocksize])
        out_blocks = [1,1,1] + [x + 2*y for x, y in zip(blocksize, overlaps)] + [3,]

        warps = da.map_overlap(
            my_deformable_align, fixed_da, moving_da,
            depth=overlaps,
            boundary='reflect',
            trim=False,
            align_arrays=False,
            dtype=np.float32,
            new_axis=[3,4,5,6,],
            chunks=out_blocks,
        )

        # stitch neighboring displacement fields
        warps = stitch.stitch_fields(warps, blocksize)

        # crop any pads
        warps = warps[:orig_sh[0], :orig_sh[1], :orig_sh[2]]

        # convert to position field
        warps = warps + transform.position_grid_dask(orig_sh, blocksize)

        # write result to zarr file
        compressor = Blosc(cname='zstd', clevel=9, shuffle=Blosc.BITSHUFFLE)
        warps_disk = zarr.open(write_path, 'w',
            shape=warps.shape, chunks=tuple(blocksize + [3,]),
            dtype=warps.dtype, compressor=compressor,
        )
        da.to_zarr(warps, warps_disk)

        # return reference to zarr data store
        return warps_disk

