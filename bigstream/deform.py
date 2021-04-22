import numpy as np
import greedypy.greedypy_registration_method as gprm
from bigstream import stitch
from bigstream import transform
import dask.array as da
import zarr
from numcodecs import Blosc
from itertools import product
import ClusterWrap


def deformable_align(
    fix, mov,
    fix_spacing, mov_spacing,
    cc_radius=12,
    gradient_smoothing=[3., 0., 1., 2.],
    field_smoothing=[0.5, 0., 1., 6.],
    iterations=[200, 100],
    shrink_factors=[2, 1],
    smooth_sigmas=[4, 2],
    step=None,
):
    """
    """

    fix = fix.astype(np.float32)
    mov = mov.astype(np.float32)

    register = gprm.greedypy_registration_method(
        fix, fix_spacing,
        mov, mov_spacing,
        iterations,
        shrink_factors,
        smooth_sigmas,
        radius=cc_radius,
        gradient_abcd=gradient_smoothing,
        field_abcd=field_smoothing,
        step=step,
    )

    register.mask_values(0)
    register.optimize()
    return register.get_warp()


def tiled_deformable_align(
    fix, mov,
    fix_spacing, mov_spacing,
    blocksize,
    transpose=[False]*2,
    global_affine=None,
    local_affines=None,
    write_path=None,
    lazy=True,
    deform_kwargs={},
    cluster_kwargs={},
):
    """
    """

    # get number of blocks required
    block_grid = np.ceil(np.array(fix.shape) / blocksize)
    nblocks = np.prod(block_grid)

    # get true field shape
    original_shape = fix.shape
    if transpose[0]:
        original_shape = original_shape[::-1]

    # get affine position field
    affine_pf = None
    if global_affine is not None or local_affines is not None:
        if local_affines is None:
            local_affines = np.empty(
                block_grid + (4,4), dtype=np.float32,
            )
            local_affines[..., :, :] = np.eye(4)
        affine_pf = transform.local_affines_to_position_field(
            original_shape, fix_spacing, blocksize,
            local_affines, global_affine=global_affine,
            lazy=True, cluster_kwargs=cluster_kwargs,
        )

    # distributed computations done in cluster context
    with ClusterWrap.cluster(**cluster_kwargs) as cluster:
        if write_path is not None or not lazy:
            cluster.scale_cluster(nblocks + 1)

        # wrap images as dask arrays
        fix_da = da.from_array(fix)
        mov_da = da.from_array(mov)

        # in case xyz convention is flipped for input file
        if transpose[0]:
            fix_da = fix_da.transpose(2,1,0)
        if transpose[1]:
            mov_da = mov_da.transpose(2,1,0)

        # pad the ends to fill in the last blocks
        pads = []
        for x, y in zip(original_shape, blocksize):
            pads += [(0, y - x % y) if x % y > 0 else (0, 0)]
        fix_da = da.pad(fix_da, pads)
        mov_da = da.pad(mov_da, pads)

        # chunk to blocksize
        fix_da = fix_da.rechunk(tuple(blocksize))
        mov_da = mov_da.rechunk(tuple(blocksize))

        # wrap deformable function
        def wrapped_deformable_align(x, y):
            warp = deformable_align(
                x, y, fix_spacing, mov_spacing,
                **deform_kwargs,
            )
            return warp.reshape((1,1,1)+warp.shape)

        # deform all chunks
        overlaps = tuple([int(round(x/8)) for x in blocksize])
        out_blocks = [x + 2*y for x, y in zip(blocksize, overlaps)]
        out_blocks = [1,1,1] + out_blocks + [3,]

        warps = da.map_overlap(
            wrapped_deformable_align, fix_da, mov_da,
            depth=overlaps,
            boundary=0,
            trim=False,
            align_arrays=False,
            dtype=np.float32,
            new_axis=[3,4,5,6,],
            chunks=out_blocks,
        )

        # stitch neighboring displacement fields
        warps = stitch.stitch_fields(warps, blocksize)

        # crop any pads
        warps = warps[:original_shape[0],
                      :original_shape[1],
                      :original_shape[2]]


        # TODO refactor transform.compose_position_fields
        #      replace this approximation
        # compose with affine position field
        if affine_pf is not None:
            final_field = affine_pf + warps
        else:
            final_field = warps + transform.position_grid_dask(
                original_shape, blocksize,
            )

        # if user wants to write to disk
        if write_path is not None:
            compressor = Blosc(cname='zstd', clevel=9, shuffle=Blosc.BITSHUFFLE)
            final_field_disk = zarr.open(write_path, 'w',
                shape=final_field.shape, chunks=tuple(blocksize + [3,]),
                dtype=final_field.dtype, compressor=compressor,
            )
            da.to_zarr(final_field, final_field_disk)

        # if user wants to compute and return full field
        if not lazy:
            return final_field.compute()

        # if user wants to return compute graph w/o executing
        if lazy:
            return final_field

