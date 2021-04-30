import numpy as np
import greedypy.greedypy_registration_method as gprm
import dask_stitch as ds
from bigstream import transform
import dask.array as da


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


def prepare_piecewise_deformable_align(
    fix, mov,
    fix_spacing, mov_spacing,
    blocksize,
    transpose=[False]*2,
    global_affine=None,
    local_affines=None,
    **kwargs,
):
    """
    """

    # get number of blocks required
    block_grid = np.ceil(np.array(fix.shape) / blocksize)

    # get true field shape
    original_shape = fix.shape
    if transpose[0]:
        original_shape = original_shape[::-1]

    # compose global/local affines
    total_affines = None
    if local_affines is not None and global_affine is not None:
        total_affines = transform.compose_affines(
            global_affine, local_affines,
        )
    elif global_affine is not None:
        total_affines = np.empty(tuple(block_grid) + (4, 4))
        total_affines[..., :, :] = global_affine
    elif local_affines is not None:
        total_affines = np.copy(local_affines)

    # get affine position field
    overlap = tuple([int(round(x/8)) for x in blocksize])
    affine_pf = None
    if total_affines is not None:
        affine_pf = ds.local_affine.local_affines_to_field(
            original_shape, fix_spacing, total_affines,
            blocksize, overlap,
            displacement=False,
        )

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
        return deformable_align(
            x, y, fix_spacing, mov_spacing,
            **kwargs,
        )

    # deform all chunks
    out_blocks = [x + 2*y for x, y in zip(blocksize, overlap)] + [3,]
    warps = da.map_overlap(
        wrapped_deformable_align,
        fix_da, mov_da,
        depth=overlap,
        boundary=0,
        trim=False,
        align_arrays=False,
        dtype=np.float32,
        new_axis=[3,],
        chunks=out_blocks,
    )

    # stitch neighboring displacement fields
    warps = ds.stitch.stitch_blocks(warps, blocksize, overlap)

    # crop any pads
    warps = warps[:original_shape[0],
                  :original_shape[1],
                  :original_shape[2]]

    # compose with affine position field
    # TODO refactor transform.compose_position_fields
    #      replace this approximation
    if affine_pf is not None:
        final_field = affine_pf + warps
    else:
        final_field = warps + ds.local_affine.position_grid(
            original_shape, blocksize,
        )

    return final_field

