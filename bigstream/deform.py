import numpy as np
import greedypy.greedypy_registration_method as gprm
from bigstream import distributed
import dask.array as da
import zarr
from numcodecs import Blosc


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
        gradient_abcd=gradient_smoothing,
        field_abcd=field_smoothing,
    )

    register.mask_values(0)
    register.optimize()
    return register.get_warp()


def slice_dict(xy_overlap, z_overlap):
    """
    """

    a = slice(None, None)
    b = slice(-xy_overlap, None)
    c = slice(None, xy_overlap)
    d = slice(-z_overlap, None)
    e = slice(None, z_overlap)

    SD =   { '100':{'000':(b, a, a), '100':(c, a, a)},
             '010':{'000':(a, b, a), '010':(a, c, a)},
             '001':{'000':(a, a, d), '001':(a, a, e)},
             '110':{'000':(b, b, a), '100':(c, b, a), '010':(b, c, a), '110':(c, c, a)},
             '101':{'000':(b, a, d), '001':(b, a, e), '100':(c, a, d), '101':(c, a, e)},
             '011':{'000':(a, b, d), '010':(a, c, d), '001':(a, b, e), '011':(a, c, e)},
             '111':{'000':(b, b, d), '100':(c, b, d), '010':(b, c, d), '001':(b, b, e),
                    '111':(c, c, e), '011':(b, c, e), '101':(c, b, e), '110':(c, c, d)} }

    w   = np.linspace(0, 1, xy_overlap)
    x   = np.linspace(1, 0, xy_overlap)
    y   = np.linspace(0, 1,  z_overlap)
    z   = np.linspace(1, 0,  z_overlap)
    opr = lambda a, b, i: np.expand_dims( np.outer(a, b), i )
    trp = lambda a, b, c: np.einsum('i,j,k->ijk', a, b, c)

    W =    { '100':{'000':x[:, None, None], '100':w[:, None, None]},
             '010':{'000':x[None, :, None], '010':w[None, :, None]},
             '001':{'000':z[None, None, :], '001':y[None, None, :]},
             '110':{'000':opr(x,x,2), '100':opr(w,x,2), '010':opr(x,w,2), '110':opr(w,w,2)},
             '101':{'000':opr(x,z,1), '100':opr(w,z,1), '001':opr(x,y,1), '101':opr(w,y,1)},
             '011':{'000':opr(x,z,0), '010':opr(w,z,0), '001':opr(x,y,0), '011':opr(w,y,0)},
             '111':{'000':trp(x,x,z), '100':trp(w,x,z), '010':trp(x,w,z), '001':trp(x,x,y),
                    '111':trp(w,w,y), '011':trp(x,w,y), '101':trp(w,x,y), '110':trp(w,w,z)} }

    return SD, W


def merge_neighbors(block, overlaps, block_info=None):
    """
    """

    # get the block index
    block_index = block_info[0]['chunk-location']

    # iterate over dimensions
    for axis in range(3):

        # average on the low end
        if block_index[axis] != 0:


        # average on the high end
        # TODO: criteria for being an end block?


        # TODO: do double and triple overlap regions separately?
            
                


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

        # wrap deformable function to simplify passing parameters
        def my_deformable_align(x, y):
            return deformable_align(
                x, y, fixed_vox, moving_vox,
                cc_radius, gradient_smoothing, field_smoothing,
                iterations, shrink_factors, smooth_sigmas, step,
            )

        # deform all chunks
        overlaps = [int(round(x/8)) for x in blocksize]
        out_blocks = [x + 2*y for x, y in zip(blocksize, overlaps)] + [3,]
        warps = da.map_overlap(
            my_deformable_align, fixed_da, moving_da,
            depth=tuple(overlaps),
            boundary='reflect',
            trim=False,
            align_arrays=False,
            dtype=np.float32,
            new_axis=[4,],
            chunks=out_blocks,
        )

        # wrap merge function
        my_merge_neighbors = lambda x: merge_neighbors(x, overlaps)

        # merge overlap regions
        warps = da.map_overlap(
            my_merge_neighbors, warps,
            depth=tuple(overlaps + [0,]),
            boundary='reflect',
            trim=False,
            dtype=np.float32,
            chunks=blocksize + [3,],
        )

        # write result to zarr file
        compressor = Blosc(cname='zstd', clevel=9, shuffle=Blosc.BITSHUFFLE)
        warp_on_disk = zarr.open(write_path, 'w',
            shape=fixed_da.shape+(3,), chunks=blocksize+(3,),
            dtype=np.float32, compressor=compressor,
        )
        da.to_zarr(warps, warp_on_disk)

        # return reference to zarr data store
        return warp_on_disk

