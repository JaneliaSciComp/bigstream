import numpy as np
import dask.array as da
import copy
from itertools import product


def weight_block(block, blocksize, block_info=None):
    """
    """

    # compute fixed overlap size
    overlaps = np.array([int(round(x/8)) for x in blocksize])

    # determine which faces need linear weighting
    core_shape = []
    pads = []
    block_index = block_info[0]['chunk-location']
    block_grid = block_info[0]['num-chunks']
    for i in range(3):
        p, bl = overlaps[i], blocksize[i]
        bi, bg = block_index[i], block_grid[i]
        pad, core = [2*p+1, 2*p+1], bl - 2*p
        if bi == 0:
            pad[0], core = 0, core + 2*p+1
        if bi == bg-1:
            pad[1], core = 0, core + 2*p+1
        pads.append(tuple(pad))
        core_shape.append(core)

    # create weights
    weights = da.ones(core_shape, dtype=np.float32)
    weights = da.pad(weights, pads, mode='linear_ramp', end_values=0)
    weights = weights[1:-1, 1:-1, 1:-1]
    weights = weights.reshape(weights.shape + (1,))

    # multiply data by weights and return
    return da.multiply(block, weights)


def merge_overlaps(block, blocksize):
    """
    """

    p = np.array([int(round(x/8)) for x in blocksize])
    core = [slice(2*x, -2*x) for x in p]
    result = np.copy(block[tuple(core)])

    # faces
    for ax in range(3):
        # the left side
        slc1 = [slice(None, None)]*3
        slc1[ax] = slice(0, p[ax])
        slc2 = copy.deepcopy(core)
        slc2[ax] = slice(0, p[ax])
        result[tuple(slc1)] += block[tuple(slc2)]
        # the right side
        slc1 = [slice(None, None)]*3
        slc1[ax] = slice(-1*p[ax], None)
        slc2 = copy.deepcopy(core)
        slc2[ax] = slice(-1*p[ax], None)
        result[tuple(slc1)] += block[tuple(slc2)]

    # edges
    for edge in product([0, 1], repeat=2):
        for ax in range(3):
            pp = np.delete(p, ax)
            left = [slice(None, pe) for pe in pp]
            right = [slice(-1*pe, None) for pe in pp]
            slc1 = [left[i] if e == 0 else right[i] for i, e in enumerate(edge)]
            slc2 = copy.deepcopy(slc1)
            slc1.insert(ax, slice(None, None))
            slc2.insert(ax, core[ax])
            result[tuple(slc1)] += block[tuple(slc2)]

    # corners
    left = [slice(None, x) for x in p]
    right = [slice(-1*x, None) for x in p]
    for corner in product([0, 1], repeat=3):
        slc = [left[i] if c == 0 else right[i] for i, c in enumerate(corner)]
        result[tuple(slc)] += block[tuple(slc)]

    return result


def stitch_fields(fields, blocksize):
    """
    """

    # weight block edges
    weighted_fields = da.map_blocks(
        weight_block, fields, blocksize=blocksize, dtype=np.float32,
    )

    # remove block index dimensions
    sh = fields.shape[:3]
    list_of_blocks = [[[[weighted_fields[i,j,k]] for k in range(sh[2])]
                                                 for j in range(sh[1])]
                                                 for i in range(sh[0])]
    aug_fields = da.block(list_of_blocks)

    # merge overlap regions
    overlaps = tuple([int(round(x/8)) for x in blocksize] + [0,])
    return da.map_overlap(
        merge_overlaps, aug_fields, blocksize=blocksize,
        depth=overlaps, boundary=0., trim=False,
        dtype=np.float32, chunks=blocksize+[3,],
    )

