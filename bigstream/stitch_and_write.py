import sys
import numpy as np
import glob
import nrrd
from os.path import dirname, isfile
import zarr
import gc
import n5_metadata_utils as n5mu
from itertools import product


# READERS
def read_coords(path):
    with open(path, 'r') as f:
        offset = np.array(f.readline().split(' ')).astype(np.float64)
        extent = np.array(f.readline().split(' ')).astype(np.float64)
        index  = np.array(f.readline().split(' ')).astype(np.uint16)
    return offset, extent, index


def read_fields(neighbors, suffix):
    fields = {}
    keys = list(neighbors.keys())
    keys.sort()
    for key in keys:
        if neighbors[key]:
            if isfile(neighbors[key] + suffix):
                fields[key], m = nrrd.read(neighbors[key] + suffix)
        else:
            fields[key] = np.zeros_like( fields['000'] )
    return fields


# WRITERS
def create_n5_dataset(n5_path, subpath, sh, xy_overlap, z_overlap):
    sh = tuple([x.item() for x in sh])
    n5im = zarr.open(store=zarr.N5Store(n5_path), mode='a')
    try:
        n5im.create_dataset('/c0'+subpath, shape=sh[::-1], 
                            chunks=(z_overlap, xy_overlap, xy_overlap),
                            dtype=np.float32)
        n5im.create_dataset('/c1'+subpath, shape=sh[::-1],
                            chunks=(z_overlap, xy_overlap, xy_overlap),
                            dtype=np.float32)
        n5im.create_dataset('/c2'+subpath, shape=sh[::-1],
                            chunks=(z_overlap, xy_overlap, xy_overlap),
                            dtype=np.float32)
    except Exception as e:
        # TODO: should only pass if it's a "File already exists" exception
        pass
    return n5im


def write_updated_transform(n5im, subpath, updated_warp, oo):
    extent = np.array(updated_warp.shape[:-1])
    ee = oo + extent
    utx = np.moveaxis(updated_warp[..., 0], (0, 2), (2, 0))
    uty = np.moveaxis(updated_warp[..., 1], (0, 2), (2, 0))
    utz = np.moveaxis(updated_warp[..., 2], (0, 2), (2, 0))
    n5im['/c0'+subpath][oo[2]:ee[2], oo[1]:ee[1], oo[0]:ee[0]] = utx
    n5im['/c1'+subpath][oo[2]:ee[2], oo[1]:ee[1], oo[0]:ee[0]] = uty
    n5im['/c2'+subpath][oo[2]:ee[2], oo[1]:ee[1], oo[0]:ee[0]] = utz


# FOR AFFINE TRANSFORM
def position_grid(sh, dtype=np.uint16):
    """Return a position array in physical coordinates with shape sh"""
    coords = np.array(np.meshgrid(*[range(x) for x in sh], indexing='ij'), dtype=dtype)
    return np.ascontiguousarray(np.moveaxis(coords, 0, -1))


def transform_grid(matrix, grid):
    """Apply affine matrix to position grid"""
    mm = matrix[:, :-1]
    tt = matrix[:, -1]
    return np.einsum('...ij,...j->...i', mm, grid) + tt


# HANDLE OVERLAPS
def get_neighbors(tiledir, index, suffix='/*/coords.txt'):

    bin_strs = [''.join(p) for p in product('10', repeat=3)]
    neighbors = { a:b for a, b in zip(bin_strs, [False,]*8) }
    tiles = glob.glob(tiledir + suffix)
    for tile in tiles:
        oo, ee, ii = read_coords(tile)
        key = ''.join( [str(i) for i in ii - index] )
        if key in neighbors.keys(): neighbors[key] = dirname(tile)
    return neighbors


def slice_dict(xy_overlap, z_overlap):

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


def average_neighbors(lcc, warps, bin_str, xy_overlap, z_overlap, eps=0.8):

    SD, W = slice_dict(xy_overlap, z_overlap)
    SD, W = SD[bin_str], W[bin_str]
    update = np.zeros_like( warps['000'][ SD['000'] ])
    denom  = np.zeros_like(   lcc['000'][ SD['000'] ])
    for key in SD.keys():
        denom  += lcc[key][ SD[key] ]
    denom[ denom == 0 ] = 1.0  # avoids divide by 0 warning
    for key in SD.keys():
        w       = eps * W[key] + (1 - eps) * lcc[key][ SD[key] ] / denom
        update += w[..., None] * warps[key][ SD[key] ]
    warps['000'][ SD['000'] ] = update
    return warps['000']


def reconcile_warps(lcc, warps, xy_overlap, z_overlap):

    # ones before twos before threes
    bin_strs = [''.join(p) for p in product('10', repeat=3)]
    for bin_str in bin_strs:
        if np.sum(np.array( [int(i) for i in bin_str] )) == 1:
            warps['000'] = average_neighbors(lcc, warps, bin_str, xy_overlap, z_overlap)
    for bin_str in bin_strs:
        if np.sum(np.array( [int(i) for i in bin_str] )) == 2:
            warps['000'] = average_neighbors(lcc, warps, bin_str, xy_overlap, z_overlap)
    for bin_str in bin_strs:
        if np.sum(np.array( [int(i) for i in bin_str] )) == 3:
            warps['000'] = average_neighbors(lcc, warps, bin_str, xy_overlap, z_overlap)
    return warps['000']



# TODO: 
#       modify to accommodate overlaps in Z
#       add simpler overlap reconciliation methods: averaging, weighted averaging
if __name__ == '__main__':

    tile            = sys.argv[1]
    xy_overlap      = int(sys.argv[2])
    z_overlap       = int(sys.argv[3])
    reference       = sys.argv[4]
    ref_subpath     = sys.argv[5]
    global_affine   = sys.argv[6]
    output          = sys.argv[7]
    invoutput       = sys.argv[8]
    output_subpath  = sys.argv[9]


    # read basic elements
    tiledir = dirname(tile)
    vox = n5mu.read_voxel_spacing(reference, ref_subpath)
    offset, extent, index = read_coords(tile + '/coords.txt')

    # initialize updated warp fields with global affine
    matrix = np.float32(np.loadtxt(global_affine))
    grid = np.round(extent/vox).astype(np.uint16)
    grid = position_grid(grid) * vox + offset
    updated_warp = transform_grid(matrix, grid)

    inv_matrix = np.array([ matrix[0],
                            matrix[1],
                            matrix[2],
                            [0, 0, 0, 1] ])
    inv_matrix = np.linalg.inv(inv_matrix)[:-1]
    updated_invwarp = transform_grid(inv_matrix, grid)
    del grid; gc.collect()

    # handle overlap regions
    neighbors = get_neighbors(tiledir, index)

    if isfile(neighbors['000']+'/final_lcc.nrrd'):
        lcc = read_fields(neighbors, suffix='/final_lcc.nrrd')
        for key in lcc.keys():
            lcc[key][lcc[key] > 1.0] = 1.0  # typically in noisy regions
            lcc[key][np.isnan(lcc[key])] = 0.0  # where lcc could not be evaluated
        warps = read_fields(neighbors, suffix='/warp.nrrd')
        updated_warp += reconcile_warps(lcc, warps, xy_overlap, z_overlap)
        del warps; gc.collect()  # need space for inv_warps
        inv_warps = read_fields(neighbors, suffix='/invwarp.nrrd')
        updated_invwarp += reconcile_warps(lcc, inv_warps, xy_overlap, z_overlap)
        
    # OPTIONAL: SMOOTH THE OVERLAP REGIONS
    # OPTIONAL: USE WEIGHTED COMBINATION BASED ON LCC AT ALL VOXELS

    # update offset to avoid writing left and top overlap regions
    oo = np.round(offset/vox).astype(np.uint16)
    if index[0] != 0 and index[1] != 0:
        updated_warp = updated_warp[xy_overlap:, xy_overlap:]
        updated_invwarp = updated_invwarp[xy_overlap:, xy_overlap:]
        oo[0:2] += xy_overlap
    elif index[1] != 0:
        updated_warp = updated_warp[:, xy_overlap:]
        updated_invwarp = updated_invwarp[:, xy_overlap:]
        oo[1] += xy_overlap
    elif index[0] != 0:
        updated_warp = updated_warp[xy_overlap:, :]
        updated_invwarp = updated_invwarp[xy_overlap:, :]
        oo[0] += xy_overlap

    if index[2] != 0:
        updated_warp = updated_warp[..., z_overlap:, :]
        updated_invwarp = updated_invwarp[..., z_overlap:, :]
        oo[2] += z_overlap

    # write results
    ref_grid = n5mu.read_voxel_grid(reference, ref_subpath)
    n5im = create_n5_dataset(output, output_subpath, ref_grid, xy_overlap, z_overlap)
    write_updated_transform(n5im, output_subpath, updated_warp, oo)
    n5im = create_n5_dataset(invoutput, output_subpath, ref_grid, xy_overlap, z_overlap)
    write_updated_transform(n5im, output_subpath, updated_invwarp, oo)

