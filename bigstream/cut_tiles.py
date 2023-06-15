#!/groups/scicompsoft/home/fleishmang/bin/miniconda3/bin/python3

import sys
import numpy as np
import n5_metadata_utils as n5mu
from os.path import abspath
from os import makedirs


def write_coords_file(path, offset, extent, index):
    with open(path, 'w') as f:
        print(*offset, file=f)
        print(*extent, file=f)
        print(*index, file=f)


if __name__ == '__main__':

    ref_img_path           = abspath(sys.argv[1])
    ref_img_subpath        = sys.argv[2]
    tiles_dir              = abspath(sys.argv[3])
    xy_stride              = int(sys.argv[4])
    xy_overlap             = int(sys.argv[5])
    z_stride               = int(sys.argv[6])
    z_overlap              = int(sys.argv[7])
    min_tile_size          = 128

    grid          = n5mu.read_voxel_grid(ref_img_path, ref_img_subpath)
    stride        = np.array([xy_stride, xy_stride, z_stride], dtype=np.uint16)
    overlap       = np.array([xy_overlap, xy_overlap, z_overlap], dtype=np.uint16)
    tile_grid     = [ x//y+1 if x % y >= min_tile_size else x//y for x, y in zip(grid, stride-overlap) ]
    # make sure there are no zeros in the tile_grid
    tile_grid     = [ x if x != 0 else 1 for x in tile_grid ]
   
    vox           = n5mu.read_voxel_spacing(ref_img_path, ref_img_subpath) 
    grid          = grid * vox
    stride        = stride * vox
    overlap       = overlap * vox
    offset        = np.array([0., 0., 0.])


    for zzz in range(tile_grid[2]):
        for yyy in range(tile_grid[1]):
            for xxx in range(tile_grid[0]):

                ttt = tiles_dir + '/' + str(xxx + yyy*tile_grid[0] + zzz*tile_grid[0]*tile_grid[1])
                makedirs(ttt, exist_ok=True)

                iii = [xxx, yyy, zzz]
                extent = [grid[i]-offset[i] if iii[i] == tile_grid[i]-1 else stride[i] for i in range(3)]
                write_coords_file(ttt + '/coords.txt', offset, extent, iii)

                offset[0] += stride[0] - overlap[0]

            offset[0] = 0.
            offset[1] += stride[1] - overlap[1]

        offset[0] = 0
        offset[1] = 0
        offset[2] += stride[2] - overlap[2]
 
