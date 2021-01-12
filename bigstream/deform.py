import sys
# temporary hack for importing greedypy module
sys.path += ['/groups/scicompsoft/home/fleishmang/source/greedypy']
import json
import numpy as np
import n5_metadata_utils as n5mu
import greedypy as gp


def read_coords(path):
    with open(path, 'r') as f:
        offset = np.array(f.readline().split(' ')).astype(np.float64)
        extent = np.array(f.readline().split(' ')).astype(np.float64)
    return offset, extent


if __name__ == '__main__':

    fixed               = sys.argv[1]
    fixed_subpath       = sys.argv[2]
    moving              = sys.argv[3]
    moving_subpath      = sys.argv[4]
    coords              = sys.argv[5]
    output              = sys.argv[6]
    initial_transform   = sys.argv[7]
    final_lcc           = sys.argv[8]
    inverse             = sys.argv[9]
    iterations          = sys.argv[10]
    auto_mask           = sys.argv[11]

    vox = n5mu.read_voxel_spacing(fixed, fixed_subpath)
    offset, extent = read_coords(coords)
    oo = np.round(offset/vox).astype(np.uint16)
    ee = oo + np.round(extent/vox).astype(np.uint16)
    n5_slice = str(oo[2])+':'+str(ee[2])+':x'+ \
               str(oo[1])+':'+str(ee[1])+':x'+ \
               str(oo[0])+':'+str(ee[0])+':'
    
    gp.set_fixed(fixed)
    gp.set_moving(moving)
    gp.set_output(output)
    gp.set_iterations(iterations)
    gp.set_initial_transform(initial_transform)
    gp.set_n5_fixed_path(fixed_subpath, n5_slice)
    gp.set_n5_moving_path(moving_subpath, n5_slice)
    gp.set_final_lcc(final_lcc)
    gp.set_compose_output_with_it()
    gp.set_inverse(inverse)
    gp.set_auto_mask([int(x) for x in auto_mask.split(',')])
    gp.register()
    

