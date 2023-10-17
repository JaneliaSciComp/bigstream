import numpy as np
from ClusterWrap.decorator import cluster
import bigstream.utility as ut
from bigstream.align import affine_align
from bigstream.transform import apply_transform
from scipy.ndimage import zoom
import zarr
from zarr import blosc
from aicsimageio.readers import CziReader
from xml.etree import ElementTree
from itertools import product
import json
from distributed import Event
import time
import os
import aicspylibczi
from ngff_zarr import to_ngff_image, to_multiscales, to_ngff_zarr


def _get_tile_info(czi_file_path):
    """"""

    # access czi file, get spacing, get channel axis, get spatial axes
    reader = CziReader(czi_file_path)
    spacing = reader.physical_pixel_sizes
    channel_axis = reader.dims.order.index('C')
    spatial_axes = tuple(reader.dims.order.index(x) for x in 'ZYX')

    # get tile/mosaic/vector axis (has different names), get tile positions
    if 'M' in reader.dims.order:
        tile_axis = reader.dims.order.index('M')
        tile_positions = np.array(reader.get_mosaic_tile_positions())
        # TODO: ensure all axes are present in tile_positions for this case
    elif 'V' in reader.dims.order:
        tile_axis = reader.dims.order.index('V')
        tile_positions = [x.attrib for x in reader.metadata.findall('.//TilesSetup//Position')]
        tile_positions = np.array([[float(x[y]) for y in 'ZYX'] for x in tile_positions])
        tile_positions = (tile_positions - np.min(tile_positions, axis=0)) / spacing / 1e-6  # spacing in microns
        # TODO: keep physical tile positions for sub voxel accuracy later
        tile_positions = np.round(tile_positions).astype(int)
    else:
        print("Error: no tile axis found\n")
        # TODO: graceful exit

    # get (i, j, k) tile grid positions
    tile_grid_indices = []
    steps = [np.sort(np.unique(tile_positions[:, x])) for x in range(tile_positions.shape[1])]
    for tile in tile_positions:
        tile_grid_indices.append( tuple(np.where(s == x)[0][0] for s, x in zip(steps, tile)) )

    # get tile shape
    tile_shape = np.array([reader.shape[x] for x in spatial_axes])

    # get overlap shapes
    smallest_diffs = np.min(np.ma.masked_equal(tile_positions, 0), axis=0) + 1
    smallest_diffs[smallest_diffs.mask] = 0
    overlaps = tile_shape - smallest_diffs + 1
    overlaps = np.array([o if o != s+1 else 0 for o, s in zip(overlaps, tile_shape)])

    return (reader, spacing, channel_axis,
            spatial_axes, tile_axis, tile_positions,
            tile_grid_indices, tile_shape, overlaps,)


@cluster
def distributed_stitch(
    czi_file_path,
    channel=0,
    minimum_overlap_correlation=0.3,
    global_optimization_iterations=100,
    global_optimization_learning_rate=0.1,
    affine_kwargs={},
    cluster=None,
    cluster_kwargs={},
):
    # TODO: think over the function API
    # TODO: complete docstring
    """
    Stitch the tiles in a czi file into one continuous volume.
    Overlapping regions are rigid aligned.

    Parameters
    ----------
    czi_file_path : string
        Path to the czi file

    channel : int (default: 0)
        Which channel to use for stitching

    cluster : ClusterWrap.cluster object (default: None)
        Only set if you have constructed your own static cluster. The default behavior
        is to construct a cluster for the duration of this function, then close it
        when the function is finished.

    cluster_kwargs : dict (default: {})
        Arguments passed to ClusterWrap.cluster
        If working with an LSF cluster, this will be
        ClusterWrap.janelia_lsf_cluster. If on a workstation
        this will be ClsuterWrap.local_cluster.
        This is how distribution parameters are specified.
    """

    # get all the relevant info about tiles
    tile_info = _get_tile_info(czi_file_path)
    reader = tile_info[0]
    spacing = tile_info[1]
    channel_axis = tile_info[2]
    spatial_axes = tile_info[3]
    tile_axis = tile_info[4]
    tile_positions = tile_info[5]
    tile_grid_indices = tile_info[6]
    tile_shape = tile_info[7]
    overlaps = tile_info[8]

    # construct list of neighbors/alignments to do
    neighbors_list = []
    fixed_image = {0:True}
    smallest_diffs = np.min(np.ma.masked_equal(tile_positions, 0), axis=0) + 1
    smallest_diffs[smallest_diffs.mask] = 0
    for iii, jjj in product(range(len(tile_positions)), repeat=2):
        diffs = tile_positions[jjj] - tile_positions[iii]
        diffs_indx = diffs.nonzero()[0]
        if len(diffs_indx) == 1 and 0 < diffs[diffs_indx[0]] <= smallest_diffs[diffs_indx[0]]:
            fixed_image[jjj] = False if fixed_image[iii] else True
            neighbors_list.append((iii, jjj, diffs_indx[0]))

    # define how to align a single pair of neighbors
    def align_neighbors(neighbors):

        # get number of cores
        ncores = ut.get_number_of_cores()

        # read the first region
        other_reader = aicspylibczi.CziFile(czi_file_path)
        A_read_spec = {
            reader.dims.order[channel_axis]:channel,
            reader.dims.order[tile_axis]:neighbors[0],
            'cores':2*ncores,
        }
        A_slice = [slice(None),] * len(spatial_axes)
        A_slice[neighbors[2]] = slice(-overlaps[neighbors[2]], None)
        # the copy prevents storage of entire tile in memory
        # without the copy, it stores a view into the entire tile
        A = np.copy(other_reader.read_image(**A_read_spec)[0].squeeze()[tuple(A_slice)])

        # read the second region
        B_read_spec = {
            reader.dims.order[channel_axis]:channel,
            reader.dims.order[tile_axis]:neighbors[1],
            'cores':2*ncores,
        }
        B_slice = [slice(None),] * len(spatial_axes)
        B_slice[neighbors[2]] = slice(0, overlaps[neighbors[2]])
        B = np.copy(other_reader.read_image(**B_read_spec)[0].squeeze()[tuple(B_slice)])

        # determine fix and moving, define origin relative to whole image
        fix, mov = (A, B) if fixed_image[neighbors[0]] else (B, A)
        origin = tile_positions[neighbors[1]] * spacing

        # check if overlap has sufficient common foreground to try and register
        corr = np.corrcoef(fix.flatten(), mov.flatten())[0, 1]
        if corr < minimum_overlap_correlation:
            print(f'Insufficient overlap correlation for tile pair {neighbors}.', flush=True)
            return None

        # define registration parameters
        default_affine_kwargs = {
            'rigid':True,
            'metric':'MS',
            'alignment_spacing':np.min(spacing) * 4,
            'shrink_factors':(2,),
            'smooth_sigmas':(np.min(spacing) * 8,),
            'optimizer_args':{
                'learningRate':0.05,
                'minStep':0.01,
                'numberOfIterations':100,
            },
        }
        kwargs = {**default_affine_kwargs, **affine_kwargs}

        # run the alignment
        return affine_align(
            fix, mov, spacing, spacing,
            fix_origin=origin, mov_origin=origin,
            **kwargs,
        )

    # map align_neighbors to all neighbors
    neighbor_transforms = cluster.client.map(align_neighbors, neighbors_list, resources={'concurrency':1})
    neighbor_transforms = cluster.client.gather(neighbor_transforms)

    # filter out bad overlaps
    new_neighbors_list, new_neighbor_transforms = [], []
    for a, b in zip(neighbors_list, neighbor_transforms):
        if b is not None:
            new_neighbors_list.append(a)
            new_neighbor_transforms.append(b)
    neighbors_list = new_neighbors_list
    neighbor_transforms = np.array(new_neighbor_transforms)

    # build transform composition matrix
    A = np.zeros((len(tile_positions), len(tile_positions), len(neighbors_list)))
    for iii, neighbors in enumerate(neighbors_list):
        a, b = neighbors[0], neighbors[1]
        fi, mi = (a, b) if fixed_image[a] else (b, a)
        A[mi, fi, iii] = 1

    # initialize tile transforms as identity
    tile_transforms = np.empty((len(tile_positions), 4, 4))
    for iii in range(len(tile_positions)):
        tile_transforms[iii] = np.eye(4)

    # gradient descent loop
    print('Starting global consistency optimization')
    for iii in range(global_optimization_iterations):

        # with respect to moving parameters
        factor = np.einsum('mij,nmo', tile_transforms, A)
        reconstruction = np.einsum('nij,jkno', tile_transforms, factor)
        left = np.einsum('ijno,jko', factor, reconstruction)
        right = np.einsum('ijno,ojk', factor, neighbor_transforms)
        gradient = left - right

        # with respect to fixed parameters
        factor = np.einsum('nij,nmo', tile_transforms, A)
        reconstruction = np.einsum('nij,jkno', tile_transforms, factor)
        left = np.einsum('ijno,jko', factor, reconstruction)
        right = np.einsum('ijno,ojk', factor, neighbor_transforms)
        gradient = (gradient + left - right).transpose(2, 0, 1)

        # print feedback
        objective = np.sum( (neighbor_transforms - reconstruction.transpose(2, 0, 1))**2 )
        print(f'ITERATION: {iii}  OBJECTIVE VALUE: {objective}')

        # take a step
        tile_transforms = tile_transforms - global_optimization_learning_rate * gradient

    # invert all fixed transforms
    for iii in range(len(tile_transforms)):
        if fixed_image[iii]:
            tile_transforms[iii] = np.linalg.inv(tile_transforms[iii])

    # TODO: consider fixing one tile
    #       i.e. find inverse of one transform and compose that with
    #       all other transforms

    # all done!
    return tile_transforms


def save_transforms(path, transforms):
    """
    """
    # TODO: discuss with JB best format to be consistent with other tools
    n = transforms.shape[0]
    d = {i:transforms[i].tolist() for i in range(n)}
    with open(path, 'w') as f:
        json.dump(d, f, indent=4)


def read_transforms(path):
    """
    """

    with open(path, 'r') as f:
        d = json.load(f)
    return np.array([d[str(i)] for i in range(len(d))])


@cluster
def distributed_apply_stitch(
    czi_file_path,
    transforms,
    write_path,
    resample_padding=0.2,
    write_group_interval=60,
    channel=0,
    cluster=None,
    cluster_kwargs={},
):
    """
    """

    # get all the relevant info about tiles
    tile_info = _get_tile_info(czi_file_path)
    reader = tile_info[0]
    spacing = tile_info[1]
    channel_axis = tile_info[2]
    spatial_axes = tile_info[3]
    tile_axis = tile_info[4]
    tile_positions = tile_info[5]
    tile_grid_indices = tile_info[6]
    tile_shape = tile_info[7]
    overlaps = tile_info[8]

    # generate zarr file for writing
    zarr_blocks = tuple(np.round(tile_shape / 2).astype(int))
    full_shape = np.max(tile_positions, axis=0) + reader.dims[['Z', 'Y', 'X']]
    output_zarr = ut.create_zarr(
        write_path, full_shape, zarr_blocks, reader.dtype, multithreaded=True,
    )

    def resample_tile(tile_number, transform):

        print(f'starting {tile_number}', flush=True)

        # get number of cores
        ncores = ut.get_number_of_cores()

        # read tile data
        other_reader = aicspylibczi.CziFile(czi_file_path)
        read_spec = {
            reader.dims.order[channel_axis]:channel,
            reader.dims.order[tile_axis]:tile_number,
            'cores':2*ncores,
        }
        tile = other_reader.read_image(**read_spec)[0].squeeze()
        mov_origin = tile_positions[tile_number]

        print(f'weighting tile {tile_number}', flush=True)

        # apply linear blending weights to overlap region, per axis
        for axis in range(3):
            # only if we cut tiles along this axis
            if overlaps[axis] != 0:
                # construct weights array for this axis
                shape = list(tile.shape)
                shape[axis] = 1
                pads = [(0, 0),]*3
                pads[axis] = (overlaps[axis], 0)
                weights = np.pad(np.ones(shape, dtype=np.float32), pads, mode='linear_ramp')
                # left side, only if it's not on the left edge
                if tile_grid_indices[tile_number][axis] > 0:
                    region = [slice(None),]*3
                    region[axis] = slice(0, overlaps[axis]+1)
                    region = tuple(region)
                    tile[region] = np.round( tile[region] * weights ).astype(tile.dtype)
                # right side, only if it's not on the right edge
                if tile_grid_indices[tile_number][axis] < np.max(tile_grid_indices, axis=0)[axis]:
                    region = [slice(None),]*3
                    region[axis] = slice(-overlaps[axis]-1, None)
                    region = tuple(region)
                    reflect = [slice(None),]*3
                    reflect[axis] = slice(None, None, -1)
                    reflect = tuple(reflect)
                    tile[region] = np.round( tile[region] * weights[reflect] ).astype(tile.dtype)

        # generate reference
        fix_origin = mov_origin - np.round(np.array(tile.shape) * resample_padding).astype(int)
        fix_end = fix_origin + np.round(np.array(tile.shape) * (1 + 2*resample_padding)).astype(int)
        fix_origin = np.maximum(fix_origin, 0)
        fix_end = np.minimum(fix_end, output_zarr.shape)
        fix_shape = tuple(int(b - a) for a, b in zip(fix_origin, fix_end))

        print(f'aligning {tile_number}', flush=True)

        # apply transform
        aligned = apply_transform(
            fix_shape, tile, spacing, spacing,
            transform_list=[transform,],
            fix_origin=fix_origin * spacing,
            mov_origin=mov_origin * spacing,
        )

        # register as ready to write
        write_region = tuple(slice(a, b) for a, b in zip(fix_origin, fix_end))
        blosc.set_nthreads(2*ncores)

        # get neighbors info
        neighbor_events = []
        for delta in product((-1, 0, 1), repeat=3):
            if delta == (0, 0, 0): continue
            neighbor_index = tuple(a + b for a, b in zip(tile_grid_indices[tile_number], delta))
            neighbor_events.append(Event(f'{neighbor_index}'))

        # wait until its clear to write
        print(f'waiting {tile_number}', flush=True)
        while True:
            if np.all( [not e.is_set() for e in neighbor_events] ):

                # some robustness to race conditions
                seed = int(time.time()) + int(''.join([str(x) for x in tile_grid_indices[tile_number]]))
                np.random.seed(seed)
                time.sleep(int(np.random.rand() * 8 + 2))
                if np.any( [e.is_set() for e in neighbor_events] ): continue

                done_event = Event(f'{tile_grid_indices[tile_number]}')
                done_event.set()
                break
            else: time.sleep(1)

        # write result to disk
        print(f'writing {tile_number}, {time.ctime(time.time())}', flush=True)
        output_zarr[write_region] = output_zarr[write_region] + aligned
        print(f'done writing {tile_number}, {time.ctime(time.time())}', flush=True)

        # unset write flag, return
        done_event.clear()
        print(f'done with {tile_number}', flush=True)
        return True

    futures = cluster.client.map(resample_tile, range(len(transforms)), transforms, resources={'concurrency':1})
    all_events = cluster.client.gather(futures)
    return output_zarr


@cluster
def generate_ome_ngff_zarr(
    input_zarr_array,
    spacing,
    write_path,
    scale_factors,
    chunks,
    cluster=None,
    cluster_kwargs={},
    **kwargs,
):
    """
    """
    
    print('calling to_ngff_image', flush=True)
    ngff_image = to_ngff_image(
        input_zarr_array,
        dims=('z', 'y', 'x'),
        scale={a:b for a, b in zip('zyx', spacing)},
        axes_units={a:'micrometer' for a in 'zyx'}
    )
    print('calling to_multiscales', flush=True)
    multiscales = to_multiscales(
        ngff_image,
        scale_factors,
        chunks=chunks,
    )
    print('calling to_ngff_zarr', flush=True)
    to_ngff_zarr(
        write_path,
        multiscales,
        **kwargs,
    )
    return zarr.open(write_path, 'r+')

