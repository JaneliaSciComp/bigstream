import numpy as np
from ClusterWrap.decorator import cluster
import bigstream.utility as ut
from bigstream.align import affine_align
from bigstream.transform import apply_transform
from scipy.ndimage import zoom
import zarr
from aicsimageio.readers import CziReader
from xml.etree import ElementTree
from itertools import product


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

    return reader, spacing, channel_axis, spatial_axes, tile_axis, tile_positions


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

    # determine overlap sizes per axis
    tile_shape = np.array([reader.shape[x] for x in spatial_axes])
    overlaps = tile_shape - smallest_diffs + 1

    # define how to align a single pair of neighbors
    def align_neighbors(neighbors):

        # determine fixed tile slice object
        A_tile_coords = [slice(None),] * len(reader.dims.order)
        A_tile_coords[channel_axis] = slice(channel, channel+1)
        A_tile_coords[tile_axis] = slice(neighbors[0], neighbors[0]+1)
        A_tile_coords[spatial_axes[neighbors[2]]] = slice(-overlaps[neighbors[2]], None)

        # determine moving tile slice object
        B_tile_coords = [slice(None),] * len(reader.dims.order)
        B_tile_coords[channel_axis] = slice(channel, channel+1)
        B_tile_coords[tile_axis] = slice(neighbors[1], neighbors[1]+1)
        B_tile_coords[spatial_axes[neighbors[2]]] = slice(0, overlaps[neighbors[2]])

        # read data, assign fixed and moving, define origin relative to complete image
        # TODO: reading data like this in distributed case is calling dask inside dask
        #       it spawns a bunch of extra tasks and data is shuffled between workers
        #       I need a more streamlined way to read the data
        lazy_data = CziReader(czi_file_path).dask_data
        A = lazy_data[tuple(A_tile_coords)].compute().squeeze()
        B = lazy_data[tuple(B_tile_coords)].compute().squeeze()
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
    neighbor_transforms = cluster.client.map(align_neighbors, neighbors_list)
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

    # all done!
    return tile_transforms


def save_transforms(path, transforms):
    """
    """
    # TODO: discuss with JB best format to be consistent with other tools
    return None


@cluster
def distributed_apply_stitch(
    czi_file_path,
    transforms,
    channel=0,
    affine_kwargs={},
    cluster=None,
    cluster_kwargs={},
):
    """
    """

    # TODO: each tile is resampled on a separate worker
    #       reference for resampling is the same coordinate region as the tile
    #       but with a buffer around it to make sure no data is lost
    #       tiles are resampled into a pyramid
    #       tiles are merged into an OME-NGFF-ZARR array on disk

    # get all the relevant info about tiles
    tile_info = _get_tile_info(czi_file_path)
    reader = tile_info[0]
    spacing = tile_info[1]
    channel_axis = tile_info[2]
    spatial_axes = tile_info[3]
    tile_axis = tile_info[4]
    tile_positions = tile_info[5]

    
