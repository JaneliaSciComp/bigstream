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


@cluster
def distributed_stitch(
    czi_file_path,
    channel=0,
    affine_kwargs={},
    cluster=None,
    cluster_kwargs={},
):
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
        tile_positions = np.round(tile_positions).astype(int)
    else:
        print("Error: no tile axis found\n")
        # TODO: graceful exit

    # construct list of neighbors/alignments to do
    neighbors_list = []
    smallest_diffs = np.min(np.ma.masked_equal(tile_positions, 0), axis=0) + 1
    smallest_diffs[smallest_diffs.mask] = 0
    for iii, jjj in product(range(len(tile_positions)), repeat=2):
        diffs = tile_positions[jjj] - tile_positions[iii]
        diffs_indx = diffs.nonzero()[0]
        if len(diffs_indx) == 1 and 0 < diffs[diffs_indx[0]] <= smallest_diffs[diffs_indx[0]]:
            neighbors_list.append((iii, jjj, diffs_indx[0]))

    # determine overlap sizes per axis
    tile_shape = np.array([reader.shape[x] for x in spatial_axes])
    overlaps = tile_shape - smallest_diffs + 1

    # define how to align a single pair of neighbors
    def align_neighbors(neighbors):

        # determine fixed tile slice object
        fix_tile_coords = [slice(None),] * len(reader.dims.order)
        fix_tile_coords[channel_axis] = slice(channel, channel+1)
        fix_tile_coords[tile_axis] = slice(neighbors[0], neighbors[0]+1)
        fix_tile_coords[spatial_axes[neighbors[2]]] = slice(-overlaps[neighbors[2]], None)

        # determine moving tile slice object
        mov_tile_coords = [slice(None),] * len(reader.dims.order)
        mov_tile_coords[channel_axis] = slice(channel, channel+1)
        mov_tile_coords[tile_axis] = slice(neighbors[1], neighbors[1]+1)
        mov_tile_coords[spatial_axes[neighbors[2]]] = slice(0, overlaps[neighbors[2]])

        # read data
        # TODO: reading data like this in distributed case is calling dask inside dask
        #       it spawns a bunch of extra tasks and data is shuffled between workers
        #       I need a more elegant way to read the data, streamlined
        lazy_data = CziReader(czi_file_path).dask_data
        fix = lazy_data[tuple(fix_tile_coords)].compute().squeeze()
        mov = lazy_data[tuple(mov_tile_coords)].compute().squeeze()

        # define origin relative to complete image grid
        origin = tile_positions[neighbors[1]] * spacing

        # TODO: probably need a way to not align overlaps that contain no foreground data

        # define registration parameters
        default_affine_kwargs = {
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
        kwargs = {**default_affine_kwargs, **affine_kwargs, 'rigid':True}

        # run the alignment
        return affine_align(
            fix, mov, spacing, spacing,
            fix_origin=origin, mov_origin=origin,
            **kwargs,
        )

    # map align_neighbors to all neighbors
    transforms = cluster.client.gather(cluster.client.map(align_neighbors, neighbors_list))

    # TODO: global adjustement of transforms - translation and rotation must be on the same scale
    # or possibly just done separately

    # TODO: what else do I need to keep in order to properly identify transforms - the neighbors_list?
    # TODO: user needs a way to write tranforms to disk. Look at motion correction json save.

