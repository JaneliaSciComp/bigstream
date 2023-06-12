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

    # access czi file, get spacing, get channel and spatial axes
    reader = CziReader(czi_file_path)
    spacing = reader.physical_pixel_sizes
    axes = tuple(reader.dims.order.index(x) for x in 'CZYX')

    # get tile/mosaic/vector axis (has different names)
    if 'M' in reader.dims.order:
        tile_axis = reader.dims.order.index('M')
        tile_positions = np.array(reader.get_mosaic_tile_positions())
    elif 'V' in reader.dims.order:
        tile_axis = reader.dims.order.index('V')
        tile_positions = [x.attrib for x in reader.metadata.findall('.//TilesSetup//Position')]
        tile_positions = np.array([[float(x[y]) for y in 'ZYX'] for x in tile_position])
        tile_positions = (tile_positions - np.min(tile_positions, axis=0)) / spacing / 1e-6  # spacing in microns
        tile_positions = np.round(tile_positions).astype(int)
    else:
        print("Error: no tile axis found\n")
        # TODO: graceful exit

    # construct list of alignments
    neighbors_list = []
    smallest_diffs = np.min(np.ma.masked_equal(tile_positions, 0), axis=0) + 1
    smallest_diffs[smallest_diffs.mask] = 0
    smallest_diffs = tuple(smallest_diffs)
    for iii, jjj in product(range(len(tile_positions)), repeat=2):
        diffs = tile_positions[jjj] - tile_positions[iii]
        neighbors = (0 < diffs <= smallest_diffs).nonzero()
        if neighbors: neighbors_list.append(iii, jjj, neighbors)

    print(neighbors)
        
        
