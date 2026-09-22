"""
Distributed I/O helpers.

Two unrelated groups of things live here:

  * **bulk format conversion** - `distributed_directory_of_hdf5_to_zarr` and
    `distributed_directory_of_stack_to_zarr`, one-shot converters that each
    stand up their own cluster.
  * **block write locking** - `BlockWriter` and the helpers below it. This is
    what lets a distributed writer use any block size against a chunked or
    sharded zarr output, instead of having to pick a block size that suits
    the chunk grid.

They share nothing except that both are about getting bytes into a store
from more than one worker at a time.

`h5py` and `glob` are imported inside the two converters rather than at
module scope: the locking half is on the alignment path, which would
otherwise pay ~100 ms of `h5py` import on every worker start and CLI
invocation for something it never calls.
"""

import logging
import numpy as np

from contextlib import contextmanager

import bigstream.utility as ut

from ClusterWrap.decorator import cluster
from dask.distributed import MultiLock


logger = logging.getLogger(__name__)


@cluster
def distributed_directory_of_hdf5_to_zarr(
    directory,
    write_path,
    dataset_path=None,
    chunks=None,
    suffix='.h5',
    cluster=None,
    cluster_kwargs={},
):
    """
    Copy a directory of hdf5 files to a zarr array on disk, data is duplicated.
    Any file with an hdf5 extension is included. All hdf5 files must contain
    an array and all such arrays must have the same shape. The lexicographical order
    determined by glob.glob will determine the order the arrays are indexed across
    the first axis in the result. A dask cluster is used to distribute this work
    over parallel resources.

    Parameters
    ----------
    directory : string
        The path to the directory containing the hdf5 files

    write_path : string
        The path to the zarr array you will create

    dataset_path : string (default: None)
        A path to the array dataset within the hdf5 files

    chunks : tuple (default: None)
        The shape of individual chunks in the zarr array.
        If None, each array dataset will be one chunk.
        WARNING: this function is not protected against parallel writes
        If you use the chunks keyword argument, you must know that your
        chunk size and array size will not result in parallel writes

    suffix : string (default: '.h5')
        The file extension for all the hdf5 files

    cluster : ClusterWrap.cluster object (default: None)
        Only set if you have constructed your own static cluster. The default behavior
        is to construct a cluster for the duration of this function, then close it
        when the function is finished.

    cluster_kwargs : dict (default: {})
        Arguments passed to ClusterWrap.cluster
        If working with an LSF cluster, this will be ClusterWrap.janelia_lsf_cluster.
        If on a workstation this will be ClusterWrap.local_cluster.
        This is how distribution parameters are specified.

    Returns
    -------
    dataset_as_zarr : zarr.Array
        A reference to the zarr array on disk
    """

    import glob
    import h5py

    # get all paths, look at first file to get shape and datatype
    paths = glob.glob(directory + '/*' + suffix)
    with h5py.File(paths[0], 'r') as ex:
        example_array = ex[dataset_path] if dataset_path else ex
        shape = example_array.shape
        dtype = example_array.dtype

    # create zarr array
    if chunks is None: chunks = (1,) + shape
    shape = (len(paths),) + shape
    zarr_array = ut.create_zarr(write_path, shape, chunks, dtype)

    # define write function
    def write_frame(path, index, zarr_array):
        with h5py.File(path, 'r') as a:
            array = a[dataset_path] if dataset_path else a
            data = array[...]
            zarr_array[index] = data
        return True

    # distribute and wait for completion
    futures = cluster.client.map(
        write_frame, paths, range(0, len(paths)),
        zarr_array=zarr_array,
    )
    all_written = np.all( cluster.client.gather(futures) )
    if not all_written: print('SOMETHING FAILED, CHECK LOGS')
    return zarr_array


@cluster
def distributed_directory_of_stack_to_zarr(
    directory,
    write_path,
    shape,
    dtype,
    chunks=None,
    suffix='.stack',
    cluster=None,
    cluster_kwargs={},
):
    """
    Copy a directory of stack files to a zarr array on disk, data is duplicated.
    Any file with a stack extension is included. All stack files must contain
    an array and all such arrays must have the same shape. The lexicographical order
    determined by glob.glob will determine the order the arrays are indexed across
    the first axis in the result. A dask cluster is used to distribute this work
    over parallel resources.

    Parameters
    ----------
    directory : string
        The path to the directory containing the stack files

    write_path : string
        The path to the zarr array you will create

    shape : tuple
        The array dimensions of the dataset in each stack file

    dtype : a numpy datatype (e.g. np.uint16)
        The datatype of the dataset in each stack file

    chunks : tuple (default: None)
        The shape of individual chunks in the zarr array.
        If None, each array dataset will be one chunk.
        WARNING: this function is not protected against parallel writes
        If you use the chunks keyword argument, you must know that your
        chunk size and array size will not result in parallel writes

    suffix : string (default: '.stack')
        The file extension for all the stack files

    cluster : ClusterWrap.cluster object (default: None)
        Only set if you have constructed your own static cluster. The default behavior
        is to construct a cluster for the duration of this function, then close it
        when the function is finished.

    cluster_kwargs : dict (default: {})
        Arguments passed to ClusterWrap.cluster
        If working with an LSF cluster, this will be ClusterWrap.janelia_lsf_cluster.
        If on a workstation this will be ClusterWrap.local_cluster.
        This is how distribution parameters are specified.

    Returns
    -------
    dataset_as_zarr : zarr.Array
        A reference to the zarr array on disk
    """

    import glob

    # get all paths, look at first file to get shape and datatype
    paths = glob.glob(directory + '/*' + suffix)

    # create zarr array
    if chunks is None: chunks = (1,) + shape
    zarr_array = ut.create_zarr(
        write_path,
        (len(paths),) + shape,
        chunks,
        dtype,
    )

    # define write function
    def write_frame(path, index, zarr_array):
        data = np.fromfile(path, dtype=dtype).reshape(shape)
        zarr_array[index] = data
        return True

    # distribute and wait for completion
    futures = cluster.client.map(
        write_frame, paths, range(0, len(paths)),
        zarr_array=zarr_array,
    )
    all_written = np.all( cluster.client.gather(futures) )
    if not all_written: print('SOMETHING FAILED, CHECK LOGS')
    return zarr_array


# -------------------------------------------------------------------------
# block write locking: one writer at a time per write unit (chunk or shard)
# -------------------------------------------------------------------------
#
# Two granularities, kept distinct throughout:
#
#   write unit   what the store rewrites atomically - a chunk, or a shard on
#                a sharded v3 array. Fixed by the output array.
#   lock cell    what one lock name covers. Always a whole number of write
#                units, so "same write unit" always implies "same cell".
#
# Everything here works in the output array's **full index space** rather
# than in its spatial axes. A write unit is a property of the whole array,
# so a key built from a subset of the axes cannot express a collision on the
# rest: two channel writes into a `(t,c,z,y,x)` output whose chunk spans
# both channels really do clobber each other, and a spatial-only key would
# never say so. Axes the write does not span cost nothing - they contribute
# one cell and one key component.
#
# The limit this does not remove: `MultiLock` is a dask scheduler object, so
# it only excludes writers running under one cluster. Two independent runs
# writing the same array share no scheduler and are not protected from each
# other.

# One lock per write unit touched is the precise answer, but a large block
# over a fine chunk grid touches hundreds of them. Past MAX_WRITE_LOCKS the
# cell grows to a larger multiple of the write unit: still correct, because
# it locks a superset, and it only costs parallelism.
MAX_WRITE_LOCKS = 64

# Ceiling on how many write units one lock cell may span, so a pathological
# max_locks cannot search forever.
_MAX_LOCK_CELL_MULTIPLIER = 1024


def storage_write_unit(output_array):
    """
    The smallest region the store rewrites as a whole, over every axis.

    Writing a single voxel of a zarr array rewrites its entire chunk - or its
    entire shard, for a sharded v3 array, because a shard is one object. So
    two workers whose regions land in the same unit clobber each other *even
    when they share no voxel*: each reads the unit, patches its own part and
    writes the whole thing back, and the last one wins.

    Returns the full chunk/shard shape, one entry per array axis. Returns
    `None` for an in-memory array, which assigns element by element and has
    no such unit, and for a dask array, whose `chunks` is a tuple of per-axis
    tuples rather than a shape.
    """
    unit = (getattr(output_array, 'shards', None)
            or getattr(output_array, 'chunks', None))
    if not unit:
        return None
    unit = tuple(unit)
    if not all(isinstance(v, (int, np.integer)) for v in unit):
        return None
    return np.asarray(unit, dtype=int)


def _array_shape(output_array):
    return tuple(int(s) for s in output_array.shape)


def full_rank_region(output_array, region_size):
    """
    `region_size` padded to the array's rank, **trailing**, with the whole
    extent of every axis it does not mention.

    Trailing, because that is what indexing with fewer slices than axes
    means: `a[0:4, 0:4, 0:4]` on a `(z,y,x,d)` array writes all of `d`. So a
    rank-3 block footprint `B` against a rank-4 deform field is the region
    `(*B, d)` and needs no help from the caller, while a caller writing one
    channel of a `(t,c,z,y,x)` output has to say `(1, 1, *B)` itself -
    nothing here can guess that the leading axes are singletons.
    """
    shape = _array_shape(output_array)
    region = [int(v) for v in region_size]
    region += [shape[a] for a in range(len(region), len(shape))]
    return np.asarray(region, dtype=int)


def full_rank_coords(output_array, block_coords):
    """
    `block_coords` padded to the array's rank (trailing, as above) and
    resolved against its shape, so lock keys are computed from plain ints.
    """
    shape = _array_shape(output_array)
    coords = [
        slice(0 if s.start is None else int(s.start),
              shape[a] if s.stop is None else int(s.stop))
        for a, s in enumerate(block_coords)
    ]
    coords += [slice(0, shape[a]) for a in range(len(coords), len(shape))]
    return tuple(coords)


def _max_cells_touched(region, cell, extent):
    """
    Upper bound on how many lock cells one write of `region` can touch.

    `ceil(size/cell) + 1` per axis: a region that size straddles one extra
    cell or not depending on where it happens to start, and the grid is
    fixed for the whole run so the placement is not known here. Capped by
    how many cells the axis actually has, which is what keeps an axis the
    write always spans whole - the `d` of a deform field, a singleton
    channel - contributing exactly one cell rather than a spurious two.
    """
    per_axis = np.minimum(-(-region // cell), -(-extent // cell) - 1) + 1
    return int(np.prod(np.maximum(per_axis, 1)))


def write_lock_grid(output_array, region_size, max_locks=MAX_WRITE_LOCKS):
    """
    Lock cell size for writes into `output_array`, over its full index space.

    `None` means the array has no write unit to protect, i.e. it is in
    memory.

    Computed **once for a whole run**, from the nominal write region rather
    than from any individual block. Every writer has to agree on the grid:
    two blocks that derived different cell sizes would lock different names
    for the same write unit and so would not exclude each other at all. Edge
    blocks are smaller than the nominal region, which only means they lock
    fewer cells of the same grid.

    Each cell is a whole number of write units, so a write unit is never
    split between two cells - which is what makes "same write unit implies
    same cell" hold, and with it the whole guarantee.
    """
    unit = storage_write_unit(output_array)
    if unit is None:
        return None
    region = full_rank_region(output_array, region_size)
    extent = np.asarray(_array_shape(output_array), dtype=int)
    # smallest whole multiple that fits, walked one step at a time rather
    # than doubled: any multiple keeps a write unit inside a single cell, so
    # doubling only over-coarsens. On a 538 voxel footprint over 128 voxel
    # units with max_locks=16, doubling lands on 1024 where 640 would do.
    for multiplier in range(1, _MAX_LOCK_CELL_MULTIPLIER + 1):
        cell = unit * multiplier
        if _max_cells_touched(region, cell, extent) <= max_locks:
            break
    return unit * multiplier


def write_lock_namespace(output_array, fallback):
    """
    Lock names are global to the cluster, so they have to identify the array
    as well as the region within it. A zarr array's path does that; anything
    else falls back to a caller supplied label.
    """
    name = getattr(output_array, 'name', None)
    return str(name) if name else str(fallback)


def lock_cell_keys(output_array, block_coords, lock_grid, namespace):
    """
    One lock key per lock cell the write to `block_coords` touches.

    A cell is a whole number of write units, so two writes that share a
    write unit always share a key. Empty when `lock_grid` is None, i.e. when
    the output is in memory and has no write unit to protect.
    """
    if lock_grid is None:
        return []
    coords = full_rank_coords(output_array, block_coords)
    start = np.array([s.start for s in coords])
    stop = np.array([s.stop for s in coords])
    first = start // lock_grid
    counts = np.maximum(-(-stop // lock_grid) - first, 0)
    return [f'{namespace}/cell{tuple(int(f + o) for f, o in zip(first, offset))}'
            for offset in np.ndindex(*counts)]


@contextmanager
def write_lock(lock_keys, context=''):
    """Hold every key in `lock_keys` for the duration of a write."""
    if not lock_keys:
        yield
        return
    lock = MultiLock(list(lock_keys))
    lock.acquire()
    logger.debug(f'{context} holds {len(lock_keys)} write locks: {lock_keys}')
    try:
        yield
    finally:
        lock.release()
        logger.debug(f'{context} released {len(lock_keys)} write locks')


def log_write_locking(output_array, lock_grid, region_size, context):
    """Report how writes will be serialized - it is the main cost knob here."""
    unit = storage_write_unit(output_array)
    if unit is None:
        logger.info((
            f'{context}: in-memory output, no write unit to lock on'
        ))
        return
    region = full_rank_region(output_array, region_size)
    extent = np.asarray(_array_shape(output_array), dtype=int)
    cells = _max_cells_touched(region, lock_grid, extent)
    aligned = bool(np.all(region % lock_grid == 0))
    logger.info((
        f'{context}: output write unit {tuple(unit)}, lock cell '
        f'{tuple(lock_grid)}, up to {cells} locks per write'
        + ('' if aligned else
           f'; the {tuple(region)} write region is not a multiple of the '
           'lock cell, so two blocks sharing a cell are serialized')
    ))


class BlockWriter:
    """
    Serializes writes into one output array on the store's write unit.

    Build it **once per run**, on the client, and ship the same instance to
    every worker. Two writers that derived their own grid could name
    different locks for the same chunk and then not exclude each other at
    all, so the grid being decided in one place is the invariant the whole
    scheme rests on - making this an object rather than loose functions is
    what enforces it.

    `region_size` is the nominal write region in the array's *own* index
    space: `(1, 1, *B)` for one channel of a `(t,c,z,y,x)` output, `B` for a
    `(z,y,x,d)` deform field (the `d` axis is filled in, see
    `full_rank_region`). It only sizes the lock grid; individual writes may
    be smaller, as edge blocks are.

    Nothing requires `region_size` to be a multiple of the write unit. When
    it is not, blocks that share a unit are serialized against each other -
    correct, and the price of not constraining the block size. `log` reports
    which regime a run is in.

    Picklable by construction: a zarr array handle (which the callers ship to
    workers anyway), a numpy grid, a string and nothing else. The `MultiLock`
    is built inside `write`, on the worker.
    """

    def __init__(self, output_array, region_size, *,
                 max_locks=MAX_WRITE_LOCKS, namespace=None):
        self.output = output_array
        if output_array is None:
            self.region_size = None
            self.lock_grid = None
            self.namespace = str(namespace or 'write')
            return
        self.region_size = full_rank_region(output_array, region_size)
        self.lock_grid = write_lock_grid(output_array, region_size,
                                         max_locks=max_locks)
        self.namespace = write_lock_namespace(output_array,
                                              namespace or 'write')

    def log(self, context):
        """Report the locking regime once, at run start."""
        if self.output is None:
            logger.info(f'{context}: no output array, nothing to write')
            return
        log_write_locking(self.output, self.lock_grid, self.region_size,
                          context)

    def lock_keys(self, block_coords):
        if self.output is None:
            return []
        return lock_cell_keys(self.output, block_coords, self.lock_grid,
                              self.namespace)

    def write(self, block_index, block_coords, block_data, context=''):
        """
        Write one block under the locks its region touches.

        Returns the coordinates written, or None when there was nothing to
        write - same contract the callers had before locking existed.
        """
        if self.output is None or block_data is None:
            return None
        keys = self.lock_keys(block_coords)
        label = context or f'block {block_index}'
        with write_lock(keys, context=label):
            logger.debug((
                f'Write {block_data.shape} block {block_index} at '
                f'{block_coords} to {self.output}({self.output.shape})'
            ))
            self.output[block_coords] = block_data
            logger.debug((
                f'Done writing {block_data.shape} block {block_index} at '
                f'{block_coords} to {self.output}({self.output.shape})'
            ))
        return block_coords
