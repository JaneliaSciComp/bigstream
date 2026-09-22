"""
Tests for the write path of `bigstream.distributed_transform`.

Both transform paths used to avoid write races by *forbidding* block sizes
that did not suit the output's chunk (or shard) grid, via
`distutils.validate_processing_block_size`. That check was unsound: it only
required `unit <= block`, which lets two adjacent blocks land in one chunk
whenever the block size is not a whole multiple of the unit.

They now lock per write unit instead, the way the alignment path already
did, so any block geometry is safe and the check is gone. These tests pin
both halves of that: the shared locking machinery in
`bigstream.distributed_io_utility`, and the two public functions end to end
against a real multi-worker cluster.

Notation, all extents in voxels and matching the rest of the codebase:

    write unit   what the store rewrites atomically - a chunk, or a shard on
                 a sharded v3 array
    lock cell    what one lock name covers; always a whole number of write
                 units
    B            the processing/partition block size, i.e. the step between
                 block origins
"""

import numpy as np
import pytest
import zarr

from bigstream.distributed_io_utility import (
    BlockWriter,
    full_rank_coords,
    full_rank_region,
    lock_cell_keys,
    storage_write_unit,
    write_lock_grid,
)
from bigstream.distributed_transform import (
    distributed_apply_transform,
    distributed_invert_displacement_vector_field,
)
from bigstream.image_data import ImageData


@pytest.fixture(scope='module')
def cluster_client_mt():
    """Several workers, so two writes into one write unit really can
    collide and the locking has something to do."""
    distributed = pytest.importorskip('dask.distributed')
    cluster = distributed.LocalCluster(n_workers=4, threads_per_worker=1,
                                       processes=False, dashboard_address=':0')
    client = distributed.Client(cluster)
    yield client
    client.close()
    cluster.close()


def _zarr(tmp_path, name, shape, chunks, shards=None, dtype='f4'):
    return zarr.create_array(store=str(tmp_path / name), shape=shape,
                             chunks=chunks, shards=shards, dtype=dtype,
                             zarr_format=3)


def _fill_block(task):
    """Write one constant-valued block through a shared `BlockWriter`."""
    writer, index, coords, value = task
    shape = tuple(s.stop - s.start for s in coords)
    return writer.write(index, coords, np.full(shape, value, dtype=np.float32))


# --------------------------------------------------------------------------
# the write unit spans every axis, not just the spatial ones
# --------------------------------------------------------------------------


def test_write_unit_covers_the_whole_index_space(tmp_path):
    """
    The unit is a property of the array, so it has to name every axis. Taking
    only the leading three is right for a `(z,y,x,d)` deform field and simply
    wrong for a `(t,c,z,y,x)` warp output, whose spatial axes are trailing.
    """
    field = _zarr(tmp_path, 'field', (32, 32, 32, 3), (8, 8, 8, 3))
    warped = _zarr(tmp_path, 'warped', (1, 2, 32, 32, 32), (1, 1, 8, 8, 8))

    assert list(storage_write_unit(field)) == [8, 8, 8, 3]
    assert list(storage_write_unit(warped)) == [1, 1, 8, 8, 8]

    # a sharded array rewrites the whole shard, so that is the unit
    sharded = _zarr(tmp_path, 'sharded', (32, 32, 32, 3), (4, 4, 4, 3),
                    shards=(16, 16, 16, 3))
    assert list(storage_write_unit(sharded)) == [16, 16, 16, 3]

    # nothing to protect in memory, and a dask array's `chunks` is a tuple
    # of per-axis tuples rather than a shape
    assert storage_write_unit(np.zeros((8, 8, 8), dtype=np.float32)) is None


def test_short_coords_are_padded_the_way_numpy_indexes(tmp_path):
    """
    `a[0:8, 0:8, 0:8]` on a `(z,y,x,d)` array writes all of `d`, so a rank-3
    block against a rank-4 field must produce the same keys as spelling the
    trailing axis out.
    """
    field = _zarr(tmp_path, 'f', (32, 32, 32, 3), (8, 8, 8, 3))
    grid = write_lock_grid(field, (8, 8, 8))

    assert list(full_rank_region(field, (8, 8, 8))) == [8, 8, 8, 3]
    assert full_rank_coords(field, (slice(0, 8),) * 3) == (
        (slice(0, 8),) * 3 + (slice(0, 3),))

    short = lock_cell_keys(field, (slice(0, 8),) * 3, grid, 'ns')
    explicit = lock_cell_keys(field, (slice(0, 8),) * 3 + (slice(0, 3),),
                              grid, 'ns')
    assert short == explicit
    # the vector axis is spanned whole by every write, so it splits nothing
    assert len(short) == 1


def test_grid_is_the_same_however_many_writers_derive_it(tmp_path):
    """
    The invariant the whole scheme rests on. Two writers that disagreed on
    the cell size would name different locks for one chunk and so would not
    exclude each other at all - which is why the grid is derived once, from
    the array and the nominal region, and never from an individual block.
    """
    array = _zarr(tmp_path, 'a', (64, 64, 64), (10, 10, 10))
    a = BlockWriter(array, (17, 17, 17), namespace='x')
    b = BlockWriter(array, (17, 17, 17), namespace='y')

    assert list(a.lock_grid) == list(b.lock_grid)
    # the namespace falls back to the array's own path, so even the labels
    # agree and the two writers really do contend
    assert a.namespace == b.namespace
    coords = (slice(9, 26),) * 3
    assert a.lock_keys(coords) == b.lock_keys(coords)


def test_no_locking_for_an_in_memory_output():
    """An ndarray assigns element by element - there is no unit to protect."""
    output = np.zeros((16, 16, 16), dtype=np.float32)
    writer = BlockWriter(output, (8, 8, 8), namespace='mem')

    assert writer.lock_grid is None
    assert writer.lock_keys((slice(0, 8),) * 3) == []

    coords = (slice(0, 8),) * 3
    assert writer.write((0, 0, 0), coords, np.ones((8, 8, 8), np.float32)) \
        == coords
    assert np.all(output[coords] == 1)
    assert not np.any(output[8:, 8:, 8:])

    # nothing to write is not an error, it just reports that it wrote nothing
    assert writer.write((0, 0, 0), coords, None) is None
    assert BlockWriter(None, (8, 8, 8)).write((0,), coords, np.ones(1)) is None


# --------------------------------------------------------------------------
# the race the retired validator let through
# --------------------------------------------------------------------------


def test_adjacent_blocks_sharing_a_chunk_do_not_clobber_each_other(
        cluster_client_mt, tmp_path):
    """
    The concrete hole in `unit <= B`: chunks of 64, `B = 100`. The check
    passed, but block 0 writes `[0,100)` and block 1 writes `[100,200)`,
    and chunk 1 spans `[64,128)` and is touched by both. Each worker
    read-modify-writes the whole chunk, so without a lock the last one wins
    and the other's voxels are gone.
    """
    array = _zarr(tmp_path, 'race', (200, 8, 8), (64, 8, 8))
    writer = BlockWriter(array, (100, 8, 8), namespace='race')
    # both blocks land in chunk 1, so they must share a key
    left = (slice(0, 100), slice(0, 8), slice(0, 8))
    right = (slice(100, 200), slice(0, 8), slice(0, 8))
    assert set(writer.lock_keys(left)) & set(writer.lock_keys(right))

    tasks = [(writer, (0,), left, 1.0), (writer, (1,), right, 2.0)]
    assert all(r is not None for r in
               cluster_client_mt.gather(
                   cluster_client_mt.map(_fill_block, tasks, pure=False)))

    assert np.all(array[0:100] == 1.0)
    assert np.all(array[100:200] == 2.0)


@pytest.mark.parametrize('chunk_extent', [2, 1],
                         ids=['chunk-spans-both-channels', 'chunk-per-channel'])
def test_two_channels_in_one_chunk_are_serialized(cluster_client_mt, tmp_path,
                                                  chunk_extent):
    """
    Channels are written concurrently, so a chunk whose channel extent is
    `> 1` is shared by two channel writes and they collide exactly as two
    spatial blocks would. A spatial-only lock key could not express that.

    The other half matters as much: with one channel per chunk the two
    writes must hold *disjoint* keys, so the lock does not serialize what it
    need not.
    """
    array = _zarr(tmp_path, f'ch{chunk_extent}', (1, 2, 16, 16, 16),
                  (1, chunk_extent, 8, 8, 8))
    writer = BlockWriter(array, (1, 1, 8, 8, 8), namespace='ch')

    spatial = (slice(0, 16),) * 3
    zero = (slice(0, 1), slice(0, 1)) + spatial
    one = (slice(0, 1), slice(1, 2)) + spatial
    shared = set(writer.lock_keys(zero)) & set(writer.lock_keys(one))
    assert bool(shared) == (chunk_extent == 2)

    tasks = [(writer, (0,), zero, 3.0), (writer, (1,), one, 4.0)]
    cluster_client_mt.gather(
        cluster_client_mt.map(_fill_block, tasks, pure=False))

    assert np.all(array[0, 0] == 3.0)
    assert np.all(array[0, 1] == 4.0)


# --------------------------------------------------------------------------
# end to end, through the two public functions
# --------------------------------------------------------------------------


@pytest.mark.parametrize('block_size,chunks,shards', [
    # B smaller than the shard: several whole blocks per shard, all of them
    # serialized against each other
    ((8, 8, 8), (4, 4, 4), (16, 16, 16)),
    # B larger than the shard, and not a multiple of it
    ((25, 25, 25), (8, 8, 8), None),
    # B coprime with the chunk in every axis
    ((11, 13, 17), (4, 4, 4), (8, 8, 8)),
], ids=['block-smaller-than-shard', 'block-not-a-multiple-of-the-chunk',
        'block-coprime-with-the-chunk'])
def test_apply_transform_matches_an_in_memory_run_for_any_block_size(
        cluster_client_mt, tmp_path, block_size, chunks, shards):
    """
    The capability this change exists to allow: the block lattice and the
    chunk/shard grid are now independent in both directions, and the warped
    volume has to come out the same either way.

    An in-memory output is the reference - it assigns element by element,
    so it has no write unit and cannot suffer the read-modify-write loss
    being guarded against here.
    """
    shape = (40, 44, 48)
    rng = np.random.default_rng(7)
    fix = rng.random(shape, dtype=np.float32)
    mov = rng.random(shape, dtype=np.float32)
    fix_image = ImageData(image_arraydata=fix, read_attrs=False)
    mov_image = ImageData(image_arraydata=mov, read_attrs=False)
    spacing = np.array([1.0, 1.0, 1.0])

    def run(output):
        return distributed_apply_transform(
            fix_image, spacing, mov_image, spacing,
            block_size, [np.eye(4)], cluster_client_mt,
            overlap_factor=0.25, aligned_data=output)

    reference = np.zeros(shape, dtype=np.float32)
    run(reference)
    assert np.any(reference)

    written = _zarr(tmp_path, 'warped', shape, chunks, shards=shards)
    run(written)
    np.testing.assert_array_equal(written[...], reference)


def test_invert_field_matches_an_in_memory_run_for_an_unaligned_block_size(
        cluster_client_mt, tmp_path):
    """
    Same guarantee for the inverse path, whose output is `(z,y,x,d)` - the
    vector axis is never split, so only the spatial axes can collide, but
    they collide in exactly the same way.
    """
    shape = (24, 24, 24)
    field = np.zeros(shape + (3,), dtype=np.float32)
    field[..., 0] = 0.5

    block_size = (10, 10, 10)     # neither divides nor is divided by 8
    kwargs = dict(overlap_factor=0.25, iterations=(2,),
                  shrink_spacings=(None,), smooth_sigmas=(0.,),
                  use_root=False, verbose=False)

    reference = np.zeros_like(field)
    distributed_invert_displacement_vector_field(
        field, np.array([1.0, 1.0, 1.0]), block_size, reference,
        cluster_client_mt, **kwargs)
    assert np.any(reference)

    written = _zarr(tmp_path, 'inv', shape + (3,), (8, 8, 8, 3))
    distributed_invert_displacement_vector_field(
        field, np.array([1.0, 1.0, 1.0]), block_size, written,
        cluster_client_mt, **kwargs)
    np.testing.assert_array_equal(written[...], reference)
