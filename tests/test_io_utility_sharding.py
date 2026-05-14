import logging

import pytest
import zarr

from bigstream import io_utility


def _chunk_files(container_path, subpath):
    base = container_path / subpath / 'c'
    return sorted(p for p in base.rglob('*') if p.is_file())


def test_shard_creation_basic(tmp_path):
    container = tmp_path / 'shard.zarr'
    arr = io_utility.create_dataset_array(
        str(container), 's0',
        shape=(512, 512, 512), chunks=(64, 64, 64),
        dtype='uint16',
        overwrite=True,
        zarr_format=3,
        shard_shape=(256, 256, 256),
    )
    assert arr.shards == (256, 256, 256)
    assert arr.chunks == (64, 64, 64)
    arr[...] = 1

    reopened = zarr.open(str(container))['s0']
    assert reopened.shards == (256, 256, 256)
    assert reopened.chunks == (64, 64, 64)
    assert int(reopened[0, 0, 0]) == 1
    assert int(reopened[100, 100, 100]) == 1
    assert int(reopened[511, 511, 511]) == 1

    # 2x2x2 shard grid -> 8 on-disk chunk-key files (one per shard)
    assert len(_chunk_files(container, 's0')) == 8


def test_shard_not_multiple_of_chunks_raises(tmp_path):
    container = tmp_path / 'bad.zarr'
    with pytest.raises(ValueError, match='must be a positive multiple of chunks'):
        io_utility.create_dataset_array(
            str(container), 's0',
            shape=(512, 512, 512), chunks=(64, 64, 64),
            dtype='uint16',
            overwrite=True,
            zarr_format=3,
            shard_shape=(100, 100, 100),
        )


def test_shard_zero_component_raises(tmp_path):
    container = tmp_path / 'bad.zarr'
    with pytest.raises(ValueError, match='must be a positive multiple of chunks'):
        io_utility.create_dataset_array(
            str(container), 's0',
            shape=(512, 512, 512), chunks=(64, 64, 64),
            dtype='uint16',
            overwrite=True,
            zarr_format=3,
            shard_shape=(0, 64, 64),
        )


def test_shard_rank_mismatch_raises(tmp_path):
    container = tmp_path / 'bad.zarr'
    with pytest.raises(ValueError, match='different rank'):
        io_utility.create_dataset_array(
            str(container), 's0',
            shape=(512, 512, 512), chunks=(64, 64, 64),
            dtype='uint16',
            overwrite=True,
            zarr_format=3,
            shard_shape=(256, 256),
        )


def test_shard_with_v2_logs_and_writes_unsharded(tmp_path, caplog):
    container = tmp_path / 'v2.zarr'
    with caplog.at_level(logging.INFO, logger='bigstream.io_utility'):
        arr = io_utility.create_dataset_array(
            str(container), 's0',
            shape=(128, 128, 128), chunks=(64, 64, 64),
            dtype='uint16',
            overwrite=True,
            zarr_format=2,
            shard_shape=(128, 128, 128),
        )
    assert any('Ignoring shard_shape' in rec.message for rec in caplog.records)
    arr[...] = 4
    reopened = zarr.open(str(container))['s0']
    assert reopened.metadata.zarr_format == 2
    # v2 has no shards concept; the array attr should be falsy
    assert not getattr(reopened, 'shards', None)
    assert int(reopened[0, 0, 0]) == 4


def test_v3_without_shards_regression(tmp_path):
    container = tmp_path / 'v3.zarr'
    arr = io_utility.create_dataset_array(
        str(container), 's0',
        shape=(128, 128, 128), chunks=(64, 64, 64),
        dtype='uint16',
        overwrite=True,
        zarr_format=3,
    )
    arr[...] = 7
    assert arr.shards is None
    assert arr.chunks == (64, 64, 64)
    # 2x2x2 = 8 chunk files (no sharding)
    assert len(_chunk_files(container, 's0')) == 8
    reopened = zarr.open(str(container))['s0']
    assert int(reopened[0, 0, 0]) == 7


def test_v2_without_shards_regression(tmp_path):
    container = tmp_path / 'v2.zarr'
    arr = io_utility.create_dataset_array(
        str(container), 's0',
        shape=(128, 128, 128), chunks=(64, 64, 64),
        dtype='uint16',
        overwrite=True,
        zarr_format=2,
    )
    arr[...] = 3
    reopened = zarr.open(str(container))['s0']
    assert reopened.metadata.zarr_format == 2
    assert int(reopened[0, 0, 0]) == 3


def test_shard_shape_none_is_no_op(tmp_path):
    container = tmp_path / 'v3.zarr'
    arr = io_utility.create_dataset_array(
        str(container), 's0',
        shape=(128, 128, 128), chunks=(64, 64, 64),
        dtype='uint16',
        overwrite=True,
        zarr_format=3,
        shard_shape=None,
    )
    assert arr.shards is None


def test_shard_shape_empty_tuple_is_no_op(tmp_path):
    """inttuple parser yields () for an unset/empty CLI value; must be a no-op."""
    container = tmp_path / 'v3.zarr'
    arr = io_utility.create_dataset_array(
        str(container), 's0',
        shape=(128, 128, 128), chunks=(64, 64, 64),
        dtype='uint16',
        overwrite=True,
        zarr_format=3,
        shard_shape=(),
    )
    assert arr.shards is None


def test_shard_root_array(tmp_path):
    """Sharding works when the container itself is the array (no subpath)."""
    container = tmp_path / 'root.zarr'
    arr = io_utility.create_dataset_array(
        str(container), '',
        shape=(256, 256, 256), chunks=(64, 64, 64),
        dtype='uint16',
        overwrite=True,
        zarr_format=3,
        shard_shape=(128, 128, 128),
    )
    arr[...] = 5
    reopened = zarr.open(str(container))
    assert reopened.shards == (128, 128, 128)
    assert int(reopened[0, 0, 0]) == 5


def test_shard_with_leading_one_axis(tmp_path):
    """Shard shape with a leading singleton dim (time/channel) is preserved."""
    container = tmp_path / 'leading.zarr'
    arr = io_utility.create_dataset_array(
        str(container), 's0',
        shape=(1, 256, 256, 256), chunks=(1, 64, 64, 64),
        dtype='uint16',
        overwrite=True,
        zarr_format=3,
        shard_shape=(1, 128, 128, 128),
    )
    assert arr.shards == (1, 128, 128, 128)
    arr[...] = 9
    reopened = zarr.open(str(container))['s0']
    assert int(reopened[0, 0, 0, 0]) == 9


def test_shard_with_trailing_vector_axis(tmp_path):
    """Shard shape mirroring the chunks layout for displacement-vector arrays."""
    container = tmp_path / 'vec.zarr'
    arr = io_utility.create_dataset_array(
        str(container), 's0',
        shape=(256, 256, 256, 3), chunks=(64, 64, 64, 3),
        dtype='float32',
        overwrite=True,
        zarr_format=3,
        shard_shape=(128, 128, 128, 3),
    )
    assert arr.shards == (128, 128, 128, 3)
