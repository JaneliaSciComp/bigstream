from bigstream.tools.utils import get_processing_size


def test_no_storage_unit_returns_input():
    """Without a shard or block shape there is nothing to align to."""
    assert get_processing_size(
        input_processing_size=(100, 100, 100)) == (100, 100, 100)


def test_unsharded_block_multiple():
    """Input already a multiple of the blocksize is returned unchanged."""
    assert get_processing_size(
        input_processing_size=(64, 64, 64),
        blocksize=(64, 64, 64),
    ) == (64, 64, 64)


def test_unsharded_rounds_up_to_block_multiple():
    """Input not aligned to the blocksize is rounded up, never down: an
    oversized processing block is clipped to the array bounds when the
    block grid is built, while an undersized one would let two workers
    share a storage unit."""
    assert get_processing_size(
        input_processing_size=(100, 100, 100),
        blocksize=(64, 64, 64),
    ) == (128, 128, 128)


def test_sharded_chunk_sized_input_becomes_shard_sized():
    """A chunk-sized processing size on a sharded array must grow to the
    shard size so each worker owns a whole shard."""
    assert get_processing_size(
        input_processing_size=(64, 64, 64),
        shard_shape=(256, 256, 256),
    ) == (256, 256, 256)


def test_sharded_rounds_up_to_shard_multiple():
    """A processing size above one shard is rounded up to a whole number of
    shards so no shard is split between workers."""
    assert get_processing_size(
        input_processing_size=(300, 300, 300),
        shard_shape=(256, 256, 256),
    ) == (512, 512, 512)
