import numpy as np
import pytest

from bigstream.distutils import validate_processing_block_size


class _FakeOutputArray:

    def __init__(self, chunks, shards=None):
        self.chunks = chunks
        self.shards = shards


def test_unsharded_block():
    """Block size matching chunk size on an unsharded array is fine."""
    output = _FakeOutputArray(chunks=(64, 64, 64))
    validate_processing_block_size(output, np.array([64, 64, 64]))


def test_processing_block_is_chunksize():
    """Chunk-sized block on a sharded array must be rejected: multiple such
    blocks can land in the same shard with no synchronization."""
    output = _FakeOutputArray(chunks=(64, 64, 64), shards=(256, 256, 256))
    with pytest.raises(ValueError, match='shard size'):
        validate_processing_block_size(output, np.array([64, 64, 64]))


def test_processing_block_is_shardsize():
    """Shard-sized block on a sharded array is fine: each block owns a whole shard."""
    output = _FakeOutputArray(chunks=(64, 64, 64), shards=(256, 256, 256))
    validate_processing_block_size(output, np.array([256, 256, 256]))
