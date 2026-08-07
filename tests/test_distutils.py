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
