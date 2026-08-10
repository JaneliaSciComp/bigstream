from bigstream.tools.utils import get_processing_size


def test_no_storage_unit_returns_input():
    """Without a shard or block shape there is nothing to align to."""
    assert get_processing_size(
        input_processing_size=(100, 100, 100)) == (100, 100, 100)
