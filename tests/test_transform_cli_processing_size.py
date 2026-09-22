"""
How the two transform CLIs resolve their block partition.

`--processing-size` exists because writes are now locked per write unit -
the block partition no longer has to line up with the output's chunk or
shard grid, so it is worth being able to set it independently of either.
Unset, it keeps the previous behaviour: a whole write unit per block, which
is the size at which no two blocks share a unit and the locks never contend.

Both CLIs take geometry in **xyz** order and reverse it to the zyx the rest
of the codebase uses, so every assertion here is on the reversed value.
"""

import logging
import sys

import numpy as np
import pytest

import bigstream.io_utility as io_utility
import bigstream.tools.main_apply_local_transform as apply_cli
import bigstream.tools.main_compute_local_inverse as inverse_cli


SHAPE = (40, 44, 48)
CHUNKS = (8, 8, 8)


@pytest.fixture(scope='module')
def shared_cluster():
    distributed = pytest.importorskip('dask.distributed')
    cluster = distributed.LocalCluster(n_workers=1, threads_per_worker=1,
                                       processes=False, dashboard_address=':0',
                                       silence_logs=logging.CRITICAL)
    yield cluster
    cluster.close()


@pytest.fixture
def apply_spy(monkeypatch, shared_cluster):
    """
    Run `main_apply_local_transform` far enough to resolve its geometry, and
    stop at the distributed call - what that call is handed is the whole
    question here, and actually warping a volume answers none of it.
    """
    seen = {}

    def spy(fix, fix_spacing, mov, mov_spacing, process_blocksize,
            transform_list, cluster_client, **kwargs):
        seen['process_blocksize'] = tuple(process_blocksize)
        seen.update(kwargs)
        return None

    monkeypatch.setattr(apply_cli, 'distributed_apply_transform', spy)
    monkeypatch.setattr(apply_cli, 'LocalCluster',
                        lambda **kwargs: shared_cluster)
    return seen


@pytest.fixture
def inverse_spy(monkeypatch, shared_cluster):
    seen = {}

    def spy(field, spacing, blocksize, output, cluster_client, **kwargs):
        seen['blocksize'] = tuple(blocksize)
        seen.update(kwargs)
        return output

    monkeypatch.setattr(inverse_cli,
                        'distributed_invert_displacement_vector_field', spy)
    monkeypatch.setattr(inverse_cli, 'LocalCluster',
                        lambda **kwargs: shared_cluster)
    return seen


@pytest.fixture
def volumes(tmp_path):
    rng = np.random.default_rng(5)
    paths = {}
    for name in ('fix', 'mov'):
        path = str(tmp_path / f'{name}.zarr')
        array = io_utility.create_dataset_array(
            path, 's0', SHAPE, CHUNKS, np.float32,
            overwrite=True, zarr_format=3)
        array[...] = rng.random(SHAPE, dtype=np.float32)
        paths[name] = path
    return paths


@pytest.fixture
def deform_field(tmp_path):
    """A `(z,y,x,d)` field for the inverse CLI to read."""
    path = str(tmp_path / 'deform.zarr')
    array = io_utility.create_dataset_array(
        path, 's0', SHAPE + (3,), CHUNKS + (3,), np.float32,
        overwrite=True, zarr_format=3)
    array[...] = 0.25
    return path


def _run_apply(volumes, tmp_path, extra):
    args = apply_cli._define_args().parse_args([
        '--fix', volumes['fix'], '--fix-subpath', 's0',
        '--mov', volumes['mov'], '--mov-subpath', 's0',
        '--output', str(tmp_path / 'warped.zarr'), '--output-subpath', 's0',
        '--output-blocksize', '8,8,8',
        '--local-dask-workers', '1',
    ] + extra)
    apply_cli.logger = logging.getLogger('apply-cli-test')
    apply_cli._run_apply_transform(args)


def _run_inverse(deform_field, tmp_path, extra):
    args = inverse_cli._define_args().parse_args([
        '--transform-dir', deform_field, '--transform-subpath', 's0',
        '--inv-transform-dir', str(tmp_path / 'inv.zarr'),
        '--inv-transform-subpath', 's0',
        '--inv-transform-blocksize', '8,8,8',
        '--local-dask-workers', '1',
    ] + extra)
    inverse_cli.logger = logging.getLogger('inverse-cli-test')
    inverse_cli._run_compute_inverse(args)


# --------------------------------------------------------------------------
# main_apply_local_transform
# --------------------------------------------------------------------------


def test_apply_processing_size_is_taken_in_xyz_and_used_in_zyx(
        volumes, tmp_path, apply_spy):
    """
    The point of the flag: a partition that is neither the chunk nor the
    shard, and is not a multiple of either.
    """
    _run_apply(volumes, tmp_path, ['--processing-size', '20,15,11'])
    assert apply_spy['process_blocksize'] == (11, 15, 20)


def test_apply_processing_size_defaults_to_the_shard_when_sharding_is_on(
        volumes, tmp_path, apply_spy):
    """Zarr v3 with a sharding factor: the write unit is the shard."""
    _run_apply(volumes, tmp_path,
               ['--output-zarr-format', '3', '--output-sharding-factor', '4,2,2'])
    # chunk 8,8,8 (zyx) times the factor reversed to zyx -> 2,2,4
    assert apply_spy['process_blocksize'] == (16, 16, 32)


@pytest.mark.parametrize('zarr_args', [
    ['--output-zarr-format', '2'],
    ['--output-zarr-format', '3'],
], ids=['zarr-v2', 'zarr-v3-unsharded'])
def test_apply_processing_size_defaults_to_the_chunk_without_sharding(
        volumes, tmp_path, apply_spy, zarr_args):
    """No shards - v2 never has them, v3 only with a factor - so the write
    unit is the chunk and that is the default partition."""
    _run_apply(volumes, tmp_path, zarr_args)
    assert apply_spy['process_blocksize'] == (8, 8, 8)


def test_apply_processing_size_wins_over_the_shard(volumes, tmp_path,
                                                   apply_spy):
    _run_apply(volumes, tmp_path,
               ['--output-zarr-format', '3', '--output-sharding-factor', '4,2,2',
                '--processing-size', '20,15,11'])
    assert apply_spy['process_blocksize'] == (11, 15, 20)


def test_apply_processing_overlap_factor_reaches_the_distributed_call(
        volumes, tmp_path, apply_spy):
    """This replaced `--blocks-overlap-factor`, which no longer exists."""
    _run_apply(volumes, tmp_path, ['--processing-overlap-factor', '0.3'])
    assert apply_spy['overlap_factor'] == 0.3

    with pytest.raises(SystemExit):
        apply_cli._define_args().parse_args(['--blocks-overlap-factor', '0.3'])


def test_apply_default_overlap_factor_is_unchanged(volumes, tmp_path,
                                                   apply_spy):
    _run_apply(volumes, tmp_path, [])
    assert apply_spy['overlap_factor'] == 0.1


# --------------------------------------------------------------------------
# main_compute_local_inverse
# --------------------------------------------------------------------------


def test_inverse_processing_size_is_taken_in_xyz_and_used_in_zyx(
        deform_field, tmp_path, inverse_spy):
    _run_inverse(deform_field, tmp_path, ['--processing-size', '20,15,11'])
    assert inverse_spy['blocksize'] == (11, 15, 20)


def test_inverse_processing_size_defaults_to_the_shard_when_sharding_is_on(
        deform_field, tmp_path, inverse_spy):
    """The vector axis is never sharded, so only the spatial shape is the
    partition."""
    _run_inverse(deform_field, tmp_path,
                 ['--output-zarr-format', '3',
                  '--output-sharding-factor', '4,2,2'])
    assert inverse_spy['blocksize'] == (16, 16, 32)


@pytest.mark.parametrize('zarr_args', [
    ['--output-zarr-format', '2'],
    ['--output-zarr-format', '3'],
], ids=['zarr-v2', 'zarr-v3-unsharded'])
def test_inverse_processing_size_defaults_to_the_blocksize_without_sharding(
        deform_field, tmp_path, inverse_spy, zarr_args):
    _run_inverse(deform_field, tmp_path, zarr_args)
    assert inverse_spy['blocksize'] == (8, 8, 8)


def test_inverse_processing_size_wins_over_the_shard(deform_field, tmp_path,
                                                     inverse_spy):
    _run_inverse(deform_field, tmp_path,
                 ['--output-zarr-format', '3',
                  '--output-sharding-factor', '4,2,2',
                  '--processing-size', '20,15,11'])
    assert inverse_spy['blocksize'] == (11, 15, 20)


def test_inverse_processing_overlap_factor_reaches_the_distributed_call(
        deform_field, tmp_path, inverse_spy):
    _run_inverse(deform_field, tmp_path, ['--processing-overlap-factor', '0.3'])
    assert inverse_spy['overlap_factor'] == 0.3
