"""
End-to-end tests for the prototype local alignment CLI.

`main_local_pipeline_prototype` does one thing - write the deformation field.
The tests below check that it does that through the multi-pass pipeline, and
that it does *not* quietly do the other two things its predecessor does
(invert the field, warp the moving image).
"""

import logging
import sys

import numpy as np
import pytest
import yaml

import bigstream.distributed_align_prototype as dap
import bigstream.io_utility as io_utility
import bigstream.tools.main_local_pipeline_prototype as cli

from bigstream.image_data import ImageData

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from test_blockwise_alignment import _fake_alignment_pipeline  # noqa: E402


SHAPE = (40, 44, 48)


@pytest.fixture
def volumes(tmp_path):
    rng = np.random.default_rng(99)
    paths = {}
    for name in ('fix', 'mov'):
        path = str(tmp_path / f'{name}.zarr')
        array = io_utility.create_dataset_array(
            path, 's0', SHAPE, (16, 16, 16), np.float32,
            overwrite=True, zarr_format=3)
        array[...] = rng.random(SHAPE, dtype=np.float32)
        paths[name] = path
    return paths


@pytest.fixture(scope='module')
def shared_cluster():
    """One in-process cluster for the module, reused by every CLI run."""
    distributed = pytest.importorskip('dask.distributed')
    cluster = distributed.LocalCluster(n_workers=2, threads_per_worker=1,
                                       processes=False, dashboard_address=':0',
                                       silence_logs=logging.CRITICAL)
    yield cluster
    cluster.close()


@pytest.fixture
def in_process_cluster(monkeypatch, shared_cluster):
    """
    Point the CLI at an in-process cluster.

    It builds its own `LocalCluster`, which defaults to subprocess workers,
    and the deterministic stand-in for the per-block fit is installed with
    monkeypatch - which does not cross a process boundary. Handing back a
    shared cluster also keeps the CLI's own `client.close()` intact, so the
    lifecycle under test is the real one.
    """
    monkeypatch.setattr(cli, 'LocalCluster', lambda **kwargs: shared_cluster)
    monkeypatch.setattr(dap, 'alignment_pipeline', _fake_alignment_pipeline)
    # the real one calls logging.basicConfig, which would take over pytest's
    # own logging for the rest of the session
    monkeypatch.setattr(cli, 'configure_logging',
                        lambda *a, **k: logging.getLogger('prototype-cli-test'))


def _write_config(tmp_path, passes):
    path = tmp_path / 'align.yml'
    path.write_text(yaml.safe_dump({'local_align': passes}))
    return str(path)


def _run(argv):
    saved = sys.argv
    sys.argv = ['main_local_pipeline_prototype'] + argv
    try:
        cli.main()
    finally:
        sys.argv = saved


def _base_argv(volumes, tmp_path, config):
    return [
        '--local-fix', volumes['fix'], '--local-fix-subpath', 's0',
        '--local-mov', volumes['mov'], '--local-mov-subpath', 's0',
        '--local-output-dir', str(tmp_path / 'out'),
        '--local-transform-name', 'deform.zarr',
        '--local-transform-subpath', 's0',
        '--local-output-blocksize', '10,10,10',
        '--align-config', config,
        '--local-dask-workers', '2',
    ]


def test_two_pass_config_writes_the_field_and_its_per_pass_parts(
        volumes, tmp_path, in_process_cluster):
    """
    The whole point of the tool: a multi-pass config runs, the composed field
    lands at the transform subpath, and each pass's own residual field is
    kept beside it so the pass-over-pass diagnostics can be read back.
    """
    config = _write_config(tmp_path, {
        'processing_size': [16, 16, 16],
        'processing_halo_factor': 0.25,
        'alignment_passes': [
            {'processing_offset': [0, 0, 0],
             'alignment_steps': [{'deform': {}}]},
            {'processing_offset': [4, 4, 4],
             'processing_halo_factor': 0.125,
             'alignment_steps': [{'deform': {}}]},
        ],
    })
    _run(_base_argv(volumes, tmp_path, config))

    out = tmp_path / 'out' / 'deform.zarr'
    composed = ImageData(str(out), 's0', open_image=True).image_array
    assert composed.shape == SHAPE + (3,)
    assert np.any(composed[...])

    first = ImageData(str(out), 's0_passes/pass1', open_image=True).image_array
    second = ImageData(str(out), 's0_passes/pass2', open_image=True).image_array
    assert np.any(first[...]) and np.any(second[...])
    # the composed field is neither pass on its own
    assert not np.array_equal(composed[...], first[...])
    assert not np.array_equal(composed[...], second[...])


def test_only_the_deformation_field_is_produced(volumes, tmp_path,
                                                in_process_cluster):
    """No inverse field, no warped volume - that is the whole scope change."""
    config = _write_config(tmp_path, {
        'processing_size': [16, 16, 16],
        'processing_halo_factor': 0.25,
        'alignment_passes': [{'alignment_steps': [{'deform': {}}]}],
    })
    _run(_base_argv(volumes, tmp_path, config))

    produced = sorted(p.name for p in (tmp_path / 'out').iterdir())
    assert produced == ['deform.zarr']
    # a single pass writes straight to the output, so there are no leftovers
    assert not (tmp_path / 'out' / 'deform.zarr' / 's0_passes').exists()


def test_a_single_steps_config_still_runs_as_one_pass(volumes, tmp_path,
                                                      in_process_cluster):
    """
    A config written for the single-pass pipeline has no `alignment_passes`.
    Refusing it would make an A/B against `main_local_align_pipeline`
    impossible, so its flat `steps:` list is run as one pass.
    """
    config = _write_config(tmp_path, {
        'steps': ['deform'],
        'block_size': [16, 16, 16],
        'block_overlap': 0.25,
    })
    _run(_base_argv(volumes, tmp_path, config))
    field = ImageData(str(tmp_path / 'out' / 'deform.zarr'), 's0',
                      open_image=True).image_array
    assert np.any(field[...])


def test_processing_size_need_not_divide_the_chunk(volumes, tmp_path,
                                                   in_process_cluster):
    """
    The predecessor raises when the chunk does not divide the processing
    size. Here the block footprint is 16 + 2*4 = 24 voxels over a 10 voxel
    chunk, which is exactly the case write locking now handles.
    """
    config = _write_config(tmp_path, {
        'processing_size': [16, 16, 16],
        'processing_halo_factor': 0.25,
        'alignment_passes': [{'alignment_steps': [{'deform': {}}]}],
    })
    argv = _base_argv(volumes, tmp_path, config)
    _run(argv + ['--local-transform-blocksize', '10,10,10'])

    field = ImageData(str(tmp_path / 'out' / 'deform.zarr'), 's0',
                      open_image=True).image_array
    assert field.chunks[:3] == (10, 10, 10)
    assert np.any(field[...])


@pytest.mark.parametrize('flag,value', [
    ('--local-inv-transform-name', 'inv.zarr'),
    ('--local-align-name', 'warped.zarr'),
])
def test_options_this_tool_cannot_honour_are_refused(volumes, tmp_path,
                                                     in_process_cluster,
                                                     flag, value):
    """
    The inverse and warp flags come from the shared input definition. They
    are hidden from --help, but passing one has to fail rather than be
    silently dropped - the caller asked for an output they would not get.
    """
    config = _write_config(tmp_path, {
        'processing_size': [16, 16, 16],
        'processing_halo_factor': 0.25,
        'alignment_passes': [{'alignment_steps': [{'deform': {}}]}],
    })
    with pytest.raises(SystemExit) as excinfo:
        _run(_base_argv(volumes, tmp_path, config) + [flag, value])
    assert 'not supported' in str(excinfo.value)
    assert 'main_local_align_pipeline' in str(excinfo.value)


def test_help_hides_the_unsupported_options(capsys):
    parser = cli._define_args(cli.CliArgsHelper('local'))
    parser.print_help()
    text = capsys.readouterr().out
    assert '--local-transform-name' in text
    assert '--local-inv-transform-name' not in text
    assert '--local-align-name' not in text
