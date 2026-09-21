"""
End-to-end tests for the local alignment CLI.

`main_local_align_pipeline` writes the deformation field, and warps the
moving image through it when an aligned output is named. The tests below
check both, and that it does *not* quietly compute the inverse field - the
one thing the tool it replaced did that this one leaves to
`main_compute_local_inverse`.
"""

import logging
import sys

import numpy as np
import pytest
import yaml

import bigstream.distributed_align as da
import bigstream.io_utility as io_utility
import bigstream.tools.main_local_align_pipeline as cli

from bigstream.distributed_align import (default_blend_ramp_from_config,
                                         default_pass_geometry_from_config)
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
    monkeypatch.setattr(da, 'alignment_pipeline', _fake_alignment_pipeline)
    # the real one calls logging.basicConfig, which would take over pytest's
    # own logging for the rest of the session
    monkeypatch.setattr(cli, 'configure_logging',
                        lambda *a, **k: logging.getLogger('local-pipeline-cli-test'))


def _write_config(tmp_path, passes):
    path = tmp_path / 'align.yml'
    path.write_text(yaml.safe_dump({'local_align': passes}))
    return str(path)


def _run(argv):
    saved = sys.argv
    sys.argv = ['main_local_align_pipeline'] + argv
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


def test_resume_from_pass_skips_recomputing_earlier_passes(
        volumes, tmp_path, in_process_cluster, monkeypatch):
    """
    Standing in for "pass 2 failed, don't redo pass 1": --resume-from-pass 2
    must not recompute pass 1's blocks, must read pass 1's field back from
    disk to seed pass 2's cascade, and must recompose to the same result a
    normal, un-resumed run produces.

    Pass 1 is compared exactly - it is untouched bytes on disk, so anything
    but equality means it was recomputed or clobbered. Pass 2 and the
    composed field are compared numerically, not bitwise: this fixture's
    cluster has two workers, and a pass overlap-adds its blocks into shared
    regions by read-modify-write, so float32 accumulation order - and with
    it the last bits - varies run to run. Only the single-worker fixture in
    test_blockwise_alignment.py can assert bit-for-bit.
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
    argv = _base_argv(volumes, tmp_path, config)
    _run(argv)

    out = tmp_path / 'out' / 'deform.zarr'
    baseline_composed = ImageData(str(out), 's0', open_image=True).image_array[...]
    baseline_pass1 = ImageData(str(out), 's0_passes/pass1',
                               open_image=True).image_array[...]
    baseline_pass2 = ImageData(str(out), 's0_passes/pass2',
                               open_image=True).image_array[...]

    contexts = []

    def recording_pipeline(*args, context='', **kwargs):
        contexts.append(context)
        return _fake_alignment_pipeline(*args, context=context, **kwargs)

    monkeypatch.setattr(da, 'alignment_pipeline', recording_pipeline)

    _run(argv + ['--resume-from-pass', '2'])

    assert contexts and all(not c.startswith('pass1') for c in contexts)
    assert any(c.startswith('pass2') for c in contexts)

    resumed_composed = ImageData(str(out), 's0', open_image=True).image_array[...]
    resumed_pass1 = ImageData(str(out), 's0_passes/pass1',
                              open_image=True).image_array[...]
    resumed_pass2 = ImageData(str(out), 's0_passes/pass2',
                              open_image=True).image_array[...]
    assert np.array_equal(resumed_pass1, baseline_pass1)
    assert np.allclose(resumed_pass2, baseline_pass2)
    assert np.allclose(resumed_composed, baseline_composed)


def test_resume_from_pass_beyond_configured_passes_is_ignored(
        volumes, tmp_path, in_process_cluster, monkeypatch):
    """A resume target with nothing to resume from just runs everything."""
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
    argv = _base_argv(volumes, tmp_path, config)

    contexts = []

    def recording_pipeline(*args, context='', **kwargs):
        contexts.append(context)
        return _fake_alignment_pipeline(*args, context=context, **kwargs)

    monkeypatch.setattr(da, 'alignment_pipeline', recording_pipeline)

    _run(argv + ['--resume-from-pass', '9'])

    assert any(c.startswith('pass1') for c in contexts)
    assert any(c.startswith('pass2') for c in contexts)


def test_only_the_deformation_field_is_produced(volumes, tmp_path,
                                                in_process_cluster):
    """
    Warping is opt-in: with no aligned output named, the field is the only
    thing written. No inverse either, ever.
    """
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


def test_naming_an_aligned_output_also_warps_the_moving_image(
        volumes, tmp_path, in_process_cluster):
    """
    The warp stage, restored from the ome-dev tool: naming an aligned output
    writes the field *and* resamples the moving image onto the fixed grid
    through it.
    """
    config = _write_config(tmp_path, {
        'processing_size': [16, 16, 16],
        'processing_halo_factor': 0.25,
        'alignment_passes': [{'alignment_steps': [{'deform': {}}]}],
    })
    _run(_base_argv(volumes, tmp_path, config)
         + ['--local-align-name', 'warped.zarr',
            '--local-align-subpath', 's0'])

    produced = sorted(p.name for p in (tmp_path / 'out').iterdir())
    assert produced == ['deform.zarr', 'warped.zarr']

    warped = ImageData(str(tmp_path / 'out' / 'warped.zarr'), 's0',
                       open_image=True).image_array
    assert warped.shape == SHAPE
    assert np.any(warped[...])
    # it is the moving image resampled, not a copy of either input
    mov = ImageData(volumes['mov'], 's0', open_image=True).image_array
    fix = ImageData(volumes['fix'], 's0', open_image=True).image_array
    assert not np.array_equal(warped[...], mov[...])
    assert not np.array_equal(warped[...], fix[...])


def test_the_warp_uses_the_align_blocksize_when_given(volumes, tmp_path,
                                                      in_process_cluster):
    """
    `--local-align-blocksize` chunks the warped output independently of the
    field's own chunking - they are different arrays with different access
    patterns.
    """
    config = _write_config(tmp_path, {
        'processing_size': [16, 16, 16],
        'processing_halo_factor': 0.25,
        'alignment_passes': [{'alignment_steps': [{'deform': {}}]}],
    })
    _run(_base_argv(volumes, tmp_path, config)
         + ['--local-align-name', 'warped.zarr',
            '--local-align-subpath', 's0',
            '--local-align-blocksize', '20,20,20',
            '--local-transform-blocksize', '10,10,10'])

    warped = ImageData(str(tmp_path / 'out' / 'warped.zarr'), 's0',
                       open_image=True).image_array
    field = ImageData(str(tmp_path / 'out' / 'deform.zarr'), 's0',
                      open_image=True).image_array
    assert warped.chunks[:3] == (20, 20, 20)
    assert field.chunks[:3] == (10, 10, 10)


def test_the_warp_reads_map_coordinates_args_from_the_config(
        volumes, tmp_path, in_process_cluster, monkeypatch):
    """
    How the warp resamples is configured under `apply_deform`, separately
    from the alignment steps, and has to reach `distributed_apply_transform`
    as keyword arguments.
    """
    seen = {}
    real = cli.distributed_apply_transform

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(cli, 'distributed_apply_transform', spy)

    path = tmp_path / 'align.yml'
    path.write_text(yaml.safe_dump({
        'local_align': {
            'processing_size': [16, 16, 16],
            'processing_halo_factor': 0.25,
            'alignment_passes': [{'alignment_steps': [{'deform': {}}]}],
        },
        'apply_deform': {
            'steps': ['map_coordinates'],
            'map_coordinates': {'order': 1, 'mode': 'constant'},
        },
    }))
    _run(_base_argv(volumes, tmp_path, str(path))
         + ['--local-align-name', 'warped.zarr',
            '--local-align-subpath', 's0',
            '--local-transform-overlap-factor', '0.25'])

    assert seen['order'] == 1
    assert seen['mode'] == 'constant'
    assert seen['overlap_factor'] == 0.25


def test_a_single_steps_config_still_runs_as_one_pass(volumes, tmp_path,
                                                      in_process_cluster):
    """
    A config written for the single-pass pipeline has no `alignment_passes`.
    Refusing it would strand every existing config, so its flat `steps:`
    list is run as one pass.
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
    ('--local-inv-transform-subpath', 's0'),
    ('--local-inv-transform-blocksize', '64,64,64'),
])
def test_options_this_tool_cannot_honour_are_refused(volumes, tmp_path,
                                                     in_process_cluster,
                                                     flag, value):
    """
    The inverse flags come from the shared input definition. They are hidden
    from --help, but passing one has to fail rather than be silently dropped
    - the caller asked for an output they would not get.
    """
    config = _write_config(tmp_path, {
        'processing_size': [16, 16, 16],
        'processing_halo_factor': 0.25,
        'alignment_passes': [{'alignment_steps': [{'deform': {}}]}],
    })
    with pytest.raises(SystemExit) as excinfo:
        _run(_base_argv(volumes, tmp_path, config) + [flag, value])
    assert 'not supported' in str(excinfo.value)


def test_a_flag_left_at_its_default_is_not_reported_as_given(capsys):
    """
    The refusal keys off "differs from the parser default", so the options
    this tool does not implement must not trip it when nobody passed them -
    otherwise every run fails.
    """
    descriptor = cli.CliArgsHelper('local')
    parser = cli._define_args(descriptor)
    args = parser.parse_args(['--local-transform-name', 'deform.zarr'])
    cli._reject_unsupported_args(parser, args, descriptor)   # must not raise


def test_help_hides_the_unsupported_options(capsys):
    parser = cli._define_args(cli.CliArgsHelper('local'))
    parser.print_help()
    text = capsys.readouterr().out
    assert '--local-transform-name' in text
    assert '--local-inv-transform-name' not in text
    # warping is supported again, so its flags are back in the help
    assert '--local-align-name' in text


# --------------------------------------------------------------------------
# the bundled default config
# --------------------------------------------------------------------------


def _geometry_for(tmp_path, user_config):
    path = tmp_path / 'c.yml'
    path.write_text(yaml.safe_dump(user_config))
    return default_pass_geometry_from_config(cli._load_align_config(str(path)))


def test_bundled_defaults_supply_the_pass_geometry(tmp_path):
    size, halo_factor = _geometry_for(tmp_path, {})
    assert size == [128, 128, 128]
    assert halo_factor == 0.5


@pytest.mark.parametrize('user_config,expected', [
    # the older spelling, which is what the bundled defaults use
    ({'block_size': [384, 384, 384], 'block_overlap': 0.2},
     ([384, 384, 384], 0.2)),
    # the newer spelling, which takes precedence when both are present
    ({'processing_size': [256, 256, 256], 'processing_halo_factor': 0.1},
     ([256, 256, 256], 0.1)),
])
def test_either_spelling_of_the_geometry_survives_the_default_merge(
        tmp_path, user_config, expected):
    """
    The bundled defaults are merged *under* the user's config, and
    `processing_size`/`processing_halo_factor` win over
    `block_size`/`block_overlap` when both are present. So the defaults must
    use exactly one spelling - put both in and a user setting only
    `block_size` would be silently overridden by the default
    `processing_size`.
    """
    size, halo_factor = _geometry_for(tmp_path, {'local_align': user_config})
    assert (size, halo_factor) == expected


def test_bundled_defaults_leave_the_blend_ramp_unset(tmp_path):
    """
    The bundled config must not set `blend_ramp`, or every run would silently
    switch ramp - and, through the fold bound, every run's displacement
    ceiling. Unset is linear, which is what the pipeline has always done.
    """
    path = tmp_path / 'c.yml'
    path.write_text(yaml.safe_dump({}))
    assert default_blend_ramp_from_config(cli._load_align_config(str(path))) is None


def test_configured_blend_ramp_survives_the_default_merge(tmp_path):
    path = tmp_path / 'c.yml'
    path.write_text(yaml.safe_dump({'local_align': {'blend_ramp': 'cosine'}}))
    assert (default_blend_ramp_from_config(cli._load_align_config(str(path)))
            == 'cosine')


def test_the_documented_alignment_passes_shape_parses(tmp_path):
    """The commented example in configure_bigstream, uncommented."""
    config = {'local_align': {'alignment_passes': [
        {'processing_offset': [0, 0, 0],
         'processing_halo_factor': [0.2, 0.2, 0.2],
         'alignment_steps': [{'ransac': {'alignment_spacing': 4}},
                             {'affine': {'alignment_spacing': 4.0}},
                             {'deform': {'control_point_spacing': 128}}]},
        {'processing_size': [384, 384, 384],
         'processing_offset': [96, 96, 96],
         'processing_halo_factor': [0.1, 0.1, 0.1],
         'alignment_steps': [{'deform': {'control_point_spacing': 128}}]},
    ]}}
    path = tmp_path / 'c.yml'
    path.write_text(yaml.safe_dump(config))
    merged = cli._load_align_config(str(path))
    passes = cli._get_alignment_passes(merged, [])
    assert len(passes) == 2
    resolved = [p.resolved(3, *_geometry_for(tmp_path, config)[:2])
                for p in passes]
    assert resolved[0].processing_size == (128, 128, 128)   # inherited
    assert resolved[1].processing_size == (384, 384, 384)   # restated
    assert resolved[1].processing_offset == (96, 96, 96)
    # the bundled per-step defaults are layered under the inline arguments
    deform_args = dict(resolved[0].alignment_steps)['deform']
    assert deform_args['control_point_spacing'] == 128      # from the pass
    assert deform_args['metric'] == 'MMI'                   # from the defaults


# --------------------------------------------------------------------------
# configs written for the single-pass pipeline
# --------------------------------------------------------------------------


@pytest.mark.parametrize('cli_steps,config_steps,expected', [
    # steps named on the command line win
    (['ransac', 'deform'], ['affine'], ['ransac', 'deform']),
    # otherwise local_align.steps
    ([], ['affine', 'deform'], ['affine', 'deform']),
])
def test_steps_come_from_the_command_line_or_the_config(tmp_path, cli_steps,
                                                        config_steps, expected):
    path = tmp_path / 'c.yml'
    path.write_text(yaml.safe_dump({'local_align': {'steps': config_steps}}))
    passes = cli._get_alignment_passes(cli._load_align_config(str(path)),
                                       cli_steps)
    assert len(passes) == 1
    assert [name for name, _ in passes[0].alignment_steps] == expected


def test_a_steps_config_layers_args_the_way_it_always_did(tmp_path):
    """
    With no `alignment_passes`, a step's arguments come from the top level
    per-step section, overridden by the `local_align` section for that step
    - the same precedence `get_algorithm_parameters` applies, so a config
    written for the single-pass pipeline keeps its exact meaning.
    """
    path = tmp_path / 'c.yml'
    path.write_text(yaml.safe_dump({'local_align': {
        'ransac': {'point_matches_threshold': 15},
        'deform': {'control_point_spacing': 128},
    }}))
    cli_steps = ['ransac', 'deform']

    from bigstream.tools.cli import get_algorithm_parameters
    old, _ = get_algorithm_parameters(str(path), 'local_align', cli_steps)
    new = cli._get_alignment_passes(cli._load_align_config(str(path)),
                                    cli_steps)[0].alignment_steps
    assert dict(old) == dict(new)
    # the local_align override on top of the bundled per-step defaults
    assert dict(new)['deform']['control_point_spacing'] == 128
    assert dict(new)['deform']['metric'] == 'MMI'
    assert dict(new)['ransac']['point_matches_threshold'] == 15
    assert dict(new)['ransac']['nspots'] == 2000


def test_a_null_valued_step_section_is_tolerated(tmp_path):
    """
    The bundled defaults carry `local_align.affine:` with no value - the key
    exists, holding None. `get_algorithm_parameters` feeds that straight to
    `deep_update` and raises; this path treats it as "no overrides".
    """
    path = tmp_path / 'c.yml'
    path.write_text(yaml.safe_dump({'local_align': {}}))
    passes = cli._get_alignment_passes(cli._load_align_config(str(path)),
                                       ['affine', 'deform'])
    assert [name for name, _ in passes[0].alignment_steps] == ['affine', 'deform']


def test_no_steps_anywhere_means_nothing_to_do(tmp_path):
    path = tmp_path / 'c.yml'
    path.write_text(yaml.safe_dump({'local_align': {}}))
    assert cli._get_alignment_passes(cli._load_align_config(str(path)), []) == []
