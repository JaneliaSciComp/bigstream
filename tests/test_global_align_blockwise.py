"""
The blockwise branch of the global alignment tool.

`main_global_align_pipeline` runs `blockwise_alignment_pipeline` whenever a
processing size is given. It runs the `global_align.alignment_passes` from
the config, read the same way the local tool reads its own, or - when there
are none - the global steps as a single pass. Unlike the local tool it
keeps the rest of its job: warping the moving image and computing the
inverse field. These tests pin the passes and the knobs, not the
downstream stages.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pytest

import bigstream.distributed_align as da
import bigstream.tools.main_global_align_pipeline as gap

from bigstream.alignment_config import AlignmentPass
from bigstream.image_data import ImageData

sys.path.insert(0, str(Path(__file__).parent))
from test_blockwise_alignment import _fake_alignment_pipeline  # noqa: E402


SHAPE = (32, 36, 40)


@pytest.fixture
def images():
    rng = np.random.default_rng(11)
    fix = ImageData(image_arraydata=rng.random(SHAPE, dtype=np.float32),
                    read_attrs=False)
    mov = ImageData(image_arraydata=rng.random(SHAPE, dtype=np.float32),
                    read_attrs=False)
    return fix, mov


@pytest.fixture
def patched(monkeypatch):
    """Deterministic per-block fit and an in-process cluster."""
    distributed = pytest.importorskip('dask.distributed')
    monkeypatch.setattr(da, 'alignment_pipeline', _fake_alignment_pipeline)
    monkeypatch.setattr(gap, 'LocalCluster', lambda **kwargs:
                        distributed.LocalCluster(n_workers=2,
                                                 threads_per_worker=1,
                                                 processes=False,
                                                 dashboard_address=':0',
                                                 silence_logs=logging.CRITICAL))


def _align(images, monkeypatch, **kwargs):
    fix, mov = images
    seen = {}
    real = da.blockwise_alignment_pipeline

    def spy(*args, **kw):
        seen['passes'] = args[4]
        seen['kwargs'] = kw
        return real(*args, **kw)

    monkeypatch.setattr(gap, 'blockwise_alignment_pipeline', spy)
    # returns (alignment_passes, transform, aligned_volume); the warped
    # volume is the stage the global tool keeps and this file does not
    # exercise
    _passes, transform, _aligned = gap._align_global_data(
        fix, None, mov, None,
        None,                            # no roi
        [],                              # no prealign steps
        [('deform', {})],                # the global steps
        kwargs.pop('processing_size', (16, 16, 16)),
        kwargs.pop('processing_overlap_factor', 0.25),
        0.0,                             # foreground percentage
        None,                            # mov origin transform
        [],                              # static transforms
        **kwargs,
    )
    return transform, seen


def test_blockwise_global_alignment_runs_exactly_one_pass(images, patched,
                                                          monkeypatch):
    transform, seen = _align(images, monkeypatch)
    assert len(seen['passes']) == 1
    assert seen['passes'][0].name == 'global'
    assert [n for n, _ in seen['passes'][0].alignment_steps] == ['deform']
    # a deformation field over the whole fixed image, materialized for the
    # apply/inverse stages that follow
    assert isinstance(transform, np.ndarray)
    assert transform.shape == SHAPE + (3,)
    assert np.any(transform)


def test_overlap_factor_maps_onto_the_halo_factor(images, patched, monkeypatch):
    _, seen = _align(images, monkeypatch, processing_overlap_factor=0.25)
    assert seen['kwargs']['processing_halo_factor'] == 0.25


def test_a_missing_overlap_factor_falls_back_to_a_default(images, patched,
                                                          monkeypatch):
    """
    There is no `global_align` config default for the halo the way the local
    stage has one, and a blockwise alignment cannot run without a halo to
    blend over.
    """
    _, seen = _align(images, monkeypatch, processing_overlap_factor=None)
    assert (seen['kwargs']['processing_halo_factor']
            == gap.DEFAULT_GLOBAL_OVERLAP_FACTOR)


def test_the_new_knobs_reach_the_pipeline(images, patched, monkeypatch):
    _, seen = _align(images, monkeypatch, max_write_locks=8,
                     displacement_diagnostics='FINAL_STEP')
    assert seen['kwargs']['max_write_locks'] == 8
    assert seen['kwargs']['displacement_diagnostics'] == 'FINAL_STEP'


def test_block_footprints_sharing_a_chunk_do_not_corrupt_the_field(
        images, patched, monkeypatch):
    """
    The temp store is chunked at the block *step*, but a block writes its
    footprint - step plus halo on each side - so blocks two apart land in a
    shared chunk without overlapping. Locking per write unit is what makes
    that safe; the old per-block-index lock did not cover it.
    """
    single, _ = _align(images, monkeypatch)
    again, _ = _align(images, monkeypatch)
    assert np.any(single)
    np.testing.assert_allclose(again, single, rtol=0, atol=1e-5)


# --------------------------------------------------------------------------
# global_align.alignment_passes
# --------------------------------------------------------------------------


def test_configured_passes_run_in_order_and_compose(images, patched,
                                                    monkeypatch):
    passes = [
        AlignmentPass(alignment_steps=[('deform', {})], name='coarse'),
        AlignmentPass(alignment_steps=[('deform', {})],
                      processing_offset=(8, 8, 8), name='fine'),
    ]
    transform, seen = _align(images, monkeypatch, alignment_passes=passes)
    assert [p.name for p in seen['passes']] == ['coarse', 'fine']
    # the per-pass fields go to the temp store rather than process memory
    assert seen['kwargs']['deformfield_output_factory'] is not None
    assert isinstance(transform, np.ndarray)
    assert transform.shape == SHAPE + (3,)
    assert np.any(transform)


class _Stop(Exception):
    pass


class _FakeRegArgs:
    """The few RegistrationInputs members `_run_global_align` reads before
    it hands over to `_align_global_data`."""

    def __init__(self, processing_size=None, processing_overlap_factor=None,
                 registration_steps=None):
        self.processing_size = processing_size
        self.processing_overlap_factor = processing_overlap_factor
        self.registration_steps = registration_steps or []
        self.prealign_steps = []
        self.foreground_percentage = 0.

    def get_mov_origin_transform(self):
        return None

    def get_static_transforms(self):
        return [], ()


def _run_global(images, monkeypatch, tmp_path, config, **reg_kwargs):
    import yaml
    fix, mov = images
    path = tmp_path / 'global.yml'
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(gap, 'get_input_images',
                        lambda reg_args: (fix, None, mov, None, None))
    seen = {}

    def spy(*args, **kwargs):
        seen['args'] = args
        seen['kwargs'] = kwargs
        raise _Stop()

    monkeypatch.setattr(gap, '_align_global_data', spy)
    with pytest.raises(_Stop):
        gap._run_global_align(_FakeRegArgs(**reg_kwargs), str(path),
                              None, {}, 3)
    return seen


def test_blockwise_global_alignment_reads_alignment_passes(images, monkeypatch,
                                                            tmp_path):
    seen = _run_global(images, monkeypatch, tmp_path, {'global_align': {
        'steps': ['affine'],                 # replaced by the passes
        'processing_halo_factor': 0.3,
        'blend_ramp': 'cosine',
        'alignment_passes': [
            {'alignment_steps': [{'affine': {}}]},
            {'processing_offset': [4, 4, 4],
             'alignment_steps': [{'deform': {'control_point_spacing': 64}}]},
        ],
    }}, processing_size=(16, 16, 16))
    passes = seen['kwargs']['alignment_passes']
    assert [p.name for p in passes] == ['pass1', 'pass2']
    assert [[n for n, _ in p.alignment_steps] for p in passes] == [
        ['affine'], ['deform']]
    # per-step defaults are layered under the inline arguments, as for local
    deform_args = dict(passes[1].alignment_steps)['deform']
    assert deform_args['control_point_spacing'] == 64
    assert deform_args['metric'] == 'MMI'
    # the global_align level defaults reach the pipeline
    assert seen['args'][8] == 0.3            # processing_overlap_factor
    assert seen['kwargs']['blend_ramp'] == 'cosine'


def test_command_line_overlap_factor_wins_over_the_config(images, monkeypatch,
                                                          tmp_path):
    seen = _run_global(images, monkeypatch, tmp_path, {'global_align': {
        'processing_halo_factor': 0.3,
        'alignment_passes': [{'alignment_steps': ['deform']}],
    }}, processing_size=(16, 16, 16), processing_overlap_factor=0.2)
    assert seen['args'][8] == 0.2


def test_without_alignment_passes_the_steps_run_as_one_pass(images,
                                                            monkeypatch,
                                                            tmp_path):
    seen = _run_global(images, monkeypatch, tmp_path,
                       {'global_align': {'steps': ['affine', 'deform']}},
                       processing_size=(16, 16, 16))
    passes = seen['kwargs']['alignment_passes']
    assert len(passes) == 1
    assert [n for n, _ in passes[0].alignment_steps] == ['affine', 'deform']
    # same arguments the whole-image path gets from get_algorithm_parameters
    assert dict(passes[0].alignment_steps) == dict(seen['args'][6])


def test_whole_image_alignment_ignores_alignment_passes(images, monkeypatch,
                                                         tmp_path):
    seen = _run_global(images, monkeypatch, tmp_path, {'global_align': {
        'steps': ['affine'],
        'alignment_passes': [{'alignment_steps': ['deform']}],
    }})
    assert seen['kwargs']['alignment_passes'] is None
    assert [n for n, _ in seen['args'][6]] == ['affine']
