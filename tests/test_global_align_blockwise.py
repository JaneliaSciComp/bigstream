"""
The blockwise branch of the global alignment tool.

`main_global_align_pipeline` runs `blockwise_alignment_pipeline` whenever a
processing size is given. Global alignment deliberately uses a *single*
pass - multi-pass cascading is a local-stage tool - and unlike the local
tool it keeps the rest of its job: warping the moving image and computing
the inverse field. These tests pin the pass count and the knobs, not the
downstream stages.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pytest

import bigstream.distributed_align as da
import bigstream.tools.main_global_align_pipeline as gap

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
    # returns (transform, aligned_volume); the warped volume is the stage
    # the global tool keeps and this file does not exercise
    transform, _aligned = gap._align_global_data(
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
