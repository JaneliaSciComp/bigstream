"""
Tests for the multi-pass blockwise alignment pipeline.

Historical note: this file used to carry
`test_single_pass_matches_distributed_alignment`, which asserted that at one
pass and a zero lattice offset the pipeline reproduced
the older `distributed_alignment_pipeline` bit for bit. That guard passed,
and the old implementation has since been removed - there is no second
implementation left to compare against, so the test went with it. What it
covered indirectly is still pinned here by
`test_blending_weights_are_a_partition_of_unity`,
`test_block_cores_tile_the_volume_exactly` and
`test_explicit_clips_reproduce_the_index_derived_crop`.
"""

import logging
import re

from contextlib import contextmanager
from unittest import mock

import numpy as np
import pytest
import yaml

import bigstream.distributed_align as da

from bigstream.align_constraints import blend_safe_displacement_bound
from bigstream.distutils import validate_processing_block_size
from bigstream.distributed_align import (
    AlignmentPass,
    DisplacementDiagnostics,
    blockwise_alignment_pipeline,
    _BlockLattice,
    alignment_passes_from_config,
    alignment_steps_from_config,
    _axis_block_starts,
    _compose_fields_block,
    default_blend_ramp_from_config,
    default_pass_geometry_from_config,
    _get_transform_weights,
    _lock_cell_keys,
    _overlapping_block_lock_keys,
    _parse_displacement_diagnostics,
    _storage_write_unit,
    _write_lock_grid,
)
from bigstream.image_data import ImageData


# --------------------------------------------------------------------------
# phase 1 - the pass description and config parsing
# --------------------------------------------------------------------------


def test_alignment_steps_converts_single_key_dicts():
    steps = alignment_steps_from_config([
        {'ransac': {'alignment_spacing': 4}},
        {'deform': {'control_point_spacing': 128}},
    ])
    assert steps == [('ransac', {'alignment_spacing': 4}),
                     ('deform', {'control_point_spacing': 128})]


def test_alignment_steps_accepts_bare_step_names():
    assert alignment_steps_from_config(['affine']) == [('affine', {})]


def test_alignment_steps_layers_over_defaults():
    steps = alignment_steps_from_config(
        [{'deform': {'control_point_spacing': 128}}],
        step_defaults={'deform': {'control_point_spacing': 50, 'metric': 'MMI'}},
    )
    assert steps == [('deform', {'control_point_spacing': 128, 'metric': 'MMI'})]


def test_alignment_steps_rejects_multi_key_entries():
    with pytest.raises(ValueError, match='single-key mapping'):
        alignment_steps_from_config([{'affine': {}, 'deform': {}}])


def test_config_parses_into_two_passes():
    with open('tests/configs/bigstream_config.yml') as f:
        config = yaml.safe_load(f)
    size, halo_factor = default_pass_geometry_from_config(config)
    passes = alignment_passes_from_config(config)

    assert len(passes) == 2
    assert size == [192, 192, 192]
    assert halo_factor == 0.5

    first = passes[0].resolved(3, default_processing_size=size,
                               default_halo_factor=halo_factor)
    second = passes[1].resolved(3, default_processing_size=size,
                                default_halo_factor=halo_factor)
    # pass 1 omits processing_size and inherits the top level default
    assert first.processing_size == (192, 192, 192)
    assert first.processing_offset == (0, 0, 0)
    assert first.processing_halo == (96, 96, 96)  # 0.5 * 192, per side
    # pass 2 restates the size
    assert second.processing_halo_factor == [0.1, 0.1, 0.1]
    assert second.processing_size == (128, 128, 128)
    assert second.processing_offset == (32, 32, 32)
    assert second.processing_halo == (13, 13, 13)

    first_step_configs = {name: value for name, value in first.alignment_steps }
    second_step_configs = {name: value for name, value in second.alignment_steps }

    assert list(first_step_configs.keys()) == [
        'ransac', 'affine', 'deform'
    ]

    assert first_step_configs['ransac']['blob_sizes'] == [6, 14]
    assert first_step_configs['affine']['alignment_spacing'] == 4.0
    assert first_step_configs['affine']['optimizer'] == 'LBFGSB'

    assert list(second_step_configs.keys()) == [
        'affine', 'deform'
    ]
    assert second_step_configs['affine']['alignment_spacing'] == 1.0
    assert second_step_configs['affine']['optimizer'] == 'RSGD'


BLEND_RAMP_CONFIG = """
local_align:
    processing_size: [64, 64, 64]
    processing_halo_factor: 0.25
    blend_ramp: cosine
    alignment_passes:
        - name: coarse
          alignment_steps: [deform]
        - name: fine
          blend_ramp: linear
          alignment_steps: [deform]
        - name: explicit_default
          blend_ramp: cosine
          alignment_steps: [deform]
"""


def test_blend_ramp_default_and_per_pass_override():
    """
    `blend_ramp` mirrors `processing_halo_factor`: a top level default that
    every pass inherits, and a per-pass override that wins.
    """
    config = yaml.safe_load(BLEND_RAMP_CONFIG)
    default_ramp = default_blend_ramp_from_config(config)
    assert default_ramp == 'cosine'

    passes = alignment_passes_from_config(config)
    resolved = [p.resolved(3, default_processing_size=(64, 64, 64),
                           default_halo_factor=0.25,
                           default_blend_ramp=default_ramp)
                for p in passes]
    assert [p.blend_ramp for p in resolved] == ['cosine', 'linear', 'cosine']
    # the unresolved passes keep "unset" distinct from "explicitly linear"
    assert [p.blend_ramp for p in passes] == [None, 'linear', 'cosine']


def test_blend_ramp_unset_everywhere_is_linear():
    config = yaml.safe_load(BLEND_RAMP_CONFIG.replace(
        '    blend_ramp: cosine\n', '', 1))
    assert default_blend_ramp_from_config(config) is None
    resolved = alignment_passes_from_config(config)[0].resolved(
        3, default_processing_size=(64, 64, 64), default_halo_factor=0.25,
        default_blend_ramp=default_blend_ramp_from_config(config))
    assert resolved.blend_ramp == 'linear'


def test_blend_ramp_is_linear_in_the_shared_test_config():
    """The config every other test here uses must stay on the default path."""
    with open('tests/configs/bigstream_config.yml') as f:
        config = yaml.safe_load(f)
    assert default_blend_ramp_from_config(config) is None


def test_unknown_blend_ramp_is_refused_when_the_pass_resolves():
    """
    Refused at `resolved()`, which is before a single block runs - not
    silently treated as linear, which would blend with one ramp while the
    fold ceiling was computed for another.
    """
    with pytest.raises(ValueError, match='unsupported blend_ramp'):
        AlignmentPass(processing_size=(16,) * 3, processing_halo_factor=0.25,
                      blend_ramp='hann').resolved(3)
    with pytest.raises(ValueError, match='unsupported blend_ramp'):
        AlignmentPass(processing_size=(16,) * 3,
                      processing_halo_factor=0.25).resolved(
                          3, default_blend_ramp='tukey')


def test_pass_without_a_block_size_is_an_error():
    with pytest.raises(ValueError, match='no processing_size'):
        AlignmentPass(alignment_steps=[]).resolved(3, default_halo_factor=0.2)


def test_halo_factor_must_be_a_per_side_fraction():
    with pytest.raises(ValueError, match=r'\(0, 1\)'):
        AlignmentPass(processing_size=(16,) * 3,
                      processing_halo_factor=1.5).resolved(3)


def test_explicit_halo_wins_over_the_factor():
    resolved = AlignmentPass(processing_size=(100, 100, 100),
                             processing_halo_factor=0.2,
                             processing_halo=(7, 8, 9)).resolved(3)
    assert resolved.processing_halo == (7, 8, 9)


def test_zero_offset_lattice_matches_the_old_partition():
    """offset 0 must give exactly ceil(extent / B) blocks starting at 0."""
    for extent, block in [(1000, 384), (768, 384), (100, 384), (48, 16)]:
        starts = _axis_block_starts(extent, block, 0)
        assert starts[0] == 0
        assert len(starts) == int(np.ceil(extent / block))


def test_offset_lattice_emits_the_partial_block_before_the_offset():
    starts = _axis_block_starts(1000, 384, 96)
    # the cell covering voxels 0..95 starts at -288, outside the volume
    assert list(starts) == [-288, 96, 480, 864]


def test_lattice_is_anchored_at_the_volume_origin_not_the_roi():
    """
    The same block must come out of the lattice whatever region is being
    processed - that is what makes results reproducible across chunkings.
    """
    small = _BlockLattice((16, 16, 16), (4, 4, 4), (5, 5, 5), (48, 48, 48))
    large = _BlockLattice((16, 16, 16), (4, 4, 4), (5, 5, 5), (96, 96, 96))
    # the first three cells of each axis are identical
    for axis in range(3):
        assert list(small.starts[axis]) == list(large.starts[axis][:len(small.starts[axis])])


@pytest.mark.parametrize('offset', [(0, 0, 0), (4, 4, 4), (5, 3, 7)])
def test_block_cores_tile_the_volume_exactly(offset):
    extent = (40, 44, 48)
    lattice = _BlockLattice((16, 16, 16), (4, 4, 4), offset, extent)
    coverage = np.zeros(extent, dtype=int)
    for index in lattice.indices():
        coverage[lattice.core_slices(index)] += 1
    assert np.all(coverage == 1)


@pytest.mark.parametrize('offset', [(0, 0, 0), (4, 4, 4), (5, 3, 7)])
def test_clip_amounts_describe_the_clipped_footprint(offset):
    extent = (40, 44, 48)
    lattice = _BlockLattice((16, 16, 16), (4, 4, 4), offset, extent)
    nominal = np.array(lattice.block_size) + 2 * np.array(lattice.halo)
    for index in lattice.indices():
        footprint = lattice.footprint_slices(index)
        before, after = lattice.clip_amounts(index)
        actual = np.array([s.stop - s.start for s in footprint])
        assert np.all(actual == nominal - before - after)


@pytest.mark.parametrize('blend_ramp', [None, 'linear', 'cosine'])
@pytest.mark.parametrize('offset', [(0, 0, 0), (4, 4, 4), (5, 3, 7)])
def test_blending_weights_are_a_partition_of_unity(offset, blend_ramp):
    """
    Every voxel must receive a total weight of exactly 1, including at the
    volume faces where the missing off-volume neighbour's share is rebalanced
    into the blocks that remain. Without this the stitched field is silently
    scaled down (or up) wherever the sum is off.

    Parametrized over the ramp shape because this is the property the whole
    overlap-add stitch rests on, and it is not obviously shape-independent:
    it holds for any `w` with `w(t) + w(1-t) = 1`, and the N-D case follows
    only because the weights are a separable product of such profiles.
    """
    extent = (40, 44, 48)
    block, halo = (16, 16, 16), (4, 4, 4)
    lattice = _BlockLattice(block, halo, offset, extent)
    nblocks = lattice.nblocks
    selected = set(lattice.indices())
    total = np.zeros(extent)
    for index in lattice.indices():
        neighbors = {
            o: tuple(a + b for a, b in zip(index, o)) in selected
            for o in np.ndindex(*(3,) * 3)
        }
        neighbors = {tuple(np.array(o) - 1): v for o, v in neighbors.items()}
        neighbors = {
            o: tuple(a + b for a, b in zip(index, o)) in selected
            for o in neighbors
        }
        before, after = lattice.clip_amounts(index)
        weights = _get_transform_weights(index, np.array(block), np.array(halo),
                                         neighbors, nblocks, True,
                                         blend_ramp=blend_ramp,
                                         clip_before=before, clip_after=after)
        total[lattice.footprint_slices(index)] += weights
    np.testing.assert_allclose(total, 1.0, rtol=0, atol=1e-6)


def test_explicit_clips_reproduce_the_index_derived_crop():
    """
    `_get_transform_weights` crops its weight array by the amounts the
    caller passes, and falls back to deriving them from the block index when
    it gets none. That fallback is only right for a lattice anchored at
    voxel 0 - an offset lattice has partial blocks at both ends - but there
    the two must agree exactly, because that is the case the retired
    single-pass pipeline ran and the case every existing config expects.
    """
    extent = (40, 44, 48)
    block, halo = (16, 16, 16), (4, 4, 4)
    lattice = _BlockLattice(block, halo, (0, 0, 0), extent)
    nblocks = lattice.nblocks
    selected = set(lattice.indices())
    for index in lattice.indices():
        neighbors = {tuple(o): tuple(a + b for a, b in zip(index, o)) in selected
                     for o in np.array(list(np.ndindex(*(3,) * 3))) - 1}
        derived = _get_transform_weights(index, np.array(block),
                                         np.array(halo), neighbors,
                                         nblocks, True)
        before, after = lattice.clip_amounts(index)
        explicit = _get_transform_weights(index, np.array(block),
                                          np.array(halo), neighbors, nblocks,
                                          True, clip_before=before,
                                          clip_after=after)
        # the derived crop only removes one halo at the far face, so on a
        # volume that is not a whole number of blocks it leaves the weights
        # longer than the data; `_write_block_transform` trims that remainder
        # against the block shape, which is what the lattice computes up front
        assert explicit.shape == tuple(
            s.stop - s.start for s in lattice.footprint_slices(index))
        trimmed = derived[tuple(slice(0, s) for s in explicit.shape)]
        assert np.array_equal(trimmed, explicit)


# --------------------------------------------------------------------------
# phase 3 - the single pass path
# --------------------------------------------------------------------------


def _fake_alignment_pipeline(fix, mov, fix_spacing, mov_spacing, steps,
                             **kwargs):
    """
    A stand-in for `alignment_pipeline` that is local and deterministic.

    The real one is neither (elastix draws random samples and its optimizer
    couples the whole block), which is exactly why blockwise alignment needs
    a blending ramp at all. For the equivalence test we only care that both
    pipelines hand the same block data to the same function and stitch the
    results the same way, so a deterministic stand-in makes the comparison
    exact instead of statistical.
    """
    ndim = len(fix_spacing)
    shape = fix.shape
    base = float(np.mean(fix))
    field = np.zeros(shape + (ndim,), dtype=np.float32)
    axes = [np.linspace(0.0, 1.0, s, dtype=np.float32) for s in shape]
    for q in range(ndim):
        ramp = axes[q].reshape([-1 if a == q else 1 for a in range(ndim)])
        field[..., q] = 0.01 * base * (q + 1) + np.sin(6.0 * ramp + q)
    return field


def _patch_alignment_pipeline(monkeypatch, replacement):
    """Swap the per-block fit for a local, deterministic stand-in."""
    monkeypatch.setattr(da, 'alignment_pipeline', replacement)


@pytest.fixture(scope='module')
def cluster_client():
    """One worker, one thread: block writes land in a deterministic order,
    which is what makes the bit-for-bit comparison below meaningful."""
    distributed = pytest.importorskip('dask.distributed')
    cluster = distributed.LocalCluster(n_workers=1, threads_per_worker=1,
                                       processes=False, dashboard_address=':0')
    client = distributed.Client(cluster)
    yield client
    client.close()
    cluster.close()


@pytest.fixture(scope='module')
def cluster_client_mt():
    """Several workers, so concurrent writes into one write unit really
    can collide and the locking has something to do."""
    distributed = pytest.importorskip('dask.distributed')
    cluster = distributed.LocalCluster(n_workers=4, threads_per_worker=1,
                                       processes=False, dashboard_address=':0')
    client = distributed.Client(cluster)
    yield client
    client.close()
    cluster.close()


@pytest.fixture
def synthetic_volumes():
    rng = np.random.default_rng(1234)
    shape = (40, 44, 48)
    fix = rng.random(shape, dtype=np.float32)
    mov = rng.random(shape, dtype=np.float32)
    return (ImageData(image_arraydata=fix, read_attrs=False),
            ImageData(image_arraydata=mov, read_attrs=False))


def test_two_passes_cascade_and_compose(cluster_client, synthetic_volumes,
                                        monkeypatch):
    """
    Pass 2 must see pass 1's field as a static transform (so it fits only the
    residual), and the written output must be the composition of the two.
    """
    seen_static_counts = []

    def recording_pipeline(fix, mov, fix_spacing, mov_spacing, steps,
                           static_transform_list=(), **kwargs):
        seen_static_counts.append(len(static_transform_list))
        return _fake_alignment_pipeline(fix, mov, fix_spacing, mov_spacing,
                                        steps, **kwargs)

    _patch_alignment_pipeline(monkeypatch, recording_pipeline)
    fix_image, mov_image = synthetic_volumes
    shape = tuple(fix_image.spatial_dims)
    spacing = np.array([1.0, 1.0, 1.0])

    pass_fields = {}

    def factory(pass_index, field_shape):
        pass_fields[pass_index] = np.zeros(field_shape, dtype=np.float32)
        return pass_fields[pass_index]

    output = np.zeros(shape + (3,), dtype=np.float32)
    assert blockwise_alignment_pipeline(
        fix_image, spacing, mov_image, spacing,
        [AlignmentPass(alignment_steps=[('deform', {})],
                       processing_offset=(0, 0, 0)),
         AlignmentPass(alignment_steps=[('deform', {})],
                       processing_offset=(4, 4, 4),
                       processing_halo_factor=0.125)],
        cluster_client,
        processing_size=(16, 16, 16),
        processing_halo_factor=0.25,
        deformfield_final_result=output,
        deformfield_output_factory=factory,
    )

    assert set(pass_fields) == {0, 1}
    # pass 1 blocks see no static transform, pass 2 blocks see exactly one
    assert min(seen_static_counts) == 0 and max(seen_static_counts) == 1
    assert np.any(pass_fields[0]) and np.any(pass_fields[1])
    # the composed output is neither pass on its own
    assert not np.array_equal(output, pass_fields[0])
    assert not np.array_equal(output, pass_fields[1])
    # away from the boundary the composition is close to, but not equal to,
    # the plain sum - the difference is exactly the term the cascade adds
    interior = (slice(8, -8),) * 3
    total = pass_fields[0][interior] + pass_fields[1][interior]
    assert np.allclose(output[interior], total, atol=0.5)


def test_blend_ramp_reaches_the_weight_builder_per_pass(
        cluster_client, synthetic_volumes, monkeypatch):
    """
    The threading test: the ramp configured on a pass has to survive the trip
    through `_run_alignment_pass` -> the write closure ->
    `_write_block_transform` -> `_get_transform_weights`. Every link is a
    keyword with a default, so a dropped one fails silently as "linear".
    """
    seen = []
    real_weights = da._get_transform_weights

    def recording_weights(*args, blend_ramp=None, **kwargs):
        seen.append(blend_ramp)
        return real_weights(*args, blend_ramp=blend_ramp, **kwargs)

    monkeypatch.setattr(da, '_get_transform_weights', recording_weights)
    _patch_alignment_pipeline(monkeypatch, _fake_alignment_pipeline)
    fix_image, mov_image = synthetic_volumes
    spacing = np.array([1.0, 1.0, 1.0])

    assert blockwise_alignment_pipeline(
        fix_image, spacing, mov_image, spacing,
        [AlignmentPass(alignment_steps=[('deform', {})], name='a'),
         AlignmentPass(alignment_steps=[('deform', {})], name='b',
                       blend_ramp='linear')],
        cluster_client,
        processing_size=(16, 16, 16),
        processing_halo_factor=0.25,
        blend_ramp='cosine',
        deformfield_final_result=np.zeros(
            tuple(fix_image.spatial_dims) + (3,), dtype=np.float32),
    )
    # pass 'a' inherits the pipeline default, pass 'b' overrides it; both
    # appear once per block, and nothing arrives unresolved
    assert set(seen) == {'cosine', 'linear'}
    assert None not in seen


def test_cosine_pipeline_output_differs_from_linear(
        cluster_client, synthetic_volumes, monkeypatch):
    """
    End to end, the ramp choice has to change the field that reaches the
    output - and only in the overlaps, since the block cores carry weight 1
    under either shape.
    """
    fix_image, mov_image = synthetic_volumes
    spacing = np.array([1.0, 1.0, 1.0])
    shape = tuple(fix_image.spatial_dims)

    def run(ramp):
        _patch_alignment_pipeline(monkeypatch, _fake_alignment_pipeline)
        output = np.zeros(shape + (3,), dtype=np.float32)
        assert blockwise_alignment_pipeline(
            fix_image, spacing, mov_image, spacing,
            [AlignmentPass(alignment_steps=[('deform', {})])],
            cluster_client,
            processing_size=(16, 16, 16), processing_halo_factor=0.25,
            blend_ramp=ramp, deformfield_final_result=output,
        )
        return output

    linear, cosine = run('linear'), run('cosine')
    assert np.array_equal(linear, run(None))   # unset is linear
    assert not np.allclose(linear, cosine)


def _two_pass_setup(cluster_client, synthetic_volumes, monkeypatch,
                    record_into):
    _patch_alignment_pipeline(
        monkeypatch,
        lambda fix, mov, fix_spacing, mov_spacing, steps,
               static_transform_list=(), **kwargs: (
            record_into.append(len(static_transform_list)) or
            _fake_alignment_pipeline(fix, mov, fix_spacing, mov_spacing,
                                     steps, **kwargs)
        ),
    )
    fix_image, mov_image = synthetic_volumes
    shape = tuple(fix_image.spatial_dims)
    spacing = np.array([1.0, 1.0, 1.0])
    passes = [AlignmentPass(alignment_steps=[('deform', {})],
                            processing_offset=(0, 0, 0)),
             AlignmentPass(alignment_steps=[('deform', {})],
                            processing_offset=(4, 4, 4),
                            processing_halo_factor=0.125)]
    return fix_image, mov_image, shape, spacing, passes


def test_resuming_from_pass_two_skips_pass_one_and_matches_a_full_run(
        cluster_client, synthetic_volumes, monkeypatch):
    """
    Standing in for "pass 3 failed, don't redo passes 1-2": resuming from
    pass 2 with pass 1's field supplied must not recompute pass 1, and the
    composed output must be bit-for-bit what a full run produces (the fake
    pipeline is deterministic, so there is no wiggle room to hide behind).
    """
    baseline_static_counts = []
    fix_image, mov_image, shape, spacing, passes = _two_pass_setup(
        cluster_client, synthetic_volumes, monkeypatch, baseline_static_counts)

    baseline_pass_fields = {}

    def baseline_factory(pass_index, field_shape):
        baseline_pass_fields[pass_index] = np.zeros(field_shape, dtype=np.float32)
        return baseline_pass_fields[pass_index]

    baseline_output = np.zeros(shape + (3,), dtype=np.float32)
    assert blockwise_alignment_pipeline(
        fix_image, spacing, mov_image, spacing, passes, cluster_client,
        processing_size=(16, 16, 16), processing_halo_factor=0.25,
        deformfield_final_result=baseline_output, deformfield_output_factory=baseline_factory,
    )

    resumed_static_counts = []
    _patch_alignment_pipeline(
        monkeypatch,
        lambda fix, mov, fix_spacing, mov_spacing, steps,
               static_transform_list=(), **kwargs: (
            resumed_static_counts.append(len(static_transform_list)) or
            _fake_alignment_pipeline(fix, mov, fix_spacing, mov_spacing,
                                     steps, **kwargs)
        ),
    )
    resumed_pass_fields = {}

    def resumed_factory(pass_index, field_shape):
        resumed_pass_fields[pass_index] = np.zeros(field_shape, dtype=np.float32)
        return resumed_pass_fields[pass_index]

    resumed_output = np.zeros(shape + (3,), dtype=np.float32)
    assert blockwise_alignment_pipeline(
        fix_image, spacing, mov_image, spacing, passes, cluster_client,
        processing_size=(16, 16, 16), processing_halo_factor=0.25,
        deformfield_final_result=resumed_output, deformfield_output_factory=resumed_factory,
        start_pass=2, resumed_pass_fields=[baseline_pass_fields[0]],
    )

    # pass 1 never ran a block the second time - only pass 2's blocks did,
    # each seeing the one resumed static transform
    assert set(resumed_pass_fields) == {1}
    assert resumed_static_counts and all(c == 1 for c in resumed_static_counts)
    assert np.array_equal(resumed_pass_fields[1], baseline_pass_fields[1])
    assert np.array_equal(resumed_output, baseline_output)


def test_resume_pass_beyond_configured_passes_is_ignored(
        cluster_client, synthetic_volumes, monkeypatch):
    """A resume target with nothing to resume from just runs everything."""
    static_counts = []
    fix_image, mov_image, shape, spacing, passes = _two_pass_setup(
        cluster_client, synthetic_volumes, monkeypatch, static_counts)

    pass_fields = {}

    def factory(pass_index, field_shape):
        pass_fields[pass_index] = np.zeros(field_shape, dtype=np.float32)
        return pass_fields[pass_index]

    output = np.zeros(shape + (3,), dtype=np.float32)
    assert blockwise_alignment_pipeline(
        fix_image, spacing, mov_image, spacing, passes, cluster_client,
        processing_size=(16, 16, 16), processing_halo_factor=0.25,
        deformfield_final_result=output, deformfield_output_factory=factory,
        start_pass=5,
    )

    assert set(pass_fields) == {0, 1}
    assert min(static_counts) == 0 and max(static_counts) == 1


def test_resume_without_matching_resumed_fields_is_an_error(
        cluster_client, synthetic_volumes, monkeypatch):
    static_counts = []
    fix_image, mov_image, shape, spacing, passes = _two_pass_setup(
        cluster_client, synthetic_volumes, monkeypatch, static_counts)

    output = np.zeros(shape + (3,), dtype=np.float32)
    with pytest.raises(ValueError):
        blockwise_alignment_pipeline(
            fix_image, spacing, mov_image, spacing, passes, cluster_client,
            processing_size=(16, 16, 16), processing_halo_factor=0.25,
            deformfield_final_result=output,
            start_pass=2, resumed_pass_fields=[],
        )


def test_block_context_names_the_pass_as_well_as_the_block(cluster_client,
                                                           synthetic_volumes,
                                                           monkeypatch):
    """
    The `context` handed to `alignment_pipeline` is logging only, but in a
    multi-pass run the same block index is aligned once per pass, so the
    block index alone cannot tell two log lines apart. Every context must
    carry the pass label too: the pass's `name`, or `passN` when unnamed.
    """
    seen_contexts = []

    def recording_pipeline(fix, mov, fix_spacing, mov_spacing, steps,
                           context='', **kwargs):
        seen_contexts.append(context)
        return _fake_alignment_pipeline(fix, mov, fix_spacing, mov_spacing,
                                        steps, **kwargs)

    _patch_alignment_pipeline(monkeypatch, recording_pipeline)
    fix_image, mov_image = synthetic_volumes
    shape = tuple(fix_image.spatial_dims)
    spacing = np.array([1.0, 1.0, 1.0])

    output = np.zeros(shape + (3,), dtype=np.float32)
    assert blockwise_alignment_pipeline(
        fix_image, spacing, mov_image, spacing,
        [AlignmentPass(alignment_steps=[('deform', {})], name='coarse'),
         AlignmentPass(alignment_steps=[('deform', {})])],
        cluster_client,
        processing_size=(16, 16, 16),
        processing_halo_factor=0.25,
        deformfield_final_result=output,
        deformfield_output_factory=lambda i, s: np.zeros(s, dtype=np.float32),
    )

    assert seen_contexts
    # the named pass keeps its name; the unnamed one falls back to its
    # 1-based position
    labels = {c.split(' ', 1)[0] for c in seen_contexts}
    assert labels == {'coarse', 'pass2'}
    # and the block index is still there, so a context identifies exactly
    # one block of one pass
    assert all(c.split(' ', 1)[1].startswith('(') for c in seen_contexts)
    assert len(set(seen_contexts)) == len(seen_contexts)


def test_each_pass_logs_its_own_completion(cluster_client,
                                           synthetic_volumes,
                                           monkeypatch, caplog):
    """
    A pass is the unit a run is read in, so each one has to close itself
    out: the start line alone leaves a reader unable to tell a slow pass
    from a hung one, or which pass a later failure came after.
    """
    _patch_alignment_pipeline(monkeypatch, _fake_alignment_pipeline)
    fix_image, mov_image = synthetic_volumes
    shape = tuple(fix_image.spatial_dims)
    spacing = np.array([1.0, 1.0, 1.0])

    output = np.zeros(shape + (3,), dtype=np.float32)
    with caplog.at_level(logging.INFO, logger='bigstream.distributed_align'):
        assert blockwise_alignment_pipeline(
            fix_image, spacing, mov_image, spacing,
            [AlignmentPass(alignment_steps=[('deform', {})], name='coarse'),
             AlignmentPass(alignment_steps=[('deform', {})])],
            cluster_client,
            processing_size=(16, 16, 16),
            processing_halo_factor=0.25,
            deformfield_final_result=output,
            deformfield_output_factory=lambda i, s: np.zeros(s, dtype=np.float32),
        )

    done = [r.getMessage() for r in caplog.records
            if 'completed in' in r.getMessage()]
    assert len(done) == 2
    assert done[0].startswith('--- coarse (1/2) completed in')
    assert done[1].startswith('--- pass2 (2/2) completed in')
    assert all(r.levelno == logging.INFO for r in caplog.records
               if 'completed in' in r.getMessage())


def test_a_failed_pass_says_so_at_error_level(cluster_client,
                                              synthetic_volumes,
                                              monkeypatch, caplog):
    """
    A pass that loses blocks does not stop the run - later passes still
    cascade from its partial field - so the only record that anything went
    wrong is this line. It has to be findable.
    """
    # patched at the pass boundary rather than by failing a block: a block
    # that *raises* propagates out of `_collect_results` and kills the run
    # (`as_completed(raise_errors=True)`), so the only way a pass returns
    # False today is a cancelled future - a worker that died. This stands in
    # for that without needing to kill a worker.
    monkeypatch.setattr(da, '_run_alignment_pass',
                        lambda *a, **k: False)
    fix_image, mov_image = synthetic_volumes
    shape = tuple(fix_image.spatial_dims)
    spacing = np.array([1.0, 1.0, 1.0])

    output = np.zeros(shape + (3,), dtype=np.float32)
    with caplog.at_level(logging.INFO, logger='bigstream.distributed_align'):
        ok = blockwise_alignment_pipeline(
            fix_image, spacing, mov_image, spacing,
            [AlignmentPass(alignment_steps=[('deform', {})], name='coarse')],
            cluster_client,
            processing_size=(16, 16, 16),
            processing_halo_factor=0.25,
            deformfield_final_result=output,
        )
    assert not ok

    failed = [r for r in caplog.records if 'FAILED' in r.getMessage()]
    assert len(failed) == 1
    assert failed[0].levelno == logging.ERROR
    assert failed[0].getMessage().startswith('--- coarse (1/1) FAILED in')


def test_per_block_log_lines_all_name_the_pass(cluster_client,
                                               synthetic_volumes,
                                               monkeypatch, caplog):
    """
    Every per-block log line, not just the alignment context, has to say
    which pass it came from - the whole point is that a log can be read
    without guessing which lattice a block index belongs to. Any line that
    names a block index but no pass label is a line that is still ambiguous.
    """
    _patch_alignment_pipeline(monkeypatch, _fake_alignment_pipeline)
    fix_image, mov_image = synthetic_volumes
    shape = tuple(fix_image.spatial_dims)
    spacing = np.array([1.0, 1.0, 1.0])

    output = np.zeros(shape + (3,), dtype=np.float32)
    with caplog.at_level(logging.DEBUG, logger='bigstream.distributed_align'):
        assert blockwise_alignment_pipeline(
            fix_image, spacing, mov_image, spacing,
            [AlignmentPass(alignment_steps=[('deform', {})], name='coarse'),
             AlignmentPass(alignment_steps=[('deform', {})])],
            cluster_client,
            processing_size=(16, 16, 16),
            processing_halo_factor=0.25,
            deformfield_final_result=output,
            deformfield_output_factory=lambda i, s: np.zeros(s, dtype=np.float32),
        )

    messages = [r.getMessage() for r in caplog.records
                if r.name == 'bigstream.distributed_align']
    # a block index renders as a python tuple, e.g. "(0, 1, 2)"
    index_re = re.compile(r'\(-?\d+, -?\d+, -?\d+\)')
    with_index = [m for m in messages if index_re.search(m)]
    assert with_index, 'no per-block log lines were captured'

    # the compose stage is not a pass and legitimately has no label
    unlabelled = [m for m in with_index
                  if 'coarse' not in m and 'pass2' not in m
                  and not m.startswith('compose')
                  and 'Compose' not in m]
    assert not unlabelled, ('per-block log lines without a pass label:\n  '
                            + '\n  '.join(sorted(set(unlabelled))[:10]))
    # and both passes actually produced lines
    assert any('coarse' in m for m in with_index)
    assert any('pass2' in m for m in with_index)


# --------------------------------------------------------------------------
# phase 4 - field composition
# --------------------------------------------------------------------------


def test_compose_reproduces_the_analytic_composition():
    """
    `F1(F2(x))` for a linear `F1` and a constant `F2`. Linear interpolation
    reproduces a linear field exactly, so this pins down the composition
    convention (which field is applied first) as well as the arithmetic.
    """
    shape = (24, 26, 28)
    spacing = np.array([1.0, 2.0, 0.5])
    ndim = 3
    gradient = np.array([0.01, -0.02, 0.03])
    shift = np.array([0.5, -1.0, 0.25])

    coords = np.stack(np.meshgrid(
        *[np.arange(s, dtype=np.float64) for s in shape], indexing='ij'),
        axis=-1) * spacing
    first = (coords * gradient).astype(np.float32)
    second = np.broadcast_to(shift.astype(np.float32),
                             shape + (ndim,)).copy()

    output = np.zeros(shape + (ndim,), dtype=np.float32)
    block = tuple(slice(0, s) for s in shape)
    _compose_fields_block(block, fields=[first, second], spacing=spacing,
                          shape=shape, output=output)

    # u(x) = shift + gradient * (x + shift)
    expected = shift + gradient * (coords + shift)
    interior = (slice(2, -2),) * 3
    np.testing.assert_allclose(output[interior], expected[interior],
                               rtol=1e-4, atol=1e-5)


# --------------------------------------------------------------------------
# ROI handling
# --------------------------------------------------------------------------


def test_three_element_roi_extends_to_the_volume_bounds(cluster_client,
                                                        synthetic_volumes,
                                                        monkeypatch):
    """
    A ROI given as the min corner alone must extend to the array bounds
    rather than raising - that is what `_phys_roi_to_voxel` already
    documents, and reusing it is why this path does not have to rediscover
    the rule.
    """
    _patch_alignment_pipeline(monkeypatch, _fake_alignment_pipeline)
    fix_image, mov_image = synthetic_volumes
    shape = tuple(fix_image.spatial_dims)
    spacing = np.array([1.0, 1.0, 1.0])
    output = np.zeros(shape + (3,), dtype=np.float32)
    assert blockwise_alignment_pipeline(
        fix_image, spacing, mov_image, spacing,
        [AlignmentPass(alignment_steps=[('deform', {})])], cluster_client,
        processing_size=(16, 16, 16), processing_halo_factor=0.25,
        roi=(20.0, 20.0, 20.0),
        deformfield_final_result=output,
    )
    # block (0,0,0) spans voxels 0..19 and so falls entirely below the ROI;
    # the next block's footprint starts at voxel 12
    assert not np.any(output[:12])
    assert np.any(output[20:])


def test_roi_selects_blocks_without_moving_them(cluster_client,
                                                synthetic_volumes,
                                                monkeypatch):
    """
    The ROI decides *which* blocks run; it must not shift the lattice. A
    block that a ROI-restricted run computes has to be bit-for-bit the block
    the full run computed, which is the point of anchoring block coordinates
    at the volume origin.
    """
    _patch_alignment_pipeline(monkeypatch, _fake_alignment_pipeline)
    fix_image, mov_image = synthetic_volumes
    shape = tuple(fix_image.spatial_dims)
    spacing = np.array([1.0, 1.0, 1.0])

    def run(roi):
        output = np.zeros(shape + (3,), dtype=np.float32)
        assert blockwise_alignment_pipeline(
            fix_image, spacing, mov_image, spacing,
            [AlignmentPass(alignment_steps=[('deform', {})],
                           processing_offset=(4, 4, 4))],
            cluster_client,
            processing_size=(16, 16, 16), processing_halo_factor=0.25,
            roi=roi, deformfield_final_result=output,
        )
        return output

    full = run(None)
    restricted = run((0.0, 0.0, 0.0, 20.0, 44.0, 48.0))
    # the ROI drops only the last block of axis 0, whose footprint is
    # voxels 32..39; everything below that is written by exactly the same
    # blocks with exactly the same weights
    assert np.array_equal(full[:32], restricted[:32])
    assert not np.array_equal(full[32:], restricted[32:])


# --------------------------------------------------------------------------
# write locking
# --------------------------------------------------------------------------


def _zarr_field(tmp_path, name, shape, chunks, shards=None, zarr_format=3):
    import zarr
    return zarr.create_array(store=str(tmp_path / name), shape=shape,
                             chunks=chunks, shards=shards, dtype='f4',
                             zarr_format=zarr_format)


def test_write_unit_is_the_shard_when_sharded_else_the_chunk(tmp_path):
    """
    A sharded v3 array rewrites a whole shard per write, so the shard - not
    the inner chunk - is what two writers must not share.
    """
    shape, ndim = (64, 64, 64, 3), 3
    v2 = _zarr_field(tmp_path, 'v2', shape, (16, 16, 16, 3), zarr_format=2)
    v3 = _zarr_field(tmp_path, 'v3', shape, (16, 16, 16, 3))
    sharded = _zarr_field(tmp_path, 'v3s', shape, (8, 8, 8, 3),
                          shards=(32, 32, 32, 3))

    assert list(_storage_write_unit(v2, ndim)) == [16, 16, 16]
    assert list(_storage_write_unit(v3, ndim)) == [16, 16, 16]
    assert list(_storage_write_unit(sharded, ndim)) == [32, 32, 32]
    # an in-memory array assigns element by element - no unit to protect
    assert _storage_write_unit(np.zeros(shape, dtype=np.float32), ndim) is None


def test_no_lock_grid_for_an_in_memory_output():
    assert _write_lock_grid(np.zeros((32, 32, 32, 3)), 3, (16, 16, 16)) is None


def test_lock_grid_coarsens_only_when_the_key_count_would_blow_up(tmp_path):
    """
    A block much larger than the chunk touches a lot of chunks. Rather than
    hold hundreds of lock names, the grid is coarsened - but always to a
    whole multiple of the unit, so a unit is never split between two cells.
    """
    array = _zarr_field(tmp_path, 'fine', (512, 512, 512, 3), (16, 16, 16, 3))
    unit = np.array([16, 16, 16])

    # a block the size of one chunk locks at chunk granularity
    fine = _write_lock_grid(array, 3, (16, 16, 16), max_locks=64)
    assert list(fine) == list(unit)

    # a block spanning many chunks coarsens
    coarse = _write_lock_grid(array, 3, (256, 256, 256), max_locks=64)
    assert np.all(coarse % unit == 0)
    assert np.all(coarse > unit)
    touched = int(np.prod(-(-np.array([256, 256, 256]) // coarse) + 1))
    assert touched <= 64


@pytest.mark.parametrize('block_size,halo,chunk', [
    ((16, 16, 16), (4, 4, 4), (16, 16, 16)),     # footprint is not a multiple
    ((16, 16, 16), (4, 4, 4), (10, 10, 10)),     # chunk does not divide block
    ((24, 24, 24), (6, 6, 6), (7, 11, 13)),      # nothing lines up at all
])
def test_blocks_sharing_a_write_unit_share_a_lock_key(tmp_path, block_size,
                                                        halo, chunk):
    """
    The guarantee the whole scheme rests on: if two writes touch a common
    write unit then their key sets intersect, so they cannot run at the
    same time. This must hold for *any* relationship between the block
    lattice and the chunk grid - which is the point, since the block size no
    longer has to be a multiple of the chunk or shard.
    """
    extent = (48, 48, 48)
    array = _zarr_field(tmp_path, 'f', extent + (3,), tuple(chunk) + (3,))
    lattice = _BlockLattice(block_size, halo, (5, 3, 7), extent)
    footprint = np.array(block_size) + 2 * np.array(halo)
    grid = _write_lock_grid(array, 3, footprint)
    unit = _storage_write_unit(array, 3)

    def write_units_touched(coords):
        starts = np.array([s.start for s in coords]) // unit
        stops = -(-np.array([s.stop for s in coords]) // unit)
        return {tuple(int(starts[a] + o[a]) for a in range(3))
                for o in np.ndindex(*(stops - starts))}

    blocks = [(index, lattice.footprint_slices(index))
              for index in lattice.indices()]
    keys = {i: set(_lock_cell_keys(c, grid, 'ns')) for i, c in blocks}
    units = {i: write_units_touched(c) for i, c in blocks}

    checked = 0
    for a, _ in blocks:
        for b, _ in blocks:
            if a >= b:
                continue
            if units[a] & units[b]:
                assert keys[a] & keys[b], (
                    f'{a} and {b} write into a shared write unit but hold '
                    'no lock key in common')
                checked += 1
    assert checked > 0


def test_in_memory_overlapping_blocks_share_a_lock_key():
    """
    With no write unit to key on, the hazard is the overlap-add itself, so
    the keys are the block and its neighbours - and A's set names B exactly
    when B's set names A.
    """
    lattice = _BlockLattice((16, 16, 16), (4, 4, 4), (0, 0, 0), (80, 80, 80))
    selected = set(lattice.indices())
    offsets = [tuple(int(v) - 1 for v in d) for d in np.ndindex(*(3,) * 3)]
    neighbors = {
        index: {o: tuple(a + b for a, b in zip(index, o)) in selected
                for o in offsets}
        for index in lattice.indices()
    }
    keys = {i: set(_overlapping_block_lock_keys(i, neighbors[i], 'ns'))
            for i in lattice.indices()}

    # every pair whose footprints overlap must share a key
    for a in lattice.indices():
        for b in lattice.indices():
            if a >= b:
                continue
            overlaps = all(
                max(x.start, y.start) < min(x.stop, y.stop)
                for x, y in zip(lattice.footprint_slices(a),
                                lattice.footprint_slices(b)))
            if overlaps:
                assert keys[a] & keys[b], (a, b)

    # the scheme over-locks a little - two blocks that are two apart share a
    # common neighbour and so share its key without overlapping - but it
    # does not lock the whole volume
    assert not keys[(0, 0, 0)] & keys[(3, 0, 0)]

    # keys are matched as strings, so they must not depend on whether the
    # index came out of numpy or out of a plain tuple
    assert all('np.int' not in k for k in keys[(1, 1, 1)])


def test_compose_block_write_is_locked_per_write_unit(tmp_path):
    """Composition writes disjoint blocks, but they can still share a chunk."""
    array = _zarr_field(tmp_path, 'c', (32, 32, 32, 3), (16, 16, 16, 3))
    grid = _write_lock_grid(array, 3, (10, 10, 10))
    left = _lock_cell_keys((slice(0, 10),) * 3, grid, 'ns')
    right = _lock_cell_keys((slice(10, 20),) * 3, grid, 'ns')
    # the two blocks do not overlap but both land in chunk (0, 0, 0)
    assert set(left) & set(right)


def test_unaligned_zarr_output_matches_the_in_memory_result(cluster_client_mt,
                                                            synthetic_volumes,
                                                            monkeypatch,
                                                            tmp_path):
    """
    End to end on several workers, writing into a zarr whose chunks neither
    divide nor are divided by the block footprint, with sharding on so the
    write unit is the shard.

    Measured with the unit locking disabled, this does not merely lose a
    contribution - two workers rewriting one shard produce a torn object and
    the next read raises `Zstd decompression error: invalid input data`. So
    the assertion below is the mild end of the failure mode.
    """
    _patch_alignment_pipeline(monkeypatch, _fake_alignment_pipeline)
    fix_image, mov_image = synthetic_volumes
    shape = tuple(fix_image.spatial_dims)
    spacing = np.array([1.0, 1.0, 1.0])
    # block footprint is 16 + 2*4 = 24 voxels; shard is 20, chunk 10
    passes = [AlignmentPass(alignment_steps=[('deform', {})])]
    kwargs = dict(processing_size=(16, 16, 16), processing_halo_factor=0.25)

    reference = np.zeros(shape + (3,), dtype=np.float32)
    assert blockwise_alignment_pipeline(
        fix_image, spacing, mov_image, spacing, passes, cluster_client_mt,
        deformfield_final_result=reference, **kwargs)

    field = _zarr_field(tmp_path, 'out', shape + (3,), (10, 10, 10, 3),
                        shards=(20, 20, 20, 3))
    assert list(_storage_write_unit(field, 3)) == [20, 20, 20]
    assert blockwise_alignment_pipeline(
        fix_image, spacing, mov_image, spacing, passes, cluster_client_mt,
        deformfield_final_result=field, **kwargs)

    assert np.any(reference)
    # exact equality is not available - the blocks accumulate in whatever
    # order the workers happen to finish - but that reorder only moves the
    # last bits of a float32 sum of at most eight terms, measured at ~1e-7,
    # whereas a lost write drops a whole weighted block
    np.testing.assert_allclose(field[...], reference, rtol=0, atol=1e-5)


@pytest.mark.parametrize('processing_size,chunks,shards,max_locks', [
    # several whole blocks land inside one shard. This is the case the old
    # `validate_processing_block_size` rejected outright, and the one with
    # the most lock contention: every block on a shard is serialized.
    ((12, 12, 12), (6, 6, 6), (24, 24, 24), 64),
    # one block spans many shards. The old check permitted this (it only
    # required unit <= processing size) but that was never sufficient - with
    # a halo the footprints still straddle shard boundaries. `max_locks` is
    # set low so the lock grid has to coarsen, which is the branch a block
    # much larger than its write unit takes.
    ((16, 16, 16), (4, 4, 4), (8, 8, 8), 8),
], ids=['processing-size-smaller-than-shard', 'processing-size-bigger-than-shard'])
def test_field_is_correct_whatever_the_block_to_shard_ratio(
        cluster_client_mt, synthetic_volumes, monkeypatch, tmp_path,
        processing_size, chunks, shards, max_locks):
    """
    The block lattice and the shard grid are now independent, in both
    directions. Whichever is larger, the assembled field has to come out the
    same as the single-threaded in-memory one.
    """
    _patch_alignment_pipeline(monkeypatch, _fake_alignment_pipeline)
    fix_image, mov_image = synthetic_volumes
    shape = tuple(fix_image.spatial_dims)
    spacing = np.array([1.0, 1.0, 1.0])
    passes = [AlignmentPass(alignment_steps=[('deform', {})])]
    kwargs = dict(processing_size=processing_size, processing_halo_factor=0.25)

    reference = np.zeros(shape + (3,), dtype=np.float32)
    assert blockwise_alignment_pipeline(
        fix_image, spacing, mov_image, spacing, passes, cluster_client_mt,
        deformfield_final_result=reference, **kwargs)

    field = _zarr_field(tmp_path, 'out', shape + (3,), tuple(chunks) + (3,),
                        shards=tuple(shards) + (3,))
    unit = _storage_write_unit(field, 3)
    assert list(unit) == list(shards)

    # pin down which regime this parametrization is actually in, so the two
    # cases cannot silently converge if someone edits the numbers
    block = np.array(processing_size)
    footprint = block + 2 * np.round(block * 0.25).astype(int)
    grid = _write_lock_grid(field, 3, footprint, max_locks=max_locks)
    if np.all(unit > block):
        # smaller: the whole block fits in one write unit
        assert np.all(grid == unit)
        with pytest.raises(ValueError, match='too small'):
            validate_processing_block_size(field, block,
                                           reverse_output_axes=True)
    else:
        # bigger: the block spans many units, so the grid had to coarsen
        assert np.all(unit < block)
        assert np.all(grid > unit) and np.all(grid % unit == 0)
        # the old check waved this through even though blocks still share units
        validate_processing_block_size(field, block, reverse_output_axes=True)

    assert blockwise_alignment_pipeline(
        fix_image, spacing, mov_image, spacing, passes, cluster_client_mt,
        deformfield_final_result=field, max_write_locks=max_locks, **kwargs)

    assert np.any(reference)
    np.testing.assert_allclose(field[...], reference, rtol=0, atol=1e-5)


# --------------------------------------------------------------------------
# displacement diagnostics
# --------------------------------------------------------------------------


class _DiagnosticsRecorder(logging.Handler):
    """Counts whole-field diagnostic sweeps and per-block diagnostic lines."""

    def __init__(self):
        super().__init__()
        self.sweeps = []
        self.blocks = 0
        self.records = []

    def emit(self, record):
        self.records.append(record)
        message = record.getMessage()
        if 'Compute displacement diagnostics for' in message:
            self.sweeps.append(message.split('regions of the ')[-1])
        if 'block displacement diagnostics' in message:
            self.blocks += 1


@contextmanager
def _recording_logs(level):
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    recorder = _DiagnosticsRecorder()
    root.handlers = [recorder]
    root.setLevel(level)
    try:
        yield recorder
    finally:
        root.handlers, root.level = saved_handlers, saved_level


def _run_with_diagnostics(cluster_client, volumes, npasses, mode, level):
    fix_image, mov_image = volumes
    shape = tuple(fix_image.spatial_dims)
    spacing = np.array([1.0, 1.0, 1.0])
    passes = [AlignmentPass(alignment_steps=[('deform', {})],
                            processing_offset=(0, 0, 0) if i == 0 else (4, 4, 4))
              for i in range(npasses)]
    with _recording_logs(level) as recorder:
        assert blockwise_alignment_pipeline(
            fix_image, spacing, mov_image, spacing, passes, cluster_client,
            processing_size=(16, 16, 16), processing_halo_factor=0.25,
            deformfield_final_result=np.zeros(shape + (3,), dtype=np.float32),
            displacement_diagnostics=mode,
            deformfield_output_factory=lambda i, sh: np.zeros(sh, dtype=np.float32),
        )
    return recorder


@pytest.mark.parametrize('npasses,mode,expected', [
    # unset: the field is never swept
    (1, None, []),
    (2, None, []),
    # PER_STEP: after every pass, and on the composed result
    (1, 'PER_STEP', ['pass1']),
    (2, 'PER_STEP', ['pass1', 'pass2', 'composed multi-pass field']),
    # FINAL_STEP: the final field only. With one pass that pass *is* the
    # final field - there is no composition to report on afterwards
    (1, 'FINAL_STEP', ['pass1']),
    (2, 'FINAL_STEP', ['composed multi-pass field']),
])
def test_diagnostics_mode_decides_which_fields_are_swept(
        cluster_client, synthetic_volumes, monkeypatch, npasses, mode,
        expected):
    _patch_alignment_pipeline(monkeypatch, _fake_alignment_pipeline)
    recorder = _run_with_diagnostics(cluster_client, synthetic_volumes,
                                     npasses, mode, logging.INFO)
    assert recorder.sweeps == expected


@pytest.mark.parametrize('mode', [None, 'PER_STEP', 'FINAL_STEP'])
def test_per_block_diagnostics_are_debug_only_and_mode_independent(
        cluster_client, synthetic_volumes, monkeypatch, mode):
    """
    A block's own field is not the field that reaches disk - its overlap
    regions only reach their final values once every neighbour has added its
    weighted share - so the per-block sweep is debug noise, never promoted
    by the mode and never suppressed by it either.
    """
    _patch_alignment_pipeline(monkeypatch, _fake_alignment_pipeline)
    at_info = _run_with_diagnostics(cluster_client, synthetic_volumes, 1,
                                    mode, logging.INFO)
    at_debug = _run_with_diagnostics(cluster_client, synthetic_volumes, 1,
                                     mode, logging.DEBUG)
    assert at_info.blocks == 0
    assert at_debug.blocks > 0
    # the mode governs whole-field sweeps only
    assert at_info.sweeps == at_debug.sweeps


def test_the_field_sweep_runs_on_the_cluster(cluster_client):
    """
    Every region is a cluster task, not a client-side loop.

    The jacobian determinant is the expensive part of a sweep, and on a
    production field there are many regions of it, so this is the whole
    reason the sweep is distributed. Asserted by counting what was
    submitted - one task per region - rather than by timing it.
    """
    shape = (32, 32, 32)
    field = np.zeros(shape + (3,), dtype=np.float32)
    field[..., 0] = np.linspace(0, 1, shape[0], dtype=np.float32)[:, None, None]

    ids, coords = da._tile_volume(shape, (16, 16, 16))
    assert len(ids) == 8

    submitted = []
    real_map = cluster_client.map

    def counting_map(fn, items, **kwargs):
        submitted.append(len(items))
        return real_map(fn, items, **kwargs)

    with mock.patch.object(cluster_client, 'map', counting_map):
        da._display_displacement_diagnostics(
            field, ids, coords, np.array([1.0, 1.0, 1.0]),
            cluster_client, context='swept field')

    assert submitted == [len(ids)]


def test_a_failing_region_is_reported_without_sinking_the_sweep(cluster_client):
    """A diagnostic is never worth failing a run over."""
    shape = (16, 16, 16)
    field = np.zeros(shape + (3,), dtype=np.float32)
    ids, coords = da._tile_volume(shape, (8, 8, 8))

    boom = mock.Mock(side_effect=ValueError('no jacobian for you'))
    with mock.patch.object(da, 'deform_field_diagnostics', boom):
        with _recording_logs(logging.INFO) as recorder:
            da._display_displacement_diagnostics(
                field, ids, coords, np.array([1.0, 1.0, 1.0]),
                cluster_client, context='doomed field')

    errors = [r for r in recorder.records if r.levelno == logging.ERROR]
    assert len(errors) == len(ids)
    assert 'no jacobian for you' in errors[0].getMessage()


def test_diagnostics_mode_accepts_the_enum_and_any_case():
    assert _parse_displacement_diagnostics(None) is None
    for value in ('PER_STEP', 'per_step', ' Per_Step ',
                  DisplacementDiagnostics.PER_STEP):
        assert (_parse_displacement_diagnostics(value)
                is DisplacementDiagnostics.PER_STEP)


def test_unknown_diagnostics_mode_is_refused():
    with pytest.raises(ValueError, match='PER_STEP'):
        _parse_displacement_diagnostics('EVERY_BLOCK')
