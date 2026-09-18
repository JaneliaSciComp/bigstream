"""
Tests for the multi-pass blockwise alignment pipeline.

Historical note: this file used to carry
`test_single_pass_matches_distributed_alignment`, which asserted that at one
pass, a zero lattice offset and no neighbour clamp the pipeline reproduced
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

import numpy as np
import pytest
import yaml

import bigstream.distributed_align as da

from bigstream.align_constraints import (blend_safe_displacement_bound,
                                         neighbor_disagreement_bound)
from bigstream.distutils import validate_processing_block_size
from bigstream.distributed_align import (
    AlignmentPass,
    DisplacementDiagnostics,
    blockwise_alignment_pipeline,
    _BlockLattice,
    _NeighborhoodEstimate,
    alignment_passes_from_config,
    alignment_steps_from_config,
    _axis_block_starts,
    _block_summary,
    _clamp_block_to_neighbors,
    _compose_fields_block,
    default_pass_geometry_from_config,
    _get_transform_weights,
    _is_abstaining_block,
    _neighborhood_estimate,
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
    size, halo_factor, _ = default_pass_geometry_from_config(config)
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


@pytest.mark.parametrize('offset', [(0, 0, 0), (4, 4, 4), (5, 3, 7)])
def test_blending_weights_are_a_partition_of_unity(offset):
    """
    Every voxel must receive a total weight of exactly 1, including at the
    volume faces where the missing off-volume neighbour's share is rebalanced
    into the blocks that remain. Without this the stitched field is silently
    scaled down (or up) wherever the sum is off.
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
        output_transform=output,
        pass_output_factory=factory,
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
        output_transform=output,
        pass_output_factory=lambda i, s: np.zeros(s, dtype=np.float32),
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
            output_transform=output,
            pass_output_factory=lambda i, s: np.zeros(s, dtype=np.float32),
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
# phase 5 - abstention and the neighbour consistency clamp
# --------------------------------------------------------------------------


def test_delta_bound_is_twice_the_displacement_bound():
    """
    The two bounds must stay in lockstep: the amplitude ceiling is derived by
    substituting the worst case `delta <= 2U` into the disagreement bound, so
    if they ever drift apart one of them is wrong.
    """
    halo = np.array([8, 8, 8])
    spacing = np.array([1.0, 1.0, 1.0])
    for k in (0.05, 0.1, 0.32):
        assert neighbor_disagreement_bound(halo, spacing, k) == pytest.approx(
            2.0 * blend_safe_displacement_bound(halo, spacing, k))


def test_zero_field_is_read_as_an_abstention():
    assert _is_abstaining_block(np.zeros((4, 4, 4, 3), dtype=np.float32))
    field = np.zeros((4, 4, 4, 3), dtype=np.float32)
    field[0, 0, 0, 0] = 1e-6
    assert not _is_abstaining_block(field)


def test_block_summary_averages_the_core_only():
    field = np.zeros((6, 6, 6, 3), dtype=np.float32)
    core = (slice(2, 4),) * 3
    field[core] = 5.0
    field[0] = 100.0  # halo territory another block owns
    index, mean, confidence = _block_summary(((1, 1, 1), None, None, field),
                                             core)
    assert index == (1, 1, 1)
    assert confidence == 1.0
    np.testing.assert_allclose(mean, [5.0, 5.0, 5.0])


def test_abstaining_block_gets_zero_confidence():
    field = np.zeros((6, 6, 6, 3), dtype=np.float32)
    _, mean, confidence = _block_summary(((0, 0, 0), None, None, field),
                                         (slice(None),) * 3)
    assert confidence == 0.0
    np.testing.assert_allclose(mean, np.zeros(3))


def test_estimate_inpaints_a_zero_confidence_node():
    """
    A failed block must not drag the estimate toward zero. Normalized
    convolution divides out the confidence, so a hole is filled from its
    neighbours rather than averaged with a zero that nobody measured.
    """
    lattice = _BlockLattice((16, 16, 16), (4, 4, 4), (0, 0, 0), (48, 48, 48))
    summaries = {}
    for index in lattice.indices():
        confident = index != (1, 1, 1)
        summaries[index] = (np.array([3.0, 0.0, 0.0]) if confident
                            else np.zeros(3), 1.0 if confident else 0.0)
    estimate = _neighborhood_estimate(lattice, summaries, sigma=1.0)
    # the hole picks up its neighbours' value, not zero
    np.testing.assert_allclose(estimate.nodes[1, 1, 1], [3.0, 0.0, 0.0],
                               rtol=1e-6)


def test_clamp_bounds_the_deviation_from_the_estimate():
    lattice = _BlockLattice((8, 8, 8), (2, 2, 2), (0, 0, 0), (16, 16, 16))
    summaries = {index: (np.zeros(3), 1.0) for index in lattice.indices()}
    estimate = _neighborhood_estimate(lattice, summaries)

    coords = (slice(0, 10), slice(0, 10), slice(0, 10))
    field = np.full((10, 10, 10, 3), 12.0, dtype=np.float32)
    _, _, _, clamped = _clamp_block_to_neighbors(
        ((0, 0, 0), coords, {}, field), estimate=estimate, delta_max=4.0)
    # estimate is zero everywhere, so the field may reach delta_max / 2
    np.testing.assert_allclose(clamped, 2.0)


def test_abstaining_block_adopts_the_estimate_instead_of_zero():
    lattice = _BlockLattice((8, 8, 8), (2, 2, 2), (0, 0, 0), (16, 16, 16))
    summaries = {index: (np.array([7.0, 7.0, 7.0]), 1.0)
                 for index in lattice.indices()}
    estimate = _neighborhood_estimate(lattice, summaries)

    coords = (slice(0, 10), slice(0, 10), slice(0, 10))
    field = np.zeros((10, 10, 10, 3), dtype=np.float32)
    _, _, _, filled = _clamp_block_to_neighbors(
        ((0, 0, 0), coords, {}, field), estimate=estimate, delta_max=4.0)
    np.testing.assert_allclose(filled, 7.0, rtol=1e-5)


def test_estimate_interpolation_is_exact_at_the_nodes():
    lattice = _BlockLattice((8, 8, 8), (2, 2, 2), (0, 0, 0), (24, 24, 24))
    nodes = np.zeros(lattice.nblocks + (3,))
    nodes[..., 0] = np.arange(np.prod(lattice.nblocks)).reshape(lattice.nblocks)
    origin, step = lattice.node_geometry()
    estimate = _NeighborhoodEstimate(nodes=nodes,
                                    confidence=np.ones(lattice.nblocks),
                                    node_origin=origin, node_step=step)
    # node (1, 1, 1) sits at voxel 8 + 3.5 = 11.5 -> sample the two voxels
    # either side of it and check they bracket the node value
    sampled = estimate.interpolate((slice(11, 13),) * 3)
    assert sampled[..., 0].mean() == pytest.approx(nodes[1, 1, 1, 0], abs=1e-5)


def test_clamped_blocks_agree_within_delta_max():
    """
    The guarantee the whole phase exists for: after clamping, no two
    overlapping blocks may disagree by more than `delta_max`, however wildly
    they disagreed before. That holds because every block is clamped to
    within `delta_max/2` of the *same* global estimate, so the triangle
    inequality closes - which is why the estimate has to be one field rather
    than a per-block scalar.
    """
    rng = np.random.default_rng(7)
    lattice = _BlockLattice((16, 16, 16), (4, 4, 4), (0, 0, 0), (48, 48, 48))
    delta_max = 1.0

    blocks = {}
    for index in lattice.indices():
        coords = lattice.footprint_slices(index)
        shape = tuple(s.stop - s.start for s in coords)
        # wildly disagreeing blocks: each one a different large constant
        value = rng.uniform(-50.0, 50.0, size=3).astype(np.float32)
        blocks[index] = (index, coords, {},
                         np.broadcast_to(value, shape + (3,)).copy())

    summaries = dict()
    for index, block in blocks.items():
        _, mean, confidence = _block_summary(
            block, lattice.core_within_footprint(index))
        summaries[index] = (mean, confidence)
    estimate = _neighborhood_estimate(lattice, summaries, sigma=1.0)

    clamped = {index: _clamp_block_to_neighbors(block, estimate=estimate,
                                                delta_max=delta_max)
               for index, block in blocks.items()}

    worst = 0.0
    indices = sorted(clamped)
    for a in indices:
        for b in indices:
            if a >= b or any(abs(i - j) > 1 for i, j in zip(a, b)):
                continue
            _, coords_a, _, field_a = clamped[a]
            _, coords_b, _, field_b = clamped[b]
            overlap = tuple(slice(max(x.start, y.start), min(x.stop, y.stop))
                            for x, y in zip(coords_a, coords_b))
            if any(s.stop <= s.start for s in overlap):
                continue
            sub_a = field_a[tuple(slice(s.start - c.start, s.stop - c.start)
                                  for s, c in zip(overlap, coords_a))]
            sub_b = field_b[tuple(slice(s.start - c.start, s.stop - c.start)
                                  for s, c in zip(overlap, coords_b))]
            worst = max(worst, float(np.abs(sub_a - sub_b).max()))

    assert worst > 0.0            # the blocks really do still disagree
    assert worst <= delta_max + 1e-5


def test_clamp_smooths_the_assembled_field(cluster_client, synthetic_volumes,
                                           monkeypatch):
    """
    End to end through the cluster: the largest voxel-to-voxel jump in the
    assembled field *is* delta (the spec's `max jump dx/dy/dz` diagnostic), so
    turning the clamp on has to bring it down.
    """
    _patch_alignment_pipeline(monkeypatch, _fake_alignment_pipeline)
    fix_image, mov_image = synthetic_volumes
    shape = tuple(fix_image.spatial_dims)
    spacing = np.array([1.0, 1.0, 1.0])

    def run(neighbor_consistency):
        output = np.zeros(shape + (3,), dtype=np.float32)
        assert blockwise_alignment_pipeline(
            fix_image, spacing, mov_image, spacing,
            [AlignmentPass(alignment_steps=[('deform', {})])], cluster_client,
            processing_size=(16, 16, 16),
            processing_halo_factor=0.25,
            neighbor_consistency=neighbor_consistency,
            output_transform=output,
        )
        return output

    def max_jump(field):
        return max(float(np.abs(np.diff(field, axis=a)).max())
                   for a in range(3))

    unclamped = run(None)
    clamped = run({'delta_max': 0.05})
    assert np.any(unclamped)
    assert max_jump(clamped) < max_jump(unclamped)


def test_auto_delta_max_comes_from_the_lattice(cluster_client,
                                               synthetic_volumes,
                                               monkeypatch):
    """`delta_max: auto` must resolve, run, and not blow up on a real pass."""
    _patch_alignment_pipeline(monkeypatch, _fake_alignment_pipeline)
    fix_image, mov_image = synthetic_volumes
    shape = tuple(fix_image.spatial_dims)
    spacing = np.array([1.0, 1.0, 1.0])
    output = np.zeros(shape + (3,), dtype=np.float32)
    assert blockwise_alignment_pipeline(
        fix_image, spacing, mov_image, spacing,
        [AlignmentPass(alignment_steps=[('deform', {})])], cluster_client,
        processing_size=(16, 16, 16),
        processing_halo_factor=0.25,
        neighbor_consistency={'delta_max': 'auto', 'k': 0.1},
        output_transform=output,
    )
    assert np.all(np.isfinite(output))


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
        output_transform=output,
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
            roi=roi, output_transform=output,
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
        output_transform=reference, **kwargs)

    field = _zarr_field(tmp_path, 'out', shape + (3,), (10, 10, 10, 3),
                        shards=(20, 20, 20, 3))
    assert list(_storage_write_unit(field, 3)) == [20, 20, 20]
    assert blockwise_alignment_pipeline(
        fix_image, spacing, mov_image, spacing, passes, cluster_client_mt,
        output_transform=field, **kwargs)

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
        output_transform=reference, **kwargs)

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
        output_transform=field, max_write_locks=max_locks, **kwargs)

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

    def emit(self, record):
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
            output_transform=np.zeros(shape + (3,), dtype=np.float32),
            displacement_diagnostics=mode,
            pass_output_factory=lambda i, sh: np.zeros(sh, dtype=np.float32),
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


def test_diagnostics_mode_accepts_the_enum_and_any_case():
    assert _parse_displacement_diagnostics(None) is None
    for value in ('PER_STEP', 'per_step', ' Per_Step ',
                  DisplacementDiagnostics.PER_STEP):
        assert (_parse_displacement_diagnostics(value)
                is DisplacementDiagnostics.PER_STEP)


def test_unknown_diagnostics_mode_is_refused():
    with pytest.raises(ValueError, match='PER_STEP'):
        _parse_displacement_diagnostics('EVERY_BLOCK')
