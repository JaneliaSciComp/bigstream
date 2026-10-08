"""
Multi-pass blockwise alignment.

Implements multi-pass distributed blockwise alignment.

  * several **passes** over the volume, each pass fitting only the residual
    left by its predecessors (cascading through `static_transform_list`),
  * a per-pass **lattice offset**, so pass 2's block seams land in pass 1's
    block interiors,
  * a per-pass **halo**, so a pass that fits a smaller residual can reach less
    far.

methods implemented in this module:

    blockwise_alignment_pipeline      run the passes
    AlignmentPass                     one pass; build it directly, or
    alignment_passes_from_config        read a list out of a config,
    default_pass_geometry_from_config   with the defaults those passes
                                        inherit, and
    alignment_steps_from_config         the steps-list converter the two
                                        above use
    MAX_WRITE_LOCKS                   default write lock budget,
                                      re-exported from
                                      `bigstream.distributed_io_utility`,
                                      which holds the write locking this
                                      module shares with the transform paths


One piece of the per-block machinery is worth knowing about because it is
shaped by the multi-pass design rather than by the block loop:

  * `_get_transform_weights` takes explicit `clip_before` / `clip_after`
    amounts rather than deriving the crop from the block's index. An index
    derived crop is only right for a lattice anchored at voxel 0; a lattice
    with a nonzero phase offset has partial blocks at *both* ends.

Notations

    u        displacement field. `u(x)` is how far the voxel at `x` moves,
             in the same physical units as the voxel spacing.
    U        max displacement in physical units
    w        per-block blending weight: 1 on a block's core, ramping to 0 at
             its footprint edge. "How much does this block get to vote here."
             Sum over blocks is 1 at every voxel.
    L        blend ramp length, `(2*halo - 1) * spacing` - the distance over
             which one block hands off to the next.
    delta    *disagreement*: `|u_A - u_B|` where blocks A and B overlap.
             "How much do neighbours contradict each other."
    B        block step: the spacing between block origins, i.e.
             `processing_size` in voxels. Not the footprint, which is
             `B + 2*halo`.
    P        number of passes.
"""

# # the entry point is deliberately the first thing in this file, so its
# # signature names types that are defined further down
# from __future__ import annotations

import logging
import time
import traceback

from dataclasses import dataclass, field
from enum import Enum
from itertools import product
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np
import zarr

from dask.distributed import as_completed

import bigstream.transform as bst
import bigstream.utility as ut

from .align import alignment_pipeline, _phys_roi_to_voxel
from .align_constraints import DEFAULT_K, blend_safe_displacement_bound
from .blend_ramp import (DEFAULT_BLEND_RAMP, blend_ramp_weights,
                         parse_blend_ramp, ramp_gradient_gain)
from .diagnostics import (deform_field_diagnostics,
                          log_deform_field_diagnostics)
from .distributed_io_utility import (MAX_WRITE_LOCKS,
                                     lock_cell_keys,
                                     log_write_locking,
                                     storage_write_unit,
                                     write_lock,
                                     write_lock_grid,
                                     write_lock_namespace)
from .distutils import ThrottledArraySliceReader
from .image_data import (ImageData, as_image_data)
from .transform import apply_transform_to_coordinates


logger = logging.getLogger(__name__)


@dataclass
class AlignmentPass:
    """
    One pass over the volume: a block lattice plus the steps run on each block.

    Every geometric field is in **voxels, zyx order**, matching the rest of
    the bigstream config (the xyz-ordered CLI values are reversed long before
    they reach here).

    alignment_steps : list of (str, dict)
        What to run on each block, in `alignment_pipeline` form.

    processing_size : tuple of int (default: None)
        `B`, the block step. None means "inherit the top level default".

    processing_offset : tuple of int (default: None)
        The lattice phase. Block origins sit at `offset + n*B` measured from
        **voxel 0 of the volume**, not from the ROI - see
        `_BlockLattice`. None means 0.

        A later pass wants an offset that puts its block seams where the
        previous pass had block interiors; `B/4` is a reasonable stagger.

    processing_halo_factor : float or tuple of float (default: None)
        Halo as a fraction of `processing_size`, **per side**. This is the
        same quantity the old single-pass pipeline called `overlap_factor`;
        the name differs only to make the per-side reading explicit. Ignored
        when `processing_halo` is given.

    processing_halo : tuple of int (default: None)
        Halo in voxels, per side. Takes precedence over the factor.

    blend_ramp : str (default: None)
        Shape of the overlap-add blending weight ramp, `'linear'` or
        `'cosine'`. None means "inherit the top level default", which is
        itself `'linear'` when unset. See `bigstream.blend_ramp`: a cosine
        ramp has no kink where it meets the block core but is `pi/2` steeper
        mid-ramp, which tightens the fold-safe displacement ceiling by the
        same factor.

    max_displacement, bspline_constraints
        Per-pass overrides of the blend-safety budget - see
        `blockwise_alignment_pipeline`. Both are only used for the
        *reporting* in `_check_blend_safe_displacement`; the values that
        actually constrain the fit live in the individual steps' arguments.
    """

    alignment_steps: List[Tuple[str, dict]] = field(default_factory=list)
    processing_size: Optional[Tuple[int, ...]] = None
    processing_offset: Optional[Tuple[int, ...]] = None
    processing_halo_factor: Optional[Tuple[float, ...]|float] = None
    processing_halo: Optional[Tuple[int, ...]] = None
    blend_ramp: Optional[str] = None
    name: Optional[str] = None

    def resolved(self, ndim,
                 default_processing_size=None,
                 default_halo_factor=None,
                 default_blend_ramp=None):
        """
        A copy with every geometric field filled in as a length-`ndim` tuple.

        Raises if a pass ends up with no block size, because there is no
        sensible fallback for it.
        """
        size = _as_int_tuple(
            _first_not_none(self.processing_size, default_processing_size),
            ndim, 'processing_size')
        if size is None:
            raise ValueError(
                'no processing_size for alignment pass '
                f'{self.name or "(unnamed)"}: set it on the pass or as the '
                'top level local_align processing_size'
            )
        if any(s <= 0 for s in size):
            raise ValueError(f'processing_size must be positive, got {size}')

        offset = _as_int_tuple(self.processing_offset, ndim,
                               'processing_offset')
        offset = (0,) * ndim if offset is None else offset

        halo = _as_int_tuple(self.processing_halo, ndim, 'processing_halo')
        if halo is None:
            factor = _first_not_none(self.processing_halo_factor,
                                     default_halo_factor)
            if factor is None:
                raise ValueError(
                    'no processing_halo or processing_halo_factor for '
                    f'alignment pass {self.name or "(unnamed)"}'
                )
            factor = np.broadcast_to(
                np.atleast_1d(np.asarray(factor, dtype=float)), (ndim,))
            if np.any(factor <= 0) or np.any(factor >= 1):
                raise ValueError(
                    'processing_halo_factor is a per-side fraction of the '
                    f'block size and must be in (0, 1), got {factor.tolist()}'
                )
            halo = tuple(int(h) for h in np.round(np.array(size) * factor))
        if any(h <= 0 for h in halo):
            raise ValueError(
                f'processing_halo must be positive, got {halo}; a zero halo '
                'leaves no ramp to blend over'
            )

        # raises on an unknown name; a typo here would otherwise silently
        # blend with one ramp and bound the displacement with another
        ramp = parse_blend_ramp(
            _first_not_none(self.blend_ramp, default_blend_ramp))

        return AlignmentPass(
            alignment_steps=list(self.alignment_steps),
            processing_size=size,
            processing_offset=offset,
            processing_halo_factor=self.processing_halo_factor,
            processing_halo=halo,
            blend_ramp=ramp,
            name=self.name,
        )


class DisplacementDiagnostics(str, Enum):
    """
    When to run `deform_field_diagnostics` over a whole assembled field.

    A "step" here is an alignment pass. The per-block diagnostics are a
    separate thing and are always emitted at DEBUG, because a block's field
    is not the field that reaches disk - its overlap regions only reach
    their final values once every neighbour has added its weighted share.

        (unset)     no whole-field diagnostics at all
        PER_STEP    after every pass, and on the composed result
        FINAL_STEP  on the final field only - the composed result, or the
                    single pass's field when there is only one

    Running these is not free: each one walks the assembled field block by
    block and computes a jacobian determinant per block.
    """

    PER_STEP = 'PER_STEP'
    FINAL_STEP = 'FINAL_STEP'


def _parse_displacement_diagnostics(value):
    """`None`, a `DisplacementDiagnostics`, or its name (any case)."""
    if value is None or value is False:
        return None
    if isinstance(value, DisplacementDiagnostics):
        return value
    try:
        return DisplacementDiagnostics(str(value).strip().upper())
    except ValueError:
        raise ValueError(
            f'unsupported displacement_diagnostics {value!r}; expected None '
            f'or one of {[m.value for m in DisplacementDiagnostics]}'
        ) from None

# -------------------------------------------------------------------------
# the entry point
# -------------------------------------------------------------------------


def blockwise_alignment_pipeline(
    fix_image: ImageData,
    fix_spatial_spacing,
    mov_image: ImageData,
    mov_spatial_spacing,
    alignment_passes: Sequence[AlignmentPass],
    cluster_client,
    processing_size=None,
    processing_halo_factor=None,
    blend_ramp=None,
    fix_mask=None,
    mov_mask=None,
    roi: Optional[Sequence[float]] = None,
    foreground_percentage=0.,
    mov_origin_transform: Optional[np.ndarray] = None,
    static_transform_list: Sequence[np.ndarray | zarr.Array] = (),
    deformfield_final_result=None,
    deformfield_output_factory: Optional[Callable[[int, tuple], Any]] = None,
    max_concurrent_reads=0,
    max_cluster_jobs=0,
    max_write_locks=MAX_WRITE_LOCKS,
    rebalance_for_missing_neighbors=True,
    displacement_diagnostics=None,
    error_if_displacement_check_fails=False,
    start_pass=1,
    resumed_pass_fields: Sequence[np.ndarray | zarr.Array] = (),
):
    """
    Multi-pass piecewise alignment of a moving image to a fixed image.

    Each pass partitions the volume on its own globally anchored block
    lattice, aligns every block in parallel, and overlap-adds the weighted
    per-block fields into one displacement field for that pass. Pass `n+1`
    then runs with passes `1..n` prepended to its `static_transform_list`, so
    it only ever fits the **residual** they left. The passes are composed at
    the end.

    Why more than one pass helps: the amplitude a single pass can safely carry
    is limited by how much neighbouring blocks may disagree without folding
    the stitched field. Splitting the deformation over `P` passes gives a
    total capacity of the *sum* of the per-pass budgets rather than one
    budget, and staggering each pass's lattice (`processing_offset`) puts one
    pass's seams in the next pass's block interiors, so no seam is ever
    reinforced.

    Parameters
    ----------
    fix_image, mov_image : ImageData
        The fixed and moving images.

    fix_spatial_spacing, mov_spatial_spacing : 1d array
        Physical voxel spacing, zyx, expansion corrected.

    alignment_passes : sequence of AlignmentPass
        The passes to run, in order. See `alignment_passes_from_config`.

    cluster_client : dask distributed Client
        Must already exist.

    processing_size : tuple of int (default: None)
        Default block step (`B`) for passes that do not set their own, zyx
        voxels.

    processing_halo_factor : float or tuple (default: None)
        Default per-side halo fraction for passes that do not set their own.
        Same meaning as the old single-pass pipeline's `overlap_factor`.

    blend_ramp : str (default: None)
        Default blending ramp shape for passes that do not set their own,
        `'linear'` or `'cosine'`. None is `'linear'`, which reproduces every
        field computed before this was configurable, bit for bit. See
        `bigstream.blend_ramp` for what the choice costs.

    roi : sequence of float (default: None)
        Registration ROI on the fixed image, physical coordinates, zyx
        `(zmin, ymin, xmin[, zmax, ymax, xmax])`. Selects which blocks run;
        selected blocks are not cropped, and block *coordinates* stay
        anchored to the volume origin regardless of the ROI.

    static_transform_list : sequence (default: ())
        Transforms already applied to the moving image, e.g. a global affine.

    output_transform : ndarray or zarr.Array (default: None)
        Where the final (composed) field is written.

    pass_output_factory : callable (default: None)
        `factory(pass_index, shape) -> array` producing the per-pass field
        container. Only consulted when there is more than one pass; without
        it, intermediate fields are allocated in process memory, which is
        fine for tests and not for a real volume.

    max_write_locks : int (default: 64)
        Largest number of lock names one write may hold.

        Blocks are written under a lock keyed by *what the store rewrites*
        - its **write unit**, a chunk or, on a sharded v3 array, a shard -
        not by the block. Two blocks landing in the same write unit clobber
        each other even with no voxel in common. Nothing therefore requires
        `processing_size` to be a multiple of the chunk or shard; blocks
        that share a write unit are simply serialized against each other.

        A large block over a fine chunk grid touches hundreds of write
        units. Past this many, they are grouped into larger **lock cells**
        - always a whole number of write units, so the guarantee holds - and
        the write locks those instead. Raising it buys parallelism when
        blocks are much bigger than the write unit, at the cost of a wider
        `MultiLock`.

        An in-memory output has no write unit, so it is locked per
        overlapping block instead and this has no effect.

    displacement_diagnostics : DisplacementDiagnostics or str (default: None)
        When to sweep a whole assembled field for jacobian/folding
        statistics - `None`, `'PER_STEP'` or `'FINAL_STEP'`; see
        `DisplacementDiagnostics`. Per-block diagnostics are independent of
        this and always go to DEBUG.

    start_pass : int (default: 1)
        1-indexed pass to resume from. Passes before it are not run; their
        fields must already exist and be supplied via `resumed_pass_fields`,
        so cascading (`static_transform_list`) and the final composition see
        the same fields a full run would have produced. Values `<= 1` run
        every pass, as if not given. A value greater than the number of
        configured passes is out of range and is ignored the same way,
        since there would be nothing left to resume.

    resumed_pass_fields : sequence of ndarray or zarr.Array (default: ())
        The already-computed fields for passes `1 .. start_pass - 1`, in
        order. Required (and must match that length) whenever `start_pass`
        selects a pass beyond the first; ignored otherwise.

    Other parameters behave as they did in the single-pass pipeline this
    replaces.

    Returns
    -------
    bool - True when every block of every pass completed.
    """
    fix_spatial_dims = tuple(fix_image.spatial_dims)
    mov_spatial_dims = tuple(mov_image.spatial_dims)
    spatial_ndim = fix_image.spatial_ndim
    fix_spatial_spacing = np.asarray(fix_spatial_spacing, dtype=np.float64)

    if not alignment_passes:
        raise ValueError('no alignment passes to run')

    resolved_passes = [
        p.resolved(spatial_ndim,
                   default_processing_size=processing_size,
                   default_halo_factor=processing_halo_factor,
                   default_blend_ramp=blend_ramp)
        for p in alignment_passes
    ]

    if roi is not None:
        roi_start, roi_stop = _phys_roi_to_voxel(roi, fix_spatial_dims,
                                                 fix_spatial_spacing)
        logger.info((
            f'ROI {roi} with voxel spacing {fix_spatial_spacing} '
            'selects blocks intersecting voxels '
            f'{roi_start}:{roi_stop}'
        ))
    else:
        roi_start = roi_stop = None

    fix_mask_image = as_image_data(fix_mask) if fix_mask is not None else None
    mov_mask_image = as_image_data(mov_mask) if mov_mask is not None else None

    npasses = len(resolved_passes)
    diagnostics = _parse_displacement_diagnostics(displacement_diagnostics)
    # a single pass writes straight to the output, so that pass *is* the
    # final step and there is no composed field to report on afterwards
    per_pass_diagnostics = (
        diagnostics is DisplacementDiagnostics.PER_STEP
        or (diagnostics is DisplacementDiagnostics.FINAL_STEP and npasses == 1)
    )

    if start_pass > npasses:
        logger.warning(
            f'start_pass={start_pass} is beyond the {npasses} configured '
            'pass(es); ignoring it and running from pass 1'
        )
        start_pass = 1
    start_pass = max(start_pass, 1)

    if start_pass > 1:
        if len(resumed_pass_fields) != start_pass - 1:
            raise ValueError(
                f'start_pass={start_pass} requires {start_pass - 1} '
                f'resumed_pass_fields (one per earlier pass), got '
                f'{len(resumed_pass_fields)}'
            )
        logger.info(
            f'Resuming from pass {start_pass}/{npasses}: reusing '
            f'{len(resumed_pass_fields)} already-computed pass field(s), '
            f'passes 1..{start_pass - 1} will not be recomputed'
        )
        pass_fields = list(resumed_pass_fields)
    else:
        pass_fields = []
    result = True

    for pass_index, alignment_pass in enumerate(resolved_passes):
        if pass_index < start_pass - 1:
            continue
        label = alignment_pass.name or f'pass{pass_index + 1}'
        if npasses == 1:
            pass_output = deformfield_final_result
        else:
            pass_output = _make_deformfield_output(
                pass_index, fix_spatial_dims + (spatial_ndim,),
                deformfield_output_factory,
            )
        logger.info((
            f'--- {label} ({pass_index + 1}/{npasses}): block size '
            f'{alignment_pass.processing_size}, offset '
            f'{alignment_pass.processing_offset}, halo '
            f'{alignment_pass.processing_halo}, {alignment_pass.blend_ramp} '
            'blend ramp, '
            f'{len(static_transform_list) + len(pass_fields)} static '
            f'transforms, steps {[s[0] for s in alignment_pass.alignment_steps]}'
        ))

        pass_start = time.time()
        pass_ok = _run_alignment_pass(
            alignment_pass,
            fix_image,
            fix_spatial_dims,
            fix_spatial_spacing,
            mov_image,
            mov_spatial_dims,
            mov_spatial_spacing,
            cluster_client,
            fix_mask_image=fix_mask_image,
            mov_mask_image=mov_mask_image,
            roi_start=roi_start,
            roi_stop=roi_stop,
            foreground_percentage=foreground_percentage,
            mov_origin_transform=mov_origin_transform,
            static_transform_list=list(static_transform_list) + pass_fields,
            output_transform=pass_output,
            max_concurrent_reads=max_concurrent_reads,
            max_cluster_jobs=max_cluster_jobs,
            max_write_locks=max_write_locks,
            rebalance_for_missing_neighbors=rebalance_for_missing_neighbors,
            run_displacement_diagnostics=per_pass_diagnostics,
            error_if_displacement_check_fails=error_if_displacement_check_fails,
            label=label,
        )
        # a failed pass does not stop the run - later passes still have
        # something to cascade from - so say so at a level that survives a
        # log read, and say which pass it was
        logger.log(
            logging.INFO if pass_ok else logging.ERROR,
            f'--- {label} ({pass_index + 1}/{npasses}) '
            f'{"completed" if pass_ok else "FAILED"} in '
            f'{time.time() - pass_start:.1f}s'
            + ('' if pass_ok else ' - some blocks did not align; its field '
                                  'is incomplete and every later pass '
                                  'cascades from it')
        )

        result = result and pass_ok
        if pass_output is not None:
            pass_fields.append(pass_output)

    if npasses > 1 and deformfield_final_result is not None and pass_fields:
        logger.info(f'Compose {len(pass_fields)} pass fields into the output')
        composed_ok = _distributed_compose_displacement_fields(
            pass_fields, fix_spatial_spacing, deformfield_final_result, cluster_client,
            max_write_locks=max_write_locks,
        )
        result = result and composed_ok
        if composed_ok and diagnostics is not None:
            # the passes compose, so det J_total = product of the per pass
            # det J. That product is far too pessimistic to design to, which
            # is exactly why it is measured on the composed field instead.
            # Walked block by block - the whole field does not fit in the
            # client's memory at production sizes.
            last = resolved_passes[-1]
            tile = (np.array(last.processing_size)
                    + 2 * np.array(last.processing_halo))
            ids, coords = _tile_volume(fix_spatial_dims, tile)
            _display_displacement_diagnostics(
                deformfield_final_result, ids, coords, fix_spatial_spacing,
                cluster_client, context='composed multi-pass field')

    logger.info(f'Blockwise alignment completed (ok={result})')
    return result


def _make_deformfield_output(pass_index, shape, factory):
    if factory is not None:
        return factory(pass_index, shape)
    logger.warning((
        f'No pass_output_factory given; allocating the pass {pass_index + 1} '
        f'field {shape} in process memory. Supply a factory backed by zarr '
        'for anything larger than a test volume.'
    ))
    return np.zeros(shape, dtype=np.float32)


def _run_alignment_pass(alignment_pass,
                        fix_image,
                        fix_spatial_dims,
                        fix_spatial_spacing,
                        mov_image,
                        mov_spatial_dims,
                        mov_spatial_spacing,
                        cluster_client,
                        fix_mask_image=None,
                        mov_mask_image=None,
                        roi_start=None,
                        roi_stop=None,
                        foreground_percentage=0.,
                        mov_origin_transform=None,
                        static_transform_list=(),
                        output_transform=None,
                        max_concurrent_reads=0,
                        max_cluster_jobs=0,
                        rebalance_for_missing_neighbors=True,
                        run_displacement_diagnostics=False,
                        error_if_displacement_check_fails=False,
                        max_write_locks=MAX_WRITE_LOCKS,
                        label='pass'):
    """Run one pass of the multi-pass pipeline. See `blockwise_alignment_pipeline`."""
    block_size = np.array(alignment_pass.processing_size)
    halo = np.array(alignment_pass.processing_halo)

    lattice = _BlockLattice(block_size=alignment_pass.processing_size,
                           halo=alignment_pass.processing_halo,
                           offset=alignment_pass.processing_offset,
                           extent=fix_spatial_dims)
    nblocks = lattice.nblocks
    logger.info((
        f'{label}: partition {fix_spatial_dims} into {nblocks} blocks of '
        f'{tuple(block_size)} with a {tuple(halo)} halo at lattice offset '
        f'{lattice.offset}'
    ))

    steps = [(name, args) for name, args in alignment_pass.alignment_steps]
    _check_blend_safe_displacement(
        steps, block_size, halo, fix_spatial_spacing,
        blend_ramp=alignment_pass.blend_ramp,
        error_when_check_fails=error_if_displacement_check_fails)

    # Decided once, here, and handed to every block: what a writer has to
    # lock depends on the output's write unit, and all writers must agree
    # on the cell grid or two of them can name different locks for one
    # chunk.
    # Nothing requires the block footprint to be a multiple of that unit -
    # blocks that land in a shared chunk are simply serialized.
    footprint = block_size + 2 * halo
    lock_grid = write_lock_grid(output_transform, footprint,
                                max_locks=max_write_locks)
    lock_namespace = write_lock_namespace(output_transform, label)
    if output_transform is not None:
        log_write_locking(output_transform, lock_grid, footprint, label)

    blocks_ids, blocks_coords = _select_blocks(
        lattice,
        fix_spatial_dims=fix_spatial_dims,
        fix_mask_image=fix_mask_image,
        foreground_percentage=foreground_percentage,
        roi_start=roi_start,
        roi_stop=roi_stop,
        max_concurrent_reads=max_concurrent_reads,
        label=label,
    )
    if not blocks_ids:
        logger.warning(f'{label}: no blocks to align')
        return True

    selected = set(blocks_ids)
    neighbor_offsets = list(product((-1, 0, 1), repeat=lattice.ndim))
    blocks_neighbors = [
        {o: tuple(a + b for a, b in zip(index, o)) in selected
         for o in neighbor_offsets}
        for index in blocks_ids
    ]
    blocks_infos = list(zip(blocks_ids, blocks_coords, blocks_neighbors))

    fix_mask_spatial_dims = (fix_mask_image.spatial_dims
                             if fix_mask_image is not None else None)
    mov_mask_spatial_dims = (mov_mask_image.spatial_dims
                             if mov_mask_image is not None else None)

    def align_method(block_info):
        prepared = _prepare_compute_block_spatial_transform_params(
            block_info,
            fix_shape=fix_spatial_dims,
            mov_shape=mov_spatial_dims,
            fix_spacing=fix_spatial_spacing,
            mov_spacing=mov_spatial_spacing,
            fix_fullmask_shape=fix_mask_spatial_dims,
            mov_fullmask_shape=mov_mask_spatial_dims,
            mov_origin_transform=mov_origin_transform,
            static_transform_list=list(static_transform_list),
            pass_label=label,
        )
        read = _read_blocks_for_processing(
            prepared,
            fix=fix_image,
            mov=mov_image,
            fix_mask=fix_mask_image,
            mov_mask=mov_mask_image,
            fix_block_reader=ThrottledArraySliceReader(max_concurrent_reads),
            mov_block_reader=ThrottledArraySliceReader(max_concurrent_reads),
            pass_label=label,
        )
        return _align_block(read,
                            fix_spacing=fix_spatial_spacing,
                            mov_spacing=mov_spatial_spacing,
                            align_steps=steps,
                            pass_label=label)

    def write_method(aligned_block):
        block_index = aligned_block[0]
        clip_before, clip_after = lattice.clip_amounts(block_index)
        return _write_block_transform(
            aligned_block,
            block_size=block_size,
            block_overlaps=halo,
            nblocks=nblocks,
            output_transform=output_transform,
            rebalance_for_missing_neighbors=rebalance_for_missing_neighbors,
            blend_ramp=alignment_pass.blend_ramp,
            clip_before=clip_before,
            clip_after=clip_after,
            lock_grid=lock_grid,
            lock_namespace=lock_namespace,
            pass_label=label,
        )

    def align_and_write_method(block_info):
        return write_method(align_method(block_info))

    partitions = _partition(blocks_infos, max_cluster_jobs)

    result = True
    for part_index, part in enumerate(partitions):
        logger.info(f'{label}: process partition {part_index} '
                    f'({len(part)} blocks)')
        futures = cluster_client.map(align_and_write_method, part, pure=False)
        result = _collect_results(futures, context=label) and result

    if output_transform is not None and run_displacement_diagnostics:
        _display_displacement_diagnostics(output_transform, blocks_ids,
                                          blocks_coords, fix_spatial_spacing,
                                          cluster_client, context=label)
    return result


def _partition(items, max_cluster_jobs):
    if max_cluster_jobs and max_cluster_jobs > 0:
        return [items[i:i + max_cluster_jobs]
                for i in range(0, len(items), max_cluster_jobs)]
    return [items]


def _collect_results(futures, context=''):
    """
    Drain a batch of block futures, logging and releasing each as it lands.

    `context` names the stage the blocks belong to - a pass label, or
    `compose` - so a failed block can be traced to the pass that produced
    it rather than only to a block index that recurs in every pass.
    """
    prefix = f'{context}: ' if context else ''
    res = True
    n_remaining_results = len(futures)
    for f, r in as_completed(futures, with_results=True):
        if f.cancelled():
            exc = f.exception()
            logger.error(f'{prefix}Block exception: {exc}')
            tb = f.traceback()
            traceback.print_tb(tb)
            n_remaining_results = n_remaining_results - 1
            res = False
        else:
            bi, bc = r
            n_remaining_results = n_remaining_results - 1
            logger.debug((
                f'{prefix}Finished computing deformation field for {bi} '
                f'at {bc} (remaining {n_remaining_results}) '
            ))
        # Release the future to free worker memory
        f.release()

    return res


def _display_displacement_diagnostics(output_transform, blocks_ids,
                                      blocks_coords, fix_spacing,
                                      cluster_client,
                                      context='assembled field'):
    """
    Single diagnostic sweep over a fully-assembled displacement field.

    Reads the field back region by region and reports jacobian/folding
    statistics on it. This is the run that means something: per-block
    diagnostics during accumulation are unreliable, because a block's
    overlap regions only reach their final values after every neighbour has
    added its weighted contribution.

    Region by region rather than all at once because an assembled field is
    far too large to hold in the client at production sizes - and one task
    per region, on the cluster, because the jacobian determinant is
    expensive enough that sweeping a production field serially in the client
    is the slowest thing left in a run.

    Each task returns only `deform_field_diagnostics`' scalar summary, a few
    hundred bytes, so what comes back through the scheduler is negligible
    next to the region it was computed from. The reporting happens here, in
    submission order, so the log reads the same as the serial sweep did.
    """
    logger.info(f'Compute displacement diagnostics for {len(blocks_coords)} '
                f'regions of the {context}')

    def region_diagnostics(region):
        block_index, block_coords = region
        try:
            return (block_index, block_coords,
                    deform_field_diagnostics(output_transform[block_coords],
                                             fix_spacing),
                    None)
        except Exception as e:
            # a diagnostic must never take the run down with it - report the
            # region that failed and carry on with the rest
            return block_index, block_coords, None, f'{type(e).__name__}: {e}'

    regions = list(zip(blocks_ids, blocks_coords))
    futures = cluster_client.map(region_diagnostics, regions, pure=False)
    for block_index, block_coords, stats, error in cluster_client.gather(futures):
        if error is not None:
            logger.error(f'Error computing diagnostics for {context} '
                         f'region {block_index} at {block_coords}: {error}')
        else:
            log_deform_field_diagnostics(
                stats, context=f'{context} {block_index} diagnostics')


def _tile_volume(shape, tile):
    """Plain non-overlapping cover of `shape`, for a diagnostics sweep."""
    tile = np.maximum(np.asarray(tile, dtype=int), 1)
    nblocks = np.ceil(np.array(shape) / tile).astype(int)
    ids, coords = [], []
    for index in np.ndindex(*nblocks):
        start = tile * np.array(index)
        stop = np.minimum(shape, start + tile)
        ids.append(index)
        coords.append(tuple(slice(int(a), int(b))
                            for a, b in zip(start, stop)))
    return ids, coords


# -------------------------------------------------------------------------
# what a pass is
# -------------------------------------------------------------------------



def _first_not_none(*values):
    for v in values:
        if v is not None:
            return v
    return None


def _as_int_tuple(value, ndim, name):
    if value is None:
        return None
    arr = np.atleast_1d(np.asarray(value))
    if arr.size == 1:
        arr = np.full(ndim, arr[0])
    if arr.size != ndim:
        raise ValueError(
            f'{name} must be a scalar or have {ndim} values (zyx), got {value}'
        )
    return tuple(int(v) for v in arr)


# -------------------------------------------------------------------------
# how the volume is cut into blocks
# -------------------------------------------------------------------------

@dataclass
class _BlockLattice:
    """
    The block partition for one pass.

    Two rectangles per block, both in absolute volume voxel coordinates:

      * the **core**, `[start, start + B)`, the territory the block owns and
        where its blending weight is 1. Cores tile the volume exactly.
      * the **footprint**, the core grown by `halo` on every side. This is
        what is read, aligned and written; footprints overlap, and the
        overlap is where the blend ramp lives.

    Either rectangle can stick out of the volume, at both ends once the
    lattice has a nonzero `offset`. The clipped footprint is what gets read;
    `clip_amounts` reports how much was cut off each side so the weight array
    can be cut to match.
    """

    block_size: Tuple[int, ...]
    halo: Tuple[int, ...]
    offset: Tuple[int, ...]
    extent: Tuple[int, ...]
    starts: List[np.ndarray] = field(init=False)

    def __post_init__(self):
        self.block_size = tuple(int(v) for v in self.block_size)
        self.halo = tuple(int(v) for v in self.halo)
        self.offset = tuple(int(v) % b
                            for v, b in zip(self.offset, self.block_size))
        self.extent = tuple(int(v) for v in self.extent)
        self.starts = [_axis_block_starts(e, b, o)
                       for e, b, o in zip(self.extent, self.block_size,
                                          self.offset)]

    @property
    def ndim(self):
        return len(self.block_size)

    @property
    def nblocks(self):
        return tuple(len(s) for s in self.starts)

    def indices(self):
        """Every block index, in C order - the order blocks are submitted."""
        return list(np.ndindex(*self.nblocks))

    def core_bounds(self, index):
        """Unclipped `(start, stop)` of the block core."""
        start = np.array([self.starts[a][i] for a, i in enumerate(index)])
        return start, start + np.array(self.block_size)

    def footprint_bounds(self, index):
        """Unclipped `(start, stop)` of the block footprint."""
        start, stop = self.core_bounds(index)
        return start - np.array(self.halo), stop + np.array(self.halo)

    def core_slices(self, index):
        """The block core, clipped to the volume."""
        start, stop = self.core_bounds(index)
        start = np.maximum(0, start)
        stop = np.minimum(self.extent, stop)
        return tuple(slice(int(a), int(b)) for a, b in zip(start, stop))

    def footprint_slices(self, index):
        """The block footprint, clipped to the volume."""
        start, stop = self.footprint_bounds(index)
        start = np.maximum(0, start)
        stop = np.minimum(self.extent, stop)
        return tuple(slice(int(a), int(b)) for a, b in zip(start, stop))

    def clip_amounts(self, index):
        """
        How many voxels of the nominal footprint fall outside the volume,
        `(before, after)` per axis. Fed to `_get_transform_weights` so the
        weight array is cut exactly where the data was.
        """
        start, stop = self.footprint_bounds(index)
        before = np.maximum(0, -start)
        after = np.maximum(0, stop - np.array(self.extent))
        return before.astype(int), after.astype(int)

    def core_within_footprint(self, index):
        """The core as a slice into the block's own (clipped) footprint array."""
        footprint = self.footprint_slices(index)
        core = self.core_slices(index)
        return tuple(slice(c.start - f.start, c.stop - f.start)
                     for c, f in zip(core, footprint))

def _axis_block_starts(extent, block_size, offset):
    """
    Lattice core starts along one axis, anchored at **volume voxel 0**.

    The lattice is the infinite set `{offset + n*B : n in Z}`; this returns
    every cell of it that intersects `[0, extent)`. Anchoring at the volume
    origin rather than at the ROI or chunk origin is what makes two runs, two
    chunks or two ROIs covering the same region compute the *same* blocks, so
    the result is reproducible across chunkings.

    Starts may be negative: with `offset = 96` and `B = 384` the cell
    `[-288, 96)` is the one that covers voxels 0..95, and it is a partial
    block clipped to `[0, 96)`. Emitting it (rather than starting the lattice
    at 96 and leaving the first 96 voxels uncovered) is what keeps the
    blending weights a partition of unity everywhere.
    """
    block_size = int(block_size)
    offset = int(offset) % block_size
    first = -1 if offset > 0 else 0
    last = int(np.ceil((extent - offset) / block_size))
    last = max(last, first + 1)
    return np.arange(first, last, dtype=int) * block_size + offset




def _select_blocks(lattice,
                   fix_spatial_dims=None,
                   fix_mask_image=None,
                   foreground_percentage=0.,
                   roi_start=None,
                   roi_stop=None,
                   max_concurrent_reads=0,
                   label='pass'):
    """
    Which blocks of the lattice actually get aligned.

    A block is dropped when its core holds too little foreground, or when its
    footprint misses the ROI entirely. Dropped blocks keep their lattice
    index, so the surviving blocks still know that a *neighbour that exists
    but was not aligned* is different from one that is off the volume - the
    two need opposite blending treatment (see `_get_transform_weights`).
    """
    fix_mask_spatial_dims = (np.array(fix_mask_image.spatial_dims)
                             if fix_mask_image is not None else None)
    blocks_ids, blocks_coords = [], []
    for index in lattice.indices():
        block_slice = lattice.footprint_slices(index)
        if any(s.stop <= s.start for s in block_slice):
            continue

        if fix_mask_image is not None:
            core = lattice.core_slices(index)
            ratio = fix_mask_spatial_dims / np.array(fix_spatial_dims)
            mask_start = np.round(
                ratio * np.array([s.start for s in core])).astype(int)
            mask_stop = np.round(
                ratio * np.array([s.stop for s in core])).astype(int)
            mask_coords = tuple(slice(int(a), int(b))
                                for a, b in zip(mask_start, mask_stop))
            mask_crop = _read_imagedata_block(
                mask_coords, fix_mask_image,
                ThrottledArraySliceReader(max_concurrent_reads))
            foreground_ratio = (np.sum(mask_crop) / np.prod(mask_crop.shape)
                                if mask_crop is not None and mask_crop.size
                                else 0.)
            if foreground_ratio < foreground_percentage:
                logger.debug((
                    f'{label}: ignore masked block {index} - foreground ratio '
                    f'{foreground_ratio} < {foreground_percentage}'
                ))
                continue

        if roi_start is not None:
            block_min = np.array([s.start for s in block_slice])
            block_max = np.array([s.stop for s in block_slice])
            if np.any(block_max <= roi_start) or np.any(block_min >= roi_stop):
                logger.debug(f'{label}: block {index} outside the ROI, skipping')
                continue

        blocks_ids.append(index)
        blocks_coords.append(block_slice)

    logger.info(f'{label}: {len(blocks_ids)} of '
                f'{int(np.prod(lattice.nblocks))} blocks selected')
    return blocks_ids, blocks_coords


# -------------------------------------------------------------------------
# one block: prepare, read, align, write
# -------------------------------------------------------------------------


def _block_context(pass_label, block_index):
    """
    How a block names itself in a log line: `<pass label> <block index>`.

    The block index alone is ambiguous in a multi-pass run - every pass
    walks a lattice over the same volume, so the same index comes round
    once per pass with different coords, a different static transform list
    and a different output. `pass_label` is the pass's `name`, or `passN`
    when unnamed, and is the same string the pass level messages use.
    Falls back to the bare index when there is no pass (the compose stage,
    or a direct call in a test).
    """
    return f'{pass_label} {block_index}' if pass_label else f'{block_index}'


def _prepare_compute_block_spatial_transform_params(block_info,
                                                    fix_shape=None,
                                                    mov_shape=None,
                                                    fix_spacing=None,
                                                    mov_spacing=None,
                                                    fix_fullmask_shape=None,
                                                    mov_fullmask_shape=None,
                                                    mov_origin_transform=None,
                                                    static_transform_list=[],
                                                    pass_label=''):
    block_index, fix_block_coords, fix_block_neighbors = block_info
    block_context = _block_context(pass_label, block_index)
    logger.debug(f'Prepare block coords {block_context}: {block_info[1:]}')
    fix_block_voxel_coords, fix_block_phys_coords = _get_spatial_block_corner_coords(fix_block_coords, fix_spacing)
    logger.debug((
        f'Block index: {block_context} - '
        f'fix block corner physical coords: {fix_block_phys_coords} '
        f'using fix spacing = {fix_spacing}'
    ))

    # parse initial transforms
    # recenter affines, read deforms, apply transforms to crop coordinates
    updated_block_transform_list = []
    mov_block_phys_coords = np.copy(fix_block_phys_coords)
    if mov_origin_transform is not None:
        # Convert to homogeneous coordinates (n, 3) -> (n, 4)
        homogeneous = np.c_[mov_block_phys_coords, np.ones(mov_block_phys_coords.shape[0])]
        # Apply transform and extract first 3 columns
        mov_block_phys_coords = (mov_origin_transform @ homogeneous.T).T[:, :3]
    # traverse current transformations in reverse order
    for transform in static_transform_list[::-1]:
        mov_block_phys_coords, block_transform = _get_spatial_moving_block_coords(
            fix_shape,
            fix_spacing,
            fix_block_voxel_coords[0],
            fix_block_voxel_coords[-1],
            fix_block_phys_coords,
            mov_block_phys_coords,
            transform)
        updated_block_transform_list.append(block_transform)

    logger.debug((
        f'Block {block_context} :'
        f'moving block physical coords {mov_block_phys_coords}, '
        f'using moving spacing = {mov_spacing}'
    ))

    block_transform_list = updated_block_transform_list[::-1]  # reverse it

    mov_start_phys_coords = np.min(mov_block_phys_coords, axis=0)
    mov_stop_phys_coords = np.max(mov_block_phys_coords, axis=0)

    logger.debug((
        f'Block {block_context}\n'
        f'fix block start physical coords: {fix_block_voxel_coords[0]}\n'
        f'fix block stop physical coords: {fix_block_voxel_coords[-1]}\n'
        f'moving block start physical coords: {mov_start_phys_coords}\n'
        f'moving block stop physical coords: {mov_stop_phys_coords}\n'
    ))

    # get moving image crop, read moving data
    mov_block_coords = mov_block_phys_coords / mov_spacing

    logger.debug((
        f'Block {block_context} :'
        f'moving block voxel coords {mov_block_coords},'
    ))

    mov_start = np.min(np.floor(mov_block_coords).astype(int), axis=0)
    # Slice stops are exclusive. +1 includes the voxel containing max corner index.
    mov_stop = np.max(np.ceil(mov_block_coords).astype(int), axis=0) + 1

    logger.debug((
        f'Block {block_context} \n'
        f'non-truncated moving block start (voxel coords): {mov_start}\n'
        f'non-truncated moving block stop (voxel coords): {mov_stop}'
    ))

    mov_start = np.maximum(0, mov_start)
    mov_stop = np.minimum(np.array(mov_shape)-1, mov_stop)
    mov_slices = tuple(slice(a, b) for a, b in zip(mov_start, mov_stop))

    # get moving crop origin relative to fixed crop
    mov_origin = mov_start * mov_spacing - fix_block_phys_coords[0]

    logger.debug((
        f'Block {block_context}:\n'
        f'fix block voxel coords:\n{fix_block_voxel_coords}\n'
        f'fix block phys coords:\n{fix_block_phys_coords}\n'
        f'mov origin relative to fix origin phys coords: {mov_origin}\n'
        f'mov block voxel coords {mov_slices}\n'
        f'mov block phys coords:\n{mov_block_phys_coords}\n'
    ))

    # read masks
    fix_blockmask_coords, mov_blockmask_coords = None, None
    if fix_fullmask_shape is not None:
        ratio = np.array(fix_fullmask_shape) / fix_shape
        fix_mask_start = np.round(ratio * fix_block_voxel_coords[0]).astype(int)
        fix_mask_stop = np.round(
            ratio * (fix_block_voxel_coords[-1] + 1)).astype(int)
        fix_blockmask_coords = tuple(slice(a, b)
                                     for a, b in zip(fix_mask_start,
                                                     fix_mask_stop))
        logger.debug((
            f'Fix mask block {block_context} coords: {fix_blockmask_coords}'
        ))

    if mov_fullmask_shape is not None:
        ratio = np.array(mov_fullmask_shape) / mov_shape
        mov_mask_start = np.round(ratio * mov_start).astype(int)
        mov_mask_stop = np.round(ratio * mov_stop).astype(int)
        if np.all(mov_mask_stop > 0) and np.all(mov_mask_stop > mov_mask_start):
            mov_blockmask_coords = tuple(slice(a, b)
                                        for a, b in zip(mov_mask_start,
                                                        mov_mask_stop))
        logger.debug((
            f'Mov mask block {block_context} coords: {mov_blockmask_coords}'
        ))

    logger.debug((
        'Return blocks data: '
        f'{block_context}, {fix_block_coords}, '
        f'{mov_origin}, {mov_slices}, '
        f'{fix_blockmask_coords}, {mov_blockmask_coords}'
    ))

    return (block_index,
            fix_block_coords,
            fix_block_neighbors,
            mov_slices,
            fix_blockmask_coords,
            mov_blockmask_coords,
            mov_origin,
            block_transform_list)


# get image block corners both in voxel and physical units
def _get_spatial_block_corner_coords(block_slice_coords, voxel_spacing):
    """
    The method returns corner coo
    """
    block_coords_list = []
    for corner in list(product([0, 1], repeat=3)):
        a = [x.stop-1 if y else x.start
             for x, y in zip(block_slice_coords, corner)]
        block_coords_list.append(a)

    block_corners_voxel_units = np.array(block_coords_list)
    block_corners_phys_units = block_corners_voxel_units * voxel_spacing
    return block_corners_voxel_units, block_corners_phys_units


def _get_spatial_moving_block_coords(fix_shape,
                                     fix_spacing,
                                     fix_block_min_voxel_coords,
                                     fix_block_max_voxel_coords,
                                     fix_block_phys_coords,
                                     original_mov_block_phys_coords,
                                     static_transform):
    if len(static_transform.shape) == 2:
        logger.debug(f'Apply affine transform {static_transform} to moving block at: {original_mov_block_phys_coords}')
        mov_block_phys_coords = bst.apply_transform_to_coordinates(
            original_mov_block_phys_coords,
            [static_transform,],
        )
        block_transform = bst.change_affine_matrix_origin(
            static_transform, fix_block_phys_coords[0])
    else:
        logger.debug(f'Apply deform field of shape {static_transform.shape} to {original_mov_block_phys_coords}')
        spacing = ut.relative_spacing(static_transform.shape,
                                      fix_shape,
                                      fix_spacing)
        ratio = np.array(static_transform.shape[:-1]) / fix_shape
        start = np.round(ratio * fix_block_min_voxel_coords).astype(int)
        stop = np.round(ratio * (fix_block_max_voxel_coords + 1)).astype(int)
        transform_slices = tuple(slice(a, b)
                                 for a, b in zip(start, stop))
        block_transform = static_transform[transform_slices]
        origin = spacing * start
        mov_block_phys_coords = bst.apply_transform_to_coordinates(
            original_mov_block_phys_coords, [block_transform,], spacing, origin
        )
    return mov_block_phys_coords, block_transform


def _read_blocks_for_processing(blocks_info,
                                fix=None,
                                mov=None,
                                fix_mask=None,
                                mov_mask=None,
                                fix_block_reader=ThrottledArraySliceReader(),
                                mov_block_reader=ThrottledArraySliceReader(),
                                mask_reader=ThrottledArraySliceReader(),
                                pass_label=''):
    # blocks_info is a tuple containing the fields below 
    # and the extract method knows to get the coords of the block to be read
    #    0:block_index,
    #    1:fix_block_coords,
    #    2:fix_block_neighbors,
    #    3:mov_block_coords,
    #    4:fix_mask_block_coords,
    #    5:mov_mask_block_coords,
    #    6:mov_origin,
    #    7:block_transforms
    # do not log block_transform, the last field - it is a whole array
    logger.debug(f'Read blocks {_block_context(pass_label, blocks_info[0])}: '
                 f'{blocks_info[1:-1]}')
    fix_block = _read_imagedata_block(blocks_info[1], fix, fix_block_reader)

    mov_block_coords = blocks_info[3]
    if (mov_block_coords is None
            or any(s.stop <= 0 for s in mov_block_coords)
            or any(s.stop <= s.start for s in mov_block_coords)):
        logger.info('Moving block corresponding to '
                    f'{_block_context(pass_label, blocks_info[0])} '
                    'is out of range')
        mov_block = None
    else:
        mov_block = _read_imagedata_block(mov_block_coords, mov, mov_block_reader)

    fix_mask_block = _read_imagedata_block(blocks_info[4], fix_mask, mask_reader)

    mov_mask_block_coords = blocks_info[5]
    if (mov_mask_block_coords is None
            or any(s.stop <= 0 for s in mov_mask_block_coords)
            or any(s.stop <= s.start for s in mov_mask_block_coords)):
        mov_mask_block = None
    else:
        mov_mask_block = _read_imagedata_block(mov_mask_block_coords, mov_mask, mask_reader)

    return (blocks_info,
            fix_block,
            mov_block,
            fix_mask_block,
            mov_mask_block)


def _read_imagedata_block(block_coords, image_data, image_block_reader,
                          image_timeindex=None, image_channels=None):
    image_repr = as_image_data(image_data, image_timeindex=image_timeindex,
                               image_channels=image_channels)
    if image_repr is not None:
        b = image_block_reader.read_slice(
            block_coords,
            image=image_repr.image_array,
            image_path=image_repr.image_path,
            image_subpath=image_repr.image_subpath,
            image_timeindex=image_repr.image_timeindex,
            image_channel=image_repr.image_channel,
        )
    else:
        b = None
    if b is not None:
        if np.issubdtype(b.dtype, np.floating):
            logger.debug(f'Convert block at {block_coords} to np.float32')
            return b.astype(np.float32)
    return b


def _align_block(compute_transform_params,
                 fix_spacing=None,
                 mov_spacing=None,
                 align_steps=[],
                 pass_label=''):
    """
    Run the alignment steps for a single block.

    `pass_label` names the pass this block belongs to - the pass's `name`,
    or `passN` when it is unnamed. It is carried into every log line and
    into the `context` handed to `alignment_pipeline`, because the same
    block index is aligned once per pass and the messages are otherwise
    indistinguishable in a multi-pass run.

    Returns `(block_index, block_coords, block_neighbors, transform)` where
    `transform` is the block's own displacement field, before any blending
    weight is applied. A block that could not be aligned returns a zero
    field.
    """
    start_time = time.time()
    ((block_index,
      block_coords,
      block_neighbors,
      _, # mov_block_coords,
      _, # fix_mask_block_coords,
      _, # mov_mask_block_coords,
      new_origin_phys,
      block_static_transform_list,
     ),
     fix_block,
     mov_block,
     fix_mask_block, # this can be a mask descriptor
     mov_mask_block, # this can be a mask descriptor
     ) = compute_transform_params
    block_context = _block_context(pass_label, block_index)
    logger.info((
        'Compute block transform '
        f'{block_context}: {block_coords}, {new_origin_phys} '
        f'fix shape: {fix_block.shape if fix_block is not None else 0}, '
        f'mov_shape: {mov_block.shape if mov_block is not None else 0} '
        f'using {len(block_static_transform_list)} transforms '
    ))

    # check if blocks have sufficient foreground content
    skip_alignment = False
    if fix_block is None or mov_block is None:
        logger.warning(f'Block {block_context} has no data, skipping alignment')
        skip_alignment = True

    if skip_alignment:
        # identity deform: zero displacement field
        block_shape = tuple(s.stop - s.start for s in block_coords)
        transform = np.zeros(block_shape + (len(block_coords),), dtype=np.float32)
    else:
        # run alignment pipeline
        # some pipeline algorithms use "fancy indexing" (list of tuples)
        # which is not supported yet by dask arrays
        # so in order to avoid the problem we materialize the fix and moving blocks
        transform = alignment_pipeline(
            fix_block, mov_block,
            fix_spacing, mov_spacing,
            align_steps,
            fix_mask=fix_mask_block,
            mov_mask=mov_mask_block,
            mov_origin=new_origin_phys,
            static_transform_list=block_static_transform_list,
            context=block_context,
        )
        # ensure transform is a vector field
        if len(transform.shape) == 2:
            transform = bst.matrix_to_displacement_field(
                transform, fix_block.shape, spacing=fix_spacing,
            )
        else:
            #  if it's a displacement field validate it
            # debug only: this is the block's own field, before blending,
            # so a fold here is not yet a fold in the result. Guarded rather
            # than left to the log level, because the diagnostics compute
            # unconditionally now and the jacobian determinant is one of the
            # largest allocations in a block's alignment
            if logger.isEnabledFor(logging.DEBUG):
                deform_field_diagnostics(
                    transform, fix_spacing,
                    context=f'{block_context} block displacement diagnostics',
                    level=logging.DEBUG,
                )

    # Finished computing transformation for current block_index
    logger.info((
        'Finished block alignment for '
        f'{block_context}:{block_coords} -> {transform.shape} '
        f'in {time.time()-start_time}s '
    ))

    return block_index, block_coords, block_neighbors, transform


def _write_block_transform(aligned_block,
                           block_size=None,
                           block_overlaps=None,
                           nblocks=None,
                           output_transform=None,
                           rebalance_for_missing_neighbors=True,
                           blend_ramp=None,
                           clip_before=None,
                           clip_after=None,
                           lock_grid=None,
                           lock_namespace='',
                           pass_label=''):
    """
    Weight one aligned block's field and overlap-add it into the output.

    `aligned_block` is what `_align_block` returns.

    The write is a read-modify-write and has to be locked, but *what* has to
    be locked depends on where the output lives, so the caller passes the
    grid it decided on (see `write_lock_grid`):

      * a zarr output rewrites a whole chunk or shard at a time - its
        *write unit* - so exclusion is keyed on that, not on the block: two
        blocks sharing a write unit clobber each other even with no voxel in
        common.
      * an in-memory output has no write unit; the only hazard is two
        overlapping blocks interleaving their accumulate, so it locks the
        block and its neighbours.
    """
    start_time = time.time()
    block_index, block_coords, block_neighbors, transform = aligned_block
    block_context = _block_context(pass_label, block_index)

    weights = _get_transform_weights(block_index,
                                     block_size,
                                     block_overlaps,
                                     block_neighbors,
                                     nblocks,
                                     rebalance_for_missing_neighbors,
                                     blend_ramp=blend_ramp,
                                     clip_before=clip_before,
                                     clip_after=clip_after,
                                     pass_label=pass_label)

    # handle end blocks
    if np.any(weights.shape != transform.shape[:-1]):
        crop = tuple(slice(0, s) for s in transform.shape[:-1])
        logger.debug(f'Crop weights for {block_context} ' +
                     f'from {transform.shape} to {weights.shape}')
        weights = weights[crop]

    # apply weights
    logger.debug(f'Block {block_context} :' +
                 f'Apply weights {weights.shape},' +
                 f'to transform {transform.shape}')
    transform = transform * weights[..., None]

    end_time = time.time()

    logger.debug(f'Finished computing {transform.shape} ' +
                 f'block  {block_context} transform in {end_time-start_time}s')

    if output_transform is not None:
        if lock_grid is not None:
            lock_keys = lock_cell_keys(output_transform, block_coords,
                                       lock_grid, lock_namespace)
        else:
            lock_keys = _overlapping_block_lock_keys(block_index,
                                                     block_neighbors,
                                                     lock_namespace)
        with write_lock(lock_keys, context=f'Block {block_context}'):
            # read-modify-write: every overlapping block adds its weighted
            # share, and the sum only reaches 1 once they all have
            output_block = output_transform[block_coords] + transform
            logger.info(f'Writing {output_block.shape} block {block_context} at {block_coords}')
            output_transform[block_coords] = output_block
            logger.info(f'Finished writing block {block_context} at {block_coords}')

    return block_index, block_coords


def _get_transform_weights(block_index,
                           block_size,
                           block_overlaps,
                           block_neighbors,
                           nblocks,
                           rebalance_for_missing_neighbors,
                           blend_ramp=None,
                           clip_before=None,
                           clip_after=None,
                           pass_label=''):
    """
    The blending weight array `w` for one block, shaped like the block footprint.

    `w` is 1 on the block core and ramps to 0 over the last `2*overlap`
    voxels of each face, so that two adjacent blocks' weights sum to exactly
    1 everywhere they overlap (the partition of unity the overlap-add stitch
    depends on).

    blend_ramp : str (default: None)
        Shape of that ramp, `'linear'` (the default) or `'cosine'`. See
        `bigstream.blend_ramp`; `'linear'` is bit-identical to the
        `np.pad(..., mode='linear_ramp')` this used to call directly.

    clip_before, clip_after : per axis voxel counts (default: None)
        How much of the block's *nominal* footprint falls outside the volume
        on the low and high side of each axis, i.e. how much of the weight
        array to drop. When either is None both are derived from the block's
        position in the lattice, which is correct only for a lattice anchored
        at voxel 0. A lattice with a nonzero phase offset has partial blocks
        at both ends and must pass the amounts explicitly.
    """
    block_context = _block_context(pass_label, block_index)
    logger.debug(f'Adjust transform for {block_context}')

    # create the standard weights array
    core = tuple(max(x - 2*y + 2, 0) for x, y in zip(block_size, block_overlaps))
    pad = tuple((max(2*y - 1, 0), max(2*y - 1, 0)) for y in block_overlaps)
    weights = blend_ramp_weights(core, pad, blend_ramp)

    # A neighbor can be absent for two reasons and they need opposite handling.
    #
    #   - It lies off the volume. Nothing exists beyond that face and this
    #     block's footprint is clipped there, so absorbing the neighbor's share
    #     of the blend is correct.
    #   - It lies inside the volume but was never aligned - dropped as
    #     background by fix_mask/foreground_percentage, or excluded by the roi.
    #     That territory exists and nobody writes it. Absorbing the share there
    #     would hold this block's displacement at full weight out to its last
    #     voxel and then step to zero, which folds the stitched field in a
    #     block aligned sheet along the mask boundary - outside the mask, where
    #     the deformation was never constrained by any image data. Keeping the
    #     ramp instead fades this block's deformation to identity as it
    #     reaches into the unaligned region.
    edge_neighbors, unaligned_neighbors = [], []
    for neighbor, flag in block_neighbors.items():
        if flag:
            continue
        neighbor_index = tuple(a + b for a, b in zip(block_index, neighbor))
        if any(i < 0 or i >= n for i, n in zip(neighbor_index, nblocks)):
            edge_neighbors.append(neighbor)
        else:
            unaligned_neighbors.append(neighbor)

    if unaligned_neighbors:
        logger.debug((
            f'Block {block_context} keeps its blending ramp toward '
            f'{len(unaligned_neighbors)} unaligned neighbors: '
            f'{unaligned_neighbors}'
        ))

    # rebalance only for the neighbors that are off the volume
    if rebalance_for_missing_neighbors and edge_neighbors:
        logger.debug(f'Rebalance transform {weights.shape} weights for '
                     f'{block_context}')
        # define overlap slices
        slices = {}
        slices[-1] = tuple(slice(0, 2*y) for y in block_overlaps)
        slices[0] = (slice(None),) * len(block_overlaps)
        # select the last 2*y elements per axis; index from the shape (not -2*y)
        # so a zero-overlap axis yields an empty slice instead of the whole axis
        slices[1] = tuple(slice(s - 2*y, s) for y, s in zip(block_overlaps, weights.shape))

        missing_weights = np.zeros_like(weights)
        for neighbor in edge_neighbors:
            neighbor_region = tuple(slices[-1*b][a]
                                    for a, b in enumerate(neighbor))
            region = tuple(slices[b][a]
                            for a, b in enumerate(neighbor))
            missing_weights[region] += weights[neighbor_region]

        # rebalance the weights
        with np.errstate(divide='ignore', invalid='ignore'):
            weights = weights / (1 - missing_weights)
        weights[np.isnan(weights)] = 0.  # edges of blocks are 0/0
        weights = weights.astype(np.float32)

    # crop weights if block is on edge of domain
    block_dim = len(block_index)
    if clip_before is None or clip_after is None:
        # lattice anchored at voxel 0: only the first and last block of each
        # axis stick out, and by exactly one overlap
        clip_before = [block_overlaps[i] if block_index[i] == 0 else 0
                       for i in range(block_dim)]
        clip_after = [block_overlaps[i] if block_index[i] == nblocks[i] - 1 else 0
                      for i in range(block_dim)]

    for i in range(block_dim):
        # no overlap padding on this axis -> nothing to crop. Guard also avoids
        # slice(None, -0) == slice(None, 0), which would empty the last block.
        if block_overlaps[i] <= 0:
            continue
        region = [slice(None),]*block_dim
        if clip_before[i] > 0:
            region[i] = slice(int(clip_before[i]), None)
            weights = weights[tuple(region)]
        if clip_after[i] > 0:
            region[i] = slice(None, -int(clip_after[i]))
            weights = weights[tuple(region)]

    return weights


# -------------------------------------------------------------------------
# write locking for an in-memory overlap-add
# -------------------------------------------------------------------------
#
# The zarr side of this lives in `bigstream.distributed_io_utility`, shared
# with the transform paths. What stays here is the one hazard that is
# specific to this module: an in-memory output has no write unit, and the
# only way two blocks collide is through the overlap-add itself.


def _overlapping_block_lock_keys(block_index, block_neighbors, namespace):
    """
    One lock key for this block and for every neighbour it shares voxels with.

    This is what guards an in-memory output, where the only hazard is the
    overlap-add itself: `out[coords] = out[coords] + t` is a read modify
    write, so two blocks sharing voxels must not interleave. Block A's key
    set contains B's name and B's contains A's, so they exclude each other.

    A zarr output needs nothing on top of its cell keys: two blocks that
    share a voxel also share the write unit holding it, and therefore the
    cell, so the cell keys already exclude them.
    """
    # coerce to plain int: a key is matched by string equality, and a numpy
    # integer renders as `np.int64(3)` while a python one renders as `3`
    return [f'{namespace}/block'
            f'{tuple(int(i) + int(o) for i, o in zip(block_index, offset))}'
            for offset, present in block_neighbors.items() if present]



# -------------------------------------------------------------------------
# is the configured amplitude safe to blend on this lattice
# -------------------------------------------------------------------------


# jacobian determinant held in reserve for the blend term when deriving the
# safe displacement ceiling. Shared by the computation and the messages so the
# two cannot drift apart.
_BLEND_MIN_JACOBIAN = 0.1


def _check_blend_safe_displacement(steps, block_size, block_overlaps,
                                   fix_spacing,
                                   blend_ramp=DEFAULT_BLEND_RAMP,
                                   error_when_check_fails=False):
    """
    Warn when the configured displacement bounds are too loose for this lattice.

    The per-block C4 constraint guarantees each block's own field does not
    fold, but stitching adds a `grad(w) * (u_A - u_B)` term that C4 does not
    bound - so two individually compliant blocks that fitted very different
    displacements still fold where they are blended. The safe ceiling depends
    on the blend ramp length, which only this function knows: neither the
    affine nor the deform step sees the block lattice.

    Critically, that ceiling is shared and single: `alignment_pipeline`
    composes every step of a block into one field before it is blended, so
    each step's own `max_displacement` is not an independent budget - their
    sum is what has to stay under the one ceiling, even though each one
    individually looks fine. Only a deform step's `bspline_constraints`
    affects what that ceiling actually is (via its `k`, the C4 allowance);
    `max_displacement` itself, on any step, only ever adds to the numerator.
    This assumes at most one step sets `bspline_constraints`, which matches
    every pipeline this is called from today.

    Every failing branch reports a concrete remedy. Which lever to reach for
    depends on something this function cannot know - whether the configured
    bound reflects a real deformation the data needs, or is simply too loose -
    so it gives the numbers for all of them rather than picking one.

    `blend_ramp` must be the ramp the pass actually blends with: a cosine
    ramp is `pi/2` steeper mid-ramp and lowers every ceiling reported here by
    that factor.
    """
    ndim = len(np.atleast_1d(block_overlaps))

    # only a deform step ever sets bspline_constraints (meaningless for
    # affine/rigid/ransac/random) - its k decides how much jacobian headroom
    # is left over for blending, i.e. the one ceiling every step shares.
    constrained_step_name, constraints = None, None
    for step_name, step_args in steps:
        candidate = step_args.get('bspline_constraints')
        if candidate:
            constrained_step_name, constraints = step_name, candidate
            break

    contributions = _max_displacement_contributions(steps)
    if constraints is None and not contributions:
        return  # nothing opted into blend-safety bookkeeping

    if constraints is not None:
        k = constraints.get('k')
        k = DEFAULT_K if k is None else k
    else:
        # no deform C4 step is consuming any jacobian headroom, so whatever
        # sets max_displacement (e.g. an affine step alone) gets the full
        # budget - see blend_safe_displacement_bound's k~0 case
        k = 1e-12
    ceiling = blend_safe_displacement_bound(
        block_overlaps, fix_spacing, k, min_jacobian=_BLEND_MIN_JACOBIAN,
        blend_ramp=blend_ramp,
    )
    total_configured = sum(
        float(np.max(np.atleast_1d(bound))) for _, bound in contributions)
    contributor_names = ', '.join(f"'{name}'" for name, _ in contributions)

    logger.info(f'Configured max displacements: {contributions} totaling {total_configured:.4g}')

    message = None
    if constraints is not None and ceiling == 0.0:
        # no finite bound helps: the C4 allowance alone consumes the whole
        # jacobian budget before blending contributes anything
        k_max = (1.0 - _BLEND_MIN_JACOBIAN) / ndim
        # a concrete suggestion for this lattice, and what it actually buys
        suggested_k = k_max / 2.0
        suggested_ceiling = blend_safe_displacement_bound(
            block_overlaps, fix_spacing, suggested_k,
            min_jacobian=_BLEND_MIN_JACOBIAN, blend_ramp=blend_ramp,
        )
        message = (
            f"'{constrained_step_name}' bspline_constraints k={k} leaves no "
            f'jacobian headroom for blockwise blending (sum(k)='
            f'{_total_k(k, ndim):.3g} plus the {_BLEND_MIN_JACOBIAN} reserve '
            'is already >= 1), so no finite max_displacement can keep the '
            f'stitched field fold free. Lower k below {k_max:.3f}; at '
            f'k={suggested_k:.3f} the ceiling becomes {suggested_ceiling:.4g} '
            'for this blocksize/overlap/spacing, and max_displacement must '
            'be set at or under whatever ceiling the chosen k yields.'
        )
    elif (constraints is not None
          and constrained_step_name not in dict(contributions)):
        # this deform step opted into blend-safety bookkeeping (it set
        # bspline_constraints) but never bounded its own amplitude - that
        # alone can fold the blend no matter what any other step contributes
        shared_note = (
            f' (would share the ceiling with {contributor_names} totaling '
            f'{total_configured:.4g})' if contributions else ''
        )
        message = (
            f"'{constrained_step_name}' sets bspline_constraints but no "
            'max_displacement, so a block that fits a large smooth '
            'deformation can still fold the stitched field where it blends '
            f'into a neighbour. Set max_displacement to <= {ceiling:.4g}'
            f'{shared_note} (same physical units as the spacing, i.e. '
            'expansion corrected). If your blocks legitimately need more '
            'than that, raise overlap_factor or lower k instead of raising '
            'the bound.'
        )
    elif total_configured > ceiling:
        remedy = _blend_remedy(
            total_configured, block_size, block_overlaps, fix_spacing, k,
            blend_ramp=blend_ramp)
        message = (
            f'configured max_displacement totals {total_configured:.4g} '
            f'across {contributor_names}, which exceeds the blend safe '
            f'ceiling {ceiling:.4g} for k={k}; the '
            'stitched field may fold where blocks disagree. To keep '
            f'{total_configured:.4g} total: {remedy}. Check the per block '
            '"max |c|"/displacement bound logs first: if your blocks never '
            'reach the configured bounds they are simply too loose and '
            'lowering them costs nothing, but if they do then the '
            'deformation is real and the ramp is what has to grow.'
        )
    else:
        logger.info((
            f'configured max_displacement totals {total_configured:.4g} '
            f'across {contributor_names or "(none)"}, within the blend safe '
            f'ceiling {ceiling:.4g}'
        ))

    if message is not None:
        if error_when_check_fails:
            raise ValueError(message)
        logger.warning(message)


def _max_displacement_contributions(steps):
    """
    Every step's own `max_displacement` (`affine_align`/`deformable_align`'s
    bound - see `align_constraints.bound_affine_displacement` and
    `project_bspline_transform`), keyed by step name. Both are top-level
    parameters of their respective functions, not nested under
    `bspline_constraints` (which only carries the deform step's C4 k/K).

    `alignment_pipeline` composes every step in `steps` into one field before
    blockwise blending mixes them, so these are not independent budgets -
    all of them land in the same stitched field and must share one ceiling.
    """
    return [(step_name, step_args['max_displacement'])
            for step_name, step_args in steps
            if step_args.get('max_displacement') is not None]


def _blend_remedy(wanted, block_size, block_overlaps, spacing, k,
                  min_jacobian=_BLEND_MIN_JACOBIAN,
                  blend_ramp=DEFAULT_BLEND_RAMP):
    """
    Spell out how to make a displacement bound of `wanted` legal on this lattice.

    The ceiling is `(1 - sum(k) - min_jacobian) * L / (2*ndim*g)` with
    `L = (2*overlap - 1) * spacing` the blend ramp length and `g` the ramp
    gradient gain (1 linear, `pi/2` cosine), so there are exactly three
    levers: shorten the deformation (lower max_displacement), lengthen the
    ramp (raise overlap_factor), or free jacobian budget (lower k). This
    returns the concrete value each lever would need, so the message says what
    to do rather than only what is wrong.

    This has to invert the *same* bound the check applied, gain included - a
    remedy computed for a linear ramp understates the overlap a cosine run
    needs by `pi/2` and would not actually clear the ceiling it is answering.
    """
    size = np.atleast_1d(np.asarray(block_size, dtype=np.float64))
    overlaps = np.atleast_1d(np.asarray(block_overlaps, dtype=np.float64))
    voxel = np.atleast_1d(np.asarray(spacing, dtype=np.float64))
    ndim = overlaps.size
    total_k = _total_k(k, ndim)
    gain = ramp_gradient_gain(blend_ramp)

    ramps = np.maximum(2.0 * overlaps - 1.0, 0.0) * voxel
    axis = int(np.argmin(np.where(ramps > 0, ramps, np.inf)))
    ramp = float(ramps[axis])

    options = [f'lower max_displacement to <= the ceiling']

    # lever 2: a longer ramp. L needed = 2*ndim*g*wanted / headroom
    headroom = 1.0 - total_k - min_jacobian
    if headroom > 0:
        needed_ramp = 2.0 * ndim * gain * wanted / headroom
        needed_overlap = (needed_ramp / voxel[axis] + 1.0) / 2.0
        factor = needed_overlap / size[axis]
        if factor <= 0.5:
            options.append(f'raise overlap_factor to ~{factor:.2f}')
        else:
            options.append(
                f'raise overlap_factor (would need ~{factor:.2f}, above the '
                '0.5 maximum, so this lever alone is not enough)')

    # lever 3: a smaller k. sum(k) needed = 1 - m - 2*ndim*g*wanted/L
    needed_total_k = 1.0 - min_jacobian - 2.0 * ndim * gain * wanted / ramp
    if needed_total_k > 0:
        options.append(f'lower k to ~{needed_total_k / ndim:.3f}')
    else:
        options.append(
            'lower k (no positive k works at this overlap_factor)')

    return '; '.join(options)


def _total_k(k, ndim):
    """sum(k) with a scalar broadcast across axes, as coefficient_bounds does."""
    return float(np.sum(np.broadcast_to(
        np.atleast_1d(np.asarray(k, dtype=np.float64)), (ndim,))))


# -------------------------------------------------------------------------
# composing the per-pass fields
# -------------------------------------------------------------------------


def _distributed_compose_displacement_fields(fields, spacing, output,
                                            cluster_client,
                                            processing_size=None,
                                            max_write_locks=MAX_WRITE_LOCKS,
                                            order=1):
    """
    Write `F1(F2(...Fn(x)))` into `output`, one output block per task.

    Every field must be on the same voxel grid as `output`.

    Output blocks are disjoint, so unlike the alignment write there is no
    read-modify-write to protect. A zarr output still needs locking though:
    disjoint *blocks* can land in the same chunk or shard, and the store
    rewrites that whole unit either way. An in-memory output needs no lock
    at all.
    """
    shape = tuple(output.shape[:-1])
    unit = storage_write_unit(output)
    if processing_size is None:
        if unit is not None:
            # one task per write unit: no write unit is then shared by
            # two tasks and the locks below never contend. `unit` covers the
            # vector axis too, so take only its spatial leading entries -
            # the blocks below are spatial.
            processing_size = tuple(int(u) for u in unit[:len(shape)])
        else:
            # an in-memory output has no natural unit; keep the per task
            # coordinate arrays bounded rather than walking the whole volume
            processing_size = tuple(min(s, 128) for s in shape)
    processing_size = np.asarray(processing_size, dtype=int)
    lock_grid = write_lock_grid(output, processing_size,
                                max_locks=max_write_locks)
    lock_namespace = write_lock_namespace(output, 'compose')
    log_write_locking(output, lock_grid, processing_size, 'compose')
    nblocks = np.ceil(np.array(shape) / processing_size).astype(int)
    logger.info((
        f'Compose {len(fields)} displacement fields into a {shape} output '
        f'using {tuple(nblocks)} blocks of {tuple(processing_size)}'
    ))

    blocks = []
    for index in np.ndindex(*nblocks):
        start = processing_size * np.array(index)
        stop = np.minimum(shape, start + processing_size)
        blocks.append((index, tuple(slice(int(a), int(b))
                                    for a, b in zip(start, stop))))

    def compose_method(block):
        block_index, block_coords = block
        return _compose_fields_block(block_coords, block_index=block_index,
                                     fields=fields, spacing=spacing,
                                     shape=shape, output=output, order=order,
                                     lock_grid=lock_grid,
                                     lock_namespace=lock_namespace)

    futures = cluster_client.map(compose_method, blocks, pure=False)
    return _collect_results(futures, context='compose')


def _compose_fields_block(block_coords, block_index=None, fields=None,
                          spacing=None, shape=None, output=None, order=1,
                          lock_grid=None, lock_namespace=''):
    """
    Compose several displacement fields over one output block.

    The fields are in `static_transform_list` order - last in the list is
    applied first to a fixed-image coordinate - so for `[F1, F2]` the result
    is `F1(F2(x))`, which is what `compose_transform_list` produces and what
    the cascade means: pass 2 refines on top of pass 1.

    Only the sub-region of each field that the walked coordinates actually
    land in is read, so this never materializes a whole field.
    """
    ndim = len(block_coords)
    shape = np.asarray(shape)
    spacing = np.asarray(spacing, dtype=np.float64)
    block_shape = tuple(s.stop - s.start for s in block_coords)

    grid = np.meshgrid(*[np.arange(s.start, s.stop, dtype=np.float64)
                         for s in block_coords], indexing='ij')
    points = np.stack([g.ravel() for g in grid], axis=-1) * spacing
    del grid
    origin_points = points.copy()

    for transform in fields[::-1]:
        voxels = points / spacing
        low = np.maximum(0, np.floor(voxels.min(axis=0)).astype(int) - 1)
        high = np.minimum(shape,
                          np.ceil(voxels.max(axis=0)).astype(int) + 2)
        del voxels
        crop = tuple(slice(int(a), int(b)) for a, b in zip(low, high))
        # spacing/origin are passed as 1-tuples: apply_transform_to_coordinates
        # reverses a bare array to match its reversed transform iteration, and
        # that would silently transpose an anisotropic spacing
        points = apply_transform_to_coordinates(
            points, [np.asarray(transform[crop]),],
            (spacing,), (low * spacing,), order=order, mode='nearest',
        )

    composed = (points - origin_points).reshape(block_shape + (ndim,))
    composed = composed.astype(np.float32)
    if output is not None:
        # a plain assignment, but on zarr it still rewrites whole units, so
        # another task writing a different part of the same unit has to wait
        lock_keys = lock_cell_keys(output, block_coords, lock_grid,
                                   lock_namespace)
        with write_lock(lock_keys, context=f'Compose block {block_index}'):
            output[block_coords] = composed
    return block_index, block_coords


# -------------------------------------------------------------------------
# reading passes out of a bigstream config
# -------------------------------------------------------------------------


def _deep_merge(base, override):
    merged = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def _step_defaults(config, context_config):
    """
    Per-step default arguments: top level `<step>:` sections overlaid with
    the `<context>.<step>:` sections.
    """
    defaults = {k: v for k, v in config.items() if isinstance(v, dict)}
    for k, v in context_config.items():
        if isinstance(v, dict):
            defaults[k] = _deep_merge(defaults.get(k) or {}, v)
    return defaults


def alignment_passes_from_config(config, context='local_align'):
    """
    Read the `alignment_passes` section of a bigstream config.

        local_align:
            processing_size: [384, 384, 384]        # default for every pass
            processing_halo_factor: 0.2             # default for every pass
            blend_ramp: cosine                      # default for every pass
            alignment_passes:
                - processing_offset: [0, 0, 0]
                  processing_halo_factor: [0.2, 0.2, 0.2]
                  alignment_steps:
                    - ransac: {...}
                    - deform: {...}
                - processing_offset: [96, 96, 96]
                  processing_halo_factor: [0.1, 0.1, 0.1]
                  blend_ramp: linear                # per-pass override
                  alignment_steps:
                    - deform: {...}

    Each pass's step arguments are layered over the per-step defaults: the
    config's top level sections (`ransac:`, `affine:`, `deform:` ...) with the
    `<context>` section's own (`local_align.deform:` ...) merged over them,
    exactly as `get_algorithm_parameters` does for the single-pass `steps:`
    list.

    Returns
    -------
    list of AlignmentPass, unresolved - call `resolved(ndim, ...)` on each, or
    let `blockwise_alignment_pipeline` do it.
    """
    context_config = config.get(context) or {}
    passes_spec = context_config.get('alignment_passes')
    if passes_spec is None and 'alignmentPass' in context_config:
        # the spelling the original prototype config used; everything
        # else here is snake_case
        logger.warning("'alignmentPass' is deprecated, use 'alignment_passes'")
        passes_spec = context_config.get('alignmentPass')
    if not passes_spec:
        return []

    step_defaults = _step_defaults(config, context_config)
    passes = []
    for i, spec in enumerate(passes_spec):
        spec = dict(spec or {})
        passes.append(AlignmentPass(
            alignment_steps=alignment_steps_from_config(
                spec.get('alignment_steps'), step_defaults=step_defaults),
            processing_size=spec.get('processing_size'),
            processing_offset=spec.get('processing_offset'),
            processing_halo_factor=spec.get('processing_halo_factor'),
            processing_halo=spec.get('processing_halo'),
            blend_ramp=spec.get('blend_ramp'),
            name=spec.get('name', f'pass{i + 1}'),
        ))
    return passes


def default_pass_geometry_from_config(config, context='local_align'):
    """
    The top level per-pass defaults: `(processing_size, halo_factor)`.

    `processing_halo_factor` and the pre-existing `block_overlap` /
    `overlap_factor` mean the same thing (a per-side fraction of the block
    size), so any of them serves as the default.
    """
    context_config = config.get(context) or {}
    halo_factor = _first_not_none(
        context_config.get('processing_halo_factor'),
        context_config.get('block_overlap'),
        context_config.get('overlap_factor'),
    )
    size = _first_not_none(context_config.get('processing_size'),
                           context_config.get('block_size'))
    return size, halo_factor


def default_blend_ramp_from_config(config, context='local_align'):
    """
    The top level `blend_ramp`, for passes that do not set their own.

    Kept separate from `default_pass_geometry_from_config` rather than added
    to its tuple: the ramp is not lattice geometry, and widening that return
    would break its call sites for a value most of them do not want.

    Returns the configured name unvalidated, or `None` when unset -
    `AlignmentPass.resolved` is where an unknown name is refused, so a pass
    that overrides it never has to care what the default was.
    """
    context_config = config.get(context) or {}
    return context_config.get('blend_ramp')


def alignment_steps_from_config(steps_spec, step_defaults=None):
    """
    Convert a config `alignment_steps` list into the `[(name, args), ...]`
    form `alignment_pipeline` wants.

    The config writes each step as a single-key mapping so the step name reads
    as a heading over its arguments:

        alignment_steps:
          - ransac:
              alignment_spacing: 4
          - deform:
              control_point_spacing: 128

    A bare string (`- ransac`) is also accepted and means "all defaults".

    Parameters
    ----------
    steps_spec : list
        The `alignment_steps` value from the config.

    step_defaults : dict (default: None)
        Per-step-name default arguments, i.e. the top level `ransac:` /
        `affine:` / `deform:` sections of the bigstream config. A step's
        inline arguments override these key by key, the same precedence
        `get_algorithm_parameters` applies to the single-pass config.

    Returns
    -------
    list of (str, dict)
    """
    step_defaults = step_defaults or {}
    steps = []
    for entry in (steps_spec or []):
        if isinstance(entry, str):
            name, args = entry, {}
        elif isinstance(entry, dict):
            if len(entry) != 1:
                raise ValueError(
                    'each entry of alignment_steps must be a single-key '
                    f'mapping of step name to its arguments, got {entry}'
                )
            (name, args), = entry.items()
            args = dict(args or {})
        else:
            raise ValueError(
                f'unsupported alignment_steps entry {entry!r}; expected a '
                'step name or a single-key mapping'
            )
        if name not in step_defaults:
            logger.warning(
                f'alignment step {name!r} has no config section; '
                'it will run with the library defaults plus its inline args')
        steps.append((name, {**(step_defaults.get(name) or {}), **args}))
    return steps
