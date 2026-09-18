import logging
import os
import sys

from logging.config import fileConfig


def configure_logging(config_file, verbose):
    if config_file:
        print(f'Configure logging using {config_file}')
        fileConfig(config_file)
    else:
        print(f'Configure logging using basic config - verbose: {verbose} ')
        log_level = logging.DEBUG if verbose else logging.INFO
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(level=log_level,
                            format=log_format,
                            datefmt='%Y-%m-%d %H:%M:%S',
                            handlers=[
                                logging.StreamHandler(stream=sys.stdout)
                            ])
    return logging.getLogger()


def set_cpu_resources(cpu_cores:int, threads_per_cpu=1):
    if cpu_cores:
        cpus = cpu_cores * threads_per_cpu
        print(f'Set CPU resources: {cpu_cores} * {threads_per_cpu} -> {cpus}')
        os.environ['ITK_THREADS'] = str(cpus * threads_per_cpu)
        # ITK honors this env var when its global MultiThreader initializes.
        # SimpleITK's SetGlobalDefaultNumberOfThreads does NOT affect the `itk`
        # package elastix uses (separate libraries), so bound the `itk` side here.
        os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(cpus)
        os.environ['MKL_NUM_THREADS'] = str(cpus)
        os.environ['NUM_MKL_THREADS'] = str(cpus)
        os.environ['OPENBLAS_NUM_THREADS'] = str(cpus)
        os.environ['OPENMP_NUM_THREADS'] = str(cpus)
        os.environ['OMP_NUM_THREADS'] = str(cpus)
    else:
        cpus = 1

    print('OS environment: ', os.environ)

    return cpus


default_bigstream_config_str="""
spot_detection: &spot_detection_args
  blob_method: 'log'
  threshold:
  threshold_rel: 0.1
  winsorize_limits: [0.01, 0.01]
  background_subtract: false

ransac: &ransac_args
  nspots: 2000
  blob_sizes: [6, 20]
  num_sigma_max: 15
  cc_radius: 12
  # default safeguard_exceptions to false
  safeguard_exceptions: false
  match_threshold: 0.7
  max_spot_match_distance:
  point_matches_threshold: 50
  align_threshold: 2.0
  confidence: 0.999
  diagonal_constraint: 0.25
  fix_spots_count_threshold: 100
  fix_spot_detection_kwargs:
    <<: *spot_detection_args
  mov_spots_count_threshold: 100
  mov_spot_detection_kwargs:
    <<: *spot_detection_args

affine: &affine_args
  optimizer: RSGD # see configure_irm for default RSGD args
  metric: MMI
  # shrink_factors - list of int
  shrink_factors: [1]
  # smooth_sigmas - list of float
  smooth_sigmas: [0]
  alignment_spacing: 1.0
  metric_args: {}
  optimizer_args: {}

deform: &deform_args
  <<: *affine_args
  control_point_spacing: 50
  control_point_levels: [1]
  # optional local invertibility constraint (Chun & Fessler 2009):
  # guarantees the deformation does not fold, at some cost in metric value.
  # omit or leave null to disable (default)
  bspline_constraints:
  #  k: 0.1             # scalar or [kz, ky, kx]; sum(k) < 1, min|J| >= 1-sum(k)
  #  K:                 # optional expansion allowance, defaults to k
  #
  # max_displacement is a sibling of bspline_constraints, not a key inside
  # it - it is independent of k/K and still applies even when
  # bspline_constraints above is left disabled.
  #
  # max_displacement: bound on the per-component displacement, in the same
  #  physical units as the spacing the pipeline runs at - note that is
  #  voxel_spacing divided by the expansion factor, not the raw value
  #  recorded in the zarr.
  #
  #  'k' bounds derivatives, not amplitude, so without this a smooth but huge
  #  displacement is C4 compliant yet still folds where blockwise alignment
  #  blends it against a neighbour that fitted something different (this
  #  includes another step's own max_displacement in the same pipeline, e.g.
  #  an 'affine' step run before this 'deform' step - they share one ceiling).
  #  The safe ceiling is (1 - sum(k) - 0.1) * L / (2 * ndim) with L the blend
  #  ramp length, so it depends on blocksize/overlap/spacing - leave this
  #  unset and the local align step logs the ceiling it computed for your
  #  lattice.
  #
  #  Prefer a small k with a larger max_displacement over the reverse: k=0.1
  #  still guarantees min|J| >= 0.7 per block, and spends the freed jacobian
  #  budget on amplitude, which is what actually binds.
  #
  #  Better still, do not cap amplitude at all: see
  #  local_align.neighbor_consistency below. That ceiling is only this tight
  #  because a block cannot see its neighbours, so the bound has to assume
  #  the worst case disagreement 2*max_displacement. What actually folds the
  #  stitch is the disagreement, not the amplitude, and bounding it directly
  #  gives the identical guarantee while leaving amplitude free.
  # max_displacement: 16

elastix_deform: &elastix_deform_args
  align_method: bspline
  alignment_spacing: 1.0
  control_point_spacing: 50
  NumberOfResolutions: 4
  # any additional keys here are forwarded verbatim as elastix parameter-map
  # entries e.g. RandomSeed: 42

rigid:
  <<: *affine_args
  rigid: true

random:
  <<: *affine_args
  use_patch_mutual_information: false

global_align:
  steps: [] # no default global steps
  # Passing a processing size to the global tool switches it from a single
  # whole-image fit to one blockwise pass over the volume. Always exactly
  # one pass: multi-pass cascading is a local-stage tool, and a global
  # alignment runs on a downsampled volume looking for the low frequency
  # part of the deformation, which one pass already carries.

local_align:
  # A single flat list of steps is one alignment pass over the volume.
  steps: [] # no default local steps

  # Block geometry, in VOXELS, zyx. These are the defaults every pass
  # inherits when it does not set its own.
  #
  #   block_size     the block step: the distance between the origins of two
  #                  adjacent blocks. Also spelled `processing_size`, which
  #                  wins if both are present - so set one, not both.
  #   block_overlap  the halo, as a fraction of the block size, PER SIDE. A
  #                  block reads and writes block_size + 2*halo voxels; the
  #                  halo is where it blends into its neighbours. Also
  #                  spelled `processing_halo_factor` (or `overlap_factor`),
  #                  which wins if both are present.
  #
  # Nothing requires these to be a multiple of the output chunk or shard:
  # block writes are locked on the write unit, so any block geometry is
  # safe. Blocks that land in a shared chunk are simply serialized.
  block_size: [128, 128, 128]
  block_overlap: 0.5

  # Several passes over the volume, each on its own block lattice, each
  # fitting only the residual its predecessors left, composed at the end.
  # Total deformation capacity becomes the SUM of the per-pass budgets
  # rather than one budget, and staggering `processing_offset` puts one
  # pass's block seams in the next pass's block interiors, so no seam is
  # ever reinforced. Replaces `steps` above when present.
  #
  # alignment_passes:
  #   - processing_offset: [0, 0, 0]          # lattice phase, voxels zyx.
  #                                           # Block origins sit at
  #                                           # offset + n*block_size measured
  #                                           # from voxel 0 of the WHOLE
  #                                           # volume, never from the ROI or
  #                                           # a chunk, so the same block is
  #                                           # computed however the work is
  #                                           # split.
  #     processing_halo_factor: [0.2, 0.2, 0.2]
  #     alignment_steps:                      # single-key dicts, in order
  #       - ransac: {alignment_spacing: 4}
  #       - affine: {alignment_spacing: 4.0}
  #       - deform: {control_point_spacing: 128}
  #   - processing_size: [384, 384, 384]      # restate it so the offset below
  #                                           # is the intended 1/4 stagger
  #     processing_offset: [96, 96, 96]
  #     processing_halo_factor: [0.1, 0.1, 0.1]   # a smaller residual needs
  #                                               # less reach
  #     alignment_steps:
  #       - deform: {control_point_spacing: 128}
  #
  # See configs/bigstream_config_prototype.yml for a complete example.

  # Bound on how much two overlapping blocks may DISAGREE, in the same
  # physical units as the voxel spacing (expansion corrected). Applies to
  # every pass unless the pass overrides it.
  #
  # This is the alternative to capping deform.max_displacement, and the
  # better one. `max_displacement` bounds absolute motion - naturally large,
  # it is the deformation being measured. `delta_max` bounds neighbour
  # disagreement - naturally small, because overlapping blocks see mostly
  # the same tissue. Both give the same fold guarantee; only the second
  # leaves the deformation free.
  #
  # Each block is clamped to within delta_max/2 of a smooth estimate
  # reconstructed from the whole lattice, so two overlapping blocks differ
  # by at most delta_max. A block that failed its metric check contributes
  # nothing to that estimate and adopts it wholesale, instead of asserting
  # a zero displacement it has no evidence for.
  #
  # Costs: the pass runs in two stages and the per-block fields stay
  # resident in the cluster between them. Omit the section to disable it.
  #
  # neighbor_consistency:
  #   delta_max: auto   # a number, or 'auto' to derive it from each pass's
  #                     # own halo and spacing
  #   sigma: 1.0        # reconstruction width, in lattice nodes
  #   k: 0.32           # C4 allowance 'auto' assumes; match your deform step

  affine:
  #  # optional bound on the per-component displacement this block's affine
  #  # step may contribute, physical units, same frame as deform's own
  #  # max_displacement. None (default) disables it.
  #  #
  #  # A block with too little foreground to anchor the fit can still return
  #  # an individually valid (invertible) but wildly implausible affine - a
  #  # large anisotropic scale plus a large offset - that folds where
  #  # distributed_align blends it against a better-behaved neighbour. Pick
  #  # this jointly with the deform step's own max_displacement (their
  #  # contributions add when composed) - see bound_affine_displacement and
  #  # blend_safe_displacement_bound.
  #  max_displacement: 16
  ransac:
    safeguard_exceptions: false

apply_deform:
  steps: [map_coordinates]
  map_coordinates:
    order: 3
    mode: nearest
"""
