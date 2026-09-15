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
  control_point_constraint:
  #  k: 0.1             # scalar or [kz, ky, kx]; sum(k) < 1, min|J| >= 1-sum(k)
  #  K:                 # optional expansion allowance, defaults to k
  #  mode: final
  #  # Bound on the per-component displacement, in the same physical units as
  #  # the spacing the pipeline runs at - note that is voxel_spacing divided by
  #  # the expansion factor, not the raw value recorded in the zarr.
  #  #
  #  # 'k' bounds derivatives, not amplitude, so without this a smooth but huge
  #  # displacement is C4 compliant yet still folds where distributed_align
  #  # blends it against a neighbour that fitted something different. The safe
  #  # ceiling is (1 - sum(k) - 0.1) * L / (2 * ndim) with L the blend ramp
  #  # length, so it depends on blocksize/overlap/spacing - leave this unset
  #  # and the local align step logs the ceiling it computed for your lattice.
  #  #
  #  # Prefer a small k with a larger max_displacement over the reverse: k=0.1
  #  # still guarantees min|J| >= 0.7 per block, and spends the freed jacobian
  #  # budget on amplitude, which is what actually binds.
  #  max_displacement: 16

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

local_align:
  steps: [] # no default local steps
  block_size: [128, 128, 128]
  block_overlap: 0.5
  ransac:
    safeguard_exceptions: false

apply_deform:
  steps: [map_coordinates]
  map_coordinates:
    order: 3
    mode: nearest
"""
