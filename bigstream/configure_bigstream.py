import logging
import os
import sys

from logging.config import fileConfig


def configure_logging(config_file, verbose, logger_name=None):
    if config_file:
        print(f'Configure logging using {config_file}, logger name: {logger_name}')
        fileConfig(config_file)
    else:
        print(f'Configure logging using basic config - verbose: {verbose}, logger name: {logger_name}')
        log_level = logging.DEBUG if verbose else logging.INFO
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(level=log_level,
                            format=log_format,
                            datefmt='%Y-%m-%d %H:%M:%S',
                            handlers=[
                                logging.StreamHandler(stream=sys.stdout)
                            ])
    return logging.getLogger(logger_name)


def set_cpu_resources(cpus:int):
    if cpus:
        os.environ['ITK_THREADS'] = str(cpus)
        os.environ['MKL_NUM_THREADS'] = str(cpus)
        os.environ['NUM_MKL_THREADS'] = str(cpus)
        os.environ['OPENBLAS_NUM_THREADS'] = str(cpus)
        os.environ['OPENMP_NUM_THREADS'] = str(cpus)
        os.environ['OMP_NUM_THREADS'] = str(cpus)

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

"""
