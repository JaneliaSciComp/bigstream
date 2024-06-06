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
  safeguard_exceptions: true
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
  sampling: 'NONE'
  interpolator: '1'
  # shrink_factors - list of int
  shrink_factors: [1]
  # smooth_sigmas - list of float
  smooth_sigmas: [0]
  alignment_spacing: 1.0
  sampling_percentage:
  metric_args: {}
  optimizer_args: {}
  exhaustive_step_sizes:

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
  print_running_improvements: false

global_align:
  steps: [] # no default global steps

local_align:
  steps: [] # no default local steps
  block_size: [128, 128, 128]
  block_overlap: 0.5
  ransac:
    safeguard_exceptions: false

"""
