

multifish_registration_global_affine_defaults = {
    'nspots': 2000,
    'cc_radius': 8,
    'match_threshold': 0.75,
    'align_threshold': 2.5,
}


multifish_registration_local_affine_defaults = {
    'nspots': 2000,
    'cc_radius': 8,
    'match_threshold': 0.75,
    'align_threshold': 2.5,
    'blocksize': [256,]*3,
    'cluster_extra': ['',],
}


multifish_registration_deform_defaults = {
    'cc_radius': 12,
    'gradient_smoothing': [3., 0., 1., 2.],
    'field_smoothing': [.5, 0., 1., 6.],
    'iterations': [120, 40],
    'shrink_factors': [2, 1],
    'smooth_sigmas': [4, 2],
    'step': 1.0,
    'deform_blocksize': [256,]*3,
    'cluster_extra': ['',],
}
