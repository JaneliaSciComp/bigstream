import numpy as np
from ClusterWrap import cluster as cluster_constructor
from bigstream.align import alignment_pipeline
from bigstream.transform import apply_transform
from bigstream.piecewise_align import distributed_piecewise_alignment_pipeline
from bigstream.piecewise_transform import distributed_apply_transform


def easifish_registration_pipeline(
    fix_lowres,
    fix_highres,
    mov_lowres,
    mov_highres,
    fix_lowres_spacing,
    fix_highres_spacing,
    mov_lowres_spacing,
    mov_highres_spacing,
    blocksize,
    write_directory,
    global_ransac_kwargs={},
    global_affine_kwargs={},
    local_ransac_kwargs={},
    local_deform_kwargs={},
    cluster_kwargs={},
    cluster=None,
):
    """
    """

    # ensure lowres datasets are in memory
    fix_lowres = fix_lowres[...]
    mov_lowres = mov_lowres[...]

    # configure global affine alignment at lowres
    a = {'blob_sizes':[6, 20]}
    b = {'shrink_factors':(2,),
         'smooth_sigmas':(2.5,),
         'optimizer_args':{
             'learningRate':0.25,
             'minStep':0.,
             'numberOfIterations':400,
         },
    }
    steps = [
        ('ransac', {**a, **global_ransac_kwargs}),
        ('affine', {**b, **global_affine_kwargs}),
    ]

    # run global affine alignment at lowres
    affine = alignment_pipeline(
        fix_lowres, mov_lowres,
        fix_lowres_spacing, mov_lowres_spacing,
        steps=steps,
    )

    # apply global affine and save result
    aligned = apply_transform(
        fix_lowres, mov_lowres,
        fix_lowres_spacing, mov_lowres_spacing,
        transform_list=[affine,],
    )
    np.savetxt(f'{write_directory}/affine.mat', affine)
    np.save(f'{write_directory}/affine.npy', aligned)

    # configure local deformable alignment at highres
    a = {'blob_sizes':[6, 20]}
    b = {'smooth_sigmas':(0.25,),
         'control_point_spacing':50.0,
         'control_point_levels':(1,),
         'optimizer_args':{
             'learningRate':0.25,
             'minStep':0.,
             'numberOfIterations':25,
         },
    }
    c = {'ncpus':1,
         'threads':1,
         'min_workers':10,
         'max_workers':100,
         'config':{
             'distributed.worker.memory.target':0.9,
             'distributed.worker.memory.spill':0.9,
             'distributed.worker.memory.pause':0.9,
         },
    }
    steps = [
        ('ransac', {**a, **local_ransac_kwargs}),
        ('deform', {**b, **local_deform_kwargs}),
    ]

    # closure for distributed functions
    alignment = lambda x: distributed_piecewise_alignment_pipeline(
        fix_highres, mov_highres,
        fix_highres_spacing, mov_highres_spacing,
        steps=steps,
        blocksize=blocksize,
        static_transform_list=[affine,],
        write_path=write_directory + '/deform.zarr',
        cluster=x,
    )
    resample = lambda x: distributed_apply_transform(
        fix_highres, mov_highres,
        fix_highres_spacing, mov_highres_spacing,
        transform_list=[affine, deform],
        blocksize=blocksize,
        write_path=write_directory + '/deformed.zarr',
        cluster=x,
    )

    # if no cluster was given, make one then run on it
    if cluster is None:
        with cluster_constructor(**{**c, **cluster_kwargs}) as cluster:
            deform = alignment(cluster)
            aligned = resample(cluster)
    # otherwise, use the cluster that was given
    else:
        deform = alignment(cluster)
        aligned = resample(cluster)

    return affine, deform, aligned

