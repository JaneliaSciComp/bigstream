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
    The easifish registration pipeline.
    Runs 4 registration steps: (1) global affine using ransac on feature points
    extracted with a blob detector, (2) global affine refinement using gradient
    descent of an image matching metric (a function of all image voxels),
    (3) local affine refinement using ransac on feature points again, and
    (4) local deformable refinement using gradient descent of an image
    matching metric and a cubic b-spline parameterized deformation.

    All four steps can be totally customized through their keyword arguments.
    This function assumes steps (1) and (2) can be done in memory on low
    resolution data and steps (3) and (4) must be done in distributed memory
    on high resolution data. Thus, steps (3) and (4) are run on overlapping
    blocks in parallel across distributed resources.

    Parameters
    ----------
    fix_lowres : ndarray
        the fixed image, at lower resolution

    fix_highres : zarr array
        the fixed image, at higher resolution

    mov_lowres : ndarray
        the moving image, at lower resolution

    mov_highres : zarr array
        the moving image, at higher resolution

    fix_lowres_spacing : 1d array
        the spacing in physical units (e.g. mm or um) between
        voxels of the fix_lowres image

    fix_highres_spacing : 1d array
        the spacing in physical units (e.g. mm or um) between
        voxels of the fix_highres image

    mov_lowres_spacing : 1d array
        the spacing in physical units (e.g. mm or um) between
        voxels of the mov_lowres image

    mov_highres_spacing : 1d array
        the spacing in physical units (e.g. mm or um) between
        voxels of the mov_highres image

    blocksize : iterable
        the shape of blocks in voxels

    write_directory : str
        a folder on disk where outputs will be written
        this pipeline will create the following files:
            affine.mat : the global affine transform
            affine.npy : affine.mat applied to mov_lowres
            deform.zarr : the local transform (a vector field)
            deformed.zarr : [affine.mat, deform.zarr] applied to mov_highres

    global_ransac_kwargs : dict
        Any arguments you would like to pass to the global instance of
        bigstream.align.feature_point_ransac_affine_align. See the
        docstring for that function for valid parameters.
        default : {'blob_sizes':[6, 20]}

    global_affine_kwargs : dict
        Any arguments you would like to pass to the global instance of
        bigstream.align.affine_align. See the docstring for that function
        for valid parameters.
        default : {'shrink_factors':(2,),
                   'smooth_sigmas':(2.5,),
                   'optimizer_args':{
                       'learningRate':0.25,
                       'minStep':0.,
                       'numberOfIterations':400,
                   }}

    local_ransac_kwargs : dict
        Any arguments you would like to pass to the local instances of
        bigstream.align.feature_point_ransac_affine_align. See the
        docstring for that function for valid parameters.
        default : {'blob_sizes':[6, 20]}

    local_deform_kwargs : dict
        Any arguments you would like to pass to the local instances of
        bigstream.align.deformable_align. See the docstring for that function
        for valid parameters.
        default : {'smooth_sigmas':(0.25,),
                   'control_point_spacing':50.0,
                   'control_point_levels':(1,),
                   'optimizer_args':{
                       'learningRate':0.25,
                       'minStep':0.,
                       'numberOfIterations':25,
                   }}

    cluster_kwargs : dict
        Arguments passed to ClusterWrap.cluster
        If working with an LSF cluster, this will be
        ClusterWrap.janelia_lsf_cluster. If on a workstation
        this will be ClusterWrap.local_cluster. This is how
        distribution parameters are specified.

    cluster : dask cluster object
        Only set if you have constructed your own static cluster. The default
        behavior is to construct a cluster for the duraction of this pipeline,
        then close it when the function is finished. However, if you provide
        a cluster through this keyword then all distributed computations will
        occur on that cluster

    Returns
    -------
    affine : 4x4 ndarray
        The global affine transform

    deform : zarr array
        The local displacement vector field (assumed too large for memory)

    aligned : zarr array
        mov_highres resampled to match fix_highres, i.e. application of
        [affine, deform] to mov_highres (assumed too large for memory)
               
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

