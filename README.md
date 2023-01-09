# BigStream
---

![warp](resources/warp_interpolation.gif)

BigStream is a library of tools for 3D registration including images too large to fit into memory and/or too large to register in a single (multi-threaded) process. BigStream can automate chunking of the alignment problem into overlapping blocks, distributes the blocks to independent workers running in parallel, and stitches the results into a single smooth transform. BigStream includes global affine, piecewise affine, and piecewise deformable alignments; it also includes tools for finding feature points of interest, applying transforms, and inverting transforms. The tools can be used individually to construct custom workflows, but pipelines are also provided for specific alignment problems.

## Installation
---
> pip install bigstream

## Updates
---
BigStream has just undergone a major change to prepare for releasing v1.0.0.
Improvements include:
* Better utilization of dask primitives for faster and more robust distribution
* More accurate linear blending of transform blocks for smoother transforms
* A single python function to run the easi-fish registration pipeline directly
* New alignment functions: feature point ransac affine, random affine search
* Full access to the SimpleITK ImageRegistrationMethod options for almost all alignments
* Better source code design providing easy modular access at many points in the funciton hierarchy

Development wise, bigstream has reached a milestone in stability and I can now focus on documenting, teaching, and maintaining the package instead of building core functionality.

## Branches 
---
The `master` branch is the most up to date version. With minimal modification it can be used in any distributed environment supported by [dask-jobqueue](https://jobqueue.dask.org/en/latest/ "dask-jobqueue").

The `prototype` branch is a record of the first implementation, built using a different software stack. Rather than DASK, it handles blocking, distribution, and stiching manually. The primary workflow can be seen in the `stream.sh` script. This version was built specifically for LSF clusters and minimal modification of the `submit` function in `stream.sh` would be required for using this version on other clusters. 

## Tutorials and Tutorial Data
---

For those interested in using the modular components of BigStream, ipython notebooks are provided walking you through the components that make up a pipeline. For example, [here is the tutorial for the multifish_registration_pipeline](https://github.com/GFleishman/bigstream/blob/master/notebooks/bigstream_intro_tutorial.ipynb "multifish registration tutorial"). Included in the repository are several datasets useful for testing and demonstrating functionality in the tutorials. 

## Issues
---
Please use the github issue tracker on this page for issues of any kind.

## Usage Examples
---

The tutorials are a more in depth way to learn the package, but here are some usage examples.

Bigstream is flexible toolkit that can be used in many different ways. I'll discuss some of them in order of "largest" (pipelines that chain together many steps) to "smallest" (individual functions).

### Running the easi-fish registration pipeline:
```python
from bigstream.application_pipelines import easifish_registration_pipeline

# load all the input data
fix_lowres = """ load a lowres version of your fixed image """
fix_highres = """ (lazy, e.g. zarr) load a highres version of your fixed image """
mov_lowres = """ load a lowres version of your moving image """
mov_highres = """ (lazy, e.g. zarr) load a highres version of your moving image """
fix_lowres_spacing = """ voxel spacing of lowres fixed image """
fix_highres_spacing = """ voxel spacing of highres fixed image """
mov_lowres_spacing = """ voxel spacing of lowres moving image """
mov_highres_spacing = """ voxel spacing of highres moving images """
blocksize = [128, 128, 128]  # size of individual alignment blocks in voxels
write_directory = './somewhere_to_save_transforms_and_images'

affine, deform, aligned = easifish_registration_pipeline(
    fix_lowres, fix_highres, mov_lowres, mov_highres,
    fix_lowres_spacing, fix_highres_spacing,
    mov_lowres_spacing, mov_highres_spacing,
    blocksize=blocksize,
    write_directory=write_directory,
)
```

This pipeline runs 4 steps:
* global affine based on feature point ransac
* global affine refinement based on gradient descent on image intensities
* local affine based on feature point ransac
* local deform based on gradient descent on image intensities

These four steps can be customized using these optional parameters to the pipeline:

```python
global_ransac_kwargs
global_affine_kwargs
local_ransac_kwargs
local_deform_kwargs
```

See the docstring for `easifish_registration_pipeline` for more details.

### Running custom alignment pipelines
```python
from bigstream.align import alignment_pipeline
from bigstream.piecewise_align import distributed_piecewise_alignment_pipeline
from bigstream.transform import apply_transform
from bigstream.piecewise_transform import distributed_apply_transform

# load all the input data
fix_lowres = """ load a lowres version of your fixed image """
fix_highres = """ (lazy, e.g. zarr) load a highres version of your fixed image """
mov_lowres = """ load a lowres version of your moving image """
mov_highres = """ (lazy, e.g. zarr) load a highres version of your moving image """
fix_lowres_spacing = """ voxel spacing of lowres fixed image """
fix_highres_spacing = """ voxel spacing of highres fixed image """
mov_lowres_spacing = """ voxel spacing of lowres moving image """
mov_highres_spacing = """ voxel spacing of highres moving images """
blocksize = [128, 128, 128]  # size of individual alignment blocks in voxels
write_directory = './somewhere_to_save_transforms_and_images'

# construct a global alignment pipeline
rigid_kwargs = {}  # see bigstream.align.affine_align docstring for options
affine_kwargs = {} # see bigstream.align.affine_align docstring for options
deform_kwargs = {} # see bigstream.align.deformable_align docstring for options
steps = [('rigid', rigid_kwargs), ('affine', affine_kwargs), ('deform', deform_kwargs)]

# run the alignment
global_transform = alignment_pipeline(
    fix_lowres, mov_lowres,
    fix_lowres_spacing,
    mov_lowres_spacing,
    steps=steps,
)

# apply transform
global_aligned = apply_transform(
    fix_lowres, mov_lowres,
    fix_lowres_spacing, mov_lowres_spacing,
    transform_list=[global_transform,],
)

# construct a local alignment pipeline to refine global result
rigid_kwargs = {}  # see bigstream.align.affine_align docstring for options
affine_kwargs = {} # see bigstream.align.affine_align docstring for options
deform_kwargs = {} # see bigstream.align.deformable_align docstring for options
steps = [('rigid', rigid_kwargs), ('affine', affine_kwargs), ('deform', deform_kwargs)]
blocksize = [128, 128, 128]

# run the alignment
local_transform = distributed_piecewise_alignment_pipeline(
    fix_highres, mov_highres
    fix_highres_spacing, mov_highres_spacing,
    steps=steps,
    blocksize=blocksize,
    static_transform_list=[global_transform,]
    write_path='./deform.zarr',
    cluster_kwargs={## params to control your cluster},
)

# apply the transforms
local_aligned = piecewise_apply_transform(
    fix_highres, mov_highres,
    fix_highres_spacing, mov_highres_spacing,
    transform_list=[global_transform, local_transform],
    blocksize=blocksize,
    write_path='./deformed.zarr',
    cluster_kwargs={## params to control your cluster},
)
```

For details, see the docstrings for:
* `bigstream.align.alignment_pipeline`
* `bigstream.piecewise_align.distributed_piecewise_alignment_pipeline`
* `bigstream.transform.apply_transform`
* `bigstream.piecewise_transform.distributed_apply_transform`

### Using individual functions
```python
from bigstream.align import affine_align
from bigstream.transform import apply_transform

# load all the input data
fix = """ load your fixed image """
mov = """ load your moving image """
fix_spacing = """ voxel spacing of fixed image """
mov_spacing = """ voxel spacing of moving image """

# run an affine alignment
affine = affine_align(
    fix, mov,
    fix_spacing, mov_spacing,
    # TONS of customization and efficiency options available here
)

# apply transform
aligned = apply_transform(
    fix, mov,
    fix_spacing, mov_spacing,
    transform_list=[affine,],  # this list can be arbitrarily long
)
```

The docstrings for the functions in `bigstream.align` show all the configurable options available. Any alignment function has full access to the SimpleITK ImageRegistrationMethod API.

