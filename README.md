# BigStream
---

![warp](resources/warp_interpolation.gif)

BigStream is a library of tools for 3D registration of images too large to fit into memory and/or too large to register in a single (multi-threaded) process. BigStream automates chunking of the alignment problem into overlapping blocks, distributes the blocks to independent workers running in parallel, and stitches the results into a single smooth transform. BigStream includes global affine, piecewise affine, and piecewise deformable alignments; it also includes tools for finding feature points of interest, applying transforms, and inverting transforms. The tools can be used individually to construct custom workflows, but pipelines are also provided for specific alignment problems.

## Installation
---
> pip install bigstream

## Branches 
---
The `master` branch is the most up to date version. With minimal modification it can be used in any distributed environment supported by [dask-jobqueue](https://jobqueue.dask.org/en/latest/ "dask-jobqueue").

The `prototype` branch is a record of the first implementation, built using a different software stack. Rather than DASK, it handles blocking, distribution, and stiching manually. The primary workflow can be seen in the `stream.sh` script. This version was built specifically for LSF clusters and minimal modification of the `submit` function in `stream.sh` would be required for using this version on other clusters. 

## Usage
---

The simplest way to use BigStream, which will cover a large portion of the use cases, is to call an existing pipeline. For example, to use the global affine --> piecewise affine --> piecewise deformable method used to align multiround multiFISH datasets:

```python
from bigstream import pipelines

transform = pipelines.multifish_registration_pipeline(
    fixed_file_path=fixed_path,
    fixed_lowres_dataset=fixed_lowres,
    fixed_highres_dataset=fixed_highres,
    moving_file_path=moving_path,
    moving_lowres_dataset=moving_lowres,
    moving_highres_dataset=moving_highres,
    transform_write_path=forward_write_path,
    inv_transform_write_path=inverse_write_path,
    scratch_directory=scratch_path,
    global_affine_params={},
    local_affine_params={},
    deform_params={},
)
```

All inputs other than the last three are strings. This pipeline assumes the input image format is [N5](https://zarr.readthedocs.io/en/stable/api/n5.html "N5 documentation") (can be written using [Zarr](https://zarr.readthedocs.io/en/stable/index.html "Zarr documentation")), so `fixed_file_path` and `moving_file_path` are paths to N5 files.

Roughly speaking this pipeline executes three steps:
1. global affine
1. piecewise affine
1. piecewise deformation

Affine alignments are done at a lower resolution than deformable alignments. `fixed_lowres_dataset` is the path to the low resolution scale level in the fixed N5 file, the other variable names should be clear.

The pipeline results in two outputs: a forward transform and an inverse transform, both stored as N5 files on disk. To construct these outputs, and because it is assumed that the input images are very large, several objects are written to disk for temporary storage - so a scratch directory must also be provided. The temporary files will be removed by he pipeline itself when they are no longer needed.

Finaly, the params dictionaries allow knowledgeable users to modify parameters associated with the individual steps of the pipeline.

## Tutorials
---

For those interested in using the modular components of BigStream, ipython notebooks are provided walking you through the components that make up a pipeline. For example, [here is the tutorial for the multifish_registration_pipeline](https://github.com/GFleishman/bigstream/blob/master/notebooks/bigstream_intro_tutorial.ipynb "multifish registration tutorial").

## Tutorial data
---

Included in the repository are several datasets useful for testing and demonstrating functionality in the tutorials. 

## Issues
---
Please use the github issue tracker on this page for issues of any kind.
