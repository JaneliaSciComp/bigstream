import os
import nrrd
import numpy as np
import zarr

from skimage import io


def create_dataset(n5_path, n5_subpath, shape, chunks, dtype,
                   data=None,
                   **kwargs):
    try:
        n5_store = zarr.N5Store(n5_path)
        if n5_subpath:
            print('Create dataset', n5_path, n5_subpath)
            n5_root = zarr.open_group(store=n5_store, mode='a')
            dataset = n5_root.require_dataset(
                n5_subpath,
                shape=shape,
                chunks=chunks,
                dtype=dtype,
                data=data)
            # set additional attributes
            dataset.attrs.update(**kwargs)
            return dataset
        else:
            print('Create root array', n5_path)
            return zarr.open(store=n5_store, mode='a',
                             shape=shape, chunks=chunks)
    except Exception as e:
        print('Error creating a dataset at', n5_path, n5_subpath, e)
        raise e


def open(container_path, subpath):
    path_comps = os.path.splitext(container_path)

    container_ext = path_comps[1]

    if container_ext == '.nrrd':
        return nrrd.read(container_path)
    elif container_ext == '.tif' or container_ext == '.tiff':
        im = io.imread(container_path)
        return im, {}
    elif container_ext == '.npy':
        im = np.load(container_path)
        return im, {}
    elif container_ext == '.n5' or os.path.exists(f'{container_path}/attributes.json'):
        return _open_n5(container_path, subpath)
    else:
        print('Cannot handle', container_path, subpath)
        return None, {}


def _open_n5(n5_path, n5_subpath):
    try:
        n5_container = zarr.open(store=zarr.N5Store(n5_path), mode='r')
        a = n5_container[n5_subpath] if n5_subpath else n5_container
        return a, a.attrs.asdict()
    except Exception as e:
        print('Error opening', n5_path, n5_subpath, e)
        raise e


def get_voxel_spacing(attrs):
    if (attrs.get('downsamplingFactors')):
        voxel_spacing = (np.array(attrs['pixelResolution']) * 
                         np.array(attrs['downsamplingFactors']))
    else:
        voxel_spacing = np.array(attrs['pixelResolution']['dimensions'])
    return voxel_spacing[::-1] # put in zyx order
