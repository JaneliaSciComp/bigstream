import numpy as np
import zarr


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


def open(n5_path, n5_subpath):
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
