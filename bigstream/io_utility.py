import os
import nrrd
import numpy as np
import zarr

from tifffile import TiffFile


def create_dataset(container_path, container_subpath, shape, chunks, dtype,
                   data=None,
                   **kwargs):
    try:
        real_container_path = os.path.realpath(container_path)
        n5_store = zarr.N5Store(real_container_path)
        if container_subpath:
            print('Create dataset', container_path, container_subpath)
            container_root = zarr.open_group(store=n5_store, mode='a')
            dataset = container_root.require_dataset(
                container_subpath,
                shape=shape,
                chunks=chunks,
                dtype=dtype,
                data=data)
            # set additional attributes
            dataset.attrs.update(**kwargs)
            return dataset
        else:
            print('Create root array', container_path)
            return zarr.open(store=n5_store, mode='a',
                             shape=shape, chunks=chunks)
    except Exception as e:
        print('Error creating a dataset at', container_path, container_subpath,
              e)
        raise e


def open(container_path, subpath, block_coords=None):
    real_container_path = os.path.realpath(container_path)
    path_comps = os.path.splitext(container_path)

    container_ext = path_comps[1]

    if container_ext == '.nrrd':
        print(f'Open nrrd {container_path} ({real_container_path})',
              flush=True)
        return _read_nrrd(real_container_path, block_coords=block_coords)
    elif container_ext == '.tif' or container_ext == '.tiff':
        print(f'Open tiff {container_path} ({real_container_path})',
              flush=True)
        im = _read_tiff(real_container_path, block_coords=block_coords)
        return im, {}
    elif container_ext == '.npy':
        im = np.load(real_container_path)
        bim = im[block_coords] if block_coords is not None else im
        return bim, {}
    elif (container_ext == '.n5'
          or os.path.exists(f'{real_container_path}/attributes.json')):
        print(f'Open N5 {container_path} ({real_container_path})',
              flush=True)
        return _open_zarr(real_container_path, subpath, data_store_name='n5',
                          block_coords=block_coords)
    elif container_ext == '.zarr':
        print(f'Open Zarr {container_path} ({real_container_path})',
              flush=True)
        return _open_zarr(real_container_path, subpath, data_store_name='zarr',
                          block_coords=block_coords)
    else:
        print('Cannot handle', f'{container_path} ({real_container_path})',
              subpath)
        return None, {}


def get_voxel_spacing(attrs):
    if (attrs.get('downsamplingFactors')):
        voxel_spacing = (np.array(attrs['pixelResolution']) * 
                         np.array(attrs['downsamplingFactors']))
    else:
        voxel_spacing = np.array(attrs['pixelResolution']['dimensions'])
    return voxel_spacing[::-1] # put in zyx order


def get_dimensions(attrs):
    if (attrs.get('dimensions')):
        shape = tuple(attrs['dimensions']) 
        return shape, len(shape)
    else:
        raise Exception(f'Do not know how to read shape from {attrs}')


def get_dtype(attrs):
    if (attrs.get('dataType')):
        return attrs['dataType']
    else:
        raise Exception(f'Do not know how to read data type from {attrs}')


def read_attributes(container_path, subpath):
    real_container_path = os.path.realpath(container_path)
    path_comps = os.path.splitext(container_path)

    container_ext = path_comps[1]

    if container_ext == '.nrrd':
        print(f'Read nrrd attrs {container_path} ({real_container_path})', flush=True)
        return _read_nrrd_attrs(container_path)
    elif container_ext == '.n5' or os.path.exists(f'{container_path}/attributes.json'):
        print(f'Read N5 attrs {container_path} ({real_container_path})', flush=True)
        return _open_zarr_attrs(real_container_path, subpath, data_store_name='n5')
    elif container_ext == '.zarr':
        print(f'Read Zarr attrs {container_path} ({real_container_path})', flush=True)
        return _open_zarr_attrs(real_container_path, subpath, data_store_name='zarr')
    else:
        print('Cannot read attributes for', container_path, subpath)
        return {}


def read_block(block_coords, image=None, image_path=None, image_subpath=None):
    if block_coords is None:
        return None

    if image is not None:
        return image[block_coords]

    if image_path is not None:
        block, _ = open(image_path, image_subpath, block_coords=block_coords)
        return block

    # this is when there are block coords but no image nd-array or
    # image path were provided
    return None


def _open_zarr(data_path, data_subpath, data_store_name=None, block_coords=None):
    try:
        data_container = zarr.open(store=_get_data_store(data_path,
                                                         data_store_name),
                                   mode='r')
        a = data_container[data_subpath] if data_subpath else data_container
        ba = a[block_coords] if block_coords is not None else a
        return ba, a.attrs.asdict()
    except Exception as e:
        print(f'Error opening {data_path} : {data_subpath}', e, flush=True)
        raise e


def _open_zarr_attrs(data_path, data_subpath, data_store_name=None):
    try:
        data_container = zarr.open(store=_get_data_store(data_path,
                                                         data_store_name),
                                   mode='r')
        a = data_container[data_subpath] if data_subpath else data_container
        dict = a.attrs.asdict()
        dict.update({'dataType': a.dtype, 'dimensions': a.shape})
        return dict
    except Exception as e:
        print(f'Error opening {data_path} : {data_subpath}', e, flush=True)
        raise e


def _get_data_store(data_path, data_store_name):
    if data_store_name is None or data_store_name == 'n5':
        return zarr.N5Store(data_path)
    else:
        return data_path



def _read_tiff(input_path, block_coords=None):
    with TiffFile(input_path) as tif:
        tif_store = tif.aszarr()
        tif_array = zarr.open(tif_store)
        if block_coords is None:
            img = tif_array
        else:
            img = tif_array[block_coords]
        return img


def _read_nrrd(input_path, block_coords=None):
    im, dict = nrrd.read(input_path)
    return im[block_coords] if block_coords is not None else im, dict


def _read_nrrd_attrs(input_path):
    return nrrd.read_header(input_path)
