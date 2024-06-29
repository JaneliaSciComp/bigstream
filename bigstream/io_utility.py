import logging
import nrrd
import numcodecs as codecs
import numpy as np
import os
import zarr

from tifffile import TiffFile


logger = logging.getLogger(__name__)


def create_dataset(container_path, container_subpath, shape, chunks, dtype,
                   data=None, overwrite=False,
                   compressor=None,
                   **kwargs):
    try:
        real_container_path = os.path.realpath(container_path)
        path_comps = os.path.splitext(container_path)

        container_ext = path_comps[1]

        if container_ext == '.zarr':
            store = real_container_path
        else:
            store = zarr.N5Store(real_container_path)
        if container_subpath:
            logger.info(f'Create dataset {container_path}:{container_subpath} ' +
                        f'compressor={compressor}, {kwargs}')
            container_root = zarr.open_group(store=store, mode='a')
            codec = (None if compressor is None 
                     else codecs.get_codec(dict(id=compressor)))
            if data is None and overwrite:
                dataset = container_root.create_dataset(
                    container_subpath,
                    shape=shape,
                    chunks=chunks,
                    dtype=dtype,
                    overwrite=overwrite,
                    compressor=codec,
                    data=data)
            else:
                dataset = container_root.require_dataset(
                    container_subpath,
                    shape=shape,
                    chunks=chunks,
                    dtype=dtype,
                    overwrite=overwrite,
                    compressor=codec,
                    data=data)
            # set additional attributes
            dataset.attrs.update((k, v) for k,v in kwargs.items() if v)
            return dataset
        else:
            logger.info(f'Create root array {container_path} {kwargs}')
            zarr_data = zarr.open(store=store, mode='a',
                                  shape=shape, chunks=chunks)
            # set additional attributes
            zarr_data.attrs.update((k, v) for k,v in kwargs.items() if v)
            return zarr_data
    except Exception as e:
        logger.error(f'Error creating a dataset at {container_path}:{container_subpath}: {e}')
        raise e


def open(container_path, subpath, block_coords=None):
    real_container_path = os.path.realpath(container_path)
    path_comps = os.path.splitext(container_path)

    container_ext = path_comps[1]

    if container_ext == '.nrrd':
        logger.info(f'Open nrrd {container_path} ({real_container_path})')
        return _read_nrrd(real_container_path, block_coords=block_coords)
    elif container_ext == '.tif' or container_ext == '.tiff':
        logger.info(f'Open tiff {container_path} ({real_container_path})')
        return _read_tiff(real_container_path, block_coords=block_coords)
    elif container_ext == '.npy':
        im = np.load(real_container_path)
        bim = im[block_coords] if block_coords is not None else im
        return bim, {}
    elif (container_ext == '.n5'
          or os.path.exists(f'{real_container_path}/attributes.json')):
        logger.info(f'Open N5 {container_path} ({real_container_path})')
        return _open_zarr(real_container_path, subpath, data_store_name='n5',
                          block_coords=block_coords)
    elif container_ext == '.zarr':
        logger.info(f'Open Zarr {container_path} ({real_container_path})')
        return _open_zarr(real_container_path, subpath, data_store_name='zarr',
                          block_coords=block_coords)
    else:
        logger.error(f'Cannot handle {container_path} ' +
                     f'({real_container_path}) {subpath}')
        return None, {}


def get_voxel_spacing(attrs):
    pr = None
    if attrs.get('pixelResolution'):
        pr_attr = attrs.get('pixelResolution')
        if type(pr_attr) == list:
            pr = np.array(pr_attr)
        elif type(pr_attr) == dict:
            if pr_attr.get('dimensions'):
                pr = np.array(pr_attr['dimensions'])

    if pr is not None:
        if attrs.get('downsamplingFactors'):
            ds = np.array(attrs['downsamplingFactors'])
        else:
            ds = 1

        return pr * ds
    else:
        return None


def read_attributes(container_path, subpath):
    real_container_path = os.path.realpath(container_path)
    path_comps = os.path.splitext(container_path)

    container_ext = path_comps[1]

    if container_ext == '.nrrd':
        logger.info(f'Read nrrd attrs {container_path} ({real_container_path})')
        return _read_nrrd_attrs(container_path)
    elif container_ext == '.n5' or os.path.exists(f'{container_path}/attributes.json'):
        logger.info(f'Read N5 attrs {container_path} ({real_container_path})')
        return _open_zarr_attrs(real_container_path, subpath, data_store_name='n5')
    elif container_ext == '.zarr':
        logger.info(f'Read Zarr attrs {container_path} ({real_container_path})')
        return _open_zarr_attrs(real_container_path, subpath, data_store_name='zarr')
    elif container_ext == '.tif' or container_ext == '.tiff':
        logger.info(f'Read TIFF attrs {container_path} ({real_container_path})')
        return _read_tiff_attrs(container_path)
    else:
        logger.error(f'Cannot read attributes for {container_path}:{subpath}')
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
        logger.error(f'Error opening {data_path} : {data_subpath}: {e}')
        raise e


def _open_zarr_attrs(data_path, data_subpath, data_store_name=None):
    try:
        data_container = zarr.open(store=_get_data_store(data_path,
                                                         data_store_name),
                                   mode='r')
        a = data_container[data_subpath] if data_subpath else data_container
        dict = a.attrs.asdict()
        dict.update({'dataType': a.dtype,
                     'dimensions': a.shape,
                     'blockSize': a.chunks})
        logger.info(f'{data_path}:{data_subpath} attrs: {dict}')
        return dict
    except Exception as e:
        logger.error(f'Error opening {data_path} : {data_subpath}, {e}')
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
        return img, _get_tiff_attrs(tif_array)


def _read_tiff_attrs(input_path):
    with TiffFile(input_path) as tif:
        tif_store = tif.aszarr()
        tif_array = zarr.open(tif_store)
        return _get_tiff_attrs(tif_array)

def _get_tiff_attrs(tif_array):
    dict = tif_array.attrs.asdict()
    dict.update({
        'dataType': tif_array.dtype,
        'dimensions': tif_array.shape,
    })
    return dict

def _read_nrrd(input_path, block_coords=None):
    im, dict = nrrd.read(input_path)
    return im[block_coords] if block_coords is not None else im, dict


def _read_nrrd_attrs(input_path):
    return nrrd.read_header(input_path)
