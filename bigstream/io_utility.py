import dask.array as da
import logging
import nrrd
import numcodecs as codecs
import numpy as np
import os
import re
import zarr
import traceback

from ome_zarr_models.v04.image import ImageAttrs
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


def open(container_path, subpath, block_coords=None, container_type=None):
    real_container_path = os.path.realpath(container_path)
    path_comps = os.path.splitext(container_path)

    container_ext = path_comps[1]

    if container_ext == '.nrrd' or container_type == 'nrrd':
        logger.info(f'Open nrrd {container_path} ({real_container_path})')
        return _read_nrrd(real_container_path, block_coords=block_coords)
    elif container_ext == '.tif' or container_ext == '.tiff' or container_type == 'tif':
        logger.info(f'Open tiff {container_path} ({real_container_path})')
        return _read_tiff(real_container_path, block_coords=block_coords)
    elif container_ext == '.npy' or container_type == 'npy':
        im = np.load(real_container_path)
        bim = im[block_coords] if block_coords is not None else im
        return bim, {}
    elif (container_ext == '.n5'
          or os.path.exists(f'{real_container_path}/attributes.json')
          or container_type == 'n5'):
        logger.info(f'Open N5 {container_path} ({real_container_path})')
        return _open_zarr(real_container_path, subpath, data_store_name='n5',
                          block_coords=block_coords)
    elif container_ext == '.zarr' or container_type == 'zarr':
        logger.info(f'Open Zarr {container_path} ({real_container_path})')
        return _open_zarr(real_container_path, subpath, data_store_name='zarr',
                          block_coords=block_coords)
    else:
        logger.error(f'Cannot handle {container_path} ' +
                     f'({real_container_path}) {subpath}')
        return None, {}


def get_voxel_spacing(attrs: dict):
    pr = None
    if attrs.get('coordinateTransformations'):
        scale_metadata = list(filter(lambda t: t.type == 'scale', attrs['coordinateTransformations']))
        if len(scale_metadata) > 0:
            # return voxel spacing as [dx, dy, dz]
            pr = np.array(scale_metadata[0].scale[2:][::-1])

    if attrs.get('pixelResolution'):
        pr_attr = attrs.get('pixelResolution')
        if type(pr_attr) is list:
            pr = np.array(pr_attr)
        elif type(pr_attr) is dict:
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


def read_attributes(container_path, subpath, container_type=None):
    real_container_path = os.path.realpath(container_path)
    path_comps = os.path.splitext(container_path)

    container_ext = path_comps[1]

    if container_ext == '.nrrd' or container_type == 'nrrd':
        logger.info(f'Read nrrd attrs {container_path} ({real_container_path})')
        return _read_nrrd_attrs(container_path)
    elif (container_ext == '.n5' or os.path.exists(f'{container_path}/attributes.json')
          or container_type == 'n5'):
        logger.info(f'Read N5 attrs {container_path} ({real_container_path})')
        return _open_zarr_attrs(real_container_path, subpath, data_store_name='n5')
    elif container_ext == '.zarr' or container_type == 'zarr':
        logger.info(f'Read Zarr attrs {container_path} ({real_container_path})')
        return _open_zarr_attrs(real_container_path, subpath, data_store_name='zarr')
    elif (container_ext == '.tif' or container_ext == '.tiff'
          or container_type == 'tif'):
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


def _open_zarr(data_path, data_subpath, data_store_name=None,
               data_subpath_pattern=None,
               block_coords=None):
    try:
        zarr_container_path, zarr_subpath = _adjust_data_paths(data_path, data_subpath)
        data_container = zarr.open(store=_get_data_store(zarr_container_path,
                                                         data_store_name),
                                   mode='r')
        data_container_attrs = data_container.attrs.asdict()

        if _is_ome_zarr(data_container_attrs):
            return _open_ome_zarr(data_container, data_container_attrs, zarr_subpath, 
                                  data_subpath_pattern=data_subpath_pattern,
                                  block_coords=block_coords)
        else:
            a = data_container[zarr_subpath] if zarr_subpath else data_container
            ba = a[block_coords] if block_coords is not None else a
            return ba, a.attrs.asdict()

    except Exception as e:
        logger.error(f'Error opening {data_path} : {data_subpath}: {traceback.format_exc(e)}')
        raise e


def _is_ome_zarr(data_container_attrs: dict | None) -> bool:
    if data_container_attrs is None:
        return False

    # test if multiscales attribute exists - if it does assume OME-ZARR
    multiscales = data_container_attrs.get('multiscales', [])
    return not (multiscales == [])


def _open_ome_zarr(data_container, data_container_attrs, data_subpath,
                   data_subpath_pattern=None,
                   block_coords=None):
    ome_zarr_metadata = ImageAttrs(**data_container_attrs)
    multiscale_metadata = ome_zarr_metadata.multiscales[0]
    dataset_metadata = None

    dataset_subpath_arg = data_subpath if data_subpath is not None else ''
    dataset_comps = [c for c in dataset_subpath_arg.split('/') if c]

    # lookup the dataset by path
    for ds in multiscale_metadata.datasets:
        current_ds_path_comps = [c for c in ds.path.split('/') if c]
        if (len(current_ds_path_comps) < len(dataset_comps) and
            tuple(current_ds_path_comps) == tuple(dataset_comps[-len(current_ds_path_comps):])):
            # found a dataset that has a path matching a suffix of the data_subpath arg
            dataset_metadata = ds
            # drop the matching suffix
            dataset_comps = dataset_comps[-len(current_ds_path_comps):]
            logger.debug((
                f'Found dataset: {dataset_metadata.path}, '
                f'remaining components: {dataset_comps}'
            ))
            break

    dataset_comps_pattern = (list(data_subpath_pattern) 
                             if data_subpath_pattern else [])
    ch = None
    timeindex = None
    dataset_axes = multiscale_metadata.axes

    for comp_index, comp in enumerate(dataset_comps_pattern):
        if (comp == 't' and
            any(a.type == 'time' for a in dataset_axes)):
            # if the time is present in the dataset subpath selector
            # and it is in the nd-array too - get the index to be processed
            logger.debug(f'Extract timeindex from {dataset_comps[comp_index]}')
            timeindex = _extract_numeric_comp(dataset_comps[comp_index])
        elif (comp == 'c' and
            any(a.type == 'channel' for a in dataset_axes)):
            # if the channel is present in the dataset subpath selector
            # and the nd-array is a multi-channel array get the channel to be processed
            logger.debug(f'Extract channel from {dataset_comps[comp_index]}')
            ch = _extract_numeric_comp(dataset_comps[comp_index])
        elif comp == 's' and dataset_metadata is None:
            # scale selector is in the path and dataset was not found
            # using the existing datasets paths
            logger.debug(f'Extract dataset index from {dataset_comps[comp_index]}')
            dataset_index = _extract_numeric_comp(dataset_comps[comp_index])
            dataset_metadata = multiscale_metadata.datasets[dataset_index]
        else:
            # this dataset component can be anything
            continue

    if dataset_metadata is None:
        dataset_metadata = multiscale_metadata.datasets[0]
        logger.info(f'No dataset was found so far - use the first one: {dataset_metadata.path}')

    dataset_path = dataset_metadata.path

    a = data_container[dataset_path]
    # a is potentially a 5-dim array: [timepoint?, channel?, z, y, x]
    if block_coords is not None:
        ba = _get_array_selector(dataset_axes, timeindex, ch)(a)[block_coords]
    else:
        ba = _get_array_selector(dataset_axes, timeindex, ch)(a)
    data_container_attrs.update({
        'dataset_path': dataset_path,
        'axes': dataset_axes,
        'timeindex': timeindex,
        'channel': ch,
        'coordinateTransformations': dataset_metadata.coordinateTransformations
    })
    return ba, data_container_attrs


def _get_array_selector(axes, timeindex, ch):
    selector = []
    selection_exists = False
    for a in axes:
        if a.type == 'time':
            if timeindex is not None:
                selector.append(timeindex)
                selection_exists = True
            else:
                selector.append(slice(None, None))
        elif a.type == 'channel':
            if ch is not None:
                selector.append(ch)
                selection_exists = True
            else:
                selector.append(slice(None, None))
        else:
            selector.append(slice(None, None))

    return lambda a: a.get_basic_selection(tuple(selector)) if selection_exists else a


def _open_zarr_attrs(data_path, data_subpath, data_store_name=None):
    try:
        zarr_container_path, zarr_subpath = _adjust_data_paths(data_path, data_subpath)
        data_container = zarr.open(store=_get_data_store(zarr_container_path,
                                                         data_store_name),
                                   mode='r')
        data_container_attrs = data_container.attrs.asdict()

        if _is_ome_zarr(data_container_attrs):
            _, dataset_attrs = _open_ome_zarr(data_container, data_container_attrs, zarr_subpath)
        else:
            a = data_container[zarr_subpath] if zarr_subpath else data_container
            dataset_attrs = a.attrs.asdict()
            dataset_attrs.update({'dataType': a.dtype,
                         'dimensions': a.shape,
                         'blockSize': a.chunks})
            logger.info(f'{data_path}:{data_subpath} attrs: {dataset_attrs}')
        
        return dataset_attrs
    except Exception as e:
        logger.error(f'Error opening {data_path} : {data_subpath}, {e}')
        raise e


def _adjust_data_paths(data_path, data_subpath):
    """
    This methods adjusts the container and dataset paths such that
    the container paths always contains a .attrs file
    """
    dataset_path_arg = data_subpath if data_subpath is not None else ''
    dataset_comps = [c for c in dataset_path_arg.split('/') if c]
    dataset_comps_index = 0

    # Look for the first subpath that containes .zattrs file
    while dataset_comps_index < len(dataset_comps):
        container_subpath = '/'.join(dataset_comps[0:dataset_comps_index])
        container_path = f'{data_path}/{container_subpath}'
        if os.path.exists(f'{container_path}/.zattrs'):
            break
        dataset_comps_index = dataset_comps_index + 1

    appended_container_path = '/'.join(dataset_comps[0:dataset_comps_index])
    container_path = f'{data_path}/{appended_container_path}'
    new_subpath = '/'.join(dataset_comps[dataset_comps_index:])

    return container_path, new_subpath


def _get_data_store(data_path, data_store_name):
    if data_store_name is None or data_store_name == 'n5':
        return zarr.N5Store(data_path)
    else:
        return zarr.DirectoryStore(data_path)



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


def _extract_numeric_comp(v):
    match = re.match(r'^(\D*)(\d+)$', v)
    if match:
        return int(match.groups()[1])
    else:
        raise ValueError(f'Invalid component: {v}')
