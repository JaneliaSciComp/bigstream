import dask.array as da
import logging
import nrrd
import numcodecs as codecs
import numpy as np
import os
import re
import zarr
import traceback

from ome_zarr_models.v04.image import (Dataset, ImageAttrs)
from tifffile import TiffFile


logger = logging.getLogger(__name__)


def create_dataset(container_path, container_subpath, shape, chunks, dtype,
                   data=None, overwrite=False,
                   for_timeindex=None, for_channel=None,
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
            logger.info((
                f'Create dataset {container_path}:{container_subpath} '
                f'compressor={compressor}, shape: {shape}, chunks: {chunks} '
                f'{kwargs} '
            ))
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
            i = 0
            resized_shape = ()
            to_resize = False
            if for_timeindex is not None:
                if shape[i] <= for_timeindex:
                    resized_shape = resized_shape + (for_timeindex + 1,)
                    to_resize = True
                else:
                    resized_shape = resized_shape + (shape[i],)
                i = i + 1
            if for_channel is not None:
                if shape[i] <= for_channel:
                    resized_shape = resized_shape + (for_channel + 1,)
                    to_resize = True
                else:
                    resized_shape = resized_shape + (shape[i],)
                i = i + 1

            if to_resize:
                while i < len(shape):
                    resized_shape = resized_shape + (shape[i],)
                    i = i + 1
                logger.info(f'Resize {container_path}:{container_subpath} to {resized_shape}')
                dataset.resize(resized_shape)

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


def get_voxel_spacing(attrs: dict):
    pr = None
    if attrs.get('coordinateTransformations'):
        scale_metadata = list(filter(lambda t: t.type == 'scale', attrs['coordinateTransformations']))
        if len(scale_metadata) > 0:
            # return voxel spacing as [time, ch, dz, dy, dx]
            pr = np.array(scale_metadata[0].scale)
        else:
            pr = None
    elif (attrs.get('downsamplingFactors')):
        pr = (np.array(attrs['pixelResolution']) * 
              np.array(attrs['downsamplingFactors']))
    elif attrs.get('pixelResolution'):
        pr_attr = attrs.get('pixelResolution')
        if type(pr_attr) is list:
            pr = np.array(pr_attr)
        elif type(pr_attr) is dict:
            if pr_attr.get('dimensions'):
                pr = np.array(pr_attr['dimensions'])
    logger.debug(f'Voxel spacing from attributes: {pr}')
    return pr


def open(container_path, subpath,
         data_timeindex=None, data_channels=None,
         block_coords=None, container_type=None):
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
        return _open_zarr(real_container_path, subpath,
                          data_timeindex=data_timeindex,
                          data_channels=data_channels,
                          data_store_name='n5',
                          block_coords=block_coords)
    elif container_ext == '.zarr' or container_type == 'zarr':
        logger.info(f'Open Zarr {container_path} ({real_container_path})')
        return _open_zarr(real_container_path, subpath,
                          data_timeindex=data_timeindex,
                          data_channels=data_channels,
                          data_store_name='zarr',
                          block_coords=block_coords)
    else:
        logger.error(f'Cannot handle {container_path} ' +
                     f'({real_container_path}) {subpath}')
        return None, {}


def prepare_attrs(container_path,
                  dataset_path,
                  axes=None,
                  coordinateTransformations=None,
                  **additional_attrs):
    if (coordinateTransformations is None or coordinateTransformations == []
        or axes is None):
        # coordinateTransformation is None or [] or no axes were provided
        return {k: v for k, v in additional_attrs.items()}
    else:
        dataset_path_comps = [c for c in dataset_path.split('/') if c]
        # take the last component of the dataset path to be the scale path
        dataset_scale_subpath = dataset_path_comps.pop()

        scales, translations = (1,) * len(axes), None
        for t in coordinateTransformations:
            if t.type == 'scale':
                scales = t.scale
            elif t.type == 'translation':
                translations = t.translation

        dataset = Dataset.build(path=dataset_scale_subpath, scale=scales, translation=translations)
        ome_attrs = {
            'multiscales': [
                {
                    'name': os.path.basename(container_path),
                    'axes': axes,
                    'datasets': (dataset.dict(exclude_none=True),),
                    'version': '0.4',
                }
            ]
        }
        ome_attrs.update(additional_attrs)
        return ome_attrs
    

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


def read_block(block_coords, image=None, image_path=None, image_subpath=None,
               image_timeindex=None, image_channel=None):
    if block_coords is None:
        return None

    if image is not None:
        return image[block_coords]

    if image_path is not None:
        block, _ = open(image_path, image_subpath,
                        data_timeindex=image_timeindex,
                        data_channels=image_channel,
                        block_coords=block_coords)
        logger.debug(f'Read {block.shape} block at {block_coords}')
        return block

    # this is when there are block coords but no image nd-array or
    # image path were provided
    return None


def _open_zarr(data_path, data_subpath, data_store_name=None,
               data_timeindex=None, data_channels=None,
               block_coords=None):
    try:
        zarr_container_path, zarr_subpath = _adjust_data_paths(data_path, data_subpath)
        data_container = zarr.open(store=_get_data_store(zarr_container_path,
                                                         data_store_name),
                                   mode='r')
        data_container_attrs = data_container.attrs.asdict()

        if _is_ome_zarr(data_container_attrs):
            return _open_ome_zarr(data_container, zarr_subpath,
                                  data_timeindex=data_timeindex,
                                  data_channels=data_channels,
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
    bioformats_layout = data_container_attrs.get("bioformats2raw.layout", None)
    multiscales = data_container_attrs.get('multiscales', [])
    return bioformats_layout == 3 or not (multiscales == [])


def _find_ome_multiscales(data_container, data_subpath):
    logger.info(f'Find OME multiscales group within {data_subpath}')
    dataset_subpath_arg = data_subpath if data_subpath is not None else ''
    dataset_comps = [c for c in dataset_subpath_arg.split('/') if c]

    dataset_comps_index = 0
    while dataset_comps_index < len(dataset_comps):
        group_subpath = '/'.join(dataset_comps[0:dataset_comps_index])
        dataset_item = data_container[group_subpath]
        dataset_item_attrs = dataset_item.attrs.asdict()
        if dataset_item_attrs.get('multiscales', []) == []:
            dataset_comps_index = dataset_comps_index + 1
        else:
            logger.debug(f'Found multiscales at {group_subpath}: {dataset_item_attrs}')
            # found a group that has attributes which contain multiscales list
            return dataset_item, '/'.join(dataset_comps[dataset_comps_index:]), dataset_item_attrs

    return None, None, {}


def _open_ome_zarr(data_container, data_subpath,
                   data_timeindex=None, data_channels=None, block_coords=None):
    multiscales_group, dataset_subpath, multiscales_attrs  = _find_ome_multiscales(data_container, data_subpath)

    if multiscales_group is None:
        a = (data_container[data_subpath]
             if data_subpath and data_subpath != '.'
             else data_container)
        ba = a[block_coords] if block_coords is not None else a
        return ba, a.attrs.asdict()

    logger.info((
        f'Open dataset {dataset_subpath}, timeindex: {data_timeindex}, '
        f'channels: {data_channels}, block_coords {block_coords} '
    ))

    dataset_comps = [c for c in dataset_subpath.split('/') if c]
    ome_metadata = ImageAttrs(**multiscales_attrs)
    multiscale_metadata = ome_metadata.multiscales[0]
    dataset_metadata = None
    # lookup the dataset by path
    for ds in multiscale_metadata.datasets:
        current_ds_path_comps = [c for c in ds.path.split('/') if c]
        logger.debug((
            f'Compare current dataset path: {ds.path} ({current_ds_path_comps}) '
            f'with {dataset_subpath} ({dataset_comps}) '
        ))
        if (len(current_ds_path_comps) <= len(dataset_comps) and
            tuple(current_ds_path_comps) == tuple(dataset_comps[-len(current_ds_path_comps):])):
            # found a dataset that has a path matching a suffix of the dataset_subpath arg
            dataset_metadata = ds
            # drop the matching suffix
            dataset_comps = dataset_comps[-len(current_ds_path_comps):]
            logger.info((
                f'Found dataset: {dataset_metadata.path}, '
                f'remaining dataset components: {dataset_comps}'
            ))
            break

    if dataset_metadata is None:
        # could not find a dataset using the subpath 
        # look at the last subpath component and get the dataset index from there
        # e.g., if the subpath looks like:
        #       '/s<n>' => datasets[n] if n < len(datasets) otherwise datasets[0]
        dataset_index_comp = dataset_comps[-1]
        logger.info(f'No dataset was found using {dataset_subpath} - try to use: {dataset_index_comp}')
        dataset_index = _extract_numeric_comp(dataset_index_comp)
        if dataset_index < len(multiscale_metadata.datasets):
            dataset_metadata = multiscale_metadata.datasets[dataset_index]
        else:
            dataset_metadata = multiscale_metadata.datasets[0]

    dataset_axes = multiscale_metadata.axes
    dataset_path = dataset_metadata.path
    logger.debug(f'Get array using array path: {dataset_path}:{data_timeindex}:{data_channels}')
    a = multiscales_group[dataset_path]
    # a is potentially a 5-dim array: [timepoint?, channel?, z, y, x]
    if block_coords is not None:
        ba = _get_array_selector(dataset_axes, data_timeindex, data_channels, block_coords)(a)
    else:
        ba = _get_array_selector(dataset_axes, data_timeindex, data_channels, None)(a)
    multiscales_attrs.update(a.attrs.asdict())
    multiscales_attrs.update({
        'dataset_path': dataset_path,
        'axes': [a.dict(exclude_none=True) for a in dataset_axes],
        'timeindex': data_timeindex,
        'channels': data_channels,
        'coordinateTransformations': dataset_metadata.coordinateTransformations
    })
    return ba, multiscales_attrs


def _get_array_selector(axes, timeindex: int | None, 
                        ch:int | list[int] | None,
                        block_coords: tuple | None):
    selector = []
    selection_exists = False
    spatial_selection = []
    for a in axes:
        if a.type == 'time':
            if timeindex is not None:
                selector.append(timeindex)
                selection_exists = True
            else:
                selector.append(slice(None, None))
        elif a.type == 'channel':
            if ch is None or ch == []:
                selector.append(slice(None, None))
            else:
                selector.append(ch)
                selection_exists = True
        else:
            spatial_selection.append(slice(None, None))

    if block_coords is not None:
        selector.extend(block_coords)
        selection_exists = True
    else:
        selector.extend(spatial_selection)

    return lambda a: a[tuple(selector)] if selection_exists else a


def _open_zarr_attrs(data_path, data_subpath, data_store_name=None):
    try:
        zarr_container_path, zarr_subpath = _adjust_data_paths(data_path, data_subpath)
        data_container = zarr.open(store=_get_data_store(zarr_container_path,
                                                         data_store_name),
                                   mode='r')
        data_container_attrs = data_container.attrs.asdict()

        if _is_ome_zarr(data_container_attrs):
            a, dataset_attrs = _open_ome_zarr(data_container, zarr_subpath)
        else:
            a = data_container[zarr_subpath] if zarr_subpath else data_container
            dataset_attrs = a.attrs.asdict()

        dataset_attrs.update({
            'dataType': a.dtype,
            'dimensions': a.shape,
            'blockSize': a.chunks,
        })
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
