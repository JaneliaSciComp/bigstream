import dask.array as da
import logging
import nrrd
import numcodecs as codecs
import numpy as np
import os
import re
import zarr
import traceback

from ome_zarr_models.v04.image import Dataset
from tifffile import TiffFile


logger = logging.getLogger(__name__)


def create_dataset(
    container_path,
    container_subpath,
    shape,
    chunks,
    dtype,
    data=None,
    overwrite=False,
    for_timeindex=None,
    for_channel=None,
    compressor=None,
    parent_attrs={},
    **dataset_attrs,
):
    """
    Create a Zarr dataset

    Parameters
    ----------
    container_path : string
        The path to the root Zarr container on disk
        If this path does not exist it will be created

    container_subpath : string
        The subpath within the root Zarr container to create the array
        If this subpath does not exist it will be created

    shape : tuple of ints
        The shape of the dataset

    chunks : tuple of ints
        The shape of individual chunks in the dataset

    dtype : python or numpy primitive numerical data type
        The primitive data type of the dataset

    data : array (default: None)
        Data to populate the dataset with
        If this parameter is not None then the shape and dtype parameters must
        be None, as in this case those values are inferred from this array.

    overwrite : bool (default: False)
        If False, an exception will be raised when trying to write to a dataset
        that already exists. If True then existing datasets can be overwritten.

    for_timeindex : int (default: None)
        The full size of the time axis. If data is provided, we may only be
        able to provide one frame. With this parameter we can ensure the time
        axis is large enough to accommodate the full time series.

    for_channel : int (default: None)
        The full size of the channels axis. If data is provided, we may only
        be able to provide one channel. With this parameter we can ensure the time
        axis is large enough to accommodate the full time series.

    parent_attrs : dict (default: {})
        Metadata attributes for the group that will contain the dataset

    **dataset_attrs : any additional keyword arguments
        Metadata attributes for the dataset itself

    Returns
    -------
    dataset : zarr.array
        The zarr.array reference to the newly created dataset
    """

    try:

        # parse container_path
        real_container_path = os.path.realpath(container_path)
        path_comps = os.path.splitext(container_path)
        container_ext = path_comps[1]

        # create the correct store
        if container_ext == '.zarr':
            store = zarr.DirectoryStore(real_container_path, dimension_separator='/')
        else:
            store = zarr.N5Store(real_container_path)

        # create dataset in root group at container_subpath
        if container_subpath:

            # log dataset spec, open root container, specify compression codec
            logger.info((
                f'Create dataset {container_path}:{container_subpath} '
                f'compressor={compressor}, shape: {shape}, chunks: {chunks} '
                f'parent attrs: {parent_attrs} '
                f'{dataset_attrs} '
            ))
            root_group = zarr.open_group(store=store, mode='a')
            codec = (None if compressor is None 
                     else codecs.get_codec(dict(id=compressor)))

            # total replacement with empty container
            if data is None and overwrite:
                dataset_shape = shape
                dataset = root_group.create_dataset(    # XXX ZARR API says should replace with group.create_array
                    container_subpath,
                    shape=shape,
                    chunks=chunks,
                    dtype=dtype,
                    overwrite=overwrite,
                    compressor=codec,    # XXX ZARR API prefers using compressors keyword
                    data=data)

            # we have initialization data and/or the dataset already exists
            else:

                # get dataset shape, either from given data or existing dataset
                if container_subpath in root_group:
                    dataset_shape = root_group[container_subpath].shape
                    logger.info((
                        f'Dataset {container_path}:{container_subpath} '
                        f'already exists with shape {dataset_shape} '
                    ))
                else:
                    dataset_shape = shape

                # return dataset, with possibility it already exists
                dataset = root_group.require_dataset(    # XXX ZARR API says should replace with group.require_array
                    container_subpath,
                    shape=dataset_shape,
                    chunks=chunks,
                    dtype=dtype,
                    overwrite=overwrite,
                    compressor=codec,    # XXX ZARR API prefers using compressors keyword
                    data=data)

            # ensure time and channel axes are sufficient length
            _resize_dataset(dataset, dataset_shape, for_timeindex, for_channel)

            # add group and dataset metadata
            _update_dataset_attrs(root_group, dataset,
                                  parent_attrs=parent_attrs,
                                  **dataset_attrs)
            return dataset

        # open a root zarr container only and set attributes
        # XXX currently this block will never execute because container_subpath
        #     is compulsory
        else:
            logger.info(f'Create root array {container_path} {kwargs}')
            zarr_data = zarr.open(store=store, mode='a',
                                  shape=shape, chunks=chunks)
            # set additional attributes
            zarr_data.attrs.update((k, v) for k,v in kwargs.items() if v)
            return zarr_data

    # write to log before erroring out
    except Exception as e:
        logger.error(f'Error creating a dataset at {container_path}:{container_subpath}: {e}')
        raise e


def _resize_dataset(dataset, dataset_shape, for_timeindex, for_channel):
    """
    Resize the dataset to accommodate the timeindex and channel

    The time and channels axes are assumed to be the 0 and 1 index axes
    respectively. If these axes shapes are smaller than for_timeindex or
    for_channel respectively, they are resized to those values.
    """

    # initializations
    resized_shape = ()
    to_resize = False
    i = 0

    # time axis
    if for_timeindex is not None:
        if dataset_shape[i] <= for_timeindex:
            resized_shape = resized_shape + (for_timeindex + 1,)
            to_resize = True
        else:
            resized_shape = resized_shape + (dataset_shape[i],)
        i = i + 1

    # channel axis
    if for_channel is not None:
        if dataset_shape[i] <= for_channel:
            resized_shape = resized_shape + (for_channel + 1,)
            to_resize = True
        else:
            resized_shape = resized_shape + (dataset_shape[i],)
        i = i + 1

    # if we have to expand
    if to_resize:
        while i < len(dataset_shape):
            resized_shape = resized_shape + (dataset_shape[i],)
            i = i + 1

        # log and then resize in place
        logger.info(f'Resize {dataset.store.path}:{dataset.path} to {resized_shape}')
        dataset.resize(resized_shape)


def _update_dataset_attrs(root_container, dataset,
                          parent_attrs={}, **dataset_attrs):
    """
    Store all metadata attributes for a dataset in the parent container
    """

    # get parent container as object, it will contain metadata for the group
    if dataset.path:
        dataset_parent = os.path.dirname(dataset.path)
        parent_container = (root_container if not dataset_parent
                            else root_container.require_group(dataset_parent))
    else:
        parent_container = root_container

    # write the parent (group) metadata and the dataset metadata
    parent_container.attrs.update(parent_attrs)
    dataset.attrs.update(dataset_attrs)


def get_voxel_spacing(attrs: dict):
    """
    Parse an attributes dictionary and return voxel spacing.
    Works for OME-ZARR or N5 attributes.
    N5 attributes can be given at any scale and the downsampling factor will
    be taken into account.

    For OME-ZARR, spacing for [time, ch, dz, dy, dx] is returned
    For N5, spacing for [dz, dy, dx] is returned
    """
    pr = None
    if attrs.get('coordinateTransformations'):
        # this is the OME-ZARR format
        scale_metadata = list(filter(lambda t: t.get('type') == 'scale', attrs['coordinateTransformations']))
        if len(scale_metadata) > 0:
            # return voxel spacing as [time, ch, dz, dy, dx]
            pr = np.array(scale_metadata[0]['scale'])
        else:
            pr = None
    elif (attrs.get('downsamplingFactors')):
        # N5 at scale > S0
        pr = (np.array(attrs['pixelResolution']) * 
              np.array(attrs['downsamplingFactors']))
        pr = pr[::-1]  # zyx order
    elif attrs.get('pixelResolution'):
        # N5 at scale S0
        pr_attr = attrs.get('pixelResolution')
        if type(pr_attr) is list:
            pr = np.array(pr_attr)
            pr = pr[::-1]  # zyx order
        elif type(pr_attr) is dict:
            if pr_attr.get('dimensions'):
                pr = np.array(pr_attr['dimensions'])
                pr = pr[::-1]  # zyx order
    logger.debug(f'Voxel spacing from attributes: {pr}')
    return pr


def open(container_path, subpath,
         data_timeindex=None, data_channels=None,
         block_coords=None, container_type=None):
    """
    A generalized open function that supports nrrd, tiff, npy, n5, and zarr
    containers. Maps to the appropriate format specific open function.
    """

    # parse container_path
    real_container_path = os.path.realpath(container_path)
    path_comps = os.path.splitext(container_path)
    container_ext = path_comps[1]

    # call the appropriate format specific open function
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


def prepare_parent_group_attrs(container_path,
                               dataset_path,
                               axes=None,
                               coordinateTransformations=None):
    if ((coordinateTransformations is None or coordinateTransformations == []) and
        axes is None):
        return {}

    if dataset_path:
        dataset_path_comps = [c for c in dataset_path.split('/') if c]
        logger.info(f'Lookup dataset path: {dataset_path} in {dataset_path_comps}')
        # take the last component of the dataset path to be the scale path
        dataset_scale_subpath = dataset_path_comps.pop()
    else:
        # No subpath was provided - I am using '.', but
        # this may be problematic - I don't know yet how to handle it properly
        logger.info('No dataset was provided - will use "." for dataset subpath')
        dataset_scale_subpath = '.'

    scales, translations = None, None
    if coordinateTransformations is not None:
        for t in coordinateTransformations:
            if t['type'] == 'scale':
                scales = t['scale']
            elif t['type'] == 'translation':
                translations = t['translation']

    multiscale_attrs = {
        'name': os.path.basename(container_path),
        'axes': axes if axes is not None else [],
        'version': '0.4',
    }

    if scales is not None:
        dataset = Dataset.build(path=dataset_scale_subpath, scale=scales, translation=translations)
        multiscale_attrs.update({
            'datasets': (dataset.dict(exclude_none=True),),
        })

    return {
        'multiscales': [ multiscale_attrs ],
    }
    

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
        if len(block_coords) == len(image.shape):
            return image[block_coords]
        else:
            block_selector = []
            if len(image.shape) - len(block_coords) >= 2:
                # image has 2 additional dimensions - so it's very likely timepoints are present
                if image_timeindex is not None:
                    block_selector.append(image_timeindex)
                else:
                    block_selector.append(slice(None))
            if len(image.shape) - len(block_coords) >= 1:
                # image has at least one extra dimension - very likely channel is present
                if image_channel is None or image_channel == []:
                    # this is very likely to result in an error further down
                    block_selector.append(slice(None))
                else:
                    block_selector.append(image_channel)
            block_selector.extend(block_coords)
            return image[tuple(block_selector)]

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
        zarr_container_path, zarr_subpath = _adjust_data_paths(data_path, data_subpath, data_store_name)
        data_store = _get_data_store(zarr_container_path, data_store_name)
        data_container = zarr.open(store=data_store, mode='r')
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
        logger.exception(f'Error opening {data_path} : {data_subpath}')
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

    data_container_attrs = data_container.attrs.asdict()
    if data_container_attrs.get('multiscales', []) == []:
        return None, None, {}
    else:
        # the container itself has multiscales attributes
        return data_container, '', data_container_attrs


def _open_ome_zarr(data_container, data_subpath,
                   data_timeindex=None, data_channels=None, block_coords=None):
    multiscales_group, dataset_subpath, multiscales_attrs  = _find_ome_multiscales(data_container, data_subpath)

    if multiscales_group is None:
        a = (data_container[data_subpath]
             if data_subpath and data_subpath != '.'
             else data_container)
        ba = a[block_coords] if block_coords is not None else a
        return ba, a.attrs.asdict()

    logger.debug((
        f'Open dataset {dataset_subpath}, timeindex: {data_timeindex}, '
        f'channels: {data_channels}, block_coords {block_coords} '
    ))

    dataset_comps = [c for c in dataset_subpath.split('/') if c]
    # ome_metadata = ImageAttrs.construct(**multiscales_attrs)
    multiscale_metadata = multiscales_attrs.get('multiscales', [])[0]
    # pprint.pprint(ome_metadata)
    dataset_metadata = None
    # lookup the dataset by path
    for ds in multiscale_metadata.get('datasets', []):
        ds_path = ds.get('path', '')
        current_ds_path_comps = [c for c in ds_path.split('/') if c]
        logger.debug((
            f'Compare current dataset path: {ds_path} ({current_ds_path_comps}) '
            f'with {dataset_subpath} ({dataset_comps}) '
        ))
        if (len(current_ds_path_comps) <= len(dataset_comps) and
            tuple(current_ds_path_comps) == tuple(dataset_comps[-len(current_ds_path_comps):])):
            # found a dataset that has a path matching a suffix of the dataset_subpath arg
            dataset_metadata = ds
            currrent_dataset_path = ds.get('path')
            # drop the matching suffix
            dataset_comps = dataset_comps[-len(current_ds_path_comps):]
            logger.debug((
                f'Found dataset: {currrent_dataset_path}, '
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
        if dataset_index < len(multiscale_metadata.get('datasets', [])):
            dataset_metadata = multiscale_metadata['datasets'][dataset_index]
        else:
            dataset_metadata = multiscale_metadata['datasets'][0]

    dataset_axes = multiscale_metadata.get('axes')
    dataset_path = dataset_metadata.get('path')
    dataset_transformations = dataset_metadata.get('coordinateTransformations')
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
        'axes': dataset_axes,
        'dimensions': a.shape,
        'dataType': a.dtype,
        'blockSize': a.chunks,
        'timeindex': data_timeindex,
        'channels': data_channels,
        'coordinateTransformations': dataset_transformations,
    })
    return ba, multiscales_attrs


def _get_array_selector(axes, timeindex: int | None,
                        ch:int | list[int] | None,
                        block_coords: tuple | None):
    selector = []
    selection_exists = False
    spatial_selection = []
    for a in axes:
        if a.get('type') == 'time':
            if timeindex is not None:
                selector.append(timeindex)
                selection_exists = True
            else:
                selector.append(slice(None, None))
        elif a.get('type') == 'channel':
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

    def _selector(a):
        if selection_exists:
            try:
                # try to select the data using the selector
                return a[tuple(selector)]
            except Exception  as e:
                logger.exception(f'Error selecting data with selector {selector}')
                raise e
        else:
            # no selection was made, so return the whole array
            return a

    return _selector

def _open_zarr_attrs(data_path, data_subpath, data_store_name=None):
    try:
        zarr_container_path, zarr_subpath = _adjust_data_paths(data_path, data_subpath, data_store_name)
        data_store = _get_data_store(zarr_container_path, data_store_name)
        data_container = zarr.open(store=data_store, mode='r')
        data_container_attrs = data_container.attrs.asdict()

        if _is_ome_zarr(data_container_attrs):
            a, dataset_attrs = _open_ome_zarr(data_container, zarr_subpath)
            dataset_attrs.update(a.attrs.asdict())
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


def _adjust_data_paths(data_path, data_subpath, data_store_name):
    """
    This methods adjusts the container and dataset paths such that
    the container paths always contains a .attrs file
    """
    if data_store_name == 'n5' or data_path.endswith('.n5') or data_path.endswith('.N5'):
        # N5 container path is the same as the data_path
        # and the subpath is the dataset path
        return data_path, data_subpath

    dataset_path_arg = data_subpath if data_subpath is not None else ''
    dataset_comps = [c for c in dataset_path_arg.split('/') if c]
    dataset_comps_index = 0

    # Look for the first subpath that containes .zattrs file
    while dataset_comps_index < len(dataset_comps):
        container_subpath = '/'.join(dataset_comps[0:dataset_comps_index])
        container_path = f'{data_path}/{container_subpath}'
        if (os.path.exists(f'{container_path}/.zattrs') or
            os.path.exists(f'{container_path}/attributes.json')):
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
        return zarr.DirectoryStore(data_path, dimension_separator='/')



def _read_tiff(input_path, block_coords=None):
    "Return tiff data as zarr array, slice at block_coords, include metadata"
    with TiffFile(input_path) as tif:
        tif_store = tif.aszarr()
        tif_array = zarr.open(tif_store)
        if block_coords is None:
            img = tif_array
        else:
            img = tif_array[block_coords]
        return img, _get_tiff_attrs(tif_array)


def _read_tiff_attrs(input_path):
    "Open tiff, call function to get tiff metadata"
    with TiffFile(input_path) as tif:
        tif_store = tif.aszarr()
        tif_array = zarr.open(tif_store)
        return _get_tiff_attrs(tif_array)


def _get_tiff_attrs(tif_array):
    "read tiff attributes, nicely organized by zarr, append dtype and shape"
    dict = tif_array.attrs.asdict()
    dict.update({
        'dataType': tif_array.dtype,
        'dimensions': tif_array.shape,
    })
    return dict


def _read_nrrd(input_path, block_coords=None):
    "read nrrd and metadata, slice at block_coords"
    im, dict = nrrd.read(input_path)
    return im[block_coords] if block_coords is not None else im, dict


def _read_nrrd_attrs(input_path):
    "read only the nrrd metadata"
    return nrrd.read_header(input_path)


def _extract_numeric_comp(v):
    match = re.match(r'^(\D*)(\d+)$', v)
    if match:
        return int(match.groups()[1])
    else:
        raise ValueError(f'Invalid component: {v}')
