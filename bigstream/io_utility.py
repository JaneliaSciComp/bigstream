import io
import json
import logging
import nrrd
import numpy as np
import os
import re
import tensorstore as ts
import zarr

from dataclasses import asdict as _dc_asdict

import ngff_zarr
from ngff_zarr.v04.zarr_metadata import (
    Axis as _AxisV04,
    Dataset as _DatasetV04,
    Metadata as _MetadataV04,
    Scale as _ScaleV04,
    Translation as _TranslationV04,
)
from ngff_zarr.v05.zarr_metadata import Metadata as _MetadataV05
from tifffile import TiffFile


logger = logging.getLogger(__name__)


def _get_compressor(compressor: str | None, compression_opts: dict, zarr_format: int):
    logger.info(f'Get compressor: {compressor}: {compression_opts} for zarr format {zarr_format}')
    if compressor is None:
        return None

    if zarr_format == 3:
        import zarr.codecs as zv3codecs

        compressor_map = {
            'zstd': zv3codecs.ZstdCodec,
            'gzip': zv3codecs.GzipCodec,
            'blosc': zv3codecs.BloscCodec,
            'crc32c': zv3codecs.Crc32cCodec,
            # 'packbits', 'zlib', 'bz2', 'lzma', 'lz4' → only available as numcodecs
            'packbits': zv3codecs.PackBits,
            'zlib': zv3codecs.Zlib,
            'bz2': zv3codecs.BZ2,
            'lzma': zv3codecs.LZMA,
            'lz4': zv3codecs.LZ4,
        }
        codec_cls = compressor_map.get(compressor.lower())
        if codec_cls is None:
            raise ValueError(f'Unsupported compressor {compressor} for zarr format {zarr_format}')
        return codec_cls(**compression_opts)
    else:
        import numcodecs as zv2codecs

        return zv2codecs.get_codec({'id': compressor, **compression_opts})


def _compressor_kwargs(codec, zarr_format: int):
    if zarr_format == 3:
        # zarr v3 expects a tuple of BytesBytesCodec (or an empty tuple)
        compressors = None if codec is None else (codec,)
        return {'compressors': compressors}
    else:
        return {'compressors': codec}


def _ensure_parent_groups(root_group, container_subpath):
    # Materialize each intermediate group along container_subpath so that
    # the subsequent create_array call finds every parent already on disk.
    # Workaround for zarr-python 3's ensure_parents=True path, which writes
    # missing parent zarr.json via set_if_not_exists -> os.link, failing on
    # filesystems without hardlink support (NFS/SMB/exFAT/FUSE).
    if not container_subpath:
        return
    comps = [c for c in container_subpath.split('/') if c]
    for i in range(1, len(comps)):
        prefix = '/'.join(comps[:i])
        if prefix not in root_group:
            root_group.require_group(prefix)


def create_dataset_array(
    container_path,
    container_subpath,
    shape,
    chunks,
    dtype,
    shard_shape=None,
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

    shard_shape : tuple of ints (default: None)
        Zarr v3 shard shape. When provided, multiple inner chunks are packed
        into a single shard file. Each component must be a positive multiple
        of the corresponding chunk dimension. Ignored (with a warning) when
        the target format is not zarr v3.

    **dataset_attrs : any additional keyword arguments
        Metadata attributes for the dataset itself

    Returns
    -------
    dataset : zarr.array
        The zarr.array reference to the newly created dataset
    """

    # parse container_path
    real_container_path = os.path.realpath(container_path)
    path_comps = os.path.splitext(container_path)
    container_ext = path_comps[1]

    if container_ext == '.n5':
        # create N5 array
        ...
    else:
        # default to zarr container
        return _create_dataset_zarray(
            real_container_path,
            container_subpath,
            shape,
            chunks,
            dtype,
            shard_shape=shard_shape,
            **dataset_attrs,
        )


def _create_dataset_zarray(
    container_path,
    container_subpath,
    shape,
    chunks,
    dtype,
    overwrite=False,
    for_timeindex=None,
    for_channel=None,
    compressor=None,
    compression_opts:dict={},
    parent_attrs:dict={},
    zarr_format:int=2, # default zarr v2 format
    shard_shape=None,
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
        # create the correct store
        store = zarr.storage.LocalStore(container_path)

        codec = _get_compressor(compressor, compression_opts, zarr_format)
        compressor_args = _compressor_kwargs(codec, zarr_format)
        logger.info(f'Dataset compression arguments: {codec}:{compressor_args}')

        chunk_key_separator = {'name': 'v2', 'separator': '/'} if zarr_format == 2 else None

        extra_create_kwargs = {}
        if shard_shape is not None and len(shard_shape) > 0:
            if zarr_format < 3:
                logger.info(
                    f'Ignoring shard_shape={shard_shape}; sharding requires '
                    f'zarr_format=3 (got {zarr_format})'
                )
            else:
                shards = tuple(shard_shape)
                if len(shards) != len(chunks):
                    raise ValueError(
                        f'shard_shape {shards} has different rank than chunks {chunks}'
                    )
                for s, c in zip(shards, chunks):
                    if s <= 0 or s % c != 0:
                        raise ValueError(
                            f'shard_shape {shards} must be a positive multiple '
                            f'of chunks {chunks} elementwise'
                        )
                extra_create_kwargs['shards'] = shards

        # create dataset in root group at container_subpath
        if container_subpath:
            logger.info((
                f'Create dataset {container_path}:{container_subpath} '
                f'compressor={compressor}, shape: {shape}, chunks: {chunks} '
                f'parent attrs: {parent_attrs} '
                f'{dataset_attrs} '
            ))
            try:
                root_group = zarr.open_group(store=store,
                                             mode='a',
                                             zarr_format=zarr_format)
            except zarr.errors.ContainsGroupError:
                # group already exists - open without zarr_format
                # to let zarr auto-detect the existing format
                root_group = zarr.open_group(store=store, mode='a')

            if overwrite:
                # total replacement with empty container
                final_dataset_shape = _get_array_shape(shape, for_timeindex, for_channel)
                dataset_shape = final_dataset_shape
                _ensure_parent_groups(root_group, container_subpath)
                dataset_array = root_group.create_array(
                    container_subpath,
                    shape=_to_native_type(dataset_shape),
                    chunks=chunks,
                    dtype=dtype,
                    overwrite=overwrite,
                    chunk_key_encoding=chunk_key_separator,
                    **compressor_args,
                    **extra_create_kwargs,
                )
            else:
                # get dataset shape, either from given data or existing dataset
                if container_subpath in root_group:
                    # if the dataset already exists, get its shape
                    dataset_array = root_group[container_subpath]
                    dataset_shape = dataset_array.shape
                    logger.info((
                        f'Dataset {container_path}:{container_subpath} '
                        f'already exists with shape {dataset_shape} '
                    ))
                    # if array already exists ensure time and channel axes are sufficient length
                    final_dataset_shape = _get_array_shape(dataset_shape, for_timeindex, for_channel)
                    if final_dataset_shape != dataset_shape:
                        logger.info(f'Resize {dataset_array.store}:{dataset_array.path} to {final_dataset_shape}')
                        dataset_array.resize(final_dataset_shape)
                else:
                    # this is a new dataset
                    final_dataset_shape = _get_array_shape(shape, for_timeindex, for_channel)
                    dataset_shape = final_dataset_shape
                    try:
                        _ensure_parent_groups(root_group, container_subpath)
                        dataset_array = root_group.create_array(
                            container_subpath,
                            shape=_to_native_type(dataset_shape),
                            chunks=chunks,
                            dtype=dtype,
                            overwrite=overwrite,
                            chunk_key_encoding=chunk_key_separator,
                            **compressor_args,
                            **extra_create_kwargs,
                        )
                    except zarr.errors.ContainsArrayError:
                        # array exists but was not found by __contains__
                        logger.info((
                            f'Dataset {container_path}:{container_subpath} '
                            f'already exists (detected on create) '
                        ))
                        dataset_array = root_group[container_subpath]
                        dataset_shape = dataset_array.shape
                        final_dataset_shape = _get_array_shape(dataset_shape, for_timeindex, for_channel)
                        if final_dataset_shape != dataset_shape:
                            logger.info(f'Resize {dataset_array.store}:{dataset_array.path} to {final_dataset_shape}')
                            dataset_array.resize(final_dataset_shape)

            # add group and dataset metadata
            _update_dataset_attrs(root_group, dataset_array,
                                  parent_attrs=parent_attrs,
                                  **dataset_attrs)
        else:
            # the zarr container is the array
            logger.info(f'Create root array {container_path} {dataset_attrs}')
            if overwrite:
                # doesn't matter if array exists or not we overwrite it anyway
                dataset_array = zarr.create_array(
                    store=store,
                    shape=shape,
                    chunks=chunks,
                    dtype=dtype,
                    overwrite=True,
                    chunk_key_encoding=chunk_key_separator,
                    zarr_format=zarr_format,
                    **compressor_args,
                    **extra_create_kwargs,
                )
            else:
                try:
                    # open existing array for read/write (raises ArrayNotFoundError if absent)
                    dataset_array = zarr.open_array(store=store, mode='r+')
                    final_dataset_shape = _get_array_shape(shape, for_timeindex, for_channel)
                    if final_dataset_shape != shape:
                        logger.info(f'Resize {dataset_array.store}:{dataset_array.path} to {final_dataset_shape}')
                        dataset_array.resize(final_dataset_shape)
                except zarr.errors.ArrayNotFoundError:
                    # this is a new array
                    logger.info(f'Create new array at {store}')
                    dataset_array = zarr.create_array(
                        store=store,
                        shape=shape,
                        chunks=chunks,
                        dtype=dtype,
                        chunk_key_encoding=chunk_key_separator,
                        zarr_format=zarr_format,
                        **compressor_args,
                        **extra_create_kwargs,
                    )
            # set additional attributes
            dataset_array.attrs.update(_to_native_type(dataset_attrs))

        return dataset_array

    # write to log before erroring out
    except Exception as e:
        logger.exception(f'Error creating a dataset at {container_path}:{container_subpath}')
        raise e


def _get_array_shape(dataset_shape, for_timeindex, for_channel):
    """
    Compute dataset shape to accommodate the timeindex and channel

    The time and channels axes are assumed to be the 0 and 1 index axes
    respectively. If these axes shapes are smaller than for_timeindex or
    for_channel respectively, they are resized to those values.
    """

    # initializations
    final_shape = ()
    i = 0

    # time axis
    if for_timeindex is not None:
        if dataset_shape[i] <= for_timeindex:
            final_shape = final_shape + (for_timeindex + 1,)
        else:
            final_shape = final_shape + (dataset_shape[i],)
        i = i + 1

    # channel axis
    if for_channel is not None:
        if dataset_shape[i] <= for_channel:
            final_shape = final_shape + (for_channel + 1,)
        else:
            final_shape = final_shape + (dataset_shape[i],)
        i = i + 1

    while i < len(dataset_shape):
        final_shape = final_shape + (dataset_shape[i],)
        i = i + 1

    return final_shape


def _detect_ome_version(group_attrs):
    """Return '0.4' for top-level multiscales, '0.5' for ome.multiscales, else None."""
    if not isinstance(group_attrs, dict):
        return None
    if group_attrs.get('multiscales'):
        return '0.4'
    ome = group_attrs.get('ome')
    if isinstance(ome, dict) and ome.get('multiscales'):
        return '0.5'
    return None


def _get_multiscales_entry(group_attrs, ome_version):
    """Return first multiscales entry dict from group attrs."""
    if ome_version == '0.4':
        ms = group_attrs.get('multiscales')
        return ms[0] if ms else None
    if ome_version == '0.5':
        ms = group_attrs.get('ome', {}).get('multiscales')
        return ms[0] if ms else None
    return None


def _axis_input_to_axis(a):
    if isinstance(a, _AxisV04):
        return a
    return _AxisV04(
        name=a.get('name'),
        type=a.get('type'),
        unit=a.get('unit'),
    )


def _transform_input_to_object(td):
    if td is None:
        return None
    if isinstance(td, (_ScaleV04, _TranslationV04)):
        return td
    ttype = td.get('type')
    if ttype == 'scale':
        return _ScaleV04(scale=list(td['scale']))
    if ttype == 'translation':
        return _TranslationV04(translation=list(td['translation']))
    return None


def _transform_object_to_dict(t):
    if isinstance(t, _ScaleV04):
        return {'type': 'scale', 'scale': list(t.scale)}
    if isinstance(t, _TranslationV04):
        return {'type': 'translation', 'translation': list(t.translation)}
    if hasattr(t, '__dataclass_fields__'):
        return _dc_asdict(t)
    if isinstance(t, dict):
        return t
    return t


def _clean_metadata_dict(metadata_dict):
    """Strip optional/null fields from a serialized NGFF metadata dict.

    Mirrors ngff_zarr.to_ngff_zarr._pop_metadata_optionals.
    """
    for ax in metadata_dict.get('axes', []) or []:
        if ax.get('unit') is None:
            ax.pop('unit', None)
        ax.pop('orientation', None)
    if metadata_dict.get('coordinateTransformations') is None:
        metadata_dict.pop('coordinateTransformations', None)
    if metadata_dict.get('omero') is None:
        metadata_dict.pop('omero', None)
    if metadata_dict.get('type') is None:
        metadata_dict.pop('type', None)
    if metadata_dict.get('metadata') is None:
        metadata_dict.pop('metadata', None)
    return metadata_dict


def _build_ngff_metadata(name, dataset_path, axes,
                        dataset_transformations,
                        global_transformations,
                        zarr_format):
    """Build a typed ngff-zarr Metadata object for the target zarr_format."""
    axis_objs = [_axis_input_to_axis(a) for a in (axes or [])]
    dataset_xforms = [
        _transform_input_to_object(t) for t in (dataset_transformations or [])
    ]
    dataset_xforms = [t for t in dataset_xforms if t is not None]
    datasets = [_DatasetV04(path=dataset_path,
                            coordinateTransformations=dataset_xforms)]
    if global_transformations:
        global_xforms = [
            _transform_input_to_object(t) for t in global_transformations
        ]
        global_xforms = [t for t in global_xforms if t is not None] or None
    else:
        global_xforms = None
    if zarr_format == 3:
        return _MetadataV05(
            axes=axis_objs,
            datasets=datasets,
            coordinateTransformations=global_xforms,
            name=name,
        )
    return _MetadataV04(
        axes=axis_objs,
        datasets=datasets,
        coordinateTransformations=global_xforms,
        name=name,
    )


def _serialize_ngff_metadata(metadata, zarr_format):
    """Serialize a typed metadata object to a dict laid out for the target version."""
    metadata_dict = _clean_metadata_dict(_dc_asdict(metadata))
    if zarr_format == 3:
        # v0.5: ome envelope; version sits at ome level
        return {'ome': {'version': '0.5', 'multiscales': [metadata_dict]}}
    # v0.4: top-level multiscales; version inside multiscales[0]
    return {'multiscales': [metadata_dict]}


def _merge_ngff_metadata(existing_attrs, new_attrs):
    """Merge new parent attrs with existing, preserving OME datasets list.

    Handles both v0.4 (top-level `multiscales`) and v0.5 (`ome.multiscales`).
    """
    if not new_attrs:
        return new_attrs
    new_version = _detect_ome_version(new_attrs)
    if new_version is None:
        return new_attrs
    existing_version = _detect_ome_version(existing_attrs)
    if existing_version != new_version:
        return new_attrs

    existing_ms = _get_multiscales_entry(existing_attrs, existing_version)
    new_ms = _get_multiscales_entry(new_attrs, new_version)
    if existing_ms is None or new_ms is None:
        return new_attrs

    existing_datasets = list(existing_ms.get('datasets', []))
    new_datasets = new_ms.get('datasets', [])

    for new_ds in new_datasets:
        new_path = new_ds.get('path')
        replaced = False
        for i, existing_ds in enumerate(existing_datasets):
            if existing_ds.get('path') == new_path:
                existing_datasets[i] = new_ds
                replaced = True
                break
        if not replaced:
            existing_datasets.append(new_ds)

    merged_ms = dict(new_ms)
    merged_ms['datasets'] = existing_datasets

    if new_version == '0.4':
        merged = {**existing_attrs, **new_attrs, 'multiscales': [merged_ms]}
    else:
        merged_ome = {
            **(existing_attrs.get('ome') or {}),
            **(new_attrs.get('ome') or {}),
        }
        merged_ome['multiscales'] = [merged_ms]
        merged = {**existing_attrs, **new_attrs, 'ome': merged_ome}
    return merged


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

    # merge OME multiscales datasets if parent already has metadata
    merged_attrs = _merge_ngff_metadata(
        dict(parent_container.attrs), parent_attrs
    )

    # write the parent (group) metadata and the dataset metadata
    parent_container.attrs.update(_to_native_type(merged_attrs))
    dataset.attrs.update(_to_native_type(dataset_attrs))


def _to_native_type(obj):
    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, (list, tuple)):
        return [_to_native_type(x) for x in obj]

    if isinstance(obj, dict):
        return {k: _to_native_type(v) for k, v in obj.items()}

    return obj


def get_voxel_spacing(attrs: dict):
    """
    Parse an attributes dictionary and return voxel spacing.
    Works for OME-ZARR or N5 attributes.
    N5 attributes can be given at any scale and the downsampling factor will
    be taken into account.

    Parameters
    ----------
    attrs : dict
        An attributes dictionary from an OME-ZARR or N5 container

    Returns
    -------
    spacing : 1-d iterable
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


def open_image_container(container_path, subpath,
                         data_timeindex=None, data_channels=None,
                         block_coords=None, container_type=None):
    """
    A generalized open function that supports nrrd, tiff, npy, n5, and zarr
    containers. Maps to the appropriate format specific open function.

    Parameters
    ----------
    container_path : string
        Path to the image file or data container

    subpath : string
        Path to dataset within a container. For nrrd, tiff, or npy this can be
        None

    data_timeindex : int (default: None)
        Index along time axis you want returned. None returns whole time axis.

    data_channels : int (default: None)
        Index along channels axis you want returned. None returns whole
        channels axis.

    block_coords : tuple of slice objects (default: None)
        A sub-region of the spatial axes you want returned. None returns
        the whole spatial domain.

    container_type : string (default: None)
        Explicit identifier of container type. Options include:
        'nrrd', 'tif', 'npy', 'n5', and 'zarr'.
        If None, we attempt to infer the container type from the container_path
        extension, which is not always reliable.

    Returns
    -------
    data : array like
        The image data

    metadata : dict
        The metadata for the image. Keys are specific to the container type.
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
        return _open_n5(real_container_path, subpath,
                        block_coords=block_coords)
    elif container_ext == '.zarr' or container_type == 'zarr':
        logger.info(f'Open Zarr {container_path}:{subpath} ({real_container_path})')
        return _open_zarr(real_container_path, subpath,
                          data_timeindex=data_timeindex,
                          data_channels=data_channels,
                          block_coords=block_coords)
    else:
        logger.error(f'Cannot handle {container_path} ' +
                     f'({real_container_path}) {subpath}')
        return None, {}


def prepare_parent_group_attrs(container_path,
                               dataset_path,
                               axes=None,
                               dataset_transformations=None,
                               global_transformations=None,
                               zarr_format=2):
    """
    Prepare an OME-NGFF metadata dictionary for the parent group of a dataset.

    The returned dict layout depends on ``zarr_format``:
      - ``zarr_format == 2``: emits OME-NGFF v0.4 (top-level ``multiscales``).
      - ``zarr_format == 3``: emits OME-NGFF v0.5 (under ``ome``).

    Parameters
    ----------
    container_path : string
        Path to the container which holds the dataset

    dataset_path : string
        Subpath to the dataset within the container

    axes : list of dicts or ngff_zarr.Axis (default: None)
        Which axes are present in the dataset.

    dataset_transformations : iterable of dicts or transform objects (default: None)
        Per-dataset scale and translation transformations.

    global_transformations : iterable of dicts or transform objects (default: None)
        Multiscale-level coordinate transformations.

    zarr_format : int (default: 2)
        Target zarr format. 2 → OME-NGFF v0.4. 3 → OME-NGFF v0.5.

    Returns
    -------
    attrs : dict
        Group attributes ready to be written via ``parent.attrs.update(...)``.
    """

    # case of no relevant metadata
    if ((dataset_transformations is None or dataset_transformations == []) and
        axes is None):
        return {}

    # identify which dataset in the container we're creating attributes for
    if dataset_path:
        dataset_path_comps = [c for c in dataset_path.split('/') if c]
        logger.info(f'Lookup dataset path: {dataset_path} in {dataset_path_comps}')
        # take the last component of the dataset path to be the scale path
        dataset_scale_subpath = dataset_path_comps.pop() if dataset_path_comps else '.'
    else:
        # No subpath was provided - I am using '.', but
        # this may be problematic - I don't know yet how to handle it properly
        logger.info('No dataset was provided - will use "." for dataset subpath')
        dataset_scale_subpath = '.'

    name = os.path.basename(container_path)
    metadata = _build_ngff_metadata(
        name=name,
        dataset_path=dataset_scale_subpath,
        axes=axes,
        dataset_transformations=dataset_transformations,
        global_transformations=global_transformations,
        zarr_format=zarr_format,
    )
    return _serialize_ngff_metadata(metadata, zarr_format)


def read_image_container_attributes(container_path, subpath, container_type=None):
    """
    A generalized attribute reader that supports nrrd, tiff, n5, and zarr
    containers. Maps to the appropriate format specific attributes reader.

    Parameters
    ----------
    container_path : string
        Path to the image file/container

    subpath : string
        Subpath to the dataset within the container. For nrrd and tiff can be None.

    container_type : string (default: None)
        Explicit identifier of container type. Options include:
        'nrrd', 'tif', 'n5', and 'zarr'.
        If None, we attempt to infer the container type from the container_path
        extension, which is not always reliable.

    Returns
    -------
    attrs : dict
        The metadata dictionary for the container/image.
    """

    # parse container_path
    real_container_path = os.path.realpath(container_path)
    path_comps = os.path.splitext(container_path)
    container_ext = path_comps[1]

    # map container type to appropriate attributes reader
    # read and return attributes
    if container_ext == '.nrrd' or container_type == 'nrrd':
        logger.info(f'Read nrrd attrs {container_path} ({real_container_path})')
        return _read_nrrd_attrs(container_path)
    elif (container_ext == '.n5' or os.path.exists(f'{container_path}/attributes.json')
          or container_type == 'n5'):
        logger.info(f'Read N5 attrs {container_path} ({real_container_path})')
        return _open_n5_attrs(real_container_path, subpath)
    elif container_ext == '.zarr' or container_type == 'zarr':
        logger.info(f'Read Zarr attrs {container_path} ({real_container_path})')
        return _open_zarr_attrs(real_container_path, subpath)
    elif (container_ext == '.tif' or container_ext == '.tiff'
          or container_type == 'tif'):
        logger.info(f'Read TIFF attrs {container_path} ({real_container_path})')
        return _read_tiff_attrs(container_path)
    else:
        logger.error(f'Cannot read attributes for {container_path}:{subpath}')
        return {}


def read_block(block_coords, image=None, image_path=None, image_subpath=None,
               image_timeindex=None, image_channel=None):
    """
    Read a continuous subset of voxels (a block) from an image dataset

    Parameters
    ----------
    block_coords : tuple of slice objects
        How to slice the global nd-array to get the requested block

    image : nd-array (default: None)
        If the image is already it can be passed with this parameter

    image_path : string (default: None)
        If the image is not already open, this is the path to it on disk

    image_subpath : string (default: None)
        Subpath within the container for the dataset you want to open

    image_timeindex : int or slice object (default: None)
        An index or slice for the time axis if not included in block_coords

    image_channel : int or slice object (default: None)
        An index or slice for the channel axis if not included in block_coords

    Returns
    -------
    block : nd-array
        The region of data from image (or the container at image_path-->image_subpath)
        specificed by block_coords
    """

    # if no crop is provided
    if block_coords is None:
        return None

    # image already in memory as nd-array
    if image is not None:
        if len(block_coords) == len(image.shape):
            # simplest case, just apply the crop
            return image[block_coords]
        else:
            # when block_coords only specifies some but not all axes
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

    # image must be opened first, use generalized open function
    if image_path is not None:
        block, _ = open_image_container(image_path, image_subpath,
                                        data_timeindex=image_timeindex,
                                        data_channels=image_channel,
                                        block_coords=block_coords)
        logger.debug(f'Read {block.shape} block at {block_coords}')
        return block

    # this is when there are block coords but no image nd-array or
    # image path were provided
    return None


def _open_n5(data_path, data_subpath, block_coords=None):
    """
    Open an N5 container using tensorstore, optionally read a region into memory.

    Parameters
    ----------
    data_path : string
        Path to the N5 container

    data_subpath : string
        Subpath within the N5 container (e.g., 's0' for scale level 0)

    block_coords : tuple of slice objects (default: None)
        A sub-region of the data to read. None returns the whole dataset.

    Returns
    -------
    data : numpy array
        The image data (or a slice of it if block_coords is provided)

    attrs : dict
        The attributes dictionary from the N5 container
    """
    try:
        # Build the tensorstore spec for N5
        spec = {
            "driver": "n5",
            "kvstore": {
                "driver": "file",
                "path": data_path
            },
        }
        if data_subpath:
            spec["path"] = data_subpath

        # Open the dataset synchronously using .result()
        dataset = ts.open(spec, read=True).result()

        # Read the data - either the full array or a slice
        if block_coords is not None:
            data = dataset[block_coords].read().result()
        else:
            data = dataset.read().result()

        data = data.transpose(2, 1, 0) # tensorstore reads it as X,Y,Z so I need to transpose it

        # Get attributes from the N5 container
        attrs = _open_n5_attrs(data_path, data_subpath)

        logger.debug(f'Opened N5 {data_path}:{data_subpath}, shape={dataset.shape}, attrs={attrs}')
        return data, attrs

    except Exception as e:
        logger.exception(f'Error opening N5 {data_path}:{data_subpath}')
        raise e

def _open_zarr(data_path, data_subpath,
               data_timeindex=None, data_channels=None,
               block_coords=None):
    """
    Open a zarr container, optionally read a region into memory.

    For OME-NGFF containers (v0.4 or v0.5) metadata parsing is delegated to
    ngff-zarr. Data is read directly from the underlying ``zarr.Array`` to
    preserve the existing contract (numpy on selection, lazy zarr otherwise)
    and avoid pulling in a dask scheduler.
    """
    try:
        zarr_container_path, zarr_subpath = _adjust_data_paths(data_path, data_subpath)
        data_store = _get_data_store(zarr_container_path)
        data_container = zarr.open(store=data_store, mode='r')

        ome_group, ome_group_subpath, remaining_subpath, ome_version = (
            _find_ome_in_zarr(data_container, zarr_subpath)
        )
        if ome_group is not None:
            logger.debug((
                f'Open OME container {data_container} at {ome_group_subpath}, '
                f'dataset: {remaining_subpath}, timeindex: {data_timeindex}, '
                f'channels: {data_channels}, block_coords {block_coords} '
            ))
            ome_store_path = (
                f'{zarr_container_path}/{ome_group_subpath}'
                if ome_group_subpath else zarr_container_path
            )
            multiscales = ngff_zarr.from_ngff_zarr(ome_store_path, version=ome_version)
            image, dataset_metadata = _select_ngff_dataset(multiscales, remaining_subpath)
            dataset_in_container = (
                f'{ome_group_subpath}/{dataset_metadata.path}'
                if ome_group_subpath else dataset_metadata.path
            )
            arr_node = data_container[dataset_in_container]
            data = _slice_zarr_array(arr_node, multiscales.metadata.axes,
                                     data_timeindex, data_channels, block_coords)
            attrs = _build_attrs_view(
                multiscales, dataset_metadata, arr_node,
                data_timeindex, data_channels,
            )
            arr_attrs = arr_node.attrs.asdict()
            for k in ('axes', 'coordinateTransformations',
                      'globalCoordinateTransformations', 'multiscales',
                      'dataset_path', 'dimensions', 'dataType', 'blockSize',
                      'timeindex', 'channels'):
                arr_attrs.pop(k, None)
            attrs = {**arr_attrs, **attrs}
            return data, attrs

        # plain zarr: no OME multiscales
        a = data_container[zarr_subpath] if zarr_subpath else data_container
        ba = a[block_coords] if block_coords is not None else a
        return ba, a.attrs.asdict()
    except Exception as e:
        logger.exception(
            f'Error opening {data_path} : {data_subpath} : {data_timeindex} : '
            f'{data_channels} : {block_coords} -> {e}'
        )
        raise e


def _find_ome_in_zarr(data_container, data_subpath):
    """Walk subpath looking for the first group with OME multiscales attrs.

    Returns (ome_group, group_subpath, remaining_subpath, ome_version) or
    (None, None, None, None) if no OME group is found.
    """
    subpath_arg = data_subpath if data_subpath is not None else ''
    comps = [c for c in subpath_arg.split('/') if c]
    for i in range(len(comps) + 1):
        group_subpath = '/'.join(comps[:i])
        try:
            item = data_container[group_subpath] if group_subpath else data_container
        except KeyError:
            break
        attrs = item.attrs.asdict()
        version = _detect_ome_version(attrs)
        if version is not None:
            return item, group_subpath, '/'.join(comps[i:]), version
    return None, None, None, None


def _select_ngff_dataset(multiscales, dataset_subpath):
    """Pick an NgffImage + Dataset from a parsed NgffMultiscales by subpath.

    First tries a suffix match against each dataset's stored path; falls back
    to extracting a numeric index from the last subpath component (e.g. 's2'
    → datasets[2]); otherwise returns datasets[0].
    """
    datasets = multiscales.metadata.datasets
    images = multiscales.images
    if not datasets:
        raise ValueError(f'No datasets in multiscales for subpath {dataset_subpath!r}')
    comps = [c for c in (dataset_subpath or '').split('/') if c]
    for idx, ds in enumerate(datasets):
        ds_comps = [c for c in (ds.path or '').split('/') if c]
        if (ds_comps and len(ds_comps) <= len(comps) and
            tuple(ds_comps) == tuple(comps[-len(ds_comps):])):
            return images[idx], ds
    if comps:
        match = re.match(r'^(\D*)(\d+)$', comps[-1])
        if match:
            i = int(match.group(2))
            if i < len(datasets):
                return images[i], datasets[i]
    return images[0], datasets[0]


def _slice_zarr_array(arr, axes, timeindex, channels, block_coords):
    """Apply time/channel/block selection to a zarr.Array.

    Returns numpy when any selection was applied (matching zarr indexing
    semantics); otherwise returns the lazy zarr.Array unchanged.
    """
    selector = []
    selection_exists = False
    spatial_default = []
    for ax in axes:
        ax_type = ax.type if hasattr(ax, 'type') else ax.get('type')
        if ax_type == 'time':
            if timeindex is not None:
                selector.append(timeindex)
                selection_exists = True
            else:
                selector.append(slice(None, None))
        elif ax_type == 'channel':
            if channels is None or channels == []:
                selector.append(slice(None, None))
            else:
                selector.append(channels)
                selection_exists = True
        else:
            spatial_default.append(slice(None, None))

    if block_coords is not None:
        selector.extend(block_coords)
        selection_exists = True
    else:
        selector.extend(spatial_default)

    if not selection_exists:
        return arr
    try:
        logger.debug(f'Read {selector}')
        return arr[tuple(selector)]
    except Exception as e:
        logger.exception(f'Error selecting data from {arr} with selector {selector}')
        raise e


def _axis_to_dict(ax):
    out = {'name': ax.name, 'type': ax.type}
    if ax.unit is not None:
        out['unit'] = ax.unit
    return out


def _build_attrs_view(multiscales, dataset_metadata, arr,
                      data_timeindex, data_channels):
    """Build the legacy attrs dict from parsed ngff-zarr objects and the
    underlying zarr.Array (whose ``shape``/``dtype``/``chunks`` are used for
    the dimensions/dataType/blockSize keys).
    """
    md = multiscales.metadata
    axes_dicts = [_axis_to_dict(ax) for ax in md.axes]
    dataset_xforms = [
        _transform_object_to_dict(t)
        for t in (dataset_metadata.coordinateTransformations or [])
    ]
    global_xforms = (
        [_transform_object_to_dict(t) for t in md.coordinateTransformations]
        if md.coordinateTransformations else None
    )
    metadata_dict = _clean_metadata_dict(_dc_asdict(md))
    return {
        'dataset_path': dataset_metadata.path,
        'axes': axes_dicts,
        'dimensions': arr.shape,
        'dataType': arr.dtype,
        'blockSize': arr.chunks,
        'coordinateTransformations': dataset_xforms,
        'globalCoordinateTransformations': global_xforms,
        'timeindex': data_timeindex,
        'channels': data_channels,
        'multiscales': [metadata_dict],
    }


def _open_n5_attrs(data_path, data_subpath):
    """
    Read attributes from an N5 container.

    Parameters
    ----------
    data_path : string
        Path to the N5 container

    data_subpath : string
        Subpath within the N5 container

    Returns
    -------
    attrs : dict
        The attributes dictionary from attributes.json
    """
    # Construct the full path to attributes.json
    if data_subpath:
        attrs_path = os.path.join(data_path, data_subpath, 'attributes.json')
    else:
        attrs_path = os.path.join(data_path, 'attributes.json')

    # Check if attributes.json exists
    if not os.path.exists(attrs_path):
        logger.warning(f'No attributes.json found at {attrs_path}')
        return {}

    # Read and parse the JSON file
    try:
        # use io.open to make sure it's not shadowed 
        # by some local method with same name
        with io.open(attrs_path, 'r') as f:
            attrs = json.load(f)
        logger.debug(f'Read N5 attributes from {attrs_path}: {attrs}')

        def get_spatial_attr_values(attrName):
            v = attrs.get(attrName)
            return v[::-1] if v is not None else None

        return {
            'dataType': attrs.get('dataType'),
            'dimensions': get_spatial_attr_values('dimensions'),
            'blockSize': get_spatial_attr_values('blockSize'),
            'pixelResolution': get_spatial_attr_values('pixelResolution'),
            'downsamplingFactors': get_spatial_attr_values('downsamplingFactors'),
        }
        return attrs
    except json.JSONDecodeError as e:
        logger.error(f'Error parsing attributes.json at {attrs_path}: {e}')
        return {}
    except Exception as e:
        logger.error(f'Error reading attributes.json at {attrs_path}: {e}')
        raise e


def _open_zarr_attrs(data_path, data_subpath, data_store_name=None):
    try:
        zarr_container_path, zarr_subpath = _adjust_data_paths(data_path, data_subpath)
        data_store = _get_data_store(zarr_container_path)
        data_container = zarr.open(store=data_store, mode='r')

        ome_group, ome_group_subpath, remaining_subpath, ome_version = (
            _find_ome_in_zarr(data_container, zarr_subpath)
        )
        if ome_group is not None:
            ome_store_path = (
                f'{zarr_container_path}/{ome_group_subpath}'
                if ome_group_subpath else zarr_container_path
            )
            multiscales = ngff_zarr.from_ngff_zarr(ome_store_path, version=ome_version)
            _, dataset_metadata = _select_ngff_dataset(multiscales, remaining_subpath)
            dataset_in_container = (
                f'{ome_group_subpath}/{dataset_metadata.path}'
                if ome_group_subpath else dataset_metadata.path
            )
            arr_node = data_container[dataset_in_container]
            dataset_attrs = _build_attrs_view(multiscales, dataset_metadata, arr_node,
                                              None, None)
            arr_attrs = arr_node.attrs.asdict()
            for k in ('axes', 'coordinateTransformations',
                      'globalCoordinateTransformations', 'multiscales',
                      'dataset_path', 'dimensions', 'dataType', 'blockSize',
                      'timeindex', 'channels'):
                arr_attrs.pop(k, None)
            dataset_attrs = {**arr_attrs, **dataset_attrs}
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
    # extract components of dataset path (data_subpath)
    dataset_path_arg = data_subpath if data_subpath is not None else ''
    dataset_comps = [c for c in dataset_path_arg.split('/') if c]
    dataset_comps_index = 0

    # Look for the first subpath that containes .zattrs or attributes.json file
    while dataset_comps_index < len(dataset_comps):
        container_subpath = '/'.join(dataset_comps[0:dataset_comps_index])
        container_path = f'{data_path}/{container_subpath}'
        if (os.path.exists(f'{container_path}/.zattrs') or
            os.path.exists(f'{container_path}/zarr.json')):
            break
        dataset_comps_index = dataset_comps_index + 1

    # move subpath components to container path to satisfy requirement
    appended_container_path = '/'.join(dataset_comps[0:dataset_comps_index])
    container_path = f'{data_path}/{appended_container_path}'
    new_subpath = '/'.join(dataset_comps[dataset_comps_index:])

    return container_path, new_subpath


def _get_data_store(data_path):
    """Get a local zarr store"""
    return data_path


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


