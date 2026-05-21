import json
import os

import numpy as np
import pytest
import zarr

import ngff_zarr

from bigstream import io_utility


def _axes_zyx():
    return [
        {'name': 'z', 'type': 'space', 'unit': 'micrometer'},
        {'name': 'y', 'type': 'space', 'unit': 'micrometer'},
        {'name': 'x', 'type': 'space', 'unit': 'micrometer'},
    ]


def _scale(s):
    return {'type': 'scale', 'scale': list(s)}


def _make_ome_container(tmp_path, name, zarr_format, dataset_path='s0',
                       shape=(64, 64, 64), chunks=(32, 32, 32), scale=(1.0, 1.0, 1.0)):
    container = tmp_path / name
    parent_attrs = io_utility.prepare_parent_group_attrs(
        str(container),
        dataset_path,
        axes=_axes_zyx(),
        dataset_transformations=[_scale(scale)],
        zarr_format=zarr_format,
    )
    arr = io_utility.create_dataset_array(
        str(container),
        dataset_path,
        shape=shape,
        chunks=chunks,
        dtype='uint16',
        overwrite=True,
        parent_attrs=parent_attrs,
        zarr_format=zarr_format,
    )
    arr[...] = np.arange(int(np.prod(shape)), dtype='uint16').reshape(shape) % 1000
    return container


def test_v04_metadata_layout(tmp_path):
    container = _make_ome_container(tmp_path, 'v04.zarr', zarr_format=2)
    zattrs = json.loads((container / '.zattrs').read_text())
    assert 'multiscales' in zattrs
    assert 'ome' not in zattrs
    ms = zattrs['multiscales'][0]
    assert ms['version'] == '0.4'
    assert [ax['name'] for ax in ms['axes']] == ['z', 'y', 'x']
    assert ms['datasets'][0]['path'] == 's0'
    cts = ms['datasets'][0]['coordinateTransformations']
    assert cts[0]['type'] == 'scale'
    assert cts[0]['scale'] == [1.0, 1.0, 1.0]


def test_v05_metadata_layout(tmp_path):
    container = _make_ome_container(tmp_path, 'v05.zarr', zarr_format=3)
    zjson = json.loads((container / 'zarr.json').read_text())
    attributes = zjson['attributes']
    assert 'ome' in attributes
    assert 'multiscales' not in attributes
    ome = attributes['ome']
    assert ome['version'] == '0.5'
    ms = ome['multiscales'][0]
    assert [ax['name'] for ax in ms['axes']] == ['z', 'y', 'x']
    assert ms['datasets'][0]['path'] == 's0'


def test_write_v04_then_read_via_ngff_zarr(tmp_path):
    container = _make_ome_container(tmp_path, 'rt_v04.zarr', zarr_format=2,
                                    scale=(2.0, 3.0, 4.0))
    multiscales = ngff_zarr.from_ngff_zarr(str(container), version='0.4')
    md = multiscales.metadata
    assert [a.name for a in md.axes] == ['z', 'y', 'x']
    assert md.datasets[0].path == 's0'
    scale = md.datasets[0].coordinateTransformations[0].scale
    assert list(scale) == [2.0, 3.0, 4.0]


def test_write_v05_then_read_via_ngff_zarr(tmp_path):
    container = _make_ome_container(tmp_path, 'rt_v05.zarr', zarr_format=3,
                                    scale=(0.5, 1.5, 2.5))
    multiscales = ngff_zarr.from_ngff_zarr(str(container), version='0.5')
    md = multiscales.metadata
    assert [a.name for a in md.axes] == ['z', 'y', 'x']
    assert md.datasets[0].path == 's0'
    scale = md.datasets[0].coordinateTransformations[0].scale
    assert list(scale) == [0.5, 1.5, 2.5]


def test_roundtrip_voxel_spacing_v04(tmp_path):
    container = _make_ome_container(tmp_path, 'spacing_v04.zarr', zarr_format=2,
                                    scale=(1.5, 2.5, 3.5))
    data, attrs = io_utility.open_image_container(str(container), 's0')
    spacing = io_utility.get_voxel_spacing(attrs)
    assert spacing is not None
    np.testing.assert_array_equal(spacing, np.array([1.5, 2.5, 3.5]))


def test_roundtrip_voxel_spacing_v05(tmp_path):
    container = _make_ome_container(tmp_path, 'spacing_v05.zarr', zarr_format=3,
                                    scale=(1.5, 2.5, 3.5))
    data, attrs = io_utility.open_image_container(str(container), 's0')
    spacing = io_utility.get_voxel_spacing(attrs)
    assert spacing is not None
    np.testing.assert_array_equal(spacing, np.array([1.5, 2.5, 3.5]))


def test_incremental_multiscale_merge_v04(tmp_path):
    container = tmp_path / 'merge_v04.zarr'
    attrs_s0 = io_utility.prepare_parent_group_attrs(
        str(container), 's0',
        axes=_axes_zyx(),
        dataset_transformations=[_scale((1.0, 1.0, 1.0))],
        zarr_format=2,
    )
    io_utility.create_dataset_array(
        str(container), 's0',
        shape=(32, 32, 32), chunks=(16, 16, 16),
        dtype='uint16', overwrite=True,
        parent_attrs=attrs_s0, zarr_format=2,
    )
    attrs_s1 = io_utility.prepare_parent_group_attrs(
        str(container), 's1',
        axes=_axes_zyx(),
        dataset_transformations=[_scale((2.0, 2.0, 2.0))],
        zarr_format=2,
    )
    io_utility.create_dataset_array(
        str(container), 's1',
        shape=(16, 16, 16), chunks=(8, 8, 8),
        dtype='uint16', overwrite=True,
        parent_attrs=attrs_s1, zarr_format=2,
    )
    zattrs = json.loads((container / '.zattrs').read_text())
    paths = [d['path'] for d in zattrs['multiscales'][0]['datasets']]
    assert paths == ['s0', 's1']


def test_incremental_multiscale_merge_v05(tmp_path):
    container = tmp_path / 'merge_v05.zarr'
    for sub, sc in (('s0', (1.0, 1.0, 1.0)), ('s1', (2.0, 2.0, 2.0))):
        parent_attrs = io_utility.prepare_parent_group_attrs(
            str(container), sub,
            axes=_axes_zyx(),
            dataset_transformations=[_scale(sc)],
            zarr_format=3,
        )
        io_utility.create_dataset_array(
            str(container), sub,
            shape=(16, 16, 16), chunks=(8, 8, 8),
            dtype='uint16', overwrite=True,
            parent_attrs=parent_attrs, zarr_format=3,
        )
    zjson = json.loads((container / 'zarr.json').read_text())
    paths = [d['path'] for d in zjson['attributes']['ome']['multiscales'][0]['datasets']]
    assert paths == ['s0', 's1']


def test_non_ome_input_voxel_spacing_is_none(tmp_path):
    container = tmp_path / 'plain.zarr'
    zarr.create_array(
        store=str(container),
        shape=(16, 16, 16),
        chunks=(8, 8, 8),
        dtype='uint16',
        zarr_format=2,
    )
    data, attrs = io_utility.open_image_container(str(container), None)
    spacing = io_utility.get_voxel_spacing(attrs)
    assert spacing is None


def test_open_with_block_coords(tmp_path):
    container = _make_ome_container(tmp_path, 'block.zarr', zarr_format=2,
                                    shape=(32, 32, 32), chunks=(16, 16, 16))
    block = (slice(0, 8), slice(0, 8), slice(0, 8))
    data, _ = io_utility.open_image_container(str(container), 's0', block_coords=block)
    assert isinstance(data, np.ndarray)
    assert data.shape == (8, 8, 8)


def test_attrs_view_keys_present(tmp_path):
    """ImageData.get_attr depends on these keys being present on the view dict."""
    container = _make_ome_container(tmp_path, 'view.zarr', zarr_format=3,
                                    scale=(1.0, 2.0, 3.0))
    attrs = io_utility.read_image_container_attributes(str(container), 's0')
    for key in ('axes', 'coordinateTransformations', 'globalCoordinateTransformations',
                'dimensions', 'dataType', 'blockSize', 'dataset_path'):
        assert key in attrs, f'missing key {key}'
