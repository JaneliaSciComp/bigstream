import json

import numpy as np
import pytest
import zarr

import bigstream.io_utility as io
import bigstream.utility as ut
from bigstream.image_data import ImageData

DATA = np.arange(4 * 6 * 8, dtype='u2').reshape(4, 6, 8)
CHUNKS = (2, 3, 4)


def _write_group(path, zarr_format, dtype='u2', group_attrs=None):
    group = zarr.open_group(path, mode='w', zarr_format=zarr_format)
    group.create_array('s0', shape=DATA.shape, chunks=CHUNKS, dtype=dtype)[:] = DATA
    if group_attrs:
        group.attrs.update(group_attrs)
    return path


@pytest.mark.parametrize('zarr_format', [2, 3])
def test_open_reads_zarr2_and_zarr3(tmp_path, zarr_format):
    path = _write_group(str(tmp_path / 'in.zarr'), zarr_format)
    block = (slice(1, 3), slice(None), slice(2, 5))
    data, _ = io.open(path, 's0', block_coords=block)
    assert np.array_equal(data, DATA[block])


def test_ome_zarr_05_voxel_spacing(tmp_path):
    multiscales = [{
        'version': '0.5',
        'axes': [{'name': n, 'type': 'space'} for n in 'zyx'],
        'datasets': [{'path': 's0', 'coordinateTransformations': [
            {'type': 'scale', 'scale': [2.0, 1.0, 0.5]}]}],
    }]
    path = _write_group(str(tmp_path / 'in.zarr'), 3,
                        group_attrs={'ome': {'version': '0.5', 'multiscales': multiscales}})
    spacing = io.get_voxel_spacing(io.read_attributes(path, 's0'))
    assert np.allclose(spacing, [2.0, 1.0, 0.5])


def test_n5_path_raises(tmp_path):
    path = tmp_path / 'in.n5'
    path.mkdir()
    with pytest.raises(ValueError, match='N5 is not supported'):
        io.open(str(path), 's0')
    with pytest.raises(ValueError, match='N5 is not supported'):
        io.create_dataset(str(tmp_path / 'out.n5'), 's0', DATA.shape, CHUNKS, 'u2')


@pytest.mark.parametrize('kwargs, expected_format', [({}, 3), ({'zarr_format': 2}, 2)])
def test_create_dataset_format(tmp_path, kwargs, expected_format):
    path = str(tmp_path / 'out.zarr')
    array = io.create_dataset(path, 's0', DATA.shape, CHUNKS, 'u2',
                              compressor='zstd', **kwargs)
    array[:] = DATA
    assert array.metadata.zarr_format == expected_format
    data, _ = io.open(path, 's0')
    assert np.array_equal(data[:], DATA)


def test_create_dataset_zstd_level_3(tmp_path):
    path = tmp_path / 'out.zarr'
    io.create_dataset(str(path), 's0', DATA.shape, CHUNKS, 'u2', compressor='zstd')
    codecs = json.loads((path / 's0' / 'zarr.json').read_text())['codecs']
    zstd = [c for c in codecs if c['name'] == 'zstd']
    assert zstd and zstd[0]['configuration']['level'] == 3


@pytest.mark.parametrize('compressor', ['lz4', 'bz2'])
def test_nonstandard_codec_only_for_zarr2(tmp_path, compressor):
    with pytest.raises(ValueError, match='not a standard zarr 3 codec'):
        io.validate_compression(compressor, 3)
    io.validate_compression(compressor, 2)
    array = io.create_dataset(str(tmp_path / 'out.zarr'), 's0', DATA.shape, CHUNKS, 'u2',
                              compressor=compressor, zarr_format=2)
    array[:] = DATA
    assert np.array_equal(array[:], DATA)


def test_read_image_converts_big_endian(tmp_path):
    path = _write_group(str(tmp_path / 'in.zarr'), 2, dtype='>u2')
    image = ImageData(path, 's0')
    image.read_image()
    assert image.image_ndarray.dtype == np.dtype('<u2')
    assert np.array_equal(image.image_ndarray, DATA)
    # the file on disk is not changed
    assert zarr.open_array(f'{path}/s0', mode='r').dtype == np.dtype('>u2')


@pytest.mark.parametrize('kwargs, expected_format', [({}, 3), ({'zarr_format': 2}, 2)])
def test_numpy_to_zarr_format(tmp_path, kwargs, expected_format):
    array = ut.numpy_to_zarr(DATA, CHUNKS, str(tmp_path / 'out.zarr'), **kwargs)
    assert array.metadata.zarr_format == expected_format
    assert np.array_equal(array[:], DATA)


@pytest.mark.parametrize('zarr_format, ome_version', [(3, '0.5'), (2, '0.4')])
def test_create_dataset_writes_ome_metadata(tmp_path, zarr_format, ome_version):
    path = str(tmp_path / 'out.zarr')
    parent_attrs = io.prepare_parent_group_attrs(
        path, 'grp/s0',
        axes=[{'name': n, 'type': 'space', 'unit': 'micrometer'} for n in 'zyx'],
        coordinateTransformations=[{'type': 'scale', 'scale': [2.0, 1.0, 0.5]}])
    array = io.create_dataset(path, 'grp/s0', DATA.shape, CHUNKS, 'u2',
                              parent_attrs=parent_attrs, zarr_format=zarr_format)
    array[:] = DATA
    group_attrs = zarr.open_group(f'{path}/grp', mode='r').attrs.asdict()
    if zarr_format == 3:
        assert group_attrs['ome']['version'] == ome_version
        assert 'multiscales' not in group_attrs
    else:
        assert group_attrs['multiscales'][0]['version'] == ome_version
    spacing = io.get_voxel_spacing(io.read_attributes(path, 'grp/s0'))
    assert np.allclose(spacing, [2.0, 1.0, 0.5])


@pytest.mark.parametrize('zarr_format', [3, 2])
def test_multiscale_pyramid(tmp_path, zarr_format):
    import bigstream.piecewise_transform as pt
    mov = np.random.default_rng(0).integers(0, 1000, size=(32, 32, 32), dtype=np.uint16)
    path = str(tmp_path / 'pyramid.zarr')
    pt.distributed_apply_transform_to_multiscale_pyramid(
        mov, mov, np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0]),
        [np.eye(4)], (16, 16, 16), path,
        scales=[(1, 1, 1), (2, 2, 2)], zarr_format=zarr_format,
        temporary_directory=str(tmp_path),
        cluster_kwargs={'cluster_type': 'local_cluster', 'n_workers': 1,
                        'threads_per_worker': 2, 'processes': False})
    s0, attrs0 = io.open(path, 's0')
    s1, attrs1 = io.open(path, 's1')
    assert s0.shape == (32, 32, 32) and s1.shape == (16, 16, 16)
    assert s0.metadata.zarr_format == zarr_format
    # identity warp: level 0 equals the input, except the last plane of each
    # axis (known edge problem, see the warp read region)
    assert np.array_equal(s0[:31, :31, :31], mov[:31, :31, :31])
    assert np.allclose(io.get_voxel_spacing(attrs0), [1.0, 1.0, 1.0])
    assert np.allclose(io.get_voxel_spacing(attrs1), [2.0, 2.0, 2.0])
    group_attrs = zarr.open_group(path, mode='r').attrs.asdict()
    assert ('ome' in group_attrs) == (zarr_format == 3)


def test_multiscale_pyramid_needs_scales(tmp_path):
    import bigstream.piecewise_transform as pt
    with pytest.raises(ValueError, match='scales must be a list'):
        pt.distributed_apply_transform_to_multiscale_pyramid(
            (32, 32, 32, np.uint16), np.zeros((32, 32, 32), np.uint16),
            np.ones(3), np.ones(3), [np.eye(4)], (16, 16, 16),
            str(tmp_path / 'p.zarr'), scales=None, cluster=object())


def test_shard_size_must_equal_chunks(tmp_path):
    path = tmp_path / 'out.zarr'
    io.create_dataset(str(path), 's0', DATA.shape, CHUNKS, 'u2', shard_size=CHUNKS)
    codecs = json.loads((path / 's0' / 'zarr.json').read_text())['codecs']
    assert 'sharding_indexed' not in [c['name'] for c in codecs]
    with pytest.raises(ValueError, match='must equal the chunk size'):
        io.create_dataset(str(tmp_path / 'b.zarr'), 's0', DATA.shape, CHUNKS, 'u2',
                          shard_size=(4, 6, 8))
    with pytest.raises(ValueError, match='must equal the chunk size'):
        ut.create_zarr(str(tmp_path / 'c.zarr'), DATA.shape, CHUNKS, 'u2',
                       shard_size=(4, 6, 8))
