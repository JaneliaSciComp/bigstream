import zarr
import numpy as np
import sys
import n5_metadata_utils as n5mu
from scipy.ndimage import map_coordinates
from os.path import splitext, abspath, isdir


def position_grid(sh, dtype=np.uint16):
    """Return a position array in physical coordinates with shape sh"""
    coords = np.array(np.meshgrid(*[range(x) for x in sh], indexing='ij'), dtype=dtype)
    return np.ascontiguousarray(np.moveaxis(coords, 0, -1))


def transform_grid(matrix, grid):
    """Apply affine matrix to position grid"""
    mm = matrix[:, :-1]
    tt = matrix[:, -1]
    return np.einsum('...ij,...j->...i', mm, grid) + tt


def interpolate_image(img, X, order=1):
    """Interpolate image at coordinates X"""
    X = np.moveaxis(X, -1, 0)
    return map_coordinates(img, X, order=order, mode='constant')


def read_n5_data(n5_path, subpath):
    im = zarr.open(store=zarr.N5Store(n5_path), mode='r')[subpath]
    return im[...].transpose(2, 1, 0)  # zarr loads zyx order


def write_n5(n5_path, subpath, im):
    im = im.transpose(2, 1, 0)  # zarr writes zyx order
    out = zarr.open(store=zarr.N5Store(n5_path), mode='a')
    out.create_dataset(subpath, shape=im.shape, chunks=(70, 128, 128), dtype=im.dtype)
    out[subpath][:, :, :] = im


def read_n5_transform(n5_path, subpath):
    txm_n5 = zarr.open(store=zarr.N5Store(n5_path), mode='r')
    grid = txm_n5['/c0'+subpath].shape[::-1]
    txm = np.empty(grid + (3,))
    txm[..., 0] = txm_n5['/c0'+subpath][...].transpose(2, 1, 0)
    txm[..., 1] = txm_n5['/c1'+subpath][...].transpose(2, 1, 0)
    txm[..., 2] = txm_n5['/c2'+subpath][...].transpose(2, 1, 0)
    return txm



if __name__ == '__main__':

    ref_img_path     = sys.argv[1]
    ref_img_subpath  = sys.argv[2]
    mov_img_path     = sys.argv[3]
    mov_img_subpath  = sys.argv[4]
    txm_path         = sys.argv[5]
    out_path         = sys.argv[6]

    points_path = None
    if len(sys.argv) == 8:
        points_path = sys.argv[7]


    ext   = splitext(txm_path)[1]
    vox   = n5mu.read_voxel_spacing(mov_img_path, mov_img_subpath)
    if ext == '.mat':
        matrix     = np.float32(np.loadtxt(txm_path))
        grid       = n5mu.read_voxel_grid(ref_img_path, ref_img_subpath)
        grid       = position_grid(grid) * vox
        grid       = transform_grid(matrix, grid)
    elif ext in ['', '.n5']:
        grid       = read_n5_transform(txm_path, ref_img_subpath)

    if points_path is None:
        im  = read_n5_data(mov_img_path, mov_img_subpath)
        im  = interpolate_image(im, grid/vox)
        write_n5(out_path, ref_img_subpath, im)
        n5mu.transfer_metadata(ref_img_path, ref_img_subpath, out_path, ref_img_subpath)
    else:
        points                 = np.float32(np.loadtxt(points_path, delimiter=','))
        warped_points          = np.empty_like(points)
        warped_points[:, 0]    = interpolate_image(grid[..., 0], points[:, :3]/vox)
        warped_points[:, 1]    = interpolate_image(grid[..., 1], points[:, :3]/vox)
        warped_points[:, 2]    = interpolate_image(grid[..., 2], points[:, :3]/vox)
        warped_points[:, -1]   = points[:, -1]
        np.savetxt(out_path, warped_points, delimiter=',')


