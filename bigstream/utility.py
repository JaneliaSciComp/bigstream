import numpy as np
from scipy.spatial.transform import Rotation
import SimpleITK as sitk
import zarr
from zarr.indexing import BasicIndexer
from numcodecs import Blosc
from distributed import Lock


def skip_sample(image, spacing, ss_spacing):
    """
    """

    spacing = np.array(spacing)
    ss = np.maximum(np.round(ss_spacing / spacing), 1).astype(int)
    slc = tuple(slice(None, None, x) for x in ss)
    return image[slc], spacing * ss


def numpy_to_sitk(image, spacing=None, origin=None, vector=False):
    """
    """

    # check endianness of data - some sitk operations seem to
    # only work with little endian
    if str(image.dtype)[0] == '>':
        error = "Array cannot be big endian. Convert arrays with ndarray.astype\n"
        error += "Given array dtype is " + str(image.dtype)
        raise TypeError(error)

    image = sitk.GetImageFromArray(image, isVector=vector)
    if spacing is None: spacing = np.ones(image.GetDimension())
    image.SetSpacing(spacing[::-1])
    if origin is None: origin = np.zeros(image.GetDimension())
    image.SetOrigin(origin[::-1])
    return image


def invert_matrix_axes(matrix):
    """
    """

    corrected = np.eye(4)
    corrected[:3, :3] = matrix[:3, :3][::-1, ::-1]
    corrected[:3, -1] = matrix[:3, -1][::-1]
    return corrected


def change_affine_matrix_origin(matrix, origin):
    """
    """

    tl, tr = np.eye(4), np.eye(4)
    origin = np.array(origin)
    tl[:3, -1], tr[:3, -1] = -origin, origin
    return np.matmul(tl, np.matmul(matrix, tr))


def affine_transform_to_matrix(transform):
    """
    """

    matrix = np.eye(4)
    matrix[:3, :3] = np.array(transform.GetMatrix()).reshape((3,3))
    matrix[:3, -1] = np.array(transform.GetTranslation())
    return invert_matrix_axes(matrix)


def matrix_to_affine_transform(matrix):
    """
    """

    matrix_sitk = invert_matrix_axes(matrix)
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(matrix_sitk[:3, :3].flatten())
    transform.SetTranslation(matrix_sitk[:3, -1].squeeze())
    return transform


def matrix_to_euler_transform(matrix):
    """
    """

    matrix_sitk = invert_matrix_axes(matrix)
    transform = sitk.Euler3DTransform()
    transform.SetMatrix(matrix_sitk[:3, :3].flatten())
    transform.SetTranslation(matrix_sitk[:3, -1].squeeze())
    return transform


def euler_transform_to_parameters(transform):
    """
    """

    return np.array((transform.GetAngleX(),
                     transform.GetAngleY(),
                     transform.GetAngleZ()) +
                     transform.GetTranslation()
    )


def parameters_to_euler_transform(params):
    """
    """

    transform = sitk.Euler3DTransform()
    transform.SetRotation(*params[:3])
    transform.SetTranslation(params[3:])
    return transform


def physical_parameters_to_affine_matrix(params, center):
    """
    """

    # translation
    aff = np.eye(4)
    aff[:3, -1] = params[:3]
    # rotation
    x = np.eye(4)
    x[:3, :3] = Rotation.from_rotvec(params[3:6]).as_matrix()
    x = change_affine_matrix_origin(x, -center)
    aff = np.matmul(x, aff)
    # scale
    x = np.diag(tuple(params[6:9]) + (1,))
    aff = np.matmul(x, aff)
    # shear
    shx, shy, shz = np.eye(4), np.eye(4), np.eye(4)
    shx[1, 0], shx[2, 0] = params[10], params[11]
    shy[0, 1], shy[2, 1] = params[9], params[11]
    shz[0, 2], shz[1, 2] = params[9], params[10]
    x = np.matmul(shz, np.matmul(shy, shx))
    return np.matmul(x, aff)


def matrix_to_displacement_field(matrix, shape, spacing=None):
    """
    """

    if spacing is None: spacing = np.ones(len(shape))
    nrows, ncols, nstacks = shape
    grid = np.array(np.mgrid[:nrows, :ncols, :nstacks])
    grid = grid.transpose(1,2,3,0) * spacing
    mm, tt = matrix[:3, :3], matrix[:3, -1]
    return np.einsum('...ij,...j->...i', mm, grid) + tt - grid


def field_to_displacement_field_transform(field, spacing=None, origin=None):
    """
    """

    field = field.astype(np.float64)[..., ::-1]
    transform = numpy_to_sitk(field, spacing, origin, vector=True)
    return sitk.DisplacementFieldTransform(transform)


def bspline_parameters_to_transform(parameters):
    """
    """

    t = sitk.BSplineTransform(3, 3)
    t.SetFixedParameters(parameters[:18])
    t.SetParameters(parameters[18:])
    return t


def bspline_to_displacement_field(
    bspline, shape, spacing=None, origin=None, direction=None,
):
    """
    """

    if spacing is None: spacing = np.ones(len(shape))
    if origin is None: origin = np.zeros(len(shape))
    if direction is None: direction = np.eye(len(shape))
    df = sitk.TransformToDisplacementField(
        bspline, sitk.sitkVectorFloat64,
        shape[::-1], origin[::-1], spacing[::-1],
        direction[::-1, ::-1].ravel(),
    )
    return sitk.GetArrayFromImage(df).astype(np.float32)[..., ::-1]


# TODO: function that takes a numpy array and return transform type


def relative_spacing(query, reference, reference_spacing):
    """
    """

    ndim = len(reference_spacing)
    ratio = np.array(reference.shape[:ndim]) / query.shape[:ndim]
    return reference_spacing * ratio



def transform_list_to_composite_transform(transform_list, spacing=None, origin=None):
    """
    """

    transform = sitk.CompositeTransform(3)
    for iii, t in enumerate(transform_list):
        if t.shape == (4, 4):
            t = matrix_to_affine_transform(t)
        elif len(t.shape) == 1:
            t = bspline_parameters_to_transform(t)
        else:
            a = spacing[iii] if isinstance(spacing, tuple) else spacing
            b = origin[iii] if isinstance(origin, tuple) else origin
            t = field_to_displacement_field_transform(t, a, b)
        transform.AddTransform(t)
    return transform


def create_zarr(path, shape, chunks, dtype, chunk_locked=False, client=None):
    """
    """

    compressor = Blosc(
        cname='zstd', clevel=4, shuffle=Blosc.BITSHUFFLE,
    )
    zarr_disk = zarr.open(
        path, 'w',
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        compressor=compressor,
    )

    # this code is currently never used within bigstream
    # keeping it aroung in case a use case comes up
    if chunk_locked:
        indexer = BasicIndexer(slice(None), zarr_disk)
        keys = (zarr_disk._chunk_key(idx.chunk_coords) for idx in indexer)
        lock = {key: Lock(key, client=client) for key in keys}
        lock['.zarray'] = Lock('.zarray', client=client)
        zarr_disk = zarr.open(
            store=zarr_disk.store, path=zarr_disk.path,
            synchronizer=lock, mode='r+',
        )

    return zarr_disk
    

def numpy_to_zarr(array, chunks, path):
    """
    """

    if not isinstance(array, zarr.Array):
        zarr_disk = create_zarr(path, array.shape, chunks, array.dtype)
        zarr_disk[...] = array
        return zarr_disk
    else:
        return array

