import numpy as np
import zarr

from .io_utility import open as img_open, read_attributes, get_voxel_spacing


class ImageData:

    def __init__(self, image_path=None, image_subpath=None,
                 image_arraydata=None, image_attrs=None,
                 image_timeindex=None, image_channel=None,
                 read_attrs=True):
        self.image_path = image_path
        self.image_subpath = image_subpath
        self.image_timeindex = image_timeindex
        self.image_channel = image_channel
        self.image_ndarray = image_arraydata
        self.image_voxel_spacing = None
        self.image_downsampling = None
        self.image_attrs = image_attrs
        if image_attrs is None and read_attrs:
            self.read_attrs()

    def __str__(self):
        image_path = self.image_path if self.image_path else '???'
        subpath = self.image_subpath if self.image_subpath else ''

        return (
            f'{image_path}:{subpath}:{self.image_timeindex}:{self.image_channel}, '
            f'shape: {self.shape}, voxel_spacing: {self.voxel_spacing} '
        )

    def __getitem__(self, key):
        if self.image_ndarray is not None:
            return self.image_ndarray[key]
        else:
            return 0

    def has_data(self):
        return self.shape is not None and len(self.shape) > 0

    def read_attrs(self):
        if self.image_path:
            self.image_attrs = read_attributes(self.image_path, self.image_subpath)
    
    def read_image(self, convert_to_little_endian=True):
        if self.image_path:
            imgarray, self.image_attrs = img_open(
                self.image_path,
                self.image_subpath,
                data_timeindex=self.image_timeindex,
                data_channels=self.image_channel
            )
            imgdtype = imgarray.dtype
            if convert_to_little_endian and imgdtype.byteorder == '>':
                # Bigstream algorithms do not support big-endian data
                self.image_ndarray = imgarray.byteswap().newbyteorder('<')
            else:
                self.image_ndarray = imgarray

    @property
    def attrs(self):
        return self.image_attrs

    def get_attr(self, attrname):
        if self.attrs:
            return self.attrs.get(attrname)
        else:
            return None

    @property    
    def image_array(self):
        return self.image_ndarray

    @property
    def shape(self):
        if self.image_attrs:
            return self._shape_from_attrs(self.image_attrs)               
        elif self.image_ndarray is not None:
            return self.image_ndarray.shape
        else:
            return ()
    
    def _shape_from_attrs(self, attrs):
        if attrs.get('dimensions'):
            return attrs['dimensions']
        elif attrs.get('shape'):
            return attrs['shape']
        else:
            return ()

    @property
    def spatial_dims(self):
        s = self.shape
        if s:
            return np.array(s[-3:])
        else:
            None

    @property
    def dtype(self):
        if self.image_ndarray:
            return self.image_ndarray.dtype
        else:
            return (self.image_attrs.get('dataType')
                    if self.image_attrs else None)

    @property
    def ndim(self):
        return len(self.shape) if self.shape else 0

    @property
    def spatial_ndim(self):
        dims = self.spatial_dims
        return len(dims) if dims is not None else 0

    @property
    def voxel_downsampling(self):
        if self.image_downsampling is not None:
            return self.image_downsampling
        elif self.attrs:
            downsampling_from_attrs = self.attrs.get('downsamplingFactors')
            if downsampling_from_attrs is not None:
                return np.array(downsampling_from_attrs[::-1])

        return np.ones(len(self.shape))

    @property
    def voxel_spacing(self):
        if self.image_voxel_spacing is not None:
            # no rotation or scaling here
            # the caller should have set them as needed
            return self.image_voxel_spacing
        elif self.attrs:
            # read voxel spacing from attributes
            voxel_spacing_from_attrs = get_voxel_spacing(self.attrs)
            if voxel_spacing_from_attrs is not None:
                return voxel_spacing_from_attrs

        # voxel spacing was not set => default it to 1.
        return np.ones(len(self.shape))

    @voxel_spacing.setter
    def voxel_spacing(self, value):
        self.image_voxel_spacing = np.array(value)


def as_image_data(image_data, image_timeindex=None, image_channels=None):
    if isinstance(image_data, ImageData):
        return image_data
    elif isinstance(image_data, np.ndarray):
        return ImageData(image_arraydata=image_data, read_attrs=False,
                         image_timeindex=image_timeindex,
                         image_channel=image_channels)
    elif isinstance(image_data, zarr.core.Array):
        return ImageData(image_arraydata=image_data,
                         image_attrs=image_data.attrs,
                         image_timeindex=image_timeindex,
                         image_channel=image_channels)
    else:
        return None


def get_spatial_values(values, reverse_axes=False):
    if values is None:
        return None

    if len(values) > 3:
        if reverse_axes:
            return values[:3]
        else:
            return values[-3:]

    return values if not reverse_axes else values[::-1]


def calc_full_voxel_resolution_attr(voxel_spacing, downsampling):
    """
    Calculate voxel resolution in order to store it in the dataset attribute.
    """
    if voxel_spacing is None:
        return None

    if downsampling is None:
        return list(voxel_spacing)[::-1]

    return list(np.array(voxel_spacing) / downsampling)[::-1]


def calc_downsampling_attr(downsampling):
    if downsampling is None:
        return downsampling

    return list(downsampling)[::-1]
