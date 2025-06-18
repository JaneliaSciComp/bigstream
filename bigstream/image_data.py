import numpy as np
import zarr

from .io_utility import open as img_open, read_attributes, get_voxel_spacing


class ImageData:

    def __init__(self, image_path=None, image_subpath=None,
                 image_arraydata=None, image_attrs=None,
                 image_timeindex=None, image_channels=None,
                 read_attrs=True):
        self.image_path = image_path
        self.image_subpath = image_subpath
        self.image_timeindex = image_timeindex
        self.image_channels = image_channels
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
            f'{image_path}:{subpath}:{self.image_timeindex}:{self.image_channels} '
            f'{self.shape} {self.voxel_spacing} {self.downsampling} '
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
    
    def read_image(self):
        if self.image_path:
            self.image_ndarray, self.image_attrs = img_open(self.image_path,
                                                            self.image_subpath)

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
        if self.image_ndarray:
            return self.image_ndarray.shape
        elif self.image_attrs:
            return self._shape_from_attrs(self.image_attrs)               
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

    @property
    def downsampling(self):
        """
        Return the image downsampling as TCZYX
        """
        image_downsampling = None
        if self.image_downsampling is not None:
            image_downsampling = self.image_downsampling
        elif self.attrs and self.attrs.get('downsamplingFactors'):
            # convert the downsampling factors to tczyx
            image_downsampling = np.array(self.attrs.get('downsamplingFactors'))[::-1]
        if image_downsampling is None:
            image_downsampling = np.array((1,) * self.ndim)

        return image_downsampling

    @downsampling.setter
    def downsampling(self, value):
        self.image_downsampling = value

    def get_full_voxel_resolution(self):
        voxel_spacing = self.voxel_spacing
        if voxel_spacing is not None:
            return self.voxel_spacing / self.downsampling
        else:
            return None


def as_image_data(image_data, image_timeindex=None, image_channels=None):
    if isinstance(image_data, ImageData):
        return image_data
    elif isinstance(image_data, np.ndarray):
        return ImageData(image_arraydata=image_data, read_attrs=False,
                         image_timeindex=image_timeindex,
                         image_channels=image_channels)
    elif isinstance(image_data, zarr.core.Array):
        return ImageData(image_arraydata=image_data,
                         image_attrs=image_data.attrs,
                         image_timeindex=image_timeindex,
                         image_channels=image_channels)
    else:
        return None


def get_spatial_values(values):
    if values is None:
        return None
    if len(values) > 3:
        return values[-3:]
    return values