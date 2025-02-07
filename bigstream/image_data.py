import numpy as np
import zarr

from .io_utility import open as img_open, read_attributes, get_voxel_spacing


class ImageData:

    def __init__(self, image_path=None, image_subpath=None,
                 image_arraydata=None, image_attrs=None,
                 read_attrs=True):
        self.image_path = image_path
        self.image_subpath = image_subpath
        self.image_ndarray = image_arraydata
        self.image_voxel_spacing = None
        self.image_downsampling = None
        self.image_attrs = image_attrs
        if image_attrs is None and read_attrs:
            self.read_attrs()

    def __str__(self):
        image_path = self.image_path if self.image_path else '???'
        subpath = self.image_subpath if self.image_subpath else ''

        return f'{image_path}:{subpath} {self.shape} {self.voxel_spacing} {self.downsampling}'

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
        else:
            return (self.image_attrs.get('dimensions')
                    if self.image_attrs else ())
    
    @property
    def shape_arr(self):
        s = self.shape
        if s:
            return np.array(s)
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
    def voxel_spacing(self):
        if self.image_voxel_spacing is not None:
            # no rotation or scaling here
            # the caller should have set them as needed
            return self.image_voxel_spacing
        elif self.attrs:
            voxel_spacing_from_attrs = get_voxel_spacing(self.attrs)
            if voxel_spacing_from_attrs is not None:
                # make it zyx
                return voxel_spacing_from_attrs[::-1]

        return np.ones(self.shape) # default voxel_spacing to 1

    @voxel_spacing.setter
    def voxel_spacing(self, value):
        self.image_voxel_spacing = np.array(value)

    @property
    def downsampling(self):
        image_downsampling = None
        if self.image_downsampling is not None:
            image_downsampling = self.image_downsampling
        elif self.attrs and self.attrs.get('downsamplingFactors'):
            # cenvert the downsampling factors to zyx
            image_downsampling = np.array(self.attrs.get('downsamplingFactors'))[::-1]
        if image_downsampling is None:
            image_downsampling = np.array((1,) * self.ndim)

        return image_downsampling

    @downsampling.setter
    def downsampling(self, value):
        self.image_downsampling = value

    def get_downsampled_voxel_resolution(self, to_xyz=True):
        voxel_spacing = self.voxel_spacing
        if voxel_spacing is not None:
            downsampled_spacing = self.voxel_spacing / self.downsampling
            return downsampled_spacing[::-1] if to_xyz else downsampled_spacing
        else:
            return None


def as_image_data(image_data):
    if isinstance(image_data, ImageData):
        return image_data
    elif isinstance(image_data, np.ndarray):
        return ImageData(image_arraydata=image_data, read_attrs=False)
    elif isinstance(image_data, zarr.core.Array):
        return ImageData(image_arraydata=image_data,
                         image_attrs=image_data.attrs)
    else:
        return None
