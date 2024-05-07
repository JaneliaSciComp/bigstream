import numpy as np

from .io_utility import open as img_open, read_attributes, get_voxel_spacing


class ImageData:

    def __init__(self, image_path=None, image_subpath=None,
                 image_arraydata=None, with_attrs=True):
        self.image_path = image_path
        self.image_subpath = image_subpath
        self.image_ndarray = image_arraydata
        self.image_voxel_spacing = None
        self.image_attrs = None
        if with_attrs:
            self.read_attrs()

    def __str__(self):
        image_path = self.image_path if self.image_path else '???'
        subpath = self.image_subpath if self.image_subpath else ''

        return f'{image_path}:{subpath} ({self.shape}) {self.voxel_spacing}'

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
            return self.image_voxel_spacing
        elif self.attrs:
            return get_voxel_spacing(self.attrs)
        else:
            return None

    @voxel_spacing.setter    
    def voxel_spacing(self, value):
        self.image_voxel_spacing = value
