import os
import nrrd
import numpy as np

import bigstream.n5_utils as n5_utils

from skimage import io

def open(container_path, subpath):
    path_comps = os.path.splitext(container_path)

    container_ext = path_comps[1]

    if container_ext == '.nrrd':
        return nrrd.read(container_path)
    elif container_ext == '.tif' or container_ext == '.tiff':
        im = io.imread(container_path)
        return im, {}
    elif container_ext == '.npy':
        im = np.load(container_path)
        return im, {}
    elif container_ext == '.n5':
        return n5_utils.open(container_path, subpath)
    else:
        print('Cannot handle', container_path, subpath)
        return None, {}
