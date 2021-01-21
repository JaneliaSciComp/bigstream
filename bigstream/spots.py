#!/usr/env/python3

# Difference-of-Gaussian spot detection
# Author: TW
# Date: June 2019

# Modified by GF for local testing

from scipy import spatial
from scipy.ndimage import convolve, map_coordinates
from scipy.ndimage.filters import maximum_filter
from scipy.stats import multivariate_normal
import numpy as np
import sys
import pickle
import n5_metadata_utils as n5mu
from skimage.feature.blob import _blob_overlap


def save_PKL(filename,var):
    f=open(filename,'wb')
    pickle.dump(var,f)
    f.close()


def gauss_conv_volume(image, sigmas):
    # implement DOG locally
    x = int(round(2*sigmas[1]))  # sigmas[1] > sigmas[0]
    x = np.arange(-x, x+.5, 1)
    x = np.moveaxis(np.array(np.meshgrid(x, x, x)), 0, -1)
    g1 = multivariate_normal.pdf(x, mean=[0,]*3, cov=np.diag([sigmas[0],]*3))
    g2 = multivariate_normal.pdf(x, mean=[0,]*3, cov=np.diag([sigmas[1],]*3))
    return convolve(image, g1 - g2)


def get_local_max(image, min_distance=3, threshold=None):
    image_max = maximum_filter(image, min_distance)
    if threshold:
        mask = np.logical_and(image == image_max, image >= threshold) # threshold of DoG NOT raw!
    else:
        mask = image == image_max
    return np.column_stack(np.nonzero(mask))


def tw_blob_dog(image, min_sigma=1, sigma_ratio=1.6, threshold=2.0, min_distance=5, overlap=.5):
    DoG = gauss_conv_volume(image, [min_sigma, min_sigma*sigma_ratio])
    coord = get_local_max(DoG, min_distance=min_distance)
    intensities = image[coord[:,0], coord[:,1], coord[:,2]]
    filtered = intensities > threshold
    coord = np.hstack( (coord[filtered],
                        np.array( [intensities[filtered]] ).T, 
                        np.full( (sum(filtered), 1), min_sigma) ) )
    return coord


def get_context(img, pos, radius, interpmap=False):
    w = img[pos[0]-radius:pos[0]+radius+1, pos[1]-radius:pos[1]+radius+1, pos[2]-radius:pos[2]+radius+1]
    if np.product(w.shape) != (2*radius+1)**3: #just ignore near edge
        return(False)
    if interpmap:
        return(map_coordinates(w,interpmap,order=3).reshape((width[0],width[1],width[2])))
    else:
        return(w)


def scan(img, spots, radius, interpmap=None):
    output=[]
    for spot in spots:
        w = get_context(img, spot, radius, interpmap)
        if type(w) != bool:
            output.append([spot, w])
    return(output)


def prune_blobs(blobs_array, overlap, distance):
    tree = spatial.cKDTree(blobs_array[:, :-2])
    pairs = np.array(list(tree.query_pairs(distance)))
    for (i, j) in pairs:
        blob1, blob2 = blobs_array[i], blobs_array[j]
        if _blob_overlap(blob1, blob2) > overlap:
            if blob1[-2] > blob2[-2]:
                blob2[-2] = 0
            else:
                blob1[-2] = 0
    return np.array([b for b in blobs_array if b[-2] > 0])


def read_coords(path):
    with open(path, 'r') as f:
        offset = np.array(f.readline().split(' ')).astype(np.float64)
        extent = np.array(f.readline().split(' ')).astype(np.float64)
    return offset, extent




# MAIN

mode          = sys.argv[1]
img_path      = sys.argv[2]
img_subpath   = sys.argv[3]
outpath       = sys.argv[4]
radius        = np.uint16(sys.argv[5])
spotNum       = np.uint16(sys.argv[6])

# get the data
# zarr reads data as zyx
zarr_im = zarr.open(store=zarr.N5Store(img_path), mode='r')[img_subpath]
vox = n5mu.read_voxel_spacing(img_path, img_subpath).astype(np.float64)
if mode != 'coarse':
    offset, extent = read_coords(mode)
    oo = np.round(offset/vox).astype(np.int16)
    ee = oo + np.round(extent/vox).astype(np.int16)
    oo_rad = np.maximum(0, oo-radius)
    ee_rad = ee + radius
    im = zarr_im[oo_rad[2]:ee_rad[2], oo_rad[1]:ee_rad[1], oo_rad[0]:ee_rad[0]]
else:
    im = zarr_im[:, :, :]
im = np.moveaxis(im, (0, 2), (2, 0))
im = im.astype(np.float64)

# get the spots
coord = tw_blob_dog(im, 1, 2)

# throw out spots in the margins
if mode != 'coarse':
    filtered = np.logical_and(coord[:, :3] >= (oo - oo_rad), coord[:, :3] < (ee - oo_rad))
    filtered = filtered[:, 0] * filtered[:, 1] * filtered[:, 2]
    coord = coord[filtered]


sortIdx = np.argsort(coord[:, -2])[::-1]

# prune the spots
if mode == 'coarse':
    sortedSpots=coord[sortIdx,:][:spotNum * 4]
    sortIdx=np.argsort(sortedSpots[:,0])
    sortedSpots=sortedSpots[sortIdx,:][::2]
    sortIdx=np.argsort(sortedSpots[:,1])
    sortedSpots=sortedSpots[sortIdx,:][::2]
else:
    sortedSpots=coord[sortIdx,:][:spotNum]

# final prune and save
min_distance = 6
overlap = 0.01
pruned_spots = prune_blobs(sortedSpots, overlap, min_distance)[:,:-2].astype(np.int)
context = scan(im, pruned_spots, radius)

# correct offset to remove radius
if mode != 'coarse':
    points = [ [(p[0] - (oo - oo_rad))*vox, p[1]] for p in context ]
else:
    points = [ [p[0]*vox, p[1]] for p in context]

# write output
save_PKL(outpath, points)

