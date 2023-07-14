import numpy as np
from fishspot.filter import white_tophat
from fishspot.detect import detect_spots_log, distributed_detect_spots_log
from fishspot.distribute import distributed_spot_detection
import os
import cv2


def blob_detection(
    image,
    min_blob_radius,
    max_blob_radius,
    distributed=True,
    cluster_name='blob',
    **kwargs,
):
    """
    """
    if not distributed:
        try:
            print("Running non distributed spots log on image size", image.shape)
        except:
            pass
        return blocked_blob_detection(
            image, min_blob_radius, max_blob_radius, **kwargs
        )
        spots, psf = detect_spots_log(
            array=image,
            blocksize=[256]*3,
            white_tophat_args={'radius': max_blob_radius},
            psf_estimation_args={'min_inliers': 2},
            deconvolution_args={},
            psf_retries=3,
            spot_detection_args={
                'min_radius': min_blob_radius,
                'max_radius': max_blob_radius,
                **kwargs
            },
            # cluster_kwargs={
            #         'queue':'braintv', # sbatch partition
            #         'cores':64,
            #         'processes':1,
            #         'memory':"256GB",
            #         'walltime':"24:00:00",
            #         'local_directory':'/scratch/fast/'+os.environ['SLURM_JOBID'],
            #         'log_directory':os.environ['jobdir']+'/dask_worker_logs/blob/',
            #         'n_workers': 32,
            # }
        )
        spots = spots.astype(int)
    else:
        wth = white_tophat(image, max_blob_radius)
        spots = distributed_detect_spots_log(
            wth,
            min_blob_radius,
            max_blob_radius,
            cluster_kwargs={
                    'queue':'braintv', # sbatch partition
                    'cores':64,
                    'processes':1,
                    'memory':"256GB",
                    'walltime':"24:00:00",
                    'local_directory':'/scratch/fast/'+os.environ['SLURM_JOBID'],
                    'log_directory':os.environ['jobdir']+'dask_worker_logs/'+cluster_name,
                    'job_directives_skip': ['-t'],
                    'job_extra_directives': ['-t 48:00:00', '--ntasks 64'],
                    'min_workers': 8
            },
            **kwargs,
        ).astype(int)
    intensities = image[spots[:, 0], spots[:, 1], spots[:, 2]]
    return np.hstack((spots[:, :3], intensities[..., None]))

def blocked_blob_detection(
    image,
    min_blob_radius,
    max_blob_radius,
    blocksize=[64]*3,
    **kwargs,
):
    """
    """

    wth = white_tophat(image, max_blob_radius)
    spots = detect_spots_log(
        wth,
        min_blob_radius,
        max_blob_radius,
        **kwargs,
    ).astype(int)

    

    intensities = image[spots[:, 0], spots[:, 1], spots[:, 2]]
    return np.hstack((spots[:, :3], intensities[..., None]))


# def get_spot_context(image, spots, vox, radius):
#     output = []
#     for spot in spots:
#         s = (spot / vox).astype(int)
#         x1, y1, z1 = np.maximum(s - radius, 0)
#         x2, y2, z2 = np.minimum(s + radius + 1, image.shape)
#         full_size_w = np.zeros((2 * radius + 1, 2 * radius + 1, 2 * radius + 1))
#         x_start, x_end = radius - (s[0] - x1), radius - (s[0] - x1) + (x2 - x1)
#         y_start, y_end = radius - (s[1] - y1), radius - (s[1] - y1) + (y2 - y1)
#         z_start, z_end = radius - (s[2] - z1), radius - (s[2] - z1) + (z2 - z1)
#         full_size_w[x_start:x_end, y_start:y_end, z_start:z_end] = image[x1:x2, y1:y2, z1:z2]
#         output.append([spot, full_size_w])
#     return output


def get_spot_context(image, spots, vox, radius):
    """
    """

    output = []
    for spot in spots:
        s = (spot/vox).astype(int)
        w = image[s[0]-radius:s[0]+radius+1,
                  s[1]-radius:s[1]+radius+1,
                  s[2]-radius:s[2]+radius+1]
    return output    





def get_spot_context_strict(image, spots, vox, radius, output_filepath, name='fix'):
    """
    """
    print("GETTING SPOT CONTEXT")
    print('image_shape', image.shape)
    output = []
    invalid_cropped = []
    invalid_zeros = []
    for spot in spots:
        s = (spot/vox).astype(int)
        w = image[s[0]-radius:s[0]+radius+1,
                  s[1]-radius:s[1]+radius+1,
                  s[2]-radius:s[2]+radius+1]
        if all(dim == 2*radius+1 for dim in w.shape):
            # spot is not near edge of image, not getting cropped
            if not (w == 0).any():
                # all values are non-zero, valid data
                output.append( [spot, w] )
            else:
                invalid_zeros.append(s)
        else:
            invalid_cropped.append(s)
    print("invalid_cropped", len(invalid_cropped))
    print("invalid_zeros", len(invalid_zeros))
    np.save(os.path.join(output_filepath,'{}_spots_cropped'.format(name)), invalid_cropped)
    np.save(os.path.join(output_filepath,'{}_spots_zeros'.format(name)), invalid_zeros)

    return output   

def get_spot_context_strict_warped(image, spots, vox, radius, affine_warp, name='fix'):
    
    """
    """
    output = []
    invalid_cropped = []
    invalid_zeros = []
    for spot in spots:
        s = (spot/vox).astype(int)
        w = image[s[0]-radius:s[0]+radius+1,
                  s[1]-radius:s[1]+radius+1,
                  s[2]-radius:s[2]+radius+1]
        if all(dim == 2*radius+1 for dim in w.shape):
            # spot is not near edge of image, not getting cropped
            if not (w == 0).any():
                # all values are non-zero, valid data
                output.append( [s, ndinterp.affine_transform(w, affine_warp, order=1, output_shape=w.shape)])

            else:
                invalid_zeros.append(s)
        else:
            invalid_cropped.append(s)
    print("invalid_cropped", len(invalid_cropped))
    print("invalid_zeros", len(invalid_zeros))
    np.save(os.path.join(os.environ['jobdir'],'{}_cropped'.format(name)), invalid_cropped)
    np.save(os.path.join(os.environ['jobdir'],'{}_zeros'.format(name)), invalid_zeros)

    return output  

def get_spot_context_strict_rotated(image, spots, vox, radius, name='fix'):
    """
    """
    output = []
    invalid_cropped = []
    invalid_zeros = []
    for spot in spots:
        s = (spot/vox).astype(int)
        w = image[s[0]-radius:s[0]+radius+1,
                  s[1]-radius:s[1]+radius+1,
                  s[2]-radius:s[2]+radius+1]
        if all(dim == 2*radius+1 for dim in w.shape):
            # spot is not near edge of image, not getting cropped
            if not (w == 0).any():
                # all values are non-zero, valid data
                output.append( [s, np.rot90(w, 1, axes=(1,2))] )
            else:
                invalid_zeros.append(s)
        else:
            invalid_cropped.append(s)
    print("invalid_cropped", len(invalid_cropped))
    print("invalid_zeros", len(invalid_zeros))
    np.save(os.path.join(os.environ['jobdir'],'{}_cropped'.format(name)), invalid_cropped)
    np.save(os.path.join(os.environ['jobdir'],'{}_zeros'.format(name)), invalid_zeros)

    return output  
def rotate_point(p, volume_shape):
    p_y = p[1]
    p[1] = p[0]
    p[0] = p_y
    p[0] = volume_shape[0] - p[0] - (volume_shape[0]-volume_shape[1])
    # points[:, 1] = 
    # points[:, 2] = -points[:, 2]
    return p

def _stats(arr):
    """
    """

    # compute mean and standard deviation along columns
    arr = arr.astype(np.float64)
    means = np.mean(arr, axis=1)
    sqr_means = np.mean(np.square(arr), axis=1)
    stddevs = np.sqrt( sqr_means - np.square(means) )
    # stddevs = np.std(arr, axis=1)
    print("MEANS< STDDEVS shape,", arr.shape, means.shape, stddevs.shape)
    return means, stddevs

def _blocked_stats(arr, blocksize=1024):
    """
    """
    import math

    # compute mean and standard deviation along columns
    arr = arr.astype(np.float64)
    means = []
    stddevs = []
    for i in range(math.ceil(arr.shape[0]/blocksize)):
        block_arr = arr[i*blocksize:min(arr.shape[1],(i+1)*blocksize),:]
        block_means = np.mean(block_arr, axis=1)
        block_sqr_means = np.mean(np.square(block_arr), axis=1)
        block_stddevs = np.sqrt( block_sqr_means - np.square(block_means) )
        means.extend(block_means)
        stddevs.extend(block_stddevs)
    means = np.array(means)
    stddevs = np.array(stddevs)
    # stddevs = np.std(arr, axis=1)
    print("MEANS< STDDEVS shape,", means.shape, stddevs.shape)
    return means, stddevs


def pairwise_correlation(A, B):
    """
    """
    print("IN PAIRWISE CORRELEATIONS")

    # grab and flatten context
    a_con = np.array( [a[1].flatten() for a in A] )
    b_con = np.array( [b[1].flatten() for b in B] )
    print(a_con.shape, b_con.shape)
    # get means and std for all contexts, center contexts
    a_mean, a_std = _stats(a_con)
    b_mean, b_std = _stats(b_con)
    a_con = a_con - a_mean[..., None]
    b_con = b_con - b_mean[..., None]

    # compute pairwise correlations
    corr = np.matmul(a_con, b_con.T)
    corr = corr / a_std[..., None]
    corr = corr / b_std[None, ...]
    corr = corr / a_con.shape[1]

    # contexts with no variability are nan, set to 0
    corr[np.isnan(corr)] = 0
    print(corr.shape)
    return corr

def pairwise_correlation_fourier(A, B):
    import numpy as np
    from scipy.ndimage import rotate, zoom
    from skimage.metrics import mean_squared_error

# grab and flatten context
    a_con = np.array( [a[1].flatten() for a in A] )
    b_con = np.array( [b[1].flatten() for b in B] )
    print(a_con.shape, b_con.shape)
    # get means and std for all contexts, center contexts
    a_mean, a_std = _stats(a_con)
    b_mean, b_std = _stats(b_con)
    a_con = a_con - a_mean[..., None]
    b_con = b_con - b_mean[..., None]

    # Compute the 3D Fourier Transform of all volumes in list 2
    fft2 = [np.fft.fftn(vol) for vol in b_con]

    # Initialize the array to hold the mean squared error for all vol1/vol2 comparisons
    mses = np.zeros((len(a_con), len(b_con)))

    # Loop over all volumes in list 1
    for i in range(len(a_con)):
        # Compute the 3D Fourier Transform of the current volume in list 1
        fft1 = np.fft.fftn(a_con[i])

        # Compute the cross-correlation with all volumes in list 2
        corr = [np.fft.ifftn(fft1 * fft2[j].conj()) for j in range(len(b_con))]
        print('corr', len(corr), flush=True)

        # Find the rotation angle and scale factor for each cross-correlation
        max_pos = [np.unravel_index(np.argmax(c), c.shape) for c in corr]

        # angles = [np.rad2deg(np.arctan2(max_pos[j][1], max_pos[j][0])) for j in range(len(b_con))]
        # scale_factors = [1.0 / max_pos[j][2] for j in range(len(b_con))]
        angles = []
        scale_factors = []
        for j in range(len(b_con)):
            if corr[j].shape == b_con[j].shape:  # check if shapes match
                angles.append(np.rad2deg(np.arctan2(max_pos[j][1], max_pos[j][0])))
                scale_factors.append(1.0 / max_pos[j][2])
            else:
                print('appending 0')
                angles.append(0)
                scale_factors.append(1.0)
        # Rotate and scale the current volume in list 1 for each cross-correlation
        a_con_rs = [zoom(rotate(a_con[i], angles[j], axes=(1, 0), reshape=False), scale_factors[j], order=1) for j in range(len(b_con))]

        # Compare each rotated and scaled volume with the corresponding volume in list 2
        mses[i] = [mean_squared_error(a_con_rs[j], b_con[j]) for j in range(len(b_con))]

    return mses

def match_points_fourier(A, B, scores, threshold):
    """
    """

    threshold = 1-threshold

    # split positions from context
    a_pos = np.array( [a[0] for a in A] )
    b_pos = np.array( [b[0] for b in B] )

    # get highest scores above threshold
    best_indcs = np.argmin(scores, axis=1)
    a_indcs = range(len(a_pos))
    keeps = scores[(a_indcs, best_indcs)] > threshold

    # return positions of corresponding points
    return a_pos[keeps, :3], b_pos[best_indcs[keeps], :3]




def match_points(A, B, scores, threshold):
    """
    """

    # split positions from context
    a_pos = np.array( [a[0] for a in A] )
    b_pos = np.array( [b[0] for b in B] )

    # get highest scores above threshold
    best_indcs = np.argmax(scores, axis=1)
    a_indcs = range(len(a_pos))
    keeps = scores[(a_indcs, best_indcs)] > threshold

    # return positions of corresponding points
    return a_pos[keeps, :3], b_pos[best_indcs[keeps], :3]

