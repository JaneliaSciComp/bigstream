import numpy as np
from bigstream import features
from bigstream import ransac
import dask.array as da
import os
from pathlib import Path
import traceback

from ClusterWrap.decorator import cluster

print("imported bigstream.affine")


def ransac_affine(
    fix, mov,                                       
    fix_spacing, mov_spacing,
    min_radius,
    max_radius,
    match_threshold,
    cc_radius=12,
    nspots=5000,
    align_threshold=2.0,
    num_sigma_max=15,
    verbose=True,
    fix_spots=None,
    mov_spots=None,
    matched_spots=None,
    r2_rotation=False,
    fix_subvolume=None,
    mov_subvolume=None,
    use_blocked_blob_lob=False,
    default=np.eye(4),
    initial_affine_no_translation=None,
    distributed=True,
    name='global',
    **kwargs,
):
    """
    """
    output_filepath = os.path.join(os.environ['jobdir'],'{}_affine'.format(name))
    Path(output_filepath).mkdir(parents=True, exist_ok=True)

    fix_is_centroids  = (fix_spots is not None) and type(fix_spots) == str and ('centroids.npy' in fix_spots)
    mov_is_centroids  = (mov_spots is not None) and type(mov_spots) == str and ('centroids.npy' in mov_spots)
    sort_fix_points = not fix_is_centroids 
    sort_mov_points = not mov_is_centroids

    print("r2 rotation", r2_rotation)
    print("fix_subvolume", fix_subvolume)
    print("centroids?", fix_is_centroids, mov_is_centroids)
    print("sort points?", sort_fix_points, sort_mov_points)

    if matched_spots is None:
        if verbose:
            print('Getting key points')

        # get spots
        if fix_spots is None:
            fix_spots = features.blob_detection(
                fix, min_radius, max_radius, distributed=distributed,
                # num_sigma=min(max_radius-min_radius, num_sigma_max),
                threshold=0, exclude_border=cc_radius, cluster_name='{}_fix_blob_detection'.format(name)
            )
            np.save(os.path.join(output_filepath,'fix_spots.npy'),fix_spots)
        else:
            fix_spots = np.load(fix_spots).astype(int)
        if fix_spots.shape[0] < 50:
            print('Fewer than 50 spots found in fixed image, returning default')
            return default
        if verbose:
            ns = fix_spots.shape[0]
            print(f'FIXED image: found {ns} key points')

        if mov_spots is None:
            # if use_blocked_blob_lob:
            #     mov_spots = features.blocked_blob_detection(
            #         mov, min_radius, max_radius, distributed=distributed,
            #         # num_sigma=min(max_radius-min_radius, num_sigma_max),
            #         threshold=0, exclude_border=cc_radius, cluster_name='mov_block_blob_detection'
            #     )
            # else:
            mov_spots = features.blob_detection(
                mov, min_radius, max_radius, distributed=distributed,
                # num_sigma=min(max_radius-min_radius, num_sigma_max),
                threshold=0, exclude_border=cc_radius, cluster_name='{}_mov_blob_detection'.format(name)
            )
            np.save(os.path.join(output_filepath,'mov_spots'),mov_spots)
        else:
            mov_spots = np.load(mov_spots).astype(int)

        if mov_spots.shape[0] < 50:
            print('Fewer than 50 spots found in moving image, returning default')
            return default
        if verbose:
            ns = mov_spots.shape[0]
            print(f'MOVING image: found {ns} key points')

        # sort
        if sort_fix_points:
            print("Sorting FIX points")
            sort_idx = np.argsort(fix_spots[:, 3])[::-1]
            fix_spots = fix_spots[sort_idx, :3][:nspots]
        if sort_mov_points:
            print("Sorting MOV points")
            sort_idx = np.argsort(mov_spots[:, 3])[::-1]
            mov_spots = mov_spots[sort_idx, :3][:nspots]

        # convert to physical units
        fix_spots = fix_spots * fix_spacing
        mov_spots = mov_spots * mov_spacing
        
        # # to match transpose(2,1,0) applied when openning raw data (only for centroid points)
        if fix_is_centroids:
            print("FIX points loaded from centroids")
            fix_spots[:, [0, 2]] = fix_spots[:, [2, 0]]
            if fix_subvolume is not None:
                fix_spots = filter_points_subvolume(fix_spots, fix_subvolume)
        if mov_is_centroids:
            print("MOV points loaded from centroids")
            mov_spots[:, [0, 2]] = mov_spots[:, [2, 0]]
            if mov_subvolume is not None:
                mov_spots = filter_points_subvolume(mov_spots, mov_subvolume)

        # get contexts
        fix_spots = features.get_spot_context_strict(
            fix, fix_spots, fix_spacing, cc_radius, output_filepath, name='fix'
        )
        if initial_affine_no_translation is not None:
            print("WARPING mov points")
            mov_spots = features.get_spot_context_strict_warped(
                mov, mov_spots, mov_spacing, cc_radius, initial_affine_no_translation, name='mov'
            )
        elif r2_rotation:
            print("rotating mov Points")
            mov_spots = features.get_spot_context_strict_rotated(
                mov, mov_spots, mov_spacing, cc_radius, name='mov'
            )
        else:            
            print("NOT rotating mov Points")
            mov_spots = features.get_spot_context_strict(
                mov, mov_spots, mov_spacing, cc_radius, output_filepath, name='mov'
            )

        if len(fix_spots) < 50:
            print('Fewer than 50 spots found in fixed image, Trying blob detection')
            fix_spots = features.blob_detection(
                fix, min_radius, max_radius, distributed=distributed,
                # num_sigma=min(max_radius-min_radius, num_sigma_max),
                threshold=0, exclude_border=cc_radius, cluster_name='{}_fix_blob_detection'.format(name)
            )

            np.save(os.path.join(output_filepath,'fix_spots_reblobbed.npy'),fix_spots)

            print("Sorting points")
            sort_idx = np.argsort(fix_spots[:, 3])[::-1]
            fix_spots = fix_spots[sort_idx, :3][:nspots]

            # convert to physical units
            fix_spots = fix_spots * fix_spacing
                    # get contexts
            fix_spots = features.get_spot_context_strict(
                fix, fix_spots, fix_spacing, cc_radius, output_filepath, name='fix'
            )


            np.save(os.path.join(output_filepath,'fix_spots_reblobbed.npy'),fix_spots)


        np.save(os.path.join(output_filepath,'fix_spots_processed'),np.array( [a[0]for a in fix_spots] ))
        np.save(os.path.join(output_filepath,'mov_spots_processed'),np.array( [a[0]for a in mov_spots] ))

        # get point correspondences
        correlations = features.pairwise_correlation(
            fix_spots, mov_spots,
        )
        fix_spots, mov_spots = features.match_points(
            fix_spots, mov_spots,
            correlations, match_threshold,
        )
        np.save(os.path.join(output_filepath,'matched_points'), np.concatenate((fix_spots, mov_spots), axis=1))
    else:
        print("LOADING MATCHED_POINTS")
        # Load the 'matched_points.npy' file
        matched_points = np.load(matched_spots)
        matched_points = filter_points(matched_points)
        # Split the loaded array into fix_spots and mov_spots
        fix_spots = matched_points[:, :3]
        mov_spots = matched_points[:, 3:]


    if verbose:
        ns = fix_spots.shape[0]
        print(f'MATCHED points: found {ns} matched points')
    # try:
    #     os.mkdir(os.path.join(os.environ['jobdir'],'{}_boxes'.format(name)))
    # except FileExistsError:
    #     pass
    # for i in np.random.randint(low=0, high=fix_spots.shape[0], size=10):
    #     #plot volumes of 10 random matched points for comparison
    #     try:
    #         plot_box_around_point(fix, fix_spots[i], os.path.join(os.environ['jobdir'],'{}_boxes'.format(name),'fix_{}'.format(i)))
    #         plot_box_around_point(mov, mov_spots[i], os.path.join(os.environ['jobdir'],'{}_boxes'.format(name),'mov_{}'.format(i)))

    #     except:
    #         pass


    # align
    return ransac.ransac_align_points(
        fix_spots, mov_spots, align_threshold, **kwargs,
    )

def rotate_coord_90_ccw(coord):
    z, y, x = coord
    return (z, x, -y)

def rotate_coord_90_cw(coord):
    z, y, x = coord
    return (z, -x, y)

def filter_points(arr):
    # Get the z-coordinates of all the points
    z_coords = arr[:, 0]
    # Get a boolean mask of which points have z < 389
    mask = z_coords < 389
    # Use the mask to select the points with z < 389
    filtered_points = arr[mask]
    # Convert the filtered points back to a list
    return np.array(filtered_points)

import numpy as np

def list_to_slice(lst):
    x_slice = slice(lst[0][0], lst[0][1])
    y_slice = slice(lst[1][0], lst[1][1])
    z_slice = slice(lst[2][0], lst[2][1])
    return (x_slice, y_slice, z_slice)

def filter_points_subvolume_slices(points, slices):
    slices = list_to_slice(slices)

    # Extract the start and stop values from the slices
    z_slice, y_slice, x_slice = slices
    x_start, x_stop = x_slice.start, x_slice.stop
    y_start, y_stop = y_slice.start, y_slice.stop
    z_start, z_stop = z_slice.start, z_slice.stop
    print(x_start, x_stop, y_start, y_stop, z_start, z_stop)

    # Calculate the minimum and maximum values of each coordinate in the filtered_points array
    x_min, y_min, z_min = np.min(points, axis=0)
    x_max, y_max, z_max = np.max(points, axis=0)

    # Print the results
    print(f"X: min={x_min:.3f}, max={x_max:.3f}")
    print(f"Y: min={y_min:.3f}, max={y_max:.3f}")
    print(f"Z: min={z_min:.3f}, max={z_max:.3f}")

    # Create boolean masks for each coordinate based on the min and max values
    x_mask = np.logical_and(points[:, 0] >= x_start, points[:, 0] < x_stop)
    y_mask = np.logical_and(points[:, 1] >= y_start, points[:, 1] < y_stop)
    z_mask = np.logical_and(points[:, 2] >= z_start, points[:, 2] < z_stop)

    # Combine the masks into a single boolean mask using logical AND
    mask = np.logical_and(np.logical_and(x_mask, y_mask), z_mask)

    # Filter the points array based on the mask
    filtered_points = points[mask]
    filtered_points[:, 0] -= x_start
    filtered_points[:, 1] -= y_start
    filtered_points[:, 2] -= z_start


    print('filtered points shape', filtered_points.shape)

    # Calculate the minimum and maximum values of each coordinate in the filtered_points array
    x_min, y_min, z_min = np.min(filtered_points, axis=0)
    x_max, y_max, z_max = np.max(filtered_points, axis=0)

    # Print the results
    print(f"X: min={x_min:.3f}, max={x_max:.3f}")
    print(f"Y: min={y_min:.3f}, max={y_max:.3f}")
    print(f"Z: min={z_min:.3f}, max={z_max:.3f}")
    # Return the filtered points array
    return filtered_points


def plot_box_around_point(zarr_file, point, output_file):
    import matplotlib.pyplot as plt
    # Extract the x, y, z coordinates of the point
    x, y, z = point

    # Extract a 20x20x1 subarray around the specified point
    subarray = zarr_file[z, y-10:y+10, x-10:x+10]

    # Plot the subarray using a 2D plot
    fig, ax = plt.subplots()
    ax.imshow(subarray, cmap='gray')

    # Add x, y, and z coordinates as text in the plot
    text = f'x={x}, y={y}, z={z}'
    ax.text(0.5, -0.1, text, transform=ax.transAxes, ha='center')

    # Save the plot to file
    plt.savefig(output_file)

def prepare_piecewise_ransac_affine(
    fix, mov,
    fix_spacing, mov_spacing,
    min_radius,
    max_radius,
    match_threshold,
    blocksize,
    distributed=False,
    fix_spots=None,
    mov_spots=None,
    cluster=None,
    cluster_kwargs={},
    **kwargs,
):
    """
    """
    from dask_jobqueue import SLURMCluster
    from dask.distributed import Client
    with SLURMCluster(**cluster_kwargs) as cluster, Client(cluster) as client:
        print("STARTING LOCAL AFFINES CLUSTER", flush=True)
        print(cluster.job_script(), '\n', flush=True)
        # get number of blocks required
        block_grid = np.ceil(np.array(fix.shape) / blocksize).astype(int)
        nblocks = np.prod(block_grid)
        # overlap = [int(round(x/8)) for x in blocksize]
        overlap = [1 for x in blocksize]
        print('blocksize', blocksize)
        print("Using overlap of: ", overlap)
        print('piecewise block_grid', block_grid)
        print('nblocks', nblocks)
        print('fix.shape', fix.shape)
        print('mov.shape', mov.shape)

        # wrap images as dask arrays
        fix_da = da.from_array(fix, chunks=blocksize)
        try:
            mov_da = da.from_array(mov, chunks=blocksize)
        except ValueError:
            mov_da = da.asarray(mov, chunks=blocksize)

        print('fix_da.shape', mov_da.shape)
        print('mov_da.shape', mov_da.shape)
        local_affines_meta = {
            'fix_shape': fix_da.shape,
            'mov_shape': mov_da.shape,
            'blocksize': blocksize,
            'nblocks': int(nblocks),
            'block_grid': block_grid.tolist(),
            'overlap': overlap,
        }

        import json
        with open(os.path.join(os.environ['jobdir'],'local_affines_meta.json'), "w") as outfile:
            json.dump(local_affines_meta, outfile)
        # wrap affine function

        def wrapped_ransac_affine(x, y, distributed=False, block_info=None, **kwargs):
            try:
                name='local/local_{}_{}_{}'.format(block_info[0]['chunk-location'][0], block_info[0]['chunk-location'][1], block_info[0]['chunk-location'][2])

                if fix_spots is not None:
                    block_fix_spots = filter_points_subvolume(fix_spots, block_info[0]['array-location'])
                else:
                    block_fix_spots = None
                if mov_spots is not None:
                    block_mov_spots = filter_points_subvolume(mov_spots, block_info[0]['array-location'])
                else:
                    block_mov_spots = None
                affine = ransac_affine(
                    x, y, fix_spacing, mov_spacing,
                    min_radius, max_radius, match_threshold,
                    distributed=distributed,
                    name=name,
                    fix_spots=block_fix_spots,
                    mov_spots=block_mov_spots,
                    **kwargs,
                )
                # adjust for block origin
                idx = np.array(block_info[0]['chunk-location'])
                print("IDX", idx)
                origin = (idx * blocksize - overlap) * fix_spacing
                tl, tr = np.eye(4), np.eye(4)
                tl[:3, -1], tr[:3, -1] = origin, -origin
                affine = np.matmul(tl, np.matmul(affine, tr))
                block_info[0]['affine'] = affine.tolist()
                with open(os.path.join(os.environ['jobdir'],name+"_affine",'block_info.json'), "w") as outfile:
                    json.dump(block_info[0], outfile)
            except Exception:
                print(traceback.format_exc())
                affine = np.eye(4)
            # return with block index axes
            return affine.reshape((1,1,1,4,4))
        
        import time
        # wait for at least one worker to be fully instantiated
        while ((client.status == "running") and
            (len(client.scheduler_info()["workers"]) < 1)):
            time.sleep(1.0)

        # affine align all chunks
        scattered = client.scatter(
            da.map_overlap(
            wrapped_ransac_affine, fix_da, mov_da,
            depth= overlap[0], #tuple(overlap),
            boundary='reflect',
            trim=False,
            align_arrays=True,
            dtype=np.float64,
            new_axis=[3, 4],
            chunks=[1, 1, 1, 4, 4],
            )
        )

        # scattered = da.map_overlap(
        #     wrapped_ransac_affine, fix_da, mov_da,
        #     depth= overlap[0], #tuple(overlap),
        #     boundary='reflect',
        #     trim=False,
        #     align_arrays=True,
        #     dtype=np.float64,
        #     new_axis=[3, 4],
        #     chunks=[1, 1, 1, 4, 4],
        #     )

        result = client.gather(client.submit(lambda x: x.compute(), scattered))
        return result

    futures = da.map_overlap(
        wrapped_ransac_affine, fix_da, mov_da,
        depth= overlap[0], #tuple(overlap),
        boundary='reflect',
        trim=False,
        align_arrays=True,
        dtype=np.float64,
        new_axis=[3, 4],
        chunks=[1, 1, 1, 4, 4],
        )


    result = cluster.client.gather(cluster.client.submit(lambda x: x.compute(), futures))
    return result

    # print(type(futures))
    # print(futures)
    # local_affines = cluster.client.gather(
    #     cluster.client.compute(futures),
    #         # errors='skip'
    #     )
    # print(type(local_affines))
    # print(local_affines)
    # print("Finished executing get local affines")
    # return local_affines

    return futures.compute()

    # result = cluster.client.map(lambda x: x.compute(), futures, error="skip")
    # gathered = client.gather(result)
    # return gathered


def filter_points_subvolume(points, subvolume):
    # Extract the start and stop values from the slices
    x_start = subvolume[0][0]
    x_stop = subvolume[0][1]
    y_start = subvolume[1][0]
    y_stop = subvolume[1][1]
    z_start = subvolume[2][0]
    z_stop = subvolume[2][1]

    x_min, y_min, z_min = np.min(points, axis=0)
    x_max, y_max, z_max = np.max(points, axis=0)

    # Print the results
    print(f"X: min={x_min:.3f}, max={x_max:.3f}")
    print(f"Y: min={y_min:.3f}, max={y_max:.3f}")
    print(f"Z: min={z_min:.3f}, max={z_max:.3f}")

    # Create boolean masks for each coordinate based on the min and max values
    x_mask = np.logical_and(points[:, 0] >= x_start, points[:, 0] < x_stop)
    y_mask = np.logical_and(points[:, 1] >= y_start, points[:, 1] < y_stop)
    z_mask = np.logical_and(points[:, 2] >= z_start, points[:, 2] < z_stop)

    # Combine the masks into a single boolean mask using logical AND
    mask = np.logical_and(np.logical_and(x_mask, y_mask), z_mask)

    # Filter the points array based on the mask
    filtered_points = points[mask]

    x_min, y_min, z_min = np.min(filtered_points, axis=0)
    x_max, y_max, z_max = np.max(filtered_points, axis=0)

    # Print the results
    print(f"X: min={x_min:.3f}, max={x_max:.3f}")
    print(f"Y: min={y_min:.3f}, max={y_max:.3f}")
    print(f"Z: min={z_min:.3f}, max={z_max:.3f}")

    print('shape', filtered_points.shape)
    # Return the filtered points array
    return filtered_points


def interpolate_affines(affines):
    """
    """

    # get block grid
    block_grid = affines.shape[:3]

    # construct an all identities matrix for comparison
    all_identities = np.empty_like(affines)
    for i in range(np.prod(block_grid)):
        idx = np.unravel_index(i, block_grid)
        all_identities[idx] = np.eye(4)

    # if affines are all identity, just return
    if np.all(affines == all_identities):
        return affines

    # process continues until there are no identity matrices left
    new_affines = np.copy(affines)
    identities = True
    while identities:
        identities = False

        # loop over all affine matrices
        for i in range(np.prod(block_grid)):
            idx = np.unravel_index(i, block_grid)

            # if an identity matrix is found
            if np.all(new_affines[idx] == np.eye(4)):
                identities = True
                trans, denom = np.array([0, 0, 0]), 0

                # average translations from 6 connected neighborhood
                for ax in range(3):
                    if idx[ax] > 0:
                        neighbor = tuple(
                            x-1 if j == ax else x for j, x in enumerate(idx)
                        )
                        neighbor_trans = new_affines[neighbor][:3, -1]
                        if not np.all(neighbor_trans == 0):
                            trans = trans + neighbor_trans
                            denom += 1
                    if idx[ax] < block_grid[ax]-1:
                        neighbor = tuple(
                            x+1 if j == ax else x for j, x in enumerate(idx)
                        )
                        neighbor_trans = new_affines[neighbor][:3, -1]
                        if not np.all(neighbor_trans == 0):
                            trans = trans + neighbor_trans
                            denom += 1

                # normalize then update matrix
                if denom > 0: trans /= denom
                new_affines[idx][:3, -1] = trans

    return new_affines

