import numpy as np
from ClusterWrap.decorator import cluster
import bigstream.utility as ut
from bigstream.align import affine_align, alignment_pipeline
from bigstream.transform import apply_transform, generate_random_affine_transforms_3d
from bigstream.metrics import local_correlation_coefficient
from scipy.ndimage import zoom
from scipy.linalg import logm, expm
from scipy.sparse.linalg import lsqr, norm
from scipy.sparse import csr_array, vstack
from scipy.special import factorial
import zarr
from zarr import blosc
from aicsimageio.readers import CziReader
from xml.etree import ElementTree
from itertools import product
import json
from distributed import Event
import time
import os
import aicspylibczi
#from ngff_zarr import to_ngff_image, to_multiscales, to_ngff_zarr
import nrrd
import glob


def create_synthetic_tiles(
    image,
    spacing,
    tile_size,
    overlap_factor,
    output_folder,
    max_translation,
    max_rotation,
    max_scale,
    max_shear,
    transforms_only=False,
    reconstruct=False,
):
    """
    
    """

    # get virtual tile origins (voxel units) and shape
    overlaps = np.round(tile_size * overlap_factor).astype(int)
    ntiles = np.ceil(image.shape / tile_size).astype(int)
    origins = [tile_size * (i, j, k) - overlaps for (i, j, k) in np.ndindex(*ntiles)]
    tile_size_with_overlaps = tile_size + 2 * overlaps

    # generate random affines
    centers = (np.array(origins) + tile_size_with_overlaps / 2) * spacing
    affines = generate_random_affine_transforms_3d(
        np.prod(ntiles)-1, max_translation, max_rotation, max_scale, max_shear, centers[1:],
    )
    affines = np.concatenate((np.eye(4)[None, ...], affines), axis=0)
    affines_inv = np.linalg.inv(affines)
    os.makedirs(output_folder, exist_ok=True)
    np.save(output_folder + '/affines.npy', affines)
    np.save(output_folder + '/affines_inv.npy', affines_inv)

    # if we only want the transforms
    if transforms_only:
        return affines, affines_inv

    # apply affines to virtual tiles
    fix = tuple(int(x) for x in tile_size_with_overlaps) + (image.dtype,)
    for iii, origin in enumerate(origins):
        tile = apply_transform(
            fix, image, spacing, spacing,
            transform_list=[affines[iii],],
            fix_origin=origin * spacing,
        )
        prefix = output_folder + f'/tile_{iii:05d}_'
        suffix = 'x'.join([str(x) for x in origin]) + '.nrrd'
        write_path = prefix + suffix
        nrrd.write(write_path, tile.transpose(2,1,0), compression_level=2)
    tile_paths = glob.glob(output_folder + '/tile_*.nrrd')

    if not reconstruct:
        return affines, affines_inv, tile_paths

    # create the reconstruction
    reconstructed_image = reconstruct_from_synthetic_tiles(
        tile_paths, affines_inv, image.shape, tile.dtype, spacing,
    )
    write_path = output_folder + '/reconstructed_image.nrrd'
    nrrd.write(write_path, reconstructed_image.transpose(2,1,0), compression_level=2)

    return affines, affines_inv, tile_paths



def reconstruct_from_synthetic_tiles(
    tile_paths,
    affines,
    shape,
    dtype,
    spacing,
):

    # get origins, voxel units
    F = lambda p: p.split('/')[-1].split('.')[0].split('_')[2].split('x')
    origins = [[int(x) for x in F(p)] for p in tile_paths]
    origins = np.array(origins)

    # apply inverse affines to reconstruct
    reconstructed_image = np.zeros(shape, dtype=dtype)
    for iii, tile_path in enumerate(tile_paths):

        # read and transform tile
        tile = nrrd.read(tile_path)[0].transpose(2,1,0)
        origin = origins[iii]
        fix = tile.shape + (dtype,)
        tile = apply_transform(
            fix, tile, spacing, spacing,
            transform_list=[affines[iii],],
            fix_origin=origin * spacing,
            mov_origin=origin * spacing,
        )

        # crop domain overflows and write to reconstructed image
        start = [abs(x) if x < 0 else 0 for x in origin]
        overflows = np.array(shape) - origin - tile.shape
        stop = [x if x < 0 else None for x in overflows]
        tile = tile[tuple(slice(x, y) for x, y in zip(start, stop))]
        image_crop = tuple(slice(x+y, x+y+z) for x, y, z in zip(origin, start, tile.shape))
        tile = np.maximum(reconstructed_image[image_crop], tile)
        reconstructed_image[image_crop] = tile

    return reconstructed_image


def create_neighbor_transforms_from_tile_transforms(
    affines,
    tile_grid_positions,
):

    # fix/mov assignments are a checkerboard pattern
    tile_grid = np.max(tile_grid_positions, axis=0) + 1
    fixed_flags = ~(np.arange(np.prod(tile_grid)).reshape(tile_grid) % 2).astype(bool)
    fixed_flags = [fixed_flags[x] for x in tile_grid_positions]

    # neighbors share faces
    ndim = len(tile_grid)
    neighbor_deltas = np.concatenate((np.eye(ndim), -np.eye(ndim)), axis=0).astype(int)

    # get all alignment index pairs
    alignments = []
    for iii, fixed_flag in enumerate(fixed_flags):
        if not fixed_flag: continue
        position = tile_grid_positions[iii]
        for neighbor_delta in neighbor_deltas:
            neighbor_position = position + neighbor_delta
            if np.all(neighbor_position >= 0) and np.all(neighbor_position < tile_grid):
                raster_index = int(np.ravel_multi_index(neighbor_position, tile_grid))
                alignments.append( ((iii, raster_index),) )

    # create neighbor transforms: T_ij = t_j @ t_i^-1
    neighbor_transforms = np.empty((len(alignments), 4, 4))
    for iii, alignment in enumerate(alignments):
        fix = affines[alignment[-1][0]]
        mov = affines[alignment[-1][1]]
        neighbor_transforms[iii] = np.matmul(mov, np.linalg.inv(fix))
    return neighbor_transforms, alignments


def distributed_stitch_new(
    tile_paths,
    tile_grid_positions,
    spacing,
    overlap_factor,
    steps,
    max_iterations=10,
    aligned_lcc_threshold=0.7,
    lcc_radius=8.,
    cluster=None,
    cluster_kwargs={},
):
    """
    """

    neighbor_transforms, neighbor_correlations, alignments = align_all_neighbors(
        tile_paths, tile_grid_positions, spacing, overlap_factor, steps,
        lcc_radius=lcc_radius,
        cluster=cluster,
        cluster_kwargs=cluster_kwargs,
    )

    tile_transforms = find_tile_transforms(
        neighbor_transforms, neighbor_correlations, alignments,
        tile_paths, max_iterations,
        aligned_lcc_threshold=aligned_lcc_threshold,
    )

    return tile_transforms


@cluster
def align_all_neighbors(
    tile_paths,
    tile_grid_positions,
    spacing,
    overlap_factor,
    steps,
    lcc_radius=8.,
    cluster=None,
    cluster_kwargs={},
):
    """
    """

    # TODO: ensure all steps are affine

    # get origins
    F = lambda p: p.split('/')[-1].split('.')[0].split('_')[2].split('x')
    origins = [[int(x) for x in F(p)] for p in tile_paths]
    origins = np.array(origins) * spacing

    # fix/mov assignments are a checkerboard pattern
    tile_grid = np.max(tile_grid_positions, axis=0) + 1
    fixed_flags = ~(np.arange(np.prod(tile_grid)).reshape(tile_grid) % 2).astype(bool)
    fixed_flags = [fixed_flags[x] for x in tile_grid_positions]

    # neighbors share faces
    ndim = len(tile_grid)
    neighbor_deltas = np.concatenate((np.eye(ndim), -np.eye(ndim)), axis=0).astype(int)

    # get all alignment specs
    # an alignment spec is: (paths, axis, fixed_first, origin, (fx, mx))
    #     paths are the paths to the image files, with the left tile path first
    #     axis is the axis along which the tiles are neighbors
    #     fixed_first is a bool indicated whether the left or right tile is fixed
    #     origin is the origin of the right tile (will be the origin of the registration)
    #     (fx, mx) are the indices of the fixed and moving tiles
    alignments = []
    for iii, fixed_flag in enumerate(fixed_flags):
        if not fixed_flag: continue
        position = tile_grid_positions[iii]
        for neighbor_delta in neighbor_deltas:
            neighbor_position = position + neighbor_delta
            if np.all(neighbor_position >= 0) and np.all(neighbor_position < tile_grid):
                raster_index = int(np.ravel_multi_index(neighbor_position, tile_grid))
                axis = np.nonzero(neighbor_delta)[0][0]
                if neighbor_delta[axis] < 0:
                    paths = [tile_paths[raster_index], tile_paths[iii]]
                    fixed_first = False
                    origin = origins[iii]
                else:
                    paths = [tile_paths[iii], tile_paths[raster_index]]
                    fixed_first = True
                    origin = origins[raster_index]
                alignment = (paths, axis, fixed_first, origin, (iii, raster_index))
                alignments.append(alignment)


    # define how to align a single pair of neighbors
    def align_neighbors(alignment):

        # unpack alignment spec
        paths = alignment[0]
        axis = alignment[1]
        fixed_first = alignment[2]
        origin = alignment[3]  # should be origin of whichever tile is on the right
        raster_indices = alignment[4]

        # print alignment spec
        print('ALIGNMENT SPEC')
        print(paths[0], '\n', paths[1], '\n', axis, fixed_first, origin, raster_indices, flush=True)

        # read tile_A
        tile_A = nrrd.read(paths[0])[0].transpose(2,1,0)
        crop = [slice(None),] * tile_A.ndim
        crop[axis] = slice(int(-2 * overlap_factor[axis] * tile_A.shape[axis]), None)
        tile_A = tile_A[tuple(crop)]

        # read tile_B
        tile_B = nrrd.read(paths[1])[0].transpose(2,1,0)
        crop = [slice(None),] * tile_B.ndim
        crop[axis] = slice(0, int(2 * overlap_factor[axis] * tile_B.shape[axis]))
        tile_B = tile_B[tuple(crop)]

        # turn on logging
        from bigstream.configure_bigstream import configure_logging
        configure_logging(None, True)

        # establish fix/mov and run the alignment
        fix, mov = (tile_A, tile_B) if fixed_first else (tile_B, tile_A)
        affine = alignment_pipeline(
            fix, mov, spacing, spacing, steps,
            fix_origin=origin, mov_origin=origin,
        )

        # score the result
        aligned = apply_transform(
            fix, mov, spacing, spacing,
            fix_origin=origin, mov_origin=origin,
            transform_list=[affine,],
        )
        _, corr_image = local_correlation_coefficient(
            fix, aligned, spacing, lcc_radius, return_image=True,
        )
        corr_mask = ~(np.isnan(corr_image) + np.isinf(corr_image))
        corr_mask = corr_mask * (aligned > 0)
        corr = max(0, np.nanmean(corr_image[corr_mask]))

        # XXX TEMP TEMP DEBUG
        import tifffile
        bundle = np.stack((fix, aligned,), axis=1)
        idx = raster_indices
        folder = '/'.join(paths[0].split('/')[:-1])
        tifffile.imwrite(
            f'{folder}/bundle_{idx[0]}_{idx[1]}.tiff', bundle, imagej=True, metadata={'axes':'ZCYX'},
        )
        # XXX END DEBUG

        return affine, corr

    # map align_neighbors to all neighbors
    affines_and_corrs = cluster.client.map(align_neighbors, alignments)
    affines_and_corrs = cluster.client.gather(affines_and_corrs)

   # unpack transforms and correlations
    neighbor_transforms, neighbor_correlations = [], []
    for a, b in affines_and_corrs:
        neighbor_transforms.append(a)
        neighbor_correlations.append(b)
    neighbor_transforms = np.array(neighbor_transforms)
    neighbor_correlations = np.array(neighbor_correlations)
    neighbor_correlations = neighbor_correlations / np.max(neighbor_correlations)

   # XXX TEMP TEMP DEBUG
    folder = '/'.join(tile_paths[0].split('/')[:-1])
    np.save(f'{folder}/neighbor_transforms.npy', neighbor_transforms)
    np.save(f'{folder}/neighbor_correlations.npy', neighbor_correlations)
    neighbor_transforms = np.load(f'{folder}/neighbor_transforms.npy')
    neighbor_correlations = np.load(f'{folder}/neighbor_correlations.npy')
    for iii in range(len(alignments)):
        print(alignments[iii][-1], neighbor_correlations[iii])
    # XXX END DEBUG

    return neighbor_transforms, neighbor_correlations, alignments



def find_tile_transforms(
    neighbor_transforms,
    neighbor_correlations,
    alignments,
    tile_paths,
    max_iterations=10,
    aligned_lcc_threshold=0.7,
    verbose=True,
    return_residuals=False,
):
    """
    """

    # remove bad alignments from optimization
    for iii in range(neighbor_transforms.shape[0]):
        if neighbor_correlations[iii] < aligned_lcc_threshold:
            neighbor_transforms[iii] = np.eye(4)
            neighbor_correlations[iii] = 0

    # INITIALIZE WITH ZEROTH ORDER BCH TRUNCATION
    # put observations in the Lie algebra
    neighbor_tangents = logm(neighbor_transforms)

    # define sparse constraints array
    rows, cols, data = [], [], []
    for iii, alignment in enumerate(alignments):
        if alignment[-1][0] != 0:
            rows.append(iii)
            cols.append(alignment[-1][0]-1)
            data.append(neighbor_correlations[iii])
        if alignment[-1][1] != 0:
            rows.append(iii)
            cols.append(alignment[-1][1]-1)
            data.append(neighbor_correlations[iii])
    rows, cols = np.array(rows), np.array(cols)
    N, M = neighbor_transforms.shape[0], len(tile_paths)
    bch_constraints = csr_array((data, (rows, cols)), shape=(N, M-1))

    # solve lsqr problem for each vector index to initialize tile tangents
    tile_tangents = np.zeros((M, 4, 4))
    for row in range(3):
        for col in range(4):
            tile_tangents[1:, row, col] = lsqr(
                bch_constraints, neighbor_tangents[:, row, col] * neighbor_correlations,
                atol=0, btol=1e-6, conlim=1e8,
            )[0]

    # put estimates in the Lie group
    tile_transforms = expm(tile_tangents)

    # GAUSS-NEWTON ITERATIONS
    # define operators
    def lie_bracket(A, B, order):
        if order > 1:
            B = lie_bracket(A, B, order-1)
        return np.matmul(A, B) - np.matmul(B, A)

    def dexp(A, B, order):
        result = B
        for iii in range(1, order+1):
            result += lie_bracket(A, B, iii) / factorial(iii+1, exact=True)
        return result

    def Ad(A, B):
        Ainv = np.linalg.inv(A)
        return np.matmul(A, np.matmul(B, Ainv))

    def Adinv(A, B):
        Ainv = np.linalg.inv(A)
        return np.matmul(Ainv, np.matmul(B, A))

    # run least square optimization iterations with Jacobian
    residuals = []
    neighbor_transforms_inv = np.linalg.inv(neighbor_transforms)
    for iteration in range(max_iterations):

        # compute residuals on the manifold and algebra
        residual_transforms = np.empty_like(neighbor_transforms)
        fixed_transforms_in_order = np.empty_like(neighbor_transforms)
        for iii in range(N):
            C = neighbor_transforms_inv[iii]
            B = tile_transforms[int(alignments[iii][-1][1])]
            A = tile_transforms[int(alignments[iii][-1][0])]
            residual_transforms[iii] = np.matmul(C, np.matmul(B, A))
            fixed_transforms_in_order[iii] = A
        residual_tangents = logm(residual_transforms)

        # XXX TEMP TEMP DEBUG
        xxx_r = np.sum( neighbor_correlations * np.linalg.norm(residual_transforms, axis=(1, 2)), axis=0)
        yyy_r = np.sum( neighbor_correlations * np.linalg.norm(residual_tangents, axis=(1, 2)), axis=0)
        residuals.append((xxx_r, yyy_r))
        if verbose:
            print(np.round([xxx_r, yyy_r], decimals=3))
        # XXX END DEBUG

        # apply jacobian terms and stack
        residual_tangents = dexp(residual_transforms, residual_tangents, 2)
        residual_tangents = -1 * Ad(fixed_transforms_in_order, residual_tangents)
        residual_tangents = residual_tangents * neighbor_correlations[:, None, None]

        # define least squares constraints right
        gn_constraints = csr_array((12*N, 12*(M-1)))
        for iii in range(N):
            row_start = iii * 12
            row_stop = row_start + 12
            if alignments[iii][-1][0] != 0:
                col_start = 12 * int(alignments[iii][-1][0] - 1)
                col_stop = col_start + 12
                eye_matrix = np.eye(12) * neighbor_correlations[iii]
                gn_constraints[row_start:row_stop, col_start:col_stop] = eye_matrix
            if alignments[iii][-1][1] != 0:
                col_start = 12 * int(alignments[iii][-1][1] - 1)  # conj == conjugation matrix
                col_stop = col_start + 12
                conj = tile_transforms[int(alignments[iii][-1][1])]  # no decrement, not a constraint matrix index
                conj_inv = np.linalg.inv(conj)
                kron_matrix = np.kron(conj_inv, conj.T)[:12, :12] * neighbor_correlations[iii]
                gn_constraints[row_start:row_stop, col_start:col_stop] = kron_matrix

        # normalize columns to condition the matrix
        col_norms = norm(gn_constraints, axis=0)
        col_norms[col_norms == 0] = 1
        gn_constraints = gn_constraints.multiply(1. / col_norms)

        # solve the model
        perturbation_tangents = np.zeros_like(tile_transforms)
        lsqr_solution = lsqr(
            gn_constraints, residual_tangents[:, :3, :].ravel(),
            atol=1e-6, btol=1e-6, conlim=1e8,
        )[0] / col_norms
        for iii in range(perturbation_tangents.shape[0]-1):
            perturbation_tangents[iii+1, :3, :] = lsqr_solution[12*iii:12*iii+12].reshape((3, 4))
        perturbation_transforms = expm(perturbation_tangents)

        # update the tile transforms
        tile_transforms = np.matmul(perturbation_transforms, tile_transforms)

    # invert all fixed transforms
    fixed_indices = set([x[-1][0] for x in alignments])
    for iii in fixed_indices:
        tile_transforms[iii] = np.linalg.inv(tile_transforms[iii])

    # remove unconstrained tiles from the result by marking them with the zero matrix
    neighbor_correlations_per_tile = [[],] * len(tile_transforms)
    for iii, alignment in enumerate(alignments):
        fix_idx = alignment[-1][0]
        mov_idx = alignment[-1][1]
        neighbor_correlations_per_tile[fix_idx].append(neighbor_correlations[iii])
        neighbor_correlations_per_tile[mov_idx].append(neighbor_correlations[iii])
    for iii, ncpt in enumerate(neighbor_correlations_per_tile):
        if np.sum(ncpt) == 0:
            tile_transforms[iii] = np.zeros_like(tile_transforms[0])

    if not return_residuals:
        return tile_transforms
    else:
        return tile_transforms, residuals














def _get_tile_info(czi_file_path):
    """"""

    # access czi file, get spacing, get channel axis, get spatial axes
    reader = CziReader(czi_file_path)
    spacing = reader.physical_pixel_sizes
    channel_axis = reader.dims.order.index('C')
    spatial_axes = tuple(reader.dims.order.index(x) for x in 'ZYX')

    # get tile/mosaic/vector axis (has different names), get tile positions
    if 'M' in reader.dims.order:
        tile_axis = reader.dims.order.index('M')
        tile_positions = np.array(reader.get_mosaic_tile_positions())
        # TODO: ensure all axes are present in tile_positions for this case
    elif 'V' in reader.dims.order:
        tile_axis = reader.dims.order.index('V')
        tile_positions = [x.attrib for x in reader.metadata.findall('.//TilesSetup//Position')]
        tile_positions = np.array([[float(x[y]) for y in 'ZYX'] for x in tile_positions])
        tile_positions = (tile_positions - np.min(tile_positions, axis=0)) / spacing / 1e-6  # spacing in microns
        # TODO: keep physical tile positions for sub voxel accuracy later
        tile_positions = np.round(tile_positions).astype(int)
    else:
        print("Error: no tile axis found\n")
        # TODO: graceful exit

    # get (i, j, k) tile grid positions
    tile_grid_indices = []
    steps = [np.sort(np.unique(tile_positions[:, x])) for x in range(tile_positions.shape[1])]
    for tile in tile_positions:
        tile_grid_indices.append( tuple(np.where(s == x)[0][0] for s, x in zip(steps, tile)) )

    # get tile shape
    tile_shape = np.array([reader.shape[x] for x in spatial_axes])

    # get overlap shapes
    smallest_diffs = np.min(np.ma.masked_equal(tile_positions, 0), axis=0) + 1
    smallest_diffs[smallest_diffs.mask] = 0
    overlaps = tile_shape - smallest_diffs + 1
    overlaps = np.array([o if o != s+1 else 0 for o, s in zip(overlaps, tile_shape)])

    return (reader, spacing, channel_axis,
            spatial_axes, tile_axis, tile_positions,
            tile_grid_indices, tile_shape, overlaps,)


@cluster
def distributed_stitch(
    czi_file_path,
    channel=0,
    minimum_overlap_correlation=0.3,
    global_optimization_iterations=100,
    global_optimization_learning_rate=0.1,
    affine_kwargs={},
    cluster=None,
    cluster_kwargs={},
):
    # TODO: think over the function API
    # TODO: complete docstring
    """
    Stitch the tiles in a czi file into one continuous volume.
    Overlapping regions are rigid aligned.

    Parameters
    ----------
    czi_file_path : string
        Path to the czi file

    channel : int (default: 0)
        Which channel to use for stitching

    cluster : ClusterWrap.cluster object (default: None)
        Only set if you have constructed your own static cluster. The default behavior
        is to construct a cluster for the duration of this function, then close it
        when the function is finished.

    cluster_kwargs : dict (default: {})
        Arguments passed to ClusterWrap.cluster
        If working with an LSF cluster, this will be
        ClusterWrap.janelia_lsf_cluster. If on a workstation
        this will be ClsuterWrap.local_cluster.
        This is how distribution parameters are specified.
    """

    # get all the relevant info about tiles
    tile_info = _get_tile_info(czi_file_path)
    reader = tile_info[0]
    spacing = tile_info[1]
    channel_axis = tile_info[2]
    spatial_axes = tile_info[3]
    tile_axis = tile_info[4]
    tile_positions = tile_info[5]
    tile_grid_indices = tile_info[6]
    tile_shape = tile_info[7]
    overlaps = tile_info[8]

    # construct list of neighbors/alignments to do
    neighbors_list = []
    fixed_image = {0:True}
    smallest_diffs = np.min(np.ma.masked_equal(tile_positions, 0), axis=0) + 1
    smallest_diffs[smallest_diffs.mask] = 0
    for iii, jjj in product(range(len(tile_positions)), repeat=2):
        diffs = tile_positions[jjj] - tile_positions[iii]
        diffs_indx = diffs.nonzero()[0]
        if len(diffs_indx) == 1 and 0 < diffs[diffs_indx[0]] <= smallest_diffs[diffs_indx[0]]:
            fixed_image[jjj] = False if fixed_image[iii] else True
            neighbors_list.append((iii, jjj, diffs_indx[0]))

    # define how to align a single pair of neighbors
    def align_neighbors(neighbors):

        # get number of cores
        ncores = ut.get_number_of_cores()

        # read the first region
        other_reader = aicspylibczi.CziFile(czi_file_path)
        A_read_spec = {
            reader.dims.order[channel_axis]:channel,
            reader.dims.order[tile_axis]:neighbors[0],
            'cores':2*ncores,
        }
        A_slice = [slice(None),] * len(spatial_axes)
        A_slice[neighbors[2]] = slice(-overlaps[neighbors[2]], None)
        # the copy prevents storage of entire tile in memory
        # without the copy, it stores a view into the entire tile
        A = np.copy(other_reader.read_image(**A_read_spec)[0].squeeze()[tuple(A_slice)])

        # read the second region
        B_read_spec = {
            reader.dims.order[channel_axis]:channel,
            reader.dims.order[tile_axis]:neighbors[1],
            'cores':2*ncores,
        }
        B_slice = [slice(None),] * len(spatial_axes)
        B_slice[neighbors[2]] = slice(0, overlaps[neighbors[2]])
        B = np.copy(other_reader.read_image(**B_read_spec)[0].squeeze()[tuple(B_slice)])

        # determine fix and moving, define origin relative to whole image
        fix, mov = (A, B) if fixed_image[neighbors[0]] else (B, A)
        origin = tile_positions[neighbors[1]] * spacing

        # check if overlap has sufficient common foreground to try and register
        corr = np.corrcoef(fix.flatten(), mov.flatten())[0, 1]
        if corr < minimum_overlap_correlation:
            print(f'Insufficient overlap correlation for tile pair {neighbors}.', flush=True)
            return None

        # define registration parameters
        default_affine_kwargs = {
            'rigid':True,
            'metric':'MS',
            'alignment_spacing':np.min(spacing) * 4,
            'shrink_factors':(2,),
            'smooth_sigmas':(np.min(spacing) * 8,),
            'optimizer_args':{
                'learningRate':0.05,
                'minStep':0.01,
                'numberOfIterations':100,
            },
        }
        kwargs = {**default_affine_kwargs, **affine_kwargs}

        # run the alignment
        return affine_align(
            fix, mov, spacing, spacing,
            fix_origin=origin, mov_origin=origin,
            **kwargs,
        )

    # map align_neighbors to all neighbors
    neighbor_transforms = cluster.client.map(align_neighbors, neighbors_list, resources={'concurrency':1})
    neighbor_transforms = cluster.client.gather(neighbor_transforms)

    # filter out bad overlaps
    new_neighbors_list, new_neighbor_transforms = [], []
    for a, b in zip(neighbors_list, neighbor_transforms):
        if b is not None:
            new_neighbors_list.append(a)
            new_neighbor_transforms.append(b)
    neighbors_list = new_neighbors_list
    neighbor_transforms = np.array(new_neighbor_transforms)

    # build transform composition matrix
    A = np.zeros((len(tile_positions), len(tile_positions), len(neighbors_list)))
    for iii, neighbors in enumerate(neighbors_list):
        a, b = neighbors[0], neighbors[1]
        fi, mi = (a, b) if fixed_image[a] else (b, a)
        A[mi, fi, iii] = 1

    # initialize tile transforms as identity
    tile_transforms = np.empty((len(tile_positions), 4, 4))
    for iii in range(len(tile_positions)):
        tile_transforms[iii] = np.eye(4)

    # gradient descent loop
    print('Starting global consistency optimization')
    for iii in range(global_optimization_iterations):

        # with respect to moving parameters
        factor = np.einsum('mij,nmo', tile_transforms, A)
        reconstruction = np.einsum('nij,jkno', tile_transforms, factor)
        left = np.einsum('ijno,jko', factor, reconstruction)
        right = np.einsum('ijno,ojk', factor, neighbor_transforms)
        gradient = left - right

        # with respect to fixed parameters
        factor = np.einsum('nij,nmo', tile_transforms, A)
        reconstruction = np.einsum('nij,jkno', tile_transforms, factor)
        left = np.einsum('ijno,jko', factor, reconstruction)
        right = np.einsum('ijno,ojk', factor, neighbor_transforms)
        gradient = (gradient + left - right).transpose(2, 0, 1)

        # print feedback
        objective = np.sum( (neighbor_transforms - reconstruction.transpose(2, 0, 1))**2 )
        print(f'ITERATION: {iii}  OBJECTIVE VALUE: {objective}')

        # take a step
        tile_transforms = tile_transforms - global_optimization_learning_rate * gradient

    # invert all fixed transforms
    for iii in range(len(tile_transforms)):
        if fixed_image[iii]:
            tile_transforms[iii] = np.linalg.inv(tile_transforms[iii])

    # TODO: consider fixing one tile
    #       i.e. find inverse of one transform and compose that with
    #       all other transforms

    # all done!
    return tile_transforms


def save_transforms(path, transforms):
    """
    """
    # TODO: discuss with JB best format to be consistent with other tools
    n = transforms.shape[0]
    d = {i:transforms[i].tolist() for i in range(n)}
    with open(path, 'w') as f:
        json.dump(d, f, indent=4)


def read_transforms(path):
    """
    """

    with open(path, 'r') as f:
        d = json.load(f)
    return np.array([d[str(i)] for i in range(len(d))])


@cluster
def distributed_apply_stitch(
    czi_file_path,
    transforms,
    write_path,
    resample_padding=0.2,
    write_group_interval=60,
    channel=0,
    cluster=None,
    cluster_kwargs={},
):
    """
    """

    # get all the relevant info about tiles
    tile_info = _get_tile_info(czi_file_path)
    reader = tile_info[0]
    spacing = tile_info[1]
    channel_axis = tile_info[2]
    spatial_axes = tile_info[3]
    tile_axis = tile_info[4]
    tile_positions = tile_info[5]
    tile_grid_indices = tile_info[6]
    tile_shape = tile_info[7]
    overlaps = tile_info[8]

    # generate zarr file for writing
    zarr_blocks = tuple(np.round(tile_shape / 2).astype(int))
    full_shape = np.max(tile_positions, axis=0) + reader.dims[['Z', 'Y', 'X']]
    output_zarr = ut.create_zarr(
        write_path, full_shape, zarr_blocks, reader.dtype, multithreaded=True,
    )

    def resample_tile(tile_number, transform):

        print(f'starting {tile_number}', flush=True)

        # get number of cores
        ncores = ut.get_number_of_cores()

        # read tile data
        other_reader = aicspylibczi.CziFile(czi_file_path)
        read_spec = {
            reader.dims.order[channel_axis]:channel,
            reader.dims.order[tile_axis]:tile_number,
            'cores':2*ncores,
        }
        tile = other_reader.read_image(**read_spec)[0].squeeze()
        mov_origin = tile_positions[tile_number]

        print(f'weighting tile {tile_number}', flush=True)

        # apply linear blending weights to overlap region, per axis
        for axis in range(3):
            # only if we cut tiles along this axis
            if overlaps[axis] != 0:
                # construct weights array for this axis
                shape = list(tile.shape)
                shape[axis] = 1
                pads = [(0, 0),]*3
                pads[axis] = (overlaps[axis], 0)
                weights = np.pad(np.ones(shape, dtype=np.float32), pads, mode='linear_ramp')
                # left side, only if it's not on the left edge
                if tile_grid_indices[tile_number][axis] > 0:
                    region = [slice(None),]*3
                    region[axis] = slice(0, overlaps[axis]+1)
                    region = tuple(region)
                    tile[region] = np.round( tile[region] * weights ).astype(tile.dtype)
                # right side, only if it's not on the right edge
                if tile_grid_indices[tile_number][axis] < np.max(tile_grid_indices, axis=0)[axis]:
                    region = [slice(None),]*3
                    region[axis] = slice(-overlaps[axis]-1, None)
                    region = tuple(region)
                    reflect = [slice(None),]*3
                    reflect[axis] = slice(None, None, -1)
                    reflect = tuple(reflect)
                    tile[region] = np.round( tile[region] * weights[reflect] ).astype(tile.dtype)

        # generate reference
        fix_origin = mov_origin - np.round(np.array(tile.shape) * resample_padding).astype(int)
        fix_end = fix_origin + np.round(np.array(tile.shape) * (1 + 2*resample_padding)).astype(int)
        fix_origin = np.maximum(fix_origin, 0)
        fix_end = np.minimum(fix_end, output_zarr.shape)
        fix_shape = tuple(int(b - a) for a, b in zip(fix_origin, fix_end))

        print(f'aligning {tile_number}', flush=True)

        # apply transform
        aligned = apply_transform(
            fix_shape, tile, spacing, spacing,
            transform_list=[transform,],
            fix_origin=fix_origin * spacing,
            mov_origin=mov_origin * spacing,
        )

        # register as ready to write
        write_region = tuple(slice(a, b) for a, b in zip(fix_origin, fix_end))
        blosc.set_nthreads(2*ncores)

        # get neighbors info
        neighbor_events = []
        for delta in product((-1, 0, 1), repeat=3):
            if delta == (0, 0, 0): continue
            neighbor_index = tuple(a + b for a, b in zip(tile_grid_indices[tile_number], delta))
            neighbor_events.append(Event(f'{neighbor_index}'))

        # wait until its clear to write
        print(f'waiting {tile_number}', flush=True)
        while True:
            if np.all( [not e.is_set() for e in neighbor_events] ):

                # some robustness to race conditions
                seed = int(time.time()) + int(''.join([str(x) for x in tile_grid_indices[tile_number]]))
                np.random.seed(seed)
                time.sleep(int(np.random.rand() * 8 + 2))
                if np.any( [e.is_set() for e in neighbor_events] ): continue

                done_event = Event(f'{tile_grid_indices[tile_number]}')
                done_event.set()
                break
            else: time.sleep(1)

        # write result to disk
        print(f'writing {tile_number}, {time.ctime(time.time())}', flush=True)
        output_zarr[write_region] = output_zarr[write_region] + aligned
        print(f'done writing {tile_number}, {time.ctime(time.time())}', flush=True)

        # unset write flag, return
        done_event.clear()
        print(f'done with {tile_number}', flush=True)
        return True

    futures = cluster.client.map(resample_tile, range(len(transforms)), transforms, resources={'concurrency':1})
    all_events = cluster.client.gather(futures)
    return output_zarr


#@cluster
#def generate_ome_ngff_zarr(
#    input_zarr_array,
#    spacing,
#    write_path,
#    scale_factors,
#    chunks,
#    cluster=None,
#    cluster_kwargs={},
#    **kwargs,
#):
#    """
#    """
#    
#    print('calling to_ngff_image', flush=True)
#    ngff_image = to_ngff_image(
#        input_zarr_array,
#        dims=('z', 'y', 'x'),
#        scale={a:b for a, b in zip('zyx', spacing)},
#        axes_units={a:'micrometer' for a in 'zyx'}
#    )
#    print('calling to_multiscales', flush=True)
#    multiscales = to_multiscales(
#        ngff_image,
#        scale_factors,
#        chunks=chunks,
#    )
#    print('calling to_ngff_zarr', flush=True)
#    to_ngff_zarr(
#        write_path,
#        multiscales,
#        **kwargs,
#    )
#    return zarr.open(write_path, 'r+')

