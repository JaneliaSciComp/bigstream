import zarr
import numpy as np
import shutil
from bigstream import affine
from bigstream import deform
from bigstream import transform
from bigstream import configuration


def multifish_registration_pipeline(
    fixed_file_path,
    fixed_lowres_dataset,
    fixed_highres_dataset,
    moving_file_path,
    moving_lowres_dataset,
    moving_highres_dataset,
    transform_write_path,
    inv_transform_write_path,
    scratch_directory,
    global_affine_params={},
    local_affine_params={},
    deform_params={},
    ):
    """
    """

    # merge params with defaults
    global_affine_params = {
        **(configuration.multifish_registration_global_affine_defaults),
        **global_affine_params,
    }
    local_affine_params = {
        **(configuration.multifish_registration_local_affine_defaults),
        **local_affine_params,
    }
    deform_params = {
        **(configuration.multifish_registration_deform_defaults),
        **deform_params,
    }

    # wrap inputs as zarr objects
    fix_zarr = zarr.open(store=zarr.N5Store(fixed_file_path), mode='r')
    mov_zarr = zarr.open(store=zarr.N5Store(moving_file_path), mode='r')

    # get lowres data and spacing
    fix_lowres = fix_zarr[fixed_lowres_dataset]
    mov_lowres = mov_zarr[moving_lowres_dataset]

    fix_meta = fix_lowres.attrs.asdict()
    mov_meta = mov_lowres.attrs.asdict()
    fix_lowres_spacing = np.array(fix_meta['pixelResolution'])
    fix_lowres_spacing *= fix_meta['downsamplingFactors']
    mov_lowres_spacing = np.array(mov_meta['pixelResolution'])
    mov_lowres_spacing *= mov_meta['downsamplingFactors']

    # lowres data is assumed to fit into RAM
    fix_lowres = fix_lowres[...].transpose(2, 1, 0)  # zarr loads zyx order
    mov_lowres = mov_lowres[...].transpose(2, 1, 0)

    # global affine alignment
    global_affine = affine.dog_ransac_affine(
        fix_lowres, mov_lowres, fix_lowres_spacing, mov_lowres_spacing,
        nspots=global_affine_params['nspots'],
        cc_radius=global_affine_params['cc_radius'],
        match_threshold=global_affine_params['match_threshold'],
        align_threshold=global_affine_params['align_threshold'],
    )

    # apply global affine
    mov_lowres_aligned = transform.apply_global_affine(
        fix_lowres, mov_lowres, fix_lowres_spacing, mov_lowres_spacing,
        global_affine,
    )

    # local affine alignment
    local_affines = affine.dog_ransac_affine_distributed(
        fix_lowres, mov_lowres_aligned, fix_lowres_spacing, fix_lowres_spacing,
        nspots=local_affine_params['nspots'],
        cc_radius=local_affine_params['cc_radius'],
        match_threshold=local_affine_params['match_threshold'],
        align_threshold=local_affine_params['align_threshold'],
        blocksize=local_affine_params['blocksize'],
        cluster_extra=local_affine_params['cluster_extra'],
    )


    # get highres data and spacing
    fix_highres = fix_zarr[fixed_highres_dataset]
    mov_highres = mov_zarr[moving_highres_dataset]

    fix_meta = fix_highres.attrs.asdict()
    mov_meta = mov_highres.attrs.asdict()
    fix_highres_spacing = np.array(fix_meta['pixelResolution'])
    fix_highres_spacing *= fix_meta['downsamplingFactors']
    mov_highres_spacing = np.array(mov_meta['pixelResolution'])
    mov_highres_spacing *= mov_meta['downsamplingFactors']

    # compose global and local affines to single position field
    # highres objects assumed too big for RAM, backed by disk
    dbs = configuration.multifish_registration_deform_defaults['deform_blocksize']
    global_affine = transform.global_affine_to_position_field(
        fix_highres.shape, fix_highres_spacing, global_affine,
        scratch_directory+'/global_affine_position_field.zarr',
        blocksize=dbs,
    )

    local_affines = transform.local_affine_to_position_field(
        fix_highres.shape, fix_highres_spacing, local_affines,
        scratch_directory+'/local_affines_position_field.zarr',
        blocksize=dbs, block_multiplier=[2, 2, 1],
    )

    return "WORKED!"

#    # compose
#    total_affine = transform.compose_position_fields(
#        [local_affines, global_affine,], fix_highres_spacing,
#        scratch_directory+'/total_affine_position_field.zarr',
#        blocksize=dbs,
#    )
#
#    # clean up
#    shutil.rmtree(scratch_directory+'/global_affine_position_field.zarr')
#    shutil.rmtree(scratch_directory+'/local_affines_position_field.zarr')
#
#
#    # apply total affine to highres data
#    mov_highres_aligned = transform.apply_position_field(
#        fix_highres, fix_highres_spacing, mov_highres, mov_highres_spacing,
#        total_affine,
#        scratch_directory+'/mov_highres_aligned.zarr',
#        transpose=[True, True, False],
#    )
#
#    # deform
#    deforms = deform.deformable_align_distributed(
#        fix_highres, mov_highres_aligned, fix_highres_spacing, fix_highres_spacing,
#        scratch_directory+'/deformation_position_field.zarr',
#        cc_radius=deform_params['cc_radius'],
#        gradient_smoothing=deform_params['gradient_smoothing'],
#        field_smoothing=deform_params['field_smoothing'],
#        iterations=deform_params['iterations'],
#        shrink_factors=deform_params['shrink_factors'],
#        smooth_sigmas=deform_params['smooth_sigmas'],
#        step=deform_params['step'],
#        cluster_extra=deform_params['cluster_extra'],
#        transpose=True,
#    )
#
#    # compose total affine and deformation
#    final_transform = transform.compose_position_fields(
#        [deforms, total_affine,], fix_highres_spacing,
#        transform_write_path, blocksize=dbs,
#    )
#
#    # clean up
#    shutil.rmtree(scratch_directory+'/mov_highres_aligned.zarr')
#    shutil.rmtree(scratch_directory+'/deformation_position_field.zarr')


