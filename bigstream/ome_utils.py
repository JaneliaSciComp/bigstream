import numpy as np


def get_spatial_values(values, reverse_axes=False):
    if values is None:
        return None

    if len(values) > 3:
        if reverse_axes:
            return values[:3]
        else:
            return values[-3:]

    return values if not reverse_axes else values[::-1]


def get_ome_spatial_translation(coordinate_transformations):
    """Extract spatial translation (last 3 values) from OME coordinateTransformations list."""
    if coordinate_transformations is None:
        return None
    for t in coordinate_transformations:
        if t.get('type') == 'translation':
            return np.array(get_spatial_values(t['translation']))
    return None


def compose_origin_transform(origin_transform, global_coord_transforms):
    """
    Compose the transform for the origin from a user-provided 4x4 affine and
    global coordinateTransformations from the OME-ZARR metadata

    Composition order: origin_transform @ T_global
    """
    global_translation = get_ome_spatial_translation(global_coord_transforms)

    translations = []
    if global_translation is not None:
        translations.append(global_translation)

    if origin_transform is None and len(translations) == 0:
        return None

    result = (origin_transform
              if origin_transform is not None
              else np.eye(4))

    for t in translations:
        t_matrix = np.eye(4)
        t_matrix[:3, 3] = t
        result = result @ t_matrix

    return result
