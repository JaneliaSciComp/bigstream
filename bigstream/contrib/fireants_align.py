import logging

import numpy as np
import SimpleITK as sitk

import bigstream.transform as bst
import bigstream.utility as ut
from bigstream.align import (
    realize_mask,
    apply_alignment_spacing,
    images_to_sitk,
    format_static_transform_data,
    phase_correlation_translation,
)


logger = logging.getLogger(__name__)


# per-process state: torch_threads is applied once, and each resolved device
# is logged once, so a worker running many blocks does not repeat either
_torch_threads_set = False
_logged_devices = set()


def _resolve_device(device):
    """
    Resolve 'auto' to 'cuda', then 'mps', then 'cpu', whichever is available
    first. Any other value is returned unchanged.
    """
    import torch

    if device != 'auto':
        return device
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def _empty_device_cache(device):
    import torch

    device_type = str(device).split(':')[0]
    if device_type == 'cuda':
        torch.cuda.empty_cache()
    elif device_type == 'mps':
        torch.mps.empty_cache()


def _clip_to_percentiles(image, clip_percentiles):
    """Clip a sitk image to its own (low, high) intensity percentiles"""
    if not clip_percentiles:
        return image
    arr = sitk.GetArrayFromImage(image)
    lo, hi = np.percentile(arr, clip_percentiles)
    clipped = sitk.GetImageFromArray(np.clip(arr, lo, hi))
    clipped.CopyInformation(image)
    return clipped


_INTENSITY_RANGE = 1000.0


def _rescale_intensities(fix, mov, joint):
    """
    Map fix and mov linearly onto [0, _INTENSITY_RANGE]: each image on its
    own, or both with one shared map when joint.

    FireANTs' local CC is (cross^2 + 1e-5) / (var_fix * var_mov + 1e-5) per
    window, with each variance floored at 1e-5. With low intensities (e.g.
    data in [0, 1]) the variance product is of the order of those constants,
    and a flat moving patch scores a perfect CC of 1: the optimization then
    shrinks or collapses the transform. CC and MI are unchanged by a linear
    map of either image otherwise, so each image gets its own and neither is
    squashed by the other's range; MSE compares intensities directly, so it
    needs the shared map. Both images also end up non-negative, which
    FireANTs' MI expects.
    """
    fix_arr = sitk.GetArrayFromImage(fix)
    mov_arr = sitk.GetArrayFromImage(mov)
    if joint:
        ranges = [(min(fix_arr.min(), mov_arr.min()), max(fix_arr.max(), mov_arr.max()))] * 2
    else:
        ranges = [(fix_arr.min(), fix_arr.max()), (mov_arr.min(), mov_arr.max())]

    def _rescaled(arr, image, lo, hi):
        if not hi > lo:
            return image
        out = sitk.GetImageFromArray(((arr - lo) * (_INTENSITY_RANGE / (hi - lo))).astype(np.float32))
        out.CopyInformation(image)
        return out

    return (_rescaled(fix_arr, fix, *ranges[0]),
            _rescaled(mov_arr, mov, *ranges[1]))


def _pad_fixed_to_min_size(fix, fix_mask, min_size, context):
    """
    Zero-pad the upper end of every fix axis shorter than min_size voxels up
    to min_size; the origin stays where it is. When anything is padded, the
    fixed mask is padded with zeros too (an all-ones mask over the original
    voxels when there is none), so the padding takes no part in the metric.
    Returns fix and fix_mask unchanged otherwise.
    """
    size = fix.GetSize()
    upper = [max(min_size - s, 0) for s in size]
    if not any(upper):
        return fix, fix_mask
    logger.info(f'{context} zero-padding the fixed working grid {size[::-1]} to at least {min_size} voxels per axis')
    if fix_mask is None:
        fix_mask = sitk.GetImageFromArray(np.ones(fix.GetSize()[::-1], dtype=np.float32))
        fix_mask.CopyInformation(fix)
    lower = [0] * len(size)
    return (sitk.ConstantPad(fix, lower, upper, 0.0),
            sitk.ConstantPad(fix_mask, lower, upper, 0.0))


def _is_flat(image):
    """True if a sitk image is constant (or has no finite values)"""
    arr = sitk.GetArrayViewFromImage(image)
    return not np.nanmax(arr) > np.nanmin(arr) if np.isfinite(arr).any() else True


def _ncc(a, b, mask=None):
    """
    Global normalized cross correlation of two arrays, higher is better;
    None when undefined (either side constant, or not finite)
    """
    if mask is not None:
        a, b = a[mask], b[mask]
    a = a.astype(np.float64) - a.mean()
    b = b.astype(np.float64) - b.mean()
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if not (np.isfinite(denom) and denom > 0):
        return None
    return float(np.sum(a * b) / denom)


class _LevelLossLogger:
    """
    Wraps a FireANTs loss function and logs its last value of every pyramid
    level at DEBUG. FireANTs calls set_current_scale_and_iterations at the
    start of each level when the loss has it, which marks the level boundary.
    """

    def __init__(self, loss_fn, context):
        self.loss_fn = loss_fn
        self.context = context
        self.scale = None
        self.iterations = 0
        self.last_loss = None

    def set_current_scale_and_iterations(self, scale, iters):
        self.flush()
        self.scale = scale
        self.iterations = 0
        if hasattr(self.loss_fn, 'set_current_scale_and_iterations'):
            self.loss_fn.set_current_scale_and_iterations(scale, iters)

    def __call__(self, *args, **kwargs):
        loss = self.loss_fn(*args, **kwargs)
        self.iterations += 1
        self.last_loss = loss.detach()
        return loss

    def flush(self):
        if self.scale is not None and self.last_loss is not None:
            logger.debug((
                f'{self.context} FireANTs scale {self.scale}: '
                f'{self.iterations} iterations, loss={float(self.last_loss):.6f}'
            ))
        self.last_loss = None


def _check_schedule(scales, iterations):
    if len(scales) != len(iterations):
        raise ValueError(
            f'scales and iterations must have equal length; '
            f'got {len(scales)} and {len(iterations)}'
        )
    if any(s1 <= s2 for s1, s2 in zip(scales[:-1], scales[1:])):
        raise ValueError(f'scales must be decreasing; got {list(scales)}')


def _prepare_images(
    fix, mov, fix_spacing, mov_spacing,
    alignment_spacing, fix_mask, mov_mask, fix_roi,
    fix_origin, mov_origin, static_transform_list,
    clip_percentiles, loss, context,
):
    """
    Shared preamble of the FireANTs steps, the same as demons_align's:
    realize masks, skip-sample to alignment_spacing, convert to sitk, pre-warp
    mov (and its mask) onto the fix working grid through the static
    transforms, put each mask on its image's grid, clip and rescale
    intensities for loss.
    Returns sitk images fix, mov, fix_mask, mov_mask (masks may be None).
    """
    # format static transform data
    a, b = format_static_transform_data(
        static_transform_list, fix, fix_spacing, fix_origin,
    )
    static_transform_spacing = a
    static_transform_origin = b

    # realize masks
    fix_mask = realize_mask(fix, fix_mask, roi=fix_roi)
    mov_mask = realize_mask(mov, mov_mask)

    # skip-sample and convert to SITK images (images_to_sitk casts to float32)
    X = apply_alignment_spacing(
        fix, mov,
        fix_mask, mov_mask,
        fix_spacing, mov_spacing,
        alignment_spacing,
        context=context,
    )
    fix, mov, fix_mask, mov_mask = images_to_sitk(*X, fix_origin, mov_origin)

    # pre-warp mov by static transforms onto the fix working grid, as
    # demons_align does; the step's result is the FireANTs residual only
    if static_transform_list:
        T = bst.transform_list_to_composite_transform(
            static_transform_list,
            static_transform_spacing,
            static_transform_origin,
        )
        mov = sitk.Resample(mov, fix, T, sitk.sitkLinear, 0.0)
        if mov_mask is not None:
            mov_mask = sitk.Resample(mov_mask, fix, T, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)

    # masks may have a different sampling from the images; FireANTs needs the
    # mask as an extra channel of the image, so on the image grid
    identity_tx = sitk.Transform(fix.GetDimension(), sitk.sitkIdentity)

    def _mask_on_image_grid(mask, image):
        if mask is None:
            return None
        return sitk.Resample(
            mask, image, identity_tx, sitk.sitkNearestNeighbor, 0, sitk.sitkFloat32,
        )

    fix_mask = _mask_on_image_grid(fix_mask, fix)
    mov_mask = _mask_on_image_grid(mov_mask, mov)

    fix = _clip_to_percentiles(fix, clip_percentiles)
    mov = _clip_to_percentiles(mov, clip_percentiles)
    fix, mov = _rescale_intensities(fix, mov, joint=(loss == 'mse'))
    return fix, mov, fix_mask, mov_mask


def _defer_fireants_log_levels():
    """
    Several FireANTs modules call logger.setLevel(logging.INFO) at import,
    which overrides any level configured for the 'fireants' logger (e.g. in
    logging_config.ini). Reset them so they inherit the configured level.
    """
    for name, module_logger in logging.root.manager.loggerDict.items():
        if name.startswith('fireants.') and isinstance(module_logger, logging.Logger):
            module_logger.setLevel(logging.NOTSET)


def _start_torch(torch_threads, device, context):
    """Apply torch_threads once per process and resolve the device"""
    import torch

    _defer_fireants_log_levels()

    global _torch_threads_set
    if torch_threads is not None and not _torch_threads_set:
        torch.set_num_threads(int(torch_threads))
        _torch_threads_set = True

    dev = _resolve_device(device)
    if dev not in _logged_devices:
        logger.info(f'{context} FireANTs runs on device {dev}')
        _logged_devices.add(dev)
    return dev


def _batched_images(fix, mov, fix_mask, mov_mask, loss, dev):
    """
    FireANTs batches of fix and mov on dev, with the masks appended as a last
    channel (all ones for the side without one) when either is given.
    Returns fixed, moving and the FireANTs loss_type.
    """
    from fireants.io.image import Image, BatchedImages
    from fireants.io.imagemask import apply_mask_to_image, generate_image_mask_allones

    # build the images on the CPU and move the batches to the device:
    # fireants.io.Image moves float64 matrices to its device before
    # casting them to float32, which MPS does not support
    fix_image = Image(fix, device='cpu')
    mov_image = Image(mov, device='cpu')
    loss_type = loss
    if fix_mask is not None or mov_mask is not None:
        fix_mask_image = (Image(fix_mask, device='cpu') if fix_mask is not None
                          else generate_image_mask_allones(fix_image))
        mov_mask_image = (Image(mov_mask, device='cpu') if mov_mask is not None
                          else generate_image_mask_allones(mov_image))
        fix_image = apply_mask_to_image(fix_image, fix_mask_image)
        mov_image = apply_mask_to_image(mov_image, mov_mask_image)
        loss_type = 'masked_' + loss
    fixed = BatchedImages([fix_image])
    moving = BatchedImages([mov_image])
    fixed.set_device(dev)
    moving.set_device(dev)
    return fixed, moving, loss_type


def _translated_image(image, offset):
    """
    Copy of the sitk image moved by -offset in physical space, so that its
    content at p is the content of `image` at p + offset: no resampling
    """
    shifted = sitk.Image(image)
    shifted.SetOrigin((np.asarray(image.GetOrigin()) - offset).tolist())
    return shifted


def _metric_improved(fix, mov, fix_mask, moved, context):
    """
    final_metric_check: global NCC (higher is better) between fix and mov
    before registering, and between fix and the moved image after, inside
    the fixed mask if any. mov is put on the fix grid for the 'before' value,
    since without static transforms it keeps its own grid.
    """
    identity_tx = sitk.Transform(fix.GetDimension(), sitk.sitkIdentity)
    fix_np = sitk.GetArrayViewFromImage(fix)
    mov_on_fix = sitk.Resample(mov, fix, identity_tx, sitk.sitkLinear, 0.0)
    metric_mask = (sitk.GetArrayViewFromImage(fix_mask) > 0
                   if fix_mask is not None else None)
    initial_metric_value = _ncc(fix_np, sitk.GetArrayViewFromImage(mov_on_fix), metric_mask)
    final_metric_value = _ncc(fix_np, moved, metric_mask)
    if initial_metric_value is None or final_metric_value is None:
        # e.g. a constant image inside the mask, or a NaN result: nothing
        # shows the result is an improvement
        logger.warning((
            f'{context} FireANTs align metric undefined: '
            f'initial NCC: {initial_metric_value}, '
            f'final NCC: {final_metric_value} '
        ))
        return False
    if final_metric_value < initial_metric_value:
        logger.warning((
            f'{context} FireANTs align optimization failed to improve metric: '
            f'initial NCC: {initial_metric_value}, '
            f'final NCC: {final_metric_value} '
        ))
        return False
    logger.info((
        f'{context} FireANTs align succeeded: '
        f'(initial NCC={initial_metric_value:.6f}, '
        f'final NCC={final_metric_value:.6f}) '
    ))
    return True


def _run_registration(reg, seed, context):
    """Optimize reg, logging the last loss of every level at DEBUG"""
    import torch

    loss_logger = _LevelLossLogger(reg.loss_fn, context)
    reg.loss_fn = loss_logger
    if seed is not None:
        torch.manual_seed(seed)
    reg.optimize()
    loss_logger.flush()


def fireants_deformable_align(
    fix,
    mov,
    fix_spacing,
    mov_spacing,
    method='greedy',
    scales=(4, 2, 1),
    iterations=(200, 100, 50),
    loss='cc',
    cc_kernel_size=5,
    optimizer_lr=0.25,
    smooth_warp_sigma=0.5,
    smooth_grad_sigma=1.0,
    tolerance=1e-6,
    max_tolerance_iters=10,
    device='auto',
    torch_threads=None,
    seed=None,
    clip_percentiles=(0.01, 99.99),
    alignment_spacing=None,
    fix_mask=None,
    mov_mask=None,
    fix_roi=None,
    fix_origin=None,
    mov_origin=None,
    static_transform_list=[],
    default=None,
    final_metric_check=True,
    context='',
):
    """
    Register moving to fixed image with FireANTs diffeomorphic deformable
    registration (https://github.com/rohitrango/FireANTs).

    FireANTs is optimization based (no training, no weights) and runs on
    CUDA, Apple MPS or CPU through PyTorch. The step only does the
    deformable part: run bigstream's affine/ransac steps before it, they come
    in through static_transform_list.

    Requires the optional `fireants` dependencies (torch and fireants); the
    pixi `fireants` environment provides them.

    Parameters
    ----------
    fix : ndarray
        The fixed image.

    mov : ndarray
        The moving image; fix.ndim must equal mov.ndim.

    fix_spacing : 1d array
        Physical spacing between voxels of the fixed image.

    mov_spacing : 1d array
        Physical spacing between voxels of the moving image.

    method : str (default: 'greedy')
        'greedy' (GreedyRegistration: one-directional diffeomorphic, one warp
        field) or 'syn' (SyNRegistration: symmetric diffeomorphic, two warp
        fields, roughly twice the memory).

    scales : list of int (default: (4, 2, 1))
        Downsampling factor per pyramid level, decreasing, coarse -> fine.

    iterations : list of int (default: (200, 100, 50))
        Iterations per pyramid level; same length as scales.

    loss : str (default: 'cc')
        Similarity metric: 'cc' (local normalized cross correlation), 'mi'
        (mutual information) or 'mse'. With a fix_mask or mov_mask the masked
        variant ('masked_' + loss) is used.

    cc_kernel_size : int (default: 5)
        Window size of the local CC metric, in voxels of the current level.

    optimizer_lr : float (default: 0.25)
        Adam learning rate.

    smooth_warp_sigma : float (default: 0.5)
        Gaussian sigma for smoothing the warp field, in voxels.

    smooth_grad_sigma : float (default: 1.0)
        Gaussian sigma for smoothing the warp gradient, in voxels.

    tolerance : float (default: 1e-6)
        Convergence tolerance on the loss.

    max_tolerance_iters : int (default: 10)
        Number of iterations within tolerance before a level stops early.

    device : str (default: 'auto')
        'auto' | 'cuda' | 'cuda:N' | 'mps' | 'cpu'. 'auto' picks cuda, then
        mps, then cpu. On a cluster the scheduler sets CUDA_VISIBLE_DEVICES
        per worker and 'cuda' means that worker's GPU; run one dask worker per
        GPU with one thread so blocks don't contend for the card.
        An op missing on the device raises NotImplementedError instead of
        returning default; set PYTORCH_ENABLE_MPS_FALLBACK=1 in the
        environment to have torch fall back to the CPU for such ops.

    torch_threads : int (default: None)
        If given, torch.set_num_threads(torch_threads) once per process, so
        CPU workers don't oversubscribe on top of dask's own threads.

    seed : int (default: None)
        If given, torch.manual_seed(seed) before optimizing.

    clip_percentiles : tuple of two floats (default: (0.01, 99.99))
        Clip each image to these intensity percentiles before registering, so
        a few very bright voxels don't squeeze the tissue into a few grey
        levels. None disables clipping.

    alignment_spacing : float (default: None)
        Skip-sample fix and mov to approximately this voxel spacing before
        registering. scales, cc_kernel_size and the smoothing sigmas are in
        voxels and applied equally on all axes, so for anisotropic data
        (e.g. coarser z) use this to make the working grid closer to
        isotropic.

    fix_mask : ndarray, tuple, or callable (default: None)
        Foreground mask for the fixed image, used as a metric mask (the
        images are not zeroed). The returned field is not masked.

    mov_mask : ndarray, tuple, or callable (default: None)
        Foreground mask for the moving image, used as a metric mask.

    fix_roi : tuple of slices (default: None)
        Voxel region of the fixed image the fixed mask is restricted to.

    fix_origin : 1d array (default: None)
        Physical origin of the fixed image.

    mov_origin : 1d array (default: None)
        Physical origin of the moving image.

    static_transform_list : list of ndarray (default: [])
        Transforms applied to the moving image before registering. The moving
        image is resampled onto the fixed working grid through their
        composite; the returned field is the FireANTs residual only.

    default : any (default: None)
        Returned on failure. If None, a zero displacement field matching
        fix.shape + (ndim,) is used.

    final_metric_check : bool (default: True)
        Return default if the global NCC between fix and the warped mov
        (inside the fixed mask, if any) is lower than before registering.

    context : str (default: '')
        Prefix for log messages.

    Returns
    -------
    params : 1d array
        Flattened displacement field (field.ravel(), float32).

    field : ndarray, shape fix.shape + (ndim,), float32
        Dense displacement field on the original (pre-alignment-spacing)
        fixed image grid, in physical units, zyx components.

    Notes
    -----
    Device memory: from KonfAI's declared cost for the SyN preset, about
    1150 bytes per voxel of the block including its halo. That is an upper
    bound for 'greedy', which keeps one warp field instead of SyN's two:

        block   voxels   ~ device memory (SyN, upper bound for greedy)
        128^3   2.1 M    2.4 GB
        192^3   7.1 M    8.1 GB
        256^3   16.8 M   19 GB

    On MPS the budget is the Mac's unified memory. Block size (processing
    size + halo) is the knob; running out of memory returns default.

    FireANTs needs the fixed image to be at least 32 voxels along every
    axis. A smaller fixed working grid (e.g. a small block, or a thin slab
    after alignment_spacing) is zero-padded at its upper end up to 32, with
    the padding left out of the metric, and the field is cropped back. Fits
    along a much shorter axis are weak: there is little image to register.
    """
    _check_schedule(scales, iterations)
    if method not in ('greedy', 'syn'):
        raise ValueError(f"method must be 'greedy' or 'syn'; got {method!r}")

    # store original fixed image grid; the returned field lives on this grid
    initial_fix_shape = fix.shape
    initial_fix_spacing = fix_spacing
    initial_fix_origin = fix_origin

    fix, mov, fix_mask, mov_mask = _prepare_images(
        fix, mov, fix_spacing, mov_spacing,
        alignment_spacing, fix_mask, mov_mask, fix_roi,
        fix_origin, mov_origin, static_transform_list,
        clip_percentiles, loss, context,
    )
    ndim = fix.GetDimension()

    # set default identity field
    if default is None:
        zero_field = np.zeros(initial_fix_shape + (ndim,), dtype=np.float32)
        default = (zero_field.ravel(), zero_field)

    # a constant image (e.g. a background block) has nothing to register to,
    # and FireANTs still moves it
    if _is_flat(fix) or _is_flat(mov):
        logger.info(f'{context} constant fixed or moving image, returning default')
        return default

    # torch/fireants are optional dependencies: import them only when the
    # step actually runs
    import torch
    from fireants.registration.greedy import GreedyRegistration
    from fireants.registration.syn import SyNRegistration
    from fireants.utils.globals import MIN_IMG_SIZE

    # FireANTs sizes its warp field at least MIN_IMG_SIZE (32) voxels per axis,
    # so on a smaller fixed grid its full resolution level fails. Zero-pad the
    # fixed image up to that size, and keep the padding out of the metric with
    # the fixed mask; the field is cropped back by the final resample.
    fix, fix_mask = _pad_fixed_to_min_size(fix, fix_mask, MIN_IMG_SIZE, context)

    dev = _start_torch(torch_threads, device, context)

    reg = fixed = moving = None
    try:
        fixed, moving, loss_type = _batched_images(fix, mov, fix_mask, mov_mask, loss, dev)

        registration_class = GreedyRegistration if method == 'greedy' else SyNRegistration
        reg = registration_class(
            scales=list(scales),
            iterations=list(iterations),
            fixed_images=fixed,
            moving_images=moving,
            loss_type=loss_type,
            cc_kernel_size=cc_kernel_size,
            optimizer_lr=optimizer_lr,
            smooth_warp_sigma=smooth_warp_sigma,
            smooth_grad_sigma=smooth_grad_sigma,
            tolerance=tolerance,
            max_tolerance_iters=max_tolerance_iters,
            loss_params={},
            init_affine=None,
            progress_bar=False,
        )
        _run_registration(reg, seed, context)

        with torch.no_grad():
            # c: where FireANTs samples mov for each fix voxel, normalized
            # [-1, 1] coordinates of mov, xyz components, shape [1, Z, Y, X, 3]
            c = reg.get_warped_coordinates(fixed, moving)
            # c0: the identity grid, the same coordinates of the fix voxels
            identity = torch.eye(ndim, ndim + 1, device=c.device, dtype=c.dtype)[None]
            c0 = torch.nn.functional.affine_grid(
                identity, [1, 1] + list(c.shape[1:-1]), align_corners=True,
            )
            # map both to physical space with the torch2phy matrices and
            # subtract: d_xyz = T_m c - T_f c0
            T_m = moving.get_torch2phy()[0].to(c.dtype)
            T_f = fixed.get_torch2phy()[0].to(c.dtype)
            d_xyz = (
                torch.einsum('ij,...j->...i', T_m[:ndim, :ndim], c) + T_m[:ndim, ndim]
                - torch.einsum('ij,...j->...i', T_f[:ndim, :ndim], c0) - T_f[:ndim, ndim]
            )
            d_xyz = d_xyz[0].cpu().numpy().astype(np.float32)
            if not np.all(np.isfinite(d_xyz)):
                raise ValueError('non-finite displacement')
            moved = reg.evaluate(fixed, moving)[0, 0].cpu().numpy() if final_metric_check else None
    except NotImplementedError:
        # an op missing on the device is a configuration error, not a block failure
        raise
    except Exception as e:
        logger.error(f'{context} FireANTs registration failed on {dev}: {e}')
        logger.info(f'{context} Returning default')
        return default
    finally:
        del reg, fixed, moving
        _empty_device_cache(dev)

    if final_metric_check and not _metric_improved(fix, mov, fix_mask, moved, context):
        logger.info(f'{context} FireANTs deformable align returning default')
        return default

    # resample the displacement field from the working grid back to the
    # original (pre-alignment-spacing) fix grid, so the returned field always
    # has shape fix.shape + (ndim,). d_xyz is in SITK's xyz component order,
    # as the field demons_align resamples
    disp = sitk.GetImageFromArray(d_xyz, isVector=True)
    disp.CopyInformation(fix)
    ref_origin = (
        np.asarray(initial_fix_origin, dtype=np.float64)
        if initial_fix_origin is not None
        else np.zeros(ndim)
    )
    ref = ut.numpy_to_sitk(
        np.zeros(initial_fix_shape, dtype=np.float32),
        np.asarray(initial_fix_spacing, dtype=np.float64),
        ref_origin,
    )
    # skip-sampling can end the working grid up to stride - 1 voxels before
    # the last fix voxel; extrapolate there instead of returning zero
    disp_full = sitk.Resample(
        disp, ref, sitk.Transform(ndim, sitk.sitkIdentity),
        sitk.sitkLinear, 0.0, sitk.sitkVectorFloat32,
        useNearestNeighborExtrapolator=True,
    )

    # convert from SITK XYZ vector components to bigstream ZYX convention
    field = np.ascontiguousarray(
        sitk.GetArrayFromImage(disp_full).astype(np.float32)[..., ::-1]
    )
    params = field.ravel()
    return params, field


def fireants_affine_align(
    fix,
    mov,
    fix_spacing,
    mov_spacing,
    method='affine',
    scales=(4, 2, 1),
    iterations=(200, 100, 50),
    loss='cc',
    cc_kernel_size=5,
    optimizer_lr=3e-2,
    tolerance=1e-6,
    max_tolerance_iters=10,
    device='auto',
    torch_threads=None,
    seed=None,
    clip_percentiles=(0.01, 99.99),
    initial_condition=None,
    alignment_spacing=None,
    fix_mask=None,
    mov_mask=None,
    fix_roi=None,
    fix_origin=None,
    mov_origin=None,
    static_transform_list=[],
    default=None,
    final_metric_check=True,
    context='',
):
    """
    Register moving to fixed image with FireANTs affine or rigid registration
    (https://github.com/rohitrango/FireANTs), on CUDA, Apple MPS or CPU
    through PyTorch.

    Requires the optional `fireants` dependencies (torch and fireants); the
    pixi `fireants` environment provides them.

    Parameters
    ----------
    fix : ndarray
        The fixed image.

    mov : ndarray
        The moving image; fix.ndim must equal mov.ndim.

    fix_spacing : 1d array
        Physical spacing between voxels of the fixed image.

    mov_spacing : 1d array
        Physical spacing between voxels of the moving image.

    method : str (default: 'affine')
        'affine' (AffineRegistration: rotation, scale, shear and translation)
        or 'rigid' (RigidRegistration: rotation and translation only).

    scales : list of int (default: (4, 2, 1))
        Downsampling factor per pyramid level, decreasing, coarse -> fine.
        FireANTs never downsamples a level below 32 voxels per axis.

    iterations : list of int (default: (200, 100, 50))
        Iterations per pyramid level; same length as scales.

    loss : str (default: 'cc')
        Similarity metric: 'cc' (local normalized cross correlation), 'mi'
        (mutual information) or 'mse'. With a fix_mask or mov_mask the masked
        variant ('masked_' + loss) is used.

    cc_kernel_size : int (default: 5)
        Window size of the local CC metric, in voxels of the current level.

    optimizer_lr : float (default: 3e-2)
        Adam learning rate (FireANTs' own default for affine and rigid).

    tolerance : float (default: 1e-6)
        Convergence tolerance on the loss.

    max_tolerance_iters : int (default: 10)
        Number of iterations within tolerance before a level stops early.

    device : str (default: 'auto')
        'auto' | 'cuda' | 'cuda:N' | 'mps' | 'cpu', as for
        fireants_deformable_align. An op missing on the device raises
        NotImplementedError instead of returning default.

    torch_threads : int (default: None)
        If given, torch.set_num_threads(torch_threads) once per process.

    seed : int (default: None)
        If given, torch.manual_seed(seed) before optimizing.

    clip_percentiles : tuple of two floats (default: (0.01, 99.99))
        Clip each image to these intensity percentiles before registering.
        None disables clipping.

    initial_condition : str or (ndim + 1, ndim + 1) ndarray (default: None)
        How to begin the optimization; FireANTs starts from the identity,
        which only converges when the images are already within about a
        feature size of each other. The string 'PHASE_CORRELATION' starts
        from the integer voxel translation that maximizes the
        cross-correlation of the images at the alignment spacing (the
        identity if it cannot be estimated). A matrix, in physical units
        and zyx order mapping fixed to moving coordinates as returned by
        affine_align, starts from that transform. FireANTs refines from
        there and the starting transform is composed into the returned
        matrix. The final metric check and `default` also start from it,
        as in affine_align; `default` is the identity for a string.

    alignment_spacing : float (default: None)
        Skip-sample fix and mov to approximately this voxel spacing before
        registering.

    fix_mask : ndarray, tuple, or callable (default: None)
        Foreground mask for the fixed image, used as a metric mask.

    mov_mask : ndarray, tuple, or callable (default: None)
        Foreground mask for the moving image, used as a metric mask.

    fix_roi : tuple of slices (default: None)
        Voxel region of the fixed image the fixed mask is restricted to.

    fix_origin : 1d array (default: None)
        Physical origin of the fixed image.

    mov_origin : 1d array (default: None)
        Physical origin of the moving image.

    static_transform_list : list of ndarray (default: [])
        Transforms applied to the moving image before registering. The moving
        image is resampled onto the fixed working grid through their
        composite; the returned matrix is the FireANTs residual only, applied
        before them, as for affine_align. They are also the way to give the
        optimization a starting point: FireANTs starts from the identity.

    default : any (default: None)
        Returned on failure. If None, the identity matrix.

    final_metric_check : bool (default: True)
        Return default if the global NCC between fix and the transformed mov
        (inside the fixed mask, if any) is lower than before registering.

    context : str (default: '')
        Prefix for log messages.

    Returns
    -------
    affine : ndarray, shape (ndim + 1, ndim + 1)
        Affine matrix in physical units and zyx order, mapping fixed image
        coordinates to moving image coordinates, the same convention as
        affine_align.

    Notes
    -----
    FireANTs optimizes the matrix with Adam at a fixed learning rate, so it
    stops within a fraction of a voxel to about a voxel of the optimum
    rather than converging tightly; a following fireants_deform (or deform)
    step absorbs the rest. A smaller optimizer_lr needs proportionally more
    iterations to cover the same misalignment.

    FireANTs reads zeros past the moving image's edge. When the moving image
    covers only the fixed field of view, as distributed_align's blocks do
    and as mov does after the static_transform_list pre-warp, edge voxels
    sample those zeros under any residual misalignment, which biases the fit
    slightly toward shrinking (about half a voxel for a few voxels of
    shift in tests). A moving image larger than the fixed field of view
    under the fitted transform avoids it.
    """
    _check_schedule(scales, iterations)
    if method not in ('affine', 'rigid'):
        raise ValueError(f"method must be 'affine' or 'rigid'; got {method!r}")
    initial_matrix = None
    if isinstance(initial_condition, str):
        if initial_condition != 'PHASE_CORRELATION':
            raise ValueError(
                f"initial_condition string must be 'PHASE_CORRELATION'; got {initial_condition!r}"
            )
    elif initial_condition is not None:
        initial_matrix = np.asarray(initial_condition, dtype=np.float64)

    fix, mov, fix_mask, mov_mask = _prepare_images(
        fix, mov, fix_spacing, mov_spacing,
        alignment_spacing, fix_mask, mov_mask, fix_roi,
        fix_origin, mov_origin, static_transform_list,
        clip_percentiles, loss, context,
    )
    ndim = fix.GetDimension()

    if initial_matrix is not None and initial_matrix.shape != (ndim + 1, ndim + 1):
        raise ValueError(
            f'initial_condition matrix must be {(ndim + 1, ndim + 1)}; got {initial_matrix.shape}'
        )

    # default identity affine, or the given initial transform
    if default is None:
        default = np.eye(ndim + 1) if initial_matrix is None else initial_matrix

    # a constant image (e.g. a background block) has nothing to register to,
    # and FireANTs still moves it
    if _is_flat(fix) or _is_flat(mov):
        logger.info(f'{context} constant fixed or moving image, returning default')
        return default

    # start FireANTs from the initial condition instead of the identity,
    # as xyz matrix mapping fixed to moving physical points
    initial_xyz = None
    if initial_condition == 'PHASE_CORRELATION':
        try:
            offset, voxels = phase_correlation_translation(fix, mov)
            # a pure translation: move the images' origin, no resampling
            mov = _translated_image(mov, offset)
            if mov_mask is not None:
                mov_mask = _translated_image(mov_mask, offset)
            initial_xyz = np.eye(ndim + 1)
            initial_xyz[:ndim, ndim] = offset
            logger.info((
                f'{context} PHASE_CORRELATION shift (voxels, zyx): {voxels}, '
                f'translation (physical, xyz): {offset}'
            ))
        except Exception as e:
            logger.warning(f'{context} phase correlation failed ({e}); starting from identity')
    elif initial_matrix is not None:
        tx = bst.matrix_to_affine_transform(initial_matrix)
        mov = sitk.Resample(mov, fix, tx, sitk.sitkLinear, 0.0)
        if mov_mask is not None:
            mov_mask = sitk.Resample(mov_mask, fix, tx, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        initial_xyz = bst.invert_matrix_axes(initial_matrix)
        logger.info(f'{context} initial condition matrix: {initial_matrix}')

    # torch/fireants are optional dependencies: import them only when the
    # step actually runs
    import torch
    from fireants.registration.affine import AffineRegistration
    from fireants.registration.rigid import RigidRegistration

    dev = _start_torch(torch_threads, device, context)

    reg = fixed = moving = None
    try:
        fixed, moving, loss_type = _batched_images(fix, mov, fix_mask, mov_mask, loss, dev)

        registration_class = AffineRegistration if method == 'affine' else RigidRegistration
        reg = registration_class(
            scales=list(scales),
            iterations=list(iterations),
            fixed_images=fixed,
            moving_images=moving,
            loss_type=loss_type,
            cc_kernel_size=cc_kernel_size,
            optimizer_lr=optimizer_lr,
            tolerance=tolerance,
            max_tolerance_iters=max_tolerance_iters,
            loss_params={},
            progress_bar=False,
        )
        _run_registration(reg, seed, context)

        with torch.no_grad():
            # y = A x + t: fixed physical point x -> moving physical point y,
            # xyz order, as a homogeneous (ndim + 1) x (ndim + 1) matrix
            matrix = reg.get_affine_matrix() if method == 'affine' else reg.get_rigid_matrix()
            matrix_xyz = matrix[0].cpu().numpy().astype(np.float64)
            if not np.all(np.isfinite(matrix_xyz)):
                raise ValueError('non-finite affine matrix')
            moved = reg.evaluate(fixed, moving)[0, 0].cpu().numpy() if final_metric_check else None
            if initial_xyz is not None:
                # the matrix maps fixed points to the moving image after the
                # initial condition: follow it by the initial condition
                matrix_xyz = initial_xyz @ matrix_xyz
    except NotImplementedError:
        # an op missing on the device is a configuration error, not a block failure
        raise
    except Exception as e:
        logger.error(f'{context} FireANTs registration failed on {dev}: {e}')
        logger.info(f'{context} Returning default')
        return default
    finally:
        del reg, fixed, moving
        _empty_device_cache(dev)

    if final_metric_check and not _metric_improved(fix, mov, fix_mask, moved, context):
        logger.info(f'{context} FireANTs affine align returning default')
        return default

    # bigstream matrices map fixed to moving physical coordinates too, but in
    # zyx order: reverse the axes
    affine = bst.invert_matrix_axes(matrix_xyz)
    logger.info(f'{context} FireANTs affine align matrix: {affine}')
    return affine
