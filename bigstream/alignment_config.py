"""
Reading the alignment description out of a bigstream config.

    load_align_config              the bundled defaults with a user config
                                   layered on top
    get_alignment_passes           the passes to run for a context
                                   (`local_align`, `global_align`), falling
                                   back to its single `steps:` list
    AlignmentPass                  one pass: a block lattice plus the steps
                                   run on each block

    _alignment_passes_from_config  the `<context>.alignment_passes` section
    _alignment_steps_from_config   one `alignment_steps` list, in
                                   `alignment_pipeline` form

"""

import logging

from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pydantic.v1.utils as pu
import yaml

from .blend_ramp import parse_blend_ramp
from .configure_bigstream import default_bigstream_config_str


logger = logging.getLogger(__name__)


@dataclass
class AlignmentPass:
    """
    One pass over the volume: a block lattice plus the steps run on each block.

    Every geometric field is in **voxels, zyx order**, matching the rest of
    the bigstream config (the xyz-ordered CLI values are reversed long before
    they reach here).

    alignment_steps : list of (str, dict)
        What to run on each block, in `alignment_pipeline` form.

    processing_size : tuple of int (default: None)
        `B`, the block step. None means "inherit the top level default".

    processing_offset : tuple of int (default: None)
        The lattice phase. Block origins sit at `offset + n*B` measured from
        **voxel 0 of the volume**, not from the ROI - see
        `_BlockLattice`. None means 0.

        A later pass wants an offset that puts its block seams where the
        previous pass had block interiors; `B/4` is a reasonable stagger.

    processing_halo_factor : float or tuple of float (default: None)
        Halo as a fraction of `processing_size`, **per side**. This is the
        same quantity the old single-pass pipeline called `overlap_factor`;
        the name differs only to make the per-side reading explicit. Ignored
        when `processing_halo` is given.

    processing_halo : tuple of int (default: None)
        Halo in voxels, per side. Takes precedence over the factor.

    blend_ramp : str (default: None)
        Shape of the overlap-add blending weight ramp, `'linear'` or
        `'cosine'`. None means "inherit the top level default", which is
        itself `'linear'` when unset. See `bigstream.blend_ramp`: a cosine
        ramp has no kink where it meets the block core but is `pi/2` steeper
        mid-ramp, which tightens the fold-safe displacement ceiling by the
        same factor.

    max_displacement, bspline_constraints
        Per-pass overrides of the blend-safety budget - see
        `blockwise_alignment_pipeline`. Both are only used for the
        *reporting* in `_check_blend_safe_displacement`; the values that
        actually constrain the fit live in the individual steps' arguments.
    """

    alignment_steps: List[Tuple[str, dict]] = field(default_factory=list)
    processing_size: Optional[Tuple[int, ...]] = None
    processing_offset: Optional[Tuple[int, ...]] = None
    processing_halo_factor: Optional[Tuple[float, ...]|float] = None
    processing_halo: Optional[Tuple[int, ...]] = None
    blend_ramp: Optional[str] = None
    name: Optional[str] = None

    def resolved(self, ndim,
                 default_processing_size=None,
                 default_halo_factor=None,
                 default_blend_ramp=None):
        """
        A copy with every geometric field filled in as a length-`ndim` tuple.

        Raises if a pass ends up with no block size, because there is no
        sensible fallback for it.
        """
        size = _as_int_tuple(
            _first_not_none(self.processing_size, default_processing_size),
            ndim, 'processing_size')
        if size is None:
            raise ValueError(
                'no processing_size for alignment pass '
                f'{self.name or "(unnamed)"}: set it on the pass or as the '
                'top level local_align processing_size'
            )
        if any(s <= 0 for s in size):
            raise ValueError(f'processing_size must be positive, got {size}')

        offset = _as_int_tuple(self.processing_offset, ndim,
                               'processing_offset')
        offset = (0,) * ndim if offset is None else offset

        halo = _as_int_tuple(self.processing_halo, ndim, 'processing_halo')
        if halo is None:
            factor = _first_not_none(self.processing_halo_factor,
                                     default_halo_factor)
            if factor is None:
                raise ValueError(
                    'no processing_halo or processing_halo_factor for '
                    f'alignment pass {self.name or "(unnamed)"}'
                )
            factor = np.broadcast_to(
                np.atleast_1d(np.asarray(factor, dtype=float)), (ndim,))
            if np.any(factor <= 0) or np.any(factor >= 1):
                raise ValueError(
                    'processing_halo_factor is a per-side fraction of the '
                    f'block size and must be in (0, 1), got {factor.tolist()}'
                )
            halo = tuple(int(h) for h in np.round(np.array(size) * factor))
        if any(h <= 0 for h in halo):
            raise ValueError(
                f'processing_halo must be positive, got {halo}; a zero halo '
                'leaves no ramp to blend over'
            )

        # raises on an unknown name; a typo here would otherwise silently
        # blend with one ramp and bound the displacement with another
        ramp = parse_blend_ramp(
            _first_not_none(self.blend_ramp, default_blend_ramp))

        return AlignmentPass(
            alignment_steps=list(self.alignment_steps),
            processing_size=size,
            processing_offset=offset,
            processing_halo_factor=self.processing_halo_factor,
            processing_halo=halo,
            blend_ramp=ramp,
            name=self.name,
        )


def _first_not_none(*values):
    for v in values:
        if v is not None:
            return v
    return None


def _as_int_tuple(value, ndim, name):
    if value is None:
        return None
    arr = np.atleast_1d(np.asarray(value))
    if arr.size == 1:
        arr = np.full(ndim, arr[0])
    if arr.size != ndim:
        raise ValueError(
            f'{name} must be a scalar or have {ndim} values (zyx), got {value}'
        )
    return tuple(int(v) for v in arr)


def _deep_merge(base, override):
    merged = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def _step_defaults(config, context_config):
    """
    Per-step default arguments: top level `<step>:` sections overlaid with
    the `<context>.<step>:` sections.
    """
    defaults = {k: v for k, v in config.items() if isinstance(v, dict)}
    for k, v in context_config.items():
        if isinstance(v, dict):
            defaults[k] = _deep_merge(defaults.get(k) or {}, v)
    return defaults


def _alignment_passes_from_config(config, context):
    """
    Read the `alignment_passes` section of a bigstream config.

        local_align:
            processing_size: [384, 384, 384]        # default for every pass
            processing_halo_factor: 0.2             # default for every pass
            blend_ramp: cosine                      # default for every pass
            alignment_passes:
                - processing_offset: [0, 0, 0]
                  processing_halo_factor: [0.2, 0.2, 0.2]
                  alignment_steps:
                    - ransac: {...}
                    - deform: {...}
                - processing_offset: [96, 96, 96]
                  processing_halo_factor: [0.1, 0.1, 0.1]
                  blend_ramp: linear                # per-pass override
                  alignment_steps:
                    - deform: {...}

    Each pass's step arguments are layered over the per-step defaults: the
    config's top level sections (`ransac:`, `affine:`, `deform:` ...) with the
    `<context>` section's own (`local_align.deform:` ...) merged over them,
    exactly as `get_algorithm_parameters` does for the single-pass `steps:`
    list.

    Returns
    -------
    list of AlignmentPass, unresolved - call `resolved(ndim, ...)` on each, or
    let `blockwise_alignment_pipeline` do it.
    """
    context_config = config.get(context) or {}
    passes_spec = context_config.get('alignment_passes')
    if passes_spec is None and 'alignmentPass' in context_config:
        # the spelling the original prototype config used; everything
        # else here is snake_case
        logger.warning("'alignmentPass' is deprecated, use 'alignment_passes'")
        passes_spec = context_config.get('alignmentPass')
    if not passes_spec:
        return []

    step_defaults = _step_defaults(config, context_config)
    passes = []
    for i, spec in enumerate(passes_spec):
        spec = dict(spec or {})
        passes.append(AlignmentPass(
            alignment_steps=_alignment_steps_from_config(
                spec.get('alignment_steps'), step_defaults=step_defaults),
            processing_size=spec.get('processing_size'),
            processing_offset=spec.get('processing_offset'),
            processing_halo_factor=spec.get('processing_halo_factor'),
            processing_halo=spec.get('processing_halo'),
            blend_ramp=spec.get('blend_ramp'),
            name=spec.get('name', f'pass{i + 1}'),
        ))
    return passes


def _alignment_steps_from_config(steps_spec, step_defaults=None):
    """
    Convert a config `alignment_steps` list into the `[(name, args), ...]`
    form `alignment_pipeline` wants.

    The config writes each step as a single-key mapping so the step name reads
    as a heading over its arguments:

        alignment_steps:
          - ransac:
              alignment_spacing: 4
          - deform:
              control_point_spacing: 128

    A bare string (`- ransac`) is also accepted and means "all defaults".

    Parameters
    ----------
    steps_spec : list
        The `alignment_steps` value from the config.

    step_defaults : dict (default: None)
        Per-step-name default arguments, i.e. the top level `ransac:` /
        `affine:` / `deform:` sections of the bigstream config. A step's
        inline arguments override these key by key, the same precedence
        `get_algorithm_parameters` applies to the single-pass config.

    Returns
    -------
    list of (str, dict)
    """
    step_defaults = step_defaults or {}
    steps = []
    for entry in (steps_spec or []):
        if isinstance(entry, str):
            name, args = entry, {}
        elif isinstance(entry, dict):
            if len(entry) != 1:
                raise ValueError(
                    'each entry of alignment_steps must be a single-key '
                    f'mapping of step name to its arguments, got {entry}'
                )
            (name, args), = entry.items()
            args = dict(args or {})
        else:
            raise ValueError(
                f'unsupported alignment_steps entry {entry!r}; expected a '
                'step name or a single-key mapping'
            )
        if name not in step_defaults:
            logger.warning(
                f'alignment step {name!r} has no config section; '
                'it will run with the library defaults plus its inline args')
        steps.append((name, _deep_merge(step_defaults.get(name) or {}, args)))
    return steps


def load_align_config(config_filename):
    """The bigstream defaults with the user's config layered on top."""
    config = yaml.safe_load(default_bigstream_config_str)
    if config_filename:
        with open(config_filename) as f:
            external_config = yaml.safe_load(f)
        logger.info(f'Read external config from {config_filename}')
        config = pu.deep_update(config, external_config)
    return config


def get_alignment_passes(config, align_steps, context):
    """
    The passes to run, from `<context>.alignment_passes`.

    A config written for the older single-pass pipeline has no
    `alignment_passes`, only a flat `steps:` list. Rather than refuse it, run
    it as one pass - which is exactly what the multi-pass pipeline reduces to
    at one pass and a zero lattice offset, so such a config keeps its old
    meaning.
    """
    passes = _alignment_passes_from_config(config, context)
    if passes:
        return passes

    context_config = config.get(context) or {}
    steps = align_steps or context_config.get('steps') or []
    if not steps:
        return []
    logger.info(f'No {context}.alignment_passes in the config; running the '
                f'single steps list {steps} as one pass')
    # mirror get_algorithm_parameters' precedence: global per-step defaults,
    # then the context overrides for that step. deep_update writes into
    # its first argument, so copy rather than edit the config in place
    step_defaults = {name: pu.deep_update(deepcopy(config.get(name) or {}),
                                          deepcopy(context_config.get(name) or {}))
                     for name in steps}
    return [AlignmentPass(
        alignment_steps=_alignment_steps_from_config(list(steps),
                                                     step_defaults=step_defaults),
        name='pass1',
    )]
