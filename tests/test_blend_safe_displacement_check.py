"""
Tests for `_check_blend_safe_displacement`'s combined affine+deform budget.

`alignment_pipeline` composes every step of a block (e.g. 'affine' then
'deform') into a single field before `distributed_align` blends it, so an
affine step's own `max_displacement` and a deform step's own
`max_displacement` (both top-level parameters of `affine_align`/
`deformable_align` - see `align_constraints.bound_affine_displacement` and
`project_bspline_transform`) are not independent budgets - each can look
individually fine while their sum still exceeds the blend-safe ceiling. This
is the real regression from a run that folded with affine.max_displacement=16
and deform.max_displacement=16 simultaneously configured, on a lattice whose
actual ceiling is ~16.68 - individually compliant, together not even close.
"""
import numpy as np
import pytest

from bigstream.distributed_align import _check_blend_safe_displacement
from bigstream.align_constraints import blend_safe_displacement_bound


# the real run's lattice: processsize [256, 512, 512] zyx, overlap_factor 0.3,
# spacing [1.09, 0.6, 0.6] zyx (post 2x-expansion-corrected)
BLOCK_SIZE = np.array([256, 512, 512])
OVERLAP_FACTOR = 0.3
OVERLAPS = np.round(BLOCK_SIZE * OVERLAP_FACTOR).astype(int)
SPACING = np.array([1.09, 0.6, 0.6])
DEFORM_K = 0.1

SHARED_CEILING = blend_safe_displacement_bound(
    OVERLAPS, SPACING, DEFORM_K, min_jacobian=0.1)
AFFINE_ONLY_CEILING = blend_safe_displacement_bound(
    OVERLAPS, SPACING, 1e-12, min_jacobian=0.1)


def _steps(affine_max_displacement=None, deform_max_displacement=None,
          deform_k=DEFORM_K, include_deform=True):
    steps = [('affine', {'max_displacement': affine_max_displacement}
                        if affine_max_displacement is not None else {})]
    if include_deform:
        deform_args = {'bspline_constraints': {'k': deform_k}}
        if deform_max_displacement is not None:
            deform_args['max_displacement'] = deform_max_displacement
        steps.append(('deform', deform_args))
    return steps


def test_ceiling_sanity():
    """The lattice this whole module is built around actually has ~16.68."""
    assert SHARED_CEILING == pytest.approx(16.68, abs=0.01)


def test_individually_compliant_bounds_still_fail_combined():
    """The exact real-world regression: 16 + 16 on a ~16.68 ceiling."""
    steps = _steps(affine_max_displacement=16.0, deform_max_displacement=16.0)
    with pytest.raises(ValueError, match='exceeds the blend safe ceiling'):
        _check_blend_safe_displacement(
            steps, BLOCK_SIZE, OVERLAPS, SPACING, error_when_check_fails=True)


def test_apportioned_bounds_pass():
    """The same lattice, with the combined bound kept under the ceiling."""
    steps = _steps(affine_max_displacement=6.0, deform_max_displacement=10.0)
    _check_blend_safe_displacement(
        steps, BLOCK_SIZE, OVERLAPS, SPACING, error_when_check_fails=True)


def test_affine_alone_gets_the_full_budget_without_a_deform_constraint():
    """No deform C4 step present -> k~0, the affine step gets the whole ceiling."""
    steps = _steps(affine_max_displacement=AFFINE_ONLY_CEILING - 1.0,
                   include_deform=False)
    _check_blend_safe_displacement(
        steps, BLOCK_SIZE, OVERLAPS, SPACING, error_when_check_fails=True)

    steps = _steps(affine_max_displacement=AFFINE_ONLY_CEILING + 1.0,
                   include_deform=False)
    with pytest.raises(ValueError, match='exceeds the blend safe ceiling'):
        _check_blend_safe_displacement(
            steps, BLOCK_SIZE, OVERLAPS, SPACING, error_when_check_fails=True)


def test_deform_without_max_displacement_still_warns_regardless_of_affine():
    """Deform's own amplitude is unbounded here - no affine bound offsets that."""
    steps = _steps(affine_max_displacement=1.0, deform_max_displacement=None)
    with pytest.raises(ValueError, match='sets bspline_constraints but no'):
        _check_blend_safe_displacement(
            steps, BLOCK_SIZE, OVERLAPS, SPACING, error_when_check_fails=True)


def test_no_bounds_configured_anywhere_is_silent():
    """Backward compatible: opting into nothing raises/warns nothing."""
    steps = [('affine', {}), ('deform', {})]
    _check_blend_safe_displacement(
        steps, BLOCK_SIZE, OVERLAPS, SPACING, error_when_check_fails=True)


def test_irrelevant_steps_do_not_trigger_a_false_positive():
    """A step that never touches this mechanism at all (e.g. ransac) is silent."""
    steps = [('ransac', {'nspots': 2000}), ('rigid', {})]
    _check_blend_safe_displacement(
        steps, BLOCK_SIZE, OVERLAPS, SPACING, error_when_check_fails=True)


def test_warning_message_names_the_shared_budget(caplog):
    steps = _steps(affine_max_displacement=16.0, deform_max_displacement=16.0)
    _check_blend_safe_displacement(
        steps, BLOCK_SIZE, OVERLAPS, SPACING, error_when_check_fails=False)
    assert any("totals 32 across 'affine', 'deform'" in r.message
              for r in caplog.records)
    assert any(f'{SHARED_CEILING:.4g}' in r.message for r in caplog.records)


# --------------------------------------------------------------------------
# the ceiling the check reports has to be the one for the ramp the pass will
# actually blend with - a cosine run that was checked against the linear
# ceiling would pass configs that fold
# --------------------------------------------------------------------------


COSINE_CEILING = blend_safe_displacement_bound(
    OVERLAPS, SPACING, DEFORM_K, min_jacobian=0.1, blend_ramp='cosine')


def test_cosine_rejects_what_linear_accepts():
    """
    The whole reason the bound had to become ramp-aware. A budget that sits
    between the two ceilings is fine on a linear ramp and folds on a cosine
    one, and only the `blend_ramp` argument tells them apart.
    """
    budget = (COSINE_CEILING + SHARED_CEILING) / 2.0
    assert COSINE_CEILING < budget < SHARED_CEILING

    steps = _steps(deform_max_displacement=budget)
    _check_blend_safe_displacement(
        steps, BLOCK_SIZE, OVERLAPS, SPACING, error_when_check_fails=True)

    with pytest.raises(ValueError, match='exceeds the blend safe ceiling'):
        _check_blend_safe_displacement(
            steps, BLOCK_SIZE, OVERLAPS, SPACING, blend_ramp='cosine',
            error_when_check_fails=True)


def test_reported_ceiling_is_the_cosine_one(caplog):
    steps = _steps(affine_max_displacement=16.0, deform_max_displacement=16.0)
    _check_blend_safe_displacement(
        steps, BLOCK_SIZE, OVERLAPS, SPACING, blend_ramp='cosine',
        error_when_check_fails=False)
    assert any(f'{COSINE_CEILING:.4g}' in r.message for r in caplog.records)
    assert not any(f'{SHARED_CEILING:.4g}' in r.message
                   for r in caplog.records)


def test_unset_ramp_checks_against_the_linear_ceiling():
    """Existing callers pass no ramp and must keep the ceiling they had."""
    steps = _steps(deform_max_displacement=SHARED_CEILING - 0.01)
    _check_blend_safe_displacement(
        steps, BLOCK_SIZE, OVERLAPS, SPACING, error_when_check_fails=True)
    steps = _steps(deform_max_displacement=SHARED_CEILING + 0.01)
    with pytest.raises(ValueError, match='exceeds the blend safe ceiling'):
        _check_blend_safe_displacement(
            steps, BLOCK_SIZE, OVERLAPS, SPACING, error_when_check_fails=True)


def test_no_headroom_suggestion_is_also_ramp_aware(caplog):
    """
    The k-too-large branch quotes a ceiling for a suggested k. That figure
    is only useful if it is the one the run would actually get.
    """
    steps = _steps(deform_max_displacement=1.0, deform_k=0.32)
    _check_blend_safe_displacement(
        steps, BLOCK_SIZE, OVERLAPS, SPACING, blend_ramp='cosine',
        error_when_check_fails=False)
    k_max = (1.0 - 0.1) / 3
    suggested = blend_safe_displacement_bound(
        OVERLAPS, SPACING, k_max / 2.0, min_jacobian=0.1, blend_ramp='cosine')
    assert any(f'{suggested:.4g}' in r.message for r in caplog.records)
