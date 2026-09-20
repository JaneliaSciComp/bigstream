"""
Tests for `deform_field_diagnostics` and its reporting half.

The split exists so the numbers can be computed on a dask worker and logged
by whoever gathers them: `deform_field_diagnostics` returns a small dict of
scalars, `log_deform_field_diagnostics` renders it. What these pin down is
that the dict stays small enough to ship through the scheduler, and that
rendering it says what the single combined function used to say.
"""

import logging
import pickle

import numpy as np
import pytest

from bigstream.diagnostics import (deform_field_diagnostics,
                                   log_deform_field_diagnostics)


SPACING = np.array([1.0, 1.0, 1.0])


def _smooth_field(shape=(16, 16, 16)):
    zz, yy, xx = np.meshgrid(*[np.linspace(0, 1, s) for s in shape],
                             indexing='ij')
    field = np.zeros(shape + (3,), dtype=np.float32)
    field[..., 0] = np.sin(6.0 * zz)
    field[..., 1] = 0.5 * yy
    field[..., 2] = 0.25 * xx
    return field


def _folded_field(shape=(16, 16, 16)):
    """A ramp steep enough that det(I + grad(u)) goes negative."""
    field = np.zeros(shape + (3,), dtype=np.float32)
    field[..., 0] = -3.0 * np.arange(shape[0], dtype=np.float32)[:, None, None]
    return field


@pytest.fixture
def capture(caplog):
    caplog.set_level(logging.DEBUG, logger='bigstream.diagnostics')
    return caplog


def test_diagnostics_return_the_statistics():
    stats = deform_field_diagnostics(_smooth_field(), SPACING)
    assert stats['n_folded'] == 0
    assert stats['has_nan'] is False and stats['has_inf'] is False
    assert stats['jacobian_min'] <= stats['jacobian_mean'] <= stats['jacobian_max']
    assert stats['magnitude_min'] <= stats['magnitude_max']
    assert set(stats['smoothness']) == {'dx', 'dy', 'dz'}


def test_the_statistics_are_small_whatever_the_field_size():
    """They cross the scheduler once per region, so they have to stay tiny."""
    small = deform_field_diagnostics(_smooth_field((16, 16, 16)), SPACING)
    large = deform_field_diagnostics(_smooth_field((64, 64, 64)), SPACING)
    # a 64^3 field is 64x the voxels of a 16^3 one; the summary must not grow
    assert len(pickle.dumps(large)) < 1024
    assert abs(len(pickle.dumps(large)) - len(pickle.dumps(small))) < 64


def test_the_statistics_survive_a_round_trip():
    """They are returned from a worker, so they have to pickle exactly."""
    stats = deform_field_diagnostics(_folded_field(), SPACING)
    assert pickle.loads(pickle.dumps(stats)) == stats


def test_folding_is_counted_and_split_by_location():
    stats = deform_field_diagnostics(_folded_field(), SPACING)
    assert stats['n_folded'] > 0
    assert (stats['n_face_folded'] + stats['n_interior_folded']
            == stats['n_folded'])


def test_nan_and_inf_are_reported():
    field = _smooth_field()
    field[2, 2, 2, 0] = np.nan
    field[3, 3, 3, 1] = np.inf
    stats = deform_field_diagnostics(field, SPACING)
    assert stats['has_nan'] is True and stats['has_inf'] is True


def test_computing_does_not_log_unless_a_level_is_given(capture):
    deform_field_diagnostics(_smooth_field(), SPACING, context='QUIET')
    assert 'QUIET' not in capture.text

    deform_field_diagnostics(_smooth_field(), SPACING, context='LOUD',
                             level=logging.INFO)
    assert 'LOUD' in capture.text


def test_reporting_renders_every_section(capture):
    stats = deform_field_diagnostics(_smooth_field(), SPACING)
    log_deform_field_diagnostics(stats, context='CTX', level=logging.INFO)
    text = capture.text
    assert 'CTX Deform align jacobian determinant' in text
    assert 'CTX Deform align field stats' in text
    for axis in ('dx', 'dy', 'dz'):
        assert f'CTX Deform align field smoothness {axis}' in text


def test_folding_is_escalated_to_error_when_reported_at_info(capture):
    """
    A fold in an assembled field is a real defect, so it outranks the level
    the rest of the sweep is reported at - but only when that level is INFO
    or above. A per-block sweep at DEBUG stays debug noise, because a
    block's overlaps are not final until its neighbours have contributed.
    """
    stats = deform_field_diagnostics(_folded_field(), SPACING)

    capture.clear()
    log_deform_field_diagnostics(stats, context='CTX', level=logging.INFO)
    assert any(r.levelno == logging.ERROR for r in capture.records)

    capture.clear()
    log_deform_field_diagnostics(stats, context='CTX', level=logging.DEBUG)
    assert all(r.levelno == logging.DEBUG for r in capture.records)


def test_reporting_is_skipped_when_the_level_is_disabled(caplog):
    caplog.set_level(logging.WARNING, logger='bigstream.diagnostics')
    stats = deform_field_diagnostics(_smooth_field(), SPACING)
    log_deform_field_diagnostics(stats, context='CTX', level=logging.INFO)
    assert 'CTX' not in caplog.text
