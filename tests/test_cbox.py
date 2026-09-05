"""Tests for c-box construction and confidence intervals."""

from __future__ import annotations

import numpy as np
import pytest

from pba import cBox, cBox2, computeConfidenceInterval, computeFocalElements


class TestCboxConstruction:
    def test_scalar_input_shape(self):
        cb = cBox(2, 10, nsteps=500)
        assert set(cb.keys()) == {"support", "lb", "ub", "betaparam", "flag"}
        assert cb["support"].shape == (500,)
        assert cb["lb"].shape == (500,)
        assert cb["ub"].shape == (500,)

    def test_cdfs_are_monotone(self):
        cb = cBox(3, 10)
        assert np.all(np.diff(cb["lb"]) >= -1e-12)
        assert np.all(np.diff(cb["ub"]) >= -1e-12)

    def test_lb_dominates_ub_as_cdf(self):
        # The 'lb' key holds the lower-confidence-limit CDF, which lies
        # pointwise above the upper-confidence-limit CDF ('ub').
        cb = cBox(4, 10)
        assert np.all(cb["lb"] >= cb["ub"] - 1e-12)

    def test_cdfs_span_unit_interval(self):
        cb = cBox(2, 10)
        assert cb["lb"][0] == pytest.approx(0, abs=1e-9)
        assert cb["lb"][-1] == pytest.approx(1, abs=1e-3)

    def test_degenerate_flag_when_k_equals_n(self):
        cb = cBox(5, 5)
        assert cb["flag"] is True

    def test_non_degenerate_flag(self):
        cb = cBox(2, 10)
        assert cb["flag"] is False

    def test_zero_successes_uses_tolerance(self):
        # Should not raise.
        cb = cBox(0, 10)
        assert cb["betaparam"]["param_alpha_left"] == pytest.approx(1e-6)

    def test_interval_k(self):
        cb = cBox([1, 3], 10)
        assert cb["betaparam"]["param_alpha_left"] == 1
        assert cb["betaparam"]["param_alpha_right"] == 4  # k_hi + 1

    def test_inverted_interval_raises(self):
        with pytest.raises(ValueError, match="Lower bound"):
            cBox([5, 2], 10)

    def test_cBox2_uses_k_plus_m_as_n(self):
        cb = cBox2(2, 8)  # equivalent to cBox(2, 10)
        cb_reference = cBox(2, 10)
        assert cb["betaparam"] == cb_reference["betaparam"]


class TestComputeConfidenceInterval:
    def test_default_levels(self):
        interval = computeConfidenceInterval(cBox(2, 10))
        assert len(interval) == 2
        lo, hi = interval
        assert 0 <= lo <= hi <= 1

    def test_custom_levels(self):
        interval = computeConfidenceInterval(cBox(3, 10), alpha_level=0.1, beta_level=0.9)
        lo, hi = interval
        assert 0 <= lo <= hi <= 1

    def test_degenerate_c_box(self):
        interval = computeConfidenceInterval(cBox(5, 5))
        _, hi = interval
        assert hi == pytest.approx(1, abs=1e-3)


class TestComputeFocalElements:
    def test_length(self):
        fes = computeFocalElements(cBox(2, 10), npoints=25)
        assert len(fes) == 25

    def test_each_is_pair(self):
        fes = computeFocalElements(cBox(2, 10), npoints=10)
        for fe in fes:
            assert len(fe) == 2
            assert fe[0] <= fe[1] + 1e-9
