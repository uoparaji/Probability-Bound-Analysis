"""Interval arithmetic tests."""

from __future__ import annotations

import math

import pytest

from pba.intervals import (
    _as_interval,
    addIntervals,
    divideIntervals,
    linear_interpolate,
    multiplyIntervals,
    subtractIntervals,
)


class TestAsInterval:
    def test_int_promotes_to_degenerate(self):
        assert _as_interval(3) == (3, 3)

    def test_float_promotes_to_degenerate(self):
        assert _as_interval(2.5) == (2.5, 2.5)

    def test_list_passes_through(self):
        assert _as_interval([1, 2]) == (1, 2)

    def test_tuple_passes_through(self):
        assert _as_interval((0.5, 1.5)) == (0.5, 1.5)


class TestAddIntervals:
    def test_basic(self):
        assert addIntervals([1, 2], [3, 4]) == [4, 6]

    def test_scalar_promotion(self):
        assert addIntervals(5, [1, 2]) == [6, 7]
        assert addIntervals([1, 2], 3) == [4, 5]

    def test_negative(self):
        assert addIntervals([-1, 1], [-2, 2]) == [-3, 3]


class TestSubtractIntervals:
    def test_basic(self):
        # [a_lo - b_hi, a_hi - b_lo]
        assert subtractIntervals([5, 10], [1, 2]) == [3, 9]

    def test_scalar_promotion(self):
        assert subtractIntervals(10, [1, 2]) == [8, 9]

    def test_result_ordering(self):
        lo, hi = subtractIntervals([1, 2], [3, 4])
        assert lo <= hi


class TestMultiplyIntervals:
    def test_positive(self):
        assert multiplyIntervals([1, 2], [3, 4]) == [3, 8]

    def test_sign_change(self):
        # [-1, 2] * [-3, 4]: products {3, -4, -6, 8} => [-6, 8]
        assert multiplyIntervals([-1, 2], [-3, 4]) == [-6, 8]

    def test_scalar(self):
        assert multiplyIntervals(3, [1, 2]) == [3, 6]


class TestDivideIntervals:
    def test_positive(self):
        result = divideIntervals([2, 4], [1, 2])
        # [2/2, 4/1] = [1, 4]
        assert result == [1.0, 4.0]

    def test_divisor_touches_zero_gives_infinite(self):
        lo, hi = divideIntervals([1, 2], [0, 1])
        assert math.isinf(lo) or math.isinf(hi)


class TestLinearInterpolate:
    def test_endpoint(self):
        assert linear_interpolate([0, 1], [10, 20], 0) == 10

    def test_midpoint(self):
        assert linear_interpolate([0, 1], [10, 20], 0.5) == pytest.approx(15)
