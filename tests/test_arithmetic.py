"""Smoke tests for c-box arithmetic."""

from __future__ import annotations

import pytest

from pba import (
    addCbox,
    addCboxNnumber,
    cBox,
    cBoxcBox,
    divideCbox,
    divideCboxNnumber,
    divideNumberNcbox,
    multiplyCbox,
    multiplyCboxNnumber,
    subtractCbox,
    subtractCboxNnumber,
    subtractNumberNcbox,
)

NPOINTS = 25  # keep tests fast


def _assert_valid_cbox(cb, npoints=NPOINTS):
    assert cb["betaparam"] == "unknown"
    assert cb["flag"] == "unknown"
    assert cb["lb"].shape == (npoints,)
    assert cb["ub"].shape == (npoints,)


@pytest.fixture
def cb_a():
    return cBox(2, 10)


@pytest.fixture
def cb_b():
    return cBox(1, 3)


class TestCboxBinary:
    def test_add(self, cb_a, cb_b):
        _assert_valid_cbox(addCbox(cb_a, cb_b, npoints=NPOINTS))

    def test_subtract(self, cb_a, cb_b):
        _assert_valid_cbox(subtractCbox(cb_a, cb_b, npoints=NPOINTS))

    def test_multiply(self, cb_a, cb_b):
        _assert_valid_cbox(multiplyCbox(cb_a, cb_b, npoints=NPOINTS))

    def test_divide(self, cb_a, cb_b):
        _assert_valid_cbox(divideCbox(cb_a, cb_b, npoints=NPOINTS))

    def test_chained_ops_still_valid_cbox(self, cb_a, cb_b):
        # This is the pattern from the tutorial notebook: cb3 = cb + cb2, then cb4 = cb3 - cb.
        cb_sum = addCbox(cb_a, cb_b, npoints=NPOINTS)
        cb_diff = subtractCbox(cb_sum, cb_a, npoints=NPOINTS)
        _assert_valid_cbox(cb_diff)


class TestCboxScalar:
    def test_add_int(self, cb_a):
        _assert_valid_cbox(addCboxNnumber(cb_a, 1, npoints=NPOINTS))

    def test_add_interval(self, cb_a):
        _assert_valid_cbox(addCboxNnumber(cb_a, [1, 2], npoints=NPOINTS))

    def test_subtract_int(self, cb_a):
        _assert_valid_cbox(subtractCboxNnumber(cb_a, 1, npoints=NPOINTS))

    def test_multiply_int(self, cb_a):
        _assert_valid_cbox(multiplyCboxNnumber(cb_a, 2, npoints=NPOINTS))

    def test_divide_int(self, cb_a):
        _assert_valid_cbox(divideCboxNnumber(cb_a, 2, npoints=NPOINTS))

    def test_number_minus_cbox(self, cb_a):
        _assert_valid_cbox(subtractNumberNcbox(1, cb_a, npoints=NPOINTS))

    def test_number_divide_cbox(self, cb_a):
        _assert_valid_cbox(divideNumberNcbox(2, cb_a, npoints=NPOINTS))


class TestCboxOfCbox:
    def test_cBoxcBox_paired(self, cb_a, cb_b):
        result = cBoxcBox(cb_a, cb_b, npoints=NPOINTS)
        _assert_valid_cbox(result, npoints=1000)
