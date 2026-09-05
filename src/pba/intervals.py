"""Interval arithmetic primitives.

An *interval* is a two-element sequence ``[a, b]`` with ``a <= b`` representing
the closed interval of the real line. Scalars (``int`` or ``float``) are
promoted to degenerate intervals ``(x, x)``. All operations return a
two-element ``list`` ``[lower, upper]``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Union

import numpy as np
import scipy.interpolate

Interval = Union[Sequence[float], "np.ndarray"]
IntervalLike = Union[int, float, Interval]


def _as_interval(x: IntervalLike) -> tuple[float, float]:
    """Promote a scalar to a degenerate interval; otherwise return ``(lo, hi)``."""
    if isinstance(x, (int, float)):
        return (x, x)
    return (x[0], x[1])


def addIntervals(interval_a: IntervalLike, interval_b: IntervalLike) -> list[float]:
    """Return ``interval_a + interval_b`` under interval arithmetic."""
    a_lo, a_hi = _as_interval(interval_a)
    b_lo, b_hi = _as_interval(interval_b)
    return [a_lo + b_lo, a_hi + b_hi]


def subtractIntervals(interval_a: IntervalLike, interval_b: IntervalLike) -> list[float]:
    """Return ``interval_a - interval_b`` under interval arithmetic."""
    a_lo, a_hi = _as_interval(interval_a)
    b_lo, b_hi = _as_interval(interval_b)
    return [a_lo - b_hi, a_hi - b_lo]


def multiplyIntervals(interval_a: IntervalLike, interval_b: IntervalLike) -> list[float]:
    """Return ``interval_a * interval_b`` under interval arithmetic.

    Handles sign changes correctly by taking the min/max over all four
    endpoint products.
    """
    a_lo, a_hi = _as_interval(interval_a)
    b_lo, b_hi = _as_interval(interval_b)
    products = (a_lo * b_lo, a_lo * b_hi, a_hi * b_lo, a_hi * b_hi)
    return [float(np.min(products)), float(np.max(products))]


def divideIntervals(interval_a: IntervalLike, interval_b: IntervalLike) -> list[float]:
    """Return ``interval_a / interval_b`` under interval arithmetic.

    If the denominator interval touches zero, the result is unbounded and this
    function returns the multiplication of ``interval_a`` by the extended
    interval ``(-inf, +inf)``.
    """
    a = _as_interval(interval_a)
    b_lo, b_hi = _as_interval(interval_b)
    if b_lo == 0 or b_hi == 0:
        reciprocal: list[float] = [-float("inf"), float("inf")]
    else:
        reciprocal = [1 / b_hi, 1 / b_lo]
    return multiplyIntervals(a, reciprocal)


def linear_interpolate(x_values: Sequence[float], y_values: Sequence[float], x: float) -> float:
    """Linear interpolation of ``y`` at ``x`` from tabulated ``(x_values, y_values)``."""
    y_interp = scipy.interpolate.interp1d(x_values, y_values)
    return y_interp(x).tolist()
