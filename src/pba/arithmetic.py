"""Confidence box arithmetic and focal-element decomposition.

C-box arithmetic follows the standard imprecise-probability recipe:

1. Discretize each operand c-box into ``npoints`` focal elements
   (:math:`\\alpha`-cut intervals of the upper and lower CDFs).
2. Compute the Cartesian product of focal elements and apply the interval
   operation pointwise.
3. Re-aggregate the sorted results into the lower and upper CDFs of a new
   c-box.

The result is a *free* combination -- it makes no independence assumptions
about the operands and is guaranteed to bracket the true CDF.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy.stats import beta as _beta

from pba.cbox import Cbox, cBox, cBox2
from pba.intervals import (
    IntervalLike,
    _as_interval,
    addIntervals,
    divideIntervals,
    linear_interpolate,
    multiplyIntervals,
    subtractIntervals,
)

_TOL = 1e-6

BinaryIntervalOp = Callable[[Any, Any], list[float]]


def _cdf_grid(npoints: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``(Fx_left, Fx_right)`` grid used for focal-element sampling.

    ``Fx_left`` runs from ``0`` to ``(npoints - 1) / npoints``;
    ``Fx_right`` runs from ``1 / npoints`` to ``1``. Together they define the
    :math:`\\alpha`-levels at which the lower and upper CDFs are inverted.
    """
    Fx_left = np.linspace(0, (npoints - 1) / npoints, npoints)
    Fx_right = np.linspace(1, npoints, npoints) / npoints
    return Fx_left, Fx_right


def _focal_elements_from_grid(
    cbox: Cbox, Fx_left: np.ndarray, Fx_right: np.ndarray, npoints: int
) -> list[list[float]]:
    """Compute ``npoints`` focal elements of ``cbox``.

    If ``cbox`` was constructed analytically from Beta parameters, uses the
    inverse Beta CDF; otherwise interpolates from the (already-sampled) lower
    and upper CDFs.
    """
    betaparam = cbox.get("betaparam")
    flag = cbox.get("flag")

    if betaparam != "unknown" and flag is False:
        p = betaparam
        left = _beta.ppf(Fx_left, p["param_alpha_left"], p["param_beta_left"])
        right = _beta.ppf(Fx_right, p["param_alpha_right"], p["param_beta_right"])
        return [[float(lf), float(rf)] for lf, rf in zip(left.tolist(), right.tolist())]

    if betaparam != "unknown" and flag is True:
        p = betaparam
        left = _beta.ppf(Fx_left, p["param_alpha_left"], p["param_beta_left"])
        right = np.repeat(1 - _TOL, npoints)
        return [[float(lf), float(rf)] for lf, rf in zip(left.tolist(), right.tolist())]

    # Non-analytic c-box: interpolate.
    support = cbox["support"]
    lb = cbox["lb"].tolist()
    ub = cbox["ub"].tolist()

    if isinstance(support, list):
        x_lo = support[0].tolist()
        x_hi = support[1].tolist()
    else:
        x_lo = x_hi = support.tolist()

    left_fe = [linear_interpolate(x_lo, lb, Fx_left[i]) for i in range(npoints)]
    right_fe = [linear_interpolate(x_hi, ub, Fx_right[i]) for i in range(npoints)]
    return [[lf, rf] for lf, rf in zip(left_fe, right_fe)]


def computeFocalElements(
    Cbox: Cbox, npoints: int = 100, show_plot: bool = False
) -> list[list[float]]:
    """Return ``npoints`` focal elements (:math:`\\alpha`-cuts) of ``Cbox``.

    Each focal element is a 2-list ``[left, right]``. When ``show_plot`` is
    ``True``, the focal elements are overlaid on the c-box plot.
    """
    Fx_left, Fx_right = _cdf_grid(npoints)
    focal_elements = _focal_elements_from_grid(Cbox, Fx_left, Fx_right, npoints)

    if show_plot:
        _plot_focal_elements(Cbox, focal_elements, Fx_left, Fx_right)

    return focal_elements


def _plot_focal_elements(
    Cbox: Cbox,
    focal_elements: list[list[float]],
    Fx_left: np.ndarray,
    Fx_right: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    left = [fe[0] for fe in focal_elements]
    right = [fe[1] for fe in focal_elements]

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.scatter(left, Fx_left)
    plt.scatter(right, Fx_right)
    plt.step(left, Fx_left, where="pre")
    plt.step(right, Fx_right, where="post")

    support = Cbox.get("support")
    lb = Cbox.get("lb")
    ub = Cbox.get("ub")
    if isinstance(support, list):
        plt.step(lb, support[0], where="pre")
        plt.step(ub, support[1], where="post")
    else:
        plt.plot(support, lb)
        plt.plot(support, ub)


def _combine_focal_elements(
    fe_a: list[list[float]],
    fe_b: list[list[float]],
    op: BinaryIntervalOp,
    npoints: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply ``op`` to every pair ``(fe_a[j], fe_b[k])`` and re-aggregate.

    The Cartesian product of ``npoints^2`` interval results is sorted
    column-wise. The output c-box lower CDF is formed by taking one value
    from every group of ``npoints`` sorted lower endpoints; the upper CDF
    from the largest value in each group of upper endpoints. This matches
    the original reference implementation.
    """
    out = np.zeros((npoints**2, 2))
    for j in range(npoints):
        for k in range(npoints):
            out[j + k * npoints] = op(fe_a[j], fe_b[k])

    sort_out = np.sort(out, 0)
    lower_sort = sort_out[:, 0]
    upper_sort = sort_out[:, 1]

    lower_bound = np.zeros(npoints)
    upper_bound = np.zeros(npoints)
    for l in range(npoints):
        lower_bound[l] = lower_sort[l * npoints]
        upper_bound[l] = upper_sort[l * npoints + npoints - 1]
    return lower_bound, upper_bound


def _plot_cdf_bounds(
    lower_bound: np.ndarray, upper_bound: np.ndarray, Fx_left: np.ndarray, Fx_right: np.ndarray
) -> None:
    import matplotlib.pyplot as plt

    plt.step(lower_bound, Fx_left, where="pre")
    plt.step(upper_bound, Fx_right, where="post")


def _cbox_binary_op(
    Cbox1: Cbox,
    Cbox2: Cbox,
    op: BinaryIntervalOp,
    npoints: int,
    show_plot: bool,
) -> Cbox:
    Fx_left, Fx_right = _cdf_grid(npoints)
    fe_1 = _focal_elements_from_grid(Cbox1, Fx_left, Fx_right, npoints)
    fe_2 = _focal_elements_from_grid(Cbox2, Fx_left, Fx_right, npoints)

    lower_bound, upper_bound = _combine_focal_elements(fe_1, fe_2, op, npoints)

    if show_plot:
        _plot_cdf_bounds(lower_bound, upper_bound, Fx_left, Fx_right)

    return {
        "support": [Fx_left, Fx_right],
        "lb": lower_bound,
        "ub": upper_bound,
        "betaparam": "unknown",
        "flag": "unknown",
    }


def _cbox_number_op(
    Cbox: Cbox,
    num: IntervalLike,
    op: BinaryIntervalOp,
    npoints: int,
    show_plot: bool,
    *,
    number_first: bool = False,
) -> Cbox:
    Fx_left, Fx_right = _cdf_grid(npoints)
    focal_elements = _focal_elements_from_grid(Cbox, Fx_left, Fx_right, npoints)
    number_interval = list(_as_interval(num))
    focal_element_num = [number_interval] * npoints

    if number_first:
        combined_op: BinaryIntervalOp = lambda a, b: op(b, a)  # noqa: E731
    else:
        combined_op = op

    lower_bound, upper_bound = _combine_focal_elements(
        focal_elements, focal_element_num, combined_op, npoints
    )

    if show_plot:
        _plot_cdf_bounds(lower_bound, upper_bound, Fx_left, Fx_right)

    return {
        "support": [Fx_left, Fx_right],
        "lb": lower_bound,
        "ub": upper_bound,
        "betaparam": "unknown",
        "flag": "unknown",
    }


# ---------------------------------------------------------------------------
# Public c-box arithmetic
# ---------------------------------------------------------------------------


def addCbox(Cbox1: Cbox, Cbox2: Cbox, npoints: int = 100, show_plot: bool = False) -> Cbox:
    """Return ``Cbox1 + Cbox2`` (free combination)."""
    return _cbox_binary_op(Cbox1, Cbox2, addIntervals, npoints, show_plot)


def subtractCbox(Cbox1: Cbox, Cbox2: Cbox, npoints: int = 100, show_plot: bool = False) -> Cbox:
    """Return ``Cbox1 - Cbox2`` (free combination)."""
    return _cbox_binary_op(Cbox1, Cbox2, subtractIntervals, npoints, show_plot)


def multiplyCbox(Cbox1: Cbox, Cbox2: Cbox, npoints: int = 100, show_plot: bool = False) -> Cbox:
    """Return ``Cbox1 * Cbox2`` (free combination)."""
    return _cbox_binary_op(Cbox1, Cbox2, multiplyIntervals, npoints, show_plot)


def divideCbox(Cbox1: Cbox, Cbox2: Cbox, npoints: int = 100, show_plot: bool = False) -> Cbox:
    """Return ``Cbox1 / Cbox2`` (free combination)."""
    return _cbox_binary_op(Cbox1, Cbox2, divideIntervals, npoints, show_plot)


# ---------------------------------------------------------------------------
# C-box with scalar / interval number
# ---------------------------------------------------------------------------


def addCboxNnumber(
    Cbox: Cbox, num: IntervalLike, npoints: int = 100, show_plot: bool = False
) -> Cbox:
    """Return ``Cbox + num`` where ``num`` is a scalar or 2-interval."""
    return _cbox_number_op(Cbox, num, addIntervals, npoints, show_plot)


def subtractCboxNnumber(
    Cbox: Cbox, num: IntervalLike, npoints: int = 100, show_plot: bool = False
) -> Cbox:
    """Return ``Cbox - num``."""
    return _cbox_number_op(Cbox, num, subtractIntervals, npoints, show_plot)


def multiplyCboxNnumber(
    Cbox: Cbox, num: IntervalLike, npoints: int = 100, show_plot: bool = False
) -> Cbox:
    """Return ``Cbox * num``."""
    return _cbox_number_op(Cbox, num, multiplyIntervals, npoints, show_plot)


def divideCboxNnumber(
    Cbox: Cbox, num: IntervalLike, npoints: int = 100, show_plot: bool = False
) -> Cbox:
    """Return ``Cbox / num``."""
    return _cbox_number_op(Cbox, num, divideIntervals, npoints, show_plot)


def subtractNumberNcbox(
    num: IntervalLike, Cbox: Cbox, npoints: int = 100, show_plot: bool = False
) -> Cbox:
    """Return ``num - Cbox``."""
    return _cbox_number_op(Cbox, num, subtractIntervals, npoints, show_plot, number_first=True)


def divideNumberNcbox(
    num: IntervalLike, Cbox: Cbox, npoints: int = 100, show_plot: bool = False
) -> Cbox:
    """Return ``num / Cbox``."""
    return _cbox_number_op(Cbox, num, divideIntervals, npoints, show_plot, number_first=True)


# ---------------------------------------------------------------------------
# Cbox-of-Cbox: averaging c-boxes constructed from focal elements of
# imprecise ``k`` and ``n``.
# ---------------------------------------------------------------------------


def _cboxcbox_average(
    kCbox: Cbox,
    nCbox: Cbox,
    npoints: int,
    *,
    cross_product: bool,
    builder: Callable[[list[float], list[float]], Cbox],
) -> Cbox:
    focal_elements_k = computeFocalElements(kCbox, npoints=npoints)
    focal_elements_n = computeFocalElements(nCbox, npoints=npoints)

    lower_values: list[np.ndarray] = []
    upper_values: list[np.ndarray] = []
    denominator = 0

    if cross_product:
        for i in range(npoints):
            for k in range(npoints):
                cb = builder(focal_elements_k[i], focal_elements_n[k])
                lower_values.append(cb["lb"])
                upper_values.append(cb["ub"])
        denominator = npoints * npoints
    else:
        for i in range(npoints):
            cb = builder(focal_elements_k[i], focal_elements_n[i])
            lower_values.append(cb["lb"])
            upper_values.append(cb["ub"])
        denominator = npoints

    average_lower = np.sum(lower_values, axis=0) / denominator
    average_upper = np.sum(upper_values, axis=0) / denominator

    # All builders return the same support grid, so cell 0 is representative.
    representative_support = builder(focal_elements_k[0], focal_elements_n[0])["support"]

    return {
        "support": representative_support,
        "lb": average_lower,
        "ub": average_upper,
        "betaparam": "unknown",
        "flag": "unknown",
    }


def cBoxcBox(kCbox: Cbox, nCbox: Cbox, npoints: int = 100) -> Cbox:
    """Averaged c-box over paired focal elements of ``kCbox`` and ``nCbox``."""
    return _cboxcbox_average(kCbox, nCbox, npoints, cross_product=False, builder=cBox)


def cBoxcBox2(kCbox: Cbox, nCbox: Cbox, npoints: int = 100) -> Cbox:
    """Averaged c-box over paired focal elements using :func:`cBox2`."""
    return _cboxcbox_average(kCbox, nCbox, npoints, cross_product=False, builder=cBox2)


def cBoxcBox3(kCbox: Cbox, nCbox: Cbox, npoints: int = 100) -> Cbox:
    """Averaged c-box over the Cartesian product of focal elements."""
    return _cboxcbox_average(kCbox, nCbox, npoints, cross_product=True, builder=cBox)


def cBoxcBox4(kCbox: Cbox, nCbox: Cbox, npoints: int = 100) -> Cbox:
    """Cartesian variant using :func:`cBox2`."""
    return _cboxcbox_average(kCbox, nCbox, npoints, cross_product=True, builder=cBox2)
