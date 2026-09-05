"""Confidence box construction from binomial data.

A *confidence box* (c-box) for a Bernoulli success probability observed as
``k`` successes in ``n`` trials is the pair of Beta CDFs

.. math::

    F_L(p) = \\mathrm{Beta}(k,\\; n - k + 1).\\mathrm{cdf}(p)
    F_R(p) = \\mathrm{Beta}(k + 1,\\; n - k).\\mathrm{cdf}(p)

These are the pointwise Clopper--Pearson lower and upper confidence limits at
every confidence level (see Ferson et al., 2003; Balch, 2012).

When ``k`` and/or ``n`` themselves are intervals, the returned c-box brackets
all possible c-boxes consistent with the imprecise observation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import beta as _beta

from pba.intervals import IntervalLike, _as_interval, linear_interpolate, multiplyIntervals

_TOL = 1e-6

Cbox = dict[str, Any]


def _normalize_k(k: IntervalLike) -> tuple[float, float]:
    """Promote ``k`` to an interval and replace exact-zero endpoints with ``TOL``.

    ``Beta(0, ...)`` is undefined; ``TOL`` is used as a stand-in whenever a
    shape parameter would otherwise be zero.
    """
    if isinstance(k, (int, float)):
        return (_TOL, _TOL) if k == 0 else (float(k), float(k))
    lo = _TOL if k[0] == 0 else float(k[0])
    hi = _TOL if k[1] == 0 else float(k[1])
    return (lo, hi)


def _normalize_n(n: IntervalLike) -> tuple[float, float]:
    """Promote ``n`` to an interval."""
    return _as_interval(n)


def cBox(k: IntervalLike, n: IntervalLike, nsteps: int = 1000) -> Cbox:
    """Construct a c-box from binomial data ``k`` successes in ``n`` trials.

    Parameters
    ----------
    k : int, float, or 2-sequence
        Number of successes. May be an interval ``[k_lo, k_hi]`` to represent
        uncertainty about ``k``.
    n : int, float, or 2-sequence
        Number of trials. May also be an interval.
    nsteps : int, optional
        Number of grid points on the probability axis ``[0, 1]`` for the
        returned lower and upper CDFs. Defaults to 1000.

    Returns
    -------
    dict
        Dictionary with keys ``support``, ``lb``, ``ub``, ``betaparam``,
        ``flag``. ``flag`` is ``True`` when the c-box degenerates on the
        right edge (``k[1] >= n[0]``), else ``False``.

    Raises
    ------
    ValueError
        If either interval is inverted (``lo > hi``).
    """
    k_lo, k_hi = _normalize_k(k)
    n_lo, n_hi = _normalize_n(n)

    if k_lo > k_hi or n_lo > n_hi:
        raise ValueError("Lower bound cannot be greater than upper bound.")

    if k_hi >= n_lo:
        dist_left = _beta(k_lo, n_hi - k_lo + 1)
        dist_right = _beta(k_hi + 1, _TOL)
        param_all = {
            "param_alpha_left": k_lo,
            "param_beta_left": n_hi - k_lo + 1,
            "param_alpha_right": k_hi + 1,
            "param_beta_right": _TOL,
        }
    else:
        dist_left = _beta(k_lo, n_hi - k_lo + 1)
        dist_right = _beta(k_hi + 1, n_lo - k_hi)
        param_all = {
            "param_alpha_left": k_lo,
            "param_beta_left": n_hi - k_lo + 1,
            "param_alpha_right": k_hi + 1,
            "param_beta_right": n_lo - k_hi,
        }

    Fx = np.linspace(0, 1, nsteps)
    lower_bound = dist_left.cdf(Fx)

    degenerate = (k_lo, k_hi) == (n_lo, n_hi)
    upper_bound = np.repeat(1 - _TOL, nsteps) if degenerate else dist_right.cdf(Fx)

    return {
        "support": Fx,
        "lb": lower_bound,
        "ub": upper_bound,
        "betaparam": param_all,
        "flag": degenerate,
    }


def cBox2(k: IntervalLike, m: IntervalLike, nsteps: int = 1000) -> Cbox:
    """Convenience form: c-box with ``n = k + m`` (successes and failures)."""
    from pba.intervals import addIntervals

    return cBox(k, addIntervals(k, m), nsteps=nsteps)


def phi(a: float, b: float) -> float:
    """Return ``a / (a + b)`` -- a normalized ratio used in Bayesian test analyses."""
    return a / (a + b)


def g(a: float, b: float, phi: float) -> float:
    """Bayesian test statistic used with :func:`f`."""
    return (phi * a + (1 - phi) * b) / (2 * phi * (1 - phi) * (a + b)) - 1


def f(a: float, b: float, phi: float) -> Cbox:
    """Compose a scaled c-box from Bayesian test parameters."""
    from pba.arithmetic import multiplyCboxNnumber

    return multiplyCboxNnumber(cBox(g(a, b, phi), g(a, b, phi) + g(b, a, phi)), a + b)


def computeConfidenceInterval(
    Cbox: Cbox,
    alpha_level: float = 0.05,
    beta_level: float = 0.95,
    show_plot: bool = False,
) -> list[float]:
    """Return the two-sided confidence interval ``[l, r]`` at the given levels.

    ``l`` is the ``alpha_level`` quantile of the *lower* CDF and ``r`` is the
    ``beta_level`` quantile of the *upper* CDF. Set ``show_plot=True`` to
    overlay the interval on the c-box plot.
    """
    betaparam = Cbox.get("betaparam")
    flag = Cbox.get("flag")

    if betaparam != "unknown" and flag is False:
        parameters = betaparam
        left_interval = _beta.ppf(
            alpha_level, parameters["param_alpha_left"], parameters["param_beta_left"]
        )
        right_interval = _beta.ppf(
            beta_level, parameters["param_alpha_right"], parameters["param_beta_right"]
        )
    elif betaparam != "unknown" and flag is True:
        parameters = betaparam
        left_interval = _beta.ppf(
            alpha_level, parameters["param_alpha_left"], parameters["param_beta_left"]
        )
        right_interval = 1 - _TOL
    elif betaparam == "unknown" and flag == "unknown":
        support = Cbox.get("support")
        if not isinstance(support, list):
            left_interval = linear_interpolate(support.tolist(), Cbox["lb"].tolist(), alpha_level)
            right_interval = linear_interpolate(support.tolist(), Cbox["ub"].tolist(), beta_level)
        else:
            left_interval = linear_interpolate(
                support[0].tolist(), Cbox["lb"].tolist(), alpha_level
            )
            right_interval = linear_interpolate(
                support[1].tolist(), Cbox["ub"].tolist(), beta_level
            )
    else:  # pragma: no cover -- unreachable given cBox invariants
        raise ValueError(f"Unexpected Cbox state: betaparam={betaparam!r}, flag={flag!r}")

    if show_plot:
        _plot_confidence_interval(Cbox, alpha_level, beta_level, left_interval, right_interval)

    return [left_interval, right_interval]


def _plot_confidence_interval(
    Cbox: Cbox,
    alpha_level: float,
    beta_level: float,
    left_interval: float,
    right_interval: float,
) -> None:
    """Overlay a horizontal/vertical guide on the c-box plot."""
    import matplotlib.pyplot as plt

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.plot((0, left_interval), (alpha_level, alpha_level), "r--")
    plt.plot((left_interval, left_interval), (0, alpha_level), "r--")
    plt.plot((0, right_interval), (beta_level, beta_level), "b--")
    plt.plot((right_interval, right_interval), (0, beta_level), "b--")

    x = Cbox.get("support")
    lb = Cbox.get("lb")
    ub = Cbox.get("ub")
    if isinstance(x, list):
        plt.step(lb, x[0], where="pre")
        plt.step(ub, x[1], where="post")
    else:
        plt.plot(x, lb)
        plt.plot(x, ub)


# Re-export a private helper for `arithmetic` -- avoids a heavy import cycle.
_multiplyIntervals = multiplyIntervals
