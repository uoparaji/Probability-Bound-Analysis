"""Plot helpers for c-boxes."""

from __future__ import annotations

from typing import Any

_TOL = 1e-6

Cbox = dict[str, Any]


def plotcBox(Cbox: Cbox) -> None:
    """Plot the lower and upper CDFs of a c-box on ``[0, 1] x [0, 1]``.

    Handles three cases:

    - Degenerate right edge (``flag=True``): draws the lower CDF and a
      vertical line at ``1 - TOL``.
    - Analytic c-box (``flag=False``): plots both CDFs.
    - Non-analytic c-box (``flag='unknown'``): plots step CDFs from the
      sampled support pair.
    """
    import matplotlib.pyplot as plt

    flag = Cbox.get("flag")
    support = Cbox.get("support")
    lb = Cbox.get("lb")
    ub = Cbox.get("ub")

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    if flag is True:
        if isinstance(support, list):
            plt.step(lb, support[0], where="pre")
            plt.step(ub, support[1], where="post")
        else:
            plt.plot(support, lb, "r")
            plt.axvline(x=1 - _TOL, color="b")
    elif flag is False:
        if isinstance(support, list):
            plt.step(lb, support[0], where="pre")
            plt.step(ub, support[1], where="post")
        else:
            plt.plot(support, lb, "r")
            plt.plot(support, ub, "b")
    elif flag == "unknown":
        # Unknown-flag c-boxes come from arithmetic and use axis-free bounds.
        plt.autoscale()
        if isinstance(support, list):
            plt.step(lb, support[0], where="pre")
            plt.step(ub, support[1], where="post")
        else:
            plt.plot(support, lb, "r")
            plt.plot(support, ub, "b")
    else:  # pragma: no cover
        raise ValueError(f"Unexpected c-box flag: {flag!r}")
