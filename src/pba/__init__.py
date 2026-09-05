"""Probability Bound Analysis (PBA): confidence boxes and interval arithmetic.

A confidence box (c-box) is an imprecise generalization of a confidence
distribution. It encodes frequentist confidence intervals at every confidence
level for a parameter of interest, providing rigorous statistical guarantees
even from sparse or imprecise sample data.

This package exposes:

- :func:`cBox`, :func:`cBox2` -- construct a c-box from binomial data.
- :func:`plotcBox` -- plot a c-box.
- :func:`computeFocalElements`, :func:`computeConfidenceInterval`.
- Interval arithmetic: :func:`addIntervals`, :func:`subtractIntervals`,
  :func:`multiplyIntervals`, :func:`divideIntervals`.
- C-box arithmetic: :func:`addCbox`, :func:`subtractCbox`,
  :func:`multiplyCbox`, :func:`divideCbox` and their scalar variants.
"""

from pba._version import __version__
from pba.arithmetic import (
    addCbox,
    addCboxNnumber,
    cBoxcBox,
    cBoxcBox2,
    cBoxcBox3,
    cBoxcBox4,
    computeFocalElements,
    divideCbox,
    divideCboxNnumber,
    divideNumberNcbox,
    multiplyCbox,
    multiplyCboxNnumber,
    subtractCbox,
    subtractCboxNnumber,
    subtractNumberNcbox,
)
from pba.cbox import cBox, cBox2, computeConfidenceInterval, f, g, phi
from pba.intervals import (
    addIntervals,
    divideIntervals,
    linear_interpolate,
    multiplyIntervals,
    subtractIntervals,
)
from pba.plotting import plotcBox

__all__ = [
    "__version__",
    "addCbox",
    "addCboxNnumber",
    "addIntervals",
    "cBox",
    "cBox2",
    "cBoxcBox",
    "cBoxcBox2",
    "cBoxcBox3",
    "cBoxcBox4",
    "computeConfidenceInterval",
    "computeFocalElements",
    "divideCbox",
    "divideCboxNnumber",
    "divideIntervals",
    "divideNumberNcbox",
    "f",
    "g",
    "linear_interpolate",
    "multiplyCbox",
    "multiplyCboxNnumber",
    "multiplyIntervals",
    "phi",
    "plotcBox",
    "subtractCbox",
    "subtractCboxNnumber",
    "subtractIntervals",
    "subtractNumberNcbox",
]
