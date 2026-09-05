# Probability Bound Analysis (PBA)

[![CI](https://github.com/uoparaji/Probability-Bound-Analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/uoparaji/Probability-Bound-Analysis/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-cf9?logo=ruff)](https://github.com/astral-sh/ruff)

A Python library for constructing and computing with **confidence boxes (c-boxes)** — an
imprecise generalization of confidence distributions that provide rigorous frequentist
uncertainty quantification even from sparse or interval-valued data.

C-boxes let you carry statistical uncertainty about a parameter through arithmetic
operations without collapsing to a point estimate, and without pretending you know a
prior. They are used in reliability engineering, risk analysis, medical diagnostics, and
any setting where you must reason about "what confidence intervals *could* look like"
given imprecise observations.

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [What is a c-box?](#what-is-a-c-box)
- [API overview](#api-overview)
- [Examples](#examples)
- [Development](#development)
- [References](#references)
- [License](#license)

## Installation

```bash
pip install -e .
# or, with dev tools:
pip install -e ".[dev]"
# or, with notebook support:
pip install -e ".[notebooks]"
```

Requirements: Python 3.9+, `numpy`, `scipy`, `matplotlib`.

## Quick start

```python
import pba

# Build a c-box from k = 2 successes in n = 10 trials.
cb = pba.cBox(2, 10)

# Two-sided 90% confidence interval.
lo, hi = pba.computeConfidenceInterval(cb, alpha_level=0.05, beta_level=0.95)
print(f"90% CI: [{lo:.3f}, {hi:.3f}]")

# Combine c-boxes arithmetically.
cb2 = pba.cBox(1, 3)
cb_sum = pba.addCbox(cb, cb2)
cb_scaled = pba.multiplyCboxNnumber(cb_sum, [1, 2])  # multiply by an interval

# Plot.
pba.plotcBox(cb_sum)
```

## What is a c-box?

Given `k` successes in `n` binomial trials, the Clopper–Pearson confidence limits at
significance level `α` are given by two Beta quantiles. A c-box packages these into a
pair of CDFs:

```
F_L(p) = Beta(k,     n − k + 1).cdf(p)   ← lower confidence CDF
F_R(p) = Beta(k + 1, n − k    ).cdf(p)   ← upper confidence CDF
```

The area between `F_L` and `F_R` is the *confidence band*: for any two levels
`α < β`, the interval `[F_L⁻¹(α), F_R⁻¹(β)]` is a valid confidence interval for the
success probability at coverage `β − α`.

When `k` or `n` is itself interval-valued (e.g. "between 2 and 4 successes out of
around 10 trials"), this library returns a c-box that brackets every c-box consistent
with the imprecise observation. Arithmetic on c-boxes uses the standard focal-element
recipe (α-cut discretization → interval Cartesian product → re-aggregation) and is
guaranteed to bracket the true CDF without assuming independence.

## API overview

**Construction**

| Function                          | Purpose                                            |
| --------------------------------- | -------------------------------------------------- |
| `cBox(k, n, nsteps=1000)`         | C-box from `k` successes in `n` trials             |
| `cBox2(k, m, nsteps=1000)`        | Convenience form with `n = k + m`                  |

**Analysis**

| Function                                                            | Purpose                              |
| ------------------------------------------------------------------- | ------------------------------------ |
| `plotcBox(cb)`                                                      | Plot lower and upper CDFs            |
| `computeFocalElements(cb, npoints=100, show_plot=False)`            | Sample α-cut intervals               |
| `computeConfidenceInterval(cb, alpha_level=0.05, beta_level=0.95)`  | Two-sided confidence interval        |

**Interval arithmetic**

`addIntervals`, `subtractIntervals`, `multiplyIntervals`, `divideIntervals`,
`linear_interpolate`.

**C-box arithmetic**

`addCbox`, `subtractCbox`, `multiplyCbox`, `divideCbox`,
`addCboxNnumber`, `subtractCboxNnumber`, `multiplyCboxNnumber`, `divideCboxNnumber`,
`subtractNumberNcbox`, `divideNumberNcbox`,
`cBoxcBox`, `cBoxcBox2`, `cBoxcBox3`, `cBoxcBox4`.

## Examples

See the `examples/` directory:

- [`tutorial_cbox.ipynb`](examples/tutorial_cbox.ipynb) — construction, plotting, focal
  elements, confidence intervals, and arithmetic.
- [`medical_test_analysis.ipynb`](examples/medical_test_analysis.ipynb) — computing
  positive/negative predictive value, sensitivity, specificity, and accuracy from
  imprecise diagnostic-test data.

## Development

```bash
# Editable install with dev tools:
pip install -e ".[dev]"

# Run tests with coverage:
pytest --cov=pba --cov-report=term-missing

# Lint and format:
ruff check .
ruff format .
```

CI runs lint + tests on Python 3.9–3.12 across Linux and macOS.

### Project layout

```
src/pba/
  __init__.py       ← public API
  cbox.py           ← c-box construction and confidence intervals
  intervals.py      ← interval arithmetic primitives
  arithmetic.py     ← c-box arithmetic (focal-element decomposition)
  plotting.py       ← matplotlib helpers
tests/              ← pytest suite
examples/           ← Jupyter notebooks
```

## References

- Ferson, S., Kreinovich, V., Ginzburg, L., Myers, D., & Sentz, K. (2003).
  *Constructing Probability Boxes and Dempster–Shafer Structures.*
  Sandia National Laboratories Report SAND2002-4015.
- Balch, M. S. (2012). *Mathematical foundations for a theory of confidence structures.*
  International Journal of Approximate Reasoning, 53(7), 1003–1019.

## License

MIT — see [LICENSE](LICENSE).
