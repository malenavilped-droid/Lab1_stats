"""Exact and log-space multiplicities Omega(N, k) of N coins (questions 1c, 1d). Owner: Pablo.

Omega(N, k) = N! / (k! (N - k)!) is the number of microstates with k heads. It is
computed in three ways:

* Exact: Python integers and the multiplicative formula. Python ints have no
  maximum size, so the result is exact for any N; it only gets slower as N grows.
* Log-space: ln Omega from the log-gamma function, using ln Gamma(n + 1) = ln n!.
  The values stay small (ln Omega <= N ln 2), so nothing overflows for any N.
* Naive float: the three factorials in float64. Only used to show where this
  approach breaks (question 1d).
"""

from __future__ import annotations

import math
import sys

import numpy as np
from scipy.special import gammaln

LN2 = math.log(2.0)


def binomial_exact(n: int, k: int) -> int:
    """Exact Omega(n, k) with the multiplicative formula.

    Omega(n, i) = Omega(n, i - 1) * (n + 1 - i) / i, starting from Omega(n, 0) = 1.
    Every step gives Omega(n, i), which is an integer, so the integer division
    is exact. Uses the symmetry Omega(n, k) = Omega(n, n - k) to do at most n/2 steps.

    Args:
        n: Number of coins (n >= 0).
        k: Number of heads. Values outside [0, n] give 0 (no such microstate).

    Returns:
        The exact number of microstates with k heads.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    omega = 1
    for i in range(1, k + 1):
        omega = omega * (n + 1 - i) // i
    return omega


def multiplicities_exact(n: int) -> list[int]:
    """Exact Omega(n, k) for every k = 0, ..., n (one row of Pascal's triangle).

    Same recurrence as :func:`binomial_exact`, but keeping every step, so the
    whole row costs n multiplications instead of n separate binomials.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    row = [1]
    for k in range(1, n + 1):
        row.append(row[-1] * (n + 1 - k) // k)
    return row


def log_binomial(n: int, k: int) -> float:
    """ln Omega(n, k) = ln Gamma(n + 1) - ln Gamma(k + 1) - ln Gamma(n - k + 1).

    Returns -inf for k outside [0, n], since Omega = 0 there.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if k < 0 or k > n:
        return -math.inf
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def log_multiplicities(n: int) -> np.ndarray:
    """ln Omega(n, k) for every k = 0, ..., n at once, with scipy's gammaln.

    Returns:
        A float64 array of length n + 1.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    k = np.arange(n + 1, dtype=np.float64)
    return gammaln(n + 1.0) - gammaln(k + 1.0) - gammaln(n - k + 1.0)


def probabilities(n: int) -> np.ndarray:
    """P_k = Omega(n, k) / 2^n for every k, computed as exp(ln Omega - n ln 2).

    All 2^n microstates are equally likely, so P_k is the fraction of them with
    k heads. Working in log-space avoids dividing two huge numbers; P_k <= 1
    never overflows (very unlikely k can underflow to 0.0).
    """
    return np.exp(log_multiplicities(n) - n * LN2)


def _factorial_float(n: int) -> float:
    """n! in float64. A Python float product gives inf on overflow (n >= 171)."""
    result = 1.0
    for i in range(2, n + 1):
        result *= i
    return result


def binomial_float_naive(n: int, k: int) -> float:
    """Omega(n, k) = n! / (k! (n - k)!) with float64 factorials (question 1d).

    This is the textbook formula done badly on purpose: n! overflows to inf for
    n >= 171 even when Omega itself is small, giving inf or nan.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if k < 0 or k > n:
        return 0.0
    return _factorial_float(n) / (_factorial_float(k) * _factorial_float(n - k))


def largest_n_naive_float() -> int:
    """Largest n for which :func:`binomial_float_naive` (at k = n // 2) is finite."""
    n = 0
    while math.isfinite(binomial_float_naive(n + 1, (n + 1) // 2)):
        n += 1
    return n


def largest_n_float_omega() -> int:
    """Largest n for which the peak value Omega(n, n // 2) itself fits in a float64.

    Beyond it, Omega can only be stored as an exact int or as its logarithm.
    Checked with exact integers against the largest float, sys.float_info.max.
    """
    n = 0
    while binomial_exact(n + 1, (n + 1) // 2) <= sys.float_info.max:
        n += 1
    return n
