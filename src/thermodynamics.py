"""Probabilities, entropy differences and waiting times for N = 100 + M (Part 2). Owner: Andrés.

Every quantity is computed from ln Omega(N, k) (log-gamma, see src/combinatorics.py),
never from Omega or 2^N themselves, so nothing overflows:

    ln P_k = ln Omega(N, k) - N ln 2          (P_k = Omega / 2^N)
    Delta S / k_B = ln Omega(k1) - ln Omega(k2)  (Boltzmann, S = k_B ln Omega)

Sums of probabilities use logsumexp: ln sum_k exp(x_k) = x_max + ln sum_k exp(x_k - x_max),
which keeps every exponential <= 1.

Convention for odd N: N/2 is not an integer, so we use k = floor(N/2). By the symmetry
Omega(N, k) = Omega(N, N - k), floor(N/2) and ceil(N/2) have exactly the same
probability, so nothing depends on this choice except the label.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.constants import k as K_B  # Boltzmann constant, J/K (exact since SI 2019)
from scipy.special import logsumexp

from src.combinatorics import LN2, log_binomial, log_multiplicities

# Age of the universe, 13.8 Gyr (Planck 2018: 13.787 +/- 0.020 Gyr), in seconds.
AGE_OF_UNIVERSE_S = 13.787e9 * 365.25 * 24 * 3600
SECONDS_PER_YEAR = 365.25 * 24 * 3600


def harmonic_mean_m(digits: Sequence[int]) -> tuple[int, float]:
    """M = harmonic mean of the last two NIA digits of the team members.

    M = n / sum(1 / d_i), rounded to the nearest integer (halves round up).

    Args:
        digits: The two-digit number at the end of each member's NIA (e.g. 3 for "...03").

    Returns:
        (M rounded, M before rounding).
    """
    if len(digits) == 0:
        raise ValueError("need at least one NIA")
    if any(d <= 0 for d in digits):
        raise ValueError("the harmonic mean is undefined if a value is 0 (or negative)")
    m_raw = len(digits) / sum(1.0 / d for d in digits)
    return math.floor(m_raw + 0.5), m_raw


def half(n: int) -> int:
    """The macrostate 'k = N/2' used for N coins: floor(N/2) (see the module docstring)."""
    return n // 2


def log_prob(n: int, k: int) -> float:
    """ln P_k = ln Omega(n, k) - n ln 2. Returns -inf for k outside [0, n]."""
    return log_binomial(n, k) - n * LN2


def prob(n: int, k: int) -> float:
    """P_k = Omega(n, k) / 2^n, the probability of exactly k heads."""
    return math.exp(log_prob(n, k))


def cumulative_prob(n: int, k1: int, k2: int) -> float:
    """Delta P = sum of P_k for k = k1, ..., k2 (both included), with logsumexp.

    The range is clipped to [0, n], since P_k = 0 outside it.
    """
    lo, hi = max(k1, 0), min(k2, n)
    if lo > hi:
        return 0.0
    log_p = log_multiplicities(n)[lo : hi + 1] - n * LN2
    return float(np.exp(logsumexp(log_p)))


def tails_window_as_heads(n: int, t1: int, t2: int) -> tuple[int, int]:
    """A range of tails t1..t2 is the range of heads n - t2 .. n - t1."""
    return n - t2, n - t1


def entropy_change(n: int, k1: int, k2: int) -> float:
    """Delta S_{k1 k2} / k_B = S(k1)/k_B - S(k2)/k_B = ln Omega(n, k1) - ln Omega(n, k2).

    Dimensionless. Multiply by K_B to get J/K.
    """
    return log_binomial(n, k1) - log_binomial(n, k2)


def waiting_time(n: int, k: int, throws_per_second: float = 1.0) -> float:
    """Expected time (s) until a throw of the n coins gives exactly k heads.

    Each throw succeeds with probability p = P_k, independently, so the number of
    throws until the first success is geometric with mean 1/p. At one throw of
    all n coins per second ("a tossing rate of N coins per second"), <t> = 1/P_k s.
    """
    return math.exp(-log_prob(n, k)) / throws_per_second


def likelihood_ratio(n: int, k_ref: int, k: int) -> float:
    """P_{k_ref} / P_k = Omega(n, k_ref) / Omega(n, k) (the 2^n cancels)."""
    return math.exp(log_binomial(n, k_ref) - log_binomial(n, k))


def part2_results(n: int, m: int) -> dict[str, float]:
    """Every number asked in Part 2 (a-i) for N = n coins and the team parameter m.

    Keys are the question letters (with a suffix when a question has several numbers).
    Delta S values are in units of k_B; waiting times in seconds.
    """
    h = half(n)
    tails_lo, tails_hi = tails_window_as_heads(n, h - 20, h + 20)
    return {
        "a_P30": prob(n, 30),
        "b_P_half": prob(n, h),
        "c_heads_pm10": cumulative_prob(n, h - 10, h + 10),
        "d_tails_pm20": cumulative_prob(n, tails_lo, tails_hi),
        "e_dS": entropy_change(n, h + 25, h),
        "f_dS": entropy_change(n, h + 10, h),
        "g_dS": entropy_change(n, h, 20),
        "h_t8_s": waiting_time(n, 8),
        "i_ratio_M+10": likelihood_ratio(n, h, m + 10),
        "i_ratio_M-10": likelihood_ratio(n, h, m - 10),
        "i_ratio_half+10": likelihood_ratio(n, h, h + 10),
        "i_ratio_half-10": likelihood_ratio(n, h, h - 10),
    }


def _log_binomial_small_k(n: float, k: int) -> float:
    """ln Omega(n, k) for huge n and small k: sum_{i<k} ln(n - i) - ln k!.

    lgamma(n + 1) - lgamma(n - k + 1) would subtract two numbers ~ n ln n that
    differ by only ~ k ln n, losing every digit for n ~ 1e23. This sum does not.
    """
    return sum(math.log(n - i) for i in range(k)) - math.lgamma(k + 1)


def large_n_limits(n: float, m: int) -> dict[str, float]:
    """Part 2 for a macroscopic N (e.g. 1e23), from Stirling / the Gaussian limit (2k).

    For large N, ln Omega(N, k) ~ N ln 2 - (1/2) ln(pi N / 2) - 2 (k - N/2)^2 / N,
    a Gaussian centred at N/2 with sigma = sqrt(N)/2. Fixed-k quantities (k = 30, 8,
    20, M +/- 10) are far in the tail and use the exact small-k sum instead.

    Probabilities that underflow are returned as log10 values (keys ending in "_log10").
    """
    sigma = math.sqrt(n) / 2
    log_p_half = -0.5 * math.log(math.pi * n / 2)  # ln P_{N/2}
    ln10 = math.log(10)

    def log_p_small(k: int) -> float:
        return _log_binomial_small_k(n, k) - n * LN2

    def window(w: int) -> float:
        # P(|k - N/2| <= w) with the continuity correction (+0.5) for a discrete k.
        return math.erf((w + 0.5) / (sigma * math.sqrt(2)))

    return {
        "sigma_over_mu": sigma / (n / 2),
        "a_P30_log10": log_p_small(30) / ln10,
        "b_P_half": math.exp(log_p_half),
        "c_heads_pm10": window(10),
        "d_tails_pm20": window(20),
        "e_dS": -2 * 25**2 / n,
        "f_dS": -2 * 10**2 / n,
        "g_dS": (log_p_half + n * LN2) - _log_binomial_small_k(n, 20),
        "h_t8_s_log10": -log_p_small(8) / ln10,
        "i_ratio_M+10_log10": (log_p_half - log_p_small(m + 10)) / ln10,
        "i_ratio_half+10": math.exp(2 * 10**2 / n),
    }
