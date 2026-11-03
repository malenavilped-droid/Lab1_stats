"""Unit tests for src/thermodynamics.py: every log-space result is checked against exact integers."""

import math
from fractions import Fraction
from math import comb

import pytest

from src.thermodynamics import (
    cumulative_prob,
    entropy_change,
    half,
    harmonic_mean_m,
    large_n_limits,
    likelihood_ratio,
    log_prob,
    part2_results,
    prob,
    tails_window_as_heads,
    waiting_time,
)


def exact_prob(n: int, k: int) -> Fraction:
    return Fraction(comb(n, k), 2**n)


# --- M from the NIAs --------------------------------------------------------


def test_harmonic_mean_by_hand():
    # 2 / (1/3 + 1/27) = 2 / (10/27) = 5.4 -> 5
    assert harmonic_mean_m([3, 27]) == (5, pytest.approx(5.4))


def test_harmonic_mean_of_our_team():
    # 4 / (1/3 + 1/27 + 1/17 + 1/51) = 8.9126...
    m, m_raw = harmonic_mean_m([3, 27, 17, 51])
    assert m_raw == pytest.approx(8.9126, abs=1e-4)
    assert m == 9


def test_harmonic_mean_rounds_halves_up():
    # 2 / (1/3 + 1/9) = 2 / (4/9) = 4.5 -> 5
    assert harmonic_mean_m([3, 9]) == (5, pytest.approx(4.5))
    assert harmonic_mean_m([2, 6])[0] == 3  # exactly 3.0


@pytest.mark.parametrize("bad", [[], [3, 0], [3, -1]])
def test_harmonic_mean_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        harmonic_mean_m(bad)


# --- Probabilities ----------------------------------------------------------


@pytest.mark.parametrize("n", [100, 109, 150, 199])
def test_prob_matches_exact_fraction(n):
    for k in (0, 8, 20, 30, half(n), n - 1, n):
        assert prob(n, k) == pytest.approx(float(exact_prob(n, k)), rel=1e-12)


def test_prob_out_of_range_is_zero():
    assert prob(109, -1) == 0.0
    assert prob(109, 110) == 0.0
    assert log_prob(109, -1) == -math.inf


@pytest.mark.parametrize("n", [1, 109, 199])
def test_full_window_sums_to_one(n):
    assert cumulative_prob(n, 0, n) == pytest.approx(1.0, rel=1e-12)
    assert cumulative_prob(n, -50, n + 50) == pytest.approx(1.0, rel=1e-12)  # clipped


@pytest.mark.parametrize("n, k1, k2", [(109, 44, 64), (109, 34, 74), (150, 65, 85), (199, 0, 30)])
def test_cumulative_matches_exact_sum(n, k1, k2):
    exact = sum(exact_prob(n, k) for k in range(k1, k2 + 1))
    assert cumulative_prob(n, k1, k2) == pytest.approx(float(exact), rel=1e-12)


def test_empty_window_is_zero():
    assert cumulative_prob(109, 60, 50) == 0.0


def test_odd_n_half_has_same_probability_on_both_sides():
    n = 109
    assert half(n) == 54
    assert prob(n, 54) == pytest.approx(prob(n, 55), rel=1e-14)


def test_tails_window_equals_heads_window_by_symmetry():
    n = 109
    h = half(n)
    lo, hi = tails_window_as_heads(n, h - 20, h + 20)
    assert (lo, hi) == (35, 75)  # 34..74 tails = 35..75 heads
    assert cumulative_prob(n, lo, hi) == pytest.approx(cumulative_prob(n, h - 20, h + 20), rel=1e-12)


# --- Entropy ----------------------------------------------------------------


@pytest.mark.parametrize("k1, k2", [(79, 54), (64, 54), (54, 20), (8, 101)])
def test_entropy_change_matches_exact(k1, k2):
    n = 109
    exact = math.log(comb(n, k1)) - math.log(comb(n, k2))
    assert entropy_change(n, k1, k2) == pytest.approx(exact, rel=1e-12)


def test_entropy_change_signs_and_antisymmetry():
    n = 109
    assert entropy_change(n, 54, 54) == 0.0
    assert entropy_change(n, 79, 54) < 0  # leaving the maximum lowers S
    assert entropy_change(n, 54, 20) > 0
    assert entropy_change(n, 30, 60) == pytest.approx(-entropy_change(n, 60, 30))


# --- Waiting time and ratios ------------------------------------------------


def test_waiting_time_is_inverse_probability():
    n = 109
    assert waiting_time(n, 8) == pytest.approx(float(1 / exact_prob(n, 8)), rel=1e-12)
    assert waiting_time(n, 8, throws_per_second=10) == pytest.approx(waiting_time(n, 8) / 10)


def test_likelihood_ratio_matches_exact():
    n = 109
    exact = Fraction(comb(n, 54), comb(n, 19))
    assert likelihood_ratio(n, 54, 19) == pytest.approx(float(exact), rel=1e-12)
    assert likelihood_ratio(n, 54, 54) == 1.0


def test_ratio_to_impossible_macrostate_is_infinite():
    # M - 10 = -1 for our team: no microstate has -1 heads, so P = 0.
    assert likelihood_ratio(109, 54, -1) == math.inf


def test_part2_results_uses_the_same_functions():
    r = part2_results(109, 9)
    assert r["b_P_half"] == prob(109, 54)
    assert r["e_dS"] == entropy_change(109, 79, 54)
    assert r["g_dS"] == entropy_change(109, 54, 20)
    assert r["i_ratio_M+10"] == likelihood_ratio(109, 54, 19)
    assert r["i_ratio_M-10"] == math.inf
    assert 0 < r["c_heads_pm10"] < r["d_tails_pm20"] < 1


# --- Large-N limit (2k) -----------------------------------------------------


def test_large_n_formulas_match_exact_at_moderate_n():
    """At N = 10^4 the Stirling/Gaussian formulas must already be close to the exact values."""
    n, m = 10_000, 9
    big = large_n_limits(n, m)
    h = n // 2
    assert big["b_P_half"] == pytest.approx(prob(n, h), rel=1e-3)
    assert big["c_heads_pm10"] == pytest.approx(cumulative_prob(n, h - 10, h + 10), rel=1e-3)
    assert big["d_tails_pm20"] == pytest.approx(cumulative_prob(n, h - 20, h + 20), rel=1e-3)
    assert big["e_dS"] == pytest.approx(entropy_change(n, h + 25, h), rel=1e-2)
    assert big["g_dS"] == pytest.approx(entropy_change(n, h, 20), rel=1e-6)
    assert big["a_P30_log10"] == pytest.approx(log_prob(n, 30) / math.log(10), rel=1e-12)
    # <t> = 1/P_8 ~ 10^3000 s here, beyond float64, so compare the logarithms.
    assert big["h_t8_s_log10"] == pytest.approx(-log_prob(n, 8) / math.log(10), rel=1e-12)


def test_large_n_limit_values():
    big = large_n_limits(1e23, 9)
    assert big["sigma_over_mu"] == pytest.approx(1 / math.sqrt(1e23))
    assert big["b_P_half"] == pytest.approx(math.sqrt(2 / (math.pi * 1e23)))
    assert big["c_heads_pm10"] < 1e-9  # a fixed window of +/-10 has vanishing probability
    # A fixed offset of 25 costs (almost) no entropy: -2 * 25^2 / 1e23 = -1.25e-20.
    assert big["e_dS"] == pytest.approx(-1.25e-20)
    assert big["g_dS"] == pytest.approx(1e23 * math.log(2), rel=1e-9)  # extensive
    assert big["h_t8_s_log10"] > 1e22
