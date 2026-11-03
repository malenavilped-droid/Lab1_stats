"""Unit tests for src/combinatorics.py."""

import math
from math import comb

import numpy as np
import pytest

from src.combinatorics import (
    binomial_exact,
    binomial_float_naive,
    largest_n_float_omega,
    largest_n_naive_float,
    log_binomial,
    log_multiplicities,
    multiplicities_exact,
    probabilities,
)


# --- Exact integers ---------------------------------------------------------


@pytest.mark.parametrize("n", [0, 1, 7, 20, 50, 100, 500])
def test_binomial_exact_matches_math_comb(n):
    assert [binomial_exact(n, k) for k in range(n + 1)] == [comb(n, k) for k in range(n + 1)]


@pytest.mark.parametrize("n", [0, 1, 7, 20, 100, 1000])
def test_multiplicities_exact_matches_math_comb(n):
    assert multiplicities_exact(n) == [comb(n, k) for k in range(n + 1)]


def test_n7_row():
    assert multiplicities_exact(7) == [1, 7, 21, 35, 35, 21, 7, 1]


@pytest.mark.parametrize("n", [0, 1, 20, 50, 80, 100, 1000])
def test_row_sums_to_2_pow_n(n):
    assert sum(multiplicities_exact(n)) == 2**n


def test_symmetry():
    row = multiplicities_exact(101)
    assert row == row[::-1]


@pytest.mark.parametrize("n", [2, 10, 57, 100])
def test_pascal_rule(n):
    # Omega(n, k) = Omega(n - 1, k - 1) + Omega(n - 1, k)
    for k in range(1, n):
        assert binomial_exact(n, k) == binomial_exact(n - 1, k - 1) + binomial_exact(n - 1, k)


def test_k_out_of_range_gives_zero():
    assert binomial_exact(10, -1) == 0
    assert binomial_exact(10, 11) == 0


def test_negative_n_raises():
    for func in (binomial_exact, log_binomial, binomial_float_naive):
        with pytest.raises(ValueError):
            func(-1, 0)
    for func in (multiplicities_exact, log_multiplicities):
        with pytest.raises(ValueError):
            func(-1)


# --- Log-space against the exact values -------------------------------------


@pytest.mark.parametrize("n", [1, 7, 20, 50, 80, 100, 1000])
def test_log_multiplicities_match_exact(n):
    exact_log = np.array([math.log(x) for x in multiplicities_exact(n)])
    np.testing.assert_allclose(log_multiplicities(n), exact_log, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("n, k", [(20, 10), (100, 50), (100, 30), (1000, 1), (5000, 2500)])
def test_log_binomial_matches_exact(n, k):
    assert log_binomial(n, k) == pytest.approx(math.log(binomial_exact(n, k)), rel=1e-12)


def test_log_binomial_k_out_of_range_is_minus_inf():
    assert log_binomial(10, 11) == -math.inf


def test_log_space_has_no_overflow_for_huge_n():
    n = 10**6
    log_omega = log_multiplicities(n)
    assert np.all(np.isfinite(log_omega))
    assert log_omega.max() == pytest.approx(log_binomial(n, n // 2))
    # ln Omega <= ln 2^N, the total number of microstates.
    assert log_omega.max() < n * math.log(2)


# --- Probabilities ----------------------------------------------------------


@pytest.mark.parametrize("n", [1, 20, 100, 10_000])
def test_probabilities_sum_to_one(n):
    assert probabilities(n).sum() == pytest.approx(1.0, rel=1e-12)


def test_probabilities_match_exact_fraction():
    n = 100
    exact = np.array([x / 2**n for x in multiplicities_exact(n)])
    np.testing.assert_allclose(probabilities(n), exact, rtol=1e-12)


# --- Float64 limits (question 1d) -------------------------------------------


def test_naive_float_works_while_factorials_fit():
    assert binomial_float_naive(170, 85) == pytest.approx(comb(170, 85), rel=1e-12)


def test_naive_float_breaks_at_171():
    assert not math.isfinite(binomial_float_naive(171, 85))
    assert largest_n_naive_float() == 170


def test_largest_n_float_omega():
    n = largest_n_float_omega()
    assert float(comb(n, n // 2)) < math.inf
    with pytest.raises(OverflowError):
        float(comb(n + 1, (n + 1) // 2))
