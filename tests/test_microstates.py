"""Unit tests for src/microstates.py."""

from math import comb

import numpy as np
import pytest

from src.microstates import (
    benchmark_enumeration,
    count_heads,
    list_microstates_by_macrostate,
    macrostate_histogram,
    macrostate_histogram_fast,
    macrostate_label,
    microstates,
    state_to_string,
)


def test_generator_yields_all_states_once():
    states = list(microstates(5))
    assert states == list(range(32))


def test_generator_rejects_negative_n():
    with pytest.raises(ValueError):
        list(microstates(-1))


def test_count_heads():
    assert count_heads(0) == 0
    assert count_heads(0b1011) == 3
    assert count_heads((1 << 40) - 1) == 40


def test_state_to_string():
    assert state_to_string(0b011, 3) == "HHT"
    assert state_to_string(0, 4) == "TTTT"


def test_n7_matches_expected_multiplicities():
    assert macrostate_histogram(7) == [1, 7, 21, 35, 35, 21, 7, 1]


def test_n7_listing_is_complete_and_correct():
    groups = list_microstates_by_macrostate(7)
    all_states = [s for strings in groups.values() for s in strings]
    assert len(all_states) == 128
    assert len(set(all_states)) == 128
    for k, strings in groups.items():
        assert all(s.count("H") == k for s in strings)


@pytest.mark.parametrize("n", [0, 1, 5, 10, 16])
def test_histogram_matches_binomial(n):
    assert macrostate_histogram(n) == [comb(n, k) for k in range(n + 1)]


@pytest.mark.parametrize("n", [0, 1, 7, 12, 20, 24])
def test_fast_histogram_matches_binomial(n):
    assert macrostate_histogram_fast(n) == [comb(n, k) for k in range(n + 1)]


def test_fast_histogram_with_small_chunks():
    # Many chunks: checks that chunk boundaries are handled correctly.
    assert macrostate_histogram_fast(12, chunk_bits=3) == [comb(12, k) for k in range(13)]


def test_histogram_sums_to_2_pow_n():
    assert sum(macrostate_histogram_fast(18)) == 2**18


def test_fast_histogram_lookup_table_fallback(monkeypatch):
    # Force the numpy < 2.0 path (byte lookup table).
    monkeypatch.delattr(np, "bitwise_count", raising=False)
    assert macrostate_histogram_fast(20) == [comb(20, k) for k in range(21)]


def test_fast_histogram_rejects_n_above_63():
    with pytest.raises(ValueError):
        macrostate_histogram_fast(64)


def test_benchmark_returns_mean_std_ram():
    mean, std, ram = benchmark_enumeration(12, repeats=3)
    assert mean > 0 and std >= 0 and ram > 0
    assert benchmark_enumeration(12, repeats=1)[1] == 0.0


def test_macrostate_label_singular_and_plural():
    assert macrostate_label(1, 7) == "1 head, 6 tails"
    assert macrostate_label(6, 7) == "6 heads, 1 tail"
    assert macrostate_label(0, 7) == "0 heads, 7 tails"
