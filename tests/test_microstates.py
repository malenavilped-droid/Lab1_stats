"""Unit tests for src/microstates.py."""

from math import comb

import pytest

from src.microstates import (
    count_heads,
    list_microstates_by_macrostate,
    macrostate_histogram,
    macrostate_histogram_fast,
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
