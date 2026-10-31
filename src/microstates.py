"""Bitwise enumeration of the 2^N microstates of N coins (questions 1a, 1b). Owner: Malena.

Encoding: every integer s in [0, 2^N) is one microstate. Bit i of s is coin i
(1 = heads, 0 = tails), so the number of heads k of a microstate is the number
of 1-bits of s (its popcount). The macrostate histogram counts how many
microstates have each k, i.e. the multiplicity Omega(N, k).
"""

from __future__ import annotations

import time
import tracemalloc
from collections.abc import Iterator

import numpy as np


def microstates(n: int) -> Iterator[int]:
    """Yield every microstate of ``n`` coins as an integer in [0, 2^n).

    A generator keeps only the current state in memory, so the memory cost is
    O(1) instead of the O(n 2^n) bits needed to store the whole list.

    Args:
        n: Number of coins (n >= 0).

    Yields:
        The integers 0, 1, ..., 2^n - 1, one per microstate.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    yield from range(1 << n)


def count_heads(state: int) -> int:
    """Return the number of heads (1-bits) of a microstate."""
    return bin(state).count("1")


def state_to_string(state: int, n: int) -> str:
    """Write a microstate as a string of H/T, coin 0 first.

    Example: ``state_to_string(0b011, 3) == "HHT"``.
    """
    return "".join("H" if (state >> i) & 1 else "T" for i in range(n))


def list_microstates_by_macrostate(n: int) -> dict[int, list[str]]:
    """Group all 2^n microstates by their number of heads k (question 1a).

    Only meant for small n, because it stores every microstate as a string.

    Returns:
        A dict {k: [microstate strings with k heads]} with keys k = n, n-1, ..., 0.
    """
    groups: dict[int, list[str]] = {k: [] for k in range(n, -1, -1)}
    for s in microstates(n):
        groups[count_heads(s)].append(state_to_string(s, n))
    return groups


def macrostate_histogram(n: int) -> list[int]:
    """Count the microstates of each macrostate by visiting all 2^n states.

    Pure-Python reference version: loops over the generator one state at a time.

    Returns:
        A list ``omega`` of length n + 1 with ``omega[k]`` = number of microstates with k heads.
    """
    omega = [0] * (n + 1)
    for s in microstates(n):
        omega[count_heads(s)] += 1
    return omega


# Popcount of every byte 0..255, used when numpy has no bitwise_count (numpy < 2.0).
_BYTE_POPCOUNT = np.array([bin(b).count("1") for b in range(256)], dtype=np.uint8)


def _popcount(arr: np.ndarray, n: int) -> np.ndarray:
    """Popcount of each element of a uint64 array whose values are below 2^n."""
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(arr)
    counts = np.zeros(arr.shape, dtype=np.uint8)
    for shift in range(0, n, 8):
        counts += _BYTE_POPCOUNT[(arr >> np.uint64(shift)) & np.uint64(0xFF)]
    return counts


def macrostate_histogram_fast(n: int, chunk_bits: int = 22) -> list[int]:
    """Same result as :func:`macrostate_histogram`, vectorised with numpy.

    The states are still all visited, but in chunks of 2^chunk_bits consecutive
    integers: each chunk is built with ``np.arange``, its popcounts are computed
    at once and added to the histogram with ``np.bincount``. Memory stays
    bounded by the chunk size (about 2^chunk_bits * 9 bytes), not by 2^n.

    Args:
        n: Number of coins (0 <= n <= 63).
        chunk_bits: log2 of the number of states processed per chunk.

    Returns:
        A list of length n + 1 with the multiplicity of each k.
    """
    if not 0 <= n <= 63:
        raise ValueError("n must be between 0 and 63")
    total = 1 << n
    chunk = 1 << min(chunk_bits, n)
    omega = np.zeros(n + 1, dtype=np.int64)
    for start in range(0, total, chunk):
        states = np.arange(start, start + chunk, dtype=np.uint64)
        omega += np.bincount(_popcount(states, n), minlength=n + 1)
    return [int(x) for x in omega]


def benchmark_enumeration(
    n: int, repeats: int = 3, fast: bool = True
) -> tuple[float, float, float]:
    """Time the full enumeration of the 2^n microstates of n coins.

    Args:
        n: Number of coins.
        repeats: How many times the enumeration is repeated.
        fast: Use the numpy version if True, the pure-Python one otherwise.

    Returns:
        (mean time in s, standard deviation in s, peak extra RAM in MiB).
        The std is the sample std (ddof=1), or 0 if repeats == 1. The RAM
        is measured with tracemalloc in a separate, untimed run.
    """
    func = macrostate_histogram_fast if fast else macrostate_histogram
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        func(n)
        times.append(time.perf_counter() - t0)

    tracemalloc.start()
    func(n)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    t = np.array(times)
    std = float(t.std(ddof=1)) if repeats > 1 else 0.0
    return float(t.mean()), std, peak / 2**20
