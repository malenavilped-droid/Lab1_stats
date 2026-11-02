"""Entry point: runs every part of Lab 1 and writes the figures to report/figures/.

Each part is added by its owner as the modules in src/ are completed.

Usage:
    python main.py                 # all parts, quick benchmark (N up to 24)
    python main.py --max-n 30      # benchmark up to a larger N
    python main.py --find-limit    # 1b: largest N enumerated in at most 5 minutes
"""

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only, no window
import matplotlib.pyplot as plt
import numpy as np

from src.combinatorics import (
    largest_n_float_omega,
    largest_n_naive_float,
    log_multiplicities,
    multiplicities_exact,
    probabilities,
)
from src.microstates import (
    benchmark_enumeration,
    list_microstates_by_macrostate,
    macrostate_histogram_fast,
    macrostate_label,
)

FIGURES = Path(__file__).parent / "report" / "figures"
TIME_LIMIT_S = 300.0  # 5 minutes of wall-clock time (question 1b)


def _time_call(func, n: int, repeats: int = 3) -> tuple[float, float]:
    """Mean and standard deviation (s) of the wall-clock time of func(n) over repeats."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        func(n)
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)), float(np.std(times))


def part_1a(n: int = 7) -> None:
    """1a: list the 2^7 microstates grouped by macrostate and print their multiplicity."""
    print(f"\n=== 1a: microstates of N = {n} coins ===")
    groups = list_microstates_by_macrostate(n)
    print(f"{'macrostate':<22}{'# microstates':>14}")
    for k, states in groups.items():
        print(f"{macrostate_label(k, n):<22}{len(states):>14}")
    print(f"{'total':<22}{sum(len(s) for s in groups.values()):>14}  (2^{n} = {2**n})")

    # Full list for the report's appendix.
    out = FIGURES.parent / f"microstates_N{n}.txt"
    with open(out, "w", encoding="utf-8") as f:
        for k, states in groups.items():
            f.write(f"{macrostate_label(k, n)} ({len(states)}): {', '.join(states)}\n")
    print(f"Full list written to {out}")


def part_1b(max_n: int, repeats: int = 3) -> None:
    """1b: time the full enumeration for growing N and plot time vs N."""
    print(f"\n=== 1b: enumeration time vs N ({repeats} repeats each) ===")
    print(f"{'N':>3}{'2^N':>16}{'time (s)':>22}{'peak RAM (MiB)':>16}")
    ns, means, stds = [], [], []
    for n in range(10, max_n + 1, 2):
        mean, std, ram = benchmark_enumeration(n, repeats=repeats)
        ns.append(n)
        means.append(mean)
        stds.append(std)
        print(f"{n:>3}{2**n:>16}{f'{mean:.4f} +/- {std:.4f}':>22}{ram:>16.2f}")

    # Fit ln t = a + b N on the larger N (small N is dominated by overhead).
    ns_arr, means_arr = np.array(ns), np.array(means)
    fit = ns_arr >= max(16, ns_arr.min())
    b, a = np.polyfit(ns_arr[fit], np.log(means_arr[fit]), 1)
    print(f"Fit ln t = a + b N: b = {b:.3f} (expected ln 2 = {np.log(2):.3f})")
    n_limit = (np.log(TIME_LIMIT_S) - a) / b
    print(f"Extrapolated largest N within {TIME_LIMIT_S:.0f} s: N ~ {n_limit:.1f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(ns, means, yerr=stds, fmt="o", capsize=3, label="measured (mean +/- std)")
    n_line = np.linspace(min(ns), max(n_limit, max(ns)), 100)
    ax.plot(n_line, np.exp(a + b * n_line), "--", label=rf"fit $t \propto e^{{{b:.3f}N}}$")
    ax.axhline(TIME_LIMIT_S, color="gray", linestyle=":", label="5 min limit")
    ax.set_yscale("log")
    ax.set_xlabel("Number of coins $N$")
    ax.set_ylabel("Enumeration time (s)")
    ax.legend()
    fig.tight_layout()
    out = FIGURES / "enumeration_time.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Figure written to {out}")


N_VALUES_1C = (20, 50, 80, 100)
# Categorical palette (blue, orange, aqua, yellow), one colour per N.
COLORS_1C = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")


def part_1c() -> None:
    """1c: Omega(N, k) for N = 20, 50, 80, 100 on one figure with two panels.

    (a) is what the guide asks for: exact Omega against k, on a log axis because
    Omega(100, 50) ~ 1e29 and Omega(20, 10) ~ 2e5. (b) is the same data normalised,
    N P_k against k/N, so every curve has area 1 and the narrowing peak is visible.
    """
    print(f"\n=== 1c: macrostate distributions for N = {', '.join(map(str, N_VALUES_1C))} ===")
    print(f"{'N':>4}{'2^N':>14}{'max Omega (k = N/2)':>24}{'sum Omega = 2^N':>18}")
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 4))
    rows = {}
    for n, color in zip(N_VALUES_1C, COLORS_1C):
        omega = multiplicities_exact(n)  # exact Python ints
        # Cross-check the exact values with the log-space version.
        assert np.allclose(np.log(np.array(omega, dtype=float)), log_multiplicities(n), rtol=1e-12)
        rows[n] = omega
        print(f"{n:>4}{float(2**n):>14.3e}{float(max(omega)):>24.4e}{str(sum(omega) == 2**n):>18}")
        k = np.arange(n + 1)
        ax_a.plot(k, [float(x) for x in omega], "o-", color=color, lw=1.5, ms=3, label=f"$N = {n}$")
        ax_b.plot(k / n, n * probabilities(n), "-", color=color, lw=1.8, label=f"$N = {n}$")

    ax_a.set_yscale("log")
    ax_a.set_xlabel("Number of heads $k$")
    ax_a.set_ylabel(r"Multiplicity $\Omega(N, k)$")
    ax_a.set_title("(a) Number of microstates per macrostate", fontsize=10, loc="left")
    ax_b.set_xlabel("Fraction of heads $k/N$")
    ax_b.set_ylabel(r"Probability density $N\,P_k$")
    ax_b.set_title(r"(b) Normalised: $P_k = \Omega / 2^N$", fontsize=10, loc="left")
    ax_b.set_xlim(0, 1)
    ax_b.set_ylim(bottom=0)
    for ax in (ax_a, ax_b):
        ax.legend(frameon=False)
        ax.grid(True, color="#e4e3df", lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    out = FIGURES / "multiplicity_distribution.pdf"
    fig.savefig(out)
    plt.close(fig)
    print("Exact integer values (no uncertainty); panel (b) uses the log-gamma probabilities.")
    print(f"Figure written to {out}")

    # Exact values for the report's appendix.
    out = FIGURES.parent / "multiplicities_1c.txt"
    with open(out, "w", encoding="utf-8") as f:
        for n, omega in rows.items():
            f.write(f"N = {n}: " + ", ".join(str(x) for x in omega) + "\n")
    print(f"Exact values written to {out}")


def part_1d(repeats: int = 3) -> None:
    """1d: largest N for which the distribution can be computed and plotted, per method."""
    print("\n=== 1d: largest N for the macrostate distribution ===")
    n_fact = largest_n_naive_float()
    n_omega = largest_n_float_omega()
    print(f"float64 factorials N!/(k!(N-k)!): works up to N = {n_fact} ({n_fact + 1}! overflows to inf)")
    print(f"Omega(N, N/2) itself as a float64: fits up to N = {n_omega} (largest float ~ 1.8e308);")
    print("  beyond it Omega can only be kept as an exact int or as its logarithm, and")
    print("  matplotlib needs floats, so Omega cannot be plotted directly")

    print(f"\nExact ints (no overflow, cost grows ~ N^2), full row k = 0..N, {repeats} repeats:")
    for n in (10**3, 3 * 10**3, 10**4, 3 * 10**4):  # N = 1e5 already takes ~7 s per row
        mean, std = _time_call(multiplicities_exact, n, repeats)
        print(f"  N = {n:>8}: {mean:.4f} +/- {std:.4f} s")

    print(f"\nLog-space (no overflow for any N), full row k = 0..N, {repeats} repeats:")
    print(f"{'N':>10}{'time (s)':>22}{'|sum P_k - 1|':>16}{'k with P_k > 0':>16}{'array (MiB)':>13}")
    for n in (10**2, 10**4, 10**6, 10**7):
        mean, std = _time_call(log_multiplicities, n, repeats)
        p = probabilities(n)
        nonzero = int(np.count_nonzero(p))
        print(f"{n:>10}{f'{mean:.4f} +/- {std:.4f}':>22}{abs(p.sum() - 1):>16.1e}"
              f"{f'{nonzero}/{n + 1}':>16}{8 * (n + 1) / 2**20:>13.1f}")
    print("Limits of log-space: precision (ln Omega ~ N ln 2, so rounding grows with N),")
    print("  P_k of the far tails underflows to 0, and memory/plotting of N + 1 points.")


def find_limit(start_n: int = 24) -> None:
    """1b: enumerate N = start_n, start_n + 1, ... until one run exceeds 5 minutes."""
    print(f"\n=== 1b: largest N enumerated in at most {TIME_LIMIT_S:.0f} s ===")
    best = None
    n = start_n
    while True:
        t0 = time.perf_counter()
        omega = macrostate_histogram_fast(n)
        elapsed = time.perf_counter() - t0
        assert sum(omega) == 2**n
        print(f"N = {n}: {elapsed:.2f} s")
        if elapsed > TIME_LIMIT_S:
            break
        best = n
        # Each extra coin doubles the time: stop if the next one cannot fit.
        if 2 * elapsed > 1.2 * TIME_LIMIT_S:
            print(f"N = {n + 1} would take about {2 * elapsed:.0f} s, stopping.")
            break
        n += 1
    print(f"Largest N fully enumerated within {TIME_LIMIT_S:.0f} s: {best}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Statistical Physics Lab 1")
    parser.add_argument("--max-n", type=int, default=24, help="largest N in the 1b benchmark")
    parser.add_argument("--find-limit", action="store_true", help="search the 5-minute limit (slow)")
    args = parser.parse_args()
    # The fit in part_1b uses N >= 16 and needs at least two points (N = 16 and 18).
    if args.max_n < 18:
        parser.error("--max-n must be >= 18")

    print("Statistical Physics - Lab 1: Microstates of a System of Coins")
    FIGURES.mkdir(parents=True, exist_ok=True)

    part_1a()
    part_1b(args.max_n)
    part_1c()
    part_1d()
    if args.find_limit:
        find_limit()


if __name__ == "__main__":
    main()
