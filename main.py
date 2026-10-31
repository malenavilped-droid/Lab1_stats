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

from src.microstates import (
    benchmark_enumeration,
    list_microstates_by_macrostate,
    macrostate_histogram_fast,
)

FIGURES = Path(__file__).parent / "report" / "figures"
TIME_LIMIT_S = 300.0  # 5 minutes of wall-clock time (question 1b)


def part_1a(n: int = 7) -> None:
    """1a: list the 2^7 microstates grouped by macrostate and print their multiplicity."""
    print(f"\n=== 1a: microstates of N = {n} coins ===")
    groups = list_microstates_by_macrostate(n)
    print(f"{'macrostate':<22}{'# microstates':>14}")
    for k, states in groups.items():
        print(f"{f'{k} heads, {n - k} tails':<22}{len(states):>14}")
    print(f"{'total':<22}{sum(len(s) for s in groups.values()):>14}  (2^{n} = {2**n})")

    # Full list for the report's appendix.
    out = FIGURES.parent / f"microstates_N{n}.txt"
    with open(out, "w", encoding="utf-8") as f:
        for k, states in groups.items():
            f.write(f"{k} heads and {n - k} tails ({len(states)}): {', '.join(states)}\n")
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

    print("Statistical Physics - Lab 1: Microstates of a System of Coins")
    FIGURES.mkdir(parents=True, exist_ok=True)

    part_1a()
    part_1b(args.max_n)
    if args.find_limit:
        find_limit()


if __name__ == "__main__":
    main()
