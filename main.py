"""Entry point: runs every part of Lab 1 and writes the figures to report/figures/.

Each part is added by its owner as the modules in src/ are completed.

Usage:
    python main.py                 # all parts, quick benchmark (N up to 24)
    python main.py --max-n 30      # benchmark up to a larger N
    python main.py --find-limit    # 1b: largest N enumerated in at most 5 minutes
    python main.py --m 9           # Part 2 with another M (default: from NIA_DIGITS)
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
from src.thermodynamics import (
    AGE_OF_UNIVERSE_S,
    K_B,
    SECONDS_PER_YEAR,
    half,
    harmonic_mean_m,
    large_n_limits,
    part2_results,
)

FIGURES = Path(__file__).parent / "report" / "figures"
TIME_LIMIT_S = 300.0  # 5 minutes of wall-clock time (question 1b)

# Last two digits of each member's NIA (Part 2): Andrés ...03, Pablo ...27,
# Malena ...17, Enrique ...51. M = harmonic mean, N = 100 + M.
NIA_DIGITS = (3, 27, 17, 51)


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


def part_2(m: int, m_raw: float | None = None) -> None:
    """Part 2 (a-i) for N = 100 + M: Table 3, a figure, and the N ~ 1e23 limit (2k)."""
    n = 100 + m
    h = half(n)
    r = part2_results(n, m)
    print(f"\n=== Part 2: probabilities and entropy for N = 100 + M = {n} ===")
    if m_raw is not None:
        print(f"M = harmonic mean of the NIA digits {NIA_DIGITS} = {m_raw:.4f} -> rounded to {m}")
    if n % 2:
        print(f"N is odd: 'N/2' is taken as floor(N/2) = {h} (P_{h} = P_{h + 1} by symmetry)")
    tails_lo, tails_hi = n - (h + 20), n - (h - 20)
    t8 = r["h_t8_s"]

    def ratio_text(value: float) -> str:
        return "inf (k < 0: no microstate, P = 0)" if value == float("inf") else f"{value:.6e}"

    rows = [
        ("a", "P_30", f"{r['a_P30']:.6e}"),
        ("b", f"P_{h} (k = N/2)", f"{r['b_P_half']:.6e}"),
        ("c", f"sum P_k, k = {h - 10}..{h + 10} heads", f"{r['c_heads_pm10']:.6f}"),
        ("d", f"sum P_k, {h - 20}..{h + 20} tails (= {tails_lo}..{tails_hi} heads)",
         f"{r['d_tails_pm20']:.6f}"),
        ("e", f"dS(k1={h + 25}, k2={h}) / k_B", f"{r['e_dS']:.6f}  ({r['e_dS'] * K_B:.4e} J/K)"),
        ("f", f"dS(k1={h + 10}, k2={h}) / k_B", f"{r['f_dS']:.6f}  ({r['f_dS'] * K_B:.4e} J/K)"),
        ("g", f"dS(k1={h}, k2=20) / k_B", f"{r['g_dS']:.6f}  ({r['g_dS'] * K_B:.4e} J/K)"),
        ("h", "<t> to get 8 heads (1 throw/s)",
         f"{t8:.4e} s = {t8 / SECONDS_PER_YEAR:.4e} yr = {t8 / AGE_OF_UNIVERSE_S:.3e} x age of universe"),
        ("i", f"P_{h} / P_(M+10 = {m + 10})", ratio_text(r["i_ratio_M+10"])),
        ("i", f"P_{h} / P_(M-10 = {m - 10})", ratio_text(r["i_ratio_M-10"])),
        ("i", f"P_{h} / P_(N/2+10 = {h + 10})", ratio_text(r["i_ratio_half+10"])),
        ("i", f"P_{h} / P_(N/2-10 = {h - 10})", ratio_text(r["i_ratio_half-10"])),
    ]
    lines = [f"Table 3: Part 2 for N = {n} (M = {m}). Exact values, computed in log-space "
             "(relative error < 1e-12 against exact integers); no statistical uncertainty."]
    lines += [f"({q}) {name:<52}{value}" for q, name, value in rows]
    for line in lines:
        print(line)
    out = FIGURES.parent / "table3_part2.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Table written to {out}")

    # Figure: (a) P_k with the windows of c and d; (b) log scale with every k of a-i marked.
    k = np.arange(n + 1)
    p = probabilities(n)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 4))
    ax_a.fill_between(k, p, where=(k >= tails_lo) & (k <= tails_hi), step="mid",
                      color=COLORS_1C[1], alpha=0.25, label=r"(d) $\pm 20$ tails")
    ax_a.fill_between(k, p, where=(k >= h - 10) & (k <= h + 10), step="mid",
                      color=COLORS_1C[0], alpha=0.45, label=r"(c) $\pm 10$ heads")
    ax_a.plot(k, p, "-", color="#333333", lw=1.2)
    ax_a.set_xlabel("Number of heads $k$")
    ax_a.set_ylabel(r"$P_k = \Omega(N,k)/2^N$")
    ax_a.set_title(f"(a) $N = {n}$: cumulative windows", fontsize=10, loc="left")
    ax_a.set_xlim(h - 35, h + 35)  # the tails beyond are ~0 on a linear axis
    ax_a.set_ylim(bottom=0)
    ax_b.semilogy(k[p > 0], p[p > 0], "-", color="#333333", lw=1.2)
    marks = {8: "h", 20: "g", 30: "a", m + 10: "i", h: "b", h + 10: "f", h + 25: "e"}
    for kk, q in marks.items():
        ax_b.plot(kk, p[kk], "o", color=COLORS_1C[2], ms=5)
        # (i) k = M + 10 sits next to (g) k = 20: put its label below the curve.
        offset = (8, -12) if q == "i" else (4, 6)
        ax_b.annotate(f"({q}) k={kk}", (kk, p[kk]), textcoords="offset points",
                      xytext=offset, fontsize=7)
    ax_b.set_xlabel("Number of heads $k$")
    ax_b.set_ylabel(r"$P_k$ (log scale)")
    ax_b.set_title("(b) Macrostates used in questions a-i", fontsize=10, loc="left")
    ax_b.set_xlim(0, n)
    for ax in (ax_a, ax_b):
        ax.grid(True, color="#e4e3df", lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    ax_a.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = FIGURES / "part2_distribution.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Figure written to {out}")

    # 2k: the same questions for a macroscopic N (Stirling / Gaussian limit).
    big_n = 1e23
    big = large_n_limits(big_n, m)
    print(f"\n=== 2k: the same quantities for N = {big_n:.0e} (Stirling / Gaussian limit) ===")
    print(f"sigma/mu = 1/sqrt(N)                 {big['sigma_over_mu']:.3e}")
    print(f"(a) log10 P_30                       {big['a_P30_log10']:.4e}")
    print(f"(b) P_N/2 ~ sqrt(2/(pi N))           {big['b_P_half']:.3e}")
    print(f"(c) P(+/-10 heads around N/2)        {big['c_heads_pm10']:.3e}")
    print(f"(d) P(+/-20 tails around N/2)        {big['d_tails_pm20']:.3e}")
    print(f"(e) dS/k_B ~ -2 (25)^2 / N           {big['e_dS']:.3e}")
    print(f"(f) dS/k_B ~ -2 (10)^2 / N           {big['f_dS']:.3e}")
    print(f"(g) dS/k_B ~ N ln 2                  {big['g_dS']:.4e}")
    print(f"(h) log10 <t> (s) for 8 heads        {big['h_t8_s_log10']:.4e}")
    print(f"(i) log10 P_N/2 / P_(M+10)           {big['i_ratio_M+10_log10']:.4e}")
    print(f"(i) P_N/2 / P_(N/2+10) ~ e^(200/N)   {big['i_ratio_half+10']:.15f}")


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
    parser.add_argument("--m", type=int, default=None,
                        help="Part 2 parameter M (default: harmonic mean of NIA_DIGITS)")
    args = parser.parse_args()
    # The fit in part_1b uses N >= 16 and needs at least two points (N = 16 and 18).
    if args.max_n < 18:
        parser.error("--max-n must be >= 18")
    # The guide restricts N = 100 + M to below 200, and question (i) needs M + 10 <= N.
    if args.m is not None and not 0 < args.m < 100:
        parser.error("--m must be between 1 and 99 (N = 100 + M < 200)")

    print("Statistical Physics - Lab 1: Microstates of a System of Coins")
    FIGURES.mkdir(parents=True, exist_ok=True)

    part_1a()
    part_1b(args.max_n)
    part_1c()
    part_1d()
    if args.m is None:
        m, m_raw = harmonic_mean_m(NIA_DIGITS)
        part_2(m, m_raw)
    else:
        part_2(args.m)
    if args.find_limit:
        find_limit()


if __name__ == "__main__":
    main()
