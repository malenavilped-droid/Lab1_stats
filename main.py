"""Reproduce every result of the report: python main.py

Each part writes its tables to the terminal and its figures (vector .pdf) to report/figures/.
"""

from pathlib import Path

FIGURES = Path(__file__).parent / "report" / "figures"


def part1() -> None:
    """Computational analysis of microstates (questions 1a-1d)."""


def part2() -> None:
    """Probabilities, entropy and thermodynamics for N = 100 + M (questions 2a-2i)."""


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    part1()
    part2()


if __name__ == "__main__":
    main()
