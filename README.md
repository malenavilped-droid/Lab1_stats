# Statistical Physics — Lab 1: Microstates of a System of Coins

Section 1, Group B (UC3M, 2026/27): Enrique Cortés Tirado, Malena Vílchez Pedrera,
Andrés Martín Muñoz, Pablo Aguado González.

## Setup

```bash
conda env create -f environment.yml
conda activate statphys_lab1
```

## Run

```bash
python main.py   # runs every part and writes the figures to report/figures/
pytest           # runs the unit tests
```

## Layout

| Path | Content |
|---|---|
| `src/microstates.py` | Bitwise enumeration of the 2^N microstates (1a, 1b) |
| `src/combinatorics.py` | Exact and log-space binomial coefficients (1c, 1d) |
| `src/thermodynamics.py` | Probabilities, entropy differences, waiting times (Part 2) |
| `tests/` | pytest unit tests |
| `notebooks/` | Exploratory notebooks |
| `report/` | LaTeX report (`main.tex`) and figures |
| `AI_STATEMENT.md` | Log of every AI query and how we verified it |

## Workflow

`main` is protected: work on a `feature/...` branch, open a Pull Request and get one approval from
another member before merging.
