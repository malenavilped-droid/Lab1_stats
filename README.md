# Lab 1 — Microstates of a System of Coins

Statistical Physics (UC3M, 2026/27) · Laboratory Practice 1 · Section 1, Group B

Team: Malena Vílchez Pedrera (@malenavilped-droid) · Enrique Cortés Tirado (@quiquecortestirado-afk) · Andrés Martín Muñoz (@andrespepsi) · Pablo Aguado González (@PABLOSKYAGU)

**Report deadline:** Wed 4 Nov 2026, 23:58 (single PDF on Aula Global, with the link to this repository, which must be public by then).

## Setup

```bash
conda env create -f environment.yml
conda activate statphys_lab1
python main.py      # reproduces every figure and table of the report
pytest              # runs the unit tests
```

In VS Code, select the `statphys_lab1` interpreter (Ctrl+Shift+P → *Python: Select Interpreter*).

## Repository layout

| Path | Content | Lab questions |
|---|---|---|
| `src/microstates.py` | Bitwise enumeration of the $2^N$ microstates (generators), benchmark | 1a, 1b |
| `src/combinatorics.py` | $\Omega(N,k)$ exact (integers) and in log-space (log-gamma) | 1c, 1d |
| `src/thermodynamics.py` | Probabilities, entropy changes, waiting times, likelihood ratios | 2a–2i |
| `tests/` | `pytest` unit tests (log-space vs exact integers, sums, symmetry) | — |
| `notebooks/` | Exploratory work only; final results come from `main.py` | — |
| `report/` | LaTeX source (`main.tex`); `main.py` writes vector figures (`.pdf`) to `report/figures/` | 1e, 2j, 2k + all |
| `AI_STATEMENT.md` | Log of every AI query, its output and what we changed | — |

## Workflow (required by the lab guidelines)

1. Never push to `main`. Start every task from an up-to-date `main` on its own branch:
   ```bash
   git switch main && git pull
   git switch -c feature/<task>      # e.g. feature/bitwise-enumeration
   ```
2. Commit small, meaningful steps (`git add <files>`, `git commit -m "Add popcount histogram"`) and push the branch: `git push -u origin feature/<task>`.
3. Open a Pull Request into `main` on GitHub. **Another team member reviews and approves it** before it is merged.
4. Any use of AI goes into `AI_STATEMENT.md` in the same PR (prompt, output, what we changed, how we verified it).
5. Before the submission: all PRs merged, `pytest` green, `python main.py` regenerates all figures, tag `v1.0`, repository made public.
