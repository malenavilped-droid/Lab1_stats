# AI Collaboration & Audit Statement

Every query to an AI tool is logged here: the prompt, what the AI produced, and what we changed or
how we verified it. All AI-generated logic is backed by unit tests in `tests/`.

| # | Date | Member | Tool | Prompt (summary) | What the AI produced | What we changed / how we verified it |
|---|------|--------|------|------------------|----------------------|--------------------------------------|
| 1 | 2026-10-30 | Malena | Claude | Asked for the steps to set up the repository according to Lab_Project.pdf | Setup steps; `environment.yml`, `pytest.ini`, `.gitignore` additions, folder skeleton, `README.md` | Checked the files against Lab_Project.pdf (Phases 1, 2 and 4); built the Conda env locally |
