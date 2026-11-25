# Agent & Research Workflow (`marcelolx/math`)

This document codifies the standing scientific, verification, peer-review, and publication rules for every existing and future research stream in this repository. Any human contributor or AI agent working in `marcelolx/math` should follow these gates end-to-end.

> **Methodology reference:** Several structural and publication-gate practices below (audit receipts, verification tiers, `CITATION.cff`, `HISTORY.md`, and automated repo checks) are adapted from [`1337isnot1337/solved-conjectures`](https://github.com/1337isnot1337/solved-conjectures).

## 1. Branches & Publication Discipline

- **`main` is the reviewed publication branch** (and feeds the GitHub Pages log in `docs/`). Promote entries to `main` only after verification scripts run cleanly, figures are inspected, and—for headline entries—an independent cross-model audit receipt is recorded and hardened.
- For open multi-entry campaigns or unreviewed draft explorations, work on a `research` or `review/<topic>` branch and promote narrowly scoped, verified packages to `main`.
- Never commit unrun, broken, or placeholder results as verified. Never push to `origin/main` without explicit user authorization.

## 2. Verification Tiers (Honest Labeling)

Every entry in the root catalogue and each topic `README.md` carries an explicit verification badge:

| Badge | Label | Meaning |
| :---: | :--- | :--- |
| ✅ | `verified (audited)` | Reproducible script runs cleanly, metrics/figures match prose, and an independent cross-model peer review (`reviews/`) has been completed and hardened. |
| 🟡 | `verified (pending audit / conditional)` | Locally verified with runnable code and metrics, but either awaiting independent peer review/hardening or conditional on an explicit modeling closure (state scope clearly). |
| ⬜ | `draft / in-progress` | Exploratory work, partial result, or unreviewed candidate. |

## 3. Mandatory Per-Entry Scientific Loop

For every new entry (`<topic>/entries/XXX-short-title.md`) in any stream:

1. **Shrink the target & state assumptions:** Use [`templates/progress-block.md`](templates/progress-block.md). State exact governing equations, boundary conditions, parameter ranges, and intentional reductions.
2. **Independent analytical/spectral anchor:** Whenever possible, pair time-domain PDE/ODE solvers with an exact analytical law, dispersion root, conservation law, or spectral eigen-solver.
3. **Numerical rigor checklist (lessons from Peer Reviews #1–#7):**
   - Check machine-precision steady-state preservation and inviscid conservation where applicable.
   - Always include a **per-mode, per-grid numerical-damping table** for multi-mode PDE solvers so numerical dissipation never masks weak higher-harmonic growth.
   - Measure linear growth rates strictly in the linear amplitude regime ($A_0 \le 10^{-4}$ or $p'/\bar{p} < 0.5\%$), never contaminated by nonlinear saturation or subcritical terms.
   - Match simulation horizons to the slowest physical mode time scale ($\tau = 1/|\alpha_{\min}|$).
   - Inspect every generated figure visually before committing.
4. **Independent cross-model peer review:**
   - Dispatch a read-only audit using a different model when available (`git status -s` must stay clean; auditors run `uv run python -B` and keep scratch files under `/tmp/`).
   - Archive the review receipt in [`reviews/`](reviews/README.md) with reviewer model, verdict, findings, and resolution commit.
   - Harden code, metrics, figures, prose, and `<topic>/SYNTHESIS.md` against every Critical and Moderate finding before finalizing.
5. **Index & validation gate before commit/push:**
   - Update `<topic>/README.md`, `<topic>/SYNTHESIS.md`, root [`README.md`](README.md), and [`HISTORY.md`](HISTORY.md) together—never leave stale entry or review counts.
   - Run the repository consistency validator and require exit code `0`:
     ```bash
     uv run python scripts/check_repo.py
     ```

## 4. Python & Repository Hygiene

- Manage all Python dependencies via `uv` (`pyproject.toml` / `uv.lock`).
- **Never** create a `math.py` file or `math/` package directory inside the repository (prevents shadowing Python's standard library `math` module).
- Keep large temporary scratch data out of git; commit only lightweight reproducible scripts, `results/*_metrics.json`, and compact summary figures.
