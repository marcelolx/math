# XXX — Short Descriptive Title

- **Date:** YYYY-MM-DD
- **Status:** `in-progress` | `verified` | `partial` | `negative-result` | `dead-end`
- **Verification Tier:** `⬜ Draft` | `🟡 Verified (Pending Audit)` | `✅ Verified & Audited`
- **Verification Script:** `../sim/<script-name>`
- **Peer Review Receipt:** *(link to `../../reviews/peer-review-XX-<topic>-XXX.md` once audited)*

## 1. Target

The exact, narrow mathematical or physical question tested in this session.

## 2. The Math, Assumptions & Checked Scope

- Governing equations / definitions used.
- Intentional simplifications (e.g., 1D/2D reduction, inviscid flow, linearized regime, toy lattice).
- Boundary / initial conditions, parameter ranges, and explicit scope limits.

## 3. The Verification

How to reproduce the run:

```bash
# Exact command to reproduce
uv run python <topic>/sim/<script-name>
```

Raw results (key metrics, plots saved in `../results/`, per-mode numerical-damping / grid-convergence checks, or proof-checker output):

- Metric / observation 1
- Metric / observation 2

## 4. Verdict & Dead Ends

- **What worked:** ...
- **What broke / dead ends:** ...
- **Next single hypothesis to test:** ...
