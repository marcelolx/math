# math

> **Open-notebook computational physics & engineering research.** Every entry pairs exact analytical or spectral benchmarks with reproducible Python verification code, explicit scope limits, documented dead ends, and independent cross-model adversarial peer reviews.
>
> **Progress Blog (GitHub Pages):** [marcelolx.github.io/math](https://marcelolx.github.io/math/) · [Agent Workflow](AGENTS.md) · [Peer Review Receipts](reviews/README.md) · [History](HISTORY.md) · [Citation](CITATION.cff)

## What This Repo Is

An open, general-purpose research notebook for tackling narrow, verifiable problems across mathematics, physics, and engineering. Every investigation follows a strict local **write → run → verify → cross-model audit → harden** loop (codified in [`AGENTS.md`](AGENTS.md)):

1. **Shrink the target** to a specific, testable mathematical or physical sub-problem.
2. **State the math, assumptions & checked scope** explicitly (including intentional reductions like 1D/2D models).
3. **Verify locally** with runnable simulation scripts (Python via `uv`) or formal proofs, backed by conservation, per-mode numerical-damping, and grid-convergence audits.
4. **Adversarially peer-review & archive** the audit receipt under [`reviews/`](reviews/README.md), harden every Critical/Moderate finding, and gate commits with `uv run python scripts/check_repo.py`.

## Verification Tiers

| Badge | Status | Meaning |
| :---: | :--- | :--- |
| ✅ | Audited & Hardened | Reproducible code + metrics verified, independent cross-model peer review completed and hardened. |
| 🟡 | Verified (Hardening / Conditional) | Locally verified with runnable code; either hardening in progress or conditional on a documented closure. |
| ⬜ | Draft / Exploratory | Unreviewed candidate or work-in-progress. |

## Research Tracks & Complete Catalogue

### 1. Rocket Combustion Instability ([`rocket-combustion/`](./rocket-combustion/) · [`SYNTHESIS.md`](./rocket-combustion/SYNTHESIS.md))

| Entry | Title | Verification | Audit |
| :--- | :--- | :---: | :---: |
| [`000`](rocket-combustion/entries/000-baseline.md) | 1D Longitudinal Combustor Baseline (Crocco $n$–$\tau$ + Choked Nozzle) | ✅ | Review #1 (`72e5c13`) |
| [`001`](rocket-combustion/entries/001-spatial-vs-timelag.md) | Finite Flame Width vs. Causal Time-Lag Dispersion | ✅ | Review #1 (`72e5c13`) |
| [`002`](rocket-combustion/entries/002-helmholtz-cavity.md) | Passive Acoustic Damping: Faceplate Helmholtz Resonator & EP Tuning | ✅ | Review #1 (`72e5c13`) |
| [`003`](rocket-combustion/entries/003-nonlinear-triggering.md) | Nonlinear Triggering (Bounded Subcritical Flame) & Calibrated Orifice Drag | ✅ | Review #1 (`72e5c13`) |
| [`004`](rocket-combustion/entries/004-2d-transverse-baffles.md) | 2D Transverse / Spinning Tangential Modes ($1T, 2T, 3T$) & Radial Baffles | ✅ | Review #2 (`074ac25`) |
| [`005`](rocket-combustion/entries/005-baffle-free-2d.md) | Baffle-Free 2D Stabilization: Dual-Tuned Cavity Rings & Azimuthal Staggering | ✅ | Review #2 (`074ac25`) |
| [`006`](rocket-combustion/entries/006-2d-nonlinear-modes.md) | 2D Nonlinear Spinning vs. Standing Selection & $t = 900$ Cavity Bomb Tests | ✅ | Reviews #2–#3 (`a3a1957`) |
| [`007`](rocket-combustion/entries/007-compressible-euler-steepening.md) | 1D Compressible Euler Shock Steepening, Mode-Selection Reversal & Cavities | ✅ | Review #5 (`ea4f8f3`) |
| [`008`](rocket-combustion/entries/008-chiral-shock-competition.md) | Chiral Swirl vs. Shock Competition, Even-$|u|$ & Rotating-Shock Control | ✅ | Review #5 (`ea4f8f3`) |
| [`009`](rocket-combustion/entries/009-2d-euler-throughflow.md) | 2D $(x, \theta)$ Compressible Euler Annular Combustor with Axial Throughflow | ✅ | [Review #6](reviews/peer-review-06-rocket-009.md) (`2a294f4`) |
| [`010`](rocket-combustion/entries/010-2d-throughflow-baffles-cavities.md) | 2D Euler Throughflow with Protruding Baffles & Wide-Mouth Faceplate Cavities | ✅ | [Review #7](reviews/peer-review-07-rocket-010.md) (`47ceabc`) |
| [`011`](rocket-combustion/entries/011-2d-viscous-ns-baffle-tips.md) | 2D Compressible Navier–Stokes: Vorticity Attenuation & No-Slip Baffle Wakes | ✅ | [Review #8](reviews/peer-review-08-rocket-011.md) (`8804d53`) |
| [`012`](rocket-combustion/entries/012-2d-transonic-cd-nozzle.md) | 2D Thin-Shell Transonic CD Nozzle: Bell Evanescent Collapse & `120×300` $2T$ Cascade | ✅ | [Review #9](reviews/peer-review-09-rocket-012.md) (`d5c03b3`) |
| [`013`](rocket-combustion/entries/013-dual-retuning-viscous-bell.md) | Dual-Harmonic Baffle–Cavity Retuning & Coupled 2D NS Transonic Bell Nozzle | ✅ | [Review #10](reviews/peer-review-10-rocket-013.md) (`e8e5796`) |
| [`014`](rocket-combustion/entries/014-3d-cylindrical-bell-bessel.md) | Full 3D Cylindrical $(x, r, \theta)$ Chamber & Bell Nozzle: Bessel Cutoff, Chiral $1T\to 1R$ & Shroud/Hub Cavities | 🟡 | Pending Review #11 |

### 2. Plasma Nozzles & MHD Propulsion ([`plasma-nozzles/`](./plasma-nozzles/) · [`SYNTHESIS.md`](./plasma-nozzles/SYNTHESIS.md))

| Entry | Title | Verification | Audit |
| :--- | :--- | :---: | :---: |
| [`000`](plasma-nozzles/entries/000-baseline.md) | Quasi-1D Paraxial Magnetic Nozzle Baseline & Alfvén Detachment Optimum | ✅ | Review #3 (`e115a12`) |
| [`001`](plasma-nozzles/entries/001-mhd-kelvin-helmholtz.md) | 2D Resistive-MHD Kelvin–Helmholtz Shear: Stuart-Number Law & $\mathrm{Re}$-Audit | ✅ | Review #3 (`e115a12`) |
| [`002`](plasma-nozzles/entries/002-self-consistent-nozzle.md) | Compliant Flux-Tube MHD Nozzle: Diamagnetic Inflation & Tube-Speed Choking | ✅ | Review #3 (`e115a12`) |

### 3. Power Grid Synchronization & Renewable Stability ([`grid-stability/`](./grid-stability/) · [`SYNTHESIS.md`](./grid-stability/SYNTHESIS.md))

| Entry | Title | Verification | Audit |
| :--- | :--- | :---: | :---: |
| [`000`](grid-stability/entries/000-baseline.md) | Nonlinear AC Swing Baseline, Exact 4-Bus Braess's Paradox & GFM Cliff | ✅ | Review #4 (`cf452d3`) |
| [`001`](grid-stability/entries/001-gfm-placement-and-saturation.md) | Spatial Placement of GFM Inverters: Import Asymmetry & Droop Saturation | ✅ | Review #4 (`cf452d3`) |

## Repository Layout

```text
math/
├── README.md               # Top-level catalogue & methodology overview
├── AGENTS.md               # Standing research, review & publication workflow
├── HISTORY.md              # Chronological milestone & audit log
├── CITATION.cff            # Citation metadata (GitHub "Cite this repository")
├── LICENSE
├── docs/                   # GitHub Pages progress blog (marcelolx.github.io/math)
├── reviews/                # Archived independent cross-model peer review receipts
├── scripts/
│   └── check_repo.py       # Automated link, metadata, index & JSON validation gate
├── templates/
│   └── progress-block.md   # Standardized entry template for new investigations
└── <topic>/
    ├── README.md           # Track index with verification badges
    ├── SYNTHESIS.md        # Consolidated design laws, dead ends & open problems
    ├── entries/            # Numbered Progress Blocks (000-baseline.md, ...)
    ├── sim/                # Runnable Python verification scripts (or Lean files)
    └── results/            # Lightweight plots and *_metrics.json outputs
```

> **Note for Python scripts:** Never name any file `math.py` or package directory `math/` inside the repo so Python's standard library `math` module is never shadowed.

## Running Simulations & Validation Locally

Python dependencies are managed with [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync

# Run repository-wide consistency gate (links, metadata, JSON, docs assets):
uv run python scripts/check_repo.py

# Run any entry's verification script:
uv run python rocket-combustion/sim/exp009_2d_euler_throughflow.py
uv run python rocket-combustion/sim/exp010_2d_throughflow_baffles_cavities.py
```

## References & Methodology Inspiration

Our repository structure, open-notebook publishing discipline, and automated validation workflow draw inspiration from the following reference projects (kept listed here so we can revisit their conventions as this project evolves):

- [`1337isnot1337/solved-conjectures`](https://github.com/1337isnot1337/solved-conjectures) — Curated open mathematical discovery repository (combinatorics & number theory, Lean 4 + Python verification). Directly inspired our adoption of [`AGENTS.md`](AGENTS.md), archived audit receipts ([`reviews/`](reviews/README.md)), honest verification-tier badges (`✅`/`🟡`/`⬜`), [`CITATION.cff`](CITATION.cff), [`HISTORY.md`](HISTORY.md), and automated pre-commit validation ([`scripts/check_repo.py`](scripts/check_repo.py)).

## License

- **Code & simulations:** [MIT License](./LICENSE)
- **Written notes, derivations & figures:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
