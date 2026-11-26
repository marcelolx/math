# Rocket Combustion Instability

Investigating thermoacoustic combustion instability in rocket chambers—modeling the feedback loop between acoustic pressure waves and unsteady heat release (Rayleigh criterion) to explore stabilization mechanisms, geometric damping, nonlinear triggering, and 1D/2D compressible Euler shock dynamics.

> **Consolidated Monograph:** See [`SYNTHESIS.md`](./SYNTHESIS.md) for the unified quantitative design laws, documented dead ends, and verified scope across Entries `000`–`013`. Audit receipts are archived in [`../reviews/README.md`](../reviews/README.md).

## Entries

| Entry | Title | Verification | Peer Review |
| :--- | :--- | :---: | :---: |
| [000-baseline.md](./entries/000-baseline.md) | 1D Longitudinal Combustor Baseline (Crocco $n$–$\tau$ + Choked Nozzle) | ✅ | Review #1 (`72e5c13`) |
| [001-spatial-vs-timelag.md](./entries/001-spatial-vs-timelag.md) | Finite Flame Width vs. Causal Time-Lag Dispersion | ✅ | Review #1 (`72e5c13`) |
| [002-helmholtz-cavity.md](./entries/002-helmholtz-cavity.md) | Passive Acoustic Damping: Faceplate Helmholtz Resonator & Exceptional-Point Tuning | ✅ | Review #1 (`72e5c13`) |
| [003-nonlinear-triggering.md](./entries/003-nonlinear-triggering.md) | Nonlinear Triggering (Bounded Subcritical Flame) & Calibrated Cavity Orifice Drag | ✅ | Review #1 (`72e5c13`) |
| [004-2d-transverse-baffles.md](./entries/004-2d-transverse-baffles.md) | 2D Transverse / Spinning Tangential Modes ($1T, 2T, 3T$) & Radial Faceplate Baffles | ✅ | Review #2 (`074ac25`) |
| [005-baffle-free-2d.md](./entries/005-baffle-free-2d.md) | Baffle-Free 2D Stabilization: Dual-Tuned Cavity Rings ($1/\omega^2$ Law) & Azimuthal Lag Staggering | ✅ | Review #2 (`074ac25`) |
| [006-2d-nonlinear-modes.md](./entries/006-2d-nonlinear-modes.md) | 2D Nonlinear Spinning vs. Standing Mode Selection & Verified Nonlinear Cavity Bomb Tests | ✅ | Reviews #2–#3 (`a3a1957`) |
| [007-compressible-euler-steepening.md](./entries/007-compressible-euler-steepening.md) | Compressible Nonlinear Euler Transverse Acoustics: Shock Steepening, Mode-Selection Reversal & Cavity Cascade Absorption | ✅ | Review #5 (`ea4f8f3`) |
| [008-chiral-shock-competition.md](./entries/008-chiral-shock-competition.md) | Chiral Swirl vs. Gasdynamic Shock Competition, Even-$|u|$ Coupling & Rotating-Shock Control | ✅ | Review #5 (`ea4f8f3`) |
| [009-2d-euler-throughflow.md](./entries/009-2d-euler-throughflow.md) | 2D $(x, \theta)$ Compressible Euler Annular Combustor with Physical Axial Throughflow | ✅ | [Review #6](../reviews/peer-review-06-rocket-009.md) (`2a294f4`) |
| [010-2d-throughflow-baffles-cavities.md](./entries/010-2d-throughflow-baffles-cavities.md) | 2D Compressible Euler Throughflow with Protruding Baffles & Wide-Mouth Faceplate Cavity Rings | ✅ | [Review #7](../reviews/peer-review-07-rocket-010.md) (`47ceabc`) |
| [011-2d-viscous-ns-baffle-tips.md](./entries/011-2d-viscous-ns-baffle-tips.md) | 2D Compressible Navier–Stokes: Vorticity-Wave Attenuation, Baffle Convergence & No-Slip Wakes | ✅ | [Review #8](../reviews/peer-review-08-rocket-011.md) (`8804d53`) |
| [012-2d-transonic-cd-nozzle.md](./entries/012-2d-transonic-cd-nozzle.md) | 2D Thin-Shell Transonic CD Nozzle: Bell Evanescent Collapse, Entropy Filtering & `120×300` $2T$ Cascade | ✅ | [Review #9](../reviews/peer-review-09-rocket-012.md) (`d5c03b3`) |
| [013-dual-retuning-viscous-bell.md](./entries/013-dual-retuning-viscous-bell.md) | Dual-Harmonic Baffle–Cavity Retuning & Coupled 2D NS Transonic Bell Nozzle: Breaking the $1T \to 2T$ Cascade | ✅ | [Review #10](../reviews/peer-review-10-rocket-013.md) |
