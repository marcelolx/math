# Peer Review #11: Rocket Combustion Entry 014

**Entry:** `rocket-combustion/entries/014-3d-cylindrical-bell-bessel.md`
**Script:** `rocket-combustion/sim/exp014_3d_cylindrical_bell_bessel.py`
**Metrics:** `rocket-combustion/results/exp014_metrics.json`
**Synthesis:** `SYNTHESIS.md` Law #17 and Dead End #16
**Reviewer role:** independent adversarial CFD and acoustics audit
**Date:** 2026-11-26

## Verdict: **Major Revisions**

The 3D conical-frustum finite-volume operator is sound. The Bessel constants, the modal weights, the chiral selection ratio, and the micro-hub 1R advantage survive independent tests. But two claims in the headline Law #17 do not survive refinement:

1. The "3D PDE locks onto the vorticity-damped plateau" result is not valid. In the straight cylinder (Exp 1D/1E), the match is unconverged numerical dissipation: the 1T rate moves toward the inviscid value under axial refinement. In the bell (Exp 2B), the true-3D damping is robust (about `-0.045` to `-0.049` up to nx = 120), but a per-case tuned `Re_vort` produces the "match", and vorticity attenuation does not explain the gap.
2. The Exp 4C verdict "decays both spinning and standing 1T bombs" is false at t = 120. Both cases re-grow.

The report also found a modal-energy normalization bug (a factor of nth² = 2304), a fit-window mismatch in the Exp 4B Δα values, and one claim (`M_wall = 0.126`) with no supporting run.

All scripts and raw outputs are in `/tmp/peer_review_014/`. I imported the author's module read-only, without any modification. Where I needed a different output (for example, full time series), I wrote my own driver around the author's operators, or an independent copy of the disk solver (`disk.py`). I used `uv run python -B` throughout. `git status -s` is empty.

---

## Critical findings

### C1. The 3D straight-cylinder "vorticity-damped plateau" is unconverged numerical damping (Exp 1D/1E, Law #17)

The entry says the `24×12×48` straight PDE value (`-0.09333`) "locks onto" the `Re_vort = 1000` Chebyshev plateau (`-0.09121`, 2.3% match). It attributes the gap to the inviscid Chebyshev value (`-0.04239`) to physical vorticity attenuation. I refined only the axial grid (nr = 12, nth = 48, A₀ = 1e-4, M_c = 0.12, t = 32), keeping everything else identical (`jobs3d.py`, `analyze3d.py`):

| nx | α (1T), fit [12, 30] | fit [8, 20] | fit [16, 30] | 1R, fit [12, 30] |
| :---: | :---: | :---: | :---: | :---: |
| 24 (author grid) | **-0.09333** (reproduced exactly) | -0.09414 | -0.09314 | -0.15656 |
| 48 | -0.08740 | -0.10358 | -0.08275 | -0.15659 |
| 96 | **-0.05417** | -0.08992 | -0.04669 | -0.15996 |
| Inviscid Chebyshev (N = 96) | -0.04239 | | | -0.03589 |
| `Re_vort = 1000` Chebyshev | -0.09121 | | | -0.09509 |

* The 1T rate does not settle on a plateau. It moves monotonically toward the inviscid Chebyshev value as dx falls (`-0.0933 → -0.0874 → -0.0542`). At nx = 96 the late-window value (`-0.0467`) is within 10% of the inviscid result.
* On finer grids, fit-window sensitivity grows to 2× (`-0.090` vs `-0.047` at nx = 96). This shows a non-exponential (multi-wave) signal once the vorticity wave survives to the exit.
* The 2T rate at nx = 96 is `-0.0876` (vs `-0.1128` on nx = 24), so the same trend applies.
* The 1R rate does not change with nx, because radial resolution controls it (see C2 and M3).

**Conclusion:** the 2.3% "match" is a coincidence of one grid. Numerical dissipation of the inlet vorticity wave on `dx = 0.0625` (about 6.6 cells per λ_vort = 0.41) sets it. It does not confirm a physical attenuation law. The statement "finite vorticity dissipation rescues the inviscid cutoff collapse" and the "3.4–14.7× faster 3D vorticity attenuation" claim have no PDE support.

### C2. `Re_vort` is a free parameter tuned per case, so the Exp 2B "1.7% / 2.6% matches" are not predictions

The quasi-3D `Re_vort` term moves the bell 1T rate continuously over a factor of 25. I reproduced the author's three plateau values exactly (`cheb_checks.py`):

| `Re_vort` | Bell L_n = 0.60 | Bell L_n = 0.80 | Straight |
| :---: | :---: | :---: | :---: |
| 500 | -0.05612 | -0.05197 | -0.09161 |
| 1000 | -0.05307 | **-0.04468** | **-0.09121** |
| 2000 | **-0.04278** | -0.03351 | -0.08287 |
| 4000 | -0.02704 | -0.02152 | -0.06742 |
| 8000 | -0.01571 | -0.01299 | -0.05601 |
| ∞ | -0.00227 | -0.00265 | -0.04239 |
| 3D PDE (author) | -0.04460 (40×12) / -0.04206 (56×16) | -0.04583 (44×12) | -0.09333 |

* The L_n = 0.60 match uses `Re_vort = 2000`. The L_n = 0.80 and straight matches use `Re_vort = 1000`. At one common value (2000), the L_n = 0.80 bell differs by 27% (`-0.0335` vs `-0.0458`). At 1000, L_n = 0.60 differs by 26%.
* With one tuned knob per row, a percent-level "match" carries no evidential weight.
* N-convergence of the quasi-3D values is good (`-0.04279 / -0.04278 / -0.04277` for N = 64, 96, 128 at Re = 2000). So the knob is the issue, not the eigen-solver.

**Base drift (task checklist item):** I tested whether the `rhs_max = 4.19e-5` residual of the 56×16 base contaminates the A₀ = 1e-4 fit. It does not. With A₀ = 0 on the same base, E_1T stays at the round-off floor (`1.6e-42`) for all t ∈ [14, 33]. The axisymmetric drift cannot project onto e^{-iθ}. The E_1R floor (`1.4e-12`) is also four orders below the 1R signal. Linearity also holds: A₀ = 1e-3 gives `-0.04500` vs `-0.04460` at A₀ = 1e-4 (40×12).

**Fit-window sensitivity is as large as the reported grid change:**

| Run | [14, 33] (author) | [10, 25] | [20, 34] | [8, 18] |
| :--- | :---: | :---: | :---: | :---: |
| Bell 40×12, A₀ = 1e-4 | **-0.04460** | -0.04207 | -0.04306 | -0.04260 |
| Bell 40×12, A₀ = 1e-3 | -0.04500 | -0.04266 | -0.04335 | -0.04319 |
| Bell 80×12, A₀ = 1e-4 | -0.04648 | -0.04334 | -0.04420 | -0.04313 |
| Bell 120×12, A₀ = 1e-4 | -0.04908 | -0.04637 | -0.04644 | -0.04501 |
| Bell 1R 40×12, A₀ = 1e-3 | -0.08293 | -0.08335 | -0.08658 | -0.08528 |

The author's "40×12 → 56×16" change (`-0.0446 → -0.0421`, 6%) is the same size as the window spread on one grid (`-0.0421` to `-0.0446`). **Axial refinement of the bell behaves differently from the straight cylinder.** At nr = 12, the bell 1T rate becomes slightly *more* damped as nx rises (`-0.0446` → `-0.0465` → `-0.0491` for nx = 40, 80, 120; dx = 0.064 → 0.021, or 6 → 19 cells per λ_vort). It does not move toward the inviscid quasi-3D value (`-0.0023`). So the true-3D bell damping of about `-0.045` is robust to axial resolution over this range, and the 20× gap to the quasi-3D WKB model is real. But the attribution of this gap to vorticity attenuation has no support: the straight cylinder (C1) shows that resolved vorticity transport *removes* damping, while the bell gains damping. The more likely cause is the meridional physics that the quasi-3D model leaves out (base u_r, throat bowing, J_m(s r/R(x)) mode-shape coupling). That is exactly what Exp 2B was meant to audit. Report Exp 2B as "the quasi-3D WKB model underpredicts true-3D bell damping by about 20×", and do not claim agreement with a tuned `Re_vort` plateau.

**Required action:** remove the "lock-on" and "1.7% / 2.6% match" claims. Keep the robust result (true-3D bell 1T ≈ `-0.045` to `-0.049`, versus quasi-3D inviscid `-0.0023`) and add an nr refinement (nr ≥ 20) of the bell. Report the 3D PDE as unconverged with respect to the inviscid quasi-3D limit, or add a physical viscosity with one fixed Re for all cases.

### C3. Exp 4C "decays both spinning and standing 1T bombs" is false at t = 120

I extended the Exp 4C cases from t = 48 to t = 120 on the same grid (20×12×48, n_p = 0.30, A₀ = 0.04) and added the missing `nl_shroud_st1T` and `nl_bare_st1T` runs (`a4c.py`). The slope is the e_ch tail slope. P is the mean p′/p̄ peak over the last 9.6 time units.

| Case | Slope [28.8, 48] | Slope [48, 84] | Slope [84, 120] | P @ 48 | P @ 84 | P @ 120 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Bare spin 1T | +0.00177 | +0.00021 | +0.00001 | 9.62% | 9.79% | 9.80% (saturated LC) |
| Bare stand 1T *(new)* | +0.00158 | +0.00017 | +0.00004 | 7.24% | 7.49% | 7.70% |
| **Dual-zone spin 1T** | -0.00130 | -0.00146 | **+0.00156** | 4.89% | 5.20% | **6.45%** |
| **Dual-zone stand 1T** | -0.00288 | -0.00126 | **+0.00095** | 3.78% | 3.90% | **4.77%** |
| Shroud-only stand 1T *(new)* | -0.00604 | -0.00452 | **+0.01101** | 3.39% | 3.18% | 3.81% |
| Dual-zone spin 2T | +0.00004 | -0.00049 | -0.00134 | 5.37% | 5.43% | 5.65% |
| Shroud-only 1R | -0.00980 | -0.03098 | -0.05552 | 2.73% | 1.03% | 0.15% |
| Dual-zone 1R | -0.06298 | -0.02921 | -0.02835 | 0.024% | 0.006% | 0.002% |

* The dual-zone spinning-1T and standing-1T cases both turn to growth after t ≈ 84. The late peak p′/p̄ rises by 32% and 26%. The tail slopes `-0.0013` / `-0.0029` that the entry quotes are transients, not decay.
* In all three regrowing cavity cases, E_1T keeps falling while the total chamber energy rises. 2T, 1R, and 3T stay small (`late.py`). An untracked component (not in the {1T, 2T, 1R, 3T} projections, probably 1L or a mean-state drift) carries the regrowth. The entry does not track or report this component.
* "1R daughter killed!" (`nl_dualz_st1T`) had no measurement in the author script. My measurement at t = 48 gives E_1R,ac (true normalization) = `3.1e-7` for dual-zone, `1.07e-6` for shroud-only, and `1.66e-5` for bare. So the hub cuts the daughter 53× vs bare and 3.4× vs shroud-only. It reduces the daughter. It does not kill it.
* "Shroud-only 1R: Exposed" is not true on this grid. The coarse shroud-only 1R case decays to 0.15% by t = 120. See M3 for the finer grid, where the claim *does* hold.

---

## Moderate findings

### M1. Modal energies in the 3D solver carry a factor of nth² (2304×), and e_ch a factor of nth (48×)

In `simulate_3d_cylindrical`, `V[ch_mask]` has shape `(n_ch, nr, 1)`, so `w_ch = V_ch / sum(V_ch)` sums to 1 over one azimuthal slice only. When it broadcasts over `nth` cells, the total weight is `nth` (`t_shape.py`: sum = 48.0). Consequences:

* `e_1t, e_2t, e_1r, e_3t` are 48² = 2304× too large. `e_ch` is 48× too large.
* This explains the anomaly in the task checklist: `E_1T_late = 4.69` at `p_peak = 7.68%` is impossible physically. The true value is 4.69 / 2304 = **2.04e-3**, which agrees with the 2D disk at the same amplitude (E_1T ≈ 2.4e-3 at A₀ = 0.06).
* Ratios between runs on the same nth (427×, 20,217×, 5.27×, 3.91×, 2.36×) are not affected. Growth rates are not affected.
* The absolute values in the Exp 3B text and in the Exp 4C "Late E_wave" column are wrong (for example, bare 1T `8.05e-2` is really `1.68e-3`).
* Ratios such as e_1t / e_ch are off by a factor of nth.

**Fix:** use `w_ch = np.broadcast_to(V, V.shape[:2] + (nth,))[ch_mask]` normalized over all cells, and regenerate the JSON.

### M2. Exp 4B Δα values use a bare rate from a different run and fit window

The bare rates (`bare_cold_3d_alpha`) come from the Exp 1D runs (t = 32, stride 12, window [12, 30]). The cavity rates use t = 24, stride 10, window [7, 22]. My matched-window bare runs (`b4_*_c`, 24×12×48, stride 6):

* The 1R "2.0% uniform" run is **bit-identical** to bare (`-0.15643` in every window). This is expected, because ∫₀¹ J₀(s₀₂ r) r dr = J₁(s₀₂)/s₀₂ = 0 exactly, so a uniform port has zero 1R coupling (`analytic.py`: coupling = 0.0). The entry's ranking "micro-hub beats ... `2.0%` uniform (`-0.15649`)" compares with a no-op. Its apparent Δα = +0.0007 is only the window mismatch.
* The 1T "2.0% hub" (`-0.09400`) is also bit-equal to matched-window bare (`-0.09400`). Its quoted Δα = `-0.00068` is window noise, not "essentially zero damping".
* With a matched bare (`-0.09400`), the 0.5% shroud/uniform ratio is **1.462**, not 1.450. The 3-digit agreement with the discrete port ratio (1.4500, `analytic.py`) is partly a coincidence of the mismatch. The ratio itself is robust to about 1%: 1.456 on 24×20×64, and the discrete theory gives 1.472 at nr = 20 and 1.461 in the continuum.

### M3. Exp 4B and 4C 1R conclusions depend strongly on radial resolution and fit window

Linear 1R placement on 24×12×48 (matched [7, 22]) and on the finer 24×20×64 grid:

| Case | 12×48 α | 12×48 Δα | 20×64 α | 20×64 Δα | 20×64 Δα, window [14, 24] |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Bare 1R | -0.15643 | 0 | -0.10769 | 0 | 0 |
| 2.0% shroud (ξ = 0.14) | -0.15909 | -0.0027 | -0.11798 | -0.0103 | -0.0270 |
| 0.35% micro-hub (ξ = 0.14) | -0.17712 | -0.0207 | -0.13222 | **-0.0245** | -0.0357 |
| 2.0% hub (ξ = 0.38) | -0.17115 | -0.0147 | -0.12513 | -0.0174 | -0.0289 |
| 2.0% uniform | -0.15643 | 0 (exact) | -0.10769 | 0 (exact) | 0 |
| Bare 1T | -0.09400 | | -0.08791 | | |
| 0.5% shroud 1T | -0.12785 | -0.03385 | -0.12095 | -0.03304 | |
| 0.5% uniform 1T | -0.11715 | -0.02315 | -0.11060 | -0.02269 | |

* The micro-hub keeps first place on 1R on both grids and in every window. **Confirmed qualitatively.**
* The hub/shroud Δα advantage falls from 7.8× (coarse) to 2.4× (fine, [7, 22]) and 1.3× (fine, [14, 24]). The weak-coupling estimate β·c (normalized port coupling c from `coupling2.py`: hub 4.90, shroud 0.82 at nr = 20) gives 0.0171 vs 0.0164, about 1.04×. The large coarse-grid advantage is mostly a coarse-grid artifact.
* The bare 1R rate changes by 45% between grids (`-0.156 → -0.108`), so the "`-0.1572` bare 1R" baseline is mostly numerical.

The reacting 1R case on the finer grid (20×20×64, t = 48):

| Case | Slope [28.8, 48] | P @ 48 | E_ch (true norm.) |
| :--- | :---: | :---: | :---: |
| Bare 1R | +0.00364 | 9.18% (coarse grid: 4.90%) | 6.8e-4 |
| Shroud-only 1R | **+0.00463 (growing)** | **7.41%** | 4.6e-4 |
| Dual-zone 1R | -0.08497 | 0.106% (coarse grid: 0.024%) | 1.0e-7 |

On the finer grid, the shroud-only design fails on 1R and the dual-zone hub still quenches it (about 6,800× in E_ch vs bare). This supports the dual-zone motivation more strongly than the coarse run does. But the "20,217×" factor and the coarse "4.90%" bare amplitude depend on the grid. Report them with the nr = 20 values.

### M4. The 3D vs. 2D thin-shell Hopf comparison depends on the retuned flame lag `TAU_BAR`

The entry lowers `TAU_BAR` from 0.40 to 0.16 "so 3D modes sit inside the active Rayleigh band". It then applies the same τ to the 2D thin-shell reference. Because the 2D thin-shell mode (ω ≈ 1) then sits out of the band, part of the "3.93× lower Hopf" comes from flame-phase alignment, not from the Bessel cutoff collapse (`tau_check.py`, L_n = 0.50, N = 96):

| τ̄ | M_c | 3D 1T n_pH | 2D 1T n_pH | 2D 2T n_pH | 2D min / 3D 1T |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0.16 | 0.12 | 0.0366 | 0.1203 | 0.0408 | 1.12 |
| 0.16 | 0.14 | 0.0504 | 0.1998 | 0.1982 | **3.93** |
| 0.40 | 0.12 | 0.0579 | 0.1372 | 0.0471 | **0.81** (3D more stable) |
| 0.40 | 0.14 | 0.0578 | 0.1944 | 0.2365 | 3.36 |

The 2D thin-shell numbers in this entry also differ from Entry 012's (for example, Entry 012 reports 0.060 at M_c = 0.12, here 0.0408). The cold-damping ratio (4.77× at L_n = 0.50) does not depend on τ and stands. Restate the Hopf claims as "at τ̄ = 0.16".

### M5. "Spinning-1T AC 1R is a small discrete-grid floor" is wrong: it is a physical second-order effect

I extended the disk sweep to small amplitudes and later windows (`disk.py`, `analyze_disk.py`). My driver reproduces the author's table exactly (for example, stand A₀ = 0.08: E_1R,ac = `1.106e-4`, spin `1.797e-7`).

| A₀ | Spin E_1R,ac | Stand E_1R,ac | Ratio [6.4, 16] | Ratio [10, 16] | Ratio [16, 32] |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0.005 | 4.11e-12 | 2.48e-9 | 603 | 807 | 967 |
| 0.01 | 6.54e-11 | 3.95e-8 | 604 | 809 | 958 |
| 0.02 | 1.02e-9 | 6.21e-7 | 607 | 813 | 928 |
| 0.04 | 1.50e-8 | 9.24e-6 | 614 | 820 | 805 |
| 0.08 | 1.80e-7 | 1.11e-4 | 615 | 770 | 399 |
| Fitted slope (all 5) | **3.87** | **3.88** | | | |
| Slope, A₀ ≤ 0.02 | | **3.99** | | | |

* The spinning AC 1R scales as A₀^3.87, the same as the standing case. A grid-noise floor would not scale with amplitude at all. Its most likely source is the O(A₀²) mismatch between the linear eigenmode IC and the nonlinear rotating wave (start-up transient), plus genuine second-order coupling through the finite-amplitude meridional field. It is physical, not discrete.
* The quartic law is **confirmed** at small amplitude (slope 3.99 for A₀ ≤ 0.02). The author's 3.67 comes from the large-A₀ points (0.08 to 0.10).
* The selection ratio is robust across grids and limiters: 615 (24×96), 579 (32×128), **552 (48×192)**, and 613 (24×96 minmod). It depends on the window (600 to 970 at small A₀). Quote it as "about 600× in window [6.4, 16]", not as an exact band.
* Stand E_1R,ac / E_1T at A₀ = 0.08 keeps rising with the grid: 2.71% → 2.98% → **3.18%** (48×192). It is not yet converged.

### M6. Unsupported claim: steep-bell separation (`M_wall = 0.126`)

Dead End #16(b) and §4 state that the Entry 012 L_n = 0.50 bell separates in true 3D axisymmetric flow (`M_wall = 0.126`). The script never relaxes a `bell_c2` base (the `mode="bell_c2"` branch is unused), and the JSON has no such entry. Either add the run and its metric, or remove the claim.

### M7. The quasi-3D "1R" curve hops between modal branches, and the L_n = 0.05 row is not N-converged

* The L_n = 0.60 1T value (`+0.00042`) is a clean, isolated, N-converged eigenvalue (`+0.000420, +0.000418, +0.000419, +0.000420` for N = 64, 96, 128, 160; next candidate `-0.069`). It is a genuine property of the inviscid quasi-3D model. **Confirmed.** (It is not a physical whistle, because of C1 and C2.)
* The L_n = 0.40 "1R" value (`+0.00295`) is also converged, but it sits at ω = 4.357. The branch near s₀₂ = 3.81 has α = `-0.036`. The "highest real part in a wide window" rule switches branch along L_n (ω = 3.81, 4.35, 3.86, 4.36, 4.27, 3.87, 4.31). The plotted 1R curve is an envelope of several axial overtones, not one mode.
* The L_n = 0.05 row is not N-converged. 1R: `+0.0460` (N = 64), `+0.00012` (N = 96, JSON), `-0.0297` (N = 128), `-0.0312` (N = 160). 1T: `+0.0133 / -0.0120 / -0.0352 / -0.0375`. 2T: `+0.0512 / -0.0096 / -0.0531 / -0.0554`. All three JSON values at L_n = 0.05 are wrong by 0.02 to 0.05. The figure plots them.
* The 1R and 3T values at L_n = 0.50 used in the Hopf sweep are converged (`-0.014219`, `-0.004239` for N = 96 and 128).
* `n_pH_1R = 0.9` at M_c = 0.16 is a cap, not a value. The real part goes from `-0.0264` (cold) to `-0.0069` at n_p = 0.9 and then to `-0.0096` at n_p = 2.0 (it is non-monotonic and never crosses). Report it as "no Hopf crossing for n_p ≤ 2", not as 0.9.
* Missing modes: 4T (s₄₁ = 5.318), 1T1R (s₁₂ = 5.331), and 2T1R (s₂₂ = 6.706) have n_pH between 0.161 and 0.432 for M_c = 0.10 to 0.22. These are all above the reported global values, so the global envelope **is confirmed** for this mode set.

---

## Minor findings

* **m1.** "Detuned 2T stays `< 1.12%`": the A₀ = 0.10 value is 1.1205%. Write "≤ 1.12%".
* **m2.** Uniform 0.5% Δα is `-0.023829`, which rounds to `-0.02383`, not `-0.02382`.
* **m3.** The M_c = 0.28 net value `-0.09029` subtracts the 12×48 closed-cylinder damping from a 40×14×48 run. Use matched nr and nth.
* **m4.** The GCL check `(ΔS_x) - (ΔS_ηx) = 0` is an algebraic identity of the chosen formulas. It holds for any R (4.2e-17 for random R in `analytic.py`). It verifies the code, not the geometry. The actual geometric metrics use Simpson quadrature. The maximum relative error vs. exact quadrature on the cosine bell is 2.5e-4 for V and 1.1e-4 for r̄. This is consistent and well-balanced, but it is not "exact".
* **m5.** The radial-face and θ-face radial projections use Δθ instead of the Cartesian-exact 2 sin(Δθ/2). Both sides use the same factor, so well-balancing is exact. Radial momentum has an O(Δθ²) consistency error (0.07% at nth = 48). This is standard cylindrical FV and is acceptable, but "exact" overstates it.
* **m6.** "Proved the 3D EP scaling law ξ_H^EP ∝ √(βW)": the runs test two ξ values per configuration. They do not sweep ξ to locate the EP. Write "consistent with", not "proved".
* **m7.** Exp 3B (427×) uses only one amplitude, one n_p, and t = 26. With M1 fixed, report the true E values.

---

## Confirmed

| Item | Independent result |
| :--- | :--- |
| Bessel zeros s₁₁, s₂₁, s₀₂, s₃₁, s₁₂ and ratios 1.6588, 2.0811 | Confirmed |
| W_mn(1), W_mn(0) (1.4184, 1.7507, 1.0000, 2.0405; 6.1647) and normalization (1/π)∫∫W r dr dθ = 1 | Confirmed (quadrature: 1.000000) |
| Frustum metrics V, r̄, S_x, S_ηr, S_ηx, S_θ, r_ef, r_xf, r_θf, hoop | Confirmed analytically (surface-normal derivation). Simpson error ≤ 2.5e-4 (m4) |
| `nozzle_R_smooth_3d` is C² at the throat: R'' = ½π²(1 - R_t)/L_n² = k_t on both sides, and atan(0.28) = 15.64° | Confirmed analytically |
| Angular momentum r ρ u_θ: flux form with face radii, no source | Confirmed conservative |
| Hoop source (p + ρw²)·hoop = (p + ρw²) ∫ dV / r | Confirmed for cell-uniform state (m5) |
| Antipodal centerline ghost: flips u_r and u_θ, keeps u_x | Confirmed correct for even nth |
| Outer wall mirror v_n,R = -v_n,L, tangential copied | Confirmed: HLLC gives s* = 0 and zero mass flux |
| Cavity coupling: discrete adjointness Σ p̄_k q_k = Σ_cells Δp u_port dA; cavity energy balance dE_c/dt = Σ q p̄ - 2ξω q²/(ω²V) | Confirmed analytically (passive, adjoint) |
| `inlet_choked_3d`: ρ(u + u_port) = M_c, isentropic, zero swirl; sign consistent with q = inflow to cavity | Confirmed |
| Bare 1T straight PDE (24×12×48) `-0.09333` | Reproduced exactly |
| Disk cascade table (24×96) | Reproduced exactly |
| Chiral selection about 600× (grid- and limiter-robust: 552 to 615) | Confirmed with window caveat (M5) |
| Quartic A₀⁴ law | Confirmed at small A₀ (slope 3.99) |
| L_n = 0.60 quasi-3D 1T `+0.00042`, L_n = 0.50 `-0.00513`, 4.77× vs thin-shell | Confirmed and N-converged (model result) |
| Base-drift contamination of the 56×16 bell 1T fit | Ruled out (E_1T floor 1.6e-42) |
| Bell 1T linearity (A₀ = 1e-4 vs 1e-3) | Confirmed (0.9%) |
| 0.5% shroud/uniform ≈ 1.45 | Confirmed to about 1% (1.456 to 1.462) |
| Micro-hub beats 2% shroud and 2% hub on 1R | Confirmed on 24×20×64 (smaller margin, M3) |
| Dual-zone 1R quench, reacting | Confirmed on nr = 20, where shroud-only fails |
| Global 3D Hopf envelope vs. 4T, 1T1R, 2T1R | Confirmed (those modes have n_pH ≥ 0.161) |
| All other numbers in the entry and in SYNTHESIS #17 / DE #16 vs. JSON | Programmatic scan (`prose.py`) plus manual check of each derived ratio: consistent, except `0.126` (M6), m1, m2 |

---

## Required revisions (summary)

1. Remove the "vorticity-damped plateau lock-on" claims in Exp 1D/1E, Exp 2B, Law #17, and the takeaways. Add an nx-refinement table (C1, C2).
2. Extend Exp 4C to t ≥ 120 and correct the 1T and standing-1T verdicts. Identify the untracked growing component. Measure the 1R daughter instead of asserting it (C3).
3. Fix the `w_ch` normalization and regenerate all absolute energies (M1).
4. Use matched bare runs and windows for every Δα. Remove the uniform-1R and hub-1T "damping" entries (M2).
5. Report the 1R placement results on nr ≥ 20 (M3), restate Hopf claims at fixed τ̄ (M4), restate the spin-AC floor as physical second order (M5), remove or support `M_wall = 0.126` (M6), and fix the L_n = 0.05 row and the 1R branch labelling (M7).

## Reproduction

All in `/tmp/peer_review_014/`:
`common.py` (read-only import), `t_shape.py` (M1), `analytic.py` and `coupling2.py` (geometry, weights, coupling), `cheb_checks.py` and `tau_check.py` (Chebyshev N-convergence, mode hopping, missing modes, Re_vort and τ̄ sweeps), `jobs3d.py`, `bell120.py`, `analyze3d.py`, `a4c.py`, `late.py` (3D PDE refinement and long runs), `disk.py` and `analyze_disk.py` (cascade), `prose.py` (prose vs. JSON). Raw time series: `runs/*.npz`.
