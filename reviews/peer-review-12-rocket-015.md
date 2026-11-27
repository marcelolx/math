# Peer Review #12: Rocket Combustion Entry 015

**Entry:** `rocket-combustion/entries/015-3d-ns-partial-baffles-chugging.md` (commit `c9ad29f`)
**Script:** `rocket-combustion/sim/exp015_3d_ns_baffles_bell.py`
**Metrics:** `rocket-combustion/results/exp015_metrics.json`
**Synthesis items:** Law #18, Dead End #17
**Reviewer:** independent, adversarial, read-only (Claude Opus 5.5)
**Date:** 2026-11-27

## Verdict: **Major Revisions**

The 3D NS solver is conservative and mostly correct. The curved-wall stress-free Stokes–Kirchhoff (SK) shift is exact to leading order. The 1D chugging eigen-solver is exact. But three headline claims do not survive independent audit:

1. The straight-cylinder "fixed-Re convergence to the exact Bessel-NS Chebyshev solver" is a coincidence. The PDE and the Chebyshev model disagree by 2.4× at Re = ∞, and the PDE has no resolved viscous increment (C1).
2. The "2.57–2.90×" partial-span 2T boost is a fit-window artifact of a two-peak beating signal. A robust total-energy measure gives 1.5–2.2×, and the boost is not viscous (C2).
3. Mass-conserving `m_dot` cavities do not cure the k = 0 drift. Extended to t = 180, the `ω_hp = 0.15` mass-conserving case grows into full k = 0 chugging (|p̄'| ≈ 6.5%, mass excursion 3.5%) and extinguishes 1T. The 1D eigen model does not describe this oscillation (C3).

All independent scripts and raw outputs are in `/tmp/peer_review_015/` (`exact_sf_disk.py`, `cheb_check.py`, `cheb_resweep.py`, `cheb_bc_test.py`, `metric_check.py`, `chug_shoot.py`, `exp015_instr.py` (diagnostic-only instrumented copy), `exp015_fixed.py` (metric-corrected copy), `batch_*.py`, `analyze.py`, `out/*.pkl`). The repository stayed unmodified (`git status -s` is empty).

---

## Critical findings

### C1. The straight-cylinder "fixed-Re convergence to exact 1D Bessel-NS Chebyshev" (`4.1–5.5%`) is coincidental. Entry 014 C1/C2 is not closed.

**Evidence (independent runs, fit window [8, 22] on `e_1t`, `20/24/48 × nr × 48`, A = 1e-4):**

| Model | Re = ∞ | Re = 500 | Re = 1000 | Viscous increment (Re 500 minus ∞) |
| :--- | :---: | :---: | :---: | :---: |
| Author Chebyshev (`N = 48/80/120`, converged) | **-0.04239** | -0.09392 | -0.09236 | **-0.05153** |
| Chebyshev, stress-free injector face (`d u_perp'/dx = 0` at x = 0) | -0.04239 | -0.04911 | -0.04621 | -0.00672 |
| Bulk SK only (no flow) | 0 | -0.00570 | -0.00285 | -0.00570 |
| 3D PDE `48×12×48` | **-0.10105** | -0.09907 (author) | -0.09996 (author) | **+0.00198 (wrong sign)** |
| 3D PDE `48×18×48` | **-0.09550** | -0.09440 | n/a | **+0.00110 (wrong sign)** |

- At Re = ∞ the PDE (-0.101 / -0.0955) and the Chebyshev model (-0.0424) disagree by 2.3–2.4×. The two models do not describe the same inviscid problem (inlet, outlet and numerical damping differ).
- 87% of the Chebyshev "viscous" increment (-0.0448 of -0.0515) comes from the injector-face Stokes layer that `u_perp'(0) = 0` creates. The Stokes-layer thickness is δ = √(2ν/ω) ≈ 0.047 at Re = 500. The PDE resolves it with only 1.5 cells (`nx = 48`, dx = 0.031) or 0.75 cells (`nx = 24`).
- In the PDE, Re = 500 is *less* damped than Re = ∞ at both nr = 12 and nr = 18, and Re = 1000 is *more* damped than Re = 500. The PDE does not resolve any viscous increment.
- The "convergence" moves away from the target under refinement (4.1% at `nx = 24`, then 5.5% at `nx = 48`).
- The Chebyshev model also omits the stress-free wall shift that the PDE contains (+2/(Re(s₁₁² - 1)) = +0.00167 at Re = 500).

**Required:** Remove the "converges / closes C1/C2" claim. Compare PDE and Chebyshev at Re = ∞ first, then use matched inlet BCs. Resolve the faceplate Stokes layer (≥ 5 cells in δ) before you quote a viscous increment. The bell "85% meridional + 15–20% viscous" split (Law #18) inherits the same problem: the viscous part is dominated by an unresolved injector-face layer, not by "fixed physical viscosity".

### C2. The partial-span "2.57–2.90× 2T boost" is a fit-window artifact. The mechanism ("inner-tip vortex shedding") is unsupported.

**Evidence:** For each case I reran Exp 2 (`24×12×50`, Re = 500), at Re = ∞, and on a refined `36×18×50` grid. I recorded the full azimuthal Fourier energy spectrum and the total chamber energy `E_ch`. The table gives the ratio Δα(r_b) / Δα(full span).

| Grid / Re | Measure | r_b = 0.42 | r_b = 0.58 |
| :--- | :--- | :---: | :---: |
| `24×12×50`, Re 500 | `e_2t`, [7, 22] (author) | **2.57** | **2.90** |
| `24×12×50`, Re 500 | `e_2t`, [10, 22] | **1.02** | 1.80 |
| `24×12×50`, Re 500 | `e_2t`, [12, 24] | 1.17 | 2.58 |
| `24×12×50`, Re 500 | `E_ch` (total), [7, 22] | **1.59** | **2.16** |
| `24×12×50`, Re = ∞ | `e_2t`, [7, 22] | 2.74 | 2.48 |
| `24×12×50`, Re = ∞ | `E_ch`, [7, 22] | 1.50 | 1.86 |
| `36×18×50`, Re 500 | `e_2t`, [7, 22] | 3.17 | 3.57 |
| `36×18×50`, Re 500 | `e_2t`, [10, 22] | 1.10 | 1.52 |
| `36×18×50`, Re 500 | `E_ch`, [7, 22] | **1.69** | **2.07** |

Raw α values (Re 500, `24×12×50`): bare -0.12975, full span -0.16801 ([7, 22]) / -0.18237 ([10, 22]), r_b = 0.42 -0.22819 / -0.18320, r_b = 0.58 -0.24088 / -0.22527. The bare 2T reference itself moves 8.6% under refinement (-0.12975 to -0.11865).

- **Two-peak signal:** The `c2_re` spectrum at r_b = 0.42 has two peaks of near-equal amplitude: 3.378 (1.00) and 2.592 (0.73). The fine grid gives 3.377 / 2.582 (0.77). The log of `e_2t` beats, so the slope depends on the window. The `r_b = 0.42 → 0.58` "branch jump" (3.386 → 2.748) is only a change of which peak is higher. On the fine grid, r_b = 0.58 is also bimodal (2.739 / 3.328, 0.72).
- **Modal transfer:** N_b = 5 Bragg-couples m = +2 to m = -3. The author's `e_3t` projects only onto m = +3 and misses this. Late energy ratio e(m = -3) / e(m = +2): full span 0.15, r_b = 0.42 **0.56–0.64**, r_b = 0.58 0.11–0.22. So part of the `e_2t` decay is redistribution into |m| = 3, not loss.
- **Not viscous:** The boost is the same size at Re = ∞ (2.74 / 2.48 on `e_2t`), so "viscous vortex shedding" is not shown.
- **What survives:** Partial-span baffles do damp total chamber energy more than full-span baffles, by 1.5–1.7× (r_b = 0.42) and 1.9–2.2× (r_b = 0.58), on both grids and at both Re.
- **Full-span upshift is physical:** The +9.3% upshift (3.337) is consistent with the 72° sector compartment mode ν = 2.5, whose cutoff is j'₂.₅,₁ = 3.633 (+18.9% if fully compartmented). With L_b / L_A = 0.30, a partial upshift is expected. It is not a 3T alias.

**Required:** Replace "2.57–2.90×" with an `E_ch`-based or two-mode-fit (Prony or matrix pencil) value, and give the window sensitivity. Report the m = -3 transfer. Remove "vortex shedding" as the mechanism, or demonstrate it (vorticity flux at the inner tip versus Re).

### C3. Mass-conserving `m_dot` does not cure k = 0 drift. The 1D "root cause" does not match the PDE oscillation. The "cure" is a flame-model change.

**Evidence (independent extension of `e015_mcons_1T` to t = 180, signed chamber-mean pressure recorded):**

| Window | max \|p̄'\| (k = 0) | max \|ΔM/M₀\| | E_1T | E_mean | P |
| :---: | :---: | :---: | :---: | :---: | :---: |
| [0, 40) | 2.31e-3 | 0.156% | | | |
| [40, 80) | 6.15e-3 | 0.273% | 7.41e-4 | 2.76e-5 | 5.42% |
| [80, 120) | 1.91e-2 | 0.934% | 5.96e-4 | 1.92e-4 | 5.88% |
| [120, 160) | 5.77e-2 | 2.807% | 2.68e-4 | 9.07e-4 | 6.36% |
| [160, 180] | **6.52e-2** | **3.521%** | **3.33e-6** | **2.47e-3** | 6.27% |

- The k = 0 oscillation grows by about 3× per 40 time units until it saturates at |p̄'| ≈ 6.5% of p̄. By t = 180 it has fully replaced the 1T mode. With mass-conserving cavities the run reaches the same chugging state as the legacy `u_port` run. Figure 2 already shows this growth in both runs up to t = 120.
- The quoted "1.63× / 2.52× E_mean cut", "slope flips negative" and "dm = -0.515%" come from the window t ≥ 105 (and slope [84, 120]). That window is shorter than one chugging period, so these values are phase snapshots of a growing oscillation, not means. This answers checklist item 3: the -0.515% is the oscillation phase, not a steady offset.
- **Period mismatch:** The signed-mean zero crossings are at 24.6, 43.0, 63.9, 83.1, 103.2, 122.8, 142.6, 161.6. That gives T = 39.3 (ω = 0.160). The 1D eigenmodes have ω = 0.110–0.114 (T = 55–57) and are *linearly stable* (α = -0.020 at n_p = 0.30). The 1D model therefore does not describe the PDE oscillation. The entry's "ω_chug ≈ 0.11–0.16, T ≈ 39" merges two different numbers. "Subcritical triggering" is asserted but not shown.
- **Attribution:** `ω_hp` is the high-pass constant of the flame-response model (`dfp[0] = ω_hp (p_sense - fp0)`), not a modeled feed system. Raising it removes the flame's low-frequency gain, so "eliminating chugging" is a model change. The "44,500×" ratio compares legacy at `ω_hp = 0.15` with mass-conserving at `ω_hp = 0.50`. The `ω_hp` change alone gives 27,400× (2.29e-4 → 8.38e-9), and `m_dot` gives at most 1.6× (a snapshot). `ω_hp` also changes the flame phase at 1T: arg H(1.84i) = -12.2° (0.15) versus -1.6° (0.50).
- **Confirmed:** At `ω_hp = 0.50` both runs stay chugging-free to t = 180 (E_mean ≤ 1.8e-8, |ΔM| ≤ 0.04%). `e015_hp50_mcons_1T` is a stationary 1T limit cycle (P = 4.53% at t ≈ 111, 4.47% at t ≈ 171). `e015_hp50_st1T` is **not** a limit cycle: it decays (P = 2.65% → 1.82% → **0.52%** at t ≈ 171, slope -0.054).

**Required:** Remove "mass-conserving coupling cuts E_mean 1.63–2.52×" and "root cause cracked". Describe `ω_hp = 0.50` as a flame-model low-frequency cutoff. Find the PDE's T ≈ 39 k = 0 mode in a matching linear model (3D-consistent, with the nonlinear flame operating point), or label the mechanism unresolved.

---

## Moderate findings

### M1. On curved bells the viscous x-derivatives use ∂/∂x at fixed η, not at fixed r. The η-face heat flux also drops n_x q_x.

`simulate_3d_ns` and `relax_axi_3d_bell_base` difference along a fixed η index. On the mapped grid r = ηR(x), the cylindrical stress needs ∂x|_r = ∂x|_η - ηR'∂r. With a manufactured field f = r², the spurious ∂f/∂x is 104% (`40×12`), 106% (`56×16`) and 108% (`80×24`) of max ∂f/∂r. This is a zeroth-order error that does not shrink under refinement (max |R'| = 1.44). `qx_ef = 0` also drops -n_x q_x on η-faces, where |n_x| ≤ 0.82 (13% of interior faces have |n_x| > 0.3).

I corrected both terms (`exp015_fixed.py`) and reran the bell 1T: Re = 500 gives -0.05223 → **-0.05241** (+0.3%), and Re = 1000 gives -0.04776 → -0.04796. The impact on the reported numbers is small, but the operator is inconsistent and must be fixed. Conservation stays exact either way, because the fluxes telescope. The straight-cylinder runs are not affected (R' = 0).

### M2. `hyb3d_1R` is not quenched. 1L grows exponentially, and the "9,524× 1R cut" uses the wrong reference.

Extended to t = 240 (Re = 500, `20×12×50`):

| Window | E_wave | E_1L | E_1R | P |
| :---: | :---: | :---: | :---: | :---: |
| [80, 120] | 1.88e-7 | 1.84e-7 | 4.8e-11 | 0.088% |
| [160, 200] | 4.86e-7 | 4.67e-7 | 1.2e-10 | 0.141% |
| [200, 240] | 7.92e-7 | 7.75e-7 | 2.0e-10 | 0.181% |

The amplitude growth rate is a constant +0.0058 to +0.0062, so this is linear 1L growth. Extrapolation reaches the `hyb3d_1L` limit-cycle energy (1.1e-3, 6.65%) near t ≈ 800. The 1R state is decided by the initial seed, not by the hybrid.

The "9,524×" figure divides by the bare `bare_ns_1R` E_wave, which is 1L (E_1L = 1.87e-3; E_1R = 1.2e-6). The bare 1R itself is not self-sustaining at n_p = 0.34. "1R dead" is therefore not something the hybrid achieves.

### M3. The 1L "migration" values for `hyb3d_st1T` and `hyb3d_2T` are mid-transient snapshots. The "48×" and "74×" factors use mismatched references.

At t = 120, both runs still grow (tail slope +0.018 and +0.025; Figure 3 shows E_wave rising). Their E_1L (6.0–6.5e-4) is still below the `hyb3d_1L` limit cycle (1.03e-3). "Dominant late state" and Dead End #17's "6.0–6.4%" must say "still growing at t = 120". "1T cut 48×" divides a *standing* hybrid by a *spinning* bare (no bare `stand_1T` run), at different t_end (120 versus 90). The cause "retuning 1.84 → 1.72 detunes from 1L = 2.09" was not tested (no run with the cavity at 1.84).

### M4. The SK curved-wall formula is exact to O(Re⁻¹). The residual "~5%" is part O(Δr²) dispersion and part a neglected O(Re^-3/2) term.

I solved the exact linear NS dispersion relation for a stress-free, adiabatic disk (acoustic + entropy + vortical Bessel modes, 3×3 determinant). For m = 1, exact minus formula = **+0.873 Re^-3/2** (2.6% at Re 300, 1.8% at Re 600). For m = 2 it is +1.22 Re^-3/2. For m = 0 the correction is ≈ 0, as claimed.

PDE refinement (net α = Re 300 minus Re ∞, `6 × nr × 48`):

| nr | 1T net | err vs exact root | err vs formula | 1R net | err vs exact |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 14 | -0.00634 | 3.04% | 5.48% | -0.03903 | 5.13% |
| 20 | -0.00645 | 1.34% | 3.82% | -0.03904 | 5.12% |
| 28 | -0.00652 | **0.25%** | 2.76% | -0.04083 | **0.77%** |

1T converges to the exact root at about second order. 1R does not follow O(Δr²) from nr 14 to 20: the error stays flat, then drops. The "uniform ~5% O(Δr²) dispersion" description is therefore not accurate. Also, the inviscid numerical baseline (-0.0072 for 1T and -0.058 for 1R at nr = 14) is as large as the viscous signal or larger.

### M5. The bell base flow is grid-sensitive at the throat.

The 36×12 Re = 500 base used for Exp 4 converged (`rhs_max = 7.76e-8`). But throat Mach (center / wall) is 0.765 / 1.102 on 36×12, 0.734 / 1.008 on 40×12, and 0.811 / 1.237 on 56×16 (author). The quoted "Sauer/Hall throat 0.811 → 1.237" holds for one grid only.

---

## Minor findings

- **m1.** "1T baffle damping scales super-linearly with metal area (84.1% at 75%)" is wrong. 0.841 = 0.75^0.60, so the scaling is *sub*-linear.
- **m2.** "ω_chug ≈ 0.11–0.16, T ≈ 39": the 1D eigen-periods are 55–57 (and 62–84 at `ω_hp = 0.5`). T ≈ 39 is the PDE period (see C3).
- **m3.** "keeps chamber mass drift below 0.03%": the metric is +0.0306%.
- **m4.** "9–20× deeper": the actual ratios are 8.8× to 25.4× (bare n_p = 0.34: 0.1730 / 0.00682 = 25.4).
- **m5.** `measure_fft_omega` divides by a moving RMS over 41 samples (≈ 0.4 of a 2T period). This biases the peak: r_b = 0.58 gives 2.748 (author) versus 2.808 (plain Hann FFT), and the inviscid bare case gives 3.062 versus 3.054.
- **m6.** E_1T and E_2T (2|c|², pressure only) use a different normalization than E_wave (0.5 Σ(p² + |u|²)). As a result, E_1T > E_wave in several rows (for example 3.92e-3 versus 2.05e-3). The table puts these side by side without a note.
- **m7.** `hyb3d_f16_1T` (t = 90) was compared with `hyb3d_1T` (t = 120). At equal t = 120 I get P = 5.586% and E = 4.365e-4, so the 8.0% spread is **confirmed**. Note that E_1L differs by 3× (1.55e-7 versus 4.93e-7).
- **m8.** The free-slip outer wall and slip baffles zero the *whole* viscous flux, including the normal stress τ_nn (and τ_θθ on the baffles). Because u_n = 0 this term does no work, but the radial-momentum flux is not a rigid-wall condition.
- **m9.** "1R completely unperturbed behind baffles" is exact by symmetry: an m = 0 mode has u_θ = 0, so slip θ-baffles are invisible to it. It is not evidence for the shroud-ring design.
- **m10.** The 1D model `exact_1d_chugging_eig` puts the full β = 3% into one cavity at ω_H = 1.84. The 3D configuration uses three cavities (1.85% at 1.84, 0.80% at 3.05, 0.35% at 3.83). This is acceptable for chugging (compliance limit), but not for the 1L values quoted beside it.

---

## Confirmed

1. **Viscous stress tensor (straight geometry):** τ_rθ = μ(∂_r w - w/r + r⁻¹∂_θ v), τ_θθ = μ(2(r⁻¹∂_θ w + v/r) - ⅔∇·u), τ_xθ = μ(r⁻¹∂_θ u + ∂_x w). The viscous hoop term -τ_θθ·hoop and the face-radius-weighted angular-momentum fluxes (`VFx_w`, `VFe_w`, `VG_w`) match the exact cylindrical divergence form r(∇·τ)_θ = ∇·(r τ_{·θ}).
2. **Curved-wall SK shift:** The +2m²/(Re(s² - m²)) shift is exact at O(Re⁻¹), including the factor 2 and the sign. I checked it two ways: the energy integral ∫Φ = (4/3)μ∫D² - 2μ∮u_θ² dS, and the exact dispersion root. The exact roots at Re 300 are -0.006537 (1T), -0.020888 (2T) and -0.041144 (1R). The adiabatic wall needs no thermal layer, because J'_m(s_mn) = 0.
3. **1D chugging and 1L eigenvalues:** My independent frequency-domain shooting solver reproduces all 16 values in `cheb_1d_chugging_eigs` to 5 digits. The cavity coupling `Mat[n, ib+6] = 1.0` matches the 3D `ρ_g u_g + m_port = M_c` condition in sign and in per-area normalization.
4. **Bessel-NS Chebyshev bulk operator:** The viscous terms (4/3 ∂xx - s², ⅓ is ∂x cross terms) and the thermal term (1/(Pr Re))∇²(γp' - ρ') are correct, and N = 48/80/120 converge to 6 digits. Its limitations are in C1.
5. **Reproducibility:** I reproduced the Exp 2 sweep rows (bare, r_b = 0.00 / 0.42 / 0.58, 1T and 2T) and `bell_re500_40 = -0.05223` exactly.
6. **Full-span 2T upshift (+9.3%):** physical (ν = 2.5 sector mode), not a numerical alias.
7. **Cavity retuning:** `wH_2T = 3.386` is the best of the three tested values (linear α_2T = -0.279 versus -0.257 at 3.054 and -0.238 at 2.75).
8. **`bell_hyb_1T`:** stationary 7.03% 1T limit cycle out to t = 150 (E_1L flat at 2.6e-9, no 1L migration). The `8.45×` energy ratio against `bell_bare_1T` stands.
9. **`ω_hp = 0.50` runs:** chugging-free out to t = 180.
10. **Prose numbers:** Every number in the Exp 1–4 tables, Law #18 and Dead End #17 matches `exp015_metrics.json`, except m1–m4 above. The derived ratios (5.08×, 8.45×, 74×, 2.0×, 44,500×, 45,200×) are arithmetically correct, but some use mismatched references (C3, M2, M3).

## Required revisions (summary)

1. Withdraw the Chebyshev "convergence" claim (C1). Do a matched Re = ∞ comparison and resolve the faceplate Stokes layer.
2. Re-measure the partial-span 2T effect with total energy or a two-mode fit. Report window sensitivity and m = -3 transfer. Drop the "vortex shedding" mechanism (C2).
3. Rewrite Exp 3: mass-conserving coupling does not stop chugging, and `ω_hp` is a flame-model constant. Report T ≈ 39 as unexplained by the 1D model (C3).
4. Fix the bell metric terms (M1). Relabel `hyb3d_1R`, `hyb3d_st1T` and `hyb3d_2T` as non-stationary (M2, M3).
5. Correct Law #18 and Dead End #17 in `SYNTHESIS.md` accordingly.
