# Peer Review #8: Rocket Combustion Entry 011 and SYNTHESIS.md Law #14

- **Target:** `rocket-combustion/entries/011-2d-viscous-ns-baffle-tips.md` (the Entry), `rocket-combustion/sim/exp011_2d_viscous_ns_baffles.py` (the Script), `rocket-combustion/results/exp011_metrics.json` (the JSON), `rocket-combustion/SYNTHESIS.md` line 26 (Law #14)
- **Mode:** Read-only. All audit scripts, logs and raw outputs are in `/tmp/peer_review_011/` (`cheb_audit.py`, `jobs.py`, `prose_check.py`, `wake_loc.py`, `summ.py`, `out/*.pkl`, `out/*.npy`). The repository `git status -s` stayed empty.
- **Verdict:** **Major revision. Do not promote Law #14 in its current form.** The NS operator, the spectral solver and the free-slip baffle convergence are sound. But two headline claims fail independent tests: (1) the "2T trap disarmed for all Re ≤ 5000" claim fails nonlinearly at Re = 2000 and 5000; (2) the retuned-hybrid linear margin (`-0.0530 / -0.0603`, "2.45×") is contaminated by an unrelaxed base wake and is not grid- or window-converged. Several mechanism labels (wake `u_min`, M=0 "Stokes layer", "Chebyshev matches PDE to 1e-4") are also wrong.

---

## Critical

### C1. The M̄ = 0.16 2T Strouhal trap is NOT disarmed for Re ≤ 5000: it triggers subcritically at Re = 2000 and 5000
Entry:57, :62, :67, :109; SYNTHESIS:26 ("completely disarmed for all Re ≤ 5000").

The claim rests only on the linear Chebyshev root (`m016_chiral_vs_Re`, Script:758). I ran the nonlinear 2D NS PDE on `64×200`, M̄ = 0.16, n_p = 0.22, n_v = 0.04, `spin_2T`, t = 200 (`out/nl2T_*.pkl`):

| Re | A₀ | Linear α₂T | Late P_peak/p̄ | E_wave late | late/mid | Result |
|---|---|---|---|---|---|---|
| 1000 | 0.040 | −0.0340 (Cheb) | 0.39 % | 1e-6 | 0.034 | quenched |
| 1000 | 0.055 | | 0.53 % | 2e-6 | 0.044 | quenched |
| **2000** | **0.020** | −0.0159 (Cheb), **−0.0182 (PDE, A₀=1e-4)** | **4.37 %** | 4.78e-4 | 1.025 | **pure 2T limit cycle** |
| **2000** | **0.040** | | **4.38 %** | 4.79e-4 | 1.000 | **pure 2T limit cycle** |
| **5000** | **0.040** | −0.0017 (Cheb) | **5.95 %** | 8.56e-4 | 1.001 | **pure 2T limit cycle** |
| ∞ | 0.040 | +0.0094 | 7.17 % | 1.34e-3 | 1.001 | limit cycle |

In all runs the late energy is 100 % in m = 2 (e₁ ≈ 1e-33). Viscosity only lowers the limit-cycle amplitude (7.2 % → 6.0 % → 4.4 %). The trap is nonlinearly quenched only at Re ≤ 1000. Also, the linear margin at Re = 5000 is only −0.0017, and the linear crossing is at **Re ≈ 5960** (`cheb_audit.json: Re_cross_2T_chiral`). The Entry must restate the law as "linearly stabilized for Re ≲ 6000; nonlinearly quenched only for Re ≲ 1000 (A₀ ≤ 0.055)".

### C2. The retuned-hybrid linear margin (`−0.0530 → −0.0603`, "2.45×") is contaminated by base-wake drift and is not converged
Entry:95, :107, :111; SYNTHESIS:26 ("reaches −0.0603").

The wake is pre-relaxed for only t = 65 (Script:842–844) and `p_ref`, `ux_ref`, `ut_ref` are frozen to it (Script:387). I ran each case twice: perturbed (A₀ = 1e-4) and unperturbed ("floor", A₀ = 0, same base, flame and cavities on). Results (`out/hyblin_*.pkl`):

| Grid | Base wake | Floor/pert. energy at t = 85 | α (Script fit, t ≥ 18) | α (t 18–60) | α (t 40–85) |
|---|---|---|---|---|---|
| 40×120 | t = 65 | **0.32** | −0.0530 (reproduces Entry) | −0.0586 | −0.0478 |
| 60×160 | t = 65 | **0.59** | −0.0603 (reproduces Entry) | −0.0589 | −0.0606 |
| 80×240 | t = 65 | **0.86** | −0.0612 | −0.0566 | −0.0646 |
| 40×120 | t = 150 | 8e-4 | **−0.0553** | −0.0591 | −0.0517 |
| 60×160 | t = 150 | 4e-4 | **−0.0652** | −0.0594 | −0.0697 |
| 80×240 | t = 150 | 5e-4 | **−0.0695** | −0.0571 | −0.0805 |

This answers the "why more negative from 40 to 60" question. (a) At t = 65 the wake drift supplies 32–86 % of the measured E_wave at t = 85. This floor flattens the late slope, most strongly on 40×120. (b) With a clean base, the late-time decay is not a single exponential. The 40–85 slope keeps steepening with grid (−0.052 → −0.070 → −0.081). Only the 18–60 window is grid-consistent (−0.057 to −0.059, ±2 %). The single numbers `−0.0530`, `−0.0603` and the derived "2.45×" gain (also computed against a contaminated `−0.0216` untuned case) are fit-window and drift artifacts. Re-run with a t ≥ 150 base and a stated, converged window. Report a range, not a three-significant-figure value.

---

## Moderate

### M1. The "steady" no-slip wake at t = 65 is not steady
Entry:25 ("pre-relaxed ... steady 2D viscous wake"), Script:794, :842–844.

The residual is max |dU/dt|, estimated by continuing the base 0.3 time units (`out/wakeconv_*.pkl`):

| Grid, Re | max\|dU/dt\| at t = 65 | at t = 150 | max\|U₁₅₀ − U₆₅\| | α₁T (65 base) | α₁T (150 base) | floor/pert E at t = 34 (65 base) |
|---|---|---|---|---|---|---|
| 40×120, 500 | 1.33e-5 | 1.0e-9 | 5.9e-6 | −0.11252 | −0.11389 | 0.15 |
| 60×160, 500 | 1.48e-5 | 1.8e-9 | 6.1e-6 | −0.10536 | −0.10641 | 0.11 |
| 80×240, 500 | 1.43e-5 | 1.8e-9 | 8.2e-6 | −0.10239 | −0.10351 | 0.11 |
| 80×240, 1000 | 1.95e-5 | 3.4e-9 | 8.5e-6 | −0.09791 | −0.09875 | 0.08 |

The residual drops by four orders of magnitude between t = 65 and t = 150. In cold Exp 3 the bias is about 1 % in α. In the reacting A₀ = 1e-4 runs it dominates (see C2). Relax to t ≥ 150, or subtract the unperturbed floor run.

### M2. The no-slip Re = 500 damping drifts through 80×240. The converged wake penalty is about 19 %, not 16.5 %
Entry:76, :81, :111; SYNTHESIS:26.

Same Exp 3 protocol (t = 65 base, fit window 10–30):

| Grid | 40×120 | 60×160 | 80×240 | **100×300** | **120×360** |
|---|---|---|---|---|---|
| No-slip Re = 500 | −0.11252 | −0.10536 | −0.10239 | **−0.09935** | **−0.09966** |
| Free-slip Re = 500 | −0.12273 | −0.12276 | −0.12261 | **−0.12273** | **−0.12285** |

The no-slip value does settle by 100–120 (0.3 %). It does not keep drifting from the tip singularity. But the 80×240 value is still 2.7 % off. The converged penalty is **1 − 0.0997/0.1229 = 18.9 %**, not 16.5 %. The no-slip α is also strongly window-sensitive: a 20–34 window gives −0.0908 on both 100×300 and 120×360 (9 % lower), against only 2 % for free-slip. The Re = 1000 penalty (19.3 %) comes from a 60→80 pair that moved by 10 % (−0.1088 → −0.0979). It has no convergence evidence.

### M3. The "trailing-edge wake u_min ≈ 0.021" is the wall-adjacent cell, not a wake
Entry:76–77, :81, :111; SYNTHESIS:26 ("u_min = 0.021"); Script:657 (`np.min(ux_w)` over the whole field), Script:946 (colorbar "tip wake u_min=0.021").

On 100×300 the global minimum (0.0159) sits at x = 0.352, in the cell next to a baffle face, **upstream** of the tip (L_b = 0.45) (`wake_loc.py`). The grid sequence 0.0494 → 0.0341 → 0.0206 → 0.0159 tends to the no-slip wall value 0. The true minimum downstream of the tips (x > 0.45) is 0.0275. The "u_min" figure measures grid spacing and does not describe wake physics. The mechanism prose ("low-speed trailing-edge wakes ... directly at the baffle tips") is not supported by this metric.

### M4. "Chebyshev NS matches 60×160 PDE to 1e-4" overstates the agreement. The −0.0105 gap is a fit-window artifact, not a solver error
SYNTHESIS:26, Entry:109 ("verified to 1e-4"), Entry:50–53, Script:723.

Only 1 of 4 cases agrees to 1e-4. The others differ by 1.2e-3, 3.4e-3 and 1.05e-2 (8.2 %). Cause: with a `spin_1T` initial field, other m = 1 modes are excited. At M̄ = 0.18, Re = 500 the next mode is λ = −0.173 + 2.26i (`m1_spectrum_top3`), and it is still significant in the 12–30 window. The evidence (`out/bare_*.pkl`):

| M̄ = 0.18, Re = 500 | 60×160 | 90×160 | 120×160 | 180×160 | Chebyshev N = 60 |
|---|---|---|---|---|---|
| Fit 12–30 (Script) | −0.13945 | −0.13916 | −0.13906 | −0.13873 | −0.12893 |
| Fit 20–40 | −0.12779 | −0.12746 | −0.12728 | −0.12706 | −0.12893 |

The PDE is grid-converged in both windows. With the later window it agrees with Chebyshev to 1.4 %. At M̄ = 0.12, Re = 1000 on 120×160 the gap shrinks from 2.2 % to 0.4 % (−0.05297 against −0.05277). The solvers are consistent. Restate the claim as "agreement to 0.4–1.4 % with a t ≥ 20 window", and drop "1e-4".

### M5. The retuned hybrid is not nonlinearly all-mode safe: a 2T limit cycle survives, and 1T triggers at A₀ ∈ (0.010, 0.015]
Entry:97 ("all-mode stable!"), :107; SYNTHESIS:26.

- 60×160, no-slip Re = 500, retuned hybrid, `spin_2T` A₀ = 0.055: **a sustained pure 2T limit cycle, 4.12 %**, E = 2.28e-4, late/mid 0.998. This happens even though the linear 2T α = −0.0363.
- 60×160 `stand_1T` A₀ = 0.055: 4.36 % (E 3.91e-4), the same attractor as `spin_1T`.
- 40×120 `spin_1T` amplitude scan: A₀ = 0.005, 0.0075, 0.010 decay (α ≈ −0.047). A₀ = 0.015, 0.025, 0.040 and 0.055 all reach the same 4.23–4.27 % / 3.92e-4 attractor. **The trigger threshold is A₀ ∈ (0.010, 0.015].** The "6.9 % shootout" therefore overstates the required perturbation by about 4–5×.

"All-mode stable" must read "linearly all-mode stable". The Verdict should report the 2T upper branch and the low trigger threshold.

### M6. M̄ = 0 "physical Stokes-layer damping −0.026" is converged but mis-attributed
Entry:79, :110; SYNTHESIS:26.

Convergence holds: on 120×360 free-slip is −0.02601 and no-slip is −0.02603 (80×240: −0.02592 / −0.02610). Decomposition on 80×240, Re = 500:
- Bare chamber, no baffles: **−0.0115** (bare Euler −2e-6). This is bulk Stokes–Kirchhoff (−0.0017) plus about −0.0098 from the injector face, which the inlet ghost treats as no-slip for u_θ.
- Baffles add about −0.0145, but **no-slip minus free-slip is only −2e-5 (< 0.1 %)**. A free-slip blade has no Stokes layer. The baffle term is viscous dissipation of the 1T flow around the sharp tips.
- Re scaling (free-slip, 80×240): Re 500 / 1000 / 2000 gives −0.0260 / −0.0183 / −0.0134 (no-slip −0.0261 / −0.0190 / −0.0143). This is about μ^0.65, between a boundary-layer (μ^0.5) and a bulk (μ¹) law.

The baffle no-slip Stokes layer contributes almost nothing. About 38 % of the "−0.026" comes from the modelled injector-face shear layer.

### M7. Physical Re regime and law robustness
Entry:11, :67; SYNTHESIS:26.

Re = ρ̄c̄R/μ_eff = 500–2000 implies ν_eff ≈ 0.1–0.4 m²/s for c ≈ 1000 m/s, R ≈ 0.2 m. That is an extreme eddy viscosity, and a constant eddy viscosity acting on coherent acoustic and vortical waves is itself an unvalidated closure. In the plausible Re ≳ 6000 range, the 2T trap is linearly active again (C1). The "trap is strictly high-Re" takeaway (Entry:67) should be framed as "returns at realistic Re". The scope line (Entry:27) should state this limitation.

---

## Minor

1. **Chebyshev BC count.** The viscous system has differential order 7 (ρ 1, u_x 2, u_θ 2, T 2), but `exact_2d_ns_eig` imposes 6 BCs. It also replaces the outflow **density** row with the T-Neumann condition (Script:186). The eigenvalues still converge spectrally (below), and alternative closures move α by ≤ 1e-4 (inlet T-Neumann) and ≤ 7e-4 (outlet u_x-Neumann, itself non-convergent). Changing the injector u_θ condition from Dirichlet to Neumann moves α by 33–50 % (M̄ = 0.12, Re = 500: −0.0687 → −0.0457). So the "vorticity attenuation" result depends strongly on the u_θ′ = 0 injector-face condition. The Entry should document this.
2. **PDE inlet ghost.** The PDE ghost puts u_θ = 0 at x = −Δx/2 (Script:419, :464), not at the face. The wall shear μ∂u_θ/∂x is therefore half the face-Dirichlet value, an O(Δx) BC error. It is small at the tested grids (see M4).
3. **Survival law accuracy.** The law L_visc/L_a = Re M̄³/(ω²L_a) is leading-order only. The exact advection–diffusion root gives 0.640 against the law's 0.588 (9 %) at M̄ = 0.12, Re = 500, and 0.585 against 0.564 at M̄ = 0.16, Re = 1000.
4. **Docstring mismatch.** The docstring says "N=48/64 converged to <1e-5" (Script:16), but the code uses 44/60. It also lists Re ∈ {300, 5000} for the envelope, which the code does not compute (Script:742 loop uses only 500, 1000, 2000, ∞).
5. **Values not in the JSON.** These prose values are not in the JSON: bare Euler linear `+0.0333`, exact `+0.0351` (Entry:89); Re = 5000 cold `−0.03401` (Entry:62; I reproduced −0.034013 independently); the Entry-010 `7.42 % / 1.22e-3` row (Entry:91).
6. **Rounding slips.** "0.12 % across 40×120 → 80×240" (Entry:110, SYNTHESIS) is 0.10 %. The 0.12 % is 60→80. "2.94×" (Entry:90) is 2.93×. "Grid-converged to 0.04 %" (Entry:107) is 0.05 % in E, and P_peak differs by 0.95 % (4.27 → 4.31).
7. **Conservation disclosure.** In the periodic no-slip run, x-momentum changes by 0.204 (relative) and θ-momentum by 0.0035. This is physical wall drag, but the prose does not disclose it. The mass and energy errors of 1.89e-16 and 2.35e-16 are identical to the bare case to 17 digits (1 ulp). This is a telescoping-flux identity. It is not an accuracy test, and the Entry should not present it as "exact viscous energy conversion".
8. **Figure labels.** The Fig 2 colorbar "tip wake u_min=0.021" (Script:946) and the Fig 1 legend "−16.5 % wake penalty" carry the M2 and M3 errors. The Fig 1 right panel shows M̄ = 0 for two grids only.

---

## Confirmed

- **Operator re-derivation.** The interior operator of `exact_2d_ns_eig` equals my independently derived linearized NS symbol to 2.2e-19 (M̄ ∈ {0, 0.12, 0.18}, k ≤ 9, m ≤ 3). This covers the Stokes 4/3 and −2/3 terms (net μ/3 cross terms) and the energy term (μ/Pr)∇²(γP − ρ′), with T′ = γp′ − ρ′ and (γ−1)κ = μ/Pr. The symbol gives Stokes–Kirchhoff −0.0168055, matching the exact formula.
- **Spectral convergence.** For N = 44, 60, 80, 96, the maximum spread of the cold α is 6.4e-6 over m = 0..3, M̄ ∈ {0.08, 0.12, 0.16, 0.18, 0.25}, Re ∈ {500, 1000, 2000, ∞}. The maximum |n_pH(N44) − n_pH(N80)| is 6e-5.
- **Higher azimuthal modes.** m = ±4 and ±5 (window to ω = 7) give n_pH ≥ 0.90 (the cap) for all Re and M̄ ∈ [0.08, 0.25]. The m ≤ 3 envelope is complete.
- **Finite-volume viscous discretization.** The stress tensor matches the Entry's equations. The split-face baffle fluxes are consistent: no-slip τ_xθ = ∓2μu_x/Δθ with the wall at the half cell; τ_θθ uses the u_θ = 0 wall; the wall energy flux is 0, which is correct for adiabatic impermeable walls with u_wall·τ = 0 (no-slip) or τ_xθ = u_θ = 0 (free-slip).
- **Free-slip Re = 500 convergence.** 100×300 gives −0.12273 and 120×360 gives −0.12285. This is within 0.2 % of 80×240 and closes Review #7 M3 for free-slip blades.
- **M̄ = 0 baffle damping.** Converged to 0.4 % (80×240 → 120×360) at −0.0260 (see M6 for the attribution).
- **Hybrid nonlinear reproduction.** The hybrid 1T upper branch reproduces (40×120 A₀ = 0.055: 4.2695 %, 3.91e-4). The attractor is independent of A₀ in [0.015, 0.055].
- **Prose against JSON.** 115 numbers checked by `prose_check.py`. All match except the rounding slips in Minor 6, and the "1e-4" claim (M4).

## Required actions before promotion of Law #14
1. Restate the 2T-trap claim using the nonlinear evidence in C1.
2. Re-run Exp 4 linear cases with a base relaxed to t ≥ 150 (or floor-subtracted). Report a converged window and α range, then recompute "2.45×".
3. Report the no-slip penalty from 100×300 / 120×360 (≈ 19 %). Replace the global `u_min` with a downstream wake metric.
4. Correct the "1e-4" Chebyshev claim and the M̄ = 0 Stokes-layer attribution.
5. Replace "all-mode stable" with "linearly", and add the 2T limit cycle and the A₀ ∈ (0.010, 0.015] trigger.
