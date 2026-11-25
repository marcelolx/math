# Peer Review #7: Rocket Combustion Entry 010 and SYNTHESIS Law #13 / Dead End #12

- **Reviewer:** independent read-only audit (Claude, Orca worker `task_84ada62fd8da`)
- **Date:** 2026-11-25
- **Targets:** `rocket-combustion/entries/010-2d-throughflow-baffles-cavities.md` (abbreviated `E010`), `rocket-combustion/sim/exp010_2d_throughflow_baffles_cavities.py` (`SIM`), `rocket-combustion/results/exp010_metrics.json` (`JSON`), `rocket-combustion/SYNTHESIS.md` (`SYN`)
- **Audit artifacts:** `/tmp/peer_review_010/` (scripts `a1_cheb.py`, `a1b_baffle.py`, `a5_prose.py`, `b1.py`, `b2.py`, `b3.py`, `an1.py`, `an2.py`, `an3.py`; logs `*.log`; data `*.json`, `*.pkl`). The repository solver ran unmodified through `import`. A broadband `noise` IC and `spin_m4`/`spin_m5` ICs came from an in-memory patched copy only. `git status -s` is empty.

## Verdict

**Accept with major revisions.** The hardware implementation is correct. Conservation, wall fluxes, cavity coupling and the Chebyshev operator all pass independent checks. The retuned hybrid is all-mode stable on every IC and grid that I tested. The wide-mouth 2T trim quench also survives refinement. But three headline claims do not survive grid refinement or a clean linear measurement:

1. Production grids hide real 2T growth (a repeat of Dead End #9).
2. The "medium σ_o = 2.5 % slow bleed" side of the orifice-drag cliff is a grid artifact.
3. The reacting U-curve ratios (4.3×, 3.2×) and the −9.1 % retuning frequency rest on numbers that are contaminated by nonlinearity or not converged on the grid.

The "St_eff = 1.00 trough" explanation also has no support in the cold data.

---

## Critical

### C1. 40×120 and 48×160 damp 2T numerically more than 2T grows physically (Dead End #9 recurs)
- **Claims:** E010:11 ("match an exact … eigen-solver across grids"), E010:52 (2T excess `-2.0e-2` with no comment), E010:67 (blade-count audit at 40×120), SYN:5 ("cross-checked directly against time-domain PDE integration").
- **Evidence (`an1.log`):** Cold bare 2T at M̄ = 0.12 has the exact value α = −0.02934. The PDE (e2 fit) gives −0.0611 (40×120), −0.0420 (60×160), −0.0353 (80×240) and −0.0313 (120×360). The excess at 40×120 is −0.032. Cold 1T converges well: −0.0327, −0.0302, −0.0294 and −0.0290 against the exact −0.0284.
- **Reacting consequence:** Chebyshev gives bare 2T (co-rotating) α = **+0.01137**. The PDE spin_2T (A₀ = 0.002) gives **−0.0158 (40×120)**, +0.0003 (60×160) and +0.0063 (80×240). At M̄ = 0.16, the exact value is +0.0094. The PDE gives +0.0002 (48×160) and +0.0061 (64×200) (`an2.log`). On the production grids, a linearly unstable 2T mode looks stable.
- **Impact:** All 2T verdicts at 40×120 and 48×160 have an offset of 0.01 to 0.03 in favor of stability. The blade-count audit (E010:67) compares runs that all decay, but the true bare 2T grows. One hidden 2T case appears: at **L_b = 0.60**, spin_2T on 40×120 is almost neutral (α(e2) = −0.0011, m = 2 dominates the final spectrum, `an2.log`). With about 0.03 of numerical 2T damping removed, the L_b = 0.60 design probably has a **linearly unstable 2T** in addition to its 1T. The entry does not report this.
- **Required:** Add a per-mode numerical-damping table (1T, 2T, 3T on every production grid), as Dead End #9 requires. Re-run every 2T claim on ≥ 80×240, or correct it with the Chebyshev offset.

### C2. The orifice-drag cliff moves with the grid. The σ_o = 2.5 % "slow bleed" is a 48×160 artifact
- **Claims:** E010:110 (medium, "Slow bleed", 1.86 %), E010:104 (one linear spectrum for all three designs), SYN:25 and SYN:42(c).
- **Evidence (`an2.log`, spin_2T, A₀ = 0.040, t = 240):**

| σ_o | 48×160 late P (α) | 64×200 late P (α) |
| :--- | :---: | :---: |
| bare | 6.58 % (entry, t = 160) | **7.17 %** (l/m 1.000) |
| 1.2 % tight | 4.91 % (+0.0000) | **5.84 %** (+0.0000) |
| **2.5 % medium** | 0.21 % (−0.0424) | **4.03 % sustained 2T limit cycle** (l/m 1.000) |
| 5.0 % wide | 0.002 % (−0.0528) | 0.013 % (−0.0447) |

- **Impact:** On 64×200 the medium design does not bleed. It chokes and locks onto a 4.0 % 2T rotating shock. The cliff edge lies between σ_o = 2.5 % and 5.0 %, not between 1.2 % and 2.5 %. The limit-cycle amplitudes rise 9 % to 19 % from 48×160 to 64×200, so "6.58 %" and "4.90 %" are not grid-converged. This agrees with C1: 48×160 under-resolves 2T. The wide-mouth quench is robust (Confirmed below).

## Moderate

### M1. The reacting U-curve ratios use "linear" fits that include amplitude-dependent growth
- **Claims:** E010:60–65 (reacting column), E010:63 "4.3× cut", E010:64 and E010:124 "3.2× worse", SYN:25 and SYN:42(a).
- **Evidence:** The `lin_*` runs start at A₀ = 0.002 and run to t = 95–110. The fit uses `stats_2d(..., fit_start_frac=0.35)` (SIM:740, SIM:757). Bare runs grow to 2–8 % within the fit window, where the subcritical cubic term (κ = 4.5) raises the growth rate. A clean A₀ = 1e-4 run (`an3.log`) gives a bare reacting 1T α of 0.03335 (40×120), 0.03472 (60×160) and 0.03506 (80×240), which **matches the exact +0.03506**. The entry's "A₀ = 0.002" values are 0.0393 (40×120) and 0.0477 (60×160), and my 80×240 repeat gives 0.0478. The clean bare value gives a 60×160 cut ratio of about 0.0347/0.0110 = **3.2×, not 4.3×**. The same contamination applies to E010:91–92 (bare +0.0330 and "cavity alone" +0.0239). At A₀ = 1e-4 the cavity-alone value is +0.0146, against an exact discrete-port value of +0.0181.
- **Required:** Measure every linear rate at A₀ ≤ 1e-4, or restrict the fit window to p′/p̄ < 0.5 %.

### M2. The cold data do not support the "St_eff = 1.00 trough" mechanism, and the n_v = 0 data reverse the short-baffle claim
- **Claims:** E010:56 ("non-monotonic Strouhal U-curve"), E010:62 (L_b = 0.30 labelled "*(peak)*" while E010:63 labels 0.45 "peak"), E010:64 ("2nd trough"), E010:124, SYN:25.
- **Evidence:**
  - At 80×240 the cold damping is −0.0309, −0.0354, −0.0621, −0.1064, −0.0770 and −0.0723 for L_b = 0 to 0.75. L_b = 0.60 (St 1.02) and 0.75 (St 0.81) damp equally, at 2.3× bare. No trough appears at St = 1.00, and no rise follows it.
  - The cold peak sits at St_eff ≈ 1.25–1.30, not at a half-integer. This contradicts the 009 half-integer peak rule that the prose invokes.
  - Pure pressure coupling (n_v = 0, 40×120, `an2.log`) gives reacting α = +0.0216, **+0.0141**, −0.0039, +0.0002, **+0.0252** and +0.0176. Without chiral coupling, the short baffle (L_b = 0.15) *reduces* growth, the best length is 0.30 and not 0.45, and L_b = 0.60 is worse than bare despite 2.3× more cold damping.
- **Interpretation:** The reacting re-destabilization at L_b = 0.60 comes mainly from the baffle-induced frequency downshift, which changes the flame phase. The baffles also change u_θ at the flame (x_f = 0.25 < L_b), which acts through n_v. Strouhal acoustic-vorticity damping is not the main cause. The "short-baffle amplification" (E010:61, E010:124) occurs only with n_v ≠ 0.

### M3. Part of the inviscid baffle damping is numerical
- **Evidence (`an3.log`):** At **M̄ = 0** (no throughflow, so the exact Δα is 0 for inviscid blades), N_b = 5 and L_b = 0.45 give Δα = **−0.0113 (40×120)** and **−0.0062 (80×240)**. This is first-order convergence and therefore purely numerical. At M̄ = 0.12 the L_b = 0.45 cold damping drifts −0.1172, −0.1119, −0.1064 (40, 60, 80): a 5 % change per refinement that has not converged. The sharp-edge "Kutta" shedding in an inviscid scheme depends on numerical dissipation, so the claimed 3.5× contains about 10–15 % scheme dissipation at 40×120.

### M4. The baffled 1T frequency used for retuning does not converge on the grid. The optimum is not ω_1T(L_b)
- **Claims:** E010:95, E010:125, E010:130, SYN:25 ("−9.1 %, 0.997 → 0.899").
- **Evidence:** ω_1T(L_b = 0.45) = 0.899 (40×120), 0.915 (60×160, JSON), and 0.933 FFT or 0.923 least squares (80×240). Bare least squares gives 0.9889, against the exact 0.9893. The fine-grid downshift is about **−6.7 %**, not −9.1 %. The ratio 0.899/0.997 is −9.8 %. Only 0.899/0.9893 gives −9.1 %, so the prose pair "0.997 → 0.899" does not produce its stated −9.1 % (`a5_prose.log`).
- **Retuning test (`an3.log`):** A cavity tuned to the fine-grid ω_H = 0.923 gives α = −0.0147 (40×120) and −0.0153 (80×240). That is **worse** than ω_H = 0.899, which gives −0.0218 at 80×240. The best tuning lies below the cold baffled frequency. This is consistent with the reacting 1T shift (bare reacting ω = 0.9625 against cold 0.9893, −2.7 %). The coarse-grid 0.899 lands near the optimum by coincidence. The law must say "tune to the baffled *reacting* frequency" and must include a scan over ω_H.
- **Positive:** The 5.0× gain holds at 80×240 (−0.00417 → −0.02183, 5.2×). At 40×120 the ratio is 4.5× (E010:95 shows only the 60×160 ratio).

### M5. The retuning result depends on cavity placement relative to the blades (not reported)
- **Evidence (`an2.log`):** Retuned hybrid, 40×120, spin_1T: phase_offset 0.5 (mid-sector) gives α = −0.0214. **phase_offset 0.0 (ports at blade roots) gives α = −0.0136**, a 36 % loss of margin. 2T stays stable in both cases (−0.047 and −0.039). The Chebyshev reduction cannot show this effect (it has no baffles), so the entry must state the mid-sector placement as a design requirement.

### M6. The 1T trigger threshold at M̄ = 0.16 is much lower than the "large 6.9 % 1T bomb" wording implies
- **Claims:** E010:114–115, E010:127, SYN:25 ("large 6.9 % 1T bombs still trigger").
- **Evidence (48×160, t = 240, wide-mouth trim, spin_1T):** A₀ = 0.005 and 0.015 decay (α = −0.033). **A₀ = 0.025, 0.040 and 0.055 all lock onto the same 3.947 % cycle.** The threshold lies in (0.015, 0.025], which is about p′/p̄ ≈ 2–3 %, not 6.9 %. The bare M̄ = 0.16 chamber triggers at the same amplitudes to 4.36–4.41 %, so the trim cavity lowers the 1T cycle by only about 10 %. stand_1T converges to 3.947 % by t = 300 (the l/m = 1.068 at t = 160 was a transient).
- **Arithmetic:** E010:114 and E010:127 call −0.0306 the "bare M̄ = 0.16 1T margin". The bare value is **−0.0270**. −0.0306 is the value *with* the trim cavity.

## Minor

- **m1. Energy ratio label (E010:98).** "Wide-mouth 5 % cuts energy 2.20×" is the ratio against the tight neck (2.04e-3/9.27e-4). Against bare the ratio is **4.02×** (3.73e-3/9.27e-4).
- **m2. "Exact wall pressure p\*" (E010:20–21, E010:44).** The mirror HLLC wall flux gives zero mass, tangential-momentum and energy flux to ≤ 2.7e-15. The wall pressure differs from the exact Riemann reflection pressure by up to 26 % at |u_θ| = 0.4, and by ≤ 0.55 % at |u_θ| < 0.05 (`a1b_baffle.log`). Call it the "HLLC wall pressure".
- **m3. The torque balance is a discrete identity.** `blade_force` (SIM:466) comes from the same wall fluxes that the update uses, so the 3.8e-17 residual confirms bookkeeping only. It is not an independent physical check. SYN:25 presents it as a physics result.
- **m4. The first-order blade-adjacent stencil (SIM:436–439) adds a small, converging dissipation.** Aligned stand_2T at n_v = 0, L_b = 0.45, compared with bare: cold Δα = −0.0024 (40×120) and −0.0007 (80×240). Reacting (n_p only) Δα = −0.0021 and −0.0005. The scheme is mirror symmetric about the blade plane to 1e-12 (N_b = 4 and 5). The Law #5(b) blind spot therefore holds, up to this artifact.
- **m5. Port width depends on the grid.** `sig_th = max(1.5*dth, …)` (SIM:353) selects 1.5 Δθ for every σ_o used, so the port shape follows the grid and not the orifice area.
- **m6. The cavity-alone spectra in E010:73–83 use ξ_H = 0.10/0.14, but Exp 3B (SIM:794–801) uses 0.12/0.12.** The PDE "2 % cavity alone" (E010:91) therefore has no matching Chebyshev row. The exact value for the PDE configuration is +0.0184 continuous and +0.0181 discrete.
- **m7. 2T retuning uses 2·ω_1T(L_b) = 1.798 (SIM:800).** The baffled 2T frequency is not 2× the 1T frequency. On an unbaffled ring this detunes 2T, and Chebyshev gives m = −2 at +0.0039. The baffles damp 2T strongly (−0.033 at 80×240), so the hybrid stays stable, but the choice needs a reason.
- **m8. Spurious Chebyshev modes.** At N = 48 the operator has spurious |ω| ≈ 44 modes with Re ≈ +0.012–0.016. The window [0.05, 5.5] excludes them correctly. State the window filter in E010.

## Confirmed (independently reproduced)

- **Chebyshev operator.** An independent re-implementation matches `exact_2d_cavity_eig` to 0.0. Convergence: max |α(N=40) − α(N=64)| = **7.1e-5** over 12 configurations and m = −5…+5. No eigenvalue that is less stable than the windowed maximum exists at |ω| < 0.05 or in (5.5, 8]. Every E010 Exp 3A table entry (36 values) matches JSON to < 2e-3 relative. No hidden m = 0, ±4, ±5 or cavity-mode instability exists in any configuration.
- **N_c = 5 aliasing (m ↔ m − 5).** A coupled [2, −3], [1, −4] and [0, ±5] solve with discrete ports shifts α by ≤ 0.010 and always toward stability. The limiting trim mode (m = −1, −0.0306) is unchanged. The PDE spin_3T run shows the 3T → 2T alias transfer (e2 > e3), and it decays.
- **Cavity coupling and passivity.** ρu_in = M̄ − u_port has the absorbing sign. The Newton inlet residual is 1.4e-15. Linearization with |ŵ_m|² is algebraically exact for a single m.
- **Conservation.** Mass, x-momentum and energy stay ≤ 3.9e-16 for N_b = 4 and 5, on 40×120 and 60×160, at t = 10 and 40. Torque residual ≤ 6.2e-17. Steady preservation is 1.11e-16.
- **Exp 2 at L_b = 0.45.** 2T is robustly stable (α(e2) = −0.035 at 40×120, −0.033 at 60×160). cold_1L decays. noise and stand_1T grow on the 1T branch only. No hidden m = 0 or 2T mode appears at 0.45.
- **Hybrid retuned (Exp 3B) is all-mode stable.** At 40×120, t = 250, the decay rates are spin_1T −0.0214, stand_1T −0.0215, spin_2T −0.047, spin_3T −0.039, cold_1L −0.053 and noise −0.022. At 60×160 they are spin_1T −0.0216, spin_2T −0.043 and noise −0.022. At 80×240 they are spin_1T −0.0218 and spin_2T −0.043. The linear-cavity variant matches the nonlinear one (−0.02156 against −0.0215). The claim "60×160 converged to 1e-4" holds for the retuned case (80×240 −0.0218).
- **Wide-mouth 1 % 2T trim (Exp 4).** It quenches spin_2T on 48×160 and 64×200 (α ≈ −0.045 to −0.053). spin_m4, spin_m5, noise and cold_1L all decay (−0.034 to −0.079). No hidden m = 0, 3, 4 or 5 instability exists.
- **Prose and JSON.** 198 programmatic checks were made (`a5_prose.log`). 189 match. The 9 mismatches are the ratios in M1, M4 and M6, item m1, the 40×120 forms of the 3.5×, 4.3×, 3.2× and 5.0× ratios (3.42×, 3.86×, 3.68× and 4.47×), and a 0.5 % rounding in 62,700× (unrounded 63,012×).

## Required revisions (summary)
1. Add a per-mode, per-grid numerical-damping table. Re-run the 2T claims (blade audit, L_b = 0.60, Exp 4 cliff) on ≥ 80×240 or 64×200 (C1, C2).
2. Revise Law #13 and DE #12(c). The cliff lies between 2.5 % and 5.0 %, and σ_o = 2.5 % fails on 64×200 (C2).
3. Re-measure the reacting rates at A₀ ≤ 1e-4. Restate 4.3× → about 3.2× and recompute 3.2× (M1).
4. Remove or qualify the "St_eff = 1.00 trough" explanation. Report the n_v = 0 decomposition (M2).
5. Report the M̄ = 0 numerical baffle damping and the L_b = 0.45 non-convergence (M3).
6. Restate the retuning law as "tune to the reacting baffled frequency". Give the downshift as about −6.7 % (converged), add the ω_H scan, and state the placement sensitivity (M4, M5).
7. Correct the 1T trigger threshold to (0.015, 0.025] and the "bare 1T margin" to −0.0270 (M6).
