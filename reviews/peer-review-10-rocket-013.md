# Peer Review #10: Rocket Combustion Entry 013

- **Entry:** `rocket-combustion/entries/013-dual-retuning-viscous-bell.md`
- **Script:** `rocket-combustion/sim/exp013_dual_retuning_viscous_bell.py`
- **Artifacts:** `exp013_metrics.json`, three `exp013_*.png` figures, SYNTHESIS Law #16 and Dead End #15
- **Reviewer mode:** independent, read-only. All scratch work is in `/tmp/peer_review_013/`. The repository was not modified (`git status -s` is empty).
- **Date:** 2026-11-26

## Verdict: **Major Revisions**

Most of the numerical machinery is sound. The interior Chebyshev operator matches an independent complex-step linearization of the stated PDE to machine precision. The inviscid `120×300` cascade break holds to `t = 420`. The `3.0%` `85/15` quench at `Re = 1000` survives every scope test I ran.

One headline claim fails. The "hybrid `1T` margin is cut more than in half by viscosity" result (`-0.0930 → -0.0434`) comes from the short fit window. The inviscid hybrid run has two-mode beating. With a converged fit window, viscosity **increases** the hybrid margin (`-0.027 → -0.036` on `80×200`, `-0.028 → -0.040` on `120×300`). This claim appears in Law #16, in Dead End #15(b), and in the design logic of Exp 5 ("needs more `1T` volume"). The nonlinear re-ignition at `Re = 1000` (`4.00%`) is real and reproducible, but the linear explanation for it is wrong.

---

## Reproduction environment

- `exp013_patched.py` is a copy of the script with three switches: `XFACE_MODE` (`orig`, `onesided`, `zero`) for the cross-baffle θ-derivative on x-faces, `INLET_VISC` (on or off) for the viscous flux at the inlet face, and the new `stand_1T` and `stand_2T` initial conditions. Everything else is identical.
- `op_check.py` does an independent complex-step Jacobian of the stated PDE. `base_check.py` checks the base flow. `cheb_checks.py` and `hopf_check.py` check the eigen-solver. `runs/*.pkl` holds 31 PDE runs, and `analyze.py`, `slopes.py` and `blong.py` analyze them.

---

## Critical findings

### C1. The "viscous halving of the hybrid `1T` margin" comes from the fit window (fails on both grids)

The `hyb20_bell_*` alphas come from `fit_alpha(..., 20, 48, key="e_1t")` on runs with `t_end = 50`. The inviscid hybrid run is not single-mode. Local 10-unit slopes of `e_1t` swing between `-0.13` and `+0.066` (beating between cavity-split modes). The `[20, 48]` window lands on a beating trough.

| Case | `[20,48]` (script window) | `[50,95]` | `[30,95]` | Local slopes t = 10..95 (10-unit) |
| :-- | :-: | :-: | :-: | :-- |
| `hyb20_bell_inf_80` | **`-0.09295`** (reproduces JSON) | `-0.02677` | `-0.03108` | `-0.026 -0.060 -0.102 -0.098 +0.037 -0.010 -0.038 -0.063 -0.074` |
| `hyb20_bell_1000_80` | **`-0.04341`** (reproduces JSON) | `-0.03620` | `-0.03920` | `-0.027 -0.042 -0.048 -0.047 -0.041 -0.035 -0.033 -0.035 -0.037` |
| `hyb20_bell_inf_120` (new) | `-0.11217` | `-0.02794` | `-0.02738` | `-0.024 -0.060 -0.114 -0.132 +0.066 -0.009 -0.041 -0.072 -0.099` |
| `hyb20_bell_1000_120` (new) | `-0.04421` | `-0.03996` | `-0.04277` | smooth, `-0.036` to `-0.051` |

On any window that is not dominated by the early transient, the viscous hybrid is **more** damped than the inviscid hybrid (about `1.3–1.4×`). The claims "cut more than in half", "halving hybrid margin `-0.0930 → -0.0434`" (Law #16), and "`-0.03838` at `Re = 500`" are unsupported. The Exp 5 design inference ("a viscous bell needs more `1T` volume *because* the linear `1T` margin halves") is also unsupported.

**What survives:** the nonlinear `2.0%` `80/20` re-ignition at `Re = 1000` is robust. I reproduced it with the original stencil (JSON `3.97%` on `80×200`) and with one-sided cross-baffle stencils (`3.90%`, `e2/e1 = 0.005`, `tail = -0.0000`). The phenomenon is therefore subcritical and nonlinear. It is not a reduced linear margin (both linear margins stay at about `-0.03` to `-0.04`).

**Required:** refit every Exp 5 linear rate on a long window (`t_end ≥ 100`, discard `t < 40`), or extract modal rates with a two-mode or Prony fit. Remove or rewrite the "halving" claim in the entry, in Law #16 and in Dead End #15(b). Re-explain the `4.00%` re-ignition as a nonlinear or subcritical effect.

---

## Moderate findings

### M1. The cross-baffle viscous stencil on x-faces is a real artifact, but it does not cause the sign flip
In `simulate_2d_viscous_transonic`, `dux_dth_f` and `dut_dth_f` on x-faces use `np.roll` central differences across the solid blade at x-faces `0..nx_b`. This couples the flow on the two sides of an impermeable baffle. I replaced them with one-sided wall-aware differences (`onesided`) and with zeros (`zero`):

| Case (`spin_1T`, `A_0 = 1e-4`, Re = 1000) | orig `[20,48]` | onesided | zero | no inlet viscous flux |
| :-- | :-: | :-: | :-: | :-: |
| `react_baf_bell_1000_80` | `+0.00271` (JSON `+0.00271`) | `+0.00202` | `+0.00174` | `+0.00280` |
| `react_baf_bell_1000_120` | JSON `+0.00279` | `+0.00213` | — | — |
| `react_baf_comp_1000_80` | `-0.01111` (JSON) | `-0.01160` | — | — |
| `hyb20_bell_1000_80` | `-0.04341` | `-0.04454` | — | — |
| `R80_Re1000_20_8020_1T` (NL, `6.9%`) | `3.97%` (JSON) | `3.90%` | — | — |
| `R80_Re1000_30_8515_1T` (NL, `6.9%`) | `0.015%` (JSON) | `0.013%`, tail `-0.0426` | — | — |

- The stencil contamination accounts for about `0.0007–0.0010` of the `0.0081` viscous shift (about 9–12%). It inflates the `Re = 1000` growth rate by about 25–35%. **The sign flip `-0.0054 → +0.0020` survives the fix.**
- The inlet-face viscous flux (`u_port` ghost) has no effect (`+0.00280` vs `+0.00271`).
- The compact-nozzle stabilization does not depend on the stencil.
- **Required:** fix the stencil, or state the bias and report the corrected `+0.0020`/`+0.0021` values.

### M2. Exp 5 linear rates are window-sensitive even where the sign is robust
- `react_baf_bell_inf_80` has local slopes that settle at `-0.0087` (t > 40), against `-0.00542` on the script window.
- `react_baf_bell_1000_80` has local slopes that rise steadily: `-0.0013` at t = 40, `+0.0008` at t = 60, `+0.0025` at t = 90. A 28-unit window on a `t_end = 50` run cannot give a modal rate.
- The "grid-converged across `80×200` and `120×300`" claim therefore compares two equally transient fits.

### M3. Exp 2: the Hopf metric and the benchmark line use different flame models
- `n_pH` comes from `brentq` on `n_p` with **`n_v = 0`**. The "benchmark flame" line at `n_p = 0.22` in Fig 1 (left) and the "bare reacting" columns use `n_p = 0.22`, **`n_v = 0.04`**.
- Example: at `M_c = 0.12` with `Re = 2000`, `n_pH_1T = 0.263 > 0.22` suggests that the benchmark flame is stable. In fact `react_1T = +0.0059` (unstable). The same applies at `Re = 500` (`0.277` vs `+0.0006`) and at `M_c = 0.18` with `Re = 500` (`0.267` vs `+0.0012`).
- The figure therefore suggests stability that the benchmark flame does not have. Label the metric as `n_v = 0`, or compute it with `n_v/n_p` fixed.

### M4. Non-monotonic `Re` dependence of bare reacting `1T` (`M_c = 0.12`): real, converged, and not reported

| Re | ∞ | 4000 | 2000 | 1500 | 1200 | 1000 | 800 | 700 | 600 | 500 | 400 |
| :- | - | - | - | - | - | - | - | - | - | - | - |
| `react_1T` (N = 96) | `+0.02205` | `+0.01290` | `+0.00525` | `+0.00140` | `-0.00132` | `-0.00288` | `-0.00334` | `-0.00275` | `-0.00142` | `+0.00062` | `+0.00311` |
| N = 72 / N = 120 | `+0.0221/+0.0220` | `+0.0140/+0.0127` | `+0.0059/+0.0051` | | | `-0.00265/-0.00297` | | | | `+0.00062/+0.00060` | |
| ω (N = 96) | 1.059 | 1.052 | 1.043 | 1.037 | 1.031 | 1.025 | 1.017 | 1.013 | 1.008 | 1.004 | 1.000 |

- The JSON values (`-0.00265` at `Re = 1000`, `+0.00062` at `Re = 500`) reproduce, and the non-monotonicity is genuine. The bare reacting `1T` is linearly **stable** for about `Re ∈ (560, 1150)` and unstable on both sides of that range.
- The likely mechanism is a phase shift of the flame response (τ = 0.55) as viscosity lowers ω (`1.06 → 1.00`). Cold `1T` damping also peaks near `Re ≈ 500`.
- The prose statement "remains linearly unstable across `[0.10, 0.18]` even at `Re = 500`" is literally true. It hides the fact that the bare bell is stable at `Re = 1000`, `M_c = 0.12` (the operating point of all Exp 4 and Exp 5 runs).
- At `Re = 2000–4000`, N = 72 (used for all of Exp 2) differs from N = 120 by 9–14% in `react_1T`. The JSON quotes only the 1C spectral difference.

### M5. `m = 4, 5` "whistles" use unconverged and unrecorded eigenvalues
- Exp 2 scans `m = 0..5`, but the JSON stores Hopf values only for `m = 0..3`.
- My rerun gives the inviscid `M_c = 0.10` values below. N = 72: `m = 4` cold `+0.0212` at ω = 6.48, and `m = 5` cold `+0.0358` at ω = 6.21. N = 96: `m = 4` `+0.0037` at ω = 4.53, and `m = 5` `-0.0176` (stable) at ω = 5.05. So the N = 72 `m = 4, 5` eigenvalues are spurious, high-frequency modes from the wide (0.05, 7.0) window.
- The global Hopf value does not change (`m = 3` is converged at `+0.0022`, so `n_pH = 0`). The claim "quenching `m = 3, 4` whistles" rests on an unconverged `m = 4` that the JSON does not record. All viscous rows match at N = 72 and N = 96 (largest Hopf change `0.263 → 0.267` at `M_c = 0.12`, `Re = 2000`).

### M6. Numbers in the prose that the JSON does not record
- **`4.51%`** (`Re = 500` no-slip, `3.0%` `85/15`) is **not in the JSON**. I reproduced it independently: **`4.52%`** (`E = 5.08e-4`, `e2/e1 = 0.0097`, `tail = 0.0000`, `l/m = 1.000`), a saturated pure-`1T` cycle. The claim is correct, but the JSON must record it.
- `8.6%` (the inviscid-base Chebyshev error), `M_in 0.1200 → 0.1221` and `M_e 1.828 → 1.737` are also not recorded. My check gives Chebyshev N = 96 at `Re = 500`: `M_in = 0.12202`, `M_e = 1.7373`. These are consistent.
- The 1C `spec` values (`3.9e-5`, `2.7e-4`, `3.0e-5`, `9.4e-5`) are complex moduli `|v96 − v72|`. The JSON stores only the real parts, so these values cannot be recomputed from the JSON.

### M7. Exp 3 resonance curve: the shape holds, but the amplitude depends on the grid
| `ω_H,2T` | 80×200 inviscid (JSON) | **120×300 inviscid (new)** | **80×200 Re = 1000 (new)** |
| :-: | :-: | :-: | :-: |
| 1.45 | `0.0214` | `0.0190` | `0.0419` |
| 1.65 | `0.0392` | `0.0330` | `0.0739` |
| 1.90 | `0.0185` | `0.0149` | `0.0452` |
| 2.01 | `0.0150` | `0.0116` | `0.0406` |
| peak ratio 1.65/2.01 | `2.61×` | `2.83×` | `1.82×` |

The peak at `1.65` holds on the finer grid and at `Re = 1000`. The absolute `80×200` damping is 16–22% too high, and the `2.61×` gain is grid-specific. At `Re = 1000` the retuning gain falls to `1.82×`, because viscosity already damps `2T` strongly.

### M8. Scope of the "exact" operator
The interior operator matches the stated PDE exactly. The stated viscous stress uses the planar form `τ_xx = μ(4/3 u_x − 2/3 u_θ,θ/R)`. It has no quasi-1D or thin-shell metric dilatation terms (`u A'/A`, `u R'/R`). In the converging section `|A'/A| ~ O(1)`, so the base and perturbation viscous dissipation are model-specific. The entry should list this omission with the other "intentional reductions".

---

## Minor findings

- **m1.** The energy-ratio headlines use mixed baselines. `3.7e6×` (`3.0%` 1T) is relative to the *inviscid untuned* run. Relative to the matched `Re = 1000` `2.0%` 1T run it is `2.2e6×`. `3.9e8×` (2T) and `1.66e8×` are relative to the *inviscid* dual-retuned 2T run. Name the baseline every time.
- **m2.** Law #16 says "`4.4–5.0×` global Hopf" for `Re = 500–1000`. The `4.4×` value is the `Re = 2000` number. For `Re = 500–1000` it is `4.6–5.0×`. Prose §3 is correct (it attributes `4.4×` to `Re = 2000`).
- **m3.** Fig 1 (left): the `Re = 500` curve is clipped at `M_c = 0.22` (`0.467 > ylim 0.45`). The legend "`1T`-limited everywhere" is wrong at `M_c = 0.22` (`limiting_m = 0` at `Re = 500` and `Re = 1000`).
- **m4.** Fig 1 (right) labels the unbaffled line `ω_2T = 1.98` (the FFT value). The prose calls `2.01` "the unbaffled frequency". The `+23%` detuning uses `2.01`. Against the measured bare `1.977` the detuning is `+21%`.
- **m5.** The Fig 2 (left) title says "on 120x300" but includes an `80×200` no-slip curve. The Fig 3 (right) title says "Resolved 120x300" but includes two `80×200` curves. The Fig 3 (left) bars mix `80×200` (cold, compact, hybrid) and `120×300` (reacting bell) results. Each curve is labelled, but the titles overstate.
- **m6.** `R80_Re500_noslip_2T` ("reverse `2T → 1T` cascade") has `l/m = 2.66` and `tail = +0.0008`. The run only reached the saturated level near `t ≈ 190–200`, so its "late" statistics mix a transient with the cycle. Run it longer before you quote `5.50%`.
- **m7.** The base flow does not drift. `base_rhs_96 ≈ 3.7e-5` is a stationary residual floor from the limiter and the sonic point: `|W(300) − W(150)| = 2.8e-6` and `|W(600) − W(300)| = 1.6e-6`, and the residual falls only slowly (`4.5e-5 → 2.7e-5`). The FVM mass flux varies by ±0.6% (`0.1196–0.1211`) along the nozzle. FVM vs Chebyshev: `M_in 0.12256` vs `0.12202`, max `|ΔM| = 5.8e-3` (diverging section). This is benign, but the entry should report it as the truncation floor and not as a converged steady state.
- **m8.** The Chebyshev base and the eigen-solver use different inlet closures. The base keeps the momentum equation at `x = 0` (it replaces mass and energy). The eigen-solver keeps the energy equation (it replaces mass, `x`-momentum and angular momentum). Both are valid, but they are inconsistent. The 1C agreement (`0.4–3.5%`) suggests the effect is small.
- **m9.** The `2.5%` `85/15` run (`2.54%`, `tail = -0.0045`, `l/m = 0.74`) is still in transient at `t = 200`. "Slowly decaying" is correct, but the final outcome is undetermined.
- **m10.** The Exp 3 "linear `2T` damping" uses `e_ch` (all-mode energy), not `e_2t`. This is acceptable here, because `e2/e1 ≈ 10^4` and my `e_2t` fits agree to within 3%. Use `e_2t` for clarity.

---

## Explicit checks of every headline claim

| # | Claim | Check | Result |
| :- | :-- | :-- | :-: |
| 1 | Interior Chebyshev operator is the exact linearization | Complex-step Jacobian of my own nonlinear residual (Fourier θ) vs the script `Mat`, 4 cases (`m = 1, 2, 3`; `Re = ∞, 500, 1000`) | **Confirmed**: max diff `≤1.8e-12` against scale `~10^3`; mass matrix diff `≤2e-15` |
| 2 | Inlet BC linearized about the viscous base; zero-gradient outflow | Inspection | Confirmed (see m8 for the closure mismatch) |
| 3 | Flame coupling | Inspection against the PDE filter chain | Confirmed |
| 4 | 1A, 1B, 1C, 1D numbers | JSON vs prose; my Chebyshev rerun (N = 72, 96) | Confirmed (1C spec values cannot be recomputed from the JSON, see M6) |
| 5 | Exp 2 table (all 28 Hopf entries, 7 lift ratios, 7 reacting values) | JSON vs prose; independent rerun at N = 72 and N = 96 | Values confirmed; interpretation qualified (M3, M4, M5) |
| 6 | Non-monotonic `Re` for bare reacting `1T` at `M_c = 0.12` | 11-point `Re` sweep, N = 72/96/120 | **Genuine and converged**; not reported (M4) |
| 7 | Exp 3 table and `2.61×`, `-37%`, 60/40 split | JSON vs prose | Confirmed at `80×200`; grid-sensitive amplitude (M7) |
| 8 | `R120_inf_dualW50_1T` keeps decaying after `t = 300` | New run to `t = 420` | **Confirmed**: `e_ch` slope `-0.032` steady in [300, 350] and [350, 420]; `e_2t` slope `-0.033`, `e_1t` slope `-0.024` to `-0.035`; `E(420) = 8.1e-13`; no regrowth |
| 9 | `R120_inf_dualT15_1T` (tight neck) decays | New run to `t = 420` | **Confirmed**: same slopes (`-0.032`), `E(420) = 1.25e-12`; no hidden slow mode |
| 10 | `21,158×` cascade energy drop | JSON ratio | Confirmed (`21,158`) |
| 11 | Direct `2T` obliterated at `Re = 1000` and `500` (dual-retuned) | JSON vs prose | Confirmed (baseline is inviscid, see m1) |
| 12 | Viscous bell `1T` back-reaction `-0.00555 → +0.00279` | Reproduced `+0.00271` exactly (80); stencil fix → `+0.00202`/`+0.00213`; window → `+0.0026` | **Sign flip confirmed**; magnitude about 25% too high (M1, M2) |
| 13 | Compact nozzle stabilized by viscosity (`+0.00145 → -0.01111`) | Reproduced; one-sided stencil gives `-0.01160` | Confirmed |
| 14 | Hybrid margin halved (`-0.0930 → -0.0434 → -0.0384`) | Long-window refits on 80 and 120 | **Refuted** (C1) |
| 15 | `3.0%` `85/15` all-mode quench at `Re = 1000` | JSON; new `80×200` scope tests | **Confirmed and robust**: `stand_1T` `0.0039%` (tail `-0.0446`), `stand_2T` `0.0003%` (`-0.0426`), `A_0 = 0.08` `0.015%` (`-0.0421`), `M_c = 0.16` `spin_1T` `0.0001%` (`-0.0598`), `M_c = 0.16` `spin_2T` `0.0002%` (`-0.0485`), one-sided stencil `0.013%` (`-0.0426`). Not tested: no-slip, `Re = 500` slip, `120×300` for the new cases |
| 16 | `Re = 500` no-slip, `3.0%` → `4.51%` | New run (`t = 150` wake, `rhs = 1.48e-8` reproduced) | **Reproduced `4.52%`**, but missing from the JSON (M6) |
| 17 | No-slip `2.0%` `5.52%`, `5.50%`, untuned `4.62%` | JSON vs prose | Confirmed (see m6 for the `2T` run) |
| 18 | `R120_Re1000_dualW50_1T` re-ignites at `4.00%` | JSON; analog at 80 with stencil fix → `3.90%` | Confirmed as a nonlinear effect (C1) |
| 19 | Law #16 numbers | JSON | All match except "halving" (C1) and `4.4×` attribution (m2) |
| 20 | Dead End #15 (`+23%`, `62%` wasted, `6.6×`, `4.00%`, `5.52%`) | JSON | Numbers match (`61.7%`, `6.56×`); clause (b) "destabilizes … unless rebalanced" should drop the linear-halving rationale (C1) |

## Prose vs JSON number audit (automated)

`prose_audit.py` extracted every decimal number in the entry and matched it against all JSON values and their derived ratios at the printed precision. The only unmatched values are model parameters (`0.50`, `0.72`, `0.96`, `6.9`), numbers from earlier entries (`4.13`, `5.07`), and the unrecorded values in M6: `4.51`, `0.1221`, `1.737`, `1.74`, `1.828`, and the 1C spec moduli. I found no rounding errors and no sign errors among the recorded values.

## Required changes (summary)
1. Refit the Exp 5 linear rates on long windows or with modal fits. Remove "halving hybrid margin" from the entry, Law #16 and Dead End #15(b). Attribute the `Re = 1000` `4.00%` re-ignition to nonlinear or subcritical dynamics (C1, M2).
2. Fix the cross-baffle x-face θ-stencil, or quantify the bias (`+0.0027 → +0.0020`) (M1).
3. Record `4.51%` (and `8.6%`, `M_in`, `M_e`, spec moduli, `m = 4, 5` Hopf values) in the JSON (M6, M5).
4. Show the `n_v = 0` basis of the Hopf metric in Fig 1 and in the prose. Report the non-monotonic `Re` window at `M_c = 0.12` (M3, M4).
5. State that the `2.61×` gain is specific to `80×200` (`2.83×` on `120×300`, `1.82×` at `Re = 1000`) (M7).
6. Fix the figure titles, the clipping and the legends. Name the baselines of the energy ratios (m1–m5).
