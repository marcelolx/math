# Peer Review #13: Rocket Combustion Entry 016

- **Entry:** `rocket-combustion/entries/016-3d-feed-chugging-triband.md`
- **Code:** `rocket-combustion/sim/exp016_3d_feed_chugging_triband.py`
- **Metrics:** `rocket-combustion/results/exp016_metrics.json`
- **Reviewer:** Claude (Opus 5.5). Independent, read-only, adversarial cross-model review.
- **Date:** 2026-11-27
- **Scratch work:** `/tmp/peer_review_016/` (patched copy of the solver at `sim/e16p.py`, run pickles at `runs/`, logs at `logs/`)

The repository was not modified. All runs used `uv run --project <repo> python -B` from `/tmp/peer_review_016/`.

---

## Verdict: **Major Revisions**

The solver reproduces bit-for-bit. Every re-run of an original case (feed_pure_psi07, tri40seg_1T, tri50_1T, tri50_st1T, tri50_1R, bell_tri50_1L, bell_tri50_1R) matches the JSON to 4 significant figures when I truncate it at the original `t_end`. The pure-feed (`n_p = 0`) linear validation is also sound. Three headline claims do not survive independent checks:

1. The 1D Chebyshev "coupled" chugging benchmark does not match the 3D PDE. At `psi = 0.25` with the 5.0% Tri-Band, the 3D PDE (linear, `A_0 = 1e-4`) gives a **growing** chug mode `+0.00164 + 0.2808i`. The 1D Chebyshev solver gives `-0.0331 + 0.2657i`. So `psi_crit,coupled = 0.1619` is not the 3D threshold, and the straight-chamber "safe" design point (`psi = 0.25`) is linearly chug-unstable in 3D.
2. The straight-chamber "quenched" states (`tri50_1T`, `tri50_st1T`, `tri50_1R`, `tri40seg_1T`) are transients. When extended to `t = 200–220`, all four show a growing `k = 0` chug at `omega = 0.281`. `tri40seg_1T` grows from `P = 1.00%` to `2.66%` (`max|p̄'| = 5.8%`) by `t = 220`.
3. The `xi_H = 0.14` value `-0.3189` (the "3.18×" 1L boost) is a fit-window artifact of two-mode beating. Long-window fits give about `-0.188` (about `1.7×` over bare).

The bell results (1L and 1R exponential quench) do hold up when extended to `t = 200` and `t = 180`.

---

## What Holds Up

- **Reproducibility.** Seven re-runs match the JSON to 4 significant figures (see Table).
- **Feed-ODE linearization.** I derived the inlet and feed linearization by hand. `inlet_choked_3d` with `rho_g = c_g^{2/(gamma-1)}`, `p_g = P_BAR rho_g^gamma` and target flux `m_in` linearizes to `M_c rho' + u' = g_4'` and `p' = c^2 rho'` (with `c = 1`). This matches 1D rows `Mat[n,:]` and `Mat[0,:]`. The orifice term `0.5(m|m|/M_c - M_c)` linearizes to `m'`, and its steady form equals the square-root orifice law `m = M_c sqrt(1 - dp/(psi p))` for forward flow.
- **0D dispersion relation.** I re-derived `theta_g s + (gamma+1)/(2 gamma) + (1/(2 psi)) (1 + s tau_f/4)^-4 / (1 + s tau_inj) = 0` from chamber mass balance, an isentropic chamber, and the compact choked-nozzle condition `u' = (gamma-1)/2 M_c p'`. It is correct for the model as coded.
- **Pure-feed linear validation (`n_p = 0`).** PDE vs. 1D agreement (`0.28–0.61%` in frequency, `|dalpha| < 1.1e-3`) is in the JSON exactly as stated. The 1D result is converged in `N` (`N = 40, 56, 80` agree to 1e-10).
- **`pure_dc_max: null` is not a bug in the physics.** The 1D generalized spectrum has **no** purely real eigenvalue with `|v| < 12` for any `psi` tested. The only real eigenvalue is a spurious `-9.3e4`. The 0D sextic `(10 s + 0.9)(1 + 1.25 s)^4 (1 + s) + 1/(2 psi) = 0` has three complex-conjugate pairs and no real root. For `s >= 0` every term is positive, so no real root can be unstable. "Automatic DC stability" is therefore true, but trivially (there is no DC branch). The JSON field `null` must say this.
- **Bessel zone table.** `r_0 = 0.62761`, areas `6.76% / 35.4% / 39.16%` (total `81.3%`), `C_1R = 4.7844 / 0.0220 / 0.7851`, ratio `217×`: all correct.
- **`n_cav = 1` azimuthal orthogonality** (`dalpha_1T = 4.4e-12`, `dalpha_2T = 7.0e-13`). This is exact by construction (an azimuthal mean of a `cos(m theta)` field is zero on a uniform grid). It is correct, but it is not a physical discovery.
- **Bell Tri-Band 1L and 1R quench.** `bell_tri50_1L` extended to `t = 200`: `P = 0.021%`, envelope slope of `E_ch = -0.021`, bulk mode `-0.0139 + 0.248i`. `bell_tri50_1R` extended to `t = 180`: `P = 0.0009%`, envelope slope `-0.024`. Both decay monotonically. The refined `bell_tri50_ref_1L` (`36×12×40`, `t = 140`, new) also quenches: `P = 0.020%`, slope `-0.0185`, bulk `-0.016 + 0.247i`. `bell_tri50_2T` (new, `t = 120`): decays, `P = 0.13%`.
- **Bell base inlet offset.** The ghost-cell pressure vs. `p_face_base` mismatch on the relaxed `32×10` bell base is `dp_rel = -1.2e-5`. That gives a `0.002%` steady `m_inj` shift. It is negligible.

---

## Critical Findings

### C1. The 1D "coupled" Chebyshev benchmark fails against the 3D PDE. `psi_crit,coupled = 0.1619` is not validated, and `psi = 0.25` is chug-unstable in the straight 3D chamber.

The entry and code docstring say the `1.66×` shift is "verified in 1D Chebyshev and 3D NS PDE". The code contains **no** linear 3D PDE run with `n_p > 0`. Only `n_p = 0` was benchmarked. I ran the missing benchmark (`bulk_k0`, `A_0 = 1e-4`, linear feed, `omega_hp = 0.5`):

| Case (straight, `n_p = 0.34`) | 3D PDE `alpha + i omega` | 1D Cheb | `dalpha` |
| :--- | :---: | :---: | :---: |
| `psi = 0.16`, no cavities, `Re = inf`, `20×8×24` | `+0.0321 + 0.3007i` | `+0.0048 + 0.2999i` | `+0.027` |
| `psi = 0.25`, Tri-Band 5.0%, `Re = inf`, `20×8×24` | `-0.0014 + 0.2811i` | `-0.0331 + 0.2657i` | `+0.032` |
| `psi = 0.25`, Tri-Band 5.0% + baffles, `Re = 500`, `20×12×40` | **`+0.00164 + 0.2808i`** | `-0.0331 + 0.2657i` | `+0.035` |

The 3D PDE is about `0.03` more unstable than 1D whenever the flame is active. With `n_p = 0` the two agree to `< 1e-3`. A likely cause is entropy (indirect) noise. The flame creates entropy spots. The 3D compact-nozzle ghost condition (`u = M_c c` from the actual `c`) responds to them, but the 1D outlet row `u' = (gamma-1)/2 M_c p'` is isentropic and ignores them. The convective time `L_A/M_c = 12.5` at `omega = 0.28` gives a phase of about `3.5 rad`, which is a classic entropy-chug feedback. I did not prove the cause. The discrepancy itself is certain.

Consequences:
- `psi_crit,coupled = 0.1619`, the `1.66×` shift, the "Coupled Trap Zone" bounds, Law #19 and Dead End #18(a) are 1D-model numbers only. The 3D threshold is near or above `0.25`.
- `coupled_psi25_safe` and every straight-chamber Tri-Band run at `psi = 0.25` sit on a linearly unstable chug branch (see C2).

### C2. The straight-chamber "quenched" states are transients that grow into `k = 0` chugging.

I extended four runs. The first 140–180 time units reproduce the JSON exactly.

| Run | Original (`t_end`) | Extended | Bulk mode from `p̄'(t)`, late window |
| :--- | :--- | :--- | :---: |
| `tri40seg_1T` | `t = 140`: `P = 1.00%`, `max|p̄'| = 1.68%`, slope `+0.0023` | `t = 220`: **`P = 2.66%`, `max|p̄'| = 5.81%`, `E_mean = 5.9e-4` (8×), slope `+0.0178`** | `+0.0140 + 0.2834i` |
| `tri50_1T` | `t = 160`: `P = 0.45%`, slope `-0.0021` | `t = 220`: `P = 0.52%`, slope `+0.0047`, `E_mean` envelope rising | `+0.0026 + 0.2809i` |
| `tri50_st1T` | `t = 140`: `P = 0.20%` | `t = 200`: `P = 0.23%`, slope `+0.0033` | `+0.0019 + 0.2808i` |
| `tri50_1R` | `t = 180`: `P = 0.004%` | `t = 220`: `P = 0.0045%`, `E_mean` envelope rising monotonically | `+0.0016 + 0.2808i` |

`tri50_1R` sits at amplitude `~1e-5`, so it is deep in the linear regime. Its growth rate `+0.00164` equals the linear PDE eigenvalue in C1. This is a linear instability, not a subcritical trigger.

Consequences:
- "`1T` cut `40,500×` at 0 extra vol" (`tri40seg_1T`) trades a `1T` cycle for a growing bulk chug. By `t = 220`, `P` is `2.7×` above the headline and still climbing.
- "Quenched", "All transverse modes quenched" and "Complete `1R+1L` quench" for `tri50_*` are wrong descriptions. The `1T`, `2T`, `1R` and `1L` projections do decay, but the chamber is not quenched. The small oscillatory cycles in Fig 3 (left) are this growing chug mode.
- Dead End #18(b) explains `tri50_1L` as a "subcritical `k = 0` compliance rebound" caused by raising cavity volume to 5.0%. The data contradict this. The bulk mode is already linearly unstable at `psi = 0.25` (C1). `tri50_1L` only reaches it sooner because the `1L` bomb loads more energy into the mean pressure.

### C3. The `xi_H = 0.14` value `-0.3189` ("`3.18×` 1L boost") is a fit-window artifact.

Near the exceptional point (EP), the 1D spectrum has two close modes. At `xi_H = 0.14` these are `-0.1743 + 1.959i` and `-0.2011 + 2.180i`, with a beat period of about 28. A single-exponential fit on `[6, 20]` covers half a beat. I re-ran the EP sweep to `t = 40` with dense recording:

| `xi_H` | `[6,20]` | `[10,20]` | `[8,16]` | `[6,40]` | `[15,40]` | 1D Cheb (slowest) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| bare | `-0.1006` | `-0.1082` | `-0.1078` | `-0.1092` | `-0.1102` | `-0.0907` |
| 0.10 | `-0.189` | `-0.068` | `-0.350` | `-0.172` | `-0.207` | `-0.137` |
| 0.12 | `-0.281` | `-0.254` | `-0.360` | `-0.179` | `-0.167` | `-0.156` |
| **0.14** | **`-0.319`** | `-0.372` | `-0.291` | **`-0.188`** | `-0.145` | `-0.174` |
| 0.14 (`nx = 40`) | `-0.339` | `-0.423` | `-0.280` | `-0.226` [6,30] | n/a | `-0.174` |
| 0.14 (linear drag) | `-0.3194` | `-0.3724` | `-0.2914` | `-0.1876` | `-0.1450` | `-0.174` |
| 0.16 | `-0.255` | `-0.288` | `-0.245` | `-0.218` | `-0.188` | `-0.193` |
| 0.18 | `-0.213` | `-0.230` | `-0.216` | `-0.231` | `-0.232` | `-0.209` |

Below the EP (`xi_H <= 0.16`), the fitted rate moves by up to `5×` with the window. Grid refinement (`nx = 40`) does not remove it, so it is not a resolution effect. Above the EP (`xi_H = 0.18`), the fit is stable. On the long window `[6, 40]`, the PDE is monotone and about 8–25% steeper than 1D, with no anomaly at 0.14. The defensible cold `1L` gain at `xi_H = 0.14` is about `-0.188 / -0.109 ≈ 1.7×`, not `3.18×`. The same artifact affects the `n_cav = 5` value `-0.2909` (`2.91×`). The entry's explanation ("enhanced destructive-beating transient near the coupled EP") names the mechanism but still reports the artifact as the headline number.

### C4. The "1L migration wall broken completely" claim is false for the straight chamber. Unfavorable runs are missing from the entry table.

Every straight-chamber run initialized with a `1L` bomb at `psi = 0.25` ends on a stationary or growing state:
- `coupled_psi25_safe` (identical configuration to a "tri40ax_1L"): stationary `1L` cycle, `P = 4.16%`, `E_1L = 3.83e-4`, slope `+1.6e-5`. It is not zero amplitude. The "eliminates chugging" wording is true only for `E_mean`.
- `tri40seg_1L` (in JSON, **not in the entry table**): stationary `P = 4.23%`, `E_1L = 3.97e-4`.
- `tri50_1L`: growing `10.1%` bulk chug.
- `tri50_ref_1L` (`24×16×50`): `P = 4.02%`, `E_1L = 2.7e-4`.

The `534,000×` drop cited for `tri40ax_1R` compares it with Entry 015's `hyb3d_1R` (`E_1L = 3.88e-7`). That run had a different (rigid) inlet and a different flame filter, so the comparison is confounded.

The entry table also leaves out these JSON runs: `bare_1R` (`slope +0.028`), `tri40ax_st1T` (`slope +0.00065`), `tri40ax_2T` (`P = 5.46%`, `max|p̄'| = 3.5%`, `E_mean = 2.4e-4`, **slope `+0.0076`**: a growing chug), and `tri40seg_1L`. All of them are unfavorable.

---

## Moderate Findings

### M1. The nonlinear `feed_pure_psi07` limit cycle is not "saturated purely by the quadratic orifice law".

I instrumented the feed state (`sim/e16p.py`, records of `m_inj` and `g_4`):
- `m_inj / M_c` ranges from **`-1.85`** to `+2.49`. Injector flow is **reversed 27.5% of the time**. Hot gas flows back into the propellant manifold, which is unphysical without a backflow model.
- `g_4 / M_c` falls to `-1.10`. The floor `m_in_now = max(g_4, 0.05 M_c)` is active in **22% of time steps** (12,832 of 58,557). It adds non-physical mass of `2.44` (non-dimensional), which is **14.5%** of total throughput `M_c t = 16.8`.
- Sensitivity of the limit-cycle amplitude `max|p̄'|`: floor `0.001` gives `42.4%`. Floor `0.05` (paper) gives `40.75%`. Floor `0.20` gives `35.5%`. A check valve (`m_inj >= 0`) gives **`28.0%`**. The amplitude depends on unphysical closure choices at the 30% level.
- Milder cases without the floor: `psi = 0.095` grows to `7.7%` (`m_inj` stays in `[0.62, 1.31] M_c`, no reversal, no clamp, still growing at `t = 140`). `psi = 0.09` reaches `21.5%` and is still growing (slope `+0.019`), with reversal 2.9% of the time and no clamp. `psi = 0.15` decays (`-0.038`).

The entry must report `psi_07` as clamp- and reversal-dominated, add a check-valve closure, and use `psi = 0.09–0.095` for a clean nonlinear chug. Even `feed_pure_psi07` is not fully saturated at `t = 140`: the tail slope is `+0.0032` and the late `E_mean` envelope is still rising.

### M2. `psi_crit,coupled` is controlled by the phenomenological `omega_hp`, which the entry says it replaces.

1D Chebyshev with `n_p = 0.34` and Tri-Band 4.0%:

| `omega_hp` | 0.05 | 0.15 | 0.30 | **0.50** | 1.0 | 2.0 | 5.0 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `psi_crit,coupled` | `> 0.6` | **`0.490`** | `0.230` | **`0.162`** | `0.120` | `0.105` | `0.098` |

At `omega_hp = 0.15`, `psi = 0.25` is chug-unstable even in 1D (`+0.0196 + 0.2225i`). The "Coupled Trap Zone" width ranges from about 0 to `5×` depending on one ad hoc filter constant. Law #19 must state this dependence. The phrase "replacing phenomenological `omega_hp`" is accurate only for `n_p = 0`.

### M3. The `1.66×` ratio mixes two configurations.

`psi_crit_pure = 0.0973` has no cavities. `psi_crit_coupled = 0.1619` has the Tri-Band 4.0% cavities (`cav_tri40_ax` is passed in `np_crit_curve` and in the coupled `brentq`). This is why `psi_crit_vs_np[n_p=0] = 0.0946 ≠ 0.0973`. Cavity compliance alone lowers `psi_crit` by 2.8% (`0.0973 → 0.0946`), and by 3.5% with Tri-Band 5.0% (`0.0939`). Consistent ratios are: flame only, no cavities, `0.1702 / 0.0973 = 1.75×`. Both with Tri-Band 4.0%: `0.1619 / 0.0946 = 1.71×`. Fig 1 says `1.67×` and the text says `1.66×`. The `psi = 0.14` "chug-proof alone" value (`-0.0321`) is also cavity-free, while the trap run includes cavities and baffles.

### M4. The Exp 1A "machine-precision equilibrium with active feed ODE" test is vacuous.

`init_type = "quiescent"` sets `u = 0` and sends the solver into the `mode == "closed" or init_type == "quiescent"` branch, which uses reflecting walls at both ends. `inlet_choked_3d` is never called. `p_in_face = p_s[0] = P_BAR`, so `dp_rel = 0` and `feed_err = 0.0` exactly. The feed state never interacts with the flow. The test does verify metric well-balancedness of the closed bell. It does not verify the coupled feed BC. The cons test (`periodic_bump`) is a periodic wavy cylinder, not a bell.

### M5. The inlet thermodynamics described in the entry do not match the code.

§2.1 says "at constant flame temperature `T_g = T̄_c`". `inlet_choked_3d` imposes **isentropic** inflow (`p_g = P_BAR rho_g^gamma`), so `T_g` fluctuates. The 1D row comment ("isentropic T_c = const") is self-contradictory. The 0D `theta_g = L_A/(gamma M_c) = 10.0` is the isentropic-compliance time, not the "chamber gas residence time" (`rho̅ V / m̄ = L_A / M_c = 12.5`). A constant-`T` injector changes chamber compliance by `gamma` and shifts `psi_crit`.

### M6. Bell `1T` and `st1T` are not quenched, and `bell_tri50_st1T` was missing.

I ran the new `bell_tri50_st1T` (`32×10×40`, `t = 120`). It ends on a stationary `1T` cycle with `P = 4.25%`, `E_1T = 4.4e-4`, `1T` frequency `1.722`. Together with `bell_tri50_1T` (`5.49%`), the bell Tri-Band does not quench either `1T` polarization. The entry says "stationary" correctly. The claim "Tri-Band all-5-mode quench in ... curved bell nozzles" in §1 goal 3 and the Verdict overstates this.

### M7. Mixed-baseline ratios.

- `bell_tri50_ref_1T` "`10.11×`" divides the `32×10×40` bare `E_wave` by the `36×12×40` Tri-Band `E_wave`. No refined bare run exists.
- `tri50_ref_1T` "`14.1×` below bare" divides bare `E_wave` (`2.01e-3`) by refined `E_1T` (`1.42e-4`). That mixes two different quantities. The `E_1T` to `E_1T` ratio is `27×`, and the `E_wave` to `E_wave` ratio is `9.5×`.
- "`8.75×` cut" and "`11.88×` cut" are `E_wave` ratios. The table column header next to them is `P`.

---

## Minor Findings

- **m1.** Spurious unstable 1D Chebyshev modes: `+3.91 ± 303i` at `N = 56`, growing as `N^2` (`+7.93 ± 617i` at `N = 80`). The filter `|v| < 12` removes them. They come from strongly imposed BCs without a penalty. Document the filter.
- **m2.** The PDE `1L` fits in Exp 2B use `nonlinear = True` (`orth_*`) for `xi_H = 0.14` and `nonlinear = False` (`ep1l_*`) for the other `xi_H` values. My check shows that the difference is `< 1e-4`, but the inconsistency must be removed.
- **m3.** The "`C_1T`, `C_2T`" columns of the zone table use the radial mean of `J_m` only. Any axisymmetric zone couples with exactly zero strength to `m ≠ 0`. For `n_cav = 5` sectors, the correct coupling has an azimuthal sinc factor. The numbers `0.923`, `1.367` etc. are not coupling coefficients of any configuration that was run.
- **m4.** Several entry table cells merge two runs but show one value (for example `bare_1T / bare_st1T` `E_1R = 8.5e-7`, while `bare_st1T` has `3.6e-5`; `bare_2T / bare_1L` `E_mean = 6.0e-5`, while `bare_2T` has `2.6e-7`).
- **m5.** Fig 1 title says `1.67×`. The JSON and text say `1.66×`.
- **m6.** In `coupled_psi14_trap` the amplitude is still growing at `t = 140` (slope `+0.036`, Fig 1 right). "`68.29%`" is a snapshot, not a limit cycle.
- **m7.** The Exp 2B PDE vs. 1D comparison calls `xi_H = 0.08` "clean" (`-0.1031` vs. `-0.1188`, 13% off) and `0.28` "clean" (`-0.1528` vs. `-0.1405`, 9% off).
- **m8.** The bell `1L`/`1R` "exponential quench" is real but slower than stated. The bulk projection decays at `-0.014` (`omega = 0.248`) through `t = 200`. The `-0.0202` value is an energy-tail fit that includes `1L` decay.

---

## Independent Verification Table

| # | Check | Claimed | Independent Result | Status |
| :-- | :--- | :--- | :--- | :---: |
| 1 | Reproduce `feed_pure_psi07` (`t = 140`) | `40.75%`, `P = 24.40%`, `285%` | `40.75%`, `24.40%`, `285.1%` | ✅ |
| 2 | Reproduce `tri40seg_1T`, `tri50_1T`, `tri50_st1T`, `tri50_1R`, `bell_tri50_1L`, `bell_tri50_1R` at original `t_end` | JSON | Identical to 4 significant figures | ✅ |
| 3 | Inlet linearization `M_c rho' + u' = g_4'`, `p' = c^2 rho'` | Stated | Derived, matches 1D rows | ✅ |
| 4 | 0D dispersion relation | Stated | Re-derived, correct (isentropic chamber) | ✅ |
| 5 | `pure_dc_max` | `null` | No real eigenvalue exists (`|v| < 12`). 0D sextic has 3 complex pairs | ✅ (document) |
| 6 | `psi_crit_pure` | `0.0973` | `0.09731` | ✅ |
| 7 | `psi_crit(n_p = 0)` with Tri-Band 4.0% | `0.0946` | `0.09458` (cavity effect `-2.8%`) | ✅ explains M3 |
| 8 | `psi_crit,coupled`, no cavities | n/a | `0.1702` (`1.75×`) | ⚠️ M3 |
| 9 | `psi_crit,coupled` vs. `omega_hp` | `0.1619` at 0.50 | `0.490 / 0.230 / 0.162 / 0.120 / 0.105` for `0.15 / 0.3 / 0.5 / 1 / 2` | ❌ M2 |
| 10 | 3D linear coupled chug at `psi = 0.25`, Tri-Band 5.0% + baffles | Stable (1D `-0.0331`) | **`+0.00164 + 0.2808i`** (unstable) | ❌ C1 |
| 11 | 3D linear coupled chug at `psi = 0.16`, no cavities | 1D `+0.0048` | `+0.0321` | ❌ C1 |
| 12 | `feed_pure_psi07` flow reversal and clamp | "Saturated purely by orifice law" | Reversal 27.5% of time. Clamp on 22% of steps. Artificial mass 14.5% of throughput | ❌ M1 |
| 13 | Limit-cycle sensitivity to closure | n/a | floor 0.001: `42.4%`; 0.05: `40.8%`; 0.20: `35.5%`; check valve: `28.0%` | ⚠️ M1 |
| 14 | Mild nonlinear chug without floor | n/a | `psi = 0.095`: `7.7%`, still growing. `psi = 0.09`: `21.5%`, still growing. `psi = 0.15`: decays | ℹ️ |
| 15 | Exp 1A quiescent feed test | Feed BC verified | Closed-wall branch, inlet BC bypassed, `feed_err = 0` trivially | ❌ M4 |
| 16 | `ep1l_xi14` fit-window sensitivity | `-0.3189` | `[6,20]`: `-0.319`; `[10,20]`: `-0.372`; `[8,16]`: `-0.291`; `[6,40]`: `-0.188`; `[15,40]`: `-0.145` | ❌ C3 |
| 17 | `ep1l_xi14` at `nx = 40` | n/a | `[6,20]`: `-0.339` (artifact persists, not grid) | ❌ C3 |
| 18 | `tri40seg_1T` to `t = 220` | `P = 1.00%` "40,500× cut" | `P = 2.66%`, `max|p̄'| = 5.8%`, bulk `+0.014 + 0.283i` | ❌ C2 |
| 19 | `tri50_1T` to `t = 220` | "Quenched" | Bulk growing `+0.0026`. `P = 0.52%` | ❌ C2 |
| 20 | `tri50_st1T` to `t = 200` | "Quenched" | Bulk growing `+0.0019` | ❌ C2 |
| 21 | `tri50_1R` to `t = 220` | "Complete `1R + 1L` quench" | Bulk growing `+0.0016` (linear regime) | ❌ C2 |
| 22 | `bell_tri50_1L` to `t = 200` | Exponential quench | `P = 0.021%`, envelope `-0.021`. Bulk `-0.014 + 0.248i` | ✅ |
| 23 | `bell_tri50_1R` to `t = 180` | Exponential quench | `P = 0.0009%`, envelope `-0.024` | ✅ |
| 23b | `bell_tri50_ref_1L` (new, `36×12×40`, `t = 140`) | Not run | `P = 0.020%`, slope `-0.0185`. Bulk `-0.0161 + 0.247i`. Grid-robust quench | ✅ |
| 24 | `bell_tri50_st1T` (new) | Missing | Stationary `1T`, `P = 4.25%` | ⚠️ M6 |
| 25 | `bell_tri50_2T` (new) | Missing | Decays, `P = 0.13%`, envelope `-0.049` | ✅ |
| 26 | `bell_bare_1L` (new, reference) | n/a | Stationary `1L`, `P = 6.7%` (so the bell Tri-Band `1L` quench is a real `>300×` gain) | ✅ |
| 27 | Bell base inlet pressure offset | n/a | `dp_rel = -1.2e-5`, `m_inj` shift `0.002%` | ✅ |
| 28 | Bessel zone `C` values, areas, `r_0`, `217×` | Stated | Exact match | ✅ |
| 29 | Spurious 1D modes | Not mentioned | `+3.9 ± 303i` (`N = 56`), scale as `N^2`, filtered | ⚠️ m1 |
| 30 | `534,000×` E_1L drop | vs. ? | Entry 015 `hyb3d_1R` (different inlet BC) | ⚠️ C4 |
| 31 | Entry/SYNTHESIS numbers vs. JSON | n/a | All quoted numbers trace to the JSON. Exceptions: `1.67×` in Fig 1 title; merged-cell values (m4); mixed-metric ratios (M7) | ⚠️ |

---

## Required Revisions (summary)

1. Add linear 3D PDE benchmarks with `n_p > 0`. Report the 3D `psi_crit,coupled`. Find and fix (or document) the `~0.03` 1D/3D gap. Test an entropy-aware outlet row in 1D first.
2. Extend all straight-chamber `psi = 0.25` runs to `t >= 220` and report the growing `k = 0` mode. Retract "quenched" and "1L wall broken" for the straight chamber. Rewrite Dead End #18(b).
3. Replace single-window `alpha` fits near the EP with long windows or a two-mode fit. Retract `3.18×` and `2.91×`.
4. Report flow reversal and clamp activity in Exp 1C. Add a check-valve closure and a `psi ≈ 0.09` clean case.
5. State the `omega_hp` dependence of `psi_crit,coupled` in Law #19. Use consistent cavity configurations in the `psi_crit` ratio.
6. Make the Exp 1A feed test actually call `inlet_choked_3d` (open inlet, mean flow, `t ≥ 50`). Correct the "constant `T_g`" statement.
7. Add the omitted unfavorable runs (`tri40ax_2T`, `tri40ax_st1T`, `tri40seg_1L`, `bare_1R`) and `bell_tri50_st1T` to the entry tables.
