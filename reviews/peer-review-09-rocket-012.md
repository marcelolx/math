# Peer Review #9: `rocket-combustion` Entry `012` (2D Transonic CD Nozzle)

- **Reviewer:** independent cross-model adversarial audit (Claude Opus 5.5, read-only)
- **Date:** 2026-11-26
- **Scope:** `entries/012-2d-transonic-cd-nozzle.md`, `sim/exp012_2d_transonic_cd_nozzle.py`, `results/exp012_metrics.json`, `results/exp012_*.png`, candidate Law #15 in `SYNTHESIS.md`
- **Constraint:** strictly read-only. All scripts, logs and run histories are in `/tmp/peer_review_012/`. `git status -s` was empty at the start and at the end.

## Verdict: **Major Revisions**

The linear spectral core is sound. An independent solver reproduces every Chebyshev eigenvalue in the entry to 5 digits, and the Hopf maps and bell cutoff-collapse values are correct for the stated contour. The solver uses a sympy-derived **conservative flux-form** linearization and a separate multi-domain Chebyshev discretization.

However, three headline claims in Law #15 do not survive refinement or a longer time horizon:

1. **"Zero-hardware annular slot decays the bare 6.9% shock 42×" is a grid artifact.** On `120×300` the bare annular slot does not quench. It holds a growing, 2T-dominated rotating limit cycle at **P = 3.66% at t = 160 and 3.97% at t = 300** (the `80×200` value is 1.57%, and the `64×144` value is 0.36%).
2. **"Annular + hybrid: monotonic quench by 523× with zero 2T regrowth" is also a grid artifact.**
   - On `120×300` the identical 1T bomb ($A_0 = 0.055$) decays to t ≈ 77. It then regrows through the 2T cascade and locks onto a **5.07% pure-2T limit cycle** (E = 5.57e-4, e2/e1 = 416). This is the same cycle that a direct `spin_2T` bomb reaches (4.77% on `80×200`, 5.07% on `120×300`).
   - The bell hybrid on `120×300` ends at 5.54% (E = 6.6e-4).
   - The resolved annular/bell energy ratio is about 1.2×, not 523×. Both hybrids fail by the same 2T cascade.
3. **The Exp 2 "% of compact MC (−0.366%)" column compares unlike quantities.** Entry 009's −0.366% is the **inlet-reflected (superposed) peak at t = 11.01**, not a first pass. It was measured with a **smaller entropy amplitude** ($\delta s = 0.05$, $\delta\rho/\rho = -3.92\%$ vs. 012's $-5\%$) on a coarse `nx = 48` grid. The like-for-like compact first pass, rescaled to 012's amplitude, is about **−0.37%** (`nx = 192`). It is not a fixed −0.366% "100%" anchor.

There are also important secondary issues:
- The near-marginal bell rates (`−0.0006`, "46×") change sign under a different smooth contour.
- A **second cold inviscid instability (bell, $\bar M_c = 0.10$, m = 4, `+0.00369`)** is missed because the Hopf map only scans m ≤ 3.
- The "per-mode numerical damping" for the annular 1T (`+0.0086`) is a fit-window and beating artifact.
- Prose-JSON consistency is good. Only a few ratio and labelling slips occur.

---

## 1. Independent re-derivation of the governing PDE and linear operator (Target 1)

### 1.1 Conservative system
For a streamtube of area $A(x)$ (per radian), mean radius $R(x)$, gap $h = A/R$, and arc length $R\,d\theta$, the conservative fluxes are as follows. The $\theta$-face area is $h\,dx$ and the volume is $A\,dx\,d\theta$:

| Eq. | $\partial_t$ | $x$-flux | $\theta$-flux | Source |
|---|---|---|---|---|
| mass | $A\rho$ | $A\rho u$ | $(A/R)\rho v$ | 0 |
| $x$-mom | $A\rho u$ | $A(\rho u^2+p)$ | $(A/R)\rho u v$ | $pA' + A\rho v^2 R'/R$ |
| ang-mom | $A\rho R v$ | $A\rho u R v$ | $A(\rho v^2+p)$ | 0 |
| energy | $AE$ | $Au(E+p)$ | $(A/R)v(E+p)$ | flame |

- **Confirmed.** The angular-momentum equation in non-conservative form gives a Coriolis term $-u v R'/R$ on $v$. The work of this term, $-\rho u v^2 R'/R$, cancels the centrifugal work $+\rho u v^2 R'/R$ on $u$ exactly, so the energy equation has no geometric source. Mass, angular momentum and energy are exactly conservative.
- **Scope caveat (Moderate, M5).** The model omits the meridional metric $\sqrt{1+R'^2}$ and any radial velocity. For the bell, $R' = A'/(2\sqrt{A})$ reaches $\approx 1.2$ at $L_n = 0.5$ (steep wall), so the omitted metric is $O(1)$, not small. The model is a quasi-2D "streamtube on a cone" model. Do not present it as a bell-nozzle flow.

### 1.2 Linear operator: independent reproduction
- **Method.** `indep_eig.py` derives $F_q, G_q, S_q, J=\partial W/\partial q$ symbolically from the conservative fluxes (sympy). It then forms $q'_t = J^{-1}[-D_x(F_q q') - i m G_q q' + S_q q']$. This is flux-form differentiation of the products, not the repo's hand-derived primitive form.
- **Base flow.** The base flow and geometry use independently coded functions.
- **Inlet conditions.** The inlet uses an isentropic injector, $(\rho u)'=0$ and $v'=0$. A variant keeps the outgoing $u-c$ characteristic row instead of the raw $p$-row.

**Table 1: repo vs. independent eigenvalues (cold, N = 120).**

| Case | Repo (JSON) | Independent | Inlet-char variant |
|---|---|---|---|
| annular $\bar M_c=0.12$, $L_n=0.50$, 1T | −0.07290 + 0.8990i | −0.07290 + 0.8990i (2nd: −0.07665 + 1.0946i) | −0.07290 |
| bell 0.12, 0.50, 1T | −0.02448 + 1.0924i | −0.02448 + 1.0924i | −0.02451 |
| bell 0.12, 0.70, 1T | −0.00377 + 1.0575i | −0.00377 + 1.0575i | −0.00376 |
| bell 0.12, 0.80, 1T | −0.00061 | −0.00061 | −0.00060 |
| bell 0.18, 0.50, 1T | −0.02803 | −0.02803 | −0.02804 |
| annular 0.16, 0.50, 2T | −0.04315 | −0.04315 | −0.04316 |
| bell 0.16, 0.50, 2T | −0.00865 | −0.00865 | −0.00866 |
| compact MC 0.12 (own outlet BC $M'=0$) | −0.02843 + 0.9893i | −0.02843 + 0.9893i | n/a |

**Spectral convergence (N = 60, 84, 100, 120, 140):** the drift is below $5\times10^{-5}$ for all cases and below $5\times10^{-6}$ between N = 120 and 140. **Confirmed.**

### 1.3 m-sweep (cold, $\bar M_c = 0.12$, $L_n = 0.50$; leading mode in $\omega \in [0.05, 7]$)

| m | annular | bell |
|---|---|---|
| 0 | −0.08193 (1.685) | −0.08193 (1.685) |
| ±1 | −0.07290 (0.899) | −0.02448 (1.092) |
| ±2 | −0.02668 (1.997) | −0.00853 (2.009) |
| ±3 | −0.04928 (3.427) | −0.02096 (3.109) |
| ±4 | −0.03227 (3.990) | −0.01147 (4.554) |
| ±5 | −0.07316 (6.022) | **−0.00513 (5.430)** |

- $\pm m$ are identical (no mean swirl), as expected.
- The bell **m = 5 (−0.0051)** and **m = 4 (−0.0115)** are as weakly damped as 2T. The Hopf map and Law #15 ignore them. The reacting check (§4.2) shows that their $n_{pH} > 0.85$ at $\bar M_c = 0.12$, so the global value 0.060 stands. At $\bar M_c = 0.10$, however, see Finding C4.

### 1.4 $L_d$ (exit Mach) sensitivity: **passes**

| $L_d$ | $M_e$ ($L_n=0.5$) | annular 1T | bell 1T | bell $L_n$=0.7 1T | bell 0.16 2T |
|---|---|---|---|---|---|
| 0.15 | 1.768 | −0.07289 | −0.02448 | −0.00377 | −0.00865 |
| 0.25 | 1.825 | −0.07290 | −0.02448 | −0.00377 | −0.00865 |
| 0.40 | 1.848 | −0.07290 | −0.02449 | −0.00377 | −0.00866 |

The supersonic exit decouples the outlet, as claimed. Note: the entry says $M_e \approx 1.83$ and the JSON nx-converged value is 1.8266. My exact value at $L_d = 0.25$ is 1.8252 (minor).

### 1.5 Contour-shape sensitivity ($C^2$ quartic vs $C^1$ cosine converging section): **fails for near-marginal claims**

| Case | quartic ($C^2$) | cosine ($C^1$), N = 140 | Change |
|---|---|---|---|
| annular 0.12, 0.50, 1T | −0.07290 | −0.06457 | −11% |
| bell 0.12, 0.50, 1T | −0.02448 | −0.02482 | +1% |
| bell 0.12, 0.70, 1T | −0.00377 | −0.00269 | −29% |
| bell 0.12, **0.80**, 1T | **−0.00061** | **+0.00186 (cold-unstable!)** | sign change |
| bell 0.18, 0.50, 1T | −0.02803 | −0.02681 | −4% |
| bell 0.12, 0.50, **2T** | **−0.00853** | **−0.00056** | 15× |
| bell 0.16, 0.50, 2T | −0.00865 | −0.00722 | −17% |
| bell 0.10, 0.50, m=3 whistle | +0.00224 | +0.00691 | 3× |

The qualitative bell collapse (1T damping well below MC at $L_n \ge 0.5$) is robust. The quantitative ratios "46×" and "7.5×", and the 2T-limited Hopf value 0.060, depend on the precise contour. They are not physical invariants (see M1).

---

## 2. Compact limit and Exp 1D PDE verification (Target 2)

### 2.1 Annular $L_n \to 0$: converges to Marble–Candel, but the repo's own short-$L_n$ values are under-resolved
- **Method.** The multi-domain Chebyshev solver (`md_eig.py`) splits the domain at $[0, L_a]$, $[L_a, x_t+\delta]$ and $[x_t+\delta, L_{tot}]$. Interfaces use characteristic upwind patching.
- **Validation.** At $L_n = 0.5$ it reproduces the single-domain value to $1\times10^{-5}$.

**Table 2: 1T, annular, multi-domain (two highest resolutions agree to ≤ 3e-4).**

| $L_n$ | $\bar M_c = 0.12$ (MD) | repo `compact_limit` / `Ln_curves` | $\bar M_c = 0.18$ (MD) | repo |
|---|---|---|---|---|
| 0.01 | −0.02827 + 0.9863i | n/a | n/a | n/a |
| 0.02 | −0.02846 + 0.9851i | n/a | n/a | n/a |
| 0.05 | −0.02894 | n/a | n/a | n/a |
| 0.08 | −0.02972 | **−0.03554 (+20%)** | n/a | −0.11456 |
| 0.10 | −0.03041 | −0.03361 (+11%) | −0.11471 | −0.11671 |
| 0.12 | −0.03124 | −0.03178 | n/a | −0.11795 |
| 0.15 | −0.03276 | −0.03552 (+8%) | −0.12247 | −0.12354 |
| 0.20 | −0.03601 | −0.03625 | −0.11389 | −0.11396 |
| **0 (MC)** | **−0.02843 + 0.9893i** | n/a | **−0.09933** | n/a |

- **Result.** The annular nozzle **does** converge smoothly to Marble–Candel ($L_n = 0.02$: −0.02846 vs. −0.02843). This finding supports the model.
- **Problem.** The repo's single-domain N = 84 to 120 values at $L_n \le 0.15$ are 8 to 20% too damped. In the JSON they show a spurious non-monotonic bump (`0.08 → −0.0355, 0.12 → −0.0318`).
- Bell short-$L_n$ values are similarly off: $\bar M_c = 0.12$, $L_n = 0.10$ gives repo −0.03553 vs. MD −0.03251 (+9%), and $L_n = 0.15$ gives −0.03928 vs. −0.03690.
- **Effect on claims.** These values affect only the left edge of Fig. 1a and the `compact_limit_annular` block. They do not affect the headline claims.

### 2.2 `ann_M012_Ln050_1T` PDE discrepancy (−0.06425 vs. −0.07290)

| Grid | fit $t\in[20,52]$ (repo window) | $[40,80]$ | $[60,100]$ |
|---|---|---|---|
| `96×200` | −0.0642 (reproduces JSON) | −0.0794 | **−0.0719** |
| `144×300` | −0.0622 | −0.0785 | **−0.0735** |
| exact | −0.07290 (1st) / −0.07665 (2nd) | | |

- **Explanation.** The `+0.0086` "numerical damping" entry is **not** numerical damping. The two branches −0.0729 + 0.899i and −0.0767 + 1.095i beat with $T_{beat} = 2\pi/0.196 = 32.1$, which is exactly the length of the 32-unit fit window. On $[60, 100]$ both grids agree with the exact value to 1.4% and 0.8%.
- **Asymmetry.** A refinement from 96 to 144 does not move the short-window fit toward exact. This proves that the bias comes from the window and not from resolution.
- **Effect on the table.** Table 1D therefore mixes true numerical damping (2T rows) with a window artifact (annular 1T row).

---

## 3. Exp 2 indirect entropy-noise filtering (Target 3)

### 3.1 The −0.366% anchor is not a first-pass compact reference (Critical, C3)
I re-ran Entry 009's own `entropy_swirl_pulse` protocol (`t4_e009.py`, read-only import):

| Entry 009 grid | 1st pass (t ≈ 8.03) | inlet-reflected peak (t ≈ 11.0) |
|---|---|---|
| `nx = 48` (as published) | −0.236% | **−0.366%** (the published number) |
| `nx = 96` | −0.277% | −0.429% |
| `nx = 192` | −0.291% | −0.451% |

The two protocols differ in several ways:
- **Peak type.** Entry 009's −0.366% is the 2nd-pass superposed peak (009's own text says "center arrival 7.92 plus one acoustic round-trip 3.04"). Entry 012's table puts it in the "1st-pass, $t = 7.92$, 100%" row.
- **Amplitude.** Entry 009 sets $\rho = e^{-s/\gamma}$ with $s = 0.05$, which gives $\delta\rho/\rho = -3.92\%$, plus uniform swirl 0.04. Entry 012 sets $\delta\rho = -5\%$, which is **1.276×** larger.
- **Grid.** The 009 value is grid-unconverged (−0.366 → −0.451 from `nx` = 48 to 192).

The like-for-like compact first pass at 012's amplitude is about **−0.291 × 1.276 ≈ −0.371%** (nx = 192, still converging). The like-for-like superposed peak is about **−0.575%**. The "96.7% / 91.5% / 88.0% / 84.2% of compact MC" column and "−15.8% filtered" therefore have no valid reference.

### 3.2 Independent Exp 2 re-runs (`t11_entropy.py`, $t_{end} = 32$, 012 solver)

| $L_n$ | 1st pass, repo grid (bell = annular) | 1st pass, 2× grid | 2nd pass (true), repo grid | repo "2nd-pass" |
|---|---|---|---|---|
| 0.15 | −0.3539% (t = 8.65) | −0.3676% | −0.5397% (t = 11.79) | −0.5397% |
| 0.35 | −0.3349% | n/a | −0.5021% | −0.5021% |
| 0.70 | −0.3219% | −0.3441% | −0.4167% | −0.4167% |
| 1.10 | −0.3083% (t = 15.06) | −0.3257% | **−0.3143% (t = 18.54)** | **−0.3083% (= 1st pass)** |
| 0.05 | −1.13% (spurious, base not steady) | −0.45% / −0.66% | n/a | not run |

Findings:
- **Bell and annular are identical to 4 digits for m = 0** ($R$ does not enter an axisymmetric $v = 0$ problem). The entry labels the run "bell" without saying this. Add a note.
- **The $L_n = 1.10$ "2nd-pass superposed peak" equals the first pass.** The true 2nd pass arrives at $t = 18.54$, outside the hard-coded $[6, 18]$ window ($t_{end} = 19$). The "−42.9%" happens to land near the true −41.8%, but the tabulated value is mislabelled (M3).
- **The first-pass amplitude is grid-sensitive by 4 to 7%** (repo grid vs. 2×). That is the same order as the claimed 3 to 16% filtering. The monotonic trend survives refinement (2× grid: −0.368 → −0.344 → −0.326), but the magnitudes do not.
- **The $L_n \to 0$ recovery is not demonstrated by the PDE.** $L_n = 0.05$ is unresolvable at this resolution, so the base flow is not steady and spurious early minima appear.
- The 2nd-pass echo and inlet reflection do not affect the qualitative "longer nozzle → weaker, later echo" claim. They do affect every quantitative ratio.

---

## 4. Exp 3: bell cutoff collapse, Hopf maps and cold whistle (Target 4)

### 4.1 Cutoff mechanism check
$\omega_{cut}(x) = m c \sqrt{1-M^2}/R$ along the bell (1T):

| $\bar M_c$ | chamber $\omega_{cut}$ | max $\omega_{cut}$ in cone (at $M \approx 0.5$) | 1T frequency |
|---|---|---|---|
| 0.12 | 0.993 | **1.632** ($R = 0.523$) | 1.092 |
| 0.18 | 0.984 | 1.341 | 1.058 |

- An evanescent barrier exists. Because $\sqrt{1-M^2} \to 0$ at the throat, $\omega_{cut}$ returns to 0 there, so the barrier is a finite-width **tunnelling** barrier, not a turning point that reflects completely. Its height does not depend on $L_n$ and its width scales with $L_n$. This is consistent with reflection that grows with $L_n$.
- The damping is **non-monotonic** in $L_n$. At $\bar M_c = 0.12$ the bell is **more** damped than the annular slot for $L_n \le 0.42$ ($L_n = 0.35$: bell −0.0584 vs. annular −0.0511, and bell = 2.05× MC). Law #15's general statement that "compact MC severely overestimates radiation in realistic bell nozzles" holds only for $L_n \gtrsim 0.45$.
- The entry gives $R_t = 0.453$, but $\sqrt{A_t} = 0.4497$ at $\bar M_c = 0.12$ (minor).
- In the annular case the "1T" at $\omega = 0.899$ is **below** the chamber cutoff 0.993. It is an axially evanescent (non-propagating) chamber mode. Its high damping therefore does not come from "radiation through an unobstructed throat" as Law #15 states.

### 4.2 Collapse values and Hopf map: **confirmed** (independent solver, N = 84, robust n_p grid scan + bisection, m = 0…5)

| | repo | independent |
|---|---|---|
| bell 1T $L_n$ = 0.70 / 0.80 | −0.00377 / −0.00061 | −0.00377 / −0.00061 |
| Hopf annular / bell $\bar M_c = 0.12$ | 0.2284 / 0.0602 | 0.2283 / 0.0602 → **ratio 3.79** ✓ |
| annular 1T-only $\bar M_c = 0.12$ | 0.4303 | 0.4302 |
| bell $\bar M_c = 0.10$ / 0.16 | 0.000 / 0.0824 | 0.000 / 0.0825 |
| annular $\bar M_c = 0.10$ / 0.16 / 0.18 | 0.104 / 0.2046 / 0.2959 | 0.104 / 0.2046 / 0.2959 |
| m = 4, 5 $n_{pH}$ ($\bar M_c$ = 0.12, 0.16) | not scanned | > 0.85 (do not limit) |

### 4.3 The cold Coriolis-swirl whistle (bell, $\bar M_c = 0.10$, m = 3)

| Check | Result |
|---|---|
| single-domain N = 72, 100, 128, 140 | +0.00219, +0.00226, +0.00224, +0.00224 (3.5598i) |
| multi-domain (70, 60, 50) / (100, 80, 60) | +0.002236 / +0.002239 |
| characteristic inlet row | +0.00223 |
| $C^1$ cosine contour | **+0.00691** (3× larger) |
| annular, same $\bar M_c$ | −0.0256 (no whistle) |
| **another cold instability: bell $\bar M_c = 0.10$, m = 4** | **+0.00369 + 4.533i** (not reported) |
| viscous (ν = 1/Re on $u', v'$; Neumann outflow) Re = 1e6 | +0.00164 |
| Re = 1e5 / 5e4 / 2e4 | −0.0044 / −0.0108 / −0.0198 |
| Re = 5000 / 2000 / 500 | −0.0408 / −0.0471 / −0.0642 |
| 2D PDE `96×200` / `144×300`, t ∈ [80, 150] | −0.0407 / −0.0307 (numerical m = 3 damping ≈ −0.033 to −0.043 hides it) |

Verdict on the whistle:
- **The whistle is a genuine eigenvalue of the inviscid model.** It converges and is independent of the discretization.
- **The 2D PDE cannot confirm it.** The vorticity wavelength $2\pi/35.6 = 0.18$ spans only about 11 cells on `144×300`, so numerical damping (≈ −0.03) is 15× the growth rate.
- **The whistle disappears at Re ≲ 2×10⁵.** This is much weaker viscosity than Entry 011's Re = 500 to 5000, so the caveat in the entry is correct and conservative.
- **The proposed mechanism is not supported by the eigenvector.** The $|v'|$ ratio throat / $x = L_a$ is **0.26**, not the $R(L_a)/R_t = 2.43$ amplification that "$u'_\theta \propto 1/R$" predicts.
- **Effect on the Hopf table.** The $\bar M_c = 0.10$ bell Hopf "0.000" entry must be marked inviscid-only. The viscous-relevant bell value is limited by 2T (0.0183).

### 4.4 Thin-shell vs. 3D scope (Moderate, M5)
- **Azimuthal wavenumber.** The thin-shell model uses $k_\theta = m/R$. A 3D cylindrical duct uses $s_{mn}/R$ (for example $s_{11} = 1.841$, $s_{21} = 3.054$, $s_{01} = 3.832$). In a real bell the chamber 1T frequency is therefore about 1.84× higher relative to the 1L family. Strouhal phases ($\omega L_a/\bar u$) and the "half-integer Strouhal tuning" claims change accordingly.
- **Radial structure and wall slope.** A 3D bell has radial structure (the mode shape shifts toward the wall as $R$ contracts) and a non-small wall slope. The barrier height in a real bell is $s_{11}/R$-scaled, so the *qualitative* barrier survives. The *magnitudes* (7.5×, 46×, Hopf 0.060, 3.79×) and the plug-nozzle analogy ($R \equiv 1$ with a shrinking gap is an annulus whose radial modes are ignored) are specific to the model.
- **Required wording.** Law #15 must say "in the 2D thin-shell model".

---

## 5. Exp 4: nonlinear rotating-shock control and long-horizon fate (Target 5)

All runs use the repo's own `simulate_2d_transonic` through a read-only import (`pde_patch.py` adds `spin_3T` only). `NL_ann50_bare_1T_80` at t = 160 reproduces the JSON exactly (P = 1.575%, E = 4.715e-5, l/m = 0.374).

### 5.1 Exact reacting eigenvalues ($n_p = 0.22$, $n_v = 0.04$; both chiralities, full window $\omega \in [0.05, 7]$)

| Config | m = −1 (repo 1T) | m = +1 | **other m = ±1 mode (ω ≈ 1.85–2.07)** | m = −2 (repo 2T) | m = +2 |
|---|---|---|---|---|---|
| bell 0.50 | **+0.02204** ✓ | −0.00176 | −0.0156 | **+0.02687** ✓ | +0.01798 |
| bell 0.20 (MD-converged) | **+0.01473** (repo N = 84: +0.01438) | −0.01136 | −0.0131 | −0.01631 ✓ | −0.0451 |
| annular 0.50 | −0.02375 ✓ | −0.05303 | **−0.0132 (ω = 1.854)** | **+0.00225** ✓ | −0.00434 |

The repo's m = 1 window (0.55–1.50) excludes an m = 1 mode at ω ≈ 1.85 that is **less damped** than the reported annular "1T" (−0.0132 vs. −0.0237). The annular m = 1 margin is therefore −0.013, not −0.024 (Minor, m3).

### 5.2 Linear PDE refinement studies (fit $t \in [18, 55]$ unless noted)

| Run | `80×200` | `120×300` | `160×400` | exact |
|---|---|---|---|---|
| `lin_bell20_bare_1T` | +0.0244 (repro) | +0.0207 | +0.0185 | +0.0147 |
| `lin_ann50_bare_2T` | −0.0188 | −0.0067 | −0.0020 | +0.00225 |

- **bell20 excess.** The `+0.010` excess of `lin_bell20` over exact is a **grid artifact**. It shrinks as +0.0097 → +0.0060 → +0.0038 (order ≈ 1.2–1.6), and Richardson extrapolation gives about +0.015 ≈ exact. The probable cause is the under-resolved reflection phase of the steep $L_n = 0.2$ contraction (only about 7 cells across the nozzle on `80×200`).
- **Annular 2T.** The PDE converges toward the exact unstable value from below. The `80×200` grid hides about −0.021 of numerical 2T damping, and `160×400` still hides about −0.004.

### 5.3 Long-horizon nonlinear fate ($A_0 = 0.055$)

| Run | t = 160 (repo stats) | t ∈ [200, 240] | t ∈ [240, 300] | Fate |
|---|---|---|---|---|
| `NL_ann50_bare_1T` `80×200` | P = 1.575%, l/m = 0.374, e2/e1 = 0.53 | 0.244% | 0.091%, e2/e1 = 21, α = −0.019 | decays on this grid |
| **`NL_ann50_bare_1T` `120×300`** | **P = 3.66%, l/m = 1.13, e2/e1 = 1.38** | **3.82%** | **4.00%, e2/e1 = 4.6, α = +0.001** | **does NOT quench: growing 2T-dominated limit cycle** |
| `NL_ann50_bare_2T` `80×200` | 2.50% | 1.02% | 0.43% | decays (2T numerically over-damped) |
| `NL_bell50_hyb_1T` `80×200` | P = 2.65%, e2/e1 = 1.69 | 5.17%, e2/e1 = 609 | **5.25%, e2/e1 = 638, α ≈ 0** | **locks onto the 5.25% pure-2T cycle** (E = 5.28e-4, identical to the 2T bomb) ✓ |
| `NL_bell50_hyb_2T` `80×200` | 5.25% | 5.25% | 5.25% | stationary 2T cycle ✓ |
| `NL_ann50_hyb_1T` `80×200` | 0.449%, e2/e1 = 615 | 0.152%, e2/e1 = 4.6e4 | 0.079%, α = −0.0134 | monotonic decay of a **pure-2T** residual |
| **`NL_ann50_hyb_1T` `120×300`** | **P = 4.79%, l/m = 4.82, e2/e1 = 535** ($E_{min}$ = 8.3e-5 at t = 77, then regrowth) | **5.07%, e2/e1 = 416** | **5.07%, α ≈ 0** (E = 5.57e-4) | **does NOT quench: 2T cascade regrowth locks onto the same 5.07% pure-2T cycle as the 2T bomb** |
| `NL_bell50_hyb_1T` `120×300` | P = 3.35%, e2/e1 = 0.18 | 5.42%, e2/e1 = 365 | **5.54%, α ≈ 0** (E = 6.6e-4) | 2T lock-in confirmed on the fine grid ✓ |
| **`NL_ann50_hyb_2T` `80×200`** (new) | **4.77%, l/m = 1.000** | 4.77% | **4.77%** | **sustained pure-2T limit cycle** |
| **`NL_ann50_hyb_2T` `120×300`** (new) | **5.07%** | **5.07%** | n/a (t = 240) | **sustained pure-2T limit cycle** |

Answers to the specific questions:
- **Does `NL_ann50_bare_1T_80` truly quench?** Only on `80×200` and coarser. The fine `120×300` grid shows a 3.7–4.0% limit cycle that grows slowly. This is consistent with the exact unstable bare-annular 2T (+0.00225) plus a nonlinear 1T↔2T cascade. The "42× below Bare Bell" and "zero hardware" claims are **not grid-converged**. Their P values in three grids are 0.36%, 1.57% and 3.66–4.0%: the trend moves **away** from quench as the grid gets finer.
- **Does `NL_bell50_hyb_1T_80` lock onto 5.25%?** Yes. From t ≈ 200 it reaches 5.25% with e2/e1 = 638, so the cascade claim is confirmed. The t = 160 "P = 2.65%" in the table is a transient and should not be quoted as an outcome.
- **Does `NL_ann50_hyb_1T` stay quenched?** **No, on `120×300`.** It regrows to a 5.07% pure-2T cycle. On `80×200` it does stay quenched, at α = −0.0134 to t = 300. The residual is, however, a **pure-2T** state (e2/e1 = 615 at t = 160 and 5×10⁴ at t = 300), so the "zero 2T regrowth" wording hides that the survivor *is* 2T. Its decay rate equals the `80×200` linear hybrid 2T rate (−0.0138), and about −0.01 of that is numerical. On the `120×300` grid the nonlinear 2T branch wins.
- **2T bomb on the annular hybrid:** a sustained **4.77% / 5.07%** pure-2T limit cycle. The bell vs. annular difference in hybrid 2T upper branches is 5.25% vs. 4.77–5.07%, so it is marginal.

---

## 6. Prose-vs-JSON consistency audit (Target 6)

`t7_audit.py` extracts all 247 back-ticked tokens from the entry and 26 from Law #15 and matches every numeric atom against all 885 JSON leaves (with display rounding). It also recomputes every derived ratio.

| Claim | JSON-derived | Status |
|---|---|---|
| 7.5× / 46× / 3.5× / 6.2× | 7.544 / 46.52 / 3.543 / 6.209 | ✓ |
| 2.56× / 19.4× | 2.565 / 19.35 | ✓ |
| 3.79× / 2.48× / 1.90× / 1.09× / 1.33× / 3.22× | 3.7955 / 2.481 / 1.896 / 1.092 / 1.333 / 3.215 | ✓ |
| −63% | −63.4% | ✓ |
| 42× | 41.52 | ✓ (rounded) |
| **523×** | **521.4** | ✗ minor (should be 521×) |
| −15.8% / −42.9% | −15.76% / −42.88% | arithmetic ✓; basis invalid (C3, M3) |
| $R_t = 0.453$ | $\sqrt{A_t} = 0.4497$ | ✗ minor |
| −0.366% | not in 012 JSON (009 superposed peak) | ✗ basis (C3) |
| conservation `4.4e-16`, `2.3e-16`, `2.0e-17`, `1.4e-16` | 4.44e-16, 2.28e-16, 1.95e-17, 1.42e-16 | ✓ |
| "spectral convergence < 5e-6 across N = 96 → 120" | max 4.98e-6 | ✓ (just) |
| "max\|∂_t W\| ~ 5e-13" | nx = 100: 2.6e-12 (loop hit $t_{relax}$, not the threshold) | minor |
| Exp 1D note "7.5× … relative to compact MC −0.0993" | 7.5× is relative to −0.0284 (M = 0.12) | ✗ wording |

All other numeric atoms (E values, P values, l/m, e2/e1, eigenvalues) match the JSON. **The prose faithfully transcribes the JSON. The problems lie in the interpretation and the reference baselines, not in the transcription.**

Other cosmetic items:
- Fig. 2 legends render literal back-ticks (`` `80×200` ``).
- The script docstring lists Exp 2 $L_n \in \{0.10, 0.25, 0.50, 0.75\}$, but the code uses {0.15, 0.35, 0.70, 1.10}.
- The docstring also says "N = 72 → 96 → 120" while the code runs 96 → 120.
- The `mc_limiter` alias actually points to van Leer.

---

## 7. Findings

### Critical
- **C1: Bare annular "decay / 42× / zero hardware" is a coarse-grid artifact.**
  - On `120×300` the bare annular slot sustains a growing 3.7–4.0% 2T-dominated rotating cycle to t = 300 (§5.3).
  - The P value increases monotonically with resolution (0.36 → 1.57 → 3.66%).
  - Linear PDE refinement shows that the exact bare 2T (+0.00225) is hidden by about −0.021 numerical damping on `80×200`.
- **C2: "Annular + hybrid monotonically quenches the 6.9% shock 523× with zero 2T regrowth" is a coarse-grid artifact.**
  - On `120×300` the *same* 1T bomb reaches a minimum at t ≈ 77 ($E = 8.3\times10^{-5}$). It then regrows through the 1T→2T cascade and locks onto a **5.07% pure-2T limit cycle** ($E = 5.57\times10^{-4}$, e2/e1 = 416, α ≈ 0 for t ∈ [160, 300]).
  - A direct 2T bomb gives the same cycle (4.77% on `80×200`, 5.07% on `120×300`).
  - The bell hybrid on `120×300` ends on 5.54% ($E = 6.6\times10^{-4}$).
  - The resolved annular-vs-bell hybrid energy ratio is therefore **1.19×, not 523×**.
  - On resolved grids, both hybrids fail by the identical 2T cascade mechanism that the entry attributes only to the bell. Law #15 item 2 and the "523×" headline must be withdrawn.
- **C3: The Exp 2 compact-MC anchor (−0.366%) is mismatched in pass, amplitude and grid** (§3.1). All "% of compact MC" values, "recovers MC as $L_n \to 0$", and "−15.8% filtered" must be recomputed against a like-for-like compact first pass at $\delta\rho = -0.05$ on a converged grid.

### Moderate
- **M1: Contour sensitivity.**
  - A $C^1$ cosine contour flips bell $L_n = 0.80$ 1T to cold-unstable (+0.00186).
  - It changes bell 2T at $\bar M_c = 0.12$ by 15× (−0.0085 → −0.0006) and the annular 1T peak by 11%.
  - "46×" (a ratio to a near-zero number) and the 0.060 Hopf value are contour-specific. Report a contour family, or state the ratios as model-specific.
- **M2: Exp 1D "per-mode numerical damping" table mixes in a fit-window and beating artifact.**
  - In the annular 1T row, the 32-unit window equals $T_{beat}$. On $[60, 100]$ both `96×200` and `144×300` match exact to ≤ 1.4%.
  - Use $[60, 100]$ or a two-mode least-squares fit, and relabel the `+0.0086`.
- **M3: Exp 2 window truncation.**
  - At $L_n = 1.10$ the "2nd-pass peak" is the first pass (true 2nd pass −0.3143% at t = 18.54, outside $t_{end} = 19$).
  - First-pass values are grid-sensitive by 4–7%, which is comparable to the claimed effect.
  - $L_n \to 0$ is never run in the PDE ($L_n = 0.05$ is unresolvable at this resolution).
- **M4: Missed cold instability and Hopf scope at $\bar M_c = 0.10$.**
  - The bell m = 4 (+0.00369) is cold-unstable too, and the m = 3 mechanism narrative ($v' \propto 1/R$) is contradicted by the eigenvector ($|v'|$ ratio 0.26).
  - Both whistles disappear for Re ≲ 2×10⁵.
  - Mark the $\bar M_c = 0.10$ bell Hopf "0.000" inviscid-only, and extend the Hopf scan to m ≤ 5.
- **M5: Physical scope.**
  - The thin-shell model ($m/R$, no meridional metric with $R'$ up to about 1.2, no radial modes) is not a 3D bell nozzle.
  - The annular "1T" is below chamber cutoff. The bell is *more* damped than annular and MC for $L_n \le 0.42$ at $\bar M_c = 0.12$.
  - Law #15 should be scoped to "2D thin-shell model, $L_n \gtrsim 0.45$".
- **M6: Short-$L_n$ eigenvalues are under-resolved** (8–20% at $L_n \le 0.15$, including the `compact_limit_annular` block). Multi-domain Chebyshev confirms smooth convergence to MC (−0.02846 at $L_n = 0.02$). Replace the block with MD values.

### Minor
- **m1:** `lin_bell20_bare_1T` +0.0244 vs. exact +0.0144 is a convergent grid artifact (+0.0207 on `120×300`, +0.0185 on `160×400`). Document it. The converged exact value is +0.01473 (MD), and the repo's N = 84 value is 2.4% low.
- **m2:** Quote `NL_bell50_hyb_1T_80` at t = 300 (5.25%), not the t = 160 transient 2.65%.
- **m3:** The m = 1 window excludes a less-damped m = 1 mode at ω ≈ 1.85 (annular reacting −0.0132 vs. reported −0.0237).
- **m4:** "523×" should be 521×. $R_t$ should be 0.4497 (not 0.453). The Exp 1D note attributes "7.5×" to MC −0.0993. "rhs ~5e-13" is 2.6e-12 at nx = 100.
- **m5:** The Exp 2 runs are labelled "bell" but are geometry-independent for m = 0 (bell = annular to 4 digits). Fix the stale docstring $L_n$ list, the back-ticks in the figure legend, and the `mc_limiter` alias name.

### Confirmed claims
- Conservation and well-balancing structure.
- Chebyshev eigenvalues (all cases, 5 digits), spectral convergence, and $L_d$ independence.
- Annular → MC compact limit (with MD).
- Bell 1T cutoff collapse values for the stated contour.
- The full Hopf map including 3.79×, 2.48×, 1.90× and −63%.
- Exact reacting bare values.
- The inviscid whistle eigenvalue.
- Bell hybrid 1T → 2T lock-in (5.25% on `80×200`, 5.54% on `120×300`).
- Bell bare double-mode failure (not contradicted).

---

## 8. Remediation steps
1. **Exp 4.**
   - Re-run all nonlinear annular cases on ≥ `120×300` (preferably `160×400`) to t ≥ 300.
   - Add `NL_ann50_hyb_2T` and `NL_ann50_bare_2T` bombs.
   - Restate Law #15 item 2: the annular slot linearly improves 1T, but on resolved grids the bare slot sustains a ~4% 2T-dominated cycle, and the hybrid regrows from a 1T bomb into a 5.07% 2T cycle (bell: 5.54%).
2. **Exp 2.**
   - Run the compact-MC reference with **012's own solver** (for example $L_n \to 0$ via MD-resolved, or the 009 solver with $\delta\rho = -0.05$, no swirl, `nx ≥ 192`).
   - Compare first pass with first pass, and extend $t_{end}$ to ≥ 26 so that the $L_n = 1.10$ 2nd pass is captured.
   - Report 1× and 2× grids.
3. **Exp 1D.** Fit on $[60, 100]$ (or use two-mode Prony fits). Remove the "+0.0086 numerical damping" annular entry.
4. **Exp 3.**
   - Add a contour-family sensitivity row ($C^1$ cosine, $C^2$ quartic, at least one other).
   - Scan m ≤ 5 in Hopf.
   - Report bell $\bar M_c = 0.10$, m = 4 (+0.00369).
   - Mark whistles as inviscid (both die at Re ≈ 10⁵).
   - Correct the mechanism text with the eigenvector.
5. **Exp 1 compact limit.** Replace single-domain short-$L_n$ values with a multi-domain solver (characteristic interface patching, as in `/tmp/peer_review_012/md_eig.py`).
6. **Scope.** Add "2D thin-shell, no meridional metric, $m/R$ not $s_{mn}/R$" to Law #15. Restrict the bell-collapse statement to $L_n \gtrsim 0.45$ at $\bar M_c = 0.12$.
7. **Fix the minor numeric and labelling items (m1–m5).**

---

## 9. Reproduction (all under `/tmp/peer_review_012/`)
- **Environment.** Analysis scripts that need sympy run in an ephemeral environment: `uvr.sh` runs `uv run --no-project --with numpy,scipy,sympy,matplotlib python -B`. The PDE batch in Step 4 runs in the repo's environment: `uv run --project /Users/marcelolx/nearchx/math python -B`.
- **No bytecode.** The repo module is imported read-only via `loadmod.py` with `sys.dont_write_bytecode = True`.

| Script | Purpose |
|---|---|
| `indep_eig.py` | independent sympy conservative-flux Chebyshev solver |
| `md_eig.py` | multi-domain version, with flame |
| `t1.py` | repo vs. independent eigenvalues |
| `t2_cold.py` | N, m, $L_d$ and contour sweeps |
| `t3_compact.py` | mapped single-domain attempt (did not converge; superseded by MD) |
| `t5_md.py`, `t9_md2.py` | MD compact limit, short-$L_n$ and reacting bell20 |
| `t6_hopf.py` | Hopf maps for m ≤ 5, reacting spectra |
| `t8_cut.py` | cutoff profiles |
| `t10_whistle.py` | whistle convergence, eigenvector, viscosity |
| `t4_e009.py` | Entry 009 entropy protocol at nx = 48, 96, 192 |
| `t11_entropy.py`, `t12_ent_an.py` | Exp 2 re-runs |
| `run_jobs.py`, `pde_patch.py`, `analyze_runs.py` | long-horizon and fine-grid PDE runs; histories in `runs/*.npz` |
| `t7_audit.py` | prose-JSON audit |
