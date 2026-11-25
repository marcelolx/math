# Independent Scientific & Numerical Peer Review #6 — Rocket Combustion Entry `009` & `SYNTHESIS.md` Law #12 / Dead End #11

- **Date:** 2026-11-25
- **Reviewer:** Independent Read-Only Audit Worker (`task_3419ef152468`)
- **Targets Audited:**
  - `rocket-combustion/entries/009-2d-euler-throughflow.md`
  - `rocket-combustion/sim/exp009_2d_euler_throughflow.py`
  - `rocket-combustion/results/exp009_metrics.json`
  - `rocket-combustion/SYNTHESIS.md` (Law #12, Dead End #11, Scope/Open Problems)
- **Repository Integrity:** Verified completely clean (`git status -s` empty; all audit scripts and outputs isolated under `/tmp/peer_review_009/` using `uv run python -B`).

## Executive Verdict: **REVISE BEFORE MERGE / PUBLICATION**

**Summary:** Entry `009`'s core numerical engine—2D conservative MUSCL-MC + HLLC Euler solver with exact choked zero-swirl inlet and Marble–Candel compact choked nozzle BCs, plus its spectrally converged Chebyshev 2D linearized eigen-solver—is **mathematically sound and genuinely solves Open Problem #1** (first-principles entropy/swirl flushing and bounded stationary limit cycles with zero ad-hoc $S_{\text{ent}}$). The primary cold-$1T$ Strouhal damping peak near $\bar{M} = 0.16\text{--}0.18$, the nonlinear cubic-flame triggering mechanism of odd $M = 5$ Erlang-4 staggering, and the all-IC robustness of $\bar{M} = 0.18$ at the specific Exp 4 point ($n_p = 0.22, n_v = 0.04$) all hold up under independent adversarial testing.

**However, our audit uncovered 3 Critical and 4 Moderate scientific/numerical defects** that require revision in `009-2d-euler-throughflow.md`, `exp009_2d_euler_throughflow.py`, and `SYNTHESIS.md` Law #12 / Dead End #11 before publication:
1. **C1 (Multi-Mode Blind Spot):** The claimed "$4.8\times$ Hopf peak ($n_{pH} = 0.638$ at $\bar{M} = 0.18$)" checks $1T$ in isolation; globally, $m = 0$ chugging/entropy-nozzle ($\omega = 0.207$) and $2T$ ($\omega = 1.954$) go unstable at **$n_{pH,\text{global}} = 0.4260$ ($3.18\times$, not $4.8\times$)**. Worse, at the actual half-integer cold-$1T$ peak ($\bar{M} = 0.16\text{--}0.17$), $2T$ falls into an integer Strouhal trough ($\mathrm{St}_{2T} \approx 3$) and is **linearly unstable** at the Exp 4 chiral point ($n_p = 0.22, n_v = 0.04$), saturating into a **$6.53\%$ $2T$ rotating shock** on `48×144`/`64×192` while falsely masked on `32×96`!
2. **C2 (Exp 3 Grid-Refinement Sign Reversal & Confounded $\bar{M}$/$n_p$ Design):** The published `low_gain` point ($\bar{M} = 0.06, n_p = 0.105$) collapses to standing **only** on Entry `009`'s coarse `28×96` grid (where numerical vorticity damping cuts $P$ by $-30\%$); on `48×144`, `low_gain_spin` **reverses direction and spins up** ($S_{\text{env}}: +0.854 \to +0.896$, bistable!). Moreover, controlled-$\bar{M}$ tests show $\bar{M}$ itself (via $2T$ damping / shock cascade) confounds the claimed low-gain vs. high-gain Landau flip.
3. **C3 (Exp 1D Marble–Candel Echo Misidentification):** The reported "$+0.47\%$ exit echo at $t = 7.44$" is **not** the Marble–Candel entropy echo (100% of the entropy spot is still inside the chamber at $t = 7.5$!); it is the **3rd acoustic bounce** ($T_{\text{rt}} = 3.044$) of a spurious $+3.06\%$ $t = 0$ inlet-BC mismatch pulse. The true Marble–Candel echo from a hot entropy spot has **opposite sign (negative rarefaction, $-0.39\%$ to $-0.44\%$)** and arrives at $t \approx 10.4\text{--}13.1$.

---

## Severity-Ranked Findings

### CRITICAL FINDINGS

#### C1. Multi-Mode Blind Spot at Strouhal Peaks: Global Hopf Threshold Is $0.426$ ($3.18\times$), Not $0.638$ ($4.8\times$), and $\bar{M} = 0.16\text{--}0.17$ Is Unstable to Co-Rotating $2T$ (Masked on `32×96`)
- **Locations:**
  - `rocket-combustion/entries/009-2d-euler-throughflow.md:33, 65-72, 113`
  - `rocket-combustion/sim/exp009_2d_euler_throughflow.py:586-594`
  - `rocket-combustion/SYNTHESIS.md:24` (Law #12)
- **Root Cause:** `exp009_2d_euler_throughflow.py:590-591` computes $n_{pH}(\bar{M})$ exclusively for $m = 1$ inside `omega_window=(0.65, 1.35)` ($1T$ only). Physically, half-integer Strouhal tuning for $1T$ ($\mathrm{St}_{1T} \approx k + \tfrac{1}{2}$) **structurally forces $2T$ ($\omega_{2T} \approx 2\omega_{1T}$) into an integer Strouhal trough** ($\mathrm{St}_{2T} \approx 2k + 1$), while $m = 0$ axial/chugging modes have $u'_\theta \equiv 0$ and receive zero vorticity-shedding damping.
- **Independent All-Mode Chebyshev Verification ($N = 56\text{--}64$, $m \in \{0, 1, 2, 3\}$, all $\omega \in [0, 10]$, $n_v = 0$):**

| $\bar{M}$ | $n_{pH,1T}$ ($\omega$) | $n_{pH,m=0}$ ($\omega$) | $n_{pH,2T}$ ($\omega$) | $n_{pH,3T}$ ($\omega$) | **True $n_{pH,\text{global}}$** | **Limiting Mode** |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $0.06$ | $0.0650$ ($0.996$) | $0.2138$ ($2.016$) | $0.1496$ ($1.973$) | $0.5837$ ($2.792$) | $0.0650$ | $1T$ |
| $0.08$ | $0.0767$ ($0.990$) | $0.2527$ ($0.130$) | $0.1996$ ($1.963$) | $0.7232$ ($2.726$) | $0.0767$ | $1T$ |
| $0.10$ | $0.3031$ ($0.914$) | **$0.2839$** ($1.968$) | $0.5170$ ($1.838$) | $0.9124$ ($2.648$) | **$0.2839$** | **$1L$ ($m=0$)** |
| $0.12$ | $0.1338$ ($0.971$) | $0.3212$ ($0.165$) | $0.1822$ ($1.954$) | $0.9104$ ($2.619$) | $0.1338$ | $1T$ |
| **$0.16$** | $0.3807$ ($1.016$) | $0.3911$ ($0.194$) | **$0.2354$** ($1.917$) | $0.7777$ ($2.652$) | **$0.2354$** | **$2T$ (St trough!)** |
| **$0.17$** | $0.5007$ ($1.003$) | $0.4086$ ($0.201$) | **$0.2706$** ($1.949$) | $0.6765$ ($2.726$) | **$0.2706$** | **$2T$ (St trough!)** |
| **$0.18$** | $0.6382$ ($0.961$) | **$0.4260$** ($0.207$) | **$0.4300$** ($1.954$) | $1.3382$ ($2.431$) | **$0.4260$** | **$m=0$ chug / $2T$** |
| $0.19$ | $0.6110$ ($0.902$) | **$0.4130$** ($1.839$) | $0.8418$ ($1.734$) | $1.1910$ ($2.445$) | **$0.4130$** | **$1L$ ($m=0$)** |
| $0.20$ | $0.5286$ ($0.891$) | **$0.3813$** ($1.864$) | $0.6990$ ($1.729$) | $1.0653$ ($2.486$) | **$0.3813$** | **$1L$ ($m=0$)** |
| $0.25$ | $0.4000$ ($0.919$) | $0.5447$ ($0.245$) | $0.4284$ ($1.834$) | $1.0384$ ($2.603$) | $0.4000$ | $1T$ |

- **Direct Nonlinear 2D PDE Falsification:**
  1. **At $\bar{M} = 0.18, n_v = 0$, well below $n_{pH,1T} = 0.6382$:**
     - `n_p = 0.50` (`32×96`, `cold_1L`, $A_0 = 0.005, t = 300$): grows from $0.39\% \to 2.57\% \to \mathbf{8.06\%}$ (`p_exit_amp = 7.30%`, $E_1 = E_2 = 0$), a violent $m = 0$ chugging/longitudinal limit cycle!
     - `n_p = 0.58` (`32×96`, `cold_1L`, $t = 240$): saturates at **$9.45\%$** (`p_exit_amp = 8.56%`).
     - `n_p = 0.58` (`48×144`, `spin_2T`, $A_0 = 0.010, t = 200$): saturates into an **$8.13\%$ $2T$ rotating limit cycle** ($E_2 = 1.17 \times 10^{-3}$).
     - At $n_p = 0.638$, exact Chebyshev ($N=64$) gives THREE strongly unstable non-$1T$ modes: $m=0$ ($\alpha = +0.06898, \omega = 0.1835$), $1L$ ($\alpha = +0.03846, \omega = 1.775$), and $2T$ ($\alpha = +0.02890, \omega = 1.899$).
  2. **At the Exp 4 chiral point ($n_p = 0.22, n_v = 0.04$) near the half-integer cold peak ($\bar{M} = 0.16\text{--}0.17$):**
     - Exact Chebyshev ($N = 52$) shows co-rotating $2T$ ($m = -2$) is **linearly unstable** across $\bar{M} \in [0.160, 0.170]$: $\alpha_{-2T} = \mathbf{+0.00941}$ ($\bar{M}=0.160$), $\mathbf{+0.00802}$ ($\bar{M}=0.165$), $\mathbf{+0.00285}$ ($\bar{M}=0.170$); stability is achieved only for $\bar{M} \ge 0.175$ ($\alpha_{-2T} = -0.00603$, and $-0.01869$ at $\bar{M} = 0.180$).
     - In 2D PDE at $\bar{M} = 0.16, n_p = 0.22, n_v = 0.04$:
       - `32×96` (`spin_2T`, $A_0 = 0.005$): `alpha_fit = -0.01528`, $P = 0.06\%$ — **falsely masked by `-4.4e-2` numerical excess damping on $2T$!**
       - `48×144` (`spin_2T`, $A_0 = 0.005$): `alpha_fit = +0.00246`, `l/m = 1.392` (linearly unstable!).
       - `48×144` (`spin_2T`, $A_0 = 0.040$): saturates into a **$6.53\%$ $2T$ rotating shock limit cycle** (`E_late = 1.06e-3`, `l/m = 1.001`)!
       - `64×192` (`spin_2T`, $A_0 = 0.005$): `alpha_fit = +0.01293`, `l/m = 5.450`, growing to $2.16\%$ by $t = 150$.
- **Required Fix:** Reframe Law #12 and Entry `009` Exp 2 to distinguish $1T$-only Hopf gain ($4.8\times$) from the **all-mode global Hopf gain ($3.18\times$, $0.134 \to 0.426$ at $\bar{M} = 0.18$)**, plot $n_{pH,\text{global}}(\bar{M})$ alongside $n_{pH,1T}(\bar{M})$, and explicitly document the **harmonic Strouhal anti-resonance trap** ($\mathrm{St}_{1T} \approx k + \tfrac{1}{2} \implies \mathrm{St}_{2T} \approx 2k + 1$, destabilizing $2T$ at $\bar{M} = 0.16\text{--}0.17$).

---

#### C2. Exp 3 Grid-Refinement Sign Reversal on `low_gain` ($\bar{M} = 0.06, n_p = 0.105$) & Confounded $\bar{M}$ vs. $n_p$ Landau Demonstration
- **Locations:**
  - `rocket-combustion/entries/009-2d-euler-throughflow.md:76-86, 114`
  - `rocket-combustion/sim/exp009_2d_euler_throughflow.py:610-617`
  - `rocket-combustion/SYNTHESIS.md:24` (Law #12)
- **Root Cause:** Entry `009` ran `low_gain` ($\bar{M} = 0.06, n_p = 0.105$) **only** on `28×96` (where $\lambda_{\text{vort}} = 2\pi(0.06)/1 \approx 0.377 \implies$ only 7 cells per vorticity wavelength!) and changed **both** $\bar{M}$ ($0.06 \to 0.12$) and $n_p$ ($0.105 \to 0.220$) simultaneously between `low_gain` and `high_gain`.
- **Independent Long-Horizon, Grid-Refined & Controlled-$\bar{M}$ PDE Runs ($n_v = 0$, $A_0 = 0.035$):**

| Run Label | $\bar{M}, n_p$ | Grid, $t_{\text{end}}$ | Init | $S_{\text{env}}$ (Q1, Q2, Q3, Q4) | $P_{\text{peak}}/\bar{p}$ | $E_{2T}/E_{1T}$ | $\bar{\rho}_{\text{late}}$ | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `low_gain_spin_t720` | $0.06, 0.105$ | $28\times 96, 720$ | `spin_pert` | `+0.812, +0.570, +0.323, +0.148` | $3.68\%$ | $8.8\%$ | $1.0004$ | Collapses on coarse grid |
| **`low_gain_stand_48`** | **$0.06, 0.105$** | **$48\times 144, 360$** | `stand_seed` | `+0.082, +0.070, +0.058, +0.048` | **$5.17\%$** | $7.7\%$ | $1.0007$ | Standing stable ($+43\%$ $P$!) |
| **`low_gain_spin_48`** | **$0.06, 0.105$** | **$48\times 144, 360$** | `spin_pert` | **`+0.854, +0.877, +0.887, +0.896`** | **$4.71\%$** | $8.8\%$ | $1.0010$ | **REVERSES! Spins UP (Bistable!)** |
| `M006_lo080_spin_48` | $0.06, 0.080$ | $48\times 144, 260$ | `spin_pert` | `+0.807, +0.799, +0.765, +0.718` | $3.34\%$ | $10.0\%$ | $1.0002$ | True `48×144` standing attractor |
| `M006_hi_np180_stand` | $0.06, 0.180$ | $32\times 96, 400$ | `stand_seed` | `+0.114, +0.083, +0.048, +0.044` | **$8.06\%$** | **$13.0\%$** | $1.0015$ | **Stays standing even at $8.06\%$!** |
| `M006_hi_np180_spin` | $0.06, 0.180$ | $32\times 96, 400$ | `spin_pert` | `+0.906, +0.935, +0.955, +0.969` | $6.60\%$ | $8.4\%$ | $1.0002$ | Bistable at $\bar{M}=0.06$ |
| `M012_lo150_spin_48` | $0.12, 0.150$ | $48\times 144, 260$ | `spin_pert` | `+0.833, +0.937, +0.962, +0.976` | $5.69\%$ | $5.9\%$ | $0.9993$ | Spins up even $12\%$ above Hopf |
| `M012_lo170_stand` | $0.12, 0.170$ | $32\times 96, 480$ | `stand_seed` | `+0.164, +0.444, +0.773, +0.919` | $5.62\%$ | $4.8\%$ | $0.9997$ | Spins up from standing |
| `high_gain_stand_t600` | $0.12, 0.220$ | $28\times 96, 600$ | `stand_seed` | `+0.328, +0.768, +0.938, +0.983` | $6.75\%$ | $4.9\%$ | $0.9995$ | Reaches $+0.983$ on `28×96` |
| `high_gain_stand_48` | $0.12, 0.220$ | $48\times 144, 320$ | `stand_seed` | `+0.149, +0.182, +0.220, +0.275` | **$8.71\%$** | **$9.6\%$** | $0.9997$ | Spins up **much slower** on `48×144` |
| `high_gain_spin_48` | $0.12, 0.220$ | $48\times 144, 360$ | `spin_pert` | `+0.918, +0.961, +0.975, +0.983` | $8.00\%$ | $6.6\%$ | $0.9986$ | Converged stationary spin (`l/m=1.000`) |

- **Scientific Diagnosis:**
  1. Entry `009`'s published `low_gain` point ($\bar{M}=0.06, n_p=0.105$) is **not grid-converged**: severe `28×96` vorticity numerical damping reduced its amplitude by $-30\%$ ($5.17\% \to 3.62\%$), artificially pushing it into the standing-attractor regime; on `48×144` it is **bistable** (`low_gain_spin` spins UP to $+0.896$).
  2. Holding $\bar{M}$ fixed shows that **$\bar{M}$ itself strongly controls mode selection** (through $2T$ damping and shock-harmonic cascade $E_{2T}/E_{1T} = 10\text{--}13\%$ at $\bar{M}=0.06$ vs. $5\text{--}6\%$ at $\bar{M}=0.12$): at $\bar{M}=0.06$, standing remains locally stable even at $P = 8.06\%$ ($n_p=0.180$), whereas at $\bar{M}=0.12$, even $n_p = 0.150$ (`48×144`) spins up to $+0.976$.
- **Required Fix:** Re-run Exp 3 on a converged grid (`48×144`) at **fixed $\bar{M}$** (or explicitly separate the $\bar{M}=0.06$ standing/bistable transition at $n_p = 0.080 \to 0.105$ from the $\bar{M}=0.12$ spinning attractor), and stop claiming $(\bar{M}=0.06, n_p=0.105)$ is a global standing attractor.

---

#### C3. Exp 1D Misidentifies a Spurious $t = 0$ Inlet-BC Mismatch Bounce (`+0.47%` at $t = 7.44$) as the Marble–Candel Indirect Entropy-Noise Echo
- **Locations:**
  - `rocket-combustion/entries/009-2d-euler-throughflow.md:61`
  - `rocket-combustion/sim/exp009_2d_euler_throughflow.py:283-286, 573-580`
  - `rocket-combustion/results/exp009_metrics.json:102-103`
- **Analytical & Numerical Proof:**
  1. **Opposite sign:** From the linearized Marble–Candel BC $u'_x(L_a) = \frac{1}{2}\bar{M}(\gamma P - \rho')$, an outgoing **hot** entropy spot ($s' > 0, \rho' = -s'/\gamma < 0$) reflects a **negative (rarefaction)** pressure pulse $P_-/\bar{p} \approx -\frac{\bar{M}}{2} s' = -0.30\%$, NOT a positive $+0.47\%$ spike.
  2. **Spurious origin of $t = 7.44$:** Because `s_init` is centered at $x_0 = 0.25$ with $\sigma = 0.12$, its tail leaks to cell 0 ($s_0 = 0.00841, c_0 = 1.00337$), forcing the choked isentropic inlet BC at $t = 0$ to jump to $p_{\text{in}}/\bar{p} = +3.055\%$. This launches a spurious positive compression pulse bouncing at the acoustic round-trip period $T_{\text{rt}} = \frac{1.5}{1.12} + \frac{1.5}{0.88} = 3.044$:
     - 1st arrival: $t = 1.42$, $+1.20\%$ (`nx=32`) / $t = 1.38$, $+0.95\%$ (`nx=64`)
     - 2nd bounce: $t = 4.42$, $+0.70\%$ (`nx=32`) / $t = 4.40$, $+0.55\%$ (`nx=64`)
     - **3rd bounce: $t = 7.43$, $+0.47\%$ (`nx=32`)** / $t = 7.41$, $+0.36\%$ (`nx=64`) — at $t = 7.51$, `ent_mean = +9.86e-3`, proving **100% of the entropy spot is still inside the chamber**!
  3. **True Marble–Candel echo:** Peak exit entropy arrives at $t = 10.45$ (exact convection $(1.5-0.25)/0.12 = 10.42$), driving a genuine **negative** exit rarefaction of **`-0.39%` at $t = 12.85$ (`nx=32`)** and **`-0.44%` at $t = 13.10$ (`nx=64`)**. Centering the pulse cleanly away from $x = 0$ ($x_0 = 0.50, u_\theta = 0$) eliminates the spurious positive bounces and isolates the pure `-0.440%` Marble–Candel echo (peak exit entropy at $t = 8.37$ vs. exact $8.33$).
- **Required Fix:** Center the Exp 1D entropy pulse away from the inlet (e.g., $x_0 = 0.50, \sigma = 0.08$, or smooth compact support vanishing at $x = 0$) and report the true negative Marble–Candel rarefaction echo at $t \approx \tau_{\text{conv}}$.

---

### MODERATE FINDINGS

#### M1. Arithmetic Error in Exp 4B & `SYNTHESIS.md` Law #12 / Dead End #11: $|H_1(i)| = 0.963$ (Not `0.910`) and Cubic Asymmetry Is $6.92\times$ / $6.68\times$ (Not `$5.8\times$`)
- **Locations:** `rocket-combustion/entries/009-2d-euler-throughflow.md:100, 109`; `rocket-combustion/SYNTHESIS.md:24, 40`.
- **Verification:** For $H_{\text{Erl}}(i\omega) = (1 + i\omega\tau/4)^{-4}$:
  - At $\omega = 1.000$: $|H_1(i)| = (1 + (0.55/4)^2)^{-2} = \mathbf{0.96323}$ (`0.95258` with DC-blocker), $|H_2(i)| = (1 + (2.55/4)^2)^{-2} = \mathbf{0.50557}$ (`0.5000` with DC-blocker). Ratio $= 1.9053$, cube $= \mathbf{6.916\times}$ (independent of DC-blocker).
  - At actual cold $\omega_{1T} = 0.98928$: $|H_1| = 0.96400$ (`0.95310` with DC-blocker), $|H_2| = 0.51186$ (`0.50607` with DC-blocker). Ratio $= 1.8833$, cube $= \mathbf{6.680\times}$.
  - The claimed `|H_1(i)| = 0.910` and `5.8×` are arithmetic errors (the cubic-flame mechanism itself is **confirmed**—see Confirmed #6).

#### M2. Exp 3 Table Initial Spin Labels (`+0.17`, `+0.98`) Are Old Flux Index $S_{\text{flux}}$, Not Envelope Index $S_{\text{env}}$ (`+0.085`, `+0.818`)
- **Locations:** `rocket-combustion/entries/009-2d-euler-throughflow.md:30, 80-84`; `rocket-combustion/sim/exp009_2d_euler_throughflow.py:295-307, 476-485`.
- **Verification:** For `stand_seed` ($\epsilon = 0.12$) and `spin_pert` ($\epsilon_c = 0.10$), $S_{\text{flux}} = \frac{2Pu}{P^2+u^2}$ gives $+0.1685$ (`+0.17`) and $+0.9802$ (`+0.98`), whereas $S_{\text{env}} = \frac{|a_+|-|a_-|}{|a_+|+|a_-|} = \frac{u}{P}$ gives **`+0.0849`** and **`+0.8182`**. Quoting initial $S_{\text{flux}}$ next to final $S_{\text{env}}$ inflates the apparent drop in `low_gain_spin` by $+66\%$ (`0.98 -> 0.57` vs. actual `0.818 -> 0.572`).

#### M3. Strouhal Peaks Are Acoustic-Vorticity Mode Branch Crossings (Coarse $\Delta\bar{M}=0.01$ Misses Cusp), and Secondary Peak $\bar{M}=0.10$ Has `-3.0e-2` PDE Error Even on `48×144`
- **Locations:** `rocket-combustion/entries/009-2d-euler-throughflow.md:33, 65-73`; `rocket-combustion/sim/exp009_2d_euler_throughflow.py:587-593`; `rocket-combustion/results/exp009_metrics.json:177-196`.
- **Verification:** Dense Chebyshev ($N = 64$) scanning shows the primary cold-$1T$ peak is a **branch-crossing cusp** between $\bar{M} = 0.164$ ($\alpha = -0.1161, \omega = 1.138, \mathrm{St} = 1.657$) and $\bar{M} = 0.166$ ($\alpha = -0.1225, \omega = 0.810, \mathrm{St} = 1.165$); at $\bar{M} = 0.18$, $\mathrm{St} = 1.13$ (not half-integer $1.5$) because the least-damped mode switched to the lower-frequency branch ($\omega = 0.852$). Furthermore, Entry `009` skipped PDE verification of $\bar{M} = 0.10$; our `48×144` PDE check gives $\alpha_{\text{PDE}} = -0.09064$ vs. exact $-0.06028$ (`diff = -3.04e-2`, latching onto the neighboring branch at $\alpha = -0.0909, \omega = 1.117$).

#### M4. `e_wave` Diagnostic Blindness to $m = 0$ Bulk/Chugging Modes & `exact_2d_throughflow_eig` Single-Sign Chirality Hazard
- **Locations:** `rocket-combustion/sim/exp009_2d_euler_throughflow.py:160-163, 412-415`.
- **Verification:**
  1. `Po = Pf - float(np.mean(Pf))` removes spatially uniform pressure perturbations, masking the $m = 0$ low-frequency chugging mode ($\omega \approx 0.207$) that limits global stability at $\bar{M} = 0.18$.
  2. `exact_2d_throughflow_eig` restricts to `v.imag > 0` for a single sign of $m$; for $n_v \neq 0$, $m > 0$ with $\omega > 0$ selects **only** the counter-rotating ($-\theta$) wave and misses the far more unstable co-rotating ($+\theta$, $m < 0$) wave (e.g., at $\bar{M}=0.12, n_p=0.22, n_v=0.04$: $\alpha_{+1T} = +0.00334$ vs. $\alpha_{-1T} = +0.03506$, a $10.5\times$ error!). While Entry `009` only called it with $n_v = 0$, the function signature accepts `n_v` without checking $\pm m$ or $\pm\omega$.

---

### MINOR FINDINGS

- **m1.** `exp009_2d_euler_throughflow.py:22` docstring says `N_cheb = 36`, while line `108` defaults to `N = 34`.
- **m2.** `exp009_2d_euler_throughflow.py:42` docstring says `t=300` for Exp 4, while lines `629, 637-640` run `t=280`.
- **m3.** `SYNTHESIS.md:3` says `000`–`008` were audited across "**four** independent cross-model peer reviews", contradicting `SYNTHESIS.md:38-39` and `009-2d-euler-throughflow.md:9, 115` which cite **Peer Review #5** (five reviews for `000`–`008`).
- **m4.** `exp009_metrics.json:403` has `"harmonic_2T_over_1T_pct": 5.16e9` for `strouhal_M018_shift` (because $1T$ decays $4.6\times$ faster than $2T$, which also causes the slope bend at $t \approx 30$ in Figure 2), and `009-2d-euler-throughflow.md:96` quotes `S_env = +0.070` on a $10^{-14}$ noise floor without noting it is undefined at zero signal. Also footnote `009:86` cites `-1.1e-2` for `28×96` excess damping (interpolated, not in `exp009_metrics.json`).

---

### CONFIRMED (Verified Independently)

1. **2D Governing Equations, MUSCL-MC + HLLC Flux & Exact Linearized BCs:** Line-by-line mathematical audit confirmed exact conservative fluxes, $\theta$-direction normal/tangential momentum swap (`Ft`), Toro/Batten HLLC energy star-state, exact Newton choked isentropic inlet BC, exact closed-form Marble–Candel outlet BC, and exact consistency between nonlinear BCs and linearized Chebyshev BCs.
2. **Chebyshev Spectral Convergence ($N = 24\text{--}80$) & BC-Row Independence:** Independently verified at $N = 24, 34, 48, 64, 80$ and with explicit characteristic-compatibility boundary rows (`char`) across $\bar{M} \in [0.06, 0.30]$ for `1L, 1T, 2T, 3T`: all agree to **6+ significant digits**, confirming `N = 34` is completely converged even at $\bar{M} = 0.06$.
3. **Genuine Primary Cold-$1T$ Strouhal Damping Peak ($\bar{M} = 0.16\text{--}0.18$):** Confirmed on `48×144` nonlinear 2D PDE:
   - $\bar{M} = 0.16$: exact `-0.10212` vs. PDE `-0.10132` (`diff = +8.0e-4`)
   - $\bar{M} = 0.17$: exact `-0.11575` vs. PDE `-0.11214` (`diff = +3.6e-3`)
   - $\bar{M} = 0.18$: exact `-0.09933` vs. PDE `-0.09815` (`diff = +1.2e-3`)
   - $\bar{M} = 0.25$: exact `-0.06006` vs. PDE `-0.06104` (`diff = -9.9e-4`)
4. **First-Principles Entropy/Swirl Flushing & Stationary Limit Cycles (Zero $S_{\text{ent}}$):** Confirmed! $\bar{\rho} = 0.9986\text{--}1.0010$ and bounded $\bar{u}_\theta \sim 10^{-5}\text{--}2\times 10^{-3}$ hold even out to $t = 600\text{--}720$ (`28×96`) and $t = 360$ (`48×144`, `high_gain_spin`: `l/m = 1.000`, $P = 8.00\%$).
5. **All-Mode Stability & Multi-IC Robustness of $\bar{M} = 0.18$ at the Exp 4 Point ($n_p = 0.22, n_v = 0.04$):** Confirmed! All modes $m \in [-3, +3]$ are linearly stable ($\alpha_{\max} = -0.01869$ on co-rotating $2T$, $4.6\times$ weaker than $1T$'s $-0.08654$), and all 8 independent nonlinear PDE stress tests (`stand_seed`, `spin_2T`, `spin_3T`, `cold_1L`, `A=0.090` spinning/standing bombs on `32×96`, plus `spin_1T` & `spin_2T` on `48×144`) decay cleanly (`l/m = 0.0000–0.0314`).
6. **Exp 4B Nonlinear Cubic-Flame Mechanism for $M = 5$ Triggering & JSON-to-Prose Consistency:** Running `stagger_M5_odd` with `linear_flame=True` at $A_0 = 0.030$ and $0.055$ yields clean decay (`l/m = 0.269, 0.267`, $P = 0.43\%, 0.74\%$), proving the triggering mechanism is indeed nonlinear flame gain-asymmetry (with corrected factor $6.7\text{--}6.9\times$). All quoted JSON-derived ratios (`3.5x`, `4.8x` for 1T-only, `4.0x`, `10^6x`, `10^4x`, `7.7x`, `10^11x`, `6.9%`) match `exp009_metrics.json`.

---

## Independent Audit Scripts (in `/tmp/peer_review_009/`)
- `audit_part1_spectral_and_exp1d.py` — Chebyshev $N=24\text{--}80$ convergence, BC-style check, all-mode global Hopf curve, Exp 1D echo anatomy, Erlang-4 $|H_1|, |H_2|$.
- `audit_part2_pde_checks.py` — `48×144` cold-$1T$ Strouhal curve, `linear_flame=True` $M=5$ test, 8-IC/grid robustness audit of `strouhal_M018_shift`.
- `audit_part3_exp3_landau.py` & `audit_part8_final_checks.py` — Long-horizon ($t=600\text{--}720$), `48×144` grid-refinement, and controlled-$\bar{M}$ Landau separation tests.
- `audit_part4_details.py` — Exp 1D $t=0$ inlet-BC pulse diagnosis, all-mode chiral spectrum ($n_p=0.22, n_v=0.04$) across $\bar{M} \in [0.10, 0.25]$, dense branch-crossing scan.
- `audit_part5_prose_json.py` — Automated JSON vs. prose & initial $S_{\text{env}}$ vs. $S_{\text{flux}}$ verification.
- `audit_part6_falsification.py` & `audit_part7_m018_high_np.py` — Direct nonlinear 2D PDE falsification at $\bar{M}=0.16$ ($6.53\%$ $2T$ rotating shock) and $\bar{M}=0.18, n_p=0.50\text{--}0.58$ ($8.05\text{--}9.45\%$ $m=0$ and $8.13\%$ $2T$ limit cycles).
