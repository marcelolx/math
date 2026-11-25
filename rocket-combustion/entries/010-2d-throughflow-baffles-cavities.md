# 010 — 2D Compressible Euler Throughflow with Protruding Baffles & Wide-Mouth Faceplate Cavity Rings

- **Date:** 2026-11-25
- **Status:** `verified` (pending independent Peer Review #7)
- **Verification Script:** `../sim/exp010_2d_throughflow_baffles_cavities.py`

## 1. Target

Directly resolve Open Problem #1 from [`../SYNTHESIS.md`](../SYNTHESIS.md): embed real physical 2D passive hardware—**protruding inviscid radial baffles** and **nonlinear wide-mouth faceplate Helmholtz cavity rings**—directly into Entry `009`'s 2D $(x, \theta)$ conservative compressible Euler annular combustor with steady axial throughflow ($\gamma = 1.25, L_a = 1.5, x_f = 0.25$, zero ad-hoc entropy-flush or drag closures). Test four concrete questions when engine cycle constraints prevent freely raising $\bar{M}$ from the $\bar{M} = 0.12$ Strouhal trough to $\bar{M} = 0.18$:

1. **Exact Hardware Conservation & Linear Validation:** Do exact split-face mirror HLLC reflecting baffles preserve steady throughflow to $\sim 10^{-16}$, conserve periodic mass/$x$-momentum/total energy to $\sim 10^{-16}$, satisfy exact azimuthal blade-pressure torque balance to $\sim 10^{-16}$, and do discrete/continuous faceplate cavities match an exact 1D-in-$x$ Chebyshev linearized throughflow eigen-solver across grids (`40×120`, `60×160`)?
2. **Inviscid Baffles in 2D Axial Throughflow ($\bar{M} = 0.12$):** In zero-throughflow bulk acoustics (`004`–`006`, Law #5 & Dead End #6), inviscid baffles were strictly conservative ($\Delta\alpha_{\text{cold}} = 0$) and longer baffles monotonically helped ($L_b \gtrsim 2\,x_f$). With steady axial throughflow $\bar{M} = 0.12$, does baffle-tip vorticity shedding break Dead End #6, and how does $1T$ stability vary across $L_b \in [0, 0.75]$ and even ($N_b = 4$) vs. prime ($N_b = 5$) blade counts?
3. **Multi-Mode Faceplate Cavities & Baffle–Cavity Frequency-Retuning Synergy ($\bar{M} = 0.12$):** Across all modes ($m = 0, \pm 1, \pm 2, \pm 3$), does Law #6's $80/20$ split survive into 2D throughflow (and does throughflow stabilize $3T$?), and what happens when $N_b = 5, L_b = 0.45$ baffles are paired with a compact $2\%$ faceplate cavity ring tuned to the *unbaffled* vs. *baffled* $1T$ frequency $\omega_{1T}(L_b)$?
4. **Disarming Entry `009`'s $2T$ Strouhal Trap ($\bar{M} = 0.16$) & the 2D Euler Orifice-Drag Cliff:** Can a micro $1.0\%$ $2T$-only faceplate trim cavity disarm the $6.58\%$ $2T$ rotating shock at $\bar{M} = 0.16$ on `48×160`, does Law #9's tight-neck vs. wide-mouth orifice-drag cliff survive full 2D rotating shocks, and how does it hold up across all five initial conditions (`spin_2T`, `spin_1T`, `stand_1T`, `spin_3T`, `cold_1L`)?

## 2. The Math & Assumptions

### Exact Split-Face Reflecting Baffles (Zero Penalty / Drag Parameters)
On the $N_b$ azimuthal cell interfaces $\theta_k = 2\pi k / N_b$ for axial cells $x_i < L_b$, inter-cell HLLC fluxes are replaced with exact mirror solid-wall HLLC fluxes on both sides of the zero-thickness blade:
- Left face (out of cell $j$): $F_{L,\text{wall}} = F_{\text{HLLC}}\!\bigl((\rho_L, u_{\theta,L}, u_{x,L}, p_L),\,(\rho_L, -u_{\theta,L}, u_{x,L}, p_L)\bigr)$, enforcing zero mass, axial-momentum, and total-energy flux across the blade and exact wall pressure $p^*_{L}$ in $\theta$-momentum.
- Right face (into cell $j+1$): $F_{R,\text{wall}} = F_{\text{HLLC}}\!\bigl((\rho_R, -u_{\theta,R}, u_{x,R}, p_R),\,(\rho_R, u_{\theta,R}, u_{x,R}, p_R)\bigr)$ with wall pressure $p^*_{R}$.
- All grids (`40×120`, `48×160`, `60×160`) are chosen so every baffle length $L_b \in \{0.15, 0.30, 0.45, 0.60, 0.75\}$ and blade/cavity count $N_b \in \{4, 5\}, N_c = 5$ sits on **exact integer cell faces** with zero sector-rounding asymmetry.

### Nonlinear Faceplate Helmholtz Cavity Ring at $x = 0$
For each cavity family $j$ (total volume fraction $\beta_j = V_j / V_{\text{ch}}$, resonance $\omega_{H,j}$, linear damping ratio $\xi_{H,j}$, total open-area ratio $\sigma_{o,j}$, $N_c = 5$ discrete resonators centered mid-sector at $\theta_k = (k + \tfrac{1}{2})2\pi/N_c$ with normalized port weights $\int_0^{2\pi} w_k\,d\theta = 1$, $V_k = \beta_j 2\pi L_a / N_c$, $A_{o,k} = \sigma_{o,j} 2\pi / N_c$, discharge coefficient $C_d = 0.65$):

$$\frac{d P_{c,k}}{dt} = \frac{q_k}{V_k}, \qquad \frac{d q_k}{dt} = \omega_{H,j}^2 V_k \Bigl(\bar{P}_k - P_{c,k} - R_{\text{nl},k}\,|q_k|\,q_k\Bigr) - 2\xi_{H,j}\omega_{H,j} q_k, \qquad R_{\text{nl},k} = \frac{1}{2 C_d^2 A_{o,k}^2}$$

coupled into the exact choked zero-swirl isentropic inlet boundary condition via $\rho_{\text{in}}(\theta) u_{x,\text{in}}(\theta) = \bar{M} - \sum_k w_k(\theta)\,q_k(t)$ (solved to $10^{-14}$ by Newton iteration). Linearizing around steady flow adds $(P_{c,j}, \tilde{q}_j)$ directly into the 1D-in-$x$ Chebyshev 2D linearized Euler eigen-solver (with azimuthal port form factor $|\hat{w}_m|^2$).

## 3. The Verification

Reproduce from the repo root (uses deterministic `ProcessPoolExecutor` batching):

```bash
uv run python rocket-combustion/sim/exp010_2d_throughflow_baffles_cavities.py
```

### Experiment 1 — Exact Preservation, Conservation, Blade-Torque Balance & Chebyshev Validation

- **Exp 1A (Steady preservation with $N_b = 5, L_b = 0.45$ baffles + $N_c = 5$ `80/20` cavities, `40×120`, $t = 20$):** `max |U - U_0| = 1.11e-16`.
- **Exp 1B (Periodic-in-$x$ inviscid conservation with $N_b = 5$ solid baffles, $A = 0.10, t = 10$):**
  - Mass: `1.89e-16` | $x$-momentum: `3.88e-16` | **Total energy: `0.00e+00`**
  - **Exact azimuthal blade-torque impulse balance** $|\Delta \iint \rho u_\theta - \int_0^t \sum_{\text{blades}}(p^*_R - p^*_L)\,dx\,dt|$: **`3.82e-17`** (on a physical raw azimuthal blade impulse of `1.44e-2`).
- **Exp 1C (Exact Chebyshev vs. 2D Euler MC PDE cold damping with $4\%$ `80/20` faceplate cavity ring at $\bar{M} = 0.12$):**

| Test Case | Exact Chebyshev $\alpha$ ($\omega$) | 2D Euler MC PDE $\alpha$ | Excess Numerical Damping |
| :--- | :---: | :---: | :---: |
| Continuous ring, $1T$ (`40×120`) | $-0.08425$ ($0.950$) | $-0.08967$ | `-5.4e-3` |
| **Continuous ring, $1T$ (`60×160`)** | **$-0.08425$ ($0.950$)** | **$-0.08540$** | **`-1.2e-3`** |
| Discrete $N_c = 5$ ring, $1T$ (`40×120`) | $-0.08423$ ($0.950$) | $-0.08770$ | `-3.5e-3` |
| Continuous ring, $2T$ (`60×160`) | $-0.05488$ ($1.985$) | $-0.07472$ | `-2.0e-2` |

### Experiment 2 — Non-Monotonic Baffle-Length Strouhal Resonance & Blade-Count Blind Spot ($\bar{M} = 0.12$)

Protruding inviscid baffles advance the zero-swirl vorticity-shedding origin from $x = 0$ to the baffle trailing edge $x = L_b$, shortening the convective Strouhal length to $L_{\text{eff}} = L_a - L_b$ while downshifting $\omega_{1T}(L_b)$ by up to **$-18\%$** (`0.997 → 0.818`). This reverses zero-throughflow Law #5 into a **non-monotonic Strouhal U-curve** verified on both `40×120` and `60×160`:

| Baffle Length $L_b$ ($N_b = 5$) | $L_{\text{eff}}$ | Cold $\omega_{1T}$ | $\mathrm{St}_{\text{eff}}$ | Cold $\alpha_{1T}$ (`40×120` / `60×160`) | Reacting $\alpha_{1T}$ (`40×120` / `60×160`) | NL Shock $P$ ($E_{\text{wave}}$, `l/m`) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $0.00$ (bare) | $1.50$ | $0.997$ | $1.98$ *(trough)* | $-0.0342$ / $-0.0317$ | $+0.0393$ / $+0.0477$ | $8.86\%$ ($3.72\times 10^{-3}$, `1.011`) |
| $0.15$ ($< x_f$) | $1.35$ | $0.997$ | $1.79$ | $-0.0398$ | **`+0.0411`** *(short-baffle amp!)* | **`9.54%`** ($2.40\times 10^{-3}$, `1.001`) |
| $0.30$ | $1.20$ | $0.981$ | $1.56$ *(peak)* | $-0.0722$ / $-0.0662$ | $+0.0103$ / $+0.0135$ | $6.96\%$ ($9.96\times 10^{-4}$, `1.000`) |
| **$0.45$ (peak)** | **$1.05$** | **$0.899$** | **$1.25$** | **`-0.1172` / `-0.1119` (`3.5×`)** | **`+0.0102` / `+0.0110` (`4.3×` cut)** | $7.42\%$ ($1.22\times 10^{-3}$, `1.000`) |
| **$0.60$ (2nd trough!)** | **$0.90$** | **$0.834$** | **`1.00` *(trough!)*** | $-0.0760$ / $-0.0759$ | **`+0.0376` / `+0.0347` (`3.2×` worse!)** | $8.94\%$ ($1.98\times 10^{-3}$, `1.003`) |
| $0.75$ | $0.75$ | $0.818$ | $0.81$ | $-0.0759$ | $+0.0240$ | $8.48\%$ ($1.45\times 10^{-3}$, `1.001`) |

- **Blade-count symmetry audit (`stand_2T_aligned`, $L_b = 0.45$, `40×120`):** Under chiral driving ($n_v = 0.04$), aligned standing $2T$ decays slowly with even $N_b = 4$ (`afit = -0.0242, E = 1.57e-8`, only $2.9\times$ below bare `4.53e-8`), whereas prime **$N_b = 5$** cuts energy **$36\times$ below bare and $12.5\times$ below $N_b = 4$** (`afit = -0.0366, E = 1.26e-9`).

### Experiment 3 — Multi-Mode Cavity Spectra, Baffle–Cavity Frequency-Retuning Law & $\bar{M} = 0.12$ Shootout

#### A. Exact Multi-Mode Chebyshev Spectra ($m = 0, \pm 1, \pm 2, \pm 3$; $n_p = 0.22, n_v = 0.04$)

| Configuration | $\bar{M}$ | $m = 0$ | $1T$ (co-rot) | $2T$ (co-rot) | $3T$ (co-rot) | **Global $\alpha_{\max}$ (Mode)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Bare | $0.12$ | $-0.03961$ | $+0.03506$ | $+0.01137$ | $-0.02509$ | **`+0.03506` ($1T$)** |
| $2\%$ `80/20` ring | $0.12$ | $-0.04134$ | $+0.01599$ | $+0.00099$ | $-0.02560$ | `+0.01599` ($1T$) |
| $4\%$ `100/0` ($1T$-only) | $0.12$ | $-0.04310$ | $+0.00002$ | **`+0.00946`** | $-0.02501$ | `+0.00946` ($2T$ escapes!) |
| $4\%$ `50/50` split | $0.12$ | $-0.04301$ | **`+0.01402`** | $-0.05326$ | $-0.02780$ | `+0.01402` ($1T$ escapes!) |
| **$4\%$ `80/20` ring** | **$0.12$** | $-0.04303$ | **$-0.00172$** | **$-0.00921$** | **$-0.02621$** | **`-0.00172` (ALL STABLE!)** |
| **$5\%$ `80/20` ring** | **$0.12$** | $-0.04387$ | **$-0.00789$** | **$-0.01425$** | **$-0.02656$** | **`-0.00789` (ALL STABLE!)** |
| Bare ($2T$ trap!) | $0.16$ | $-0.07272$ | $-0.02700$ | **`+0.00941`** | $-0.06349$ | **`+0.00941` ($2T$ trap!)** |
| **Micro $0.6\%$ $2T$-trim** | **$0.16$** | $-0.07385$ | $-0.02914$ | **$-0.01808$** | $-0.06353$ | **`-0.01808` ($2T$)** |
| **Micro $1.0\%$ $2T$-trim** | **$0.16$** | **$-0.07423$** | **$-0.03059$** | **$-0.04043$** | **$-0.06358$** | **`-0.03059` (`3.9×` deeper than $5\%$ at $0.12$!)** |

*(Key structural change from 1D rings in `007`–`008`: physical axial throughflow naturally stabilizes $3T$ at $\alpha_{3T} = -0.025\text{ to }-0.064$, so Law #6's `80/20` split achieves **all-mode** stability without a $3T$ escape—though stabilizing the severe $\bar{M} = 0.12$ trough by cavities alone requires $\beta \ge 4\text{--}5\%$.)*

#### B. The Baffle–Cavity Frequency-Retuning Law ($N_b = 5, L_b = 0.45$ + $2\%$ `80/20` Cavity at $\bar{M} = 0.12$)

| Linear Test ($A_0 = 0.002$) | Grid | Fitted $\alpha_{\text{late}}$ | Late $E_{\text{wave}}$ (`l/m`) | Verdict |
| :--- | :---: | :---: | :---: | :--- |
| Bare chamber | `40×120` | $+0.0330$ *(saturating)* | $2.81\times 10^{-3}$ (`3.368`) | Violently unstable |
| $2\%$ `80/20` cavity alone | `40×120` | $+0.0239$ | $1.09\times 10^{-4}$ (`6.159`) | Unstable alone |
| $N_b = 5, L_b = 0.45$ baffles alone | `40×120` | $+0.0105$ | $1.11\times 10^{-5}$ (`2.030`) | Unstable alone |
| Naive hybrid (untuned $\omega_H = 0.989$) | `40×120` / `60×160` | $-0.0048$ / **`-0.0043`** | $4.34\text{--}5.86\times 10^{-7}$ | Weak (detuned $-9.1\%$!) |
| **Retuned hybrid ($\omega_H = \omega_{1T}(0.45) = 0.899$)** | **`40×120` / `60×160`** | **`-0.0215` / `-0.0216`** | **$1.82\text{--}4.32\times 10^{-8}$ (`0.207`)** | **`5.0×` deeper margin (`60×160` converged!)** |
| Standalone $5\%$ wide-mouth cavity | `40×120` | $-0.0215$ | $9.11\times 10^{-9}$ (`0.213`) | Same margin, **$2.5\times$ more volume** |

- **Finite-amplitude $6.9\%$ shock shootout at $\bar{M} = 0.12$ (`40×120`, $A_0 = 0.055, t = 160$):** Bare settles at $8.86\%$ ($E = 3.73\times 10^{-3}$); tight-neck $5\%$ cavity ($\sigma_o = 2.0\%$) chokes to $6.96\%$ ($2.04\times 10^{-3}$); wide-mouth $5\%$ ($\sigma_o = 7.0\%$) cuts energy $2.20\times$ to $5.10\%$ ($9.27\times 10^{-4}$), close to the ideal linear-cavity floor $4.53\%$ ($7.00\times 10^{-4}$, proving the subcritical cubic flame triggers at $A_0 = 0.055$ even with zero orifice drag); **retuned `L_b = 0.45 + 2%` hybrid** achieves the **lowest nonlinear wave energy of all physical hardware at $\bar{M} = 0.12$** ($5.56\%, E = 7.22\times 10^{-4}$, **$5.2\times$ below bare** with $2.5\times$ less cavity volume than a $5\%$ ring).

![Inviscid Baffle Strouhal Shedding and Baffle-Cavity Frequency-Retuning Law](../results/exp010_baffle_strouhal_retuning.png)

### Experiment 4 — Disarming the $\bar{M} = 0.16$ $2T$ Strouhal Trap & 2D Euler Orifice-Drag Cliff (`48×160`, $t = 160$)

All three $\sigma_o$ designs below share the **exact same linear multi-mode spectrum** ($\beta = 1.0\%$, $\omega_H = 1.94, \xi_H = 0.08$, $\alpha_{2T} = -0.0404, \alpha_{\max} = -0.0306$):

| Case (`48×160`, $\bar{M} = 0.16, n_p = 0.22, n_v = 0.04$) | IC ($A_0$) | Late $P_{\text{peak}}/\bar{p}$ | Late $E_{\text{wave}}$ (`l/m`) | Fitted $\alpha_{\text{late}}$ | Outcome |
| :--- | :---: | :---: | :---: | :---: | :--- |
| Bare $\bar{M} = 0.16$ ($2T$ Strouhal trap) | `spin_2T` ($0.040$) | **`6.58%`** | $1.06\times 10^{-3}$ (`1.003`) | $+0.0000$ | Sustained $2T$ rotating shock |
| **Micro $1.0\%$ $2T$-trim: tight neck ($\sigma_o = 1.2\%$)** | `spin_2T` ($0.040$) | **`4.90%`** | **`6.22e-4` (`1.008`)** | $+0.0001$ | **CHOKES & TRIGGERS!** (Orifice-drag cliff!) |
| Micro $1.0\%$ $2T$-trim: medium ($\sigma_o = 2.5\%$) | `spin_2T` ($0.040$) | $1.86\%$ | $4.99\times 10^{-5}$ (`0.273`) | $-0.0164$ | Slow bleed |
| **Micro $1.0\%$ $2T$-trim: wide mouth ($\sigma_o = 5.0\%$)** | **`spin_2T` ($0.040$)** | **`0.04%`** | **`1.69e-8` (`0.007`)** | **`-0.0534`** | **TOTAL QUENCH (`62,700×` vs. bare, `36,800×` vs. tight!)** |
| Micro $1.0\%$ $2T$-trim: wide mouth ($\sigma_o = 5.0\%$) | `spin_3T` ($0.040$) | **$0.00\%$** | $3.01\times 10^{-12}$ (`0.008`) | $-0.0586$ | Total quench |
| Micro $1.0\%$ $2T$-trim: wide mouth ($\sigma_o = 5.0\%$) | `cold_1L` ($0.040$) | **$0.00\%$** | $3.14\times 10^{-12}$ (`0.001`) | $-0.0797$ | Total quench |
| *Micro $1.0\%$ $2T$-trim: wide mouth ($\sigma_o = 5.0\%$)* | *`spin_1T` ($0.055$)* | *`3.95%`* | *$4.51\times 10^{-4}$ (`1.000`)* | *$+0.0000$* | *Subcritically triggered $1T$ (bare $1T$ margin `-0.0306`)* |
| *Micro $1.0\%$ $2T$-trim: wide mouth ($\sigma_o = 5.0\%$)* | *`stand_1T` ($0.055$)* | *`3.95%`* | *$4.50\times 10^{-4}$ (`1.068`)* | *$+0.0009$* | *Subcritically triggered $1T$ (spins up to $3.95\%$)* |

![2D Baffle-Tip Vorticity Shedding and Micro-Cavity Orifice-Drag Cliff](../results/exp010_orifice_cliff_and_2T_rescue.png)

Full metrics: [`../results/exp010_metrics.json`](../results/exp010_metrics.json).

## 4. Verdict & Dead Ends

- **Dead Ends & Traps Exposed in Full 2D Compressible Euler Throughflow:**
  1. **Over-long baffles hit the $\mathrm{St}_{\text{eff}} = 1.00$ Strouhal trough:** Extending $N_b = 5$ inviscid baffles past the Strouhal peak ($L_b = 0.45 \to 0.60$) moves $L_{\text{eff}} = L_a - L_b$ onto the next integer trough ($\mathrm{St}_{\text{eff}} = 1.00$) and **re-destabilizes $1T$ growth by $3.2\times$** (`+0.0110 → +0.0347` on `60×160`), reversing zero-throughflow monotonicity (Law #5). Short baffles ($L_b = 0.15 < x_f$) also amplify chiral growth (`+0.0393 → +0.0411`) and shock peaks (`8.86% → 9.54%`).
  2. **Detuned Baffle–Cavity Hybrids:** Pairing $L_b = 0.45$ baffles with a $2\%$ faceplate cavity tuned to the *unbaffled* frequency ($\omega_H = 0.989$) wastes **$80\%$** of linear damping margin (`-0.0043` vs. `-0.0216` on `60×160`) because baffles downshift $\omega_{1T}$ by $-9.1\%$ (`0.997 → 0.899`).
  3. **2D Compressible Euler Orifice-Drag Cliff:** With identical linear all-mode stability ($\alpha_{2T} = -0.0404$), a tight-neck ($\sigma_o = 1.2\%$) $1.0\%$ $2T$-trim cavity chokes on a $4.0\%$ $2T$ shock and locks onto a $4.90\%$ rotating shock (`l/m = 1.008`), whereas a wide mouth ($\sigma_o = 5.0\%$) annihilates it by $36{,}800\times$ (`E = 1.69e-8`).
  4. **Honest Subcritical Trade-off at $\bar{M} = 0.16$ vs. $\bar{M} = 0.18$:** Although a micro $1.0\%$ wide-mouth $2T$-trim cavity at $\bar{M} = 0.16$ stabilizes all four modes linearly ($\alpha_{\max} = -0.0306$) and annihilates $2T, 3T, 1L$ shocks on `48×160`, leaving $1T$ at its bare $\bar{M} = 0.16$ margin (`-0.0306`) still permits a large $6.9\%$ $1T$ bomb to trigger a $3.95\%$ cycle—showing why shifting to $\bar{M} = 0.18$ ($\alpha_{1T} = -0.0865$) remains uniquely superior for extreme $1T$ bomb immunity when cycle Mach allows.
- **What Worked (Law #13 for `SYNTHESIS.md`):**
  1. **Inviscid Baffle Strouhal Vorticity Shedding:** In steady axial throughflow, strictly inviscid reflecting baffles shed trailing-edge vortex streets (breaking Dead End #6) and triple cold $1T$ damping at $L_b = 0.45$ (`-0.0317 → -0.1119` on `60×160`, $4.3\times$ reacting growth cut).
  2. **Baffle–Cavity Frequency-Retuning Law:** Retuning a compact $2\%$ `80/20` faceplate cavity ring to the baffled $1T$ frequency $\omega_{1T}(L_b) = 0.899$ boosts hybrid linear damping **$5.0\times$** (`-0.0043 → -0.0216`, `60×160` grid-converged to `1e-4`), matching a $5\%$ standalone cavity ring with **$2.5\times$ less volume** and achieving the lowest finite-amplitude wave energy ($5.2\times$ below bare) at the $\bar{M} = 0.12$ trough.
  3. **Micro $1.0\%$ Wide-Mouth $2T$-Trim Rescue at $\bar{M} = 0.16$:** Plugs Entry `009`'s $2T$ Strouhal harmonic trap cleanly ($\alpha_{2T}: +0.0094 \to -0.0404$, $62{,}700\times$ $2T$ shock cut on `48×160`) with **$5\times$ less cavity volume** than full trough stabilization.
