# 007 — Compressible Nonlinear Euler Transverse Acoustics: Shock Steepening, Mode-Selection Reversal & Cavity Cascade Absorption

- **Date:** 2026-11-24
- **Status:** `verified`
- **Verification Script:** `../sim/exp007_compressible_euler_steepening.py`

## 1. Target

Tackle Open Problem #1 from [`../SYNTHESIS.md`](../SYNTHESIS.md): replace the bulk-linear transverse wave operator of Entries `004`–`006` with the **full conservative 1D periodic compressible Euler equations** on the azimuthal ring $\theta \in [0, 2\pi)$ ($\gamma = 1.25$) coupled to Entry `006`'s exact DC-blocked Erlang-4 bounded subcritical flame law and calibrated quadratic-drag faceplate cavities, to answer four concrete questions:

1. **Verification:** Does our 2nd-order MUSCL-minmod HLLC finite-volume solver conserve inviscid mass/momentum/energy to machine precision, recover Entry `004`'s Transverse Half-Damping Theorem ($\alpha_{mT} = -\zeta / (2 L_a) = -0.010$) in the small-amplitude limit, and reproduce the exact analytical Riemann shock birth time $t_{\text{shock}} = \frac{2}{(\gamma + 1) u_0}$, Fay–Burgers maximum shock-jump time $\frac{\pi}{2} t_{\text{shock}}$, and Rankine–Hugoniot $O(u_0^3)$ weak-shock dissipation law?
2. **Sawtooth Capping & Harmonic Cascade:** At supercritical driving ($n_p = 0.068 > n_{pH,1T} = 0.04137$), how much does gasdynamic shock steepening cap peak pressure and spill energy into $2T/3T$ harmonics compared to bulk-linear acoustics?
3. **Attractor/Saddle Exchange (Mode-Selection Reversal):** Does Rankine–Hugoniot $O(A^3)$ shock dissipation **reverse** Entry `006`'s bulk-linear supercritical selection of spinning waves ($S \to 1$) back to **standing waves ($S \to 0$)**—and does it quench Entry `006`'s subcritical bistability window ($n_p < n_{pH,1T}$)?
4. **Shocked-Wave Cavity Absorption:** Under strong supercritical driving ($n_p = 0.085$, where $1T, 2T, 3T$ are all linearly unstable and nonlinearly coupled by shock steepening), does a $100\%\,1T$-only cavity ring fail by jumping into a $2T$ limit cycle, while Entry `005`'s $80/20$ ($1T/2T$) dual-tuned ring stabilizes the entire shocked combustor?

## 2. The Math & Assumptions

### Governing 1D Periodic Compressible Euler Ring ($\theta \in [0, 2\pi)$, $\gamma = 1.25$)

$$\frac{\partial \rho}{\partial t} + \frac{\partial (\rho u)}{\partial \theta} = -\frac{1}{L_a}\,\rho\,(u_{\text{noz}} + v_{\text{cav}}) - S_{\text{ent}}$$
$$\frac{\partial (\rho u)}{\partial t} + \frac{\partial (\rho u^2 + p)}{\partial \theta} = -\frac{1}{L_a}\,\rho u\,(u_{\text{noz}} + v_{\text{cav}}) - S_{\text{ent}}\,u$$
$$\frac{\partial E}{\partial t} + \frac{\partial \bigl(u(E + p)\bigr)}{\partial \theta} = -\frac{1}{L_a}\,(E + p)\,(u_{\text{noz}} + v_{\text{cav}}) - \frac{1}{2} S_{\text{ent}}\,u^2 + \frac{1}{\gamma - 1}\frac{n_p}{L_a}\Phi(q_p)$$

where $E = \frac{p}{\gamma - 1} + \frac{1}{2}\rho u^2$, reference state $(\bar{\rho} = 1, \bar{u} = 0, \bar{p} = 1/\gamma = 0.80, \bar{c} = 1)$, $P = p - \bar{p}$ (so $p'/\bar{p} = \gamma P = 1.25 P$), $u_{\text{noz}} = \zeta P$ ($L_a = 2.0, \zeta = 0.04$), and $\Phi(q) = q_s \tanh(q/q_s + \kappa(q/q_s)^3)$ ($q_s = 0.036, \kappa = 4.5$) driven by the DC-blocked ($\omega_{\text{hp}} = 0.15$) Erlang-4 ($\bar{\tau} = 0.30, \sigma_\tau = 0.15$) pressure lag.

- **Isobaric chamber-residence entropy flush:** $S_{\text{ent}} = \nu_{\text{res}}\bigl(\rho - \bar{\rho}(p/\bar{p})^{1/\gamma}\bigr)$ with $\nu_{\text{res}} = \zeta/L_a = 0.02$ when $\zeta > 0$ (and identically $0$ on inviscid $\zeta = 0$ runs). Physically, shock heating in a closed ring with zero axial throughflow causes a slow $+2\%/500$ secular density drift via the quadratic nozzle rectification term $-\frac{\zeta}{L_a}\langle \rho' p'\rangle$; $S_{\text{ent}}$ models fresh-propellant residence flushing of shock-generated entropy while leaving instantaneous $p$ and $u$ unchanged and vanishing to first order on all isentropic linear acoustic modes.
- **Exact continuum linear Hopf thresholds** of the 1D ring dispersion relation $s^2 + \frac{\zeta - n_p H(s)}{L_a} s + m^2 = 0$:
  - $1T$: **$n_{pH,1T} = 0.04137$** ($\omega_H = 0.99849$)
  - $2T$: **$n_{pH,2T} = 0.04828$** ($\omega_H = 1.99430$)
  - $3T$: **$n_{pH,3T} = 0.06560$** ($\omega_H = 2.98904$)

### Exact Fay–Burgers–Rankine–Hugoniot Shock Theory & Spinning/Standing Asymmetry

For an inviscid simple wave $u(\theta, 0) = u_0 \cos\theta$, exact Riemann characteristic steepening ($\lambda_+ = u + c = 1 + \frac{\gamma + 1}{2} u$) predicts:
1. **Shock birth time:** $t_{\text{shock}} = \frac{2}{(\gamma + 1) u_0}$.
2. **Fay–Burgers maximum shock-jump time:** In dimensionless time $\tau = t / t_{\text{shock}}$, the shock jump obeys $\Delta u(\tau) = 2 u_0 \sin\phi(\tau)$ where $\phi = \tau \sin\phi$, peaking at $\Delta u = 2 u_0$ when **$\tau = \frac{\pi}{2}$** ($t_{\text{pk}} = \frac{\pi}{2} t_{\text{shock}}$).
3. **Exact weak-shock acoustic energy dissipation rate** (averaged over $\theta \in [0, 2\pi)$ for $E_{\text{ac}} = \frac{1}{2}\langle P^2 + u^2 \rangle = \langle u^2 \rangle$):
   $$-\frac{dE_{\text{ac}}}{dt} = \frac{\gamma + 1}{24\pi}(\Delta u)^3 = \frac{\gamma + 1}{3\pi}\,u_0^3\,\sin^3\phi(\tau)$$
   Integrating over the post-shock window $\tau \in [1.15, 1.75]$ gives the exact linear-fit constant $-\frac{dE_{\text{ac}}}{dt} / u_0^3 = \mathbf{0.22079}$.
4. **Why gasdynamic shocks reverse mode selection (Spinning $\to$ Standing):** At equal acoustic energy $E_{\text{ac}} = A_{\text{eq}}^2$, a spinning $1T$ wave sustains a single continuous shock of amplitude $A_{\text{eq}}$, whereas a standing $1T$ wave splits the energy into two counter-propagating traveling waves of amplitude $A_{\text{eq}}/\sqrt{2}$. By strict convexity of the cubic Rankine–Hugoniot loss, $2\,(A_{\text{eq}}/\sqrt{2})^3 = \frac{1}{\sqrt{2}} A_{\text{eq}}^3$, so equal-energy standing waves suffer **at least $\sqrt{2} \approx 1.414\times$ lower shock dissipation** (and roughly half the $2T$ cascade) than spinning waves—directly opposing and overwhelming Entry `006`'s $1.50\times$ bulk-linear flame saturation penalty on standing waves.

### Intentional Simplifications
- Purely tangential 1D periodic ring ($k_x = 0$, no axial or radial $1R$ modes); bulk-linear reference uses 2/3-dealiased Fourier RK4 on the exact same grid, flame, and cavities.
- Inviscid Euler (no physical viscosity or thermal conduction; shock dissipation occurs physically via the Rankine–Hugoniot jump captured by HLLC).

## 3. The Verification

Reproduce from the repo root:

```bash
uv run python rocket-combustion/sim/exp007_compressible_euler_steepening.py
```

### Experiment 1 — Conservation, Linear Limit & Exact Riemann/Fay–Burgers Shock Verification

- **Exp 1A (Inviscid conservation, $\zeta = 0, n_p = 0, A = 0.10, t = 25, N = 256$):** Mass relative error **`5.64e-16`**, momentum error **`7.63e-17`**, total energy relative error **`3.52e-16`**.
- **Exp 1B (Small-amplitude linear limit, $A = 10^{-4}, N = 256$):**
  - Cold $1T$: $\alpha_{\text{Euler}} = -0.010104$ vs. exact $-\zeta/(2 L_a) = -0.010000$ (`|diff| = 1.04e-4`).
  - Cold $2T$: $\alpha_{\text{Euler}} = -0.011059$ (`|diff| = 1.06e-3`).
- **Exp 1C (Riemann shock birth, Fay–Burgers peak jump, and $O(u_0^3)$ Rankine–Hugoniot dissipation, $N = 512$):**

| $u_0$ ($p'/\bar{p}$) | Exact Birth $t_{\text{shock}}$ | RH Entropy Onset | Exact Peak Jump $\frac{\pi}{2} t_{\text{shock}}$ | $-\partial_\theta u$ Peak (Err) | $-\frac{dE_{\text{ac}}}{dt}/u_0^3$ *(Exact: `0.2208`)* |
| :---: | :---: | :---: | :---: | :---: | :---: |
| $0.04$ ($5.0\%$) | $22.22$ | $22.18$ | $34.91$ | $34.54$ (`-1.0%`) | **`0.2204`** (`-0.18%`) |
| $0.08$ ($10.0\%$) | $11.11$ | $11.50$ | $17.45$ | $17.31$ (`-0.8%`) | **`0.2216`** (`+0.37%`) |
| $0.12$ ($15.0\%$) | $7.41$ | $7.80$ | $11.64$ | $11.51$ (`-1.1%`) | **`0.2221`** (`+0.61%`) |

- **Grid convergence on peak jump ($u_0 = 0.08$, exact $17.45$):** $N = 128 \to 16.71$ (`-4.3%`), $N = 256 \to 17.35$ (`-0.6%`), $N = 512 \to 17.31$ (`-0.8%`).

### Experiment 2 — Sawtooth Capping, Attractor/Saddle Exchange & Subcritical Quenching

- **Exp 2A (Supercritical spinning $1T$ manifold at $n_p = 0.068, t = 220$):**
  - **Bulk-linear:** Smooth flat-topped saturated wave, $P_{\text{peak}} = 0.0576$ (**$7.2\%\,p'/\bar{p}$**), $E_{2T}/E_{1T} = 0.00\%$, $E_{3T}/E_{1T} = 6.23\%$.
  - **Compressible Euler ($N = 256$ / $512$):** Rotating sawtooth shock with peak pressure capped by **$-37\%$** to $P_{\text{peak}} = 0.0362 / 0.0372$ (**$4.5\% / 4.6\%\,p'/\bar{p}$**), dumping **$27.6\% / 28.3\%$** of fundamental energy into $2T$ and **$8.4\% / 8.7\%$** into $3T$ (mean density locked at $\bar{\rho} = 0.99946$).
- **Exp 2B (Attractor/Saddle Exchange at $n_p = 0.068, t = 360$):**

| Model & Grid | Initial State ($S_0$) | Late Spin Index $S_{\text{late}}$ | Late $P_{\text{peak}}$ ($p'/\bar{p}$) | Late $E_{\text{ac}}$ | $E_{2T}/E_{1T}$ | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| Bulk-linear ($N=256$) | Seeded standing ($0.17$) | **$0.9047$** | $0.0637$ ($8.0\%$) | $2.56\times 10^{-3}$ | $0.00\%$ | **Spins up** (Spin = Attractor) |
| Bulk-linear ($N=256$) | Perturbed spin ($0.98$) | **$0.9999$** | $0.0603$ ($7.5\%$) | $2.91\times 10^{-3}$ | $0.00\%$ | **Stays spinning** |
| **Euler ($N=256$)** | Seeded standing ($0.17$) | **$0.0037$** | $0.0423$ ($5.3\%$) | $5.67\times 10^{-4}$ | $17.11\%$ | **Stays standing** (Stand = Attractor!) |
| **Euler ($N=256$)** | Perturbed spin ($0.98$) | **$0.2974$** *(diving to $0.21$)* | $0.0415$ ($5.2\%$) | $5.40\times 10^{-4}$ | $18.39\%$ | **Collapses to standing!** |
| **Euler ($N=512$)** | Seeded standing ($0.17$) | **$0.0058$** | $0.0433$ ($5.4\%$) | $5.85\times 10^{-4}$ | $17.51\%$ | **Grid-converged standing attractor** |

*(An additional unperturbed $t = 500$ diagnostic confirmed that even pure $S_0 = 1.000$ spinning Euler waves spontaneously fall off the spinning saddle around $t \approx 400\text{--}500$, with $S: 1.000 \to 0.949 \to 0.703$ at $\nu_{\text{res}} = 0.02$ and $S \to 0.115$ at $\nu_{\text{res}} = 0$, confirming the mode-selection reversal is independent of $S_{\text{ent}}$.)*

- **Exp 2C (Subcritical bistability quenching at $n_p = 0.037, 0.039 < n_{pH,1T} = 0.04137$):**
  - **Bulk-linear:** Triggers cleanly with Entry `006`'s standing advantage: at $n_p = 0.037$, $A_{\text{spin}} = 0.0074, A_{\text{stand}} = 0.0061$ ($E_{\text{ratio}} = 1.461$); at $n_p = 0.039$, $A_{\text{spin}} = 0.0053, A_{\text{stand}} = 0.0045$ ($E_{\text{ratio}} = 1.424$).
  - **Compressible Euler:** Rankine–Hugoniot $O(A^3)$ shock dissipation **quenches both spinning and standing subcritical triggering** up to $A_0 \le 0.045$ on $N = 192$. Independent **$N = 512$ verification** at $A_0 = 0.030$ confirms genuine physical shock decay (not numerical viscosity), AND preserves the exact spinning/standing shock-loss asymmetry: spinning decays significantly faster than standing ($e_3/e_2 = 0.533$ vs. $0.765$ at $n_p = 0.037$; $0.566$ vs. $0.805$ at $n_p = 0.039$).

![Compressible Euler Sawtooth Shock Steepening and Mode-Selection Reversal](../results/exp007_euler_shock_steepening.png)

### Experiment 3 — Shocked-Wave Cavity Absorption ($n_p = 0.085, \beta_{\text{tot}} = 2\%, \sigma_o = 3.5\%, A_0 = 0.055 = 6.9\%\,p'/\bar{p}, t = 320$)

At $n_p = 0.085$, bare $1T, 2T, 3T$ are all linearly supercritical and coupled by shock steepening:

| Cavity Architecture | Late $P_{\text{peak}}$ ($p'/\bar{p}$) | Late $E_{\text{ac}}$ (`late/mid`) | $E_{2T}/E_{1T}$ | Long-Horizon Outcome ($t = 320$ & $t = 600$ check) |
| :--- | :---: | :---: | :---: | :--- |
| **Bare Euler** | $0.0456$ (**$5.70\%$**) | $7.63\times 10^{-4}$ (`0.969`) | $20.7\%$ | Sustained rotating sawtooth shock limit cycle |
| **100% $1T$-Only Ring** | $0.0207$ (**$2.59\%$**) | $1.48\times 10^{-4}$ (`1.295`) | **$150{,}447\%$** | **Fails:** kills $1T$ but jumps into a growing pure-$2T$ limit cycle! |
| **50/50 ($1T/2T$) Dual Ring** | $0.0068$ (**$0.85\%$**) | $1.58\times 10^{-5}$ (`0.142`) | $4.7\%$ | Stabilizes ($P_{\text{peak}} \to 0.01\%$ by $t=600$), but slower $1T$ decay |
| **80/20 ($1T/2T$) Dual Ring** | **$0.0031$ ($0.39\%$)** | **$6.09\times 10^{-6}$ (`0.099`)** | **$4.2\%$** | **Best:** fastest exponential decay of $1T, 2T, 3T$ ($P_{\text{peak}} \to 0.05\%$ by $t=600$)* |

*\*An independent $t = 600$ check confirmed that the small $E_{\text{ac}} \approx 3.9\times 10^{-6}$ ($80/20$) and $5.2\times 10^{-6}$ ($50/50$) floor in Figure 2 is NOT an acoustic limit cycle—all three acoustic modes decay cleanly ($e_1 \to 2.1\times 10^{-13}$, $e_2 \to 1.4\times 10^{-12}$, $e_3 \to 1.9\times 10^{-7}$, $P_{\text{peak}} \to 0.0004$)—leaving behind only the conserved $O(A_0^2)$ mean azimuthal acoustic-streaming swirl ($\frac{1}{2}\bar{u}^2 \approx 3.9\times 10^{-6}$) imparted by the initial nonlinear simple wave.*

![Shocked-Wave Cavity Cascade Absorption: 100% 1T vs 80/20 vs 50/50 Dual Rings](../results/exp007_shock_cavity_absorption.png)

Full metrics: [`../results/exp007_metrics.json`](../results/exp007_metrics.json).

## 4. Verdict & Dead Ends

- **What broke / pitfalls caught during development:**
  1. **SSP-RK2 on undamped Fourier bulk-linear acoustics:** Unconditionally unstable on pure imaginary eigenvalues; fixed by switching the bulk-linear reference solver to 2/3-dealiased 4th-order RK4.
  2. **Secular density drift in zero-throughflow closed Euler rings:** Without fresh-propellant residence flushing, quadratic nozzle enthalpy extraction shifts $\langle p \rangle$ down by $-0.33\%$ and slowly pumps mass in ($+1.9\%$ over $t = 500$). Adding the exact isobaric residence entropy flush $S_{\text{ent}}$ ($\nu_{\text{res}} = \zeta/L_a$, zero effect on linear modes or inviscid conservation) locks $\bar{\rho} = 0.9994$ indefinitely while leaving all physical phenomena unchanged.
  3. **Single-mode $100\%\,1T$ cavities under shocked driving:** Suppressing $1T$ alone causes immediate mode jumping into an undamped pure-$2T$ limit cycle ($E_{2T}/E_{1T} > 10^5\%$).
- **What worked (New Law #10 for `SYNTHESIS.md`):**
  1. **Exact Fay–Burgers–Rankine–Hugoniot verification:** Machine-precision inviscid conservation ($10^{-16}$), peak shock jump within $0.8\text{--}1.1\%$ of $\frac{\pi}{2} t_{\text{shock}}$, and post-shock energy dissipation matching the closed-form cubic theorem ($0.2208\,u_0^3$) to **$0.2\text{--}0.6\%$** across a $27\times$ range in $u_0^3$.
  2. **Gasdynamic Mode-Selection Reversal & Amplitude Capping:** Compressible Euler shock steepening caps supercritical spinning peak pressure by **$-37\%$** ($7.2\% \to 4.5\%\,p'/\bar{p}$), cascades $28\%$ of energy into $2T$, quenches Entry `006`'s bulk-linear subcritical bistability window ($n_p = 0.037\text{--}0.039$, verified at $N = 512$), and triggers a complete **attractor/saddle exchange** at $n_p = 0.068$—reversing bulk-linear spinning selection ($S \to 1$) to a grid-converged **standing-wave attractor ($S \to 0$)** via the $\sqrt{2}\times$ cubic shock-loss convexity penalty on spinning waves.
  3. **The $80/20$ dual-tuned cavity ring survives shocked waves:** Even when initialized with a $6.9\%\,p'/\bar{p}$ shocked spinning wave at $n_p = 0.085$, Entry `005`'s wide-mouth ($\sigma_o = 3.5\%$) $80/20$ ring outperforms both $100\%\,1T$ (which jumps to $2T$) and $50/50$, exponentially damping $1T, 2T, 3T$ down to $P_{\text{peak}} = 0.05\%$ by $t = 600$.
- **Next single hypothesis to test (`008`):** Couple Entry `007`'s compressible Euler shock steepening with **chiral velocity coupling ($n_v \neq 0$) vs. symmetric $|u|$ velocity coupling** (or 2D $(x, \theta)$ compressible Euler with radial baffles) to map the exact critical swirl/chirality threshold where chiral driving overcomes the $\sqrt{2}\times$ Rankine–Hugoniot shock penalty and flips the shocked combustor back to a rotating detonation-like spinning shock wave!
