# 006 — 2D Nonlinear Spinning vs. Standing Mode Selection & Long-Horizon Cavity Bomb Tests

- **Date:** 2026-11-23
- **Status:** `verified` (hardened after Peer Reviews #2 & #3)
- **Verification Script:** `../sim/exp006_2d_nonlinear_modes.py`

## 1. Target

Unite our **1D nonlinear lessons (Entry `003`)** with our **2D transverse/baffle-free architectures (Entries `004`–`005`)** inside a single 2D nonlinear cylindrical-annulus PDE (carrying forward DC-blocking $\omega_{\text{hp}} = 0.15$, causal Erlang-4 time lag $\bar{\tau} = 0.30, \sigma_\tau = 0.15$, flame plane $x_f = 1/6 \approx 0.1667$, and the bounded monotonic subcritical flame law $\Phi(q) = q_s\tanh(q/q_s + \kappa(q/q_s)^3)$), to test:

1. **2D $O(2)$ Symmetry-Breaking Mode Selection (Axisymmetric Chamber, $n_v = 0$, bare linear Hopf $n_{pH} = 0.04094$):**
   - At equal bomb energy $E_0$ in the subcritical regime ($n_p < n_{pH}$), does a **standing $1T$ bomb** trigger at $\sim 1.50\times$ *lower* energy than a **spinning $1T$ bomb** (known cubic $O(2)$ theory; cf. Noiray et al. 2011; Ghirardo & Juniper 2013)?
   - How does spin selection ($S(t)$) evolve separately on **(a) the subcritical triggered upper branch ($n_p = 0.037 < n_{pH}$)** vs. **(b) the strongly saturated supercritical branch ($n_p = 0.068 > n_{pH}$)**?
2. **Long-Horizon ($t = 900$) 2D Nonlinear Bomb Scan of the $80/20$ Dual-Tuned Faceplate Cavity Ring:**
   - First verify **$t = 1000$ broadband linear stability** across all modes ($1T, 2T$, both spin directions, $1L$; the least-damped linear eigenmode has $|\alpha_{\min}| \approx 0.0018$, time scale $\sim 550$).
   - Run a **dense amplitude scan** ($A_{\text{eq}} \in \{0.010, 0.015, 0.022, 0.032, 0.045, 0.060, 0.072\}$, keeping $p'/\bar{p} \le 9.0\%$ for spinning waves) over **$t = 900$** without assuming monotonic bisection, quantifying how **(a) cavity mouth open-area ratio $\sigma_o$ ($1.5\%$ vs. $3.5\%$)** and **(b) linear damping ratio $\xi_H / \xi_H^{\text{EP}}$** govern finite-amplitude $1T$ bomb survival.

## 2. The Math & Assumptions

### Bounded Subcritical 2D Flame Law & Energy-Matched Bomb Ratio
$$\Phi(q) = q_s \tanh\!\left(\frac{q}{q_s} + \kappa\left(\frac{q}{q_s}\right)^3\right), \qquad q_s = 0.036,\quad \kappa = 4.5$$

Consider two $1T$ initial perturbations with **identical total acoustic energy** $E_0 = \pi L_a A_{\text{eq}}^2$:
- **Spinning $1T$:** $P(\theta, t) = A_{\text{eq}}\cos(\theta - \omega t)$, $U_\theta = +P$.
- **Standing $1T$:** $P(\theta, t) = \sqrt{2}\,A_{\text{eq}}\cos(\theta)\cos(\omega t)$, $U_\theta = \sqrt{2}\,A_{\text{eq}}\sin(\theta)\sin(\omega t)$.

In the small-amplitude subcritical cubic expansion ($\Phi(q) \approx q + c_3 q^3$), the **space-time averaged** quartic Rayleigh energy boost over $\theta \in [0, 2\pi)$ and one period $T = 2\pi/\omega$ is:
$$\langle P_{\text{stand}}^4 \rangle_{\theta,t} = (\sqrt{2}\,A_{\text{eq}})^4 \langle\cos^4\theta\rangle_\theta \langle\cos^4\omega t\rangle_t = 4 \cdot \frac{3}{8} \cdot \frac{3}{8}\,A_{\text{eq}}^4 = \frac{9}{16} A_{\text{eq}}^4$$
$$\langle P_{\text{spin}}^4 \rangle_{\theta,t} = A_{\text{eq}}^4 \langle\cos^4(\theta - \omega t)\rangle_{\theta,t} = \frac{3}{8} A_{\text{eq}}^4 \quad \Longrightarrow \quad \frac{\langle P_{\text{stand}}^4 \rangle_{\theta,t}}{\langle P_{\text{spin}}^4 \rangle_{\theta,t}} = \mathbf{1.50}$$
Spin coherence is tracked via the bounded spin index $S(t) = \frac{|\int P U_\theta\,d\theta|}{\frac{1}{2}\int(P^2 + U_\theta^2)\,d\theta} \in [0, 1]$.

### Calibrated 2D Cavity Orifice Drag & Discrete Energy Bookkeeping
Each rim cavity sector obeys $\frac{\partial v_r}{\partial t} + (2\xi_H\omega_H + c_{\text{nl}}|v_r|)v_r = \beta_s\omega_H^2(P(0, \theta) - P_c)$ with $c_{\text{nl}} = \frac{\beta_s\omega_H^2}{2 C_d^2 \sigma_o^2}$ ($C_d = 0.65$), pre-pressurized consistently at $t = 0$. Full 2D nonlinear energy balance (acoustic + cavity stored energy vs. nozzle loss + linear/quadratic orifice dissipation + nonlinear Rayleigh flame power) is verified directly on the nonlinear cavity PDE (`rel err = 4.89e-06`). Note that calibrated quadratic drag already modifies small-amplitude decay rates (e.g., at $P \sim 0.002$), so finite-amplitude cavity dynamics deviate from the pure linear-EP regime even well below saturation.

### Intentional Simplifications & Validity Caveats
- Thin-annulus 2D bulk-linear acoustics ($30\times 60$ grid; $x_f = 1/6 \approx 0.1667$ exact cell center).
- Dimensionless pressure converts via $p'/\bar{p} = \gamma_{\text{ref}} P$ ($\gamma_{\text{ref}} = 1.25$). In Experiment 1, subcritical settled amplitudes are $P_{\text{LC}} = 0.038\text{--}0.047$ ($p'/\bar{p} \approx 4.8\text{--}5.9\%$, bulk-linear); the supercritical spin-selection run ($n_p = 0.068$) reaches $P_{\text{peak}} = 0.0859$ ($10.7\%$), entering the weakly nonlinear acoustic regime where gasdynamic steepening would begin. In Experiment 2, all scanned spinning bombs stay within $A_{\text{eq}} \le 0.072$ ($p'/\bar{p} \le 9.0\%$).
- Chiral velocity coupling ($n_v = 0.025$) in Experiment 2 uses Entry `004`'s signed linear azimuthal velocity-coupling assumption (swirled/directional injectors).

## 3. The Verification

Reproduce from the repo root:

```bash
uv run python rocket-combustion/sim/exp006_2d_nonlinear_modes.py
```

### Experiment 1 — 2D Subcritical Standing vs. Spinning Triggering & Regime-Dependent Spin Selection ($n_v = 0$)

Exact bare 2D $1T$ linear Hopf bifurcation (at $x_f = 1/6$): **$n_{pH} = 0.04094$**, $\omega_H = 0.99923$.

| $n_p$ ($< n_{pH}$) | $A_{\text{crit}}(\text{spin})$ | $A_{\text{crit}}(\text{stand})$ | Energy Ratio $E_{\text{crit}}(\text{spin})/E_{\text{crit}}(\text{stand})$ | Settled Upper-Branch $P_{\text{LC}}(\text{spin})$ | Analytical Cubic Theory |
| :---: | :---: | :---: | :---: | :---: | :---: |
| $0.033$ | $0.0108$ | $0.0089$ | **$1.466$** *(2.3% from theory)* | $0.0383$ ($4.8\%\,p'/\bar{p}$) | $1.500$ |
| $0.037$ | $0.0070^*$ | $0.0058^*$ | **$1.488^*$** *(0.8% from theory)* | $0.0431$ ($5.4\%\,p'/\bar{p}$) | $1.500$ |
| $0.040$ | $0.0035$ | $0.0029$ | **$1.490$** *(0.7% from theory)* | $0.0468$ ($5.8\%\,p'/\bar{p}$) | $1.500$ |

*\*Horizon convergence check at $n_p = 0.037$: $t = 180$ vs. $t = 360$ yields $A_{\text{spin}} = 0.00700$, $A_{\text{stand}} = 0.00568$, $E_{\text{ratio}} = \mathbf{1.519}$ ($1.3\%$ from theory).*

- **Subcritical vs. supercritical spin selection (both initialized with seeded chiral asymmetry $\epsilon_s = 0.12 \implies S_0 = 0.169$):**
  - **Subcritical triggered standing bomb ($n_p = 0.037 < n_{pH}$, $A_{\text{eq}} = 0.014$, $t = 500$):** Settles at $P_{\text{peak}} = 0.0456$ ($5.7\%\,p'/\bar{p}$). During initial cubic-boost growth, $S(t)$ dips ($0.169 \to 0.13$), then slowly turns upward after saturation ($S_{\text{final}} = 0.159$ at $t = 500$), remaining predominantly standing over hundreds of acoustic periods.
  - **Supercritical saturated chamber ($n_p = 0.068 > n_{pH}$, $t = 360$):** Reaches deeper saturation ($P_{\text{peak}} = 0.0859 \implies 10.7\%\,p'/\bar{p}$, weakly nonlinear acoustic regime), rapidly amplifying the seeded chirality from $S_0 = 0.169$ to **$S = 0.879$** at $t = 360$.

![2D Nonlinear Symmetry Breaking: Subcritical vs Supercritical Spin Selection](../results/exp006_2d_spin_selection.png)

### Experiment 2 — Long-Horizon ($t = 900$) Dense Amplitude Scan on $80/20$ Dual-Tuned Cavity Ring ($n_p = 0.038, n_v = 0.025, \beta_{\text{tot}} = 2\%, N_{\text{pair}} = 5$)

- **Broadband linear checks ($t = 400$ for unstable baselines; $t = 1000$ for cavities):**
  - **Bare combined combustor:** **Linearly unstable** ($\alpha_{\text{late}} = +0.00505$).
  - **Pure 5-blade protruding baffles ($L_b = 0.60$, no cavities):** **Linearly unstable** ($\alpha_{\text{late}} = +0.00631$).
  - **All four $80/20$ cavity designs ($\xi/\xi_{\text{EP}} \in \{0.25, 0.50, 1.00, 2.00\}$):** **Linearly stable** over $t = 1000$ (late broadband rate $-0.0021$ to $-0.0044$; independent $t = 1200$ single-mode checks in Peer Review #3 confirm all designs share a least-damped asymptotic mode near $\alpha_{\min} \approx -0.0018$, time scale $\sim 550$).
- **2D nonlinear energy-balance check** (flame + quadratic orifice drag): **`rel err = 4.89e-06`**.
- **Strict trend-based trigger criterion** (code and doc matched identically): `triggered = (e_ratio > 1.005 and p_peak_late > 0.002) or (e_ratio >= 0.998 and p_peak_late > 0.032)` evaluated over **$t = 900$** across dense amplitude grid $A_{\text{eq}} \in \{0.010, 0.015, 0.022, 0.032, 0.045, 0.060, 0.072\}$:

| $\xi_H / \xi_H^{\text{EP}}$ | Broadband Linear $\alpha$ ($t \in [700, 1000]$) | Wide Mouth ($\sigma_o = 3.5\%$) Onset Bracket ($t=900$) | Tight Neck ($\sigma_o = 1.5\%$) Onset Bracket ($t=900$) | Wide vs. Tight Amplitude Gain |
| :---: | :---: | :---: | :---: | :---: |
| $0.25$ (Under-EP) | $-0.00213$ *(stable)* | **$(0.032,\; 0.045]$** | $(0.010,\; 0.015]$ | **$\sim 2.1\text{--}4.5\times$** *(~3x typical)* |
| $0.50$ | $-0.00304$ *(stable)* | **$(0.032,\; 0.045]$** | $(0.010,\; 0.015]$ | **$\sim 2.1\text{--}4.5\times$** |
| **$1.00$ (Linear 2D EP)** | $-0.00441$ *(stable)* | **$(0.032,\; 0.045]$** | $(0.010,\; 0.015]$ | **$\sim 2.1\text{--}4.5\times$** |
| **$2.00$ (Over-damped)** | $-0.00433$ *(stable)* | **$(0.015,\; 0.022]^\dagger$ *(DROPS ~2x!)*** | $(0.010,\; 0.015]$ | $\sim 1.0\text{--}2.2\times$ |

*$\dagger$Non-monotonicity note (visible in Figure 2 left): at over-damped $\xi_H/\xi_H^{\text{EP}} = 2.00$ ($\sigma_o = 3.5\%$), $A_{\text{eq}} = 0.022$ triggers while $A_{\text{eq}} = 0.032$ decays before $A_{\text{eq}} \ge 0.045$ triggers consistently—reinforcing why dense amplitude scanning is superior to monotonic bisection near the marginal manifold.*

- **Ideal linear cavity ($c_{\text{nl}} = 0$) at $t = 900$:** At $\xi/\xi_{\text{EP}} = 1.00$, $A_{\text{eq}} \in \{0.032, 0.060, 0.072\}$ all decay (`False`); at under-damped $\xi/\xi_{\text{EP}} = 0.25$, $A_{\text{eq}} = 0.032$ decays (`False`), while $A_{\text{eq}} = 0.060$ and $0.072$ creep upward (`True`, slow marginal-manifold regrowth), consistent with Peer Review #3's observation that long-horizon triggering near the marginal manifold can be non-monotonic in $A_{\text{eq}}$ at $c_{\text{nl}} = 0$.

![Long-Horizon 2D 80/20 Cavity Ring Nonlinear Bomb Survival Map](../results/exp006_2d_nonlinear_cavity_bombs.png)

Full metrics: [`../results/exp006_metrics.json`](../results/exp006_metrics.json).

## 4. Verdict & Dead Ends

- **Horizon artifacts caught & resolved across Peer Reviews #2 and #3:**
  - Short horizons ($t = 130\text{--}180$) missed slow regrowth on the $|\alpha_{\min}| \approx 0.0018$ ($\tau \approx 550$) manifold, previously producing a false "`NO TRIG <= 0.095`" claim for wide-mouth cavities. Extending the dense amplitude scan to **$t = 900$** (with $t = 1000$ linear verification and strict non-decaying trigger logic) reveals the true wide-mouth onset bracket **$(0.032, 0.045]$** for $\xi_H \le 1.0\,\xi_H^{\text{EP}}$.
- **What broke / failure modes verified at $t = 900$:**
  - **Pure conservative baffles fail combined pressure + velocity driving:** At $n_p = 0.038, n_v = 0.025$, $N_b = 5, L_b = 0.60$ baffles remain linearly unstable ($\alpha = +0.00631$).
  - **Tight cavity necks ($\sigma_o = 1.5\%$) collapse 2D bomb survival:** Quadratic orifice drag drops the $t = 900$ trigger onset down to **$(0.010, 0.015]$** across all $\xi_H / \xi_H^{\text{EP}}$.
  - **Over-damping above the 2D EP ($\xi_H = 2.0\,\xi_H^{\text{EP}}$) halves wide-mouth bomb survival:** Even with a wide mouth ($\sigma_o = 3.5\%$), over-damping drops the onset bracket from $(0.032, 0.045]$ down to **$(0.015, 0.022]$** ($\sim 2\times$ amplitude loss / $\sim 4\times$ bomb-energy loss).
- **What worked (long-horizon verified 2D nonlinear design laws):**
  1. **Standing bombs are the worst-case 2D subcritical test:** Space-time quartic averaging predicts $E_{\text{crit}}(\text{spin})/E_{\text{crit}}(\text{stand}) = 1.50$, verified at $1.47\text{--}1.49$ ($1.52$ at $t = 360$, within $0.7\text{--}2.3\%$ of cubic $O(2)$ theory). Triggered subcritical standing bombs stay predominantly standing ($S: 0.17 \to 0.16$ over $t = 500$), whereas deeper supercritical saturation ($n_p = 0.068$) amplifies seeded chirality into a spinning wave ($S: 0.17 \to 0.88$).
  2. **Wide-mouth ($\sigma_o = 3.5\%$), at-or-below-EP ($\xi_H \le 1.0\,\xi_H^{\text{EP}}$) $80/20$ rim cavities maximize long-horizon 2D bomb survival:** Widening mouth area from $1.5\%$ to $3.5\%$ ($2.33\times$) raises the $t = 900$ trigger threshold from $(0.010, 0.015]$ to **$(0.032, 0.045]$** ($\sim 2\text{--}4\times$ amplitude gain / $\sim 4\text{--}16\times$ bomb-energy gain, $\sim 3\times / 9\times$ midpoint), while avoiding the $2\times$ over-damping collapse at $\xi_H = 2.0\,\xi_H^{\text{EP}}$.
- **Track Milestone:** Entries `000`–`006` complete the peer-reviewed arc of `rocket-combustion/`. See [`../SYNTHESIS.md`](../SYNTHESIS.md) for the consolidated engineering design laws, verified scope, documented dead ends, and open problems.
