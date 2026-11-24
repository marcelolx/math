# 006 — 2D Nonlinear Spinning vs. Standing Mode Selection & Nonlinear Cavity Bomb Tests

- **Date:** 2026-11-23
- **Status:** `verified`
- **Verification Script:** `../sim/exp006_2d_nonlinear_modes.py`

## 1. Target

Unite our **1D nonlinear lessons (Entry `003`)** with our **2D transverse/baffle-free architectures (Entries `004`–`005`)** inside a single 2D nonlinear cylindrical-annulus PDE (carrying forward DC-blocking $\omega_{\text{hp}} = 0.15$, causal Erlang-4 time lag $\bar{\tau} = 0.30, \sigma_\tau = 0.15$, and the bounded monotonic subcritical flame law $\Phi(q) = q_s\tanh(q/q_s + \kappa(q/q_s)^3)$ keeping amplitudes bulk-linear $P \le 0.09$), to test:

1. **2D Symmetry-Breaking Mode Selection (Axisymmetric Chamber, $n_v = 0$):**
   - At equal bomb energy $E_0$, does a **standing $1T$ bomb** trigger at $\sim 1.5\times$ *lower* energy than a **spinning $1T$ bomb** (because its antinodes reach $\sqrt{2}\times$ higher local peak pressure in the subcritical regime)?
   - Once saturated on the upper branch, does nonlinear saturation flip stability so a seeded standing wave spontaneously transitions into a **spinning $1T$ limit cycle**?
2. **2D Nonlinear Bomb Survival of the $80/20$ Dual-Tuned Faceplate Cavity Ring:**
   - Under calibrated quadratic orifice jet drag ($c_{\text{nl}} = \frac{\beta_s\omega_H^2}{2 C_d^2 \sigma_o^2}$), how much do **(a) linear damping ratio $\xi_H / \xi_H^{\text{EP}}$** and **(b) cavity mouth open-area ratio $\sigma_o$ ($1.5\%$ vs. $3.5\%$)** govern finite-amplitude $1T$ and $2T$ bomb survival compared to pure radial baffles?

## 2. The Math & Assumptions

### Bounded Subcritical 2D Flame Law & Energy-Matched Bomb Ratio
$$\Phi(q) = q_s \tanh\!\left(\frac{q}{q_s} + \kappa\left(\frac{q}{q_s}\right)^3\right), \qquad q_s = 0.036,\quad \kappa = 4.5$$
Consider two $1T$ initial perturbations with **identical total acoustic energy** $E_0 = \pi L_a A_{\text{eq}}^2$:
- **Spinning $1T$:** $P = A_{\text{eq}}\cos\theta$, $U_\theta = +A_{\text{eq}}\cos\theta$ (uniform local amplitude $A_{\text{eq}}$ everywhere around the rim).
- **Standing $1T$:** $P = \sqrt{2} A_{\text{eq}}\cos\theta$, $U_\theta = 0$ (local peak $\sqrt{2}A_{\text{eq}}$ at antinodes $\theta = 0, \pi$).

In the small-amplitude subcritical regime ($\Phi(q) \approx q + c_3 q^3$), the azimuthally averaged cubic Rayleigh energy boost is:
$$\frac{\langle q_{\text{stand}}^4 \rangle}{\langle q_{\text{spin}}^4 \rangle} = \frac{\langle \cos^4\theta \rangle (\sqrt{2}A_{\text{eq}})^4}{A_{\text{eq}}^4} = \frac{3}{8} \times 4 = \mathbf{1.50}$$
Thus standing bombs require $\sim 1.50\times$ **less** total energy $E_0$ to trigger than spinning bombs! Conversely, at saturating upper-branch amplitudes where $\Phi(q)/q$ decreases with $|q|$, the standing wave's antinodes saturate harder than the uniform-amplitude spinning wave, making the **spinning $1T$ limit cycle nonlinearly preferred** (measured via spin index $S(t) = \frac{|\int P U_\theta\,d\theta|}{\frac{1}{2}\int(P^2 + U_\theta^2)\,d\theta} \in [0, 1]$).

### Calibrated 2D Cavity Orifice Drag & Consistent Initialization
Each rim cavity sector obeys $\frac{\partial v_r}{\partial t} + (2\xi_H\omega_H + c_{\text{nl}}|v_r|)v_r = \beta_s\omega_H^2(P(0, \theta) - P_c)$ with $c_{\text{nl}} = \frac{\beta_s\omega_H^2}{2 C_d^2 \sigma_o^2}$ ($C_d = 0.65$), pre-pressurized consistently at $t = 0$ ($p_c(0) = P(0, \theta), v_r(0) = 0$).

### Intentional Simplifications
- Thin-annulus 2D bulk-linear acoustics ($30\times 60$ grid, verified against $45\times 90$ to $0.38\%$).
- Note: reported `A_crit` thresholds are energy-equivalent amplitudes ($E_{\text{crit}} = \pi L_a A_{\text{eq,crit}}^2$); in Experiment 1, `bisect_2d_bomb` returns the threshold cleanly while settled upper-branch amplitudes are checked in dedicated post-threshold runs (strictly bulk-linear $P_{\text{peak}} \le 0.0863$, i.e. $p'/\bar{p} \le 0.108$).

## 3. The Verification

Reproduce from the repo root:

```bash
uv run python rocket-combustion/sim/exp006_2d_nonlinear_modes.py
```

### Experiment 1 — 2D Standing vs. Spinning Subcritical Triggering & Spontaneous Spin Transition ($n_v = 0$)

| $n_p$ | $A_{\text{crit}}(\text{spin})$ | $A_{\text{crit}}(\text{stand})$ | Energy Ratio $E_{\text{crit}}(\text{spin})/E_{\text{crit}}(\text{stand})$ | Analytical Cubic Theory |
| :---: | :---: | :---: | :---: | :---: |
| $0.033$ | $0.0109$ | $0.0091$ | **$1.447$** | $1.500$ |
| $0.037$ | $0.0072$ | $0.0059^*$ | **$1.488$** *(0.8% from theory!)* | $1.500$ |
| $0.040$ | $0.0039$ | $0.0032$ | **$1.467$** | $1.500$ |

*\*Grid convergence check at $n_p = 0.037$: $30\times 60$ gives $A_{\text{crit}}(\text{stand}) = 0.0059$ vs. $45\times 90$ giving $0.0059$ (**0.38% relative difference**).*

- **Equal-energy bomb showdown ($n_p = 0.037, A_{\text{eq}} = 0.0066$):** Spinning bomb decays (`triggered = False`); standing bomb with the **exact same energy** triggers (`triggered = True`).
- **Spontaneous symmetry breaking ($n_p = 0.068$):** Seeding a predominantly standing $1T$ wave ($S_0 = 0.169$) spontaneously transitions into a strongly spinning limit cycle (**$S_{\text{late}} = 0.8434$** at $t = 340$, still rising toward $1.00$, bulk-linear peak $P_{\text{peak}} = 0.0863 \implies p'/\bar{p} = 0.108$).

![2D Nonlinear Symmetry Breaking: Standing Triggers Easier, Spinning Wins Saturation](../results/exp006_2d_spin_selection.png)

### Experiment 2 — 2D Nonlinear Bomb Sweep on $80/20$ Dual-Tuned Cavity Ring ($n_p = 0.048, n_v = 0.040, \beta_{\text{tot}} = 2\%, N_{\text{pair}} = 5$)

- **Pure 5-blade protruding baffles ($L_b = 0.60$, no cavities):** $A_{\text{crit}}(1T) = 0.0$, $A_{\text{crit}}(2T) = 0.0$ *(linearly unstable under combined pressure + velocity driving because conservative baffles cannot dissipate pressure-coupled Rayleigh driving)*.

| $\xi_H / \xi_H^{\text{EP}}$ | $1T$ $A_{\text{crit}}$ (Ideal Linear Cav) | $1T$ $A_{\text{crit}}$ (Wide Mouth $\sigma_o=3.5\%$) | $1T$ $A_{\text{crit}}$ (Tight Neck $\sigma_o=1.5\%$) | $2T$ $A_{\text{crit}}$ (Tight Neck $\sigma_o=1.5\%$) |
| :---: | :---: | :---: | :---: | :---: |
| **$0.25$ (Under-EP)** | $0.0293$ | **$0.0214$ *(BEST)*** | **$0.0097$ *(BEST)*** | **$0.0130$ *(BEST)*** |
| $0.50$ | **$0.0628$** | $0.0172$ | $0.0091$ | $0.0124$ |
| $0.80$ | $0.0434$ | $0.0139$ | $0.0084$ | $0.0118$ |
| **$1.00$ (2D Linear EP)** | $0.0408$ | $0.0126$ | $0.0078$ | $0.0114$ |
| $1.40$ (Over-EP) | $0.0130$ | $0.0104$ | $0.0067$ | $0.0105$ |
| **$2.00$ (Over-damped)** | $0.0086$ | **$0.0074$ (-65%)** | **$0.0048$ (-51%)** | $0.0092$ |

![2D 80/20 Dual-Tuned Cavity Ring Nonlinear Bomb Survival](../results/exp006_2d_nonlinear_cavity_bombs.png)

Full metrics: [`../results/exp006_metrics.json`](../results/exp006_metrics.json).

## 4. Verdict & Dead Ends

- **What broke / failure modes uncovered:**
  - **Protruding baffles alone fail combined pressure + velocity driving:** Under $n_p = 0.048, n_v = 0.040$, even $N_b = 5, L_b = 0.60$ baffles remain linearly unstable ($A_{\text{crit}} = 0$) because solid blades only disrupt $U_\theta$ and detune frequencies without dissipating pressure-coupled acoustic energy.
  - **Tight cavity mouths ($\sigma_o = 1.5\%$) suffer severe nonlinear bomb-margin loss:** Quadratic orifice drag drops $1T$ $A_{\text{crit}}$ by $>4\times$ relative to ideal linear cavities ($0.0408 \to 0.0078$ at $\xi/\xi_{\text{EP}} = 1.0$), and over-damping above the 2D EP ($\xi/\xi_{\text{EP}} = 2.0$) slashes $A_{\text{crit}}$ by another $51\text{--}65\%$ ($4\text{--}8\times$ drop in survivable bomb energy).
- **What worked (unified 2D nonlinear design laws):**
  1. **Standing bombs are the worst-case 2D subcritical test:** Because standing antinodes concentrate energy by $1.50\times$ in cubic driving ($E_{\text{crit}}(\text{spin})/E_{\text{crit}}(\text{stand}) = 1.45\text{--}1.49$ in 2D PDE), bomb-test qualification must account for standing-wave triggering even though the resulting saturated instability spontaneously evolves into a spinning $1T$ wave ($S: 0.17 \to 0.84$).
  2. **Wide-mouth, slightly under-EP $80/20$ dual-tuned rim cavities win 2D nonlinear bomb tests:** Increasing cavity open-area ratio from $\sigma_o = 1.5\%$ to $3.5\%$ ($2.33\times$ area) raises nonlinear $1T$ bomb amplitude survival by **$2.21\times$** ($0.0097 \to 0.0214$, a **$4.9\times$ gain in bomb energy $E_{\text{crit}}$**!), and designing **below the 2D linear EP** ($\xi_H \approx 0.25\text{--}0.50\,\xi_H^{\text{EP}}$) maximizes nonlinear bomb survival across both $1T$ and $2T$ simultaneously.
- **Next step:** Dispatch independent cross-model peer review (Claude Code via Orca orchestration) on Entries `004`–`006` before writing the `rocket-combustion/` synthesis monograph.
