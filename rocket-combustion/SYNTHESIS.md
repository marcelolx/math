# Synthesis — High-Frequency Rocket Combustion Instability (`000`–`006`)

This monograph synthesizes seven verified open-notebook entries (`000`–`006`), audited across **two independent cross-model peer reviews** (Claude Code on `000`–`003`, Claude Opus on `004`–`006`), building from a 1D Crocco $n\text{--}\tau$ baseline up to a 2D nonlinear cylindrical-annulus chamber with radial baffles, dual-tuned faceplate Helmholtz cavity rings, and additive-manufactured azimuthal injector lag staggering.

Every law below is backed by an exact analytical derivation or dispersion root **cross-checked directly against time-domain PDE integration** with verified discrete energy-balance residuals ($10^{-8}\text{--}10^{-5}$), real-axis / $k = 0$ DC stability checks, and grid/horizon convergence checks.

---

## 1. Consolidated Verified Design Laws

| # | Design Law / Structural Theorem | Quantitative Rule (Verified in PDE) | Source Entry |
| :-: | :--- | :--- | :---: |
| **1** | **DC-Blocking Requirement ($k = 0$ Bulk Mode)** | An unfiltered Crocco pressure-coupled law has a positive real $k = 0$ root for any $n > \zeta$ ($F(0) = \zeta - n < 0$). High-pass filtering $H_{\text{hp}}(s) = s/(s + \omega_{\text{hp}})$ eliminates static bulk runaway for all physical $n \lesssim 0.50$. | [`000`](entries/000-baseline.md), [`001`](entries/001-spatial-vs-timelag.md) |
| **2** | **Spatial Smearing Ceiling vs. Causal Lag Spread** | Local spatial flame thickness $w_f$ cannot stabilize a 1D combustor above $n_c = 2\operatorname{arctanh}(\zeta) = 0.08004$ ($0\%$ gain above $n_c$). Causal Erlang-4 time-lag spread $\sigma_\tau$ damps high harmonics exponentially ($|\hat{g}(i\omega)| = (1 + (\omega\sigma_\tau/2)^2)^{-2}$), cutting $2L\text{--}12L$ instability sharply while leaving $1L$ exposed. | [`001`](entries/001-spatial-vs-timelag.md) |
| **3** | **1D Exceptional-Point (EP) Cavity + Lag Synergy** | A 1D faceplate Helmholtz cavity ($\beta = V_c / V_{\text{ch}}$) maximizes $1L$ damping near $\xi_H^{\text{EP, }1L} \approx \sqrt{2\beta} - \alpha_1/\pi$ (optimal $\xi_H \approx 0.19$ for $\beta = 2\%$). Pairing a $2\%$ $1L$ cavity with $\sigma_\tau = 0.25$ raises full-domain ($1L\text{--}12L$) stability from **$19.2\%$ to $89.7\%$**. | [`002`](entries/002-helmholtz-cavity.md) |
| **4** | **Transverse Half-Damping Theorem** | In a 2D cylindrical annulus, pure tangential modes ($1T, 2T$) suffer **exactly half** the cold nozzle damping of longitudinal modes ($\alpha_{mT}/\alpha_{kL} = \frac{1}{2}$, verified $0.4997$ in 2D PDE) because their axial pressure energy is uniform rather than peaking $2\times$ at the nozzle wall. | [`004`](entries/004-2d-transverse-baffles.md) |
| **5** | **Radial Baffle Length & Symmetry Blind-Spot Rules** | (a) Short baffles terminating upstream of the flame ($L_b < x_f$) **amplify** chiral spinning $1T$ ($+0.00466 \to +0.00561$, verified on 3 grids); stabilization requires $L_b \gtrsim 2.0\,x_f$. (b) Equally spaced radial blades leave any standing $mT$ mode completely unperturbed ($0.00\%$ detuning) whenever $N_b \mid 2m$ ($N_b = 2 \to 1T/2T/3T$; $N_b = 4 \to 2T$; $N_b = 6 \to 3T$). Prime **$N_b = 5$** has no blind spot below $5T$. | [`004`](entries/004-2d-transverse-baffles.md) |
| **6** | **2D Inverse-Square Dual-Tuned Cavity Allocation** | Peak tangential cavity damping scales as $\Delta\alpha_{mT} \propto -\omega_m\sqrt{\beta_m}$ with 2D EP $\xi_H^{\text{EP, }1T} \approx \sqrt{\beta_{\text{tot}}} - \alpha_{1T}$. Equalizing $1T$ and $2T$ margins under a fixed $2\%$ volume budget requires $f_m \propto 1/\omega_m^2 \implies \mathbf{80\%\,1T / 20\%\,2T}$ (exact max-min optimum $81\text{--}82\%$), **$2.1\times$ stronger** than a naive $50/50$ split and **$5.4\times$ stronger** on $1T$ than conservative baffles. | [`005`](entries/005-baffle-free-2d.md) |
| **7** | **Odd-Periodic Zero-Hardware Azimuthal Lag Staggering** | Alternating injector lags $\tau(\theta) = \bar{\tau} \pm \Delta\tau$ attenuates non-resonant modes by $\cos(\omega\Delta\tau)$, stabilizing pressure-coupled $1T$ for $\Delta\tau \ge 0.75$ with **zero cavities or baffles** when $M \in \{3, 5\}$ is odd—whereas $M = 2$ triggers a Bragg-orientation trap that worsens orthogonal standing $1T$ by **$+67\%$**. | [`005`](entries/005-baffle-free-2d.md) |
| **8** | **2D Standing-Bomb Energy Advantage ($1.50\times$)** | Under subcritical cubic driving, standing $1T$ bombs concentrate space-time quartic Rayleigh energy by $\frac{9/16}{3/8} = \mathbf{1.50\times}$ over equal-energy spinning bombs (verified $1.47\text{--}1.49$ at $t=180$, $1.52$ at $t=360$ in 2D PDE, within $0.7\text{--}2.3\%$ of cubic $O(2)$ theory). Near the subcritical fold, triggered standing waves stay standing over long times; strong spin-up ($S: 0.17 \to 0.88$, seeded $\epsilon_s=0.12$) emerges deeper in the saturating supercritical regime. | [`006`](entries/006-2d-nonlinear-modes.md) |
| **9** | **Nonlinear Cavity Orifice-Drag Cliff (1D & 2D)** | Quadratic orifice jet drag ($c_{\text{nl}} \propto 1/\sigma_o^2$) penalizes both tight necks and over-damping above the EP: in 1D (`003`), over-damping above $\xi_H^{\text{EP}}$ drops $P_{\text{crit}}$ by $-54\%$; in long-horizon ($t = 900$) 2D scans (`006`), widening the $80/20$ cavity mouth from $\sigma_o = 1.5\%$ to $3.5\%$ raises the $1T$ trigger onset from $(0.010, 0.015]$ to **$(0.032, 0.045]$** ($\sim 2\text{--}4\times$ amplitude / $\sim 4\text{--}16\times$ bomb-energy gain) for $\xi_H \le 1.0\,\xi_H^{\text{EP}}$, whereas over-damping to $\xi_H = 2.0\,\xi_H^{\text{EP}}$ halves the wide-mouth margin down to $(0.015, 0.022]$. | [`003`](entries/003-nonlinear-triggering.md), [`006`](entries/006-2d-nonlinear-modes.md) |

---

## 2. Documented Dead Ends & Traps (What Broke)

1. **Unfiltered pressure-coupled Crocco law (Peer Review #1 C1):** Produces an unphysical static $k = 0$ bulk pressure runaway for every $n > \zeta$. Caught by real-axis scanning and fixed uniformly with $H_{\text{hp}}(s)$.
2. **Sign-reversing quintic flame laws (Peer Review #1 C2):** Polynomial subcritical flames $q + c_3 q^3 - c_5 q^5$ reverse sign at finite amplitude and drive $p'/\bar{p} \sim 0.55$, invalidating linear bulk acoustics. Replaced with strictly monotonic bounded $\Phi(q) = q_s\tanh(q/q_s + \kappa(q/q_s)^3)$ keeping $p'/\bar{p} \le 0.09$ in 1D and $\le 0.06$ on subcritical 2D branches.
3. **Under-damped cavities ($\xi_H \ll \xi_H^{\text{EP}}$):** Split the chamber mode into two weakly damped peaks that remain linearly unstable even at exact tuning (confirmed in 1D `002` and 2D `005`).
4. **Over-damped or tight-neck cavities ($\xi_H > \xi_H^{\text{EP}}$ or $\sigma_o = 1.5\%$):** Choke the cavity neck during finite-amplitude pressure pulses, collapsing bomb-survival thresholds despite strong linear damping (`003`, `006`).
5. **Short radial baffles ($L_b < x_f$):** Amplify rather than damp chiral spinning $1T$ (`004`).
6. **Conservative baffles under combined pressure + velocity driving:** Cannot dissipate pressure-coupled Rayleigh energy and can even worsen linear growth via phase detuning (`006` Exp 2: $+0.0166$ vs. bare $+0.0049$).
7. **$M = 2$ azimuthal lag staggering:** Bragg-resonates with $1T$, destabilizing the orthogonal standing orientation by $+67\%$ (`005`).
8. **Short-horizon bomb bisection without slow-mode time-scale matching (Peer Reviews #2 C1 & #3 C3):** Horizons of $t = 130\text{--}180$ were shorter than the $80/20$ cavity ring's least-damped linear time scale ($\tau = 1/|\alpha_{\min}| \approx 550$), hiding slow regrowth of large wide-mouth bombs ($A_{\text{eq}} \ge 0.045$) and non-monotonic behavior near the marginal manifold. Fixed in `006` with $t = 1000$ broadband linear verification and a $t = 900$ dense amplitude scan (no monotonicity assumption) with strictly non-decaying trigger criteria.

---

## 3. Verified Scope vs. Open Problems

### What Is Verified Within This Model
- Linearized 1D and 2D thin-annulus bulk acoustics (zero mean Mach advection, no radial $r$-modes or gasdynamic shock steepening).
- Exact dispersion root-finding cross-checked against staggered-grid RK4 PDEs with discrete energy conservation checks.
- Chiral spinning $1T$ conclusions (`004`–`006`) explicitly assume **signed linear azimuthal velocity coupling** (swirled/directional injectors); non-swirled injectors with even $|U_\theta|$ or $U_\theta^2$ response have no linear chiral drive.

### High-Value Open Problems for Future Entries (`007+`)
1. **Compressible 2D Nonlinear Euler Transverse Acoustics:** Capture finite-amplitude spinning-wave steepening (rotating detonation-like shock fronts) at $p'/\bar{p} \gtrsim 0.10$, where bulk-linear acoustics breaks down.
2. **Full 3D Cylindrical $(r, \theta, x)$ Chamber with Radial ($1R$) Modes & Baffle-Tip Vortex-Shedding Loss:** Couple conservative geometric blockage with nonlinear baffle-tip acoustic dissipation.
3. **Even-Symmetry ($|U_\theta|$) Velocity Coupling & Noise-Driven Azimuthal Mode Switching:** Test stochastic switching between standing and spinning $1T$ in symmetric non-swirled chambers (following Noiray & Schuermans).
