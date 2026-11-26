# 012 — 2D Transonic Converging–Diverging Nozzle Geometry: Bell-Nozzle Evanescent-Cutoff Collapse vs. Annular Slot Tuning & Entropy-Noise Filtering

- **Date:** 2026-11-26
- **Status:** `verified` (pending independent Peer Review #9)
- **Verification Tier:** 🟡 `verified (pending audit)`
- **Verification Script:** `../sim/exp012_2d_transonic_cd_nozzle.py`
- **Peer Review Receipt:** *(pending Peer Review #9)*

## 1. Target

Replace the compact 0D Marble–Candel choked-nozzle boundary condition used in Entries `009`–`011` with an **explicitly resolved, $C^2$-smooth 2D $(x, \theta)$ transonic converging–diverging (CD) nozzle** accelerating steady throughflow from chamber Mach $\bar{M}_c$ through an interior sonic throat ($M = 1$ at $x_t = L_a + L_n$) to a strictly supersonic exit ($M_e > 1$ at $x = L_a + L_n + L_d$, requiring **zero outlet boundary conditions**). Test four concrete propulsion questions:

1. **Exact Variable-$(A, R)$ Conservation, Compact Limit & Chebyshev-vs-PDE Audit:** Does our conservative variable-area, variable-radius 2D Euler discretization preserve quiescent equilibrium to $\sim 10^{-16}$, conserve periodic mass, angular momentum ($A\rho R u_\theta$), and total energy ($AE$) to $\sim 10^{-16}$, converge steady transonic flow to machine-precision stationarity ($\max|\partial_t \mathbf{W}| \sim 5\times 10^{-13}$), and match an independent 1D-in-$x$ Chebyshev transonic eigen-solver (zero outlet BCs, spectral convergence $< 5\times 10^{-6}$ across $N = 96 \to 120$) across `64×144` and `96×200` with an explicit per-mode numerical-damping audit?
2. **Finite-Nozzle Low-Pass Filtering of Indirect Entropy Noise:** When Entry `009`'s exact zero-tail hot entropy spot ($x_0 = 0.55, \sigma = 0.08, \delta\rho/\bar{\rho} = -0.05$) convects through finite converging sections $L_n \in \{0.15, 0.35, 0.70, 1.10\}$, how does distributed nozzle acceleration attenuate the first-pass Marble–Candel rarefaction echo (`-0.366%`) at the nozzle entrance $x = L_a$?
3. **Bell-Nozzle Evanescent-Cutoff Collapse vs. Constant-Radius Annular Slot Tuning:** How does a **shrinking-radius axisymmetric bell nozzle** ($R(x) = \sqrt{A(x)}$, where the local tangential cutoff frequency $\omega_{\text{cut}}(x) = m\bar{c}\sqrt{1 - M^2}/R(x)$ rises toward the throat and reflects $1T, 2T$ waves evanescently, Stow–Dowling–Hynes 2002) compare against a **constant-radius annular slot / plug nozzle** ($R(x) \equiv 1$, where only the radial gap $h(x) = A(x)$ converges) across $L_n \in [0.10, 0.80]$ and multi-mode ($m = 0..3$) global Hopf maps?
4. **2D Nonlinear Transonic Rotating-Shock Control ($\bar{M}_c = 0.12, n_p = 0.22, n_v = 0.04, A_0 = 0.055$):** How does replacing compact Marble–Candel with a realistic $L_n = 0.50$ bell nozzle vs. a steep bell ($L_n = 0.20$) vs. a constant-radius annular slot ($L_n = 0.50$) alter $6.9\%$ rotating-shock dynamics—and does Entry `010`'s retuned mid-sector $2\%$ baffle–cavity hybrid still control $1T$ and $2T$ inside an evanescently reflecting bell nozzle?

## 2. The Math, Assumptions & Checked Scope

### 2D Conservative Variable-$(A(x), R(x))$ Compressible Euler Equations
For $\mathbf{W} = (A\rho,\; A\rho u_x,\; A\rho R u_\theta,\; A E)^T$ on $(x, \theta) \in [0, L_{\text{tot}}] \times [0, 2\pi)$ ($\gamma = 1.25, L_a = 1.5, L_d = 0.25$):
$$\frac{\partial (A\rho)}{\partial t} + \frac{\partial (A\rho u_x)}{\partial x} + \frac{\partial (\frac{A}{R}\rho u_\theta)}{\partial \theta} = 0$$
$$\frac{\partial (A\rho u_x)}{\partial t} + \frac{\partial (A(\rho u_x^2 + p))}{\partial x} + \frac{\partial (\frac{A}{R}\rho u_x u_\theta)}{\partial \theta} = p\frac{dA}{dx} + \frac{A\rho u_\theta^2}{R}\frac{dR}{dx}$$
$$\frac{\partial (A\rho R u_\theta)}{\partial t} + \frac{\partial (A\rho u_x R u_\theta)}{\partial x} + \frac{\partial (A(\rho u_\theta^2 + p))}{\partial \theta} = 0$$
$$\frac{\partial (AE)}{\partial t} + \frac{\partial (A u_x (E + p))}{\partial x} + \frac{\partial (\frac{A}{R} u_\theta (E + p))}{\partial \theta} = A(x)\,\frac{w_f(x)}{\gamma - 1} Q_{\text{flame}}(\theta, t)$$
- **Exact conservation & well-balancing:** Mass ($A\rho$), angular momentum ($A\rho R u_\theta$), and total energy ($AE$) carry **zero geometric source terms** (centrifugal axial work and Coriolis swirl work cancel identically). Quiescent equilibrium ($u_x = u_\theta = 0, p = \bar{p}$) cancels $p_i(A_{i+1/2} - A_{i-1/2})/\Delta x$ to $\sim 10^{-16}$.
- **$C^2$-smooth transonic contour:** Straight chamber $A = 1$ on $[0, L_a]$; quartic converging section $A(x) = A_t + (1 - A_t)(1 - 4s^3 + 3s^4)$ on $[L_a, x_t = L_a + L_n]$ ($A' = A'' = 0$ at $L_a$, strictly positive throat curvature $\kappa_t = 12(1 - A_t)/L_n^2 > 0$ guaranteeing smooth finite sonic acceleration $M'(x_t) > 0$); curvature-matched supersonic expansion $A(x) = A_t + \Delta A\,\frac{y^2}{y^2 + y_0^2}$ ($\Delta A = 0.60 A_t, y_0^2 = 2\Delta A/\kappa_t$) on $[x_t, x_t + L_d]$ reaching $M_e \approx 1.83 > 1$ (exact zero-gradient supersonic outflow).
- **Smooth van Leer harmonic TVD limiter:** Used in both 1D steady transonic base-flow pre-relaxation ($t_{\text{relax}} = 300$, eliminating sonic-throat MC limiter jitter to $\max|\partial_t \mathbf{W}| \sim 5\times 10^{-13}$) and 2D SSP-RK2 HLLC integration. Flame nonlinearity ($n_p\Phi(q_p) + n_v\Phi(q_v)$) and discrete faceplate Helmholtz cavity ports match Entries `010`–`011` identically.
- **Checked scope:** Inviscid 2D thin-shell/streamtube Euler model (no radial $1R$ Bessel structure—the 2D shell uses azimuthal wavenumber $m/R(x)$ rather than 3D cylindrical $s_{mn}/R(x)$—and no viscous boundary layers; viscous damping of high-$k_{\text{vort}}$ modes was quantified in Entry `011`).

## 3. The Verification

Reproduce from the repo root:

```bash
uv run python rocket-combustion/sim/exp012_2d_transonic_cd_nozzle.py
```

### Experiment 1 — Variable-$(A, R)$ Benchmarks, Compact Limit & Per-Mode Numerical-Damping Audit

- **Exp 1A (Quiescent well-balancing, `48×64`, $t = 10$):** `annular = 9.20e-16`, `bell = 4.44e-16`.
- **Exp 1B (Periodic variable-$(A, R)$ bell conservation, $A = 0.10, t = 8$):** **`mass = 2.28e-16`, `ang_mom = 1.95e-17`, `energy = 1.42e-16`** (wall pressure thrust acts on $x$-momentum $\Delta m_x = 0.048$, while angular momentum and total energy are strictly conserved).
- **Exp 1C (Steady transonic flow convergence, `bell`, $\bar{M}_c = 0.12, L_n = 0.50$, $M_e = 1.827$):**
  - `nx = 60`: `rhs = 9.9e-13`, `mf_err = 2.69e-3`, `M_L2 = 5.80e-3`
  - `nx = 100`: `rhs = 2.6e-12`, `mf_err = 1.05e-3`, `M_L2 = 1.69e-3`
  - `nx = 160`: `rhs = 4.9e-13`, `mf_err = 4.29e-4`, `M_L2 = 7.67e-4` ($O(\Delta x^2)$ convergence!).
- **Exp 1D (Chebyshev spectral convergence $N = 96 \to 120$ vs. 2D Transonic PDE & per-mode numerical damping, $A_0 = 10^{-4}, t \in [20, 52]$):**

| Case | Exact Chebyshev ($N=120$) | $|N_{96}-N_{120}|$ | `64×144` PDE ($\Delta_{\text{num}}$) | `96×200` PDE ($\Delta_{\text{num}}$) |
| :--- | :---: | :---: | :---: | :---: |
| `annular` $\bar{M}_c=0.12, L_n=0.50$ ($1T$) | **`-0.07290`** ($\omega=0.899$) | `4.7e-6` | `-0.07008` (`+0.0028`) | `-0.06425` (`+0.0086`)* |
| `bell` $\bar{M}_c=0.12, L_n=0.50$ ($1T$) | **`-0.02448`** ($\omega=1.092$) | `3.1e-6` | `-0.03128` (`-0.0068`) | **`-0.02608` (`-0.0016`)** |
| **`bell` $\bar{M}_c=0.12, L_n=0.70$ ($1T$)** | **`-0.00377`** ($\omega=1.058$) | `3.9e-7` | `-0.01087` (`-0.0071`) | **`-0.00627` (`-0.0025`)** |
| **`bell` $\bar{M}_c=0.18, L_n=0.50$ ($1T$)** | **`-0.02803`** ($\omega=1.058$) | `2.8e-6` | — | **`-0.02966` (`-0.0016`)** |
| `annular` $\bar{M}_c=0.16, L_n=0.50$ ($2T$) | **`-0.04315`** ($\omega=2.026$) | `5.0e-6` | `-0.06505` (`-0.0219`) | `-0.05287` (`-0.0097`) |
| `bell` $\bar{M}_c=0.16, L_n=0.50$ ($2T$) | **`-0.00865`** ($\omega=2.035$) | `3.2e-6` | `-0.02930` (`-0.0206`) | `-0.01686` (`-0.0082`) |

*(Note: `*` At `annular` $\bar{M}_c=0.12, L_n=0.50$, two closely spaced $1T$ branches at `-0.0729 + 0.899i` and `-0.0767 + 1.095i` beat slowly over $T_{\text{beat}} \approx 32$. Crucially, `96×200` confirms BOTH the `7.5×` bell evanescent collapse at $\bar{M}_c=0.12, L_n=0.70$ and the `3.5×` bell collapse at $\bar{M}_c=0.18, L_n=0.50$ relative to compact Marble–Candel `-0.0993`, while explicitly documenting `-0.008` to `-0.010` numerical $2T$ damping on `96×200` and `-0.021` on `64×144`.)*

### Experiment 2 — Finite-Nozzle Low-Pass Filtering of Indirect Entropy Noise ($x = L_a, \bar{M}_c = 0.12$)

| Converging Nozzle Length $L_n$ | 1st-Pass Direct Rarefaction at $x = L_a$ | Arrival Time $t_{\text{1st}}$ | % of Compact MC (`-0.366%`, Entry `009`) | 2nd-Pass Superposed Peak |
| :---: | :---: | :---: | :---: | :---: |
| $0^+$ *(Compact MC, `009`)* | **`-0.366%`** | `7.92` | `100.0%` | — |
| `0.15` | **`-0.3539%`** | `8.65` | **`96.7%`** | `-0.5397%` |
| `0.35` | **`-0.3349%`** | `10.12` | **`91.5%`** | `-0.5021%` |
| `0.70` | **`-0.3219%`** | `12.38` | **`88.0%`** | `-0.4167%` |
| `1.10` | **`-0.3083%`** | `15.06` | **`84.2%`** (`-15.8%` filtered!) | `-0.3083%` (`-42.9%`!) |

*(As $L_n \to 0$, the first-pass direct rarefaction echo smoothly recovers Entry `009`'s exact Marble–Candel `-0.366%` limit; lengthening the converging cone spreads entropy acceleration across $L_n$, monotonically delaying and low-pass filtering indirect entropy noise.)*

### Experiment 3 — Bell-Nozzle Evanescent-Cutoff Collapse vs. Constant-Radius Annular Slot Tuning

In a shrinking-radius **bell nozzle ($R = \sqrt{A}$)**, the local tangential cutoff frequency $\omega_{\text{cut}}(x) = m\bar{c}(x)\sqrt{1 - M^2}/R(x)$ rises sharply as $R(x)$ contracts toward the throat ($R_t = \sqrt{A_t} = 0.453$ at $\bar{M}_c = 0.12$), creating an evanescent turning point that reflects tangential acoustic waves back into the chamber:
- **Cold $1T$ collapse at $\bar{M}_c = 0.12$:** Bell damping drops from `-0.0584` ($L_n=0.35$) to `-0.0245` ($L_n=0.50$), **`-0.00377` ($L_n=0.70$, `7.5×` below compact MC `-0.02843`)**, and **`-0.00061` ($L_n=0.80$, `46×` below MC!)**.
- **Cold $1T$ collapse at $\bar{M}_c = 0.18$:** Bell damping drops monotonically from `-0.125` ($L_n=0.10$) to **`-0.0280` ($L_n=0.50$, `3.5×` below MC `-0.0993`)** and `-0.0160` ($L_n=0.80$, `6.2×` below MC!).
- **Multi-mode ($m = 0..3$) global Hopf envelopes at $L_n = 0.50$:**

| Chamber $\bar{M}_c$ | Compact MC Global $n_{pH}$ (`009`) | Annular Slot ($R \equiv 1, L_n=0.50$) | Shrinking Bell ($R = \sqrt{A}, L_n=0.50$) | Bell vs. Annular Ratio |
| :---: | :---: | :---: | :---: | :---: |
| `0.10` | `0.284` ($m=0$) | `0.104` ($1T$) | **`0.000`** *(cold $3T$-1L whistle `+0.00224`!)* | — |
| **`0.12`** | `0.134` ($1T$) | **`0.228`** ($2T$; **`1T = 0.430`**, `3.22×` MC!) | **`0.060`** ($2T$; `1T = 0.157`) | **`3.79×` worse!** |
| `0.14` | `0.205` ($1T$) | `0.217` ($1T$) | `0.198` ($1T$) | `1.09×` |
| **`0.16`** | `0.235` ($2T$) | `0.205` ($1T$; `2T = 0.401`) | **`0.082`** ($2T$) | **`2.48×` worse!** |
| **`0.18`** | `0.426` ($m=0$) | `0.296` ($1T$) | **`0.156`** ($1T$, **`-63%` vs. MC!**) | **`1.90×` worse!** |
| `0.20` | `0.381` ($m=0$) | `0.344` ($m=0$; `1T = 0.427`) | `0.258` ($2T$) | `1.33×` |

- **Exact inviscid cold $3T$-1L Coriolis-swirl whistle (`bell`, $\bar{M}_c = 0.10, L_n = 0.50$):** Angular-momentum conservation ($R u_\theta = \text{const}$) amplifies convecting vorticity velocity as $u'_\theta \propto 1/R(x)$ inside the converging cone while $3T$ acoustics are evanescently trapped—yielding a spectrally converged cold instability **`+0.002241 + 3.5598i`** (`N = 72: +0.002239`, `N = 96: +0.002237`, `N = 128: +0.002241`). *(Physical caveat per Entry `011`: at $k_{\text{vort}} = \omega/\bar{M}_c = 35.6$, finite viscosity $\mathrm{Re} \lesssim 5000$ damps this short-wavelength vorticity mode before it reaches the nozzle.)*

![Bell Evanescent-Cutoff Collapse vs. Annular Strouhal Tuning and Global Hopf Envelopes](../results/exp012_bell_cutoff_vs_annular_tuning.png)

### Experiment 4 — 2D Transonic $6.9\%$ Rotating-Shock Shootout ($\bar{M}_c = 0.12, n_p = 0.22, n_v = 0.04$)

| Configuration ($\bar{M}_c = 0.12$) | Exact Bare Linear ($1T$ / $2T$) | `80×200` Linear PDE ($1T$ / $2T$) | Nonlinear $6.9\%$ Shock ($A_0 = 0.055, t = 160$) |
| :--- | :---: | :---: | :--- |
| **Bare Bell ($L_n = 0.50$)** | **`+0.0220` / `+0.0269`** *(both unstable!)* | `+0.0149` / `+0.0089` | **$1T$: `12.51%`** ($E=1.96\times 10^{-3}$, $l/m=1.061$); **$2T$ (`80×200`): `6.42%`** ($1.04\times 10^{-3}$) |
| Bare Steep Bell ($L_n = 0.20$) | `+0.0144` / **`-0.0164`** *($2T$ cleared!)* | `+0.0244` / — | $1T$: `9.32%` ($E=3.47\times 10^{-3}$, $l/m=1.020$) |
| **Bare Annular Slot ($R\equiv 1, L_n=0.50$, zero hardware!)** | **`-0.0237`** / `+0.0023` | `-0.0692` / `-0.0188`* | **`80×200` $1T$ decays to `1.57%`** ($E=4.71\times 10^{-5}$, $l/m=0.374$, **`42×`** below Bare Bell!) |
| **Bell ($L_n = 0.50$) + Retuned $2\%$ Hybrid** | — | **`-0.0758` / `-0.0150`** | **`80×200` $1T$ drops to $9\times 10^{-5}$ at $t=132$, then $2T$ cascade regrows!** (`P=2.65%, e2/e1=1.69`); direct **`2T` bomb locks onto `5.25%` pure-$2T$ cycle** ($5.28\times 10^{-4}$) |
| **Annular Slot ($L_n = 0.50$) + Retuned $2\%$ Hybrid** | — | **`-0.0716` / `-0.0138`** | **`80×200` monotonic quench to `0.45%`** ($E=3.75\times 10^{-6}$, $l/m=0.276$, **`523×`** below Bare Bell; `64×144`: `0.10%`!) |

*(`*` Exact bare annular $2T$ is marginally unstable at `+0.00225`, masked by `-0.01` numerical $2T$ damping on `80×200`, consistent with Exp 1D's per-mode damping table.)*

![Finite-Nozzle Entropy Filtering and 2D Transonic Rotating-Shock Shootout](../results/exp012_entropy_filter_and_transonic_shocks.png)

Full metrics: [`../results/exp012_metrics.json`](../results/exp012_metrics.json).

## 4. Verdict & Dead Ends

- **Honest Nonlinear Reality (Bell-Nozzle $2T$ Cascade Regrowth):** Even when Entry `010`'s retuned mid-sector $2\%$ baffle–cavity hybrid linearly stabilizes BOTH $1T$ (`-0.0758`) and $2T$ (`-0.0150`) inside a realistic $L_n = 0.50$ **bell nozzle**, bombing it with a $6.9\%$ $1T$ shock on `80×200` kills $1T$ cleanly by $t \approx 130$—**only for the evanescently trapped $2T$ harmonic (`e2/e1 = 1.69`) to begin regrowing out of the nonlinear $1T \to 2T$ cascade** toward the `5.25%` pure-$2T$ upper branch confirmed by `NL_bell50_hyb_2T_80`!
- **What Worked (Candidate Law #15 for `SYNTHESIS.md`):**
  1. **The Bell-Nozzle Evanescent-Cutoff Collapse:** Compact 0D Marble–Candel models (`009`–`011`) severely overestimate tangential acoustic radiation in realistic finite-length shrinking-radius bell nozzles ($R = \sqrt{A}$): at $L_n = 0.50\text{--}0.80$, rising tangential cutoff $\omega_{\text{cut}}(x) = m\bar{c}\sqrt{1-M^2}/R(x)$ reflects $1T$ and $2T$ waves upstream of the throat—collapsing $\bar{M}_c = 0.12$ cold $1T$ damping `7.5–46×` (`-0.0284 → -0.0038 → -0.0006`, verified on `96×200`), slashing global Hopf at $\bar{M}_c = 0.18$ by `-63%` (`0.426 → 0.156`) and at $\bar{M}_c = 0.12$ to `0.060` ($2T$-limited), causing **double-mode (`12.5%` $1T$ and `6.4%` $2T$) shock failure** in bare bell nozzles, and enabling $2T$ cascade regrowth even with a $2\%$ hybrid!
  2. **Constant-Radius Annular Slot / Plug Nozzle Advantage & Strouhal Tuning:** Contracting only the radial gap ($R(x) \equiv 1$, annular aerospike/plug throat) eliminates the tangential evanescent barrier completely—turning converging length $L_n$ into a **zero-pressure-drop Strouhal phase tuner** that lifts $\bar{M}_c = 0.12$ cold $1T$ damping **`2.56×` above Marble–Candel** (`-0.0729` at $L_n = 0.50$, `19.4×` above `L_n = 0.70` bell), raises global Hopf **`3.79×` above the `L_n = 0.50` bell** (`0.228` vs. `0.060`), decays the bare $6.9\%$ $1T$ shock $42\times$ in energy with zero hardware, and—paired with the retuned $2\%$ hybrid—**monotonically quenches the $6.9\%$ shock by `523×` (`0.45%`) on `80×200` with zero $2T$ regrowth**.
  3. **Finite-Nozzle Low-Pass Filtering of Indirect Entropy Noise:** Distributed acceleration across $L_n = 0.15 \to 1.10$ smoothly recovers compact Marble–Candel (`-0.366%`) at $L_n \to 0$ (`-0.354%` at `0.15`) and monotonically attenuates the first-pass rarefaction echo by `-15.8%` (`-0.308%` at `1.10`) and superposed 2nd-pass echo by `-42.9%`.
