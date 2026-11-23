# 001 — Finite Flame Width vs. Causal Time-Lag Dispersion

- **Date:** 2026-11-23
- **Status:** `verified` (includes a clean analytical & numerical negative result)
- **Verification Script:** `../sim/exp001_spatial_vs_timelag.py`

## 1. Target

Entry `000` uncovered the **point-flame harmonic whack-a-mole problem**: when $n > \operatorname{arctanh}(\zeta)$, tuning $(x_f, \tau)$ to suppress low modes pushes instability into higher harmonics ($4L, 8L, 12L, \dots$), collapsing the $(x_f, \tau)$ linearly stable area to **1.7%** across modes $1L\text{--}12L$.

In this session we test **three physical extensions** of the 1D combustor ($n = 0.10, \zeta = 0.04$) to answer:

> *Does finite spatial flame thickness ($\sigma_f$) cure high-frequency harmonic instabilities, or do we need feed-coupled spatial averaging / causal time-lag dispersion ($\sigma_\tau$)?*

## 2. The Math & Assumptions

Starting from the verified Entry `000` 1D closed–choked acoustic equations ($U(0,t)=0$, $U(1,t)=\zeta P(1,t)$, $\zeta=0.04$, cold damping $\alpha_0 = -\operatorname{arctanh}(\zeta) = -0.040021$), we compare three heat-release models with normalized spatial profile $w(x)$ ($\int_0^1 w(x)\,dx = 1$):

### Mechanism 1 — Local Pressure Coupling + Finite Flame Width $\sigma_f$
$$\dot{Q}_{\text{loc}}(x, t) = n\,w_{\sigma_f}(x)\,P(x, t - \tau)$$
In the modal Rayleigh integral for mode $k$, the spatial coupling factor is:
$$I_k[w] = \int_0^1 w(x)\cos^2(k\pi x)\,dx = \frac{1}{2} + \frac{1}{2}\int_0^1 w(x)\cos(2k\pi x)\,dx \xrightarrow{k \to \infty} \frac{1}{2}$$
By Riemann–Lebesgue, $I_k[w]$ **does not decay to zero**—it asymptotes to $1/2$ for *any* smooth flame distribution $w(x)$. This predicts a sharp threshold theorem:
$$n_c = 2\operatorname{arctanh}(\zeta) \approx 0.080043$$
- For $n < n_c$, spatial spreading can stabilize high-$k$ modes.
- For $n > n_c$ (including our $n = 0.10$ baseline), **no spatial flame distribution $w(x)$ can stabilize high-$k$ harmonics** under single-lag local coupling; the high-$k$ asymptote remains strictly unstable ($\alpha_\infty \approx -\zeta + \frac{n}{2}e^{-\alpha_\infty\tau} \approx +0.0095 > 0$). For a uniform flame $w(x) \equiv 1$, this is confirmed by the exact dispersion relation $\kappa\sinh(\kappa) + \zeta s\cosh(\kappa) = 0$ with $\kappa = \sqrt{s(s - n e^{-s\tau})}$.

### Mechanism 2 — Injector/Feed-Coupled Distributed Flame Width $\sigma_f$
$$\dot{Q}_{\text{inj}}(x, t) = n\,w_{\sigma_f}(x)\,P(0, t - \tau)$$
Here faceplate pressure fluctuations modulate propellant mass flow, which burns across $w(x)$. Using the exact 1D acoustic Green's function yields the closed-form dispersion relation:
$$\sinh(s) + \zeta\cosh(s) = n\,e^{-s\tau}\int_0^1 w(\xi)\Big[\cosh\big(s(1-\xi)\big) + \zeta\sinh\big(s(1-\xi)\big)\Big]\,d\xi$$
Modal driving scales with $\hat{w}_k = \int_0^1 w(x)\cos(k\pi x)\,dx \sim \cos(k\pi x_f)e^{-\frac{1}{2}(k\pi\sigma_f)^2}$, giving super-exponential high-$k$ roll-off (note: because $\hat{w}_k$ can flip sign, modes where $\hat{w}_k$ and $\cos(k\pi\tau)$ are both negative can still be driven at low $k$).

### Mechanism 3 — Local Coupling + Causal Shifted Erlang-$m$ Time-Lag Spread $\sigma_\tau$
Even under local pressure coupling, polydisperse atomization/vaporization/mixing introduces a strictly causal distribution of time lags (dead time $\tau_0 = \bar{\tau} - \tau_d \ge 0$ plus an $m$-stage Erlang cascade of mean $\tau_d = \sqrt{m}\sigma_\tau$, total mean $\bar{\tau}$ and standard deviation $\sigma_\tau$). In the Laplace domain:
$$\hat{g}(s) = \frac{e^{-s\tau_0}}{\left(1 + \frac{s\tau_d}{m}\right)^m}, \qquad |\hat{g}(ik\pi)| = \left(1 + \frac{(k\pi\sigma_\tau)^2}{m}\right)^{-m/2}$$
In the time domain, this is exact via $m$ linear scalar filter ODEs coupled into our RK4 PDE solver. Any mode $k$ becomes unconditionally stable (regardless of phase) once $n|\hat{g}(ik\pi)| < \operatorname{arctanh}(\zeta)$, giving an exact analytical cutoff:
$$\sigma_\tau^*(k) = \frac{\sqrt{m}}{k\pi}\sqrt{\left(\frac{n}{\operatorname{arctanh}(\zeta)}\right)^{2/m} - 1}$$
For $n = 0.10, \zeta = 0.04, m = 4$, killing $4L$ and all higher harmonics requires only $\sigma_\tau^*(4) = \mathbf{0.1213}$.

### Intentional Simplifications
- Still 1D linear longitudinal acoustics with constant choked-nozzle admittance $\zeta = 0.04$.
- Mechanism 3 uses a point spatial flame with $m=4$ Erlang lag dispersion to isolate temporal dispersion cleanly from spatial smearing.

## 3. The Verification

Reproduce from the repo root:

```bash
uv run python rocket-combustion/sim/exp001_spatial_vs_timelag.py
```

### Uniform Local Flame Threshold Check ($n_c = 2\operatorname{arctanh}(\zeta) = 0.080043$)

| Regime | $n$ | Exact $4L$ $\alpha$ | PDE $4L$ $\alpha$ | $|\Delta\alpha|$ | Exact $8L$ $\alpha$ | Outcome |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| Above threshold ($n > n_c$) | $0.10$ | $+0.009506$ | $+0.009543$ | $3.7\times 10^{-5}$ | $+0.009506$ | **Unstable** |
| Below threshold ($n < n_c$) | $0.07$ | $-0.004851$ | $-0.004813$ | $3.8\times 10^{-5}$ | $-0.004851$ | **Stable** |

### Case D ($x_f = 0.25, \bar{\tau} = 1.00, n = 0.10, \zeta = 0.04$) Three-Mechanism Comparison

| Model | Parameter | Exact $4L$ $\alpha$ | PDE $4L$ $\alpha$ | $\max_{k \le 12} \alpha_k$ | Verdict |
| :--- | :--- | :---: | :---: | :---: | :--- |
| Entry `000` Point flame | $\sigma = 0$ | $+0.054737$ | $+0.054768$ | $+0.0547$ | Unstable ($4L, 8L, 12L$) |
| **Mech 1:** Local Gaussian | $\sigma_f = 0.12$ | $+0.009617$* | $+0.009649$ | $+0.0096$ | **FAILS** ($4L, 6L, 8L, 10L, 12L$ grow) |
| **Mech 2:** Injector Gaussian | $\sigma_f = 0.12$ | $-0.076995$ | $-0.076956$ | $-0.0011$ | **STABLE** ($3L$ barely negative) |
| **Mech 3:** Causal Erlang-4 | $\sigma_\tau = 0.15$ | $-0.018842$ | $-0.018827$ | $-0.0188$ | **STABLE** (all modes well-damped) |

*\*Self-consistent modal Rayleigh integral prediction; PDE matches to $3.2\times 10^{-5}$.*

![Three Mechanisms Comparison on Case D](../results/exp001_three_mechanisms.png)

### Broadband ($1L\text{--}12L$) Stability Area Recovery in $(x_f, \bar{\tau}) \in [0.05, 0.95] \times [0.45, 2.0]$

| Dispersion Width | Mech 1: Local Spatial $\sigma_f$ | Mech 3: Causal Erlang-4 $\sigma_\tau$ |
| :---: | :---: | :---: |
| $0.00$ | $1.7\%$ | $1.7\%$ |
| $0.04 / 0.05$ | $4.1\%$ ($\sigma_f=0.05$) | $15.3\%$ ($\sigma_\tau=0.04$) |
| $0.08$ | $4.6\%$ | $52.6\%$ |
| $0.12$ | $2.6\%$ | $72.6\%$ |
| $0.15 / 0.16$ | $2.3\%$ ($\sigma_f=0.16$) | $80.0\%$ ($\sigma_\tau=0.15$) |
| $0.20$ | **$1.9\%$ (dead end)** | **$87.5\%$ (broadband stable)** |

![Broadband Stability Recovery and Recovered Map](../results/exp001_stability_recovery.png)

Full metrics: [`../results/exp001_metrics.json`](../results/exp001_metrics.json).

## 4. Verdict & Dead Ends

- **Dead end / clean negative result (Mechanism 1):**
  - Widening a *locally* pressure-coupled combustion zone ($\dot{Q} = n w_{\sigma_f}(x) P(x, t-\tau)$) **cannot** cure high-frequency instabilities whenever $n > 2\operatorname{arctanh}(\zeta)$. Even a wide Gaussian ($\sigma_f = 0.20$) or completely uniform chamber-wide flame ($w \equiv 1$) leaves the $1L\text{--}12L$ stable operating area stuck at **1.9%–4.6%**, because $\int w(x)\cos^2(k\pi x)\,dx \to \frac{1}{2}$ instead of $0$.
  - **Design lesson:** Axial flame spreading only acts as a true high-frequency low-pass filter when the pressure sensitivity is *non-local* (e.g., feed/injector-coupled modulation at $x = 0$, Mechanism 2).

- **What worked (Mechanisms 2 & 3):**
  - **Causal time-lag dispersion ($\sigma_\tau$, Mechanism 3)** is dramatically more effective under local pressure coupling: crossing the exact analytical threshold $\sigma_\tau^*(4) = 0.1213$ extinguishes the entire $4L/8L/12L\dots$ comb (stabilizing Case D to $\max\alpha = -0.0188$), recovering the broadband $1L\text{--}12L$ stable area from **1.7% to 80.0%** at $\sigma_\tau = 0.15$ and **87.5%** at $\sigma_\tau = 0.20$. Note that low-frequency modes still resist moderate time-lag spread (e.g., Case C's $2L$ growth rate drops by 86% from $+0.0252$ to $+0.0036$ at $\sigma_\tau = 0.15$, remaining marginally unstable).
  - **Injector-coupled spatial distribution (Mechanism 2)** also kills high-$k$ modes super-exponentially, though low-$k$ sign-flips in $\hat{w}_k = \int w(x)\cos(k\pi x)\,dx$ can unexpectedly bring a previously damped odd mode (here $3L$, $\alpha_{3L} = -0.0011$) right up to the stability margin.

- **Next single hypothesis to test (`002`):**
  - **Passive Acoustic Damping via a Tuned Side-Branch Helmholtz Resonator (Acoustic Cavity):** Even with realistic time-lag spread ($\sigma_\tau = 0.15$), the **fundamental $1L$ instability lobe** (e.g., Case B at $\bar{\tau} = 0.20, x_f = 0.15$) remains strongly unstable because $\sigma_\tau$ barely attenuates low frequencies ($\omega_1 = \pi$). Test whether coupling a single compact Helmholtz resonator at optimal axial placement $x_r$ (and optimal cavity resistance $R_c$ / tuning frequency $\omega_H$) can stabilize Case B across *all* modes—and map the classic **mode-splitting / optimal-damping trap** where under-damped cavities split $1L$ into two unstable sideband peaks!
