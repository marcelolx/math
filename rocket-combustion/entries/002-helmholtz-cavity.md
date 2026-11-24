# 002 — Passive Acoustic Damping: Faceplate Helmholtz Resonator & Exceptional-Point Tuning

- **Date:** 2026-11-23 *(revised post peer review)*
- **Status:** `verified`
- **Verification Script:** `../sim/exp002_helmholtz_cavity.py`

## 1. Target

Entry `001` showed that causal time-lag dispersion ($\sigma_\tau = 0.15$) stabilizes high-$k$ harmonics (reaching **73.9%** full-domain $1L\text{--}12L$ stability on $\bar{\tau} \in [0.05, 2.0]$), while leaving low-frequency $1L$ lobes ($\bar{\tau} < 0.45$, e.g. Case B) strongly unstable.

Here we mount a compact **side-branch Helmholtz resonator (acoustic cavity)** at the injector faceplate ($x = 0$, the pressure antinode of all longitudinal modes), under the unified DC-blocked model ($H_{\text{hp}}(s) = s/(s+0.20)$, zero $k=0$ roots), to verify:

1. The **optimal cavity damping ratio $\xi_H$** for volume ratio $\beta = V_{\text{cav}}/V_{\text{chamber}}$ and its hot-fire shift relative to cold-flow bench tuning.
2. Both failure modes (**under-damped mode splitting** and **over-damped decoupling**) verified against direct 1D PDE runs using a **2-mode damped-sinusoid least-squares fit**.
3. **Uncensored** frequency-mistuning robustness ($\omega_H/\pi \in [0.65, 1.45]$) and full-domain synergy with Entry `001`'s causal time-lag dispersion (all four baselines computed on the exact same grid).

## 2. The Math & Assumptions

At $x = 0$, $U(0, t) = -v_r(t)$, with cavity pressure $P_c(t)$, volume ratio $\beta = V_{\text{cav}}/V_{\text{chamber}}$, resonance $\omega_H$, and damping ratio $\xi_H = 1/(2Q_H)$:

$$\frac{dv_r}{dt} + 2\xi_H\omega_H v_r = \beta\omega_H^2\big(P(0, t) - P_c\big), \qquad \frac{dP_c}{dt} = \frac{v_r}{\beta}$$

Exact total energy balance ($E_{\text{tot}} = \frac{1}{2}\int_0^1(P^2+U^2)\,dx + \frac{1}{2}\beta P_c^2 + \frac{1}{2\beta\omega_H^2}v_r^2$):

$$\frac{dE_{\text{tot}}}{dt} = n\,P(x_f, t)\,q(t) - \zeta\,P(1, t)^2 - \frac{2\xi_H}{\beta\omega_H}v_r(t)^2$$

Exact dispersion relation with $Y_H(s) = \frac{\beta\omega_H^2 s}{s^2 + 2\xi_H\omega_H s + \omega_H^2}$:

$$(1 + \zeta Y_H)\sinh(s) + (\zeta + Y_H)\cosh(s) = n\,H_{\text{hp}}(s)\,\hat{g}(s)\Big[\cosh(s x_f) + Y_H\sinh(s x_f)\Big]\Big[\cosh\big(s(1-x_f)\big) + \zeta\sinh\big(s(1-x_f)\big)\Big]$$

Leading-order **exceptional-point (critical coupling)** law near $\omega_H \approx \pi$:

$$\xi_H^{\text{EP}} \approx \sqrt{2\beta} - \frac{\alpha_{\text{uncoupled}}}{\pi}, \qquad \alpha_{\text{cold}}^{\text{EP}} \approx -\zeta - \pi\sqrt{\frac{\beta}{2}}$$

## 3. The Verification

Reproduce from the repo root:

```bash
uv run python rocket-combustion/sim/exp002_helmholtz_cavity.py
```

### Exceptional-Point Law vs. Exact Dispersion Optimum (Case B uncoupled: $s = +0.02469 + 3.0986i$)

| Case | Analytical $\xi_H^{\text{EP}}$ | Exact $\xi_H^{\text{opt}}$ | Exact $\max(\alpha_-, \alpha_+)$ | Notes |
| :--- | :---: | :---: | :---: | :--- |
| Cold chamber ($\beta = 2\%$) | $0.2127$ | $0.2060$ | $-0.3152$ | $7.9\times$ stronger than end loss alone |
| Hot Case B ($\beta = 2\%$) | $0.1921$ | $0.1781$ | $-0.2275$ | Shifted down by hot-fire driving |
| Hot Case B ($\beta = 0.5\%$) | $0.0921$ | $0.0858$ | $-0.1049$ | Stable window $\xi_H \in [0.0120, 0.2953]$ |

### Direct 1D PDE + Cavity ODE Verification (Hot Case B, $\beta = 0.5\%$, 2-Mode LSQ Fit)

| Regime | $\xi_H$ ($Q_H$) | Exact $\max s_\pm$ | PDE LSQ Fit | $|\Delta\alpha|$ | Energy Balance Err | Outcome |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| No cavity ($\beta = 0$) | — | $+0.025215 + 3.0983i$ | $+0.025220 + 3.0982i$ | $5.2\times 10^{-6}$ | $1.1\times 10^{-5}$ | Unstable $1L$ |
| **Under-damped trap** | $0.004$ ($125$) | $+0.009895 + 3.2498i$ | $+0.009884 + 3.2499i$ | $1.1\times 10^{-5}$ | $6.4\times 10^{-5}$ | **FAILS** (two unstable beating sidebands) |
| **Hot-EP optimal** | $0.086$ ($5.8$) | $-0.104847 + 3.1383i$ | $-0.105278 + 3.1370i$ | $4.3\times 10^{-4}$ | $4.2\times 10^{-5}$ | **STABLE** ($1L$ strongly damped) |
| **Over-damped decoupled** | $0.350$ ($1.4$) | $+0.004244 + 3.0994i$ | $+0.004251 + 3.0993i$ | $7.1\times 10^{-6}$ | $3.6\times 10^{-5}$ | **FAILS** (cavity decouples) |

![Exceptional Point, Mode Splitting, and Direct PDE Verification](../results/exp002_exceptional_point.png)

### Uncensored Mistuning Robustness ($\beta = 1\%$, $\omega_H/\pi \in [0.65, 1.45]$) & Full-Domain Synergy

- **Uncensored stable mistuning bandwidth ($\omega_H/\pi$ where $\max\alpha_\pm < 0$):**
  - Under-damped ($\xi_H = 0.015$): **11.0%** ($[0.960, 1.070]$)
  - At hot EP ($\xi_H = 0.134$): **50.0%** ($[0.820, 1.320]$) — **$4.5\times$ wider**
  - Over-EP ($\xi_H = 0.220$): **64.0%** ($[0.800, 1.440]$) — **$5.8\times$ wider**
- **Full-domain $1L\text{--}12L$ stability on $(x_f, \bar{\tau}) \in [0.05, 0.95] \times [0.05, 2.0]$ ($31 \times 40$ grid, all 4 baselines computed identically, `0` real $k=0$ hits):**
  - Point flame ($\beta = 0, \sigma_\tau = 0$): **3.2%** *(upper bound; $k > 12$ unstable)*
  - **Cavity ONLY** ($\beta = 2\%, \sigma_\tau = 0$): **7.8%** *(upper bound; dead end alone for high-$k$)*
  - Erlang-4 ONLY ($\beta = 0, \sigma_\tau = 0.15$): **73.9%**
  - **Synergy (Cavity $\beta = 2\%$ + Erlang-4 $\sigma_\tau = 0.15$):** **89.7%** (and Case B at $\bar{\tau}=0.20, x_f=0.15$ with $\sigma_\tau = 0.08$ reaches $\max_{k \le 12}\alpha_k = -0.0359$).

![Frequency Mistuning Robustness and Full-Spectrum Synergy](../results/exp002_mistuning_and_synergy.png)

Full metrics: [`../results/exp002_metrics.json`](../results/exp002_metrics.json).

## 4. Verdict & Dead Ends

- **What broke / dead ends:**
  - High-$Q$ cavities ($\xi_H = 0.004$) split $1L$ into two unstable beating sidebands (verified by 2-mode LSQ fit to $|\Delta\alpha| = 1.1\times 10^{-5}$) and narrow mistuning tolerance to **11.0%**.
  - A $1L$-tuned cavity alone cannot stabilize high-$k$ harmonics (**7.8%** $k \le 12$ upper bound) because $Y_H(i\omega) \to 0$ as $\omega \to \infty$.
- **What worked:**
  - Tuning near/slightly above the hot-fire EP $\xi_H^{\text{EP}} \approx \sqrt{2\beta} - \alpha_{\text{hot}}/\pi$ maximizes linear damping and widens uncensored mistuning tolerance by $4.5\text{--}5.8\times$.
  - Combining a $2\%$ faceplate cavity ($1L$) with causal Erlang-4 time-lag spread ($4L+$) achieves **89.7%** broadband $1L\text{--}12L$ stability across the full domain including $\bar{\tau} < 0.45$.
- **Next hypothesis (`003`):** Test nonlinear subcritical triggering and quadratic cavity orifice jet losses.
