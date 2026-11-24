# 000 — Nonlinear AC Swing Baseline, Exact Braess's Paradox & Grid-Forming Inverter Stabilization

- **Date:** 2026-11-23
- **Status:** `verified`
- **Verification Script:** `../sim/baseline_power_grid.py`

## 1. Target

As humanity decarbonizes electrical power grids by replacing heavy rotating synchronous generators with power-electronic solar/wind/battery inverters, two structural failure modes threaten grid stability: **(1) topological phase-frustration blackouts (Braess's paradox)** where adding transmission capacity reduces synchronous stability, and **(2) low-inertia frequency/synchronization collapse** under $N-1$ contingencies.

In this baseline session we build and verify a **coupled nonlinear AC swing-equation network solver (RK4)** cross-checked against exact Hamiltonian/Lyapunov conservation laws and exact linearized eigenmodes, and test:

1. Exact lossless Hamiltonian ($D = 0$) and damped Lyapunov ($D > 0$) energy conservation + slowest inter-area eigenmode verification on a 6-bus meshed AC grid.
2. An **exact closed-form analytical proof and time-domain blackout demonstration of AC Braess's Paradox** on a 4-bus ring.
3. A **12-bus two-area meshed grid** comparing Grid-Following (GFL) inverter replacement vs. Grid-Forming (GFM) low-pass-filtered droop inverters under a combined generator-trip ($\Delta P = 0.85$) + $N-1$ inter-area tie-line outage—uncovering an **asymmetric GFM placement tie-line overload blackout trap**.

## 2. The Math & Assumptions

### Nonlinear AC Swing Equations & Exact GFM Droop Equivalence

On an $N$-bus lossless AC transmission network with line susceptance capacities $K_{ij} = K_{ji} \ge 0$ and power injections $P_i$:

$$M_i \ddot{\theta}_i + D_i \dot{\theta}_i = P_i - \sum_{j=1}^N K_{ij}\sin(\theta_i - \theta_j)$$

- **Exact Grid-Forming (GFM) Droop Equivalence (D'Arco & Suul 2014):** A GFM inverter with low-pass power filter $\tau_f \dot{\tilde{P}}_i = P_{e,i} - \tilde{P}_i$ and active-power frequency droop $\dot{\theta}_i = -k_{p,i}(\tilde{P}_i - P_i)$ obeys the swing equation identically with virtual inertia $M_{\text{GFM},i} = \tau_f / k_{p,i}$ and primary droop damping $D_{\text{GFM},i} = 1 / k_{p,i}$.
- **Exact Nonlinear Lyapunov / Hamiltonian Energy Balance:** Defining $\mathcal{H}(\theta, \omega) = \frac{1}{2}\sum_i M_i\omega_i^2 - \sum_i P_i\theta_i - \frac{1}{2}\sum_{i,j} K_{ij}\cos(\theta_i - \theta_j)$ yields $\frac{d\mathcal{H}}{dt} = -\sum_i D_i \omega_i^2 \le 0$ for *any* nonlinear trajectory (including phase-slip blackouts).
- **Exact Center-of-Inertia (COI) Frequency Laws:** Under a sudden net power imbalance $-\Delta P$ ($\sum_i P_i = -\Delta P$), exact initial COI Rate of Change of Frequency (RoCoF) and steady-state primary frequency drop are:
  $$\dot{f}_{\text{COI}}(0^+) = -\frac{\Delta P}{2\pi M_{\text{tot}}}, \qquad \Delta f_{\text{ss}} = -\frac{\Delta P}{2\pi D_{\text{tot}}}, \qquad M_{\text{tot}} = \sum_i M_i,\; D_{\text{tot}} = \sum_i D_i$$
- **Exact 4-Bus AC Braess Capacity Law:** On a 4-bus ring `0-1-2-3-0` with $P = (+P_0, -P_0, +P_0, -P_0)$ and reinforced line $K_{01} = 1 + \Delta K$ ($K_{12} = K_{23} = K_{30} = 1$), reflection symmetry gives $\sin\gamma = (1 + \Delta K)\sin\alpha$ and Kirchhoff's loop constraint forces $\beta = \frac{1}{2}(\alpha + \gamma)$, yielding:
  $$P_0(\gamma) = \sin\gamma + \sin\!\left(\frac{\gamma + \arcsin\big(\sin\gamma / (1 + \Delta K)\big)}{2}\right)$$
  For $\Delta K = 0$, $P_{\max}(0) = 2.000$ ($\alpha = \beta = \gamma = \pi/2$). As $\Delta K \to \infty$ (superconducting link `0-1`, $\alpha \to 0$), maximizing $\sin\gamma + \sin(\gamma/2)$ gives $\cos(\gamma^*/2) = \frac{-1 + \sqrt{33}}{8}$ and **$P_{\max}(\infty) = 1.760173$**—a provable **$-11.99\%$ loss in total grid capacity** from adding infinite transmission capacity!

### Intentional Simplifications
- Lossless reactive lines ($R/X = 0$, constant bus voltage magnitudes $|V_i| = 1$); ignores voltage collapse, reactive power $Q\text{--}V$ coupling, and inverter current saturation limits (to be added in Entry `001`).
- Grid-Following (GFL) inverters are modeled in the swing framework with minimal residual inertia/damping ($M_{\text{GFL}} = 0.08, D_{\text{GFL}} = 0.04$) rather than explicit second-order PLL phase-tracking dynamics.

## 3. The Verification

Reproduce from the repo root:

```bash
uv run python grid-stability/sim/baseline_power_grid.py
```

### Experiment 1 — Exact Energy Conservation & Linear Eigenmode Check (6-Bus Meshed Grid)

- **Lossless Hamiltonian conservation ($D = 0$, finite amplitude `0.25 rad`):** relative error $\mathbf{5.05\times 10^{-12}}$.
- **Damped Lyapunov energy balance ($D > 0$, finite amplitude `0.30 rad`):** relative error $\mathbf{9.90\times 10^{-8}}$.
- **Slowest inter-area oscillatory eigenmode:** exact Jacobian $s = -0.094722 + 1.603813i$ vs. nonlinear RK4 damped-sinusoid LSQ fit $-0.094599 + 1.603787i$ ($|\Delta\alpha| = 1.2\times 10^{-4}$, $|\Delta\omega| = 2.6\times 10^{-5}$).

### Experiment 2 — Exact Nonlinear AC Braess's Paradox (4-Bus Ring)

- Analytical $P_{\max}(\Delta K)$ vs. full 4-bus numerical equilibrium + Fiedler stability bisection across 36 values of $\Delta K \in [0, 50]$: maximum discrepancy **$5.1\times 10^{-4}$**.
  - $\Delta K = 0$: $P_{\max} = \mathbf{2.0000}$
  - $\Delta K = 1.0$ (doubling line `0-1`): $P_{\max} = \mathbf{1.8894}$ (**$-5.53\%$**)
  - $\Delta K \to \infty$: $P_{\max} = \mathbf{1.7602}$ (**$-11.99\%$**)
- **Time-domain blackout at $P_0 = 1.85$:** Unreinforced ring settles cleanly to synchrony ($\omega_{\text{rms}} = 4.0\times 10^{-5}$); switching on parallel line $\Delta K = +2.0$ at $t = 15$ immediately destroys synchrony ($\omega_{\text{rms}} = 4.47\,\text{rad/s}$, angle spread $277.3\,\text{rad}$).

![Exact Nonlinear AC Braess's Paradox](../results/baseline_braess_paradox.png)

### Experiment 3 — 12-Bus Two-Area Grid: Low-Inertia GFL Cliff vs. GFM Stabilization ($\Delta P = 0.85$ Gen Trip in Area B + Tie-Line Outage)

| Configuration | Synced? | COI RoCoF Sim / Exact (Hz/s) | COI Nadir $|\Delta f|$ (Hz) | Max Angle Spread (rad) | Outcome |
| :--- | :---: | :---: | :---: | :---: | :--- |
| $100\%$ Synchronous (`6 Sync`) | `True` | $0.005$ / $0.005$ | $0.036$ | $2.20$ | Stable baseline |
| $50\%$ GFL (`3 Sync, 3 GFL`) | `True` | $0.010$ / $0.010$ | $0.058$ | $2.85$ | $2\times$ RoCoF, $1.6\times$ nadir |
| **$83.3\%$ GFL (`1 Sync in A, 5 GFL`)** | **`False`** | $0.026$ / $0.026$ | $0.054$ | **$77.60$** | **INTER-AREA POLE-SLIP BLACKOUT!** |
| $100\%$ GFL (`0 Sync, 6 GFL`) | `True` | **$0.113$** / $0.113$ | **$0.141$** | $2.16$ | **$22.6\times$ RoCoF, $3.9\times$ nadir plunge** |
| **$100\%$ Inv, $16.7\%$ GFM (`1 GFM in A`)** | **`False`** | $0.046$ / $0.046$ | $0.062$ | **$78.23$** | **ASYMMETRIC GFM TIE-LINE BLACKOUT!** |
| **$100\%$ Inv, $33.3\%$ GFM (`1 in A, 1 in B`)** | **`True`** | $0.029$ / $0.029$ | **$0.033$** | **$2.02$** | **Restores synchrony & beats Sync nadir!** |
| **$100\%$ Inv, $100\%$ GFM (`6 GFM`)** | **`True`** | **$0.012$** / $0.012$ | **$0.013^*$** | **$1.76$** | **$9.4\times$ lower RoCoF vs GFL, $2.75\times$ better nadir than Sync!** |

*\*RK4 timestep convergence check on $100\%$ GFM: $\Delta t = 0.0025$ vs. $\Delta t = 0.00125$ gives COI nadir $|\Delta f| = 0.013109$ on both ($|\Delta| = 6.9\times 10^{-17}$).*

![12-Bus Low-Inertia GFL Cliff vs Grid-Forming Droop Stabilization](../results/baseline_gfm_vs_gfl_cliff.png)

Full metrics: [`../results/baseline_metrics.json`](../results/baseline_metrics.json).

## 4. Verdict & Dead Ends

- **What broke / unexpected grid traps uncovered:**
  - **Exact AC Braess's Paradox:** Adding parallel capacity to a single line in a stressed loop creates Kirchhoff phase frustration that cuts synchronous capacity by up to **$-11.99\%$** ($P_{\max}: 2.000 \to 1.760$), triggering an immediate blackout at $P_0 = 1.85$.
  - **Asymmetric GFM Placement Tie-Line Overload Trap:** Deploying a *single* Grid-Forming inverter (or leaving a single synchronous generator) concentrated in the exporting Area A (`1/6 = 16.7%`) concentrates $63\%$ of total grid droop in Area A; when Area B suffers a generator trip + tie-line outage, Area A's lone GFM inverter floods emergency droop power across the surviving tie line, **tearing the grid apart (`spread = 78.2 rad`)**!
- **What worked:**
  - **Geographically Balanced Multi-Area GFM Deployment ($\ge 33.3\%$, at least one GFM per area):** Immediately eliminates the inter-area blackout (`spread = 2.02 rad`), cuts COI RoCoF by $3.9\times$ and frequency nadir by $4.3\times$ vs. $100\%$ GFL, and at $100\%$ GFM achieves **$2.75\times$ tighter frequency nadir than $100\%$ legacy synchronous generators** ($0.013\,\text{Hz}$ vs. $0.036\,\text{Hz}$).
- **Next single hypothesis to test (`001`):**
  - **Optimal Spatial Placement of Limited GFM Inverters & Braess-Mitigating Virtual Impedance / Droop Co-Design:** Given a fixed budget of $k$ Grid-Forming inverters (or fixed total virtual inertia $M_{\text{GFM,tot}}$), which graph-theoretic placement rule (area-balanced vs. high-degree hub vs. electrical-centrality / Fiedler-vector placement) maximizes transient $N-1$ fault survival—and can **inverter current-limit saturation** ($|I_i| \le I_{\max}$) break GFM synchronization during severe faults?
