# 001 — Spatial Placement of Grid-Forming Inverters: Import-Side Asymmetry & Semiconductor Saturation Cliff

- **Date:** 2026-11-23
- **Status:** `verified`
- **Verification Script:** `../sim/exp001_gfm_placement_and_saturation.py`

## 1. Target

Entry `000` revealed that asymmetrically concentrating Grid-Forming (GFM) inverters on the exporting side of a two-area AC grid can trigger an inter-area tie-line overload blackout. Here we test two fundamental engineering questions for **$100\%$ inverter-based renewable grids** under a fixed total GFM budget ($M_{\text{GFM,tot}} = 5.4, D_{\text{GFM,tot}} = 4.8$ across 6 generator buses on our 12-bus two-area grid):

1. **Spatial Placement Paradox (Linear $H_2$/Damping vs. Nonlinear $N-1$ Fault Survival):** Does optimizing linearized inter-area damping ($-\alpha_{\text{dom}}$) or the exact Lyapunov $H_2$ angle norm identify the best GFM placement for surviving large $N-1$ generator-loss + tie-line contingencies?
2. **Semiconductor Headroom Saturation Cliff:** Real IGBT/SiC inverters hit a hard active-power/current ceiling $h_i$ during faults (unlike copper-wound turbines that tolerate brief $3\text{--}5\times$ overloads). How does smooth droop saturation $P_{\text{droop},i} = -h_i\tanh(D_{\text{GFM},i}\omega_i / h_i)$ reshape the competition between concentrated tie-line gateway GFM and distributed 6-bus GFM?

## 2. The Math & Assumptions

### Saturated GFM Swing Equations & Exact Nonlinear Lyapunov Balance

Adding smooth semiconductor droop saturation to our Entry `000` swing network preserves exact Lyapunov monotonicity:

$$M_i \ddot{\theta}_i + D_{\text{base},i}\dot{\theta}_i + h_i \tanh\!\left(\frac{D_{\text{GFM},i}\dot{\theta}_i}{h_i}\right) = P_i - \sum_{j=1}^N K_{ij}\sin(\theta_i - \theta_j)$$

$$\frac{d\mathcal{H}}{dt} = -\sum_{i=1}^N \left[D_{\text{base},i}\omega_i^2 + h_i\omega_i\tanh\!\left(\frac{D_{\text{GFM},i}\omega_i}{h_i}\right)\right] \le 0$$

- For $|\omega_i| \ll h_i / D_{\text{GFM},i}$, droop is linear ($-D_{\text{GFM},i}\omega_i$); for large frequency excursions, emergency injection saturates smoothly at $\pm h_i$ (zero incremental droop stiffness).

### Exact Steady-State Import-Side Tie-Line Theorem

Consider two areas connected by a surviving $N-1$ tie line of capacity $K_{\text{tie}} = 1.60$ carrying pre-fault export $P_{\text{tie},0} = +0.90$ from Area A $\to$ Area B, with primary droop fractions $f_{D,A} = D_A / (D_A + D_B)$ and $f_{D,B} = 1 - f_{D,A}$. Under a generator trip $-\Delta P$ in **importing Area B**, steady post-fault tie flow is:

$$P_{\text{tie}}^{\text{post}}(\text{Fault in B}) = P_{\text{tie},0} + f_{D,A}\,\Delta P \le K_{\text{tie}} \quad \Longrightarrow \quad \Delta P_{\max,B}^{\text{ss}} = \frac{K_{\text{tie}} - P_{\text{tie},0}}{f_{D,A}}$$

Under a generator trip $-\Delta P$ in **exporting Area A**, Area B's droop pushes power *opposite* to $P_{\text{tie},0}$ (unloading the congested corridor first!):

$$P_{\text{tie}}^{\text{post}}(\text{Fault in A}) = P_{\text{tie},0} - f_{D,B}\,\Delta P \ge -K_{\text{tie}} \quad \Longrightarrow \quad \Delta P_{\max,A}^{\text{ss}} = \frac{K_{\text{tie}} + P_{\text{tie},0}}{f_{D,B}}$$

Because $K_{\text{tie}} + P_{\text{tie},0} = 2.50 \gg K_{\text{tie}} - P_{\text{tie},0} = 0.70$, **concentrating primary droop on the importing side ($f_{D,B} \gg f_{D,A}$) dramatically maximizes worst-case $\min(\Delta P_{\max,A}, \Delta P_{\max,B})$**—an asymmetric directional law to which symmetric linearized $H_2$ norms and eigenvalues are completely blind!

### Intentional Simplifications
- Active-power/angle dynamics (constant voltage magnitudes $|V_i| = 1$, lossless lines); omits reactive-power $Q\text{--}V$ voltage droop and current-angle limiting inside the inverter inner loop.
- Worst-case nonlinear fault survival $\Delta P_{\text{crit}} = \min(\Delta P_{\text{crit}}(\text{Bus 0 in A}),\, \Delta P_{\text{crit}}(\text{Bus 6 in B}))$ is bisected up to $2.40\,\text{pu}$ (11 bisection steps, $0.001\,\text{pu}$ resolution) under simultaneous outage of tie line `(5, 9)`.

## 3. The Verification

Reproduce from the repo root:

```bash
uv run python grid-stability/sim/exp001_gfm_placement_and_saturation.py
```

- **Nonlinear Lyapunov energy conservation with `tanh` droop saturation:** verified to relative residual **$\mathbf{7.97\times 10^{-8}}$**.
- **Horizon convergence check** (uniform GFM, $h_{\text{bus}} = 0.24$, Bus 6 fault): $t = 26$ gives $\Delta P_{\text{crit}} = 1.3283$ vs. $t = 52$ giving $1.3283$ ($|\Delta| = 0.00$).

### Experiment 1 — Unsaturated Fixed-Budget GFM Placement ($M_{\text{tot}} = 5.4, D_{\text{tot}} = 4.8$)

| Placement Policy | Linear $-\alpha_{\text{dom}}$ | Linear $H_2$ Angle Norm *(lower=better)* | $\Delta P_{\text{crit}}$ (Fault in A) | $\Delta P_{\text{crit}}$ (Fault in B) | Worst-Case $\Delta P_{\text{crit}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Area A Only *(exporting)* | $0.2676$ | $4.396$ | $\ge 2.40$ | **$0.728$ *(FAILS EARLY)*** | **$0.728$** |
| **Area B Only *(importing!)*** | $0.2759$ *(worst linear!)* | $4.405$ *(worst $H_2$!)* | $2.285$ | **$\ge 2.40$** | **$2.285$ *(3.14x vs A, +54% vs Fiedler!)*** |
| Tie-Line Gateways `{2, 6}` | $0.2770$ | $4.308$ | $\ge 2.40$ | $1.431$ | $1.431$ |
| Fiedler-Weighted (all 6) | $0.4289$ | **$2.926$ *(best $H_2$!)*** | $\ge 2.40$ | $1.481$ | $1.481$ |
| Uniform Distributed (all 6) | **$0.4298$ *(best linear!)*** | $2.986$ | $\ge 2.40$ | $1.442$ | $1.442$ |

![Optimal Spatial Allocation of Limited GFM Inverters](../results/exp001_gfm_placement.png)

### Experiment 2 — Semiconductor Headroom Saturation Cliff

**Case 2A — Equal per-bus inverter headroom ceiling $h_{\text{bus}}$** (each active GFM bus clips at $\pm h_{\text{bus}}$, so 6 distributed buses unlock $3\times$ more system headroom than 2 concentrated gateways):

| Per-Bus Cap $h_{\text{bus}}$ (pu) | Tie-Line Gateways `{2, 6}` | Fiedler-Weighted (6 buses) | Uniform Distributed (6 buses) | Distributed Gain over Gateways |
| :---: | :---: | :---: | :---: | :---: |
| $0.10$ | $0.843$ | $0.788$ | $0.761$ | $-10\%$ *(all saturated)* |
| $0.16$ | $0.862$ | $1.096$ | **$1.171$** | **$+35.8\%$** |
| **$0.24$** | **$0.902$** | **$1.331$** | **$1.328$** | **$+47.6\%$** |
| $0.35$ | $1.114$ | **$1.413$** | $1.396$ | $+26.8\%$ |
| $0.55$ | $1.262$ | **$1.457$** | $1.427$ | $+15.5\%$ |
| $2.00$ *(near-linear)* | $1.421$ | $1.479$ | $1.441$ | $+4.1\%$ |

**Case 2B — Fixed total system headroom $H_{\text{tot}} = \sum h_i$ allocated proportionally:** At tight global headroom ($H_{\text{tot}} = 0.60$), tie-line gateways `{2, 6}` retain an edge (`1.088` vs. uniform `0.761`) by injecting directly at the corridor boundary; once $H_{\text{tot}} \ge 1.20$, Fiedler-weighted placement leads (`1.308` to `1.461`).

![Inverter Semiconductor Headroom Saturation Cliff](../results/exp001_headroom_saturation.png)

Full metrics: [`../results/exp001_metrics.json`](../results/exp001_metrics.json).

## 4. Verdict & Dead Ends

- **Major counterintuitive finding — Linear $H_2$ Placement Fails Directional Corridor Congestion:**
  - Linearized damping ($-\alpha_{\text{dom}}$) and exact Lyapunov $H_2$ angle norms treat exporting-only (`Area A`) and importing-only (`Area B`) GFM placement as nearly identical ($H_2 = 4.396$ vs. $4.405$) and rank both *worst* compared to Fiedler/uniform placement ($H_2 = 2.93\text{--}2.99$).
  - Yet in true nonlinear $N-1$ fault survival, **importing-side GFM (`Area B`) achieves $\Delta P_{\text{crit}} = 2.285$—$3.14\times$ higher than exporting-side (`0.728`) and $+54\%$ higher than Fiedler-weighted (`1.481`)!** Our exact steady-state tie-line theorem explains why: placing droop on the importing side absorbs local faults without adding flow to the congested corridor while unloading the corridor during remote exporting-side faults.
- **Semiconductor Headroom Cliff favors Distributed GFM over Concentrated Gateways:**
  - Under realistic per-bus inverter current ceilings ($h_{\text{bus}} = 0.16\text{--}0.24\,\text{pu}$), relying on only 2 concentrated tie-line GFM gateways clips their droop early and slashes $N-1$ fault survival by **$-37\%$** (`1.431 -> 0.902`), whereas enabling GFM droop across all 6 existing inverter buses unlocks shared headroom and raises survival by **$+48\%$** (`1.331` vs. `0.902`).
- **Practical engineering rule:** Combine **distributed multi-bus GFM activation** (to share per-bus semiconductor headroom) with **import-side bias** across congested transmission corridors (not purely symmetric or linear-$H_2$-optimal weights).
- **Next step:** Integrate Peer Review #3 findings for `plasma-nozzles/` (`000`–`002`), write `plasma-nozzles/SYNTHESIS.md`, and run an independent peer review on `grid-stability/` Entries `000`–`001`.
