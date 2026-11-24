# Synthesis — Power Grid Synchronization & Renewable Stability (`000`–`001`)

This monograph synthesizes two verified open-notebook entries (`000`–`001`), audited by an independent cross-model peer review (Peer Review #4, Claude Opus 5.5, 7 independent spot checks), investigating nonlinear AC synchronization, topological Braess's paradox blackouts, and Grid-Forming (GFM) inverter placement under primary-droop saturation in $100\%$ renewable grids.

---

## 1. Consolidated Verified Design Laws

| # | Design Law / Structural Result | Quantitative Rule (Verified in RK4 & Peer-Reviewed) | Source Entry |
| :-: | :--- | :--- | :---: |
| **1** | **Exact AC Braess's Paradox Capacity Loss** | Reinforcing a single line $K_{01} = 1 + \Delta K$ in a stressed 4-bus ring $P = (+P_0, -P_0, +P_0, -P_0)$ cuts global synchronous capacity via Kirchhoff loop phase frustration from $P_{\max}(0) = 2.000$ to **$P_{\max}(\infty) = 1.760173$ ($-11.99\%$)**—with exact fold = Fiedler stability boundary ($\lambda_2 = 0$ to $5.8\times 10^{-10}\,\text{rad}$) and immediate time-domain blackout at $P_0 = 1.85$. | [`000`](entries/000-baseline.md) |
| **2** | **Low-Inertia GFL Cliff vs. Balanced GFM Restoration** | Replacing synchronous machines with Grid-Following (GFL) inverters explodes COI RoCoF by **$22.6\times$** and deepens frequency nadir by $3.9\times$ ($100\%$ GFL). Placing a single GFM solely in exporting Area A concentrates $81.0\%$ of droop there and causes a thin-margin first-swing tie-line blackout (`spread = 78.2 rad`), whereas moving it to importing Area B (`spread = 1.73 rad`) or deploying balanced multi-area GFM ($\ge 33.3\%$) restores synchrony and cuts RoCoF $3.9\text{--}9.4\times$. | [`000`](entries/000-baseline.md) |
| **3** | **Import-Side Asymmetry Theorem (Linear $H_2$ Blind Spot)** | Exact linear $H_2$ angle norms ($[Q^T\delta, \omega]$ realization) are nearly insensitive to corridor flow direction (`Area A` $4.397$ vs. `Area B` $4.405$) and rank `Area B only` worst, yet horizon-robust ($T = 80$, exact static check) nonlinear $N-1$ survival proves **importing-side GFM achieves $\Delta P_{\text{crit}} = 2.285$** unsaturated ($3.14\times$ `Area A` $0.728$; $+59\%$ over `Fiedler` $1.435$) and **$2.118$** under per-bus droop cap $h_{\text{bus}} = 0.24$ ($+59\%$ over `Fiedler` $1.331$). | [`001`](entries/001-gfm-placement-and-saturation.md) |
| **4** | **Per-Bus Droop-Cap Sharing Advantage** | Among balanced cross-corridor GFM allocations under equal per-bus primary-droop ceilings ($h_{\text{bus}} = 0.24\,\text{pu}$), concentrated 2-bus tie-line gateways drop **$-21.6\%$** (`1.400 -> 1.097`), whereas distributed 6-bus GFM (`Uniform` $1.328$, `Fiedler` $1.331$) shares droop headroom across all inverters for **$+21\%$** higher fault survival. | [`001`](entries/001-gfm-placement-and-saturation.md) |

---

## 2. Documented Dead Ends & Peer-Review Corrections

1. **Short-horizon ($T = 26$) survival classifier without static equilibrium check (Peer Review #4 C1–C3):** Omitting the post-fault rotating-frame equilibrium check allowed ghost-crawl trajectories to pass $T = 26$ before slipping at $t = 30\text{--}44$ (inflating balanced thresholds above the `1.400/1.435` static bounds), while a late-window `max_om_diff < 0.04` test falsely rejected slow-settling saturated cases (`Area B only` at `h_bus=0.24` was misclassified as `1.058` instead of its true `2.118`). Fixed with exact saturated `brentq` rotating-frame equilibrium verification + $T = 80$ pole-slip check (verified to $0.00$ diff at $T = 120$).
2. **Non-proportional $M, D$ $H_2$-norm COI coupling bug (Peer Review #4 M1):** Projecting out COI velocity assumed $D \propto M$ (negligible $< 0.01\%$ on the 5 reported policies, $-13.3\%$ on a non-proportional stress test). Fixed with the exact $(2n - 1)$-dimensional $[Q_{\text{orth}}^T\delta, \omega]$ Lyapunov realization.
3. **Vacuous overdamped `dt` check & droop-share arithmetic (Peer Review #4 M2, M6):** Replaced steady-state COI nadir `dt` check with transient first-swing peak tie-line angle (`|diff| = 1.01e-10`), corrected `63%` → `81.0%` Area A droop share, and explicitly documented normalized dimensionless frequency units.

---

## 3. Verified Scope & Open Problems

- **Model bounds:** Lossless reactive lines ($R/X = 0$, constant $|V_i| = 1$), dimensionless normalized swing parameters, and primary-droop-only active-power saturation (virtual-inertia surges $M_i\dot{\omega}_i$ and total semiconductor current $|I_i|$ are unclipped).
- **High-value next entries (`002+`):**
  1. **Full Current-Limited GFM + Reactive $Q\text{--}V$ Coupling & Explicit GFL PLL Instability:** Clip total inverter current $|I_i| \le I_{\max}$ (including first-swing virtual inertia!) and model voltage-angle coupling / weak-grid PLL loss of synchronism.
  2. **Braess-Proof Topology Screening & Adaptive Virtual-Impedance Mitigation:** Design local inverter virtual-reactance control laws that dynamically relieve Braess phase frustration on stressed meshed corridors.
