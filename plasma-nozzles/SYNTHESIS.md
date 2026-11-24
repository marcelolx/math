# Synthesis — Plasma Nozzles & MHD Propulsion (`000`–`002`)

This monograph synthesizes three verified open-notebook entries (`000`–`002`), audited by an independent cross-model peer review (Peer Review #3, Claude Opus 5.5, 11 independent spot checks), progressing from a rigid quasi-1D dipole magnetic nozzle baseline to 2D nonlinear resistive-MHD plume shear stability and a compliant-flux-tube MHD nozzle with diamagnetic inflation and Roberts–Webb tube-speed choking.

---

## 1. Consolidated Verified Design Laws

| # | Design Law / Structural Result | Quantitative Rule (Verified in PDE & Peer-Reviewed) | Source Entry |
| :-: | :--- | :--- | :---: |
| **1** | **Alfvén Detachment vs. Dipole Flare Optimum ($\beta_0^*$)** | Net axial thrust $F_{\text{net}}(\beta_0) = \eta_{\text{div}}(z_A) F_{\text{stream}}(z_A)$ peaks at a sharp optimal throat beta balancing premature detachment ($\beta_0 \gg \beta_0^*$) against dipole flare divergence ($\beta_0 \ll \beta_0^*$): $+9.2\%$ adiabatic ($\gamma=5/3, \beta_0^*=0.156$) vs. **$+36.8\%$** near-isothermal ($\gamma=1.12, \beta_0^*=0.0043$) at $L_n = 2.5 R_0$. | [`000`](entries/000-baseline.md) |
| **2** | **Sub-Alfvénic Resistive KH Breakdown Law** | Inside the sub-Alfvénic nozzle ($M_A < 1$), where ideal MHD is linearly stable, finite resistivity unleashes full 2D nonlinear KH vortex roll-up below the $\mathrm{Re}$-independent Stuart-number boundary **$R_{m,\text{crit}} = N_c(k_x a)\,M_A^2$** ($N_c \approx 0.90\text{--}1.09$ at $k_x a = 0.25$ across $\mathrm{Re}\in[250, 4000]$; $N_c = 1.47$ at longer wavelength $k_x a = 0.125$, $0.57$ at $k_x a = 0.50$). | [`001`](entries/001-mhd-kelvin-helmholtz.md) |
| **3** | **Exact Compliant-Flux-Tube MHD Energy Law & $c_T$ Choking** | Under fixed external vacuum pressure $p + B^2/2 = B_{\text{vac}}(z)^2/2$ and flux freezing $BA = \Phi_B$, total energy $\mathcal{E}_{\text{tot}} = (\frac{1}{2}\rho u^2 + \frac{p}{\gamma-1} + \frac{B^2}{2} + \frac{B_{\text{vac}}^2}{2})A$ is **strictly conserved with zero source term** (verified symbolically, $10^{-14}$ closed-tube, and 2nd-order MUSCL-HLL PDE to $1.3\times 10^{-5}$), primitive recovery is a closed-form quadratic, and the throat chokes at the Roberts–Webb tube speed $u_0 = c_{T0} < c_{s0}$. | [`002`](entries/002-self-consistent-nozzle.md) |
| **4** | **Far-Field Diamagnetic Inflation & Detachment Beta Identity** | Downstream plasma pressure inflates the plume area as **$A(z) \propto B_{\text{vac}}(z)^{-2/\gamma}$** (vs. rigid $B_{\text{vac}}^{-1}$). At $M_A = 1$, $\beta_{\text{in}}(z_{\text{det}}) = 2/(\gamma M_{\text{det}}^2) \ll 1$ for supersonic detachment ($M_{\text{det}} \sim 5\text{--}6$): thus adiabatic low-$\beta$ optima shift by only $-0.01\text{ to }-0.04$ pts, while near-isothermal plume flare costs $-0.58\text{ to }-1.49$ pts ($\theta: 40.9^\circ \to 42.0^\circ$). | [`002`](entries/002-self-consistent-nozzle.md) |
| **5** | **High-$\beta$ Same-$H_0$ Magnetic Gain Erosion** | Under a rigorous same-stagnation-enthalpy ($H_0$) sonic comparison, compliant magnetic gain never goes negative, but above $\beta_{\text{vac},0} \gtrsim 0.50$, upstream detachment + tube choking erode net magnetic augmentation nearly to zero ($+0.02\%$ vs. rigid $+3.70\%$ at $\beta = 0.78, \gamma=5/3$; $+0.49\%$ vs. rigid $+11.03\%$ at $\beta = 0.77, \gamma=1.12$). | [`002`](entries/002-self-consistent-nozzle.md) |

---

## 2. Documented Dead Ends & Peer-Review Corrections

1. **Finite-$\mathrm{Re}$ / finite-box intermediate-$R_m$ trough misattribution (Peer Review #3 C2):** The deep damping trough at $R_m \sim 1.5\text{--}5$ in a $2\pi$ box at $\mathrm{Re} = 250$ is a delocalized bulk Alfvén-continuum box mode scaling as $\gamma_{\min} \approx -2 k_x B_0 \sqrt{R_m^* / \mathrm{Re}}$, which weakens by $>2\times$ at $\mathrm{Re} = 4000$ and vanishes as $\mathrm{Re} \to \infty$—not a universal shear-layer stabilizer.
2. **Unequal stagnation-enthalpy ($H_0$) sonic normalization artifact (Peer Review #3 C1):** Comparing rigid ($u_0 = c_{s0}$) and compliant ($u_0 = c_{T0} < c_{s0}$) nozzles at equal throat static $(p_0, \rho_0)$ against a single rigid sonic reference penalized the compliant nozzle for having up to $19\%$ less stagnation enthalpy, creating a spurious `< 0%` negative gain. Re-normalizing to a same-$H_0$ sonic reservoir eliminated the artifact and exposed the true near-zero erosion curve.
3. **Symmetric double-layer initial-condition blind spot (Entry `001`):** An exactly symmetric perturbation across the two shear layers projected zero energy onto the fastest varicose mode ($+0.310$), locking onto the slower sinuous mode ($+0.235$); caught by eigenvalue cross-check.

---

## 3. Verified Scope & Open Problems

- **Key model bounds:** Entries `000` and `002` assume paraxial/slender-tube geometry (exceeded at near-isothermal optima where $\theta_{\text{edge}} = 42^\circ\text{--}53^\circ$), abrupt ballistic detachment at $M_A = 1$ (discarding residual post-detachment thermal enthalpy), and fixed external $B_{\text{vac}}(z)$ without 2D Grad–Shafranov exterior compression. Entry `001` assumes incompressible 2D resistive MHD without Hall physics or axial expansion.
- **High-value next entries (`003+`):**
  1. **2D Axisymmetric Self-Consistent Ideal/Resistive MHD Nozzle (Grad–Shafranov / 2D FV MHD):** Include exterior magnetic-field back-reaction and gradual resistive/inertial detachment rather than abrupt $M_A = 1$ cutoff.
  2. **Compressible / Hall-MHD Plume Shear & Drift-Alfvén Instabilities:** Extend Entry `001` to finite-$\beta$ compressible shear layers and ion-inertial-scale Hall physics relevant to VASIMR / helicon plumes.
