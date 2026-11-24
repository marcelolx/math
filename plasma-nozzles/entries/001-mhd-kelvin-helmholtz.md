# 001 — 2D Resistive-MHD Kelvin–Helmholtz Shear Instability: Low-$R_m$ Sub-Alfvénic Breakdown & $\mathrm{Re}$-Scaling Audit

- **Date:** 2026-11-23
- **Status:** `verified` (hardened after Peer Review #3)
- **Verification Script:** `../sim/exp001_mhd_kelvin_helmholtz.py`

## 1. Target

Entry `000` showed that magnetic nozzle thrust efficiency peaks when the plasma accelerates through the **Alfvén detachment surface ($M_A = u/v_A = 1$)** before dipole field-line flare bends the exhaust off-axis. However, plasma-nozzle plumes contain velocity shear layers between the fast core plasma and slower edge/ambient plasma. In ideal MHD ($R_m \to \infty$), Chandrasekhar's classic theorem states that a parallel magnetic field completely stabilizes incompressible Kelvin–Helmholtz (KH) shear whenever $M_A < 1$.

In this session we build a **2D pseudospectral nonlinear incompressible resistive-MHD PDE solver** (2/3-dealiased FFT + exact Integrating-Factor RK4) cross-checked against a **1D Fourier-collocation linearized MHD eigenvalue solver**, to test sub-Alfvénic shear stability ($M_A < 1$) at **finite magnetic Reynolds number $R_m = U_0 a / \eta \in [0.05, 300]$**, and rigorously separate **$\mathrm{Re}$-independent physical instability boundaries** from **$\mathrm{Re}$- and box-dependent dissipative continuum modes**.

## 2. The Math & Assumptions

### 2D Incompressible Resistive MHD (Vorticity–Flux Formulation)

On a periodic domain $(x, y) \in [0, 2\pi) \times [0, L_y)$ with uniform background axial field $\mathbf{B}_0 = B_0\hat{x}$ ($B_0 = U_0 / M_A$), writing $\mathbf{u} = (\partial_y\psi, -\partial_x\psi)$ and $\mathbf{B} = B_0\hat{x} + (\partial_y a_z, -\partial_x a_z)$ enforces $\nabla\cdot\mathbf{u} = 0$ and $\nabla\cdot\mathbf{B} = 0$ **identically**:

$$\frac{\partial\omega}{\partial t} + \mathbf{u}\cdot\nabla\omega = \mathbf{B}\cdot\nabla J + \nu\nabla^2\omega, \qquad \frac{\partial a_z}{\partial t} + \mathbf{u}\cdot\nabla a_z + B_0 u_y = \eta\nabla^2 a_z$$

where $\omega = -\nabla^2\psi$ and $J = -\nabla^2 a_z$. Base double-tanh shear layer (half-width $a = 0.25$, $U_0 = 1$, base $\mathrm{Re} = U_0 a / \nu = 250$):
$$U_x(y) = U_0\left[\tanh\!\left(\frac{y - L_y/4}{a}\right) - \tanh\!\left(\frac{y - 3 L_y/4}{a}\right) - 1\right]$$

- **Exact total MHD energy balance** (for unforced evolution):
  $$E_{\text{tot}} = \frac{1}{2}\langle |\mathbf{u}|^2 + |\mathbf{B}|^2 \rangle, \qquad \frac{dE_{\text{tot}}}{dt} = -\nu\langle \omega^2 \rangle - \eta\langle J^2 \rangle$$
- **Exact Integrating-Factor RK4 (IFRK4):** Diagonal Fourier diffusion factors $e^{-\nu k^2 \Delta t}$ and $e^{-\eta k^2 \Delta t}$ are integrated analytically inside RK4, eliminating explicit diffusive stiffness down to $R_m = 0.05$ ($\eta = 5.0$).
- **Linearized Fourier-collocation eigenproblem** ($k_x a = 0.25$ baseline, plus $k_x a \in \{0.125, 0.50\}$ sweep):
  $$s\hat{\omega} = -ik_x U\hat{\omega} - ik_x U_{yy} L^{-1}\hat{\omega} + ik_x B_0 L\hat{a}_z - \nu L\hat{\omega}, \qquad s\hat{a}_z = -ik_x U\hat{a}_z + ik_x B_0 L^{-1}\hat{\omega} - \eta L\hat{a}_z$$
  where $L = k_x^2 - \partial_{yy}$.
- **Quasi-static low-$R_m$ Stuart-number scaling:** At low $R_m$, induced current obeys $\eta\nabla^2 a_z \approx B_0 u_y$, turning the Lorentz force into a quasi-static magnetic friction $\propto B_0^2 k_x^2 / (\eta k^2)$ whose ratio to shear advection $U_0 / a$ is the **Stuart / magnetic interaction parameter** $N = R_m / M_A^2$. Thus sub-Alfvénic KH breakdown occurs below a critical Stuart number $R_{m,\text{crit}} = N_c(k_x a)\,M_A^2$, independent of $\mathrm{Re}$ as $\mathrm{Re} \to \infty$ but dependent on wavelength $k_x a$!

### Intentional Simplifications
- Incompressible 2D resistive MHD (no compressibility, no Hall term, constant parallel $B_0$); isolates the competition between shear advection, magnetic tension, and resistive diffusion.
- Linear/forced-shear runs add static $-\nu\nabla^2\omega_0$ forcing to hold the base shear profile steady; unforced energy conservation is verified separately with forcing switched off.

## 3. The Verification

Reproduce from the repo root:

```bash
uv run python plasma-nozzles/sim/exp001_mhd_kelvin_helmholtz.py
```

### Experiment 1 — Discretized Collocation Eigenvalues vs. 2D Nonlinear MHD PDE ($64\times 64$, IFRK4, $\mathrm{Re}=250$)

| Regime | $M_A$ | $R_m$ | Collocation Eigenvalue $\gamma$ | 2D Nonlinear PDE $\gamma$ | $|\Delta\gamma|$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Super-Alfvénic | $2.50$ | $200.0$ | $+0.549219$ | $+0.549476$ | $2.6\times 10^{-4}$ |
| Near-cutoff super-Alfvénic | $1.25$ | $200.0$ | $+0.188527$ | $+0.188518$ | $9.3\times 10^{-6}$ |
| Sub-Alfvénic high-$R_m$ | $0.75$ | $200.0$ | $-0.008192$ | $-0.008195$ | $3.1\times 10^{-6}$ |
| Sub-Alfvénic box trough | $0.75$ | $4.0$ | $-0.113142$ | $-0.113143$ | $1.0\times 10^{-6}$ |
| **Sub-Alfvénic low-$R_m$ KH** | $0.75$ | $0.20$ | **$+0.309875$** | **$+0.309211^*$** | $6.6\times 10^{-4}$ |

- **Exact unforced nonlinear 2D MHD energy conservation** ($M_A = 1.5, R_m = 50, \mathrm{Re} = 100$, finite amplitude `0.05`, no base forcing): relative error $\mathbf{4.31\times 10^{-8}}$.
- *\*Grid convergence check ($M_A = 0.75, R_m = 0.20$): $64\times 64$ gives $+0.309211$ vs. $128\times 128$ giving $+0.309902$ (matching `ny_eig = 128` collocation $+0.309903$ to $10^{-6}$).*

### Experiment 2A & 2B — Robust $R_{m,\text{crit}} \propto M_A^2$ Breakdown Law vs. $\mathrm{Re}$-/Box-Dependent Continuum Trough

At fixed $k_x a = 0.25$ ($\mathrm{Re} = 250, L_y = 2\pi$):

| Sub-Alfvénic $M_A$ | Unstable Breakdown $R_{m,\text{crit}}$ | Critical Stuart $N_c = R_{m,\text{crit}}/M_A^2$ | $\mathrm{Re}=250, L_y=2\pi$ Box Trough $\gamma_{\min}$ ($R_m^*$) |
| :---: | :---: | :---: | :---: |
| $0.50$ | $0.2261$ | **$0.904$** | $-0.2845$ ($R_m^* = 1.19$) |
| $0.60$ | $0.3288$ | **$0.913$** | $-0.2540$ ($R_m^* = 1.36$) |
| **$0.75$** | **$0.5348$** | **$0.951$** | $-0.2209$ ($R_m^* = 1.59$) |
| $0.85$ | $0.7297$ | **$1.010$** | $-0.1984$ ($R_m^* = 1.80$) |
| $0.92$ | $0.9201$ | **$1.087$** | $-0.1654$ ($R_m^* = 2.29$) |

**Peer Review #3 $\mathrm{Re}$, Box-Size ($L_y$), and Wavenumber ($k_x a$) Audit ($M_A = 0.75$):**

| Audit Case | $R_{m,\text{crit}}$ | $N_c = R_{m,\text{crit}}/M_A^2$ | Box Trough $\gamma_{\min}$ ($R_m^*$) | Bulk Continuum Scaling $-2 k_x B_0 \sqrt{R_m^*/\mathrm{Re}}$ |
| :--- | :---: | :---: | :---: | :---: |
| $\mathrm{Re} = 250, L_y = 2\pi$ ($k_x a = 0.25$) | $0.5348$ | $0.951$ | $-0.2210$ ($1.59$) | $-0.2124$ |
| $\mathrm{Re} = 1000, L_y = 2\pi$ ($k_x a = 0.25$) | $0.5926$ | $1.054$ | **$-0.1445$** ($2.83$) | $-0.1418$ |
| $\mathrm{Re} = 4000, L_y = 2\pi$ ($k_x a = 0.25$) | $0.5910$ | $1.051$ | **$-0.0948$** ($4.96$) | $-0.0939$ |
| $\mathrm{Re} = 250, L_y = 4\pi$ ($k_x a = 0.25$) | $0.4593$ | $0.816$ | **$-0.1706$** ($1.00$) | $-0.1686$ |
| **$k_x a = 0.125$** (longer wavelength, $\mathrm{Re}=250$) | **$0.8289$** | **$1.474$** | — | — |
| **$k_x a = 0.500$** (shorter wavelength, $\mathrm{Re}=250$) | **$0.3207$** | **$0.570$** | — | — |

- **Key audit conclusion:** The low-$R_m$ breakdown law **$R_{m,\text{crit}} = N_c(k_x a)\,M_A^2$** is **robust and $\mathrm{Re}$-independent** ($N_c \approx 0.95\text{--}1.05$ across $\mathrm{Re} \in [250, 4000]$ at $k_x a = 0.25$, and $0.82$ at $L_y = 4\pi$), with $N_c(k_x a)$ increasing at longer wavelengths ($1.47$ at $k_x a = 0.125$ vs. $0.57$ at $k_x a = 0.50$). Conversely, the intermediate-$R_m$ damping trough is a **delocalized bulk Alfvén-continuum box mode** balancing viscous damping $\nu k^2$ against magnetic friction $k_x^2 B_0^2 / (\eta k^2)$ ($\gamma_{\min} \approx -2 k_x B_0 \sqrt{R_m^* / \mathrm{Re}}$): its depth more than halves from $\mathrm{Re} = 250$ to $\mathrm{Re} = 4000$, vanishes as $\mathrm{Re} \to \infty$, and shifts with box height $L_y$.

![Low-Rm Sub-Alfvenic KH Breakdown and Re-Scaling Audit](../results/exp001_mhd_kh_stability.png)

### Experiment 3 — Full 2D Nonlinear Vortex Roll-Up ($96\times 96$, $\mathrm{Re} = 250, L_y = 2\pi, t = 18$, `pert_amp = 0.015`)

| Nonlinear Case ($\mathrm{Re}=250, 2\pi$ box) | $(M_A, R_m)$ | Peak $\frac{1}{2}\langle u_y^2\rangle$ | Final $\frac{1}{2}\langle u_y^2\rangle$ ($t=18$) | Max $\mathrm{rms}(u_y)$ | Physical Outcome |
| :--- | :---: | :---: | :---: | :---: | :--- |
| Super-Alfvénic | $(2.50, 200)$ | $1.94\times 10^{-2}$ | $1.50\times 10^{-2}$ | $0.1972$ | Violent KH roll-up + field-line folding |
| Sub-Alfvénic High-$R_m$ | $(0.75, 200)$ | $9.78\times 10^{-6}$ | $1.05\times 10^{-7}$ | $0.0044$ | Elastic Alfvén oscillation, slow viscous decay |
| Sub-Alfvénic Box Trough | $(0.75, 4.0)$ | $8.77\times 10^{-6}$ | $2.62\times 10^{-10}$ | $0.0042$ | Faster finite-$\mathrm{Re}$ box-mode decay ($\mathrm{Re}/L_y$-dependent) |
| **Low-$R_m$ Sub-Alfvénic KH** | **$(0.75, 0.20)$** | **$5.39\times 10^{-3}$** | **$1.70\times 10^{-3}$** | **$0.1038$** | **Full nonlinear KH vortex roll-up through straight field lines!** |

![2D Nonlinear Vorticity and Magnetic Field Lines Across Four MHD Regimes](../results/exp001_mhd_kh_nonlinear.png)

Full metrics: [`../results/exp001_metrics.json`](../results/exp001_metrics.json).

## 4. Verdict & Dead Ends

- **Artifact caught & corrected by Peer Review #3 (C2 & M3):**
  - Our earlier claim that the intermediate-$R_m$ trough represents a universal "19–34× Hartmann braking stabilizer" ($400\times$ cleaner mixing) was an **over-interpretation of a finite-$\mathrm{Re}$, finite-box continuum mode**. Independent and reproduced audits prove the trough obeys $\gamma_{\min} \approx -2 k_x B_0 \sqrt{R_m^* / \mathrm{Re}}$, halving from $\mathrm{Re} = 250$ to $\mathrm{Re} = 4000$ and vanishing in the inviscid limit $\mathrm{Re} \to \infty$.
- **What survives rigorously:**
  1. **$\mathrm{Re}$-Independent Sub-Alfvénic Resistive KH Breakdown Law ($R_{m,\text{crit}} = N_c(k_x a)\,M_A^2$):** Below the critical Stuart number $N = R_m / M_A^2 < N_c(k_x a)$ ($N_c \approx 0.90\text{--}1.09$ at $k_x a = 0.25$ across $\mathrm{Re} \in [250, 4000]$; $1.47$ at $k_x a = 0.125$; $0.57$ at $k_x a = 0.50$), magnetic diffusion decouples the plasma from the parallel magnetic field and unleashes full 2D nonlinear KH vortex roll-up **inside the sub-Alfvénic ($M_A < 1$) nozzle** right through nearly straight magnetic field lines ($1.6\times 10^4\times$ higher final transverse mixing energy than high-$R_m$ ideal MHD at $M_A = 0.75, R_m = 0.20$).
  2. **Long-wavelength vulnerability:** Because $N_c(k_x a)$ grows at smaller $k_x a$ ($1.47$ at $k_x a = 0.125$), long-wavelength plume shear modes break down at higher $R_m$ than the nominal $k_x a = 0.25$ mode.
- **Next single hypothesis tested in [`002`](002-self-consistent-nozzle.md):** Upgrade Entry `000`'s rigid dipole nozzle to a compliant flux-tube MHD nozzle obeying $p + B^2/2 = B_{\text{vac}}(z)^2/2$ to quantify diamagnetic plume inflation and Roberts–Webb tube-speed choking.
