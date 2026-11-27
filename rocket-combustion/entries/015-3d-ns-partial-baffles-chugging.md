# 015 — Full 3D Cylindrical Compressible Navier–Stokes: Partial-Span Shroud-Ring Baffles, Mass-Conserving Cavities & the 3D Transverse $\to$ Chugging $\to$ $1L$ Mode-Migration Cascade

- **Date:** 2026-11-27
- **Status:** `verified`
- **Verification Tier:** `🟡 Verified (Pending Audit)`
- **Verification Script:** [`../sim/exp015_3d_ns_baffles_bell.py`](../sim/exp015_3d_ns_baffles_bell.py)
- **Metrics JSON:** [`../results/exp015_metrics.json`](../results/exp015_metrics.json)
- **Peer Review Receipt:** *(pending independent cross-model Peer Review #12)*

---

## 1. Target

Upgrade Entry `014`'s 3D cylindrical conical-frustum compressible Euler solver to **full 3D conservative compressible Navier–Stokes ($\mathrm{Re}$ fixed, $\mathrm{Pr} = 0.72$) with 3D partial-span radial baffles and mass-conserving faceplate Helmholtz cavities**, directly resolving four open scientific questions left by Peer Review #11:

1. **Fixed-$\mathrm{Re}$ 3D NS Convergence & Meridional vs. Viscous Bell Split:** Separate physical viscosity ($\mathrm{Re} = 500\text{--}1000$) from inviscid 3D meridional bell damping across straight cylinders (against an exact 1D Bessel-NS Chebyshev eigen-solver) and true-3D smooth curved bells (`40×12×48` & `56×16×48`).
2. **3D Partial-Span Shroud-Ring ($r \in [r_b, 1]$) vs. Full-Span ($r \in [0, 1]$) Radial Baffle Law:** Quantify how cutting out the inner hub of $N_b = 5, L_b = 0.45$ radial baffles ($r_b \in [0, 0.75]$) trades blade metal area (`1 - r_b`) against $1T, 2T, 1R$ damping and frequency downshift while leaving the hub unobstructed for a `0.35%` micro-hub $1R$ cavity.
3. **Root Cause & Cure of Entry `014`'s $t = 120$ $k = 0$ Bulk Drift (`e_mean`):** Disentangle non-conservative cavity mass rectification ($\langle\rho' u_{\text{port}}\rangle > 0$ vs. exact mass-conserving $\dot{m}$ coupling) from subcritical low-frequency **$k = 0$ chugging** ($\omega_{\text{chug}} \approx 0.11\text{--}0.16, T \approx 39$) at $\omega_{\text{hp}} = 0.15$ vs. stiff feed-system cutoff $\omega_{\text{hp}} = 0.50$.
4. **Full 3D NS All-Mode (`spin_1T, stand_1T, spin_2T, 1R, 1L`) Nonlinear Bomb Audit ($t = 120$):** Test the combined `Re = 500` + partial-span $r_b = 0.42$ baffles + `3.0%` retuned mass-conserving Dual-Zone faceplate in both straight cylinders (`20×12×50`, refined `20×16×50`) and true-3D smooth curved bell nozzles (`36×12×50`).

---

## 2. The Math, Assumptions & Checked Scope

### 2.1 Conservative 3D Cylindrical Compressible Navier–Stokes & Adiabatic 3D Baffles

We augment Entry `014`'s 3D conical-frustum finite-volume Euler equations ($\gamma = 1.25$, $\mathrm{Pr} = 0.72$, constant eddy viscosity $\mu = 1/\mathrm{Re}$, Stokes hypothesis $\lambda = -\frac{2}{3}\mu$, thermal conductivity $\kappa = \frac{\gamma}{\mathrm{Pr}(\gamma-1)\mathrm{Re}}$) with full 3D cylindrical viscous stress tensor $\boldsymbol{\tau}$ and Fourier heat flux $\mathbf{q} = -\kappa \nabla T$ ($T = p/\rho$) across all three face families ($x$-faces, slanted conical $\eta$-faces projected onto $(n_{x\eta}, n_{r\eta})$, and azimuthal $\theta$-faces), conserving axial angular momentum $r\rho u_\theta$ via face-radius-weighted viscous torques ($r_{xf}\tau_{x\theta}, r_{ef}\tau_{n\theta}, r_{\theta f}\tau_{\theta\theta}$) and radial viscous hoop stress $-\tau_{\theta\theta}\,\text{hoop}$.

- **Curved-Cylinder Stress-Free Outer Wall Correction (Derived & Verified):** On a curved outer cylinder ($R = 1$) with stress-free boundary ($\tau_{r\theta}|_{r=1} = 0$), an irrotational tangential acoustic wave ($\mathbf{u} = \nabla\phi$) carries nonzero interior shear $\tau_{r\theta,\text{irrot}}|_{r=1} = -2\mu u_\theta/R \neq 0$, shifting the leading-order viscous decay rate from bulk Stokes–Kirchhoff by:

  $$\alpha_{\text{SF,3D}}(m, s_{mn}) = -\frac{s_{mn}^2}{2\,\mathrm{Re}}\!\left(\frac{4}{3} + \frac{\gamma - 1}{\mathrm{Pr}}\right) + \frac{2m^2}{\mathrm{Re}\,(s_{mn}^2 - m^2)}.$$

  For axisymmetric modes ($m = 0$, e.g., $1R$), the curvature shift is identically zero; for $1T$ ($m = 1$), it reduces bulk SK by `+29.4%` (`-0.00950 → -0.00671` at $\mathrm{Re} = 300$).
- **3D Partial-Span Radial Baffles ($N_b = 5, L_b = 0.45, r \in [r_b, 1]$):** Implemented via two-sided mirror HLLC on azimuthal $\theta$-faces with zero cross-wall MUSCL slopes, wall-aware one-sided $\theta$-differences for viscous cross-derivatives (Peer Review #10 M1 lesson), and zero viscous energy flux across adiabatic slip or no-slip rigid blades (`mass ~ 1.1–2.3e-16`, `energy = 1.42e-16`).
- **Exact Mass-Conserving Faceplate Cavity Coupling:** Solving the cavity ODE for mass flux $\dot{m}_k$ (instead of volume velocity $u_{\text{port}}$) and enforcing $\rho_g u_g + \sum_k \dot{m}_k W_k = \bar{M}_c$ at the choked inlet eliminates quadratic acoustic-absorption mass leakage $\langle \rho' u_{\text{port}}\rangle > 0$ and conserves total closed chamber + cavity mass to **`1.88e-16`**.
- **Explicit Scope Limits:** Constant-eddy-viscosity laminar/mean-turbulent closure ($\mathrm{Re} = 500\text{--}1000$, adiabatic free-slip outer bell wall); reacting baffles in Exp 2 & Exp 4 use adiabatic slip walls (no-slip wake conservation is verified in Exp 1B, while full 3D no-slip baffle wake relaxation remains for future work); Exp 4 reacting runs use `n_p = 0.34` with `TAU_BAR = 0.16` and `n_v = 0`.

---

## 3. The Verification

Reproduce all experiments, metrics, and figures with:

```bash
uv run python rocket-combustion/sim/exp015_3d_ns_baffles_bell.py
```

### 3.1 Experiment 1: 3D NS Conservation, Curved-Wall Stokes–Kirchhoff & Fixed-$\mathrm{Re}$ Convergence

1. **Machine-Precision 3D NS Conservation (`t = 2.0–4.0`, $\mathrm{Re} = 500$):**
   - Quiescent smooth 3D NS bell (`20×10×32`): drift = **`1.37e-18`**.
   - Periodic 3D wavy bell NS bare (`16×10×32`): `mass = 1.14e-16`, `ang_mom = 1.78e-18`, `energy = 1.42e-16`.
   - Periodic 3D wavy bell NS + 5 adiabatic partial-span baffles (`r_b = 0.42`, `16×10×30`): **slip** `mass = 1.14e-16, energy = 1.42e-16`; **no-slip** `mass = 2.28e-16, energy = 1.42e-16`.
   - Closed 3D cylinder + mass-conserving cavity (`12×10×32`, $A_0 = 0.04$): total chamber + cavity mass error = **`1.88e-16`**.
2. **3D Curved-Cylinder Stress-Free Stokes–Kirchhoff Attenuation (`14×48`, net of `Re = ∞` numerical baseline):**
   - **$1T$ ($m = 1$):** `Re = 300`: `-0.00634` vs. exact $\alpha_{\text{SF,3D}} = -0.00671$ (`5.48%`); `Re = 600`: `-0.00318` vs. `-0.00335` (`5.04%`).
   - **$1R$ ($m = 0$):** `Re = 300`: `-0.03903` vs. exact $\alpha_{\text{SK}} = -0.04112$ (`5.08%`); `Re = 600`: `-0.01953` vs. `-0.02056` (`5.02%`), confirming the `4.33×` stronger bulk SK scaling ($s_{02}^2 / s_{11}^2$) and uniform `~5%` second-order central-difference dispersion on `nr = 14`.
3. **Fixed-$\mathrm{Re}$ Straight-Cylinder & True-3D Smooth-Bell Convergence (Resolving Entry `014` C1/C2):**
   - **Straight cylinder `1T` ($\bar{M}_c = 0.12$):** At fixed $\mathrm{Re} = 500$, exact 1D Bessel-NS Chebyshev (`-0.09392`, $\omega = 1.825$) matches 3D NS PDE cleanly across axial refinement: `nx = 24`: **`-0.09776` (`4.1%`)**, `nx = 48`: **`-0.09907` (`5.5%`)** (and at $\mathrm{Re} = 1000$: Chebyshev `-0.09236` vs. `nx = 48` `-0.09996`, `8.2%`).
   - **True-3D smooth curved bell ($L_n = 0.60, L_d = 0.45$, `rhs = 1.19e-7` on `56×16`, Sauer/Hall throat `0.811 → 1.237`, exit `1.661–1.701`):**
     - `Re = ∞` (`40×12×48`): **`-0.04365`**
     - `Re = 1000` (`40×12×48`): **`-0.04776`** (`+9.4%`)
     - `Re = 500` (`40×12×48`): **`-0.05223`** (`+19.7%`) → refined `56×16×48`: **`-0.05011`** (`4.1%` grid spread).
   - **Clean physical split:** `~85%` of the true-3D smooth bell's cold $1T$ damping (`-0.0437`) comes from inviscid 3D meridional bell geometry, while fixed physical viscosity at $\mathrm{Re} = 500$ adds a converged `+15–20%` (`-0.0065` to `-0.0086`) on top.

### 3.2 Experiment 2: 3D Partial-Span Shroud-Ring vs. Full-Span Radial Baffle Law

![3D partial-span radial baffle law and frequency shift vs radial span](../results/exp015_3d_partial_span_baffle_law.png)

Testing $N_b = 5, L_b = 0.45$ radial baffles across inner cutout radius $r_b \in \{0.00, 0.25, 0.42, 0.58, 0.75\}$ at fixed $\mathrm{Re} = 500$ (`24×12×50`, bare `1T = -0.09785` [$\omega = 1.835$], `2T = -0.12975` [$\omega = 3.052$], `1R = -0.18577`):

| Inner Cutout $r_b / R$ | Blade Metal Area | $1T$ $\alpha$ ($\Delta\alpha$ / `%` Full) | $1T$ Shift $\Delta\omega_{1T}$ | $2T$ $\alpha$ ($\Delta\alpha$ / `%` Full) | $2T$ Shift $\Delta\omega_{2T}$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **`0.00`** *(full-span)* | `100.0%` | `-0.15481` (`-0.0570`, `100%`) | `-16.6%` (`1.531`) | `-0.16801` (`-0.0383`, `100%`) | `+9.3%` (`3.337`) |
| `0.25` | `75.0%` | `-0.14574` (`-0.0479`, `84.1%`) | `-9.6%` (`1.659`) | `-0.18687` (`-0.0571`, **`149%`**) | `+10.3%` (`3.366`) |
| **`0.42`** *(shroud-ring)* | **`58.3%`** | **`-0.13540`** (`-0.0376`, **`65.9%`**) | **`-6.4%` (`1.717`)** | **`-0.22819`** (`-0.0984`, **`257%`**) | `+10.9%` (`3.386`) |
| **`0.58`** *(outer ring)* | **`41.7%`** | `-0.12459` (`-0.0267`, `47.0%`) | `-3.7%` (`1.767`) | **`-0.24088`** (`-0.1111`, **`290%`**) | **`-10.0%` (`2.748`)** |
| `0.75` | `25.0%` | `-0.11407` (`-0.0162`, `28.5%`) | `-1.6%` (`1.806`) | `-0.19729` (`-0.0675`, **`176%`**) | `-3.2%` (`2.954`) |

- **Headline 3D discovery (Inner-Tip Vortex-Shedding Boost on $2T$):** While $1T$ baffle damping scales super-linearly with blade metal area (`84.1%` retained at `75%` metal; `65.9%` at `58.3%` metal), **cutting out the inner hub (`r_b = 0.42–0.58`) boosts $2T$ damping by `2.57–2.90×` over a full-span baffle (`-0.0984` to `-0.1111` vs. `-0.0383`) using `42–58%` less metal!** Full-span blades seal closed pie-slice wedges that only shed vorticity from the downstream axial tip $x = L_b$, whereas partial-span shroud-ring blades expose a second **inner radial tip edge ($r = r_b, x \in [0, L_b]$)** directly inside the high-velocity $2T$ annular zone. Meanwhile, axisymmetric $1R$ is completely unperturbed (`-0.18575` full, `-0.18576` shroud-ring vs. `-0.18577` bare), leaving $r < 0.42$ wide open for the `0.35%` micro-hub $1R$ cavity. Note the discrete $2T$ FFT branch jump between $r_b = 0.42$ (`3.386`, $+10.9\%$) and $r_b = 0.58$ (`2.748`, $-10.0\%$) as compartment confinement opens.

### 3.3 Experiment 3: Root Cause & Cure of Entry `014`'s $t = 120$ $k = 0$ Bulk Drift

![Root cause and cure of k=0 bulk chugging drift at t=120](../results/exp015_mass_conserving_cavity_cure.png)

1. **Exact 1D Chebyshev Chugging & $1L$ Eigenvalues (`Exp 3A`, $\bar{M}_c = 0.12$):**
   - At $\omega_{\text{hp}} = 0.15$: the low-frequency **$k = 0$ chugging mode** ($\omega_{\text{chug}} = 0.110\text{--}0.114$) sits dangerously close to the imaginary axis (`-0.0197` bare / `-0.0213` with $\beta = 3\%$ at $n_p = 0.30$; **`-0.0068` bare / `-0.0088` with $\beta = 3\%$ at $n_p = 0.34$**), where finite-amplitude perturbations readily subcritically trigger a $T \approx 39$ bulk chugging cycle!
   - Stiffening the feed-system cutoff to **$\omega_{\text{hp}} = 0.50$** pushes the $k = 0$ chugging root **`9–20×` deeper into the stable half-plane** (`-0.1882` at $n_p = 0.30$; **`-0.1753`** at $n_p = 0.34$), while leaving the $1L$ frequency ($\omega_{1L} \approx 2.04\text{--}2.12$) intact (bare $1L = +0.0461 / +0.0630$; $\beta = 3\%$ at $\omega_H = 1.84$: `-0.0267 / -0.0085`).
2. **3D $t = 120$ Head-to-Head PDE Audit (`20×12×48`, $n_p = 0.30, \mathrm{Re} = \infty$, `Exp 3B`):**

| Case (`3.0%` Dual-Zone, `t = 120`) | $\omega_{\text{hp}}$ | Cavity Coupling | $P_{120}$ | Late $E_{\text{mean}}$ ($k=0$) | Late Mass Drift | Slope `[84, 120]` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `e014_legacy_1T` | `0.15` | Legacy `u_port` | `6.19%` | `3.73e-4` | `-0.667%` | `+0.0016` |
| `e015_mcons_1T` | `0.15` | Mass-cons. `m_dot` | `6.04%` | `2.29e-4` (`1.63×` cut) | `-0.515%` | `-0.0003` |
| **`e015_hp50_mcons_1T`** | **`0.50`** | **Mass-cons. `m_dot`** | **`4.53%`** | **`8.38e-9` (`44,500×` cut!)** | **`+0.031%`** | **`-0.0004`** |
| `e014_legacy_st1T` | `0.15` | Legacy `u_port` | `4.92%` | `2.70e-4` | `-0.821%` | `+0.0010` |
| `e015_mcons_st1T` | `0.15` | Mass-cons. `m_dot` | `4.40%` | `1.07e-4` (`2.52×` cut) | `-0.530%` | `-0.0007` |
| **`e015_hp50_st1T`** | **`0.50`** | **Mass-cons. `m_dot`** | **`2.60%`** | **`5.97e-9` (`45,200×` cut!)** | **`+0.009%`** | **`-0.0055`** |

- **Double diagnosis confirmed:** Mass-conserving $\dot{m}$ coupling alone cuts $E_{\text{mean}}$ `1.63–2.52×` and flips the $n_p = 0.30$ tail slope negative, while pairing it with stiff feed-system cutoff $\omega_{\text{hp}} = 0.50$ **eliminates $k = 0$ bulk chugging by `44,500–45,200×`** and keeps chamber mass drift below `0.03%`!

### 3.4 Experiment 4: Full 3D NS All-Mode Bomb Audit & the Transverse $\to$ Chugging $\to$ $1L$ Cascade

![Full 3D NS all-mode bomb audit and true-3D smooth bell shootout](../results/exp015_3d_ns_allmode_quench.png)

Testing the `Re = 500` 3D Hybrid ($N_b = 5, L_b = 0.45, r_b = 0.42$ shroud-ring baffles + `3.0%` mass-conserving retuned Dual-Zone cavities: `1.85%` at `1.717`, `0.80%` at `3.386`, `0.35%` micro-hub at `3.832`) against bare 3D NS at supercritical $n_p = 0.34, \omega_{\text{hp}} = 0.50, A_0 = 0.04$:

| Run (`Re = 500, n_p = 0.34`) | Grid / $t_{\text{end}}$ | Late $P$ | Late $E_{\text{wave}}$ | $E_{1T}$ | $E_{2T}$ | $E_{1R}$ | $E_{1L}$ | $E_{\text{mean}}$ | Dominant Late State |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `bare_ns_1T` | `20×12×50`, `90` | `10.53%` | `2.05e-3` | `3.92e-3` | `1.70e-5` | `8.86e-7` | `1.92e-9` | `1.26e-6` | Saturated $1T$ |
| `bare_ns_2T` | `20×12×50`, `90` | `8.53%` | `8.71e-4` | `8.77e-32` | `1.66e-3` | `4.27e-8` | `6.80e-9` | `1.26e-6` | Saturated $2T$ |
| `bare_ns_1R` | `20×12×50`, `90` | `8.92%` | `2.02e-3` | `4.73e-37` | `1.14e-36` | `1.21e-6` | **`1.87e-3`** | `6.67e-5` | Transitions to $1L$ |
| `bare_ns_1L` | `20×12×50`, `90` | `9.20%` | `2.21e-3` | `5.39e-37` | `1.33e-36` | `3.20e-10` | **`1.97e-3`** | `7.99e-5` | Saturated $1L$ |
| **`hyb3d_1T`** | `20×12×50`, `120` | **`5.39%`** | **`4.04e-4`** | `7.17e-4` | `8.99e-7` | `3.31e-8` | `4.93e-7` | `1.83e-8` | **`5.08×` energy cut** |
| **`hyb3d_f16_1T`** *(grid check)* | `20×16×50`, `90` | **`5.59%`** | **`4.37e-4`** | `7.81e-4` | `1.07e-6` | `3.86e-8` | `5.96e-8` | `1.18e-8` | Grid-consistent (`8%`) |
| **`hyb3d_st1T`** | `20×12×50`, `120` | `6.38%` | `7.36e-4` | `8.25e-5` | `2.67e-7` | `2.14e-7` | **`6.54e-4`** | `2.21e-5` | **$1T$ cut `48×` → $1L$!** |
| **`hyb3d_2T`** | `20×12×50`, `120` | `6.00%` | `6.50e-4` | `1.07e-7` | `2.25e-5` | `1.75e-7` | **`6.04e-4`** | `2.12e-5` | **$2T$ cut `74×` → $1L$!** |
| **`hyb3d_1R`** | `20×12×50`, `120` | **`0.093%`** | **`2.12e-7`** | `1.51e-32` | `7.53e-33` | **`5.42e-11`** | `2.06e-7` | `5.29e-9` | **`9,524×` cut ($1R$ dead!)** |
| `hyb3d_1L` | `20×12×50`, `120` | `6.65%` | `1.10e-3` | `7.54e-33` | `3.14e-33` | `3.27e-7` | **`1.03e-3`** | `3.29e-5` | $1L$ cut `2.0×` (survives) |
| `hyb3d_hp15_2T` *($\omega_{\text{hp}}=0.15$)* | `20×12×50`, `120` | `7.05%` | `1.82e-3` | `6.11e-13` | **`1.88e-12`** | `1.49e-8` | `1.42e-6` | **`3.60e-3`** | **$2T$ dead → chugging!** |
| **`bell_bare_1T`** *(3D bell)* | `36×12×50`, `80` | **`20.21%`** | **`5.37e-3`** | `1.07e-2` | `1.09e-4` | `5.53e-6` | `1.50e-8` | `7.85e-7` | Severe 3D bell $1T$ |
| **`bell_hyb_1T`** *(3D bell)* | `36×12×50`, `80` | **`7.04%`** | **`6.35e-4`** | `1.18e-3` | `2.50e-6` | `7.40e-8` | `2.61e-9` | `1.33e-7` | **`8.45×` energy cut!** |

- **Headline 3D Multi-Mode Mode-Migration Cascade:**
  1. At aggressive $n_p = 0.34$, the `3.0%` Hybrid + `r_b = 0.42` shroud-ring baffles cuts `spin_1T` energy `5.08×` (straight) and **`8.45×` in the true-3D smooth curved bell nozzle (`20.21% → 7.04%`)**, obliterates $1R$ (`5.42e-11`, `9,524×` total energy drop), and suppresses `stand_1T` (`48×`) and `spin_2T` (`74×` at $\omega_{\text{hp}} = 0.50$; `8.8×10^8×` at $\omega_{\text{hp}} = 0.15$)!
  2. Yet suppressing transverse modes at $\omega_{\text{hp}} = 0.15$ unmasks **$k = 0$ chugging** (`hyb3d_hp15_2T`: `2T = 1.88e-12`, `e_mean = 3.60e-3`), and stiffening to $\omega_{\text{hp}} = 0.50$ eliminates chugging (`45,000×`) only to unmask **longitudinal $1L$ rumble** (`hyb3d_st1T` & `hyb3d_2T` both migrate to $E_{1L} \approx 6.0\text{--}6.5\times 10^{-4}$ after $t \approx 85$, because radial baffles do not damp $1L$ and baffle-retuning the shroud $1T$ cavity downward `1.84 → 1.72` detunes it away from $1L = 2.09$)!

---

## 4. Verdict & Dead Ends

- **What worked:**
  - Built a machine-precision conserving **full 3D cylindrical compressible Navier–Stokes solver** (`quiescent = 1.37e-18`, bare curved-bell `1.14e-16`/`1.78e-18`/`1.42e-16`, 5 adiabatic slip/no-slip 3D baffles `1.14–2.28e-16`/`1.42e-16`, closed chamber+cavity mass `1.88e-16`), derived & verified the curved-cylinder stress-free SK shift (`5.0–5.5%` on `14×48`), and closed Entry `014` C1/C2 at fixed $\mathrm{Re} = 500$ (`4.1–5.5%` vs. 1D Bessel-NS Chebyshev; `~85%` meridional + `~15–20%` viscous bell damping split).
  - Discovered the **3D Partial-Span Shroud-Ring Baffle Law**: cutting out the inner hub (`r_b = 0.42–0.58`) saves `42–58%` blade metal, leaves the hub wide open for the `0.35%` micro-hub $1R$ cavity, and **boosts $2T$ damping `2.57–2.90×` over full-span baffles** via exposed inner-tip vortex shedding.
  - Cracked the root cause of Entry `014`'s $t = 120$ $k = 0$ drift (`T ≈ 39` subcritical chugging at $\omega_{\text{hp}} = 0.15$) and eliminated it **`44,500–45,200×`** via mass-conserving $\dot{m}$ coupling + stiff feed-system cutoff $\omega_{\text{hp}} = 0.50$.
  - Demonstrated **`8.45×` wave-energy reduction in a true-3D smooth curved bell NS chamber (`20.21% → 7.04%`)** and **`9,524×` $1R$ quench** (`0.093%`), while exposing the **3D Transverse $\to$ Chugging $\to$ $1L$ Mode-Migration Cascade**.
- **What broke / dead ends:**
  - **Transverse-only baffle+cavity stabilization at $n_p = 0.34$:** Even with $k = 0$ chugging cured, suppressing `stand_1T` (`48×`) and `spin_2T` (`74×`) unmasks longitudinal $1L$ rumble (`6.0–6.4%`) after $t \approx 85$ because radial baffles cannot damp $1L$ and baffle-retuning $\omega_{H,1T}$ downward (`1.84 → 1.72`) shifts it further from $\omega_{1L} = 2.09$.
  - **2D-like naive viscous CFL (`0.28 / (4 nu / h^2)`):** Penalized polar-centerline 3D NS timesteps `6×` unnecessarily; using the true SSP-RK2 diffusion bound `0.40 / (nu / h^2)` gave a clean `5.7×` speedup with exact conservation.
- **Next single hypothesis to test:** Submit Entry `015` to independent read-only **Peer Review #12**, harden all findings, and then in **Entry `016`** test a **Tri-Band 3D Faceplate (Shroud $1T/2T$ + Hub $1R$ + $1L$ Longitudinal Damper / Axial Lag Spread)** inside the true-3D viscous bell to break the final $1L$ mode-migration wall and achieve true **5-mode (`1T, 2T, 1R, 1L, k=0`) 3D quench**!
