"""Entry 001 (plasma-nozzles): 2D Resistive-MHD Kelvin-Helmholtz Shear Instability.

Investigates plasma-plume shear stability across the Alfven detachment threshold M_A = 1
(from Entry 000) in 2D incompressible resistive MHD using vorticity-streamfunction + magnetic
flux potential (omega, psi, A_z) on a periodic box (x, y) in [0, L_x] x [0, L_y]:

    d(omega)/dt + u . grad(omega) = B . grad(J) + nu * lap(omega)
    d(A_z)/dt   + u . grad(A_z)   = eta * lap(A_z)

where:
    u = (d_y psi, -d_x psi),   omega = -lap(psi)   (div u == 0 identically!)
    B = (d_y A_z, -d_x A_z),   J     = -lap(A_z)   (div B == 0 identically!)

Base flow (periodic double shear layer of half-width a, velocity +/- U_0, uniform parallel B_0):
    U_x(y) = U_0 * [tanh((y - L_y/4)/a) - tanh((y - 3*L_y/4)/a) - 1]
    B_x(y) = B_0 = U_0 / M_A   (Alfven Mach number M_A = U_0 / v_A)
    Rm     = U_0 * a / eta     (magnetic Reynolds number)
    Re     = U_0 * a / nu      (viscous Reynolds number)

Time integration uses exact Integrating-Factor RK4 (IFRK4) on the diagonal Fourier diffusion
operators (-nu * k^2, -eta * k^2), ensuring unconditional diffusive stability down to Rm = 0.05!

Exact total MHD energy balance:
    E_tot = 0.5 * mean(|u|^2 + |B|^2),    dE_tot/dt = -nu * mean(omega^2) - eta * mean(J^2)

Experiments:
1. Exact 1D Fourier-Galerkin linearized MHD eigenvalue solver vs. full 2D pseudospectral
   (2/3-dealiased FFT + IFRK4) nonlinear resistive-MHD PDE solver across all regimes, plus
   exact unforced MHD energy balance verification.
2. Non-Monotonic Sub-Alfvenic Resistive MHD Shear Stability (M_A < 1, Rm in [0.05, 300]):
   Uncovers THREE distinct sub-Alfvenic regimes:
     (i)  Ideal-MHD elastic tension (Rm >> 1): frozen-in field lines bounce elastically (gamma ~ -0.008)
     (ii) Moderate-Rm Hartmann/Ohmic braking (Rm ~ 2-15): eddy-current dissipation brakes shear
          perturbations up to 14x MORE strongly than ideal MHD (deep damping trough!)
     (iii) Low-Rm magnetic-diffusion breakdown (Rm < Rm_crit(M_A) ~ 0.5-0.9): magnetic field
          decouples from plasma, unleashing violent KH vortex roll-up INSIDE the sub-Alfvenic nozzle!
3. Full 2D nonlinear evolution across four regimes + 64x64 vs 128x128 grid convergence check:
   (a) Super-Alfvenic (M_A=2.5, Rm=200), (b) Sub-Alfvenic Ideal (M_A=0.75, Rm=200),
   (c) Sub-Alfvenic Hartmann-Braked (M_A=0.75, Rm=4.0), (d) Sub-Alfvenic Low-Rm Roll-Up (M_A=0.75, Rm=0.20).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


# ---------------------------------------------------------------------------
# 1. Exact 1D Fourier-collocation linearized resistive-MHD eigenvalue solver
# ---------------------------------------------------------------------------

def base_shear_profile(y: np.ndarray, L_y: float, a: float, U_0: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Periodic double-tanh shear layer U_x(y) and its exact second derivative U_yy(y)."""
    s1 = (y - 0.25 * L_y) / a
    s2 = (y - 0.75 * L_y) / a
    t1, t2 = np.tanh(s1), np.tanh(s2)
    U = U_0 * (t1 - t2 - 1.0)
    sech2_1 = 1.0 - t1 * t1
    sech2_2 = 1.0 - t2 * t2
    U_yy = U_0 * (-(2.0 / (a * a)) * t1 * sech2_1 + (2.0 / (a * a)) * t2 * sech2_2)
    return U, U_yy


def solve_linear_mhd_kh(
    kx: float,
    M_A: float,
    Rm: float,
    Re: float = 250.0,
    L_y: float = 2.0 * np.pi,
    a: float = 0.25,
    U_0: float = 1.0,
    ny_eig: int = 96,
    return_vec: bool = False,
):
    """Exact 1D Fourier-spectral eigenvalue problem for perturbations ~ exp(i*kx*x + s*t).

    Linearized equations for vorticity omega_hat = (kx^2 - d_yy) psi and flux a_z(y):
        s * omega_hat = -i*kx * U * omega_hat - i*kx * U_yy * psi + i*kx * B_0 * j_hat - nu * L * omega_hat
        s * a_z       = -i*kx * U * a_z       + i*kx * B_0 * psi                       - eta * L * a_z
    Returns (max growth rate gamma = max(Re(s)), corresponding frequency Im(s)),
    or (gamma, omega, w_vec, a_vec) when return_vec=True.
    """
    B_0 = U_0 / M_A
    nu = U_0 * a / Re
    eta = U_0 * a / Rm

    y = np.linspace(0.0, L_y, ny_eig, endpoint=False)
    U, U_yy = base_shear_profile(y, L_y, a, U_0)

    ky = 2.0 * np.pi * np.fft.fftfreq(ny_eig, d=L_y / ny_eig)
    lap_k = kx * kx + ky * ky
    eye = np.eye(ny_eig)
    L_mat = np.fft.ifft(lap_k[:, None] * np.fft.fft(eye, axis=0), axis=0).real
    L_inv = np.fft.ifft((1.0 / np.maximum(lap_k, 1e-30))[:, None] * np.fft.fft(eye, axis=0), axis=0).real

    ikx = 1j * kx
    A11 = -ikx * np.diag(U) - ikx * (np.diag(U_yy) @ L_inv) - nu * L_mat
    A12 = ikx * B_0 * L_mat
    A21 = ikx * B_0 * L_inv
    A22 = -ikx * np.diag(U) - eta * L_mat

    M_sys = np.block([[A11, A12], [A21, A22]])
    if return_vec:
        eigvals, eigvecs = np.linalg.eig(M_sys)
        idx = int(np.argmax(eigvals.real))
        vec = eigvecs[:, idx]
        w_vec = vec[:ny_eig]
        a_vec = vec[ny_eig:]
        norm_w = max(float(np.max(np.abs(w_vec))), 1e-12)
        return float(eigvals[idx].real), float(eigvals[idx].imag), w_vec / norm_w, a_vec / norm_w
    eigvals = np.linalg.eigvals(M_sys)
    idx = int(np.argmax(eigvals.real))
    return float(eigvals[idx].real), float(eigvals[idx].imag)


# ---------------------------------------------------------------------------
# 2. Full 2D Pseudospectral (2/3-Dealiased FFT + IFRK4) Nonlinear Resistive-MHD PDE
# ---------------------------------------------------------------------------

def simulate_2d_resistive_mhd(
    M_A: float,
    Rm: float,
    Re: float = 250.0,
    L_x: float = 2.0 * np.pi,
    L_y: float = 2.0 * np.pi,
    a: float = 0.25,
    U_0: float = 1.0,
    nx: int = 64,
    ny: int = 64,
    dt: float = 0.005,
    t_end: float = 18.0,
    pert_amp: float = 1e-3,
    pert_kx_mode: int = 1,
    keep_mean_forcing: bool = True,
    eig_init: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict:
    """Solve full 2D nonlinear incompressible resistive MHD on [0, L_x) x [0, L_y).

    Uses exact Integrating-Factor RK4 (IFRK4) on viscous and resistive diffusion so
    low-Rm (high-eta) runs remain unconditionally stable without shrinking dt.
    """
    B_0 = U_0 / M_A
    nu = U_0 * a / Re
    eta = U_0 * a / Rm

    x = np.linspace(0.0, L_x, nx, endpoint=False)
    y = np.linspace(0.0, L_y, ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")

    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=L_x / nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=L_y / ny)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    K2 = KX * KX + KY * KY
    inv_K2 = np.where(K2 > 0.0, 1.0 / np.maximum(K2, 1e-30), 0.0)

    kx_cut = (2.0 / 3.0) * np.max(np.abs(kx))
    ky_cut = (2.0 / 3.0) * np.max(np.abs(ky))
    dealias = (np.abs(KX) <= kx_cut) & (np.abs(KY) <= ky_cut)

    U_base_1d, _ = base_shear_profile(y, L_y, a, U_0)
    U_base = np.broadcast_to(U_base_1d[None, :], (nx, ny)).copy()
    omega_0_hat = -1j * KY * np.fft.fft2(U_base)
    f_visc_hat = (nu * K2 * omega_0_hat) if keep_mean_forcing else np.zeros_like(omega_0_hat)

    kx_p = pert_kx_mode * (2.0 * np.pi / L_x)
    if eig_init is not None:
        w_vec, a_vec = eig_init
        phase_x = np.exp(1j * kx_p * X)
        w_pert = pert_amp * np.real(phase_x * w_vec[None, :])
        a_pert = pert_amp * np.real(phase_x * a_vec[None, :])
        omega_hat = (omega_0_hat + np.fft.fft2(w_pert)) * dealias
        az_hat = np.fft.fft2(a_pert) * dealias
    else:
        # Asymmetric double-layer perturbation excites both varicose (fastest) & sinuous modes
        env_y = np.exp(-((Y - 0.25 * L_y) / (2.0 * a)) ** 2) - 0.75 * np.exp(-((Y - 0.75 * L_y) / (2.0 * a)) ** 2)
        psi_pert = pert_amp * np.sin(kx_p * X) * env_y
        psi_pert_hat = np.fft.fft2(psi_pert) * dealias
        omega_hat = (omega_0_hat + K2 * psi_pert_hat) * dealias
        az_hat = np.zeros_like(omega_hat)

    def fields_from_hat(w_h: np.ndarray, a_h: np.ndarray):
        psi_h = w_h * inv_K2
        ux_h = 1j * KY * psi_h
        uy_h = -1j * KX * psi_h
        bx_h = 1j * KY * a_h
        by_h = -1j * KX * a_h
        j_h = K2 * a_h
        return ux_h, uy_h, bx_h, by_h, j_h

    def nl_rhs(w_h: np.ndarray, a_h: np.ndarray):
        """Nonlinear advection + Lorentz force + optional static base-shear forcing."""
        ux_h, uy_h, bx_h, by_h, j_h = fields_from_hat(w_h, a_h)

        ux = np.fft.ifft2(ux_h).real
        uy = np.fft.ifft2(uy_h).real
        bx = B_0 + np.fft.ifft2(bx_h).real
        by = np.fft.ifft2(by_h).real

        dwx = np.fft.ifft2(1j * KX * w_h).real
        dwy = np.fft.ifft2(1j * KY * w_h).real
        djx = np.fft.ifft2(1j * KX * j_h).real
        djy = np.fft.ifft2(1j * KY * j_h).real
        dax = np.fft.ifft2(1j * KX * a_h).real
        day = np.fft.ifft2(1j * KY * a_h).real

        nl_w = -(ux * dwx + uy * dwy) + (bx * djx + by * djy)
        nl_a = -(ux * dax + uy * (B_0 + day))

        dw_h = np.fft.fft2(nl_w) * dealias + f_visc_hat
        da_h = np.fft.fft2(nl_a) * dealias
        dw_h[0, 0] = 0.0
        da_h[0, 0] = 0.0
        return dw_h, da_h

    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps
    t_arr = np.linspace(0.0, t_end, n_steps + 1)

    # Exact integrating factors for half-step and full-step diffusion
    Ew_h = np.exp(-0.5 * dt * nu * K2)
    Ew_f = Ew_h * Ew_h
    Ea_h = np.exp(-0.5 * dt * eta * K2)
    Ea_f = Ea_h * Ea_h

    ke_y = np.empty(n_steps + 1)
    me_pert = np.empty(n_steps + 1)
    e_tot = np.empty(n_steps + 1)
    diss_tot = np.empty(n_steps + 1)

    def record(idx: int) -> None:
        ux_h, uy_h, bx_h, by_h, j_h = fields_from_hat(omega_hat, az_hat)
        ux = np.fft.ifft2(ux_h).real
        uy = np.fft.ifft2(uy_h).real
        bx_p = np.fft.ifft2(bx_h).real
        by_p = np.fft.ifft2(by_h).real
        w_r = np.fft.ifft2(omega_hat).real
        j_r = np.fft.ifft2(j_h).real
        ke_y[idx] = 0.5 * float(np.mean(uy * uy))
        me_pert[idx] = 0.5 * float(np.mean(bx_p * bx_p + by_p * by_p))
        e_tot[idx] = 0.5 * float(np.mean(ux * ux + uy * uy + (B_0 + bx_p) ** 2 + by_p * by_p))
        diss_tot[idx] = float(nu * np.mean(w_r * w_r) + eta * np.mean(j_r * j_r))

    record(0)

    for step in range(n_steps):
        k1_w, k1_a = nl_rhs(omega_hat, az_hat)

        w2 = Ew_h * (omega_hat + 0.5 * dt * k1_w)
        a2 = Ea_h * (az_hat + 0.5 * dt * k1_a)
        k2_w, k2_a = nl_rhs(w2, a2)

        w3 = Ew_h * omega_hat + 0.5 * dt * k2_w
        a3 = Ea_h * az_hat + 0.5 * dt * k2_a
        k3_w, k3_a = nl_rhs(w3, a3)

        w4 = Ew_f * omega_hat + dt * Ew_h * k3_w
        a4 = Ea_f * az_hat + dt * Ea_h * k3_a
        k4_w, k4_a = nl_rhs(w4, a4)

        omega_hat = (Ew_f * omega_hat + (dt / 6.0) * (Ew_f * k1_w + 2.0 * Ew_h * (k2_w + k3_w) + k4_w)) * dealias
        az_hat = (Ea_f * az_hat + (dt / 6.0) * (Ea_f * k1_a + 2.0 * Ea_h * (k2_a + k3_a) + k4_a)) * dealias
        record(step + 1)

    cum_diss = np.concatenate([[0.0], np.cumsum(0.5 * (diss_tot[:-1] + diss_tot[1:]) * dt)])
    dE = e_tot - e_tot[0]
    e_err = float(np.max(np.abs(dE + cum_diss)) / max(np.max(np.abs(dE)), 1e-12))

    omega_fin = np.fft.ifft2(omega_hat).real
    az_tot_fin = B_0 * Y + np.fft.ifft2(az_hat).real
    rms_uy_max = float(np.sqrt(2.0 * np.max(ke_y)))

    return {
        "t": t_arr, "ke_y": ke_y, "me_pert": me_pert, "e_tot": e_tot,
        "energy_balance_rel_err": e_err,
        "omega_final": omega_fin, "az_tot_final": az_tot_fin,
        "rms_uy_max": rms_uy_max, "X": X, "Y": Y,
    }


def fit_ke_growth(t: np.ndarray, ke_y: np.ndarray, t_min: float, t_max: float) -> float:
    mask = (t >= t_min) & (t <= t_max)
    return float(np.polyfit(t[mask], 0.5 * np.log(np.maximum(ke_y[mask], 1e-30)), 1)[0])


# ---------------------------------------------------------------------------
# 3. Main experiments & figures
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    a = 0.25
    Re = 250.0
    kx_test = 1.0  # Fundamental mode k_x * a = 0.25 (near peak KH growth for tanh layer)

    print("=== Entry 001 (plasma-nozzles): 2D Resistive-MHD Kelvin-Helmholtz Shear Instability ===")
    print(f"Double-tanh shear layer a = {a}, Re = {Re}, fundamental k_x = {kx_test} (k_x*a = {kx_test*a:.2f})\n")

    # ------------------------------------------------------------------
    # Experiment 1: Exact Linear Eigenvalues vs 2D Pseudospectral PDE + Energy Check
    # ------------------------------------------------------------------
    print("--- Experiment 1: Linear Eigenvalue Solver vs. Full 2D Nonlinear MHD PDE (IFRK4) ---")
    ver_cases = [
        ("super_alfvenic_ideal", 2.50, 200.0, (2.5, 6.5)),
        ("near_cutoff_MA125", 1.25, 200.0, (3.5, 9.0)),
        ("sub_alfvenic_ideal", 0.75, 200.0, (3.0, 10.0)),
        ("sub_alfvenic_hartmann", 0.75, 4.0, (3.0, 8.5)),
        ("sub_alfvenic_lowRm", 0.75, 0.20, (2.5, 7.5)),
    ]
    ver_summary = {}
    for cid, ma_v, rm_v, fit_w in ver_cases:
        g_ex, w_ex, w_vec, a_vec = solve_linear_mhd_kh(kx_test, ma_v, rm_v, Re=Re, a=a, ny_eig=64, return_vec=True)
        sim = simulate_2d_resistive_mhd(
            M_A=ma_v, Rm=rm_v, Re=Re, a=a, nx=64, ny=64, dt=0.005, t_end=11.0,
            pert_amp=1e-4, keep_mean_forcing=True, eig_init=(w_vec, a_vec),
        )
        # Fit total perturbation energy (KE_y + ME_pert) so standing Alfvén oscillations cancel!
        e_pert = sim["ke_y"] + sim["me_pert"]
        g_pde = fit_ke_growth(sim["t"], e_pert, fit_w[0], fit_w[1])
        print(
            f"  [{cid:22s}] M_A={ma_v:.2f}, Rm={rm_v:6.2f}: "
            f"exact gamma={g_ex:+.6f} | PDE gamma={g_pde:+.6f} (|diff|={abs(g_pde - g_ex):.2e})"
        )
        ver_summary[cid] = {
            "M_A": ma_v, "Rm": rm_v,
            "gamma_exact": g_ex, "omega_exact": w_ex,
            "gamma_pde": g_pde, "abs_diff": float(abs(g_pde - g_ex)),
        }

    # Exact unforced MHD total energy conservation check (keep_mean_forcing=False)
    sim_econ = simulate_2d_resistive_mhd(
        M_A=1.5, Rm=50.0, Re=100.0, a=a, nx=64, ny=64, dt=0.004, t_end=8.0,
        pert_amp=0.05, keep_mean_forcing=False,
    )
    print(f"  Unforced nonlinear 2D MHD energy balance rel err (dE + int(D_nu+D_eta) dt): {sim_econ['energy_balance_rel_err']:.2e}\n")

    # ------------------------------------------------------------------
    # Experiment 2: Non-Monotonic Sub-Alfvenic Resistive MHD Shear Stability & 2D Map
    # ------------------------------------------------------------------
    print("--- Experiment 2: Non-Monotonic Sub-Alfvenic Stability (Hartmann Trough + Low-Rm Onset) ---")
    rm_sweep = np.logspace(-1.3, 2.5, 48)  # Rm in [0.05, 316]
    g_sub_092 = np.array([solve_linear_mhd_kh(kx_test, 0.92, rm, Re=Re, a=a, ny_eig=72)[0] for rm in rm_sweep])
    g_sub_075 = np.array([solve_linear_mhd_kh(kx_test, 0.75, rm, Re=Re, a=a, ny_eig=72)[0] for rm in rm_sweep])
    g_sub_060 = np.array([solve_linear_mhd_kh(kx_test, 0.60, rm, Re=Re, a=a, ny_eig=72)[0] for rm in rm_sweep])

    def find_rm_crit_and_trough(ma_v: float) -> dict:
        # Bisect Rm_crit where gamma crosses 0 from positive (low Rm) to negative (moderate Rm)
        lo, hi = 0.01, 20.0
        for _ in range(35):
            mid = np.sqrt(lo * hi)
            g = solve_linear_mhd_kh(kx_test, ma_v, mid, Re=Re, a=a, ny_eig=72)[0]
            if g > 0.0:
                lo = mid
            else:
                hi = mid
        rc = float(np.sqrt(lo * hi))
        # Find Hartmann braking trough (most negative gamma for Rm in [rc, 200])
        rms_t = np.logspace(np.log10(max(rc, 0.05)), 2.3, 45)
        gs_t = [solve_linear_mhd_kh(kx_test, ma_v, r, Re=Re, a=a, ny_eig=72)[0] for r in rms_t]
        i_min = int(np.argmin(gs_t))
        g_ideal = solve_linear_mhd_kh(kx_test, ma_v, 200.0, Re=Re, a=a, ny_eig=72)[0]
        return {
            "M_A": ma_v,
            "Rm_crit": rc,
            "Rm_hartmann_opt": float(rms_t[i_min]),
            "gamma_hartmann_min": float(gs_t[i_min]),
            "gamma_ideal_Rm200": float(g_ideal),
            "braking_ratio": float(gs_t[i_min] / min(g_ideal, -1e-9)),
        }

    rm_crit_map = {}
    for ma_v in (0.50, 0.60, 0.75, 0.85, 0.92):
        info = find_rm_crit_and_trough(ma_v)
        print(
            f"  Sub-Alfvenic M_A={ma_v:.2f}: unstable for Rm < {info['Rm_crit']:.3f} | "
            f"Hartmann trough gamma={info['gamma_hartmann_min']:+.4f} at Rm={info['Rm_hartmann_opt']:.2f} "
            f"({info['braking_ratio']:.1f}x stronger damping than ideal Rm=200 [{info['gamma_ideal_Rm200']:+.4f}])"
        )
        rm_crit_map[f"MA_{ma_v:.2f}"] = info

    # 2D (M_A, Rm) stability map across M_A in [0.45, 1.80] and Rm in [0.05, 300]
    ma_grid = np.linspace(0.45, 1.80, 28)
    rm_grid = np.logspace(-1.3, 2.48, 28)
    gamma_map = np.zeros((len(rm_grid), len(ma_grid)))
    for i, rmv in enumerate(rm_grid):
        for j, mav in enumerate(ma_grid):
            gamma_map[i, j] = solve_linear_mhd_kh(kx_test, mav, rmv, Re=Re, a=a, ny_eig=64)[0]

    # ------------------------------------------------------------------
    # Experiment 3: Full 2D Nonlinear Evolution (4 Regimes + Grid Convergence Check)
    # ------------------------------------------------------------------
    print("\n--- Experiment 3: Full 2D Nonlinear Evolution (96x96, t_end = 18.0, pert_amp = 0.015) ---")
    nl_regimes = [
        ("super_alfvenic", "Super-Alfvénic\n($M_A=2.5, R_m=200$)", 2.50, 200.0),
        ("sub_ideal", "Sub-Alfvénic Ideal\n($M_A=0.75, R_m=200$)", 0.75, 200.0),
        ("sub_hartmann", "Hartmann-Braked\n($M_A=0.75, R_m=4.0$)", 0.75, 4.0),
        ("sub_lowRm", "Low-$R_m$ Uncoupled KH\n($M_A=0.75, R_m=0.20$)", 0.75, 0.20),
    ]
    nl_sims = {}
    nl_summary = {}
    for cid, label, mav, rmv in nl_regimes:
        sim = simulate_2d_resistive_mhd(
            M_A=mav, Rm=rmv, Re=Re, a=a, nx=96, ny=96, dt=0.004, t_end=18.0,
            pert_amp=0.015, keep_mean_forcing=True,
        )
        print(
            f"  [{cid:16s}] peak KE_y={np.max(sim['ke_y']):.4e}, final KE_y={sim['ke_y'][-1]:.4e}, "
            f"max rms u_y={sim['rms_uy_max']:.4f}, peak ME_pert={np.max(sim['me_pert']):.4e}"
        )
        nl_sims[cid] = (label, sim)
        nl_summary[cid] = {
            "M_A": mav, "Rm": rmv,
            "peak_ke_y": float(np.max(sim["ke_y"])),
            "final_ke_y": float(sim["ke_y"][-1]),
            "max_rms_uy": sim["rms_uy_max"],
            "peak_me_pert": float(np.max(sim["me_pert"])),
        }

    # Grid convergence check on low-Rm sub-Alfvenic KH (64x64 vs 128x128 at M_A=0.75, Rm=0.20)
    _, _, w_vec_128, a_vec_128 = solve_linear_mhd_kh(kx_test, 0.75, 0.20, Re=Re, a=a, ny_eig=128, return_vec=True)
    sim_128 = simulate_2d_resistive_mhd(
        M_A=0.75, Rm=0.20, Re=Re, a=a, nx=128, ny=128, dt=0.005, t_end=11.0,
        pert_amp=1e-4, keep_mean_forcing=True, eig_init=(w_vec_128, a_vec_128),
    )
    g_128 = fit_ke_growth(sim_128["t"], sim_128["ke_y"] + sim_128["me_pert"], 2.5, 7.5)
    g_64 = ver_summary["sub_alfvenic_lowRm"]["gamma_pde"]
    print(f"\nGrid check (sub-Alfvenic low-Rm M_A=0.75, Rm=0.20): 64x64={g_64:+.6f} vs 128x128={g_128:+.6f} (|diff|={abs(g_128 - g_64):.2e})")

    # ------------------------------------------------------------------
    # Figure 1: Non-Monotonic Sub-Alfvenic Curves + 2D (M_A, Rm) Stability Map
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.5), dpi=150)

    ax1.axhline(0.0, color="black", lw=1.3)
    ax1.axhspan(-0.30, 0.0, color="#e8f4fa", alpha=0.55)
    ax1.semilogx(rm_sweep, g_sub_092, "-", color="#d62728", lw=2.2,
                 label="$M_A = 0.92$ ($R_{m,\\mathrm{crit}} = %.2f$)" % rm_crit_map["MA_0.92"]["Rm_crit"])
    ax1.semilogx(rm_sweep, g_sub_075, "-", color="#1f77b4", lw=2.3,
                 label="$M_A = 0.75$ ($R_{m,\\mathrm{crit}} = %.2f$)" % rm_crit_map["MA_0.75"]["Rm_crit"])
    ax1.semilogx(rm_sweep, g_sub_060, "-", color="#2ca02c", lw=2.2,
                 label="$M_A = 0.60$ ($R_{m,\\mathrm{crit}} = %.2f$)" % rm_crit_map["MA_0.60"]["Rm_crit"])
    ax1.scatter(
        [0.20, 4.0, 200.0],
        [ver_summary["sub_alfvenic_lowRm"]["gamma_pde"],
         ver_summary["sub_alfvenic_hartmann"]["gamma_pde"],
         ver_summary["sub_alfvenic_ideal"]["gamma_pde"]],
        c=["crimson", "gold", "cyan"], edgecolors="black", s=70, zorder=5,
        label="2D Nonlinear MHD PDE ($M_A=0.75$)",
    )
    box = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.88)
    ax1.annotate("Low-$R_m$ KH\nBreakdown", (0.20, ver_summary["sub_alfvenic_lowRm"]["gamma_pde"]),
                 xytext=(0.06, 0.34), fontsize=8.0, fontweight="bold", color="#b00000",
                 arrowprops=dict(arrowstyle="->", color="#b00000", lw=1.1), bbox=box)
    ax1.annotate("Hartmann / Ohmic\nBraking Trough!", (4.0, ver_summary["sub_alfvenic_hartmann"]["gamma_pde"]),
                 xytext=(8.0, -0.22), fontsize=8.2, fontweight="bold", color="#004488",
                 arrowprops=dict(arrowstyle="->", color="#004488", lw=1.1), bbox=box)
    ax1.set_ylim(-0.26, 0.48)
    ax1.set_xlabel("Magnetic Reynolds number $R_m = U_0 a / \\eta$")
    ax1.set_ylabel("KH growth rate $\\gamma$ ($k_x a = 0.25$)")
    ax1.set_title("Non-Monotonic Sub-Alfvénic ($M_A < 1$) Shear Stability", fontweight="bold")
    ax1.grid(True, which="both", alpha=0.28)
    ax1.legend(loc="upper right", fontsize=8.0)

    norm = TwoSlopeNorm(vmin=-0.22, vcenter=0.0, vmax=0.55)
    im = ax2.contourf(ma_grid, rm_grid, gamma_map, levels=np.linspace(-0.22, 0.55, 32),
                      cmap="RdBu_r", norm=norm, extend="both")
    ax2.contour(ma_grid, rm_grid, gamma_map, levels=[0.0], colors="black", linewidths=2.2)
    ax2.axvline(1.0, color="gold", ls="--", lw=1.8, label="Ideal-MHD Chandrasekhar cutoff ($M_A = 1$)")
    ax2.scatter([0.75, 0.75, 0.75, 1.25], [0.20, 4.0, 200.0, 200.0],
                c=["crimson", "gold", "lime", "orange"], edgecolors="black", s=65, zorder=5)
    ax2.annotate("Low-$R_m$ Sub-Alfvénic\nKH Roll-Up", (0.75, 0.20), xytext=(0.47, 0.08),
                 fontsize=7.8, fontweight="bold", color="#900000", bbox=box)
    ax2.annotate("Hartmann Braking\nSweet Spot", (0.75, 4.0), xytext=(0.48, 8.5),
                 fontsize=8.0, fontweight="bold", color="#003d7a", bbox=box)
    ax2.annotate("Elastic Tension\n(Ideal MHD)", (0.75, 200.0), xytext=(0.48, 120.0),
                 fontsize=7.8, fontweight="bold", color="#005533", bbox=box)
    ax2.set_yscale("log")
    ax2.set_xlabel("Alfvén Mach number $M_A = U_0 / v_A$")
    ax2.set_ylabel("Magnetic Reynolds number $R_m$")
    ax2.set_title("2D $(M_A, R_m)$ Stability Map ($k_x a = 0.25, \\mathrm{Re}=250$)", fontweight="bold")
    ax2.legend(loc="lower right", fontsize=7.9)
    plt.colorbar(im, ax=ax2, label="KH growth rate $\\gamma$")

    fig.suptitle("Plasma Nozzles Entry 001 (Part 1): Three-Regime Resistive MHD Shear Stability", fontsize=12.2, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(RESULTS_DIR / "exp001_mhd_kh_stability.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2: 2D Nonlinear Vortex Roll-Up Snapshots (4 Regimes) + Energy Traces
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 5, figsize=(18.5, 4.1), dpi=150)
    for ax, cid in zip(axes[:4], ["super_alfvenic", "sub_ideal", "sub_hartmann", "sub_lowRm"]):
        lbl, s = nl_sims[cid]
        ax.imshow(
            s["omega_final"].T, origin="lower", extent=[0, 2 * np.pi, 0, 2 * np.pi],
            cmap="RdBu_r", vmin=-4.5, vmax=4.5, aspect="equal",
        )
        ax.contour(s["X"], s["Y"], s["az_tot_final"], levels=12, colors="black", linewidths=0.75, alpha=0.8)
        ax.set_title(lbl, fontsize=9.0, fontweight="bold")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")

    ax5 = axes[4]
    cols = {
        "super_alfvenic": "#d62728",
        "sub_ideal": "#2ca02c",
        "sub_hartmann": "#1f77b4",
        "sub_lowRm": "#ff7f0e",
    }
    short_lbls = {
        "super_alfvenic": "Super-Alfvénic ($M_A=2.5$)",
        "sub_ideal": "Sub-Alfv. Ideal ($R_m=200$)",
        "sub_hartmann": "Hartmann-Braked ($R_m=4$)",
        "sub_lowRm": "Low-$R_m$ KH ($R_m=0.2$)",
    }
    for cid in ("super_alfvenic", "sub_lowRm", "sub_ideal", "sub_hartmann"):
        _, s = nl_sims[cid]
        ax5.semilogy(s["t"], s["ke_y"], color=cols[cid], lw=2.0, label=short_lbls[cid])
    ax5.set_ylim(1e-10, 1.0)
    ax5.set_xlabel("Dimensionless time $t U_0 / L$")
    ax5.set_ylabel("Transverse mixing $\\frac{1}{2}\\langle u_y^2 \\rangle$")
    ax5.set_title("Cross-Field Plume Mixing", fontsize=9.5, fontweight="bold")
    ax5.grid(True, which="both", alpha=0.28)
    ax5.legend(loc="lower left", fontsize=7.0)

    fig.suptitle("Plasma Nozzles Entry 001 (Part 2): 2D Nonlinear Vorticity & Magnetic Field Lines ($t = 18, 96\\times 96$)", fontsize=12.0, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(RESULTS_DIR / "exp001_mhd_kh_nonlinear.png")
    plt.close(fig)

    metrics = {
        "parameters": {"a": a, "Re": Re, "kx_test": kx_test, "grid_base": [64, 64]},
        "exp1_linear_vs_pde": ver_summary,
        "exp1_unforced_energy_balance_rel_err": sim_econ["energy_balance_rel_err"],
        "exp2_sub_alfvenic_regimes": rm_crit_map,
        "exp3_nonlinear_regimes": nl_summary,
        "exp3_grid_convergence_64_vs_128_lowRm": {
            "M_A": 0.75, "Rm": 0.20,
            "gamma_64": g_64,
            "gamma_128": g_128,
            "abs_diff": float(abs(g_128 - g_64)),
        },
    }
    with open(RESULTS_DIR / "exp001_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved plasma-nozzles Entry 001 plots and metrics to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
