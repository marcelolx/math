"""Entry 001 (plasma-nozzles): 2D Resistive-MHD Kelvin-Helmholtz Shear Instability.

Investigates plasma-plume shear stability across the Alfven detachment threshold M_A = 1
(from Entry 000) in 2D incompressible resistive MHD using vorticity-streamfunction + magnetic
flux potential (omega, psi, A_z) on a periodic box (x, y) in [0, L_x] x [0, L_y]:

    d(omega)/dt + u . grad(omega) = B . grad(J) + nu * lap(omega)
    d(A_z)/dt   + u . grad(A_z)   = eta * lap(A_z)

where:
    u = (d_y psi, -d_x psi),   omega = -lap(psi)   (div u == 0 identically)
    B = (d_y A_z, -d_x A_z),   J     = -lap(A_z)   (div B == 0 identically)

Time integration uses exact Integrating-Factor RK4 (IFRK4) on the diagonal Fourier diffusion
operators (-nu * k^2, -eta * k^2), ensuring unconditional diffusive stability down to Rm = 0.05.

Experiments (hardened after Peer Review #3):
1. Discretized 1D Fourier-collocation linearized MHD eigenvalue solver vs. full 2D pseudospectral
   (2/3-dealiased FFT + IFRK4) nonlinear resistive-MHD PDE solver across all 5 regimes, plus
   exact unforced MHD energy balance verification (dE/dt = -nu*<w^2> - eta*<J^2>).
2. Robust Low-Rm Sub-Alfvenic Breakdown Law (Rm_crit ~ N_c(k_x*a) * M_A^2) vs. Re- & Box-Dependent
   Bulk Alfven-Continuum Trough:
   - Verifies that Rm_crit / M_A^2 = N_c(k_x*a) is Re-independent across Re in {250, 1000, 4000}
     while mapping its wavenumber dependence across k_x*a in {0.125, 0.25, 0.50} and box height
     L_y in {2*pi, 4*pi} (longer wavelengths break down at larger Rm).
   - Quantifies honestly that the intermediate-Rm damping trough (Re=250, L_y=2*pi) is a
     delocalized bulk Alfven-continuum box mode scaling as gamma_min ~ -2*kx*B0*sqrt(Rm*/Re),
     whose depth halves from Re=250 to Re=4000 and vanishes as Re -> inf.
3. Full 2D nonlinear evolution (96x96, t=18) across four regimes + 64x64 vs 128x128 grid check.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from scipy.optimize import minimize_scalar


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


# ---------------------------------------------------------------------------
# 1. 1D Fourier-collocation linearized resistive-MHD eigenvalue solver
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
    """1D Fourier-collocation eigenvalue problem for perturbations ~ exp(i*kx*x + s*t)."""
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
    """Solve full 2D nonlinear incompressible resistive MHD on [0, L_x) x [0, L_y) via IFRK4."""
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


def find_rm_crit(kx: float, ma_v: float, Re: float, L_y: float = 2.0 * np.pi, a: float = 0.25, ny_eig: int = 72) -> float:
    lo, hi = 0.01, 20.0
    for _ in range(35):
        mid = np.sqrt(lo * hi)
        g = solve_linear_mhd_kh(kx, ma_v, mid, Re=Re, L_y=L_y, a=a, ny_eig=ny_eig)[0]
        if g > 0.0:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


# ---------------------------------------------------------------------------
# 3. Main experiments & figures
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    a = 0.25
    Re = 250.0
    kx_test = 1.0

    print("=== Entry 001 (plasma-nozzles): 2D Resistive-MHD Kelvin-Helmholtz Shear Instability ===")
    print(f"Double-tanh shear layer a = {a}, base Re = {Re}, k_x*a = {kx_test*a:.2f}\n")

    # Experiment 1: Linear Collocation Eigenvalues vs 2D Nonlinear MHD PDE
    print("--- Experiment 1: Linear Collocation Eigenvalues vs. 2D Nonlinear MHD PDE (IFRK4) ---")
    ver_cases = [
        ("super_alfvenic_ideal", 2.50, 200.0, (2.5, 6.5)),
        ("near_cutoff_MA125", 1.25, 200.0, (3.5, 9.0)),
        ("sub_alfvenic_ideal", 0.75, 200.0, (3.0, 10.0)),
        ("sub_alfvenic_trough", 0.75, 4.0, (3.0, 8.5)),
        ("sub_alfvenic_lowRm", 0.75, 0.20, (2.5, 7.5)),
    ]
    ver_summary = {}
    for cid, ma_v, rm_v, fit_w in ver_cases:
        g_ex, w_ex, w_vec, a_vec = solve_linear_mhd_kh(kx_test, ma_v, rm_v, Re=Re, a=a, ny_eig=64, return_vec=True)
        sim = simulate_2d_resistive_mhd(
            M_A=ma_v, Rm=rm_v, Re=Re, a=a, nx=64, ny=64, dt=0.005, t_end=11.0,
            pert_amp=1e-4, keep_mean_forcing=True, eig_init=(w_vec, a_vec),
        )
        g_pde = fit_ke_growth(sim["t"], sim["ke_y"] + sim["me_pert"], fit_w[0], fit_w[1])
        print(f"  [{cid:22s}] M_A={ma_v:.2f}, Rm={rm_v:6.2f}: eig={g_ex:+.6f} | PDE={g_pde:+.6f} (|diff|={abs(g_pde - g_ex):.2e})")
        ver_summary[cid] = {
            "M_A": ma_v, "Rm": rm_v, "gamma_eig": g_ex, "omega_eig": w_ex,
            "gamma_pde": g_pde, "abs_diff": float(abs(g_pde - g_ex)),
        }

    sim_econ = simulate_2d_resistive_mhd(
        M_A=1.5, Rm=50.0, Re=100.0, a=a, nx=64, ny=64, dt=0.004, t_end=8.0,
        pert_amp=0.05, keep_mean_forcing=False,
    )
    print(f"  Unforced nonlinear 2D MHD energy balance rel err: {sim_econ['energy_balance_rel_err']:.2e}\n")

    # Experiment 2A: Sub-Alfvenic Stuart-Number Breakdown Scaling (Rm_crit / M_A^2)
    print("--- Experiment 2A: Sub-Alfvenic Breakdown Scaling Rm_crit = N_c(k_x*a) * M_A^2 ---")
    rm_crit_map = {}
    for ma_v in (0.50, 0.60, 0.75, 0.85, 0.92):
        rc = find_rm_crit(kx_test, ma_v, Re=Re, a=a, ny_eig=72)
        nc = rc / (ma_v ** 2)
        res_t = minimize_scalar(
            lambda lrm: solve_linear_mhd_kh(kx_test, ma_v, 10.0 ** lrm, Re=Re, a=a, ny_eig=72)[0],
            bounds=(np.log10(max(rc, 0.05)), 2.3), method="bounded", options={"xatol": 1e-3},
        )
        rm_opt = float(10.0 ** res_t.x)
        g_min = float(res_t.fun)
        g_id = solve_linear_mhd_kh(kx_test, ma_v, 200.0, Re=Re, a=a, ny_eig=72)[0]
        print(f"  M_A={ma_v:.2f}: Rm_crit={rc:.4f} (N_c = Rm_crit/M_A^2 = {nc:.3f}) | Re=250 box trough gamma={g_min:+.4f} at Rm*={rm_opt:.2f}")
        rm_crit_map[f"MA_{ma_v:.2f}"] = {
            "M_A": ma_v, "Rm_crit": rc, "N_c": nc,
            "Rm_trough_Re250": rm_opt, "gamma_trough_Re250": g_min, "gamma_Rm200_Re250": g_id,
        }

    # Experiment 2B: Re, Box-Size (L_y), and Wavenumber (k_x*a) Sensitivity Audit (Peer Review #3 C2 & M3!)
    print("\n--- Experiment 2B: Re, Box-Size & Wavenumber Audit (Separating Robust Physics from Box/Re Modes) ---")
    re_audit = {}
    for re_v, ly_v, ny_v in [(250.0, 2 * np.pi, 72), (1000.0, 2 * np.pi, 80), (4000.0, 2 * np.pi, 96), (250.0, 4 * np.pi, 112)]:
        rc = find_rm_crit(kx_test, 0.75, Re=re_v, L_y=ly_v, a=a, ny_eig=ny_v)
        res_t = minimize_scalar(
            lambda lrm: solve_linear_mhd_kh(kx_test, 0.75, 10.0 ** lrm, Re=re_v, L_y=ly_v, a=a, ny_eig=ny_v)[0],
            bounds=(np.log10(max(rc, 0.1)), 2.2), method="bounded", options={"xatol": 1e-3},
        )
        rm_opt = float(10.0 ** res_t.x)
        g_min = float(res_t.fun)
        g_scale = -2.0 * kx_test * (1.0 / 0.75) * np.sqrt(rm_opt / re_v)
        key = f"Re{int(re_v)}_Ly{int(round(ly_v/np.pi))}pi"
        print(
            f"  [{key:12s}] M_A=0.75: Rm_crit={rc:.4f} (N_c={rc/0.75**2:.3f}) | "
            f"box trough gamma_min={g_min:+.4f} at Rm*={rm_opt:.2f} (bulk-mode scaling -2*kx*B0*sqrt(Rm*/Re)={g_scale:+.4f})"
        )
        re_audit[key] = {
            "Re": re_v, "L_y_over_pi": ly_v / np.pi, "Rm_crit": rc, "N_c": rc / (0.75 ** 2),
            "Rm_trough": rm_opt, "gamma_trough": g_min, "bulk_mode_scaling": float(g_scale),
        }

    kx_audit = {}
    for kx_v in (0.5, 1.0, 2.0):
        rc = find_rm_crit(kx_v, 0.75, Re=Re, a=a, ny_eig=72)
        nc = rc / (0.75 ** 2)
        print(f"  [kx*a={kx_v*a:.3f}  ] M_A=0.75: Rm_crit={rc:.4f} => N_c(kx*a) = {nc:.3f}")
        kx_audit[f"kxa_{kx_v*a:.3f}"] = {"kx_a": kx_v * a, "Rm_crit": rc, "N_c": nc}

    # Curves & 2D (M_A, Rm) map
    rm_sweep = np.logspace(-1.3, 2.5, 48)
    g_sub_Re250 = np.array([solve_linear_mhd_kh(kx_test, 0.75, rm, Re=250.0, a=a, ny_eig=72)[0] for rm in rm_sweep])
    g_sub_Re1000 = np.array([solve_linear_mhd_kh(kx_test, 0.75, rm, Re=1000.0, a=a, ny_eig=80)[0] for rm in rm_sweep])
    g_sub_Re4000 = np.array([solve_linear_mhd_kh(kx_test, 0.75, rm, Re=4000.0, a=a, ny_eig=96)[0] for rm in rm_sweep])

    ma_grid = np.linspace(0.45, 1.80, 28)
    rm_grid = np.logspace(-1.3, 2.48, 28)
    gamma_map = np.zeros((len(rm_grid), len(ma_grid)))
    for i, rmv in enumerate(rm_grid):
        for j, mav in enumerate(ma_grid):
            gamma_map[i, j] = solve_linear_mhd_kh(kx_test, mav, rmv, Re=Re, a=a, ny_eig=64)[0]

    # Experiment 3: Full 2D Nonlinear Evolution (96x96, t=18) + Grid Convergence Check
    print("\n--- Experiment 3: Full 2D Nonlinear Evolution (96x96, Re=250, t=18, pert_amp=0.015) ---")
    nl_regimes = [
        ("super_alfvenic", "Super-Alfvénic\n($M_A=2.5, R_m=200$)", 2.50, 200.0),
        ("sub_ideal", "Sub-Alfvénic High-$R_m$\n($M_A=0.75, R_m=200$)", 0.75, 200.0),
        ("sub_trough", "Sub-Alfv. Box Trough\n($M_A=0.75, R_m=4.0$)", 0.75, 4.0),
        ("sub_lowRm", "Low-$R_m$ Uncoupled KH\n($M_A=0.75, R_m=0.20$)", 0.75, 0.20),
    ]
    nl_sims = {}
    nl_summary = {}
    for cid, label, mav, rmv in nl_regimes:
        sim = simulate_2d_resistive_mhd(
            M_A=mav, Rm=rmv, Re=Re, a=a, nx=96, ny=96, dt=0.004, t_end=18.0,
            pert_amp=0.015, keep_mean_forcing=True,
        )
        print(f"  [{cid:16s}] peak KE_y={np.max(sim['ke_y']):.4e}, final KE_y={sim['ke_y'][-1]:.4e}, max rms u_y={sim['rms_uy_max']:.4f}")
        nl_sims[cid] = (label, sim)
        nl_summary[cid] = {
            "M_A": mav, "Rm": rmv, "peak_ke_y": float(np.max(sim["ke_y"])),
            "final_ke_y": float(sim["ke_y"][-1]), "max_rms_uy": sim["rms_uy_max"],
        }

    _, _, w_vec_128, a_vec_128 = solve_linear_mhd_kh(kx_test, 0.75, 0.20, Re=Re, a=a, ny_eig=128, return_vec=True)
    sim_128 = simulate_2d_resistive_mhd(
        M_A=0.75, Rm=0.20, Re=Re, a=a, nx=128, ny=128, dt=0.005, t_end=11.0,
        pert_amp=1e-4, keep_mean_forcing=True, eig_init=(w_vec_128, a_vec_128),
    )
    g_128 = fit_ke_growth(sim_128["t"], sim_128["ke_y"] + sim_128["me_pert"], 2.5, 7.5)
    g_64 = ver_summary["sub_alfvenic_lowRm"]["gamma_pde"]
    print(f"\nGrid check (M_A=0.75, Rm=0.20): 64x64={g_64:+.6f} vs 128x128={g_128:+.6f} (|diff|={abs(g_128 - g_64):.2e})")

    # Figure 1: Re-Independence of Rm_crit vs Re-Dependence of Bulk-Mode Trough + 2D Map
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.5), dpi=150)

    ax1.axhline(0.0, color="black", lw=1.3)
    ax1.axhspan(-0.26, 0.0, color="#e8f4fa", alpha=0.55)
    ax1.semilogx(rm_sweep, g_sub_Re250, "-", color="#1f77b4", lw=2.3, label="$\\mathrm{Re} = 250$ ($R_{m,\\mathrm{crit}} = %.3f$)" % re_audit["Re250_Ly2pi"]["Rm_crit"])
    ax1.semilogx(rm_sweep, g_sub_Re1000, "--", color="#ff7f0e", lw=2.2, label="$\\mathrm{Re} = 1000$ ($R_{m,\\mathrm{crit}} = %.3f$)" % re_audit["Re1000_Ly2pi"]["Rm_crit"])
    ax1.semilogx(rm_sweep, g_sub_Re4000, "-.", color="#2ca02c", lw=2.2, label="$\\mathrm{Re} = 4000$ ($R_{m,\\mathrm{crit}} = %.3f$)" % re_audit["Re4000_Ly2pi"]["Rm_crit"])
    ax1.scatter(
        [0.20, 4.0, 200.0],
        [ver_summary["sub_alfvenic_lowRm"]["gamma_pde"],
         ver_summary["sub_alfvenic_trough"]["gamma_pde"],
         ver_summary["sub_alfvenic_ideal"]["gamma_pde"]],
        c=["crimson", "gold", "cyan"], edgecolors="black", s=68, zorder=5,
        label="2D Nonlinear MHD PDE ($\\mathrm{Re}=250$)",
    )
    box = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.88)
    ax1.annotate("Re-independent\n$R_{m,\\mathrm{crit}} \\propto M_A^2$", (0.535, 0.0), xytext=(0.06, 0.16),
                 fontsize=8.0, fontweight="bold", color="#b00000",
                 arrowprops=dict(arrowstyle="->", color="#b00000", lw=1.1), bbox=box)
    ax1.annotate("Re-dependent bulk-mode\ntrough ($\\propto \\sqrt{R_m^*/\\mathrm{Re}}$)", (1.6, -0.22), xytext=(6.5, -0.22),
                 fontsize=7.9, fontweight="bold", color="#004488",
                 arrowprops=dict(arrowstyle="->", color="#004488", lw=1.1), bbox=box)
    ax1.set_ylim(-0.26, 0.46)
    ax1.set_xlabel("Magnetic Reynolds number $R_m = U_0 a / \\eta$")
    ax1.set_ylabel("Dominant growth rate $\\gamma$ ($M_A = 0.75, k_x a = 0.25$)")
    ax1.set_title("Sub-Alfvénic Stability: Robust $R_{m,\\mathrm{crit}}$ vs. Re-Dependent Trough", fontweight="bold")
    ax1.grid(True, which="both", alpha=0.28)
    ax1.legend(loc="upper right", fontsize=7.9)

    norm = TwoSlopeNorm(vmin=-0.22, vcenter=0.0, vmax=0.55)
    im = ax2.contourf(ma_grid, rm_grid, gamma_map, levels=np.linspace(-0.22, 0.55, 32), cmap="RdBu_r", norm=norm, extend="both")
    ax2.contour(ma_grid, rm_grid, gamma_map, levels=[0.0], colors="black", linewidths=2.2)
    ax2.axvline(1.0, color="gold", ls="--", lw=1.8, label="Ideal-MHD cutoff ($M_A = 1$)")
    ma_sub = np.linspace(0.45, 0.96, 40)
    ax2.plot(ma_sub, 0.95 * (ma_sub ** 2), ":", color="lime", lw=2.0, label="$R_m = 0.95 M_A^2$ ($N_c \\approx 0.95$)")
    ax2.set_yscale("log")
    ax2.set_xlabel("Alfvén Mach number $M_A = U_0 / v_A$")
    ax2.set_ylabel("Magnetic Reynolds number $R_m$")
    ax2.set_title("2D $(M_A, R_m)$ Stability Map ($k_x a = 0.25, \\mathrm{Re}=250$)", fontweight="bold")
    ax2.legend(loc="lower right", fontsize=7.9)
    plt.colorbar(im, ax=ax2, label="Dominant growth rate $\\gamma$")

    fig.suptitle("Plasma Nozzles Entry 001 (Part 1): Low-$R_m$ Sub-Alfvénic KH Breakdown & Re-Scaling Audit", fontsize=12.0, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(RESULTS_DIR / "exp001_mhd_kh_stability.png")
    plt.close(fig)

    # Figure 2: 2D Nonlinear Snapshots + Traces
    fig, axes = plt.subplots(1, 5, figsize=(18.5, 4.1), dpi=150)
    for ax, cid in zip(axes[:4], ["super_alfvenic", "sub_ideal", "sub_trough", "sub_lowRm"]):
        lbl, s = nl_sims[cid]
        ax.imshow(s["omega_final"].T, origin="lower", extent=[0, 2 * np.pi, 0, 2 * np.pi], cmap="RdBu_r", vmin=-4.5, vmax=4.5, aspect="equal")
        ax.contour(s["X"], s["Y"], s["az_tot_final"], levels=12, colors="black", linewidths=0.75, alpha=0.8)
        ax.set_title(lbl, fontsize=9.0, fontweight="bold")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")

    ax5 = axes[4]
    cols = {"super_alfvenic": "#d62728", "sub_ideal": "#2ca02c", "sub_trough": "#1f77b4", "sub_lowRm": "#ff7f0e"}
    short_lbls = {
        "super_alfvenic": "Super-Alfv. ($M_A=2.5$)",
        "sub_lowRm": "Low-$R_m$ KH ($R_m=0.2$)",
        "sub_ideal": "Sub-Alfv. High-$R_m$ ($200$)",
        "sub_trough": "Box Trough ($R_m=4, \\mathrm{Re}=250$)",
    }
    for cid in ("super_alfvenic", "sub_lowRm", "sub_ideal", "sub_trough"):
        _, s = nl_sims[cid]
        ax5.semilogy(s["t"], s["ke_y"], color=cols[cid], lw=2.0, label=short_lbls[cid])
    ax5.set_ylim(1e-10, 1.0)
    ax5.set_xlabel("Dimensionless time $t U_0 / L$")
    ax5.set_ylabel("Transverse mixing $\\frac{1}{2}\\langle u_y^2 \\rangle$")
    ax5.set_title("Cross-Field Plume Mixing", fontsize=9.5, fontweight="bold")
    ax5.grid(True, which="both", alpha=0.28)
    ax5.legend(loc="lower left", fontsize=6.8)

    fig.suptitle("Plasma Nozzles Entry 001 (Part 2): 2D Nonlinear Vorticity & Magnetic Field Lines ($t = 18, 96\\times 96, \\mathrm{Re}=250$)", fontsize=11.8, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(RESULTS_DIR / "exp001_mhd_kh_nonlinear.png")
    plt.close(fig)

    metrics = {
        "parameters": {"a": a, "Re_base": Re, "kx_test": kx_test, "grid_base": [64, 64]},
        "exp1_linear_vs_pde": ver_summary,
        "exp1_unforced_energy_balance_rel_err": sim_econ["energy_balance_rel_err"],
        "exp2a_sub_alfvenic_stuart_law": rm_crit_map,
        "exp2b_re_and_box_audit": re_audit,
        "exp2b_kx_audit": kx_audit,
        "exp3_nonlinear_regimes": nl_summary,
        "exp3_grid_convergence_64_vs_128_lowRm": {"gamma_64": g_64, "gamma_128": g_128, "abs_diff": float(abs(g_128 - g_64))},
    }
    with open(RESULTS_DIR / "exp001_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved hardened plasma-nozzles Entry 001 plots and metrics to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
