"""Entry 008 (rocket-combustion): Chiral Swirl vs. Gasdynamic Shock Competition,
Even-Symmetry |u| Coupling & Zero-Hardware Staggered Stabilization of Rotating Shocks.

Extends Entry 007's verified 1D periodic compressible Euler ring (gamma = 1.25, L_a = 2.0,
zeta = 0.04, nu_res = 0.02, MUSCL-minmod + HLLC + SSP-RK2) with:
  1. Signed chiral azimuthal velocity coupling n_v * Phi(q_u) (swirled/directional injectors,
     matching Entries 004-006 identically in the linear limit).
  2. Even-symmetry speed coupling n_|u| * Phi(q_|u|) (non-swirled injectors whose heat release
     responds to flow speed |u| regardless of CW vs. CCW direction; Open Problem #3).
  3. Spatially varying causal Erlang-4 injector time lags tau(theta) in {tau_1, tau_2} with
     M alternating periods around the rim (Entry 005 Law #7 upgraded to shocked compressible Euler).

Experiments:
1. Verification — Exact Continuum Chiral Dispersion vs. Small-Amplitude Euler PDE (A = 1e-4):
   Verifies +1T (chiral-boosted) and -1T (chiral-damped) linear growth rates at n_p = 0.050,
   n_v = 0.020 against exact complex roots of s^2 + ((zeta - (n_p +/- n_v)*H(s))/L_a)*s + m^2 = 0.
2. Standing-to-Rotating-Shock Phase Transition (n_p = 0.068, t = 360):
   (2A) Chiral swirl sweep n_v in {0.000, 0.004, 0.007, 0.010, 0.014, 0.018, 0.024, 0.032}:
        Starts from BOTH pure standing (S_0 = 0.00) AND perturbed spinning (S_0 = 0.98) to map
        the exact critical chirality n_v* where swirled driving overcomes the >= sqrt(2) cubic
        Rankine-Hugoniot shock penalty and flips the standing attractor into a rotating detonation-
        like sawtooth shock (S -> 1), checking for hysteresis/bistability + N = 512 grid check!
   (2B) Non-swirled even-symmetry |u| coupling (n_v = 0, n_|u| in {0.02, 0.05, 0.08}):
        Tests whether |u| speed coupling breaks chirality or preserves standing dominance (S -> 0)
        while pumping 2T harmonics.
3. Rotating-Shock Control Shootout (n_p = 0.060, n_v = 0.025, A_0 = 0.055 = 6.9% p'/p_bar, t = 360):
   Compares five baffle-free configurations on the exact same violent rotating sawtooth shock:
     (1) Bare chiral Euler shock (baseline S ~ 1)
     (2) Zero-hardware M = 2 lag staggering (tau in {0.30, 2.50}) — Bragg-orientation trap!
     (3) Zero-hardware M = 5 odd-periodic lag staggering (tau in {0.30, 2.50}) — ZERO cavities/baffles!
     (4) Wide-mouth 80/20 (1T/2T) dual cavity ring (beta = 2%, sigma_o = 3.5%, uniform tau = 0.30)
     (5) Hybrid M = 5 odd staggering + 80/20 dual cavity ring
   Plus N = 512 grid verification on M = 5 vs. M = 2!
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root

from exp007_compressible_euler_steepening import (
    GAMMA,
    KAPPA_NL,
    L_A,
    OMEGA_HP,
    P_BAR,
    Q_S,
    TAU_BAR,
    ZETA,
    cons_from_prim,
    hllc_flux_periodic,
    make_dual_cavity_1d,
    make_isentropic_init,
    phi_flame,
    prim_from_cons,
)


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


# ---------------------------------------------------------------------------
# 1. Exact Continuum Chiral Dispersion Root Solver
# ---------------------------------------------------------------------------

def exact_chiral_root(m: int, spin_dir: int, n_p: float, n_v: float, tau_bar: float = TAU_BAR) -> complex:
    """Exact complex root s = alpha + i*omega of s^2 + ((zeta - (n_p + spin_dir*n_v)*H(s))/L_a)*s + m^2 = 0."""
    n_eff = n_p + spin_dir * n_v

    def res(x: np.ndarray) -> list[float]:
        s = complex(float(x[0]), float(x[1]))
        H = (s / (s + OMEGA_HP)) / ((1.0 + s * (tau_bar / 4.0)) ** 4)
        F = s * s + ((ZETA - n_eff * H) / L_A) * s + float(m * m)
        return [float(F.real), float(F.imag)]

    sol = root(res, np.array([(n_eff - ZETA) / (2.0 * L_A), float(m)]))
    if not sol.success:
        raise RuntimeError(f"Chiral root failed for m={m}, spin={spin_dir}: {sol.message}")
    return complex(float(sol.x[0]), float(sol.x[1]))


def make_staggered_tau_1d(n_cells: int, M_periods: int, tau_1: float = 0.30, tau_2: float = 2.50) -> np.ndarray:
    """Alternating causal injector time lags tau(theta) in {tau_1, tau_2} with M periods around the ring."""
    sec = np.floor(np.arange(n_cells) * (2 * M_periods) / n_cells).astype(int)
    return np.where(sec % 2 == 0, tau_1, tau_2)


# ---------------------------------------------------------------------------
# 2. Compressible Euler Ring Solver with Chiral n_v, Even |u|, and tau(theta)
# ---------------------------------------------------------------------------

def simulate_chiral_euler_ring(
    n_cells: int = 256,
    t_end: float = 360.0,
    cfl: float = 0.38,
    zeta: float = ZETA,
    n_p: float = 0.068,
    n_v: float = 0.0,
    n_uabs: float = 0.0,
    tau_arr: np.ndarray | float = TAU_BAR,
    linear_flame: bool = False,
    beta_local: np.ndarray | float = 0.0,
    omega_H_arr: np.ndarray | float = 1.0,
    xi_H_arr: np.ndarray | float = 0.14,
    c_nl_arr: np.ndarray | float = 0.0,
    init_type: str = "standing_1T_pure",
    A_eq: float = 0.035,
    record_stride: int = 8,
) -> dict:
    """1D periodic compressible Euler ring with chiral velocity coupling, even |u| coupling, tau(theta), and cavities."""
    dth = 2.0 * np.pi / n_cells
    theta = (np.arange(n_cells) + 0.5) * dth
    nu_ent = zeta / L_A if zeta > 0.0 else 0.0

    if init_type == "spin_1T_minus":
        u0 = -A_eq * np.cos(theta)
        c0 = 1.0 - 0.5 * (GAMMA - 1.0) * u0
        rho0 = c0 ** (2.0 / (GAMMA - 1.0))
        p0 = P_BAR * (rho0 ** GAMMA)
    else:
        rho0, u0, p0 = make_isentropic_init(theta, init_type, A_eq)

    U = cons_from_prim(rho0, u0, p0)

    tau_loc = np.broadcast_to(np.asarray(tau_arr, dtype=float), (n_cells,)).copy()
    inv_tau_s = 4.0 / tau_loc

    b_loc = np.broadcast_to(np.asarray(beta_local, dtype=float), (n_cells,)).copy()
    has_cav = bool(np.any(b_loc > 0.0))
    beta_s = b_loc * L_A
    wH = np.broadcast_to(np.asarray(omega_H_arr, dtype=float), (n_cells,)).copy()
    xiH = np.broadcast_to(np.asarray(xi_H_arr, dtype=float), (n_cells,)).copy()
    c_nl = np.broadcast_to(np.asarray(c_nl_arr, dtype=float), (n_cells,)).copy()
    inv_beta_s = np.where(beta_s > 0.0, 1.0 / np.maximum(beta_s, 1e-30), 0.0)

    v_r = np.zeros(n_cells)
    p_c = (p0 - P_BAR).copy() if has_cav else np.zeros(n_cells)

    # DC-blocked HPF + Erlang-4 states for pressure, signed velocity u, and even speed |u|
    p_bf = np.zeros(n_cells)
    u_bf = np.zeros(n_cells)
    ua_bf = np.zeros(n_cells)
    qp = np.zeros((4, n_cells))
    qu = np.zeros((4, n_cells))
    qua = np.zeros((4, n_cells))

    use_v = abs(n_v) > 0.0
    use_ua = abs(n_uabs) > 0.0

    def rhs(U_s, vr_s, pc_s, pbf_s, ubf_s, uabf_s, qp_s, qu_s, qua_s):
        rho, u, p, _ = prim_from_cons(U_s)
        F = hllc_flux_periodic(U_s)
        div_F = (F - np.roll(F, 1, axis=1)) / dth

        Pf = p - P_BAR
        dpbf = OMEGA_HP * (Pf - pbf_s)
        p_hp = Pf - pbf_s
        dqp = np.empty_like(qp_s)
        dqp[0] = inv_tau_s * (p_hp - qp_s[0])
        dqp[1] = inv_tau_s * (qp_s[0] - qp_s[1])
        dqp[2] = inv_tau_s * (qp_s[1] - qp_s[2])
        dqp[3] = inv_tau_s * (qp_s[2] - qp_s[3])

        q_tot = n_p * (qp_s[3] if linear_flame else phi_flame(qp_s[3]))

        if use_v:
            dubf = OMEGA_HP * (u - ubf_s)
            u_hp = u - ubf_s
            dqu = np.empty_like(qu_s)
            dqu[0] = inv_tau_s * (u_hp - qu_s[0])
            dqu[1] = inv_tau_s * (qu_s[0] - qu_s[1])
            dqu[2] = inv_tau_s * (qu_s[1] - qu_s[2])
            dqu[3] = inv_tau_s * (qu_s[2] - qu_s[3])
            q_tot = q_tot + n_v * (qu_s[3] if linear_flame else phi_flame(qu_s[3]))
        else:
            dubf = np.zeros_like(ubf_s)
            dqu = np.zeros_like(qu_s)

        if use_ua:
            u_abs = np.abs(u)
            duabf = OMEGA_HP * (u_abs - uabf_s)
            ua_hp = u_abs - uabf_s
            dqua = np.empty_like(qua_s)
            dqua[0] = inv_tau_s * (ua_hp - qua_s[0])
            dqua[1] = inv_tau_s * (qua_s[0] - qua_s[1])
            dqua[2] = inv_tau_s * (qua_s[1] - qua_s[2])
            dqua[3] = inv_tau_s * (qua_s[2] - qua_s[3])
            q_tot = q_tot + n_uabs * (qua_s[3] if linear_flame else phi_flame(qua_s[3]))
        else:
            duabf = np.zeros_like(uabf_s)
            dqua = np.zeros_like(qua_s)

        Q_flame_E = (1.0 / (GAMMA - 1.0)) * (1.0 / L_A) * q_tot

        if has_cav:
            dvr = beta_s * (wH ** 2) * (Pf - pc_s) - (2.0 * xiH * wH + c_nl * np.abs(vr_s)) * vr_s
            dpc = vr_s * inv_beta_s
        else:
            dvr = np.zeros_like(vr_s)
            dpc = np.zeros_like(pc_s)

        u_ext = zeta * Pf + vr_s
        E_cur = p / (GAMMA - 1.0) + 0.5 * rho * u * u
        s_ent = nu_ent * (rho - (p / P_BAR) ** (1.0 / GAMMA)) if nu_ent > 0.0 else 0.0

        S_U = np.vstack([
            -(1.0 / L_A) * rho * u_ext - s_ent,
            -(1.0 / L_A) * rho * u * u_ext - s_ent * u,
            -(1.0 / L_A) * (E_cur + p) * u_ext - 0.5 * s_ent * u * u + Q_flame_E,
        ])
        return -div_F + S_U, dvr, dpc, dpbf, dubf, duabf, dqp, dqu, dqua

    t_list, e_ac_list, e_wave_list, spin_list, spin_signed_list = [], [], [], [], []
    ppeak_list, e1_list, e2_list, e3_list, max_grad_list = [], [], [], [], []

    def record(t_val: float):
        _, u_f, p_f, _ = prim_from_cons(U)
        P_f = p_f - P_BAR
        # Separate oscillatory wave energy from O(A^2) mean acoustic-streaming swirl
        P_osc = P_f - float(np.mean(P_f))
        u_osc = u_f - float(np.mean(u_f))
        e_ac = 0.5 * float(np.mean(P_f * P_f + u_f * u_f))
        e_wave = 0.5 * float(np.mean(P_osc * P_osc + u_osc * u_osc))
        s_signed = float(np.mean(P_osc * u_osc)) / max(e_wave, 1e-30)

        P_hat = np.fft.rfft(P_osc) / (0.5 * n_cells)
        e1 = float(np.abs(P_hat[1]) ** 2)
        e2 = float(np.abs(P_hat[2]) ** 2)
        e3 = float(np.abs(P_hat[3]) ** 2)
        du_dth = (np.roll(u_f, -1) - np.roll(u_f, 1)) / (2.0 * dth)

        t_list.append(t_val)
        e_ac_list.append(e_ac)
        e_wave_list.append(e_wave)
        spin_list.append(abs(s_signed))
        spin_signed_list.append(s_signed)
        ppeak_list.append(float(np.max(np.abs(P_f))))
        e1_list.append(e1)
        e2_list.append(e2)
        e3_list.append(e3)
        max_grad_list.append(float(np.max(-du_dth)))

    t = 0.0
    step = 0
    record(0.0)
    while t < t_end - 1e-12:
        _, u_c, _, c_c = prim_from_cons(U)
        dt = min(cfl * dth / max(float(np.max(np.abs(u_c) + c_c)), 1.0), 0.015, t_end - t)
        k1 = rhs(U, v_r, p_c, p_bf, u_bf, ua_bf, qp, qu, qua)
        U1 = U + dt * k1[0]
        vr1 = v_r + dt * k1[1]; pc1 = p_c + dt * k1[2]
        pbf1 = p_bf + dt * k1[3]; ubf1 = u_bf + dt * k1[4]; uabf1 = ua_bf + dt * k1[5]
        qp1 = qp + dt * k1[6]; qu1 = qu + dt * k1[7]; qua1 = qua + dt * k1[8]
        k2 = rhs(U1, vr1, pc1, pbf1, ubf1, uabf1, qp1, qu1, qua1)
        U = 0.5 * (U + U1 + dt * k2[0])
        v_r = 0.5 * (v_r + vr1 + dt * k2[1]); p_c = 0.5 * (p_c + pc1 + dt * k2[2])
        p_bf = 0.5 * (p_bf + pbf1 + dt * k2[3]); u_bf = 0.5 * (u_bf + ubf1 + dt * k2[4]); ua_bf = 0.5 * (ua_bf + uabf1 + dt * k2[5])
        qp = 0.5 * (qp + qp1 + dt * k2[6]); qu = 0.5 * (qu + qu1 + dt * k2[7]); qua = 0.5 * (qua + qua1 + dt * k2[8])
        t += dt
        step += 1
        if step % record_stride == 0 or t >= t_end - 1e-12:
            record(t)

    rho_fin, u_fin, p_fin, _ = prim_from_cons(U)
    return {
        "theta": theta,
        "P_final": p_fin - P_BAR,
        "u_final": u_fin,
        "rho_mean": float(np.mean(rho_fin)),
        "u_mean": float(np.mean(u_fin)),
        "t": np.array(t_list),
        "e_ac": np.array(e_ac_list),
        "e_wave": np.array(e_wave_list),
        "spin": np.array(spin_list),
        "spin_signed": np.array(spin_signed_list),
        "p_peak": np.array(ppeak_list),
        "e1": np.array(e1_list),
        "e2": np.array(e2_list),
        "e3": np.array(e3_list),
        "max_neg_grad": np.array(max_grad_list),
    }


def summarize_run(r: dict) -> dict:
    m = r["t"] >= 0.80 * r["t"][-1]
    m_mid = (r["t"] >= 0.50 * r["t"][-1]) & (r["t"] <= 0.65 * r["t"][-1])
    e1_m = float(np.mean(r["e1"][m]))
    e2_m = float(np.mean(r["e2"][m]))
    e3_m = float(np.mean(r["e3"][m]))
    ew_late = float(np.mean(r["e_wave"][m]))
    ew_mid = float(np.mean(r["e_wave"][m_mid]))
    return {
        "spin_late": float(np.mean(r["spin"][m])),
        "spin_signed_late": float(np.mean(r["spin_signed"][m])),
        "p_peak_late": float(np.mean(r["p_peak"][m])),
        "p_over_pbar_pct": float(np.mean(r["p_peak"][m]) / P_BAR * 100.0),
        "e_wave_late": ew_late,
        "e_wave_late_over_mid": ew_late / max(ew_mid, 1e-30),
        "harmonic_2T_over_1T_pct": e2_m / max(e1_m, 1e-30) * 100.0,
        "harmonic_3T_over_1T_pct": e3_m / max(e1_m, 1e-30) * 100.0,
        "max_neg_grad_late": float(np.mean(r["max_neg_grad"][m])),
        "rho_mean": r["rho_mean"],
        "u_mean": r["u_mean"],
    }


# ---------------------------------------------------------------------------
# 3. Main Experiments & Figures
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Entry 008 (rocket-combustion): Chiral Swirl vs. Shock Competition & Rotating-Shock Control ===\n")

    # ------------------------------------------------------------------
    # Experiment 1: Exact Continuum Chiral Dispersion vs Small-Amplitude Euler PDE
    # ------------------------------------------------------------------
    print("--- Experiment 1: Exact Continuum Chiral Dispersion vs. Euler PDE (n_p=0.050, n_v=0.020, A=1e-4) ---")
    exp1_ver = {}
    for label, sdir, itype in [("1T_plus", +1, "spin_1T"), ("1T_minus", -1, "spin_1T_minus")]:
        s_ex = exact_chiral_root(1, sdir, n_p=0.050, n_v=0.020)
        rr = simulate_chiral_euler_ring(
            n_cells=256, t_end=50.0, n_p=0.050, n_v=0.020, linear_flame=True,
            init_type=itype, A_eq=1e-4, record_stride=4,
        )
        m = rr["t"] >= 8.0
        alpha_pde = float(np.polyfit(rr["t"][m], 0.5 * np.log(np.maximum(rr["e_wave"][m], 1e-30)), 1)[0])
        diff = abs(alpha_pde - s_ex.real)
        print(f"  [{label:8s}] Exact alpha={s_ex.real:+.6f} (w={s_ex.imag:.5f}) | Euler PDE alpha={alpha_pde:+.6f} | |diff|={diff:.2e}")
        exp1_ver[label] = {"alpha_exact": s_ex.real, "omega_exact": s_ex.imag, "alpha_pde": alpha_pde, "abs_diff": diff}

    # ------------------------------------------------------------------
    # Experiment 2A: Standing-to-Rotating-Shock Phase Transition (n_p = 0.068, n_v sweep)
    # ------------------------------------------------------------------
    print("\n--- Experiment 2A: Standing -> Rotating-Shock Phase Transition (n_p=0.068, t=360, N=256) ---")
    nv_grid = [0.000, 0.004, 0.007, 0.010, 0.014, 0.018, 0.024, 0.032]
    exp2a_stand = []
    exp2a_spin = []
    waveforms_2a = {}
    for nv_v in nv_grid:
        r_st = simulate_chiral_euler_ring(
            n_cells=256, t_end=360.0, n_p=0.068, n_v=nv_v,
            init_type="standing_1T_pure", A_eq=0.035, record_stride=10,
        )
        r_sp = simulate_chiral_euler_ring(
            n_cells=256, t_end=360.0, n_p=0.068, n_v=nv_v,
            init_type="spin_1T_perturbed", A_eq=0.035, record_stride=10,
        )
        st_st = summarize_run(r_st)
        st_sp = summarize_run(r_sp)
        exp2a_stand.append({"n_v": nv_v, **st_st})
        exp2a_spin.append({"n_v": nv_v, **st_sp})
        if nv_v in (0.000, 0.014, 0.032):
            waveforms_2a[f"nv_{nv_v:.3f}"] = r_st
        print(
            f"  n_v={nv_v:.3f} | from STAND: S={st_st['spin_signed_late']:+.3f}, P={st_st['p_over_pbar_pct']:4.2f}%, E2/E1={st_st['harmonic_2T_over_1T_pct']:4.1f}% | "
            f"from SPIN: S={st_sp['spin_signed_late']:+.3f}, P={st_sp['p_over_pbar_pct']:4.2f}%, E2/E1={st_sp['harmonic_2T_over_1T_pct']:4.1f}%"
        )

    # Grid refinement N=512 at two key points straddling the transition (n_v = 0.007 and n_v = 0.024)
    exp2a_grid512 = {}
    print("\n  [Grid check N=512 straddling transition (from standing, t=360)]")
    for nv_chk in (0.007, 0.024):
        r5 = simulate_chiral_euler_ring(
            n_cells=512, t_end=360.0, n_p=0.068, n_v=nv_chk,
            init_type="standing_1T_pure", A_eq=0.035, record_stride=16,
        )
        st5 = summarize_run(r5)
        exp2a_grid512[f"nv_{nv_chk:.3f}"] = st5
        print(f"    N=512 n_v={nv_chk:.3f}: S={st5['spin_signed_late']:+.4f}, P={st5['p_over_pbar_pct']:.2f}%, E2/E1={st5['harmonic_2T_over_1T_pct']:.1f}%")

    # ------------------------------------------------------------------
    # Experiment 2B: Non-Swirled Even-Symmetry |u| Coupling (Open Problem #3)
    # ------------------------------------------------------------------
    print("\n--- Experiment 2B: Non-Swirled Even-Symmetry |u| Speed Coupling (n_p=0.068, n_v=0, t=360) ---")
    exp2b_uabs = {}
    for nua in (0.02, 0.05, 0.08):
        r_st = simulate_chiral_euler_ring(
            n_cells=256, t_end=360.0, n_p=0.068, n_v=0.0, n_uabs=nua,
            init_type="standing_1T_seeded", A_eq=0.035, record_stride=10,
        )
        r_sp = simulate_chiral_euler_ring(
            n_cells=256, t_end=360.0, n_p=0.068, n_v=0.0, n_uabs=nua,
            init_type="spin_1T_perturbed", A_eq=0.035, record_stride=10,
        )
        st_st = summarize_run(r_st)
        st_sp = summarize_run(r_sp)
        exp2b_uabs[f"n_uabs_{nua:.2f}"] = {"from_standing_seeded": st_st, "from_spin_perturbed": st_sp}
        print(
            f"  n_|u|={nua:.2f} | from stand(S0=0.17): S={st_st['spin_late']:.4f}, P={st_st['p_over_pbar_pct']:.2f}%, E2/E1={st_st['harmonic_2T_over_1T_pct']:.1f}% | "
            f"from spin(S0=0.98): S={st_sp['spin_late']:.4f}, P={st_sp['p_over_pbar_pct']:.2f}%, E2/E1={st_sp['harmonic_2T_over_1T_pct']:.1f}%"
        )

    # ------------------------------------------------------------------
    # Experiment 3: Rotating-Shock Control Shootout (n_p=0.060, n_v=0.025, A_0=0.055 = 6.9% p'/p_bar)
    # ------------------------------------------------------------------
    print("\n--- Experiment 3: Rotating-Shock Control Shootout (n_p=0.060, n_v=0.025, A_0=6.9%, t=360) ---")
    b_80, wH_80, xiH_80, cnl_80 = make_dual_cavity_1d(256, 0.02, 0.80, xi_mult=1.0, C_d=0.65, sigma_o=0.035)
    tau_M2 = make_staggered_tau_1d(256, M_periods=2, tau_1=0.30, tau_2=2.50)
    tau_M5 = make_staggered_tau_1d(256, M_periods=5, tau_1=0.30, tau_2=2.50)

    ctrl_specs = [
        ("bare_chiral_shock",    TAU_BAR, 0.0,  1.0,   0.14,  0.0),
        ("stagger_M2_bragg",     tau_M2,  0.0,  1.0,   0.14,  0.0),
        ("stagger_M5_odd",       tau_M5,  0.0,  1.0,   0.14,  0.0),
        ("cav_80_20_dual",       TAU_BAR, b_80, wH_80, xiH_80, cnl_80),
        ("hybrid_M5_plus_80_20", tau_M5,  b_80, wH_80, xiH_80, cnl_80),
    ]
    exp3_summary = {}
    exp3_runs = {}
    for cid, tau_c, b_c, w_c, xi_c, cnl_c in ctrl_specs:
        rr = simulate_chiral_euler_ring(
            n_cells=256, t_end=360.0, n_p=0.060, n_v=0.025,
            tau_arr=tau_c, beta_local=b_c, omega_H_arr=w_c, xi_H_arr=xi_c, c_nl_arr=cnl_c,
            init_type="spin_1T", A_eq=0.055, record_stride=8,
        )
        st = summarize_run(rr)
        exp3_summary[cid] = st
        exp3_runs[cid] = rr
        print(
            f"  [{cid:20s}] P_peak={st['p_peak_late']:.4f} ({st['p_over_pbar_pct']:4.2f}%) | "
            f"E_wave={st['e_wave_late']:.2e} (late/mid={st['e_wave_late_over_mid']:.3f}) | "
            f"S={st['spin_signed_late']:+.3f} | E2/E1={st['harmonic_2T_over_1T_pct']:5.1f}%"
        )

    # N=512 grid check on zero-hardware M=2 Bragg trap vs M=5 odd staggering
    print("\n  [Grid check N=512 on M=2 Bragg trap vs M=5 odd staggering (t=360)]")
    exp3_grid512 = {}
    for cid, M_p in [("stagger_M2_bragg", 2), ("stagger_M5_odd", 5)]:
        tau_512 = make_staggered_tau_1d(512, M_periods=M_p, tau_1=0.30, tau_2=2.50)
        r5 = simulate_chiral_euler_ring(
            n_cells=512, t_end=360.0, n_p=0.060, n_v=0.025,
            tau_arr=tau_512, init_type="spin_1T", A_eq=0.055, record_stride=16,
        )
        st5 = summarize_run(r5)
        exp3_grid512[cid] = st5
        print(f"    N=512 [{cid:16s}] P_peak={st5['p_peak_late']:.4f} ({st5['p_over_pbar_pct']:4.2f}%) | E_wave={st5['e_wave_late']:.2e} (late/mid={st5['e_wave_late_over_mid']:.3f}) | S={st5['spin_signed_late']:+.3f}")

    # ------------------------------------------------------------------
    # Figure 1: Standing -> Rotating-Shock Phase Transition (S(n_v) + Waveforms)
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.4), dpi=150)

    nvs = [x["n_v"] for x in exp2a_stand]
    s_from_st = [x["spin_signed_late"] for x in exp2a_stand]
    s_from_sp = [x["spin_signed_late"] for x in exp2a_spin]
    ax1.plot(nvs, s_from_st, "o-", color="#d62728", lw=2.3, ms=6.5, label="Euler from pure standing ($S_0 = 0.00$)")
    ax1.plot(nvs, s_from_sp, "s--", color="#1f77b4", lw=2.1, ms=6.0, label="Euler from perturbed spin ($S_0 = 0.98$)")
    ax1.scatter(
        [0.007, 0.024],
        [exp2a_grid512["nv_0.007"]["spin_signed_late"], exp2a_grid512["nv_0.024"]["spin_signed_late"]],
        marker="*", s=140, color="#2ca02c", zorder=6, label="$N=512$ grid verification",
    )
    ax1.axhline(0.0, color="gray", lw=0.9, ls=":")
    ax1.axhline(1.0, color="gray", lw=0.9, ls=":")
    ax1.set_xlabel("Chiral azimuthal velocity coupling $n_v$ ($n_p = 0.068$)")
    ax1.set_ylabel("Late-time signed spin index $S_{\\mathrm{late}}$ ($0=$ stand, $+1=$ CW shock)")
    ax1.set_ylim(-0.08, 1.12)
    ax1.set_title("Standing $\\rightarrow$ Rotating-Shock Phase Transition", fontweight="bold")
    ax1.grid(True, alpha=0.28)
    ax1.legend(loc="lower right", fontsize=8.3, framealpha=0.92)

    wf_styles = [
        ("nv_0.000", "-",  "#d62728", "$n_v = 0.000$ (standing attractor, $S \\approx 0$)"),
        ("nv_0.014", "--", "#ff7f0e", "$n_v = 0.014$ (transition / mixed regime)"),
        ("nv_0.032", "-",  "#1f77b4", "$n_v = 0.032$ (rotating sawtooth shock, $S \\approx 1$)"),
    ]
    for key, ls, col, lbl in wf_styles:
        rr = waveforms_2a[key]
        ax2.plot(rr["theta"], rr["P_final"] / P_BAR * 100.0, ls, color=col, lw=2.2, label=lbl)
    ax2.set_xlabel("Azimuthal angle $\\theta$ (rad)")
    ax2.set_ylabel("Final pressure waveform $p'/\\bar{p}$ (%) at $t = 360$")
    ax2.set_title("Waveforms Across the Chiral-Shock Transition", fontweight="bold")
    ax2.grid(True, alpha=0.28)
    ax2.legend(loc="upper right", fontsize=8.2, framealpha=0.92)

    fig.suptitle(
        "Rocket Combustion Entry 008 (Part 1): Chiral Swirl vs. Rankine-Hugoniot Shock Competition",
        fontsize=11.8, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(RESULTS_DIR / "exp008_chiral_shock_transition.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2: Rotating-Shock Control Shootout (Wave Energy Decay + Final Waveforms)
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.4), dpi=150)

    c_styles = {
        "bare_chiral_shock":    ("-",  "#d62728", "Bare chiral shock ($S \\approx 1$)"),
        "stagger_M2_bragg":     ("--", "#ff7f0e", "Zero-HW $M=2$ stagger (Bragg trap!)"),
        "stagger_M5_odd":       ("-",  "#2ca02c", "Zero-HW $M=5$ odd stagger (no cavities!)"),
        "cav_80_20_dual":       ("-.", "#1f77b4", "80/20 dual cavity ring ($\\beta=2\\%, \\sigma_o=3.5\\%$)"),
        "hybrid_M5_plus_80_20": (":",  "#9467bd", "Hybrid $M=5$ stagger + 80/20 ring"),
    }
    for cid, (ls, col, lbl) in c_styles.items():
        rr = exp3_runs[cid]
        ax1.semilogy(rr["t"], np.maximum(rr["e_wave"], 1e-10), ls, color=col, lw=2.2, label=lbl)
    ax1.set_xlabel("Dimensionless time $t \\bar{c} / R$")
    ax1.set_ylabel("Oscillatory wave energy $E_{\\mathrm{wave}}(t)$ (log)")
    ax1.set_ylim(1e-9, 5e-3)
    ax1.set_title("Control of a $6.9\\%$ Chiral Rotating Shock ($n_p=0.060, n_v=0.025$)", fontweight="bold")
    ax1.grid(True, which="both", alpha=0.24)
    ax1.legend(loc="lower left", fontsize=8.0, framealpha=0.92)

    for cid, (ls, col, lbl) in c_styles.items():
        rr = exp3_runs[cid]
        ax2.plot(rr["theta"], rr["P_final"] / P_BAR * 100.0, ls, color=col, lw=2.1, label=lbl)
    ax2.set_xlabel("Azimuthal angle $\\theta$ (rad)")
    ax2.set_ylabel("Final pressure waveform $p'/\\bar{p}$ (%) at $t = 360$")
    ax2.set_title("Final Shocked Waveforms at $t = 360$", fontweight="bold")
    ax2.grid(True, alpha=0.28)
    ax2.legend(loc="upper right", fontsize=8.0, framealpha=0.92)

    fig.suptitle(
        "Rocket Combustion Entry 008 (Part 2): Zero-Hardware Odd Lag Staggering vs. 80/20 Cavities Under Rotating Shocks",
        fontsize=11.6, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(RESULTS_DIR / "exp008_rotating_shock_control.png")
    plt.close(fig)

    metrics = {
        "parameters": {
            "gamma": GAMMA, "L_a": L_A, "zeta": ZETA, "nu_res": ZETA / L_A,
            "omega_hp": OMEGA_HP, "tau_bar": TAU_BAR, "q_s": Q_S, "kappa": KAPPA_NL,
            "stagger_tau_1": 0.30, "stagger_tau_2": 2.50,
        },
        "exp1_chiral_linear_verification": exp1_ver,
        "exp2a_chiral_sweep_from_standing": exp2a_stand,
        "exp2a_chiral_sweep_from_spinning": exp2a_spin,
        "exp2a_grid512_check": exp2a_grid512,
        "exp2b_even_uabs_coupling": exp2b_uabs,
        "exp3_rotating_shock_control": exp3_summary,
        "exp3_grid512_check": exp3_grid512,
    }
    with open(RESULTS_DIR / "exp008_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved Entry 008 plots and metrics to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
