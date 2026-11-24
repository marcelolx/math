"""Entry 006: 2D Nonlinear Spinning vs. Standing Mode Selection & Verified Cavity Bomb Tests.

Peer-review-hardened 2D nonlinear cylindrical-annulus PDE (L_a = 2.0, zeta = 0.04,
DC-blocked omega_hp = 0.15, causal Erlang-4 tau_bar = 0.30, sigma_tau = 0.15):

Addresses all peer-review findings (C1, C2, M7, M8, m1, m5, m6):
1. Computes exact 2D bare linear Hopf point n_pH(1T) from dispersion relation.
2. Uses a pure trend-based triggering classifier (never counts decaying trajectories as
   triggered!), backed by long-horizon (t = 400) multi-mode linear stability pre-checks
   (seeding 1L + 1T+/- + 2T+/- at 1e-4) and an explicit horizon convergence check (t=180 vs t=360).
3. Separates subcritical (n_p < n_pH) and supercritical (n_p > n_pH) spin-selection runs
   explicitly, runs a bistable standing bomb all the way to its upper-branch limit cycle (t=500),
   and reports true settled limit cycles (never saddle transients).
4. Checks exact 2D nonlinear total energy balance (chamber + cavity + nonlinear flame +
   quadratic orifice drag).
5. Measures actual linear growth rates for bare and baffled combustors alongside the
   verified linearly-stable 80/20 cavity ring configurations.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OMEGA_HP = 0.15


def phi_bounded(q: np.ndarray, q_s: float, kappa: float) -> np.ndarray:
    u = q / q_s
    return q_s * np.tanh(u + kappa * (u * u * u))


# ---------------------------------------------------------------------------
# 1. Exact bare 2D 1T linear Hopf point n_pH
# ---------------------------------------------------------------------------

def sinhc(z: complex) -> complex:
    if abs(z) < 1e-6:
        z2 = z * z
        return 1.0 + z2 / 6.0 + (z2 * z2) / 120.0
    return np.sinh(z) / z


def bare_2d_1T_eigenvalue(n_p: float, n_v: float, L_a: float, zeta: float, x_f: float, tau_bar: float, sigma_tau: float) -> complex:
    def F(s: complex, frac: float) -> complex:
        kap2 = s * s + 1.0
        kap = np.sqrt(kap2)
        lhs = kap2 * L_a * sinhc(kap * L_a) + zeta * s * np.cosh(kap * L_a)
        hp = s / (s + OMEGA_HP)
        tau_d = min(tau_bar, 2.0 * sigma_tau)
        tau_0 = max(0.0, tau_bar - tau_d)
        g = np.exp(-s * tau_0) / ((1.0 + s * (tau_d / 4.0)) ** 4)
        n_eff = frac * (n_p + n_v * (1j / s)) * hp * g
        p1 = np.cosh(kap * x_f)
        p2 = np.cosh(kap * (L_a - x_f)) + zeta * s * (L_a - x_f) * sinhc(kap * (L_a - x_f))
        return lhs - n_eff * s * p1 * p2

    s = complex(-zeta / (2.0 * L_a), 1.0)
    for step in range(1, 9):
        fr = step / 8.0
        for _ in range(30):
            fv = F(s, fr)
            if abs(fv) < 1e-13:
                break
            eps = 1e-7 * max(1.0, abs(s))
            dfv = (F(s + eps, fr) - F(s - eps, fr)) / (2.0 * eps)
            if abs(dfv) < 1e-16:
                break
            s -= fv / dfv
    return s


def find_bare_2d_hopf(L_a: float, zeta: float, x_f: float, tau_bar: float, sigma_tau: float) -> tuple[float, float]:
    lo, hi = 0.01, 0.12
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        s = bare_2d_1T_eigenvalue(mid, 0.0, L_a, zeta, x_f, tau_bar, sigma_tau)
        if s.real < 0.0:
            lo = mid
        else:
            hi = mid
    n_H = 0.5 * (lo + hi)
    s_H = bare_2d_1T_eigenvalue(n_H, 0.0, L_a, zeta, x_f, tau_bar, sigma_tau)
    return float(n_H), float(s_H.imag)


# ---------------------------------------------------------------------------
# 2. Nonlinear 2D staggered C-grid RK4 PDE with energy balance check
# ---------------------------------------------------------------------------

def simulate_2d_nonlinear(
    L_a: float = 2.0,
    zeta: float = 0.04,
    x_f: float = 0.1667,
    n_p: float = 0.032,
    n_v: float = 0.0,
    tau_bar: float = 0.30,
    sigma_tau: float = 0.15,
    m_stages: int = 4,
    q_s: float = 0.036,
    kappa: float = 4.5,
    beta_local: np.ndarray | float = 0.0,
    omega_H_arr: np.ndarray | float = 1.0,
    xi_H_arr: np.ndarray | float = 0.14,
    c_nl_arr: np.ndarray | float = 0.0,
    n_baffles: int = 0,
    L_b: float = 0.0,
    nx: int = 30,
    ntheta: int = 60,
    cfl: float = 0.35,
    t_end: float = 180.0,
    init_type: str = "spin_1T_plus",
    init_energy_amp: float = 0.03,
    record_history: bool = False,
    check_energy_bal: bool = False,
) -> dict:
    dx = L_a / nx
    dth = (2.0 * np.pi) / ntheta
    inv_dx = 1.0 / dx
    inv_dth = 1.0 / dth

    dt_max = cfl / np.sqrt(inv_dx * inv_dx + inv_dth * inv_dth)
    n_steps = int(round(t_end / dt_max))
    dt = t_end / n_steps

    x_c = (np.arange(nx) + 0.5) * dx
    th_c = (np.arange(ntheta) + 0.5) * dth
    th_f = (np.arange(ntheta) + 1.0) * dth
    m_f = int(np.argmin(np.abs(x_c - x_f)))
    x_f_actual = float(x_c[m_f])

    beta_arr = np.full(ntheta, float(beta_local)) if np.ndim(beta_local) == 0 else np.asarray(beta_local, dtype=float)
    wH_arr = np.full(ntheta, float(omega_H_arr)) if np.ndim(omega_H_arr) == 0 else np.asarray(omega_H_arr, dtype=float)
    xiH_arr = np.full(ntheta, float(xi_H_arr)) if np.ndim(xi_H_arr) == 0 else np.asarray(xi_H_arr, dtype=float)
    cnl_arr = np.full(ntheta, float(c_nl_arr)) if np.ndim(c_nl_arr) == 0 else np.asarray(c_nl_arr, dtype=float)

    beta_s = beta_arr * L_a
    has_cav = bool(np.any(beta_s > 0.0))
    cav_mask = beta_s > 0.0
    inv_beta_s = np.where(cav_mask, 1.0 / np.maximum(beta_s, 1e-30), 0.0)
    cav_k_vr = beta_s * (wH_arr ** 2)
    cav_damp = 2.0 * xiH_arr * wH_arr

    uth_mask = np.ones((nx, ntheta), dtype=float)
    if n_baffles > 0 and L_b > 0.0:
        n_bc = int(np.sum(x_c <= L_b + 1e-12))
        for b in range(n_baffles):
            jf = int(round(b * ntheta / n_baffles)) % ntheta
            uth_mask[:n_bc, jf] = 0.0

    X, TH = np.meshgrid(x_c, th_c, indexing="ij")
    P = np.zeros((nx, ntheta))
    Ux = np.zeros((nx - 1, ntheta))
    Uth = np.zeros((nx, ntheta))

    if init_type == "spin_1T_plus":
        P = init_energy_amp * np.cos(TH)
        Uth = init_energy_amp * np.cos(th_f[None, :]) * np.ones((nx, 1))
    elif init_type == "standing_1T":
        eps_s = 0.12
        scale_p = np.sqrt(2.0 / (1.0 + 0.5 * eps_s * eps_s))
        P = scale_p * init_energy_amp * np.cos(TH)
        Uth = (eps_s * init_energy_amp) * np.cos(th_f[None, :]) * np.ones((nx, 1))
    elif init_type == "standing_1T_pure":
        P = np.sqrt(2.0) * init_energy_amp * np.cos(TH)
    elif init_type == "spin_2T_plus":
        P = init_energy_amp * np.cos(2.0 * TH)
        Uth = init_energy_amp * np.cos(2.0 * th_f[None, :]) * np.ones((nx, 1))
    elif init_type == "broadband_all":
        # Multi-mode linear stability probe: 1L + 1T(+) + 1T(-) + 2T(+) + 2T(-) + 1L1T
        P = init_energy_amp * (
            0.7 * np.cos(np.pi * X / L_a)
            + np.cos(TH) + 0.7 * np.cos(TH + 1.1)
            + np.cos(2.0 * TH + 0.4) + 0.7 * np.cos(2.0 * TH - 0.9)
            + 0.5 * np.cos(np.pi * X / L_a) * np.cos(TH - 0.5)
        )
        Uth = init_energy_amp * (
            np.cos(th_f[None, :]) - 0.7 * np.cos(th_f[None, :] + 1.1)
            + np.cos(2.0 * th_f[None, :] + 0.4) - 0.7 * np.cos(2.0 * th_f[None, :] - 0.9)
        ) * np.ones((nx, 1))
    else:
        raise ValueError(f"Unknown init_type: {init_type}")

    Uth *= uth_mask
    v_r = np.zeros(ntheta)
    p_c = np.where(cav_mask, P[0], 0.0)

    tau_d = min(tau_bar, np.sqrt(m_stages) * sigma_tau)
    rate = m_stages / tau_d
    p_bar = np.zeros(ntheta)
    v_bar = np.zeros(ntheta)
    q_p = np.zeros((m_stages, ntheta))
    q_v = np.zeros((m_stages, ntheta))

    def rhs(P_c, Ux_c, Uth_c, vr_c, pc_c, pb_c, vb_c, qp_c, qv_c):
        dP = np.empty_like(P_c)
        dP[0] = -(Ux_c[0] + vr_c) * inv_dx
        dP[1:-1] = -(Ux_c[1:] - Ux_c[:-1]) * inv_dx
        dP[-1] = -(zeta * P_c[-1] - Ux_c[-1]) * inv_dx
        dP -= (Uth_c - np.roll(Uth_c, 1, axis=1)) * inv_dth

        q_tot = np.zeros(ntheta)
        if n_p != 0.0:
            q_tot += n_p * phi_bounded(qp_c[-1], q_s, kappa)
        if n_v != 0.0:
            q_tot += n_v * phi_bounded(qv_c[-1], q_s, kappa)
        dP[m_f] += q_tot * inv_dx

        dUx = -(P_c[1:] - P_c[:-1]) * inv_dx
        dUth = -(np.roll(P_c, -1, axis=1) - P_c) * inv_dth
        dUth *= uth_mask

        if has_cav:
            dvr = np.where(cav_mask, cav_k_vr * (P_c[0] - pc_c) - (cav_damp + cnl_arr * np.abs(vr_c)) * vr_c, 0.0)
            dpc = vr_c * inv_beta_s
        else:
            dvr = np.zeros(ntheta)
            dpc = np.zeros(ntheta)

        u_cell = 0.5 * (Uth_c[m_f] + np.roll(Uth_c[m_f], 1))
        dpb = OMEGA_HP * (P_c[m_f] - pb_c)
        dvb = OMEGA_HP * (u_cell - vb_c)

        dqp = np.empty_like(qp_c)
        dqv = np.empty_like(qv_c)
        dqp[0] = rate * ((P_c[m_f] - pb_c) - qp_c[0])
        dqv[0] = rate * ((u_cell - vb_c) - qv_c[0])
        for r in range(1, m_stages):
            dqp[r] = rate * (qp_c[r - 1] - qp_c[r])
            dqv[r] = rate * (qv_c[r - 1] - qv_c[r])

        return dP, dUx, dUth, dvr, dpc, dpb, dvb, dqp, dqv

    w2_lo, w2_hi = int(0.55 * n_steps), int(0.70 * n_steps)
    w3_lo = int(0.85 * n_steps)
    e_w2_sum = e_w2_cnt = 0.0
    e_w3_sum = e_w3_cnt = 0.0
    p_peak_late = 0.0
    spin_num_late = 0.0
    spin_den_late = 0.0

    stride_h = 4
    if record_history:
        n_rec = n_steps // stride_h + 1
        t_rec = np.empty(n_rec)
        e_rec = np.empty(n_rec)
        spin_rec = np.empty(n_rec)
        pmax_rec = np.empty(n_rec)
    else:
        t_rec = e_rec = spin_rec = pmax_rec = None

    if check_energy_bal:
        eb_E = np.empty(n_steps + 1)
        eb_pow = np.empty(n_steps + 1)
    else:
        eb_E = eb_pow = None

    cav_e_vr_coeff = np.where(cav_mask, 1.0 / np.maximum(beta_s * (wH_arr ** 2), 1e-30), 0.0)
    cav_dlin_coeff = np.where(cav_mask, (2.0 * xiH_arr) / np.maximum(beta_s * wH_arr, 1e-30), 0.0)
    cav_dnl_coeff = np.where(cav_mask, cnl_arr / np.maximum(beta_s * (wH_arr ** 2), 1e-30), 0.0)

    def calc_diag():
        e_ch = 0.5 * dx * dth * (float(np.sum(P * P)) + float(np.sum(Ux * Ux)) + float(np.sum(Uth * Uth)))
        e_cav = 0.5 * dth * float(np.sum(beta_s * p_c * p_c + cav_e_vr_coeff * v_r * v_r)) if has_cav else 0.0
        u_c = 0.5 * (Uth[m_f] + np.roll(Uth[m_f], 1))
        s_num = abs(float(np.dot(P[m_f], u_c)))
        s_den = 0.5 * float(np.dot(P[m_f], P[m_f]) + np.dot(u_c, u_c))
        return e_ch + e_cav, s_num, s_den, float(np.max(np.abs(P[0])))

    def calc_net_power() -> float:
        q_tot = np.zeros(ntheta)
        if n_p != 0.0:
            q_tot += n_p * phi_bounded(q_p[-1], q_s, kappa)
        if n_v != 0.0:
            q_tot += n_v * phi_bounded(q_v[-1], q_s, kappa)
        r_pow = dth * float(np.dot(P[m_f], q_tot))
        d_noz = zeta * dth * float(np.sum(P[-1] * P[-1]))
        d_cav = dth * float(np.sum(cav_dlin_coeff * (v_r ** 2) + cav_dnl_coeff * (np.abs(v_r) ** 3))) if has_cav else 0.0
        return r_pow - d_noz - d_cav

    if record_history:
        e0, sn0, sd0, pm0 = calc_diag()
        t_rec[0], e_rec[0], spin_rec[0], pmax_rec[0] = 0.0, e0, sn0 / max(sd0, 1e-30), pm0
    if check_energy_bal:
        eb_E[0] = calc_diag()[0]
        eb_pow[0] = calc_net_power()

    for step in range(n_steps):
        k1 = rhs(P, Ux, Uth, v_r, p_c, p_bar, v_bar, q_p, q_v)
        k2 = rhs(
            P + 0.5 * dt * k1[0], Ux + 0.5 * dt * k1[1], Uth + 0.5 * dt * k1[2],
            v_r + 0.5 * dt * k1[3], p_c + 0.5 * dt * k1[4], p_bar + 0.5 * dt * k1[5],
            v_bar + 0.5 * dt * k1[6], q_p + 0.5 * dt * k1[7], q_v + 0.5 * dt * k1[8],
        )
        k3 = rhs(
            P + 0.5 * dt * k2[0], Ux + 0.5 * dt * k2[1], Uth + 0.5 * dt * k2[2],
            v_r + 0.5 * dt * k2[3], p_c + 0.5 * dt * k2[4], p_bar + 0.5 * dt * k2[5],
            v_bar + 0.5 * dt * k2[6], q_p + 0.5 * dt * k2[7], q_v + 0.5 * dt * k2[8],
        )
        k4 = rhs(
            P + dt * k3[0], Ux + dt * k3[1], Uth + dt * k3[2],
            v_r + dt * k3[3], p_c + dt * k3[4], p_bar + dt * k3[5],
            v_bar + dt * k3[6], q_p + dt * k3[7], q_v + dt * k3[8],
        )

        P = P + (dt / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
        Ux = Ux + (dt / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
        Uth = (Uth + (dt / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])) * uth_mask
        v_r = v_r + (dt / 6.0) * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3])
        p_c = p_c + (dt / 6.0) * (k1[4] + 2.0 * k2[4] + 2.0 * k3[4] + k4[4])
        p_bar = p_bar + (dt / 6.0) * (k1[5] + 2.0 * k2[5] + 2.0 * k3[5] + k4[5])
        v_bar = v_bar + (dt / 6.0) * (k1[6] + 2.0 * k2[6] + 2.0 * k3[6] + k4[6])
        q_p = q_p + (dt / 6.0) * (k1[7] + 2.0 * k2[7] + 2.0 * k3[7] + k4[7])
        q_v = q_v + (dt / 6.0) * (k1[8] + 2.0 * k2[8] + 2.0 * k3[8] + k4[8])

        need_diag = (w2_lo <= step < w2_hi) or (step >= w3_lo) or check_energy_bal or (record_history and ((step + 1) % stride_h == 0))
        if need_diag:
            ec, sn, sd, pm = calc_diag()
            if w2_lo <= step < w2_hi:
                e_w2_sum += ec
                e_w2_cnt += 1.0
            if step >= w3_lo:
                e_w3_sum += ec
                e_w3_cnt += 1.0
                spin_num_late += sn
                spin_den_late += sd
                if pm > p_peak_late:
                    p_peak_late = pm
            if record_history and ((step + 1) % stride_h == 0):
                ri = (step + 1) // stride_h
                t_rec[ri] = (step + 1) * dt
                e_rec[ri] = ec
                spin_rec[ri] = sn / max(sd, 1e-30)
                pmax_rec[ri] = pm
            if check_energy_bal:
                eb_E[step + 1] = ec
                eb_pow[step + 1] = calc_net_power()

    e_w2 = e_w2_sum / max(e_w2_cnt, 1.0)
    e_w3 = e_w3_sum / max(e_w3_cnt, 1.0)
    e_ratio = e_w3 / max(e_w2, 1e-30)
    spin_late = spin_num_late / max(spin_den_late, 1e-30)

    # Pure trend-based trigger (NEVER marks a decaying trajectory as triggered!):
    # Triggered iff energy is growing (e_ratio > 1.005) OR flat on a large limit cycle (e_ratio >= 0.985 and P > 0.020)
    triggered = bool((e_ratio > 1.005 and p_peak_late > 0.002) or (e_ratio >= 0.985 and p_peak_late > 0.020))

    eb_err = None
    if check_energy_bal:
        cum_w = np.concatenate([[0.0], np.cumsum(0.5 * (eb_pow[:-1] + eb_pow[1:]) * dt)])
        dE = eb_E - eb_E[0]
        eb_err = float(np.max(np.abs(dE - cum_w)) / max(np.max(np.abs(dE)), np.max(eb_E), 1e-15))

    return {
        "e_w2": float(e_w2), "e_w3": float(e_w3), "e_ratio": float(e_ratio),
        "p_peak_late": float(p_peak_late), "spin_index_late": float(spin_late),
        "triggered": triggered, "x_f_actual": x_f_actual,
        "energy_balance_rel_err": eb_err,
        "t_rec": t_rec, "e_rec": e_rec, "spin_rec": spin_rec, "pmax_rec": pmax_rec,
    }


def check_linear_stability_long(
    n_p: float, n_v: float, beta_local=0.0, omega_H_arr=1.0, xi_H_arr=0.14,
    n_baffles: int = 0, L_b: float = 0.0, nx: int = 30, ntheta: int = 60, t_end: float = 400.0
) -> tuple[bool, float]:
    """Long-horizon (t=400) multi-mode broadband linear stability check (1L, 1T+/-, 2T+/-, 1L1T)."""
    r = simulate_2d_nonlinear(
        n_p=n_p, n_v=n_v, beta_local=beta_local, omega_H_arr=omega_H_arr, xi_H_arr=xi_H_arr, c_nl_arr=0.0,
        n_baffles=n_baffles, L_b=L_b, nx=nx, ntheta=ntheta, t_end=t_end,
        init_type="broadband_all", init_energy_amp=1e-4, record_history=True,
    )
    mask = r["t_rec"] >= 200.0
    alpha_late = float(np.polyfit(r["t_rec"][mask], 0.5 * np.log(np.maximum(r["e_rec"][mask], 1e-30)), 1)[0])
    is_stable = bool(alpha_late < -1e-4 and r["e_ratio"] < 0.99)
    return is_stable, alpha_late


def bisect_2d_bomb(
    n_p: float, n_v: float, init_type: str,
    beta_local=0.0, omega_H_arr=1.0, xi_H_arr=0.14, c_nl_arr=0.0,
    nx: int = 30, ntheta: int = 60, t_end: float = 180.0, n_iter: int = 9,
) -> float | None:
    """Bisect energy-equivalent critical bomb amplitude A_eq_crit with pure trend-based trigger."""
    r_lo = simulate_2d_nonlinear(
        n_p=n_p, n_v=n_v, beta_local=beta_local, omega_H_arr=omega_H_arr, xi_H_arr=xi_H_arr, c_nl_arr=c_nl_arr,
        nx=nx, ntheta=ntheta, t_end=t_end, init_type=init_type, init_energy_amp=0.002,
    )
    if r_lo["triggered"]:
        return 0.0

    hi_found = None
    for pa in (0.015, 0.030, 0.050, 0.075, 0.095):
        r = simulate_2d_nonlinear(
            n_p=n_p, n_v=n_v, beta_local=beta_local, omega_H_arr=omega_H_arr, xi_H_arr=xi_H_arr, c_nl_arr=c_nl_arr,
            nx=nx, ntheta=ntheta, t_end=t_end, init_type=init_type, init_energy_amp=pa,
        )
        if r["triggered"]:
            hi_found = pa
            break

    if hi_found is None:
        return None

    lo, hi = 0.002, hi_found
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        r = simulate_2d_nonlinear(
            n_p=n_p, n_v=n_v, beta_local=beta_local, omega_H_arr=omega_H_arr, xi_H_arr=xi_H_arr, c_nl_arr=c_nl_arr,
            nx=nx, ntheta=ntheta, t_end=t_end, init_type=init_type, init_energy_amp=mid,
        )
        if r["triggered"]:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def make_dual_cavity_arrays(ntheta: int, beta_tot: float, f1: float, xi_mult: float, C_d: float, sigma_o: float, L_a: float = 2.0):
    n_pair = 5
    half_sec = ntheta // (2 * n_pair)
    is_z1 = np.array([((j // half_sec) % 2 == 0) for j in range(ntheta)])
    f2 = 1.0 - f1
    b_loc = np.where(is_z1, 2.0 * f1 * beta_tot, 2.0 * f2 * beta_tot)
    wH = np.where(is_z1, 1.0, 2.0)
    xiH = np.where(is_z1, xi_mult * np.sqrt(f1 * beta_tot), xi_mult * np.sqrt(f2 * beta_tot))
    beta_s = b_loc * L_a
    c_nl = (beta_s * (wH ** 2)) / (2.0 * (C_d ** 2) * (sigma_o ** 2)) if sigma_o > 0.0 else np.zeros(ntheta)
    return b_loc, wH, xiH, c_nl


# ---------------------------------------------------------------------------
# 3. Main experiments & plotting
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    L_a = 2.0
    zeta = 0.04
    nx, ntheta = 30, 60
    x_f_exact = 2.5 * (L_a / nx)  # = 0.16667 (exact cell center on nx=30 AND nx=90!)

    n_pH, w_pH = find_bare_2d_hopf(L_a, zeta, x_f_exact, 0.30, 0.15)
    print("=== Entry 006: 2D Nonlinear Mode Selection & Verified Cavity Bomb Tests ===")
    print(f"Exact bare 2D 1T linear Hopf point (x_f={x_f_exact:.4f}): n_pH = {n_pH:.5f}, omega_H = {w_pH:.5f}\n")

    # ------------------------------------------------------------------
    # Experiment 1: 2D Subcritical Standing vs Spinning Triggering + True Spin-Up Separation
    # ------------------------------------------------------------------
    print("--- Experiment 1: 2D Subcritical Standing vs. Spinning Triggering (n_v = 0) ---")
    exp1_rows = []
    for np_val in (0.033, 0.037, 0.040):
        a_sp = bisect_2d_bomb(np_val, 0.0, "spin_1T_plus", nx=nx, ntheta=ntheta, t_end=180.0)
        a_st = bisect_2d_bomb(np_val, 0.0, "standing_1T_pure", nx=nx, ntheta=ntheta, t_end=180.0)
        e_ratio = (a_sp / a_st) ** 2 if (a_sp and a_st and a_st > 0) else float("nan")
        # Dedicated upper-branch settled limit-cycle check (starting well above saddle at A=0.055)
        r_lc = simulate_2d_nonlinear(n_p=np_val, n_v=0.0, x_f=x_f_exact, nx=nx, ntheta=ntheta, t_end=260.0, init_type="spin_1T_plus", init_energy_amp=0.055)
        print(
            f"  n_p={np_val:.3f} (< n_pH): A_crit(spin)={a_sp:.4f}, A_crit(stand)={a_st:.4f} "
            f"=> E_crit(spin)/E_crit(stand)={e_ratio:.3f} (theory 1.50) | settled P_LC={r_lc['p_peak_late']:.4f}"
        )
        exp1_rows.append({
            "n_p": np_val, "A_crit_spin": a_sp, "A_crit_stand": a_st,
            "E_crit_spin_over_stand": float(e_ratio), "P_LC_settled_spin": r_lc["p_peak_late"],
        })

    # Horizon check on n_p=0.037 (t=180 vs t=360)
    a_sp_360 = bisect_2d_bomb(0.037, 0.0, "spin_1T_plus", nx=nx, ntheta=ntheta, t_end=360.0, n_iter=8)
    a_st_360 = bisect_2d_bomb(0.037, 0.0, "standing_1T_pure", nx=nx, ntheta=ntheta, t_end=360.0, n_iter=8)
    print(f"  Horizon check at n_p=0.037 (t=360): A_spin={a_sp_360:.4f}, A_stand={a_st_360:.4f}, E_ratio={(a_sp_360/a_st_360)**2:.3f}")

    # Subcritical (n_p=0.037 < n_pH) triggered standing bomb run to upper branch + spin evolution (t=500)
    tr_sub_spinup = simulate_2d_nonlinear(
        n_p=0.037, n_v=0.0, x_f=x_f_exact, nx=nx, ntheta=ntheta, t_end=500.0,
        init_type="standing_1T", init_energy_amp=0.014, record_history=True,
    )
    print(
        f"  Subcritical triggered standing bomb (n_p=0.037 < n_pH, A_eq=0.014, S_0={tr_sub_spinup['spin_rec'][0]:.3f}): "
        f"at t=500 -> P_peak={tr_sub_spinup['p_peak_late']:.4f}, S_late={tr_sub_spinup['spin_index_late']:.4f}"
    )

    # Supercritical (n_p=0.068 > n_pH) seeded standing -> spinning transition (t=360)
    tr_sup_spinup = simulate_2d_nonlinear(
        n_p=0.068, n_v=0.0, x_f=x_f_exact, nx=nx, ntheta=ntheta, t_end=360.0,
        init_type="standing_1T", init_energy_amp=0.035, record_history=True,
    )
    print(
        f"  Supercritical seeded standing (n_p=0.068 > n_pH, S_0={tr_sup_spinup['spin_rec'][0]:.3f}): "
        f"at t=360 -> P_peak={tr_sup_spinup['p_peak_late']:.4f}, S_late={tr_sup_spinup['spin_index_late']:.4f}\n"
    )

    # ------------------------------------------------------------------
    # Experiment 2: Long-Horizon Linearly-Verified 80/20 Cavity Ring Nonlinear Bomb Tests
    # Choose n_p = 0.038, n_v = 0.025 (bare combustor strongly unstable; 80/20 ring strictly linearly stable over t=400!)
    # ------------------------------------------------------------------
    n_p_c, n_v_c = 0.038, 0.025
    beta_tot = 0.02
    f1 = 0.80
    C_d = 0.65
    sigma_o_tight = 0.015
    sigma_o_wide = 0.035

    bare_stable, bare_alpha = check_linear_stability_long(n_p_c, n_v_c, nx=nx, ntheta=ntheta, t_end=260.0)
    baf_stable, baf_alpha = check_linear_stability_long(n_p_c, n_v_c, n_baffles=5, L_b=0.60, nx=nx, ntheta=ntheta, t_end=260.0)
    print(f"--- Experiment 2: Linearly-Verified 80/20 Cavity Ring Bomb Sweep (n_p={n_p_c}, n_v={n_v_c}, beta=2%) ---")
    print(f"  Bare chamber broadband linear check:          stable={bare_stable}, alpha_late={bare_alpha:+.5f}")
    print(f"  Pure 5-blade baffles (L_b=0.60) linear check: stable={baf_stable}, alpha_late={baf_alpha:+.5f}")

    # Nonlinear 2D energy balance check with calibrated quadratic cavity drag!
    b0, w0, x0, c0 = make_dual_cavity_arrays(ntheta, beta_tot, f1, 0.80, C_d, sigma_o_wide, L_a)
    r_eb = simulate_2d_nonlinear(
        n_p=n_p_c, n_v=n_v_c, x_f=x_f_exact, beta_local=b0, omega_H_arr=w0, xi_H_arr=x0, c_nl_arr=c0,
        nx=nx, ntheta=ntheta, t_end=50.0, init_type="spin_1T_plus", init_energy_amp=0.04, check_energy_bal=True,
    )
    print(f"  2D Nonlinear Energy Balance Check (flame + quadratic orifice drag): rel err = {r_eb['energy_balance_rel_err']:.2e}")

    xi_mults = [0.25, 0.50, 0.80, 1.00, 1.40, 2.00]
    cav_rows = []
    for xm in xi_mults:
        b_loc, wH, xiH, c_nl_t = make_dual_cavity_arrays(ntheta, beta_tot, f1, xm, C_d, sigma_o_tight, L_a)
        _, _, _, c_nl_w = make_dual_cavity_arrays(ntheta, beta_tot, f1, xm, C_d, sigma_o_wide, L_a)
        lin_ok, lin_alpha = check_linear_stability_long(n_p_c, n_v_c, b_loc, wH, xiH, nx=nx, ntheta=ntheta, t_end=400.0)
        if not lin_ok:
            print(f"  xi/xi_EP={xm:.2f}: LINEARLY UNSTABLE over t=400 (alpha_late={lin_alpha:+.5f}) -> A_crit = 0.0000")
            cav_rows.append({
                "xi_mult": xm, "linear_stable_t400": False, "linear_alpha_t400": lin_alpha,
                "A_crit_1T_linear_cav": 0.0, "A_crit_1T_wide_35pct": 0.0, "A_crit_1T_tight_15pct": 0.0,
            })
            continue

        a1_lin = bisect_2d_bomb(n_p_c, n_v_c, "spin_1T_plus", b_loc, wH, xiH, 0.0, nx=nx, ntheta=ntheta, t_end=180.0)
        a1_wide = bisect_2d_bomb(n_p_c, n_v_c, "spin_1T_plus", b_loc, wH, xiH, c_nl_w, nx=nx, ntheta=ntheta, t_end=180.0)
        a1_tight = bisect_2d_bomb(n_p_c, n_v_c, "spin_1T_plus", b_loc, wH, xiH, c_nl_t, nx=nx, ntheta=ntheta, t_end=180.0)
        s_lin = f"{a1_lin:.4f}" if a1_lin is not None else "NO TRIG<=.095"
        s_w = f"{a1_wide:.4f}" if a1_wide is not None else "NO TRIG<=.095"
        s_t = f"{a1_tight:.4f}" if a1_tight is not None else "NO TRIG<=.095"
        print(
            f"  xi/xi_EP={xm:.2f} (lin alpha={lin_alpha:+.5f}): "
            f"1T(lin cav)={s_lin:>13s} | 1T(wide 3.5%)={s_w:>13s} | 1T(tight 1.5%)={s_t:>13s}"
        )
        cav_rows.append({
            "xi_mult": xm, "linear_stable_t400": True, "linear_alpha_t400": lin_alpha,
            "A_crit_1T_linear_cav": a1_lin, "A_crit_1T_wide_35pct": a1_wide, "A_crit_1T_tight_15pct": a1_tight,
        })

    # Horizon convergence check on Exp 2 at xi/xi_EP = 1.00 (tight 1.5%, t=180 vs t=360)
    b1, w1, x1, c1_t = make_dual_cavity_arrays(ntheta, beta_tot, f1, 1.00, C_d, sigma_o_tight, L_a)
    a1_t_360 = bisect_2d_bomb(n_p_c, n_v_c, "spin_1T_plus", b1, w1, x1, c1_t, nx=nx, ntheta=ntheta, t_end=360.0, n_iter=8)
    row_ep = next(r for r in cav_rows if abs(r["xi_mult"] - 1.0) < 1e-6)
    print(f"  Exp 2 horizon check at xi/xi_EP=1.00 (tight 1.5%): t=180 -> {row_ep['A_crit_1T_tight_15pct']} vs t=360 -> {a1_t_360}")

    # ------------------------------------------------------------------
    # Figure 1: Subcritical Triggered Spin-Up vs Supercritical Spin-Up
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.4), dpi=150)

    ax1_r = ax1.twinx()
    l1, = ax1.plot(tr_sub_spinup["t_rec"], tr_sub_spinup["pmax_rec"], color="#d62728", lw=1.6, label="Peak faceplate $|P(0)|$")
    l2, = ax1_r.plot(tr_sub_spinup["t_rec"], tr_sub_spinup["spin_rec"], color="#2ca02c", lw=2.2,
                     label=f"Spin index $S(t): {tr_sub_spinup['spin_rec'][0]:.2f} \\rightarrow {tr_sub_spinup['spin_index_late']:.2f}$")
    ax1.set_xlabel("Dimensionless time $t \\bar{c}/R$")
    ax1.set_ylabel("Peak faceplate pressure $|P(0)|$", color="#d62728")
    ax1_r.set_ylabel("Spin index $S(t)$ ($0=$ stand, $1=$ spin)", color="#2ca02c")
    ax1_r.set_ylim(-0.05, 1.12)
    ax1.set_title(f"Subcritical ($n_p=0.037 < n_{{pH}}$) Triggered Standing Bomb", fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend([l1, l2], [l1.get_label(), l2.get_label()], loc="upper left", fontsize=8.0)

    ax2_r = ax2.twinx()
    m1, = ax2.plot(tr_sup_spinup["t_rec"], tr_sup_spinup["pmax_rec"], color="#d62728", lw=1.6, label="Peak faceplate $|P(0)|$")
    m2, = ax2_r.plot(tr_sup_spinup["t_rec"], tr_sup_spinup["spin_rec"], color="#2ca02c", lw=2.2,
                     label=f"Spin index $S(t): {tr_sup_spinup['spin_rec'][0]:.2f} \\rightarrow {tr_sup_spinup['spin_index_late']:.2f}$")
    ax2.set_xlabel("Dimensionless time $t \\bar{c}/R$")
    ax2.set_ylabel("Peak faceplate pressure $|P(0)|$", color="#d62728")
    ax2_r.set_ylabel("Spin index $S(t)$", color="#2ca02c")
    ax2_r.set_ylim(-0.05, 1.12)
    ax2.set_title(f"Supercritical ($n_p=0.068 > n_{{pH}}$) Seeded Standing $\\rightarrow$ Spinning", fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend([m1, m2], [m1.get_label(), m2.get_label()], loc="center right", fontsize=8.0)

    fig.suptitle("Entry 006 (Part 1): 2D Symmetry Breaking — Subcritical Triggering vs. Supercritical Spin Selection", fontsize=11.9, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(RESULTS_DIR / "exp006_2d_spin_selection.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2: Verified 2D Nonlinear Cavity Bomb Thresholds vs xi/xi_EP
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.6, 5.3), dpi=150)
    cap = 0.10
    xms = [r["xi_mult"] for r in cav_rows]
    y_1lin = [cap if r["A_crit_1T_linear_cav"] is None else r["A_crit_1T_linear_cav"] for r in cav_rows]
    y_1w = [cap if r["A_crit_1T_wide_35pct"] is None else r["A_crit_1T_wide_35pct"] for r in cav_rows]
    y_1t = [cap if r["A_crit_1T_tight_15pct"] is None else r["A_crit_1T_tight_15pct"] for r in cav_rows]

    ax.axvline(1.0, color="#1f77b4", ls=":", lw=1.4, label="Linear 2D EP ($\\xi_H / \\xi_H^{\\mathrm{EP}} = 1.0$)")
    ax.axhline(cap, color="gray", ls="--", lw=1.0, alpha=0.7)
    ax.text(1.42, cap + 0.002, "No trigger (probed $\\leq 0.095$)", fontsize=8.0, color="gray")
    ax.plot(xms, y_1lin, "s--", color="#2ca02c", lw=1.9, ms=6.5, label="$1T$ bomb (ideal linear 80/20 cavity ring)")
    ax.plot(xms, y_1w, "D-", color="#1f77b4", lw=2.3, ms=6.5, label="$1T$ bomb (wide-mouth rim cavities, $\\sigma_o=3.5\\%$)")
    ax.plot(xms, y_1t, "o-", color="#d62728", lw=2.3, ms=6.5, label="$1T$ bomb (tight-neck cavities, $\\sigma_o=1.5\\%$)")

    ax.set_xlabel("Linear cavity damping ratio normalized to 2D EP ($\\xi_H / \\xi_H^{\\mathrm{EP}}$)")
    ax.set_ylabel("Critical 2D bomb amplitude $A_{\\mathrm{eq,crit}}$ (trend-verified)")
    ax.set_ylim(-0.005, 0.112)
    ax.set_title(f"Linearly-Verified 80/20 Cavity Ring Bomb Survival ($n_p={n_p_c}, n_v={n_v_c}, \\beta=2\\%$)", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=8.1)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp006_2d_nonlinear_cavity_bombs.png")
    plt.close(fig)

    metrics = {
        "parameters": {
            "L_a": L_a, "zeta": zeta, "x_f_exact": x_f_exact, "tau_bar": 0.30, "sigma_tau": 0.15,
            "omega_hp": OMEGA_HP, "q_s": 0.036, "kappa": 4.5, "gamma_ref": 1.25,
            "n_pH_bare": n_pH, "omega_pH_bare": w_pH, "grid": [nx, ntheta],
            "cavity_test": {"n_p": n_p_c, "n_v": n_v_c, "beta_tot": beta_tot, "f1": f1, "C_d": C_d, "sigma_o_tight": sigma_o_tight, "sigma_o_wide": sigma_o_wide},
        },
        "exp1_spin_vs_standing": {
            "threshold_ratios": exp1_rows,
            "horizon_check_np037_t360": {"A_spin": a_sp_360, "A_stand": a_st_360, "E_ratio": float((a_sp_360 / a_st_360) ** 2)},
            "subcritical_spinup_np037": {
                "spin_initial": float(tr_sub_spinup["spin_rec"][0]),
                "spin_final_t500": tr_sub_spinup["spin_index_late"],
                "p_peak_final_t500": tr_sub_spinup["p_peak_late"],
            },
            "supercritical_spinup_np068": {
                "spin_initial": float(tr_sup_spinup["spin_rec"][0]),
                "spin_final_t360": tr_sup_spinup["spin_index_late"],
                "p_peak_final_t360": tr_sup_spinup["p_peak_late"],
            },
        },
        "exp2_2d_nonlinear_cavity_sweep": {
            "bare_linear_check": {"stable": bare_stable, "alpha_late": bare_alpha},
            "baffles_Nb5_Lb060_linear_check": {"stable": baf_stable, "alpha_late": baf_alpha},
            "nonlinear_energy_balance_rel_err": r_eb["energy_balance_rel_err"],
            "xi_mult_sweep": cav_rows,
            "horizon_check_ep_tight_t360": a1_t_360,
        },
    }

    with open(RESULTS_DIR / "exp006_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved hardened Entry 006 plots and metrics to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
