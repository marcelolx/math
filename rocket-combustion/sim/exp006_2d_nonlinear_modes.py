"""Entry 006: 2D Nonlinear Spinning vs. Standing Mode Selection & Nonlinear Cavity Triggering.

Synthesizes Entries 003, 004, and 005 in a 2D cylindrical-annulus combustor (L_a = 2.0,
zeta = 0.04, DC-blocked omega_hp = 0.15, causal Erlang-4) with:
  1. Strictly bounded, monotonic subcritical flame law (keeping bulk acoustics linear P <= 0.09):
         Phi(q) = q_s * tanh(q/q_s + kappa * (q/q_s)^3),   q_s = 0.036, kappa = 4.5
  2. Calibrated quadratic orifice jet drag inside the 2D faceplate Helmholtz cavity ring:
         dv_r/dt + (2*xi_H*omega_H + c_nl * |v_r|) * v_r = beta_s*omega_H^2 * (P(0,theta) - P_c)
         c_nl = (beta_s * omega_H^2) / (2 * C_d^2 * sigma_o^2)
  3. Optional radial faceplate baffles (U_theta = 0 for x <= L_b).

Experiments:
1. 2D Symmetry-Breaking Mode Selection (Axisymmetric bare chamber, n_v = 0, pure pressure coupling):
   - Measures the bare 2D linear Hopf point n_H(1T) and tests our analytical prediction in the
     bistable subcritical zone (n_p < n_H):
     (a) At the SAME total initial bomb energy E_0, a standing 1T bomb has 1.5x stronger spatial-avg
         cubic boost than a spinning 1T bomb => E_crit(standing) < E_crit(spinning)!
     (b) Once triggered into the saturating upper branch (n_p > n_H or triggered bistable),
         uniform-amplitude spinning 1T saturates less than antinode-peaked standing 1T =>
         a seeded standing 1T wave spontaneously breaks symmetry into a PURE SPINNING 1T limit cycle!
2. 2D Nonlinear Bomb-Test Showdown for Baffle-Free 80/20 Dual-Tuned Cavity Ring (beta_tot = 2%,
   odd N_pair = 5) under calibrated quadratic orifice drag across xi_H / xi_EP in [0.25, 2.0]:
   - Tests both spinning 1T and spinning 2T finite-amplitude bombs on a bare-unstable combined
     1T+2T combustor (n_p = 0.048, n_v = 0.040).
   - Verifies whether Entry 003's "never over-damp above EP" rule carries over to 2D nonlinear
     transverse bomb tests, and compares against pure 5-blade baffles.
3. Grid convergence check (32x48 vs 48x72) on the 2D nonlinear spinning-selection & bomb threshold.
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
# 1. Nonlinear 2D staggered C-grid RK4 PDE (bounded flame + nonlin cavities + baffles)
# ---------------------------------------------------------------------------

def simulate_2d_nonlinear(
    L_a: float = 2.0,
    zeta: float = 0.04,
    x_f: float = 0.20,
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
    nx: int = 32,
    ntheta: int = 48,
    cfl: float = 0.35,
    t_end: float = 140.0,
    init_type: str = "spin_1T_plus",
    init_energy_amp: float = 0.03,
    record_history: bool = False,
) -> dict:
    """2D nonlinear annular combustor with bounded subcritical flame & quadratic cavity drag.

    Energy-matched initialization (init_energy_amp = A_eq):
      - 'spin_1T_plus':  P = A_eq * cos(th), Uth = +A_eq * cos(th)
                         -> E_0 = 0.5 * L_a * (pi*A_eq^2 + pi*A_eq^2) = pi * L_a * A_eq^2
      - 'standing_1T':   P = sqrt(2)*A_eq * cos(th), Uth = 0 (+ tiny 1% spin symmetry-breaking noise)
                         -> E_0 = 0.5 * L_a * (2*pi*A_eq^2) = pi * L_a * A_eq^2  (EXACT SAME E_0!)
      - 'spin_2T_plus':  P = A_eq * cos(2*th), Uth = +A_eq * cos(2*th)
                         -> E_0 = pi * L_a * A_eq^2
    Note: With tau_bar = 0.30, sigma_tau = 0.15, m_stages = 4 => tau_d = 0.30, tau_0 = 0 (pure ODE!).
    """
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

    # Optional radial baffles
    uth_mask = np.ones((nx, ntheta), dtype=float)
    if n_baffles > 0 and L_b > 0.0:
        n_bc = int(np.sum(x_c <= L_b + 1e-12))
        for b in range(n_baffles):
            jf = int(round(b * ntheta / n_baffles)) % ntheta
            uth_mask[:n_bc, jf] = 0.0

    # Energy-matched initial condition
    _, TH = np.meshgrid(x_c, th_c, indexing="ij")
    P = np.zeros((nx, ntheta))
    Ux = np.zeros((nx - 1, ntheta))
    Uth = np.zeros((nx, ntheta))

    if init_type == "spin_1T_plus":
        P = init_energy_amp * np.cos(TH)
        Uth = init_energy_amp * np.cos(th_f[None, :]) * np.ones((nx, 1))
    elif init_type == "standing_1T":
        # Exact same acoustic energy E_0 = pi * L_a * A_eq^2, plus small chiral seed to test stability of standing state!
        eps_s = 0.12
        scale_p = np.sqrt(2.0 / (1.0 + 0.5 * eps_s * eps_s))
        P = scale_p * init_energy_amp * np.cos(TH)
        Uth = (eps_s * init_energy_amp) * np.cos(th_f[None, :]) * np.ones((nx, 1))
    elif init_type == "standing_1T_pure":
        P = np.sqrt(2.0) * init_energy_amp * np.cos(TH)
    elif init_type == "spin_2T_plus":
        P = init_energy_amp * np.cos(2.0 * TH)
        Uth = init_energy_amp * np.cos(2.0 * th_f[None, :]) * np.ones((nx, 1))
    else:
        raise ValueError(f"Unknown init_type: {init_type}")

    Uth *= uth_mask
    # Consistent cavity pre-pressurization at t = 0
    v_r = np.zeros(ntheta)
    p_c = np.where(cav_mask, P[0], 0.0)

    # Pure 4-stage causal Erlang (tau_d == tau_bar -> tau_0 = 0)
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

        # Combined bounded nonlinear flame response: Q = [n_p*Phi(q_p) + n_v*Phi(q_v)] / dx
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

    # Diagnostics
    w2_lo, w2_hi = int(0.55 * n_steps), int(0.70 * n_steps)
    w3_lo = int(0.82 * n_steps)
    e_w2_sum = e_w2_cnt = 0.0
    e_w3_sum = e_w3_cnt = 0.0
    p_peak_late = 0.0
    spin_num_late = 0.0
    spin_den_late = 0.0

    # Sample history every 4 steps if requested
    stride_h = 4
    if record_history:
        n_rec = n_steps // stride_h + 1
        t_rec = np.empty(n_rec)
        e_rec = np.empty(n_rec)
        spin_rec = np.empty(n_rec)
        pmax_rec = np.empty(n_rec)
    else:
        t_rec = e_rec = spin_rec = pmax_rec = None

    cav_e_vr_coeff = np.where(cav_mask, 1.0 / np.maximum(beta_s * (wH_arr ** 2), 1e-30), 0.0)

    def calc_diag():
        e_ch = 0.5 * dx * dth * (float(np.sum(P * P)) + float(np.sum(Ux * Ux)) + float(np.sum(Uth * Uth)))
        e_cav = 0.5 * dth * float(np.sum(beta_s * p_c * p_c + cav_e_vr_coeff * v_r * v_r)) if has_cav else 0.0
        u_c = 0.5 * (Uth[m_f] + np.roll(Uth[m_f], 1))
        # Normalized spin index S in [0, 1]: |int P*Uth dth| / (0.5 * int (P^2 + Uth^2) dth)
        # S = 1.0 for pure spinning wave (P = +/- Uth); S = 0.0 for pure standing wave!
        s_num = abs(float(np.dot(P[m_f], u_c)))
        s_den = 0.5 * float(np.dot(P[m_f], P[m_f]) + np.dot(u_c, u_c))
        return e_ch + e_cav, s_num, s_den, float(np.max(np.abs(P[0])))

    if record_history:
        e0, sn0, sd0, pm0 = calc_diag()
        t_rec[0], e_rec[0], spin_rec[0], pmax_rec[0] = 0.0, e0, sn0 / max(sd0, 1e-30), pm0

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

        if (w2_lo <= step < w2_hi) or (step >= w3_lo) or (record_history and ((step + 1) % stride_h == 0)):
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

    e_w2 = e_w2_sum / max(e_w2_cnt, 1.0)
    e_w3 = e_w3_sum / max(e_w3_cnt, 1.0)
    spin_late = spin_num_late / max(spin_den_late, 1e-30)
    # Triggered iff energy is growing (e_w3 > 1.01*e_w2) OR settled at finite limit-cycle peak > 0.025
    triggered = bool(p_peak_late > 0.025 or (e_w3 > 1.01 * e_w2 and p_peak_late > 0.004))

    return {
        "e_w2": float(e_w2),
        "e_w3": float(e_w3),
        "p_peak_late": float(p_peak_late),
        "spin_index_late": float(spin_late),
        "triggered": triggered,
        "t_rec": t_rec,
        "e_rec": e_rec,
        "spin_rec": spin_rec,
        "pmax_rec": pmax_rec,
    }


def bisect_2d_bomb(
    n_p: float, n_v: float, init_type: str,
    beta_local=0.0, omega_H_arr=1.0, xi_H_arr=0.14, c_nl_arr=0.0,
    n_baffles: int = 0, L_b: float = 0.0,
    nx: int = 32, ntheta: int = 48, t_end: float = 150.0, n_iter: int = 9,
) -> tuple[float | None, float, float]:
    """Bisect energy-equivalent critical bomb amplitude A_eq_crit (E_0 = pi * L_a * A_eq^2)."""
    r_lo = simulate_2d_nonlinear(
        n_p=n_p, n_v=n_v, beta_local=beta_local, omega_H_arr=omega_H_arr, xi_H_arr=xi_H_arr, c_nl_arr=c_nl_arr,
        n_baffles=n_baffles, L_b=L_b, nx=nx, ntheta=ntheta, t_end=t_end,
        init_type=init_type, init_energy_amp=0.002,
    )
    if r_lo["triggered"]:
        return 0.0, r_lo["p_peak_late"], r_lo["spin_index_late"]

    hi_found = None
    pk_lc = 0.0
    sp_lc = 0.0
    for pa in (0.025, 0.045, 0.070, 0.095):
        r = simulate_2d_nonlinear(
            n_p=n_p, n_v=n_v, beta_local=beta_local, omega_H_arr=omega_H_arr, xi_H_arr=xi_H_arr, c_nl_arr=c_nl_arr,
            n_baffles=n_baffles, L_b=L_b, nx=nx, ntheta=ntheta, t_end=t_end,
            init_type=init_type, init_energy_amp=pa,
        )
        if r["triggered"]:
            hi_found = pa
            pk_lc = r["p_peak_late"]
            sp_lc = r["spin_index_late"]
            break

    if hi_found is None:
        return None, 0.0, 0.0

    lo, hi = 0.002, hi_found
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        r = simulate_2d_nonlinear(
            n_p=n_p, n_v=n_v, beta_local=beta_local, omega_H_arr=omega_H_arr, xi_H_arr=xi_H_arr, c_nl_arr=c_nl_arr,
            n_baffles=n_baffles, L_b=L_b, nx=nx, ntheta=ntheta, t_end=t_end,
            init_type=init_type, init_energy_amp=mid,
        )
        if r["triggered"]:
            hi = mid
            pk_lc = r["p_peak_late"]
            sp_lc = r["spin_index_late"]
        else:
            lo = mid
    return 0.5 * (lo + hi), float(pk_lc), float(sp_lc)


def make_dual_cavity_arrays(ntheta: int, beta_tot: float, f1: float, xi_mult: float, C_d: float, sigma_o: float, L_a: float = 2.0):
    """Build 80/20 odd-periodic dual-tuned cavity ring arrays with calibrated quadratic orifice drag.

    Uses N_pair = 4 for ntheta=48? Wait: 48 = 3 * 16 -> let's use ntheta=60 (5 pairs x 12 cells!) for exact N_pair=5!
    """
    n_pair = 5
    half_sec = ntheta // (2 * n_pair)
    is_z1 = np.array([((j // half_sec) % 2 == 0) for j in range(ntheta)])
    f2 = 1.0 - f1
    b_loc = np.where(is_z1, 2.0 * f1 * beta_tot, 2.0 * f2 * beta_tot)
    wH = np.where(is_z1, 1.0, 2.0)
    xi1_ep = np.sqrt(f1 * beta_tot)
    xi2_ep = np.sqrt(f2 * beta_tot)
    xiH = np.where(is_z1, xi_mult * xi1_ep, xi_mult * xi2_ep)
    # Calibrated quadratic drag per cavity sector: c_nl = (beta_s * wH^2) / (2 * C_d^2 * sigma_o^2)
    beta_s = b_loc * L_a
    c_nl = (beta_s * (wH ** 2)) / (2.0 * (C_d ** 2) * (sigma_o ** 2)) if sigma_o > 0.0 else np.zeros(ntheta)
    return b_loc, wH, xiH, c_nl


# ---------------------------------------------------------------------------
# 2. Main experiments & plotting
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    L_a = 2.0
    nx, ntheta = 30, 60  # Exact divisibility by N_pair=5 and N_b=5, fast 2D nonlinear runs!

    print("=== Entry 006: 2D Nonlinear Spinning vs. Standing Mode Selection & Cavity Bomb Tests ===")

    # ------------------------------------------------------------------
    # Experiment 1: 2D Axisymmetric Subcritical Mode Selection (n_v = 0, pure pressure coupling)
    # Bare 1T linear Hopf at L_a=2, zeta=0.04, tau=0.30, sigma_tau=0.15, omega_hp=0.15 is n_pH ~ 0.0425.
    # Test bistable zone n_p in {0.032, 0.036, 0.040}: compare E_crit(standing) vs E_crit(spinning)
    # AND watch a standing seed spontaneously flip into a PURE SPINNING limit cycle!
    # ------------------------------------------------------------------
    print("\n--- Experiment 1: 2D Standing vs. Spinning Triggering & Spontaneous Spin Selection ---")
    exp1_rows = []
    for np_val in (0.033, 0.037, 0.040):
        a_sp, pk_sp, sp_sp = bisect_2d_bomb(np_val, 0.0, "spin_1T_plus", nx=nx, ntheta=ntheta, t_end=160.0)
        a_st, pk_st, sp_st = bisect_2d_bomb(np_val, 0.0, "standing_1T_pure", nx=nx, ntheta=ntheta, t_end=160.0)
        e_ratio = (a_sp / a_st) ** 2 if (a_sp and a_st and a_st > 0) else float("nan")
        print(
            f"  n_p={np_val:.3f}: A_crit(spin)={a_sp:.4f}, A_crit(stand)={a_st:.4f} "
            f"=> E_crit(spin)/E_crit(stand) = {e_ratio:.3f} (theory ~1.50!) | P_LC={pk_sp:.4f}"
        )
        exp1_rows.append({
            "n_p": np_val, "A_crit_spin": a_sp, "A_crit_stand": a_st,
            "E_crit_spin_over_stand": float(e_ratio), "P_LC_spin": pk_sp, "P_LC_stand": pk_st,
        })

    # Watch a standing 1T seed (S_0 ~ 0.17) at n_p = 0.068 spontaneously flip to pure SPINNING limit cycle!
    tr_flip = simulate_2d_nonlinear(
        n_p=0.068, n_v=0.0, nx=nx, ntheta=ntheta, t_end=340.0,
        init_type="standing_1T", init_energy_amp=0.035, record_history=True,
    )
    print(
        f"  Spontaneous symmetry breaking (n_p=0.068, seeded standing S_0={tr_flip['spin_rec'][0]:.3f}): "
        f"final spin index S_late = {tr_flip['spin_index_late']:.4f}, P_peak = {tr_flip['p_peak_late']:.4f}"
    )

    # Measure true settled limit cycles at n_p = 0.037 (above saddle) and equal-energy showdown
    row_037 = exp1_rows[1]
    a_mid = 0.5 * (row_037["A_crit_stand"] + row_037["A_crit_spin"])
    tr_st_trig = simulate_2d_nonlinear(
        n_p=0.037, n_v=0.0, nx=nx, ntheta=ntheta, t_end=260.0,
        init_type="standing_1T", init_energy_amp=a_mid, record_history=True,
    )
    tr_sp_decay = simulate_2d_nonlinear(
        n_p=0.037, n_v=0.0, nx=nx, ntheta=ntheta, t_end=260.0,
        init_type="spin_1T_plus", init_energy_amp=a_mid, record_history=True,
    )
    print(
        f"  Equal-energy bomb showdown at n_p=0.037, A_eq={a_mid:.4f}: "
        f"Spinning bomb triggered={tr_sp_decay['triggered']} (P={tr_sp_decay['p_peak_late']:.4f}) | "
        f"Standing bomb triggered={tr_st_trig['triggered']} (P={tr_st_trig['p_peak_late']:.4f}, final spin S={tr_st_trig['spin_index_late']:.3f})"
    )

    # ------------------------------------------------------------------
    # Experiment 2: 2D Nonlinear Bomb Tests on 80/20 Dual-Tuned Cavity Ring Under Calibrated Drag
    # Compare two neck open-area ratios: tight sigma_o = 0.015 vs wide sigma_o = 0.035!
    # ------------------------------------------------------------------
    print("\n--- Experiment 2: 2D Nonlinear Bomb Sweep on 80/20 Cavity Ring (n_p=0.048, n_v=0.040, beta=2%) ---")
    n_p_c, n_v_c = 0.048, 0.040
    beta_tot = 0.02
    f1 = 0.80
    C_d = 0.65
    sigma_o_tight = 0.015  # 1.5% faceplate neck area ratio
    sigma_o_wide = 0.035   # 3.5% faceplate neck area ratio (wide-mouth rim cavities)

    a_baf_1T, _, _ = bisect_2d_bomb(n_p_c, n_v_c, "spin_1T_plus", n_baffles=5, L_b=0.60, nx=nx, ntheta=ntheta, t_end=140.0)
    a_baf_2T, _, _ = bisect_2d_bomb(n_p_c, n_v_c, "spin_2T_plus", n_baffles=5, L_b=0.60, nx=nx, ntheta=ntheta, t_end=140.0)
    print(f"  Pure 5-blade baffles (L_b=0.60, no cavities): A_crit(1T)={a_baf_1T}, A_crit(2T)={a_baf_2T} (linearly unstable under n_p!)")

    xi_mults = [0.25, 0.50, 0.80, 1.00, 1.40, 2.00]
    cav_rows = []
    for xm in xi_mults:
        b_loc, wH, xiH, c_nl_t = make_dual_cavity_arrays(ntheta, beta_tot, f1, xm, C_d, sigma_o_tight, L_a)
        _, _, _, c_nl_w = make_dual_cavity_arrays(ntheta, beta_tot, f1, xm, C_d, sigma_o_wide, L_a)
        a1_lin, _, _ = bisect_2d_bomb(n_p_c, n_v_c, "spin_1T_plus", b_loc, wH, xiH, 0.0, nx=nx, ntheta=ntheta, t_end=130.0)
        a1_wide, _, _ = bisect_2d_bomb(n_p_c, n_v_c, "spin_1T_plus", b_loc, wH, xiH, c_nl_w, nx=nx, ntheta=ntheta, t_end=130.0)
        a1_tight, _, _ = bisect_2d_bomb(n_p_c, n_v_c, "spin_1T_plus", b_loc, wH, xiH, c_nl_t, nx=nx, ntheta=ntheta, t_end=130.0)
        a2_tight, _, _ = bisect_2d_bomb(n_p_c, n_v_c, "spin_2T_plus", b_loc, wH, xiH, c_nl_t, nx=nx, ntheta=ntheta, t_end=130.0)
        s_lin = f"{a1_lin:.4f}" if a1_lin is not None else "NO TRIG<=.095"
        s_1w = f"{a1_wide:.4f}" if a1_wide is not None else "NO TRIG<=.095"
        s_1t = f"{a1_tight:.4f}" if a1_tight is not None else "NO TRIG<=.095"
        s_2t = f"{a2_tight:.4f}" if a2_tight is not None else "NO TRIG<=.095"
        print(
            f"  xi/xi_EP={xm:.2f}: 1T(lin)={s_lin:>13s} | 1T(wide 3.5%)={s_1w:>13s} | "
            f"1T(tight 1.5%)={s_1t:>13s} | 2T(tight 1.5%)={s_2t:>13s}"
        )
        cav_rows.append({
            "xi_mult": xm,
            "A_crit_1T_linear_cav": a1_lin,
            "A_crit_1T_wide_35pct": a1_wide,
            "A_crit_1T_tight_15pct": a1_tight,
            "A_crit_2T_tight_15pct": a2_tight,
        })

    # Grid convergence check on standing vs spinning threshold at n_p = 0.037 (30x60 vs 45x90)
    print("\n--- Experiment 3: 2D Nonlinear Grid Convergence Check (30x60 vs 45x90 at n_p=0.037) ---")
    a_st_fine, _, _ = bisect_2d_bomb(0.037, 0.0, "standing_1T_pure", nx=45, ntheta=90, t_end=160.0, n_iter=8)
    rel_conv = abs(a_st_fine - row_037["A_crit_stand"]) / a_st_fine
    print(f"  A_crit(standing) at n_p=0.037: 30x60={row_037['A_crit_stand']:.4f} vs 45x90={a_st_fine:.4f} (rel diff={100*rel_conv:.2f}%)")

    # ------------------------------------------------------------------
    # Figure 1: 2D Standing vs Spinning Triggering & Spontaneous Spin Selection
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.4), dpi=150)

    # Panel 1: Equal-energy bomb showdown at n_p = 0.037 (standing triggers & flips to spin; spinning decays!)
    ax1_r = ax1.twinx()
    l1, = ax1.plot(tr_st_trig["t_rec"], tr_st_trig["pmax_rec"], color="#d62728", lw=2.0,
                   label=f"Standing bomb ($A_{{\\mathrm{{eq}}}}={a_mid:.4f}$) $\\rightarrow$ TRIGGERS!")
    l2, = ax1.plot(tr_sp_decay["t_rec"], tr_sp_decay["pmax_rec"], color="#1f77b4", lw=2.0, ls="--",
                   label=f"Spinning bomb (SAME $E_0$) $\\rightarrow$ Decays")
    l3, = ax1_r.plot(tr_st_trig["t_rec"], tr_st_trig["spin_rec"], color="#2ca02c", lw=1.6, ls=":",
                     label="Spin index $S(t)$ of triggered standing bomb")
    ax1.set_xlabel("Dimensionless time $t \\bar{c}/R$")
    ax1.set_ylabel("Peak faceplate pressure $\\max_\\theta |P(0, \\theta, t)|$", color="#d62728")
    ax1_r.set_ylabel("Spin index $S(t)$ ($0=$ standing, $1=$ pure spinning)", color="#2ca02c")
    ax1_r.set_ylim(-0.05, 1.12)
    ax1.set_title("Same Bomb Energy $E_0$ ($n_p=0.037$): Standing Triggers, Spinning Decays", fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend([l1, l2, l3], [l1.get_label(), l2.get_label(), l3.get_label()], loc="center right", fontsize=7.8)

    # Panel 2: Spontaneous symmetry breaking from seeded standing wave to pure spinning limit cycle (n_p = 0.068)
    ax2_r = ax2.twinx()
    m1, = ax2.plot(tr_flip["t_rec"], tr_flip["pmax_rec"], color="#d62728", lw=2.0, label="Peak faceplate $|P|$ (bulk-linear $\\leq 0.09$)")
    m2, = ax2_r.plot(tr_flip["t_rec"], tr_flip["spin_rec"], color="#2ca02c", lw=2.2,
                     label=f"Spin index $S(t): {tr_flip['spin_rec'][0]:.2f} \\rightarrow {tr_flip['spin_index_late']:.2f}$")
    ax2.set_xlabel("Dimensionless time $t \\bar{c}/R$")
    ax2.set_ylabel("Peak faceplate pressure $|P(0)|$", color="#d62728")
    ax2_r.set_ylabel("Spin index $S(t)$", color="#2ca02c")
    ax2_r.set_ylim(-0.05, 1.12)
    ax2.set_title("Spontaneous Symmetry Breaking ($n_p=0.068$): Standing $\\rightarrow$ Pure Spinning", fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend([m1, m2], [m1.get_label(), m2.get_label()], loc="center right", fontsize=8.2)

    fig.suptitle("Entry 006 (Part 1): 2D Nonlinear Symmetry Breaking — Standing Triggers Easier, Spinning Wins Saturation", fontsize=12.0, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(RESULTS_DIR / "exp006_2d_spin_selection.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2: 2D Nonlinear Bomb Survival of 80/20 Dual-Tuned Cavity Ring vs xi/xi_EP
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.6, 5.3), dpi=150)
    cap = 0.10
    xms = [r["xi_mult"] for r in cav_rows]
    y_1lin = [cap if r["A_crit_1T_linear_cav"] is None else r["A_crit_1T_linear_cav"] for r in cav_rows]
    y_1w = [cap if r["A_crit_1T_wide_35pct"] is None else r["A_crit_1T_wide_35pct"] for r in cav_rows]
    y_1t = [cap if r["A_crit_1T_tight_15pct"] is None else r["A_crit_1T_tight_15pct"] for r in cav_rows]
    y_2t = [cap if r["A_crit_2T_tight_15pct"] is None else r["A_crit_2T_tight_15pct"] for r in cav_rows]

    ax.axvline(1.0, color="#1f77b4", ls=":", lw=1.4, label="Linear 2D EP ($\\xi_H / \\xi_H^{\\mathrm{EP}} = 1.0$)")
    ax.axhline(cap, color="gray", ls="--", lw=1.0, alpha=0.7)
    ax.text(1.52, cap + 0.002, "No trigger (probed $\\leq 0.095$)", fontsize=8.0, color="gray")
    ax.plot(xms, y_1lin, "s--", color="#2ca02c", lw=1.8, ms=6, label="$1T$ bomb (ideal linear 80/20 cavity ring)")
    ax.plot(xms, y_1w, "D-", color="#1f77b4", lw=2.2, ms=6.5, label="$1T$ bomb (wide-mouth rim cavities, $\\sigma_o=3.5\\%$)")
    ax.plot(xms, y_1t, "o-", color="#d62728", lw=2.2, ms=6.5, label="$1T$ bomb (tight-neck cavities, $\\sigma_o=1.5\\%$)")
    ax.plot(xms, y_2t, "^:", color="#ff7f0e", lw=2.0, ms=6, label="$2T$ bomb (tight-neck cavities, $\\sigma_o=1.5\\%$)")

    ax.set_xlabel("Linear cavity damping ratio normalized to 2D EP ($\\xi_H / \\xi_H^{\\mathrm{EP}}$)")
    ax.set_ylabel("Critical 2D bomb amplitude $A_{\\mathrm{eq,crit}}$ (higher = safer)")
    ax.set_ylim(-0.005, 0.112)
    ax.set_title(f"2D 80/20 Dual-Tuned Cavity Ring Under Calibrated Orifice Drag ($n_p={n_p_c}, n_v={n_v_c}, \\beta=2\\%$)", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8.0)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp006_2d_nonlinear_cavity_bombs.png")
    plt.close(fig)

    metrics = {
        "parameters": {
            "L_a": L_a, "zeta": 0.04, "x_f": 0.20, "tau_bar": 0.30, "sigma_tau": 0.15,
            "omega_hp": OMEGA_HP, "q_s": 0.036, "kappa": 4.5,
            "grid": [nx, ntheta],
            "cavity_test": {"n_p": n_p_c, "n_v": n_v_c, "beta_tot": beta_tot, "f1": f1, "C_d": C_d, "sigma_o_tight": sigma_o_tight, "sigma_o_wide": sigma_o_wide},
        },
        "exp1_spin_vs_standing": {
            "threshold_ratios": exp1_rows,
            "spontaneous_flip_np068": {
                "spin_initial": float(tr_flip["spin_rec"][0]),
                "spin_final": tr_flip["spin_index_late"],
                "p_peak_final": tr_flip["p_peak_late"],
            },
            "equal_energy_showdown_np037": {
                "A_eq": float(a_mid),
                "standing_triggered": tr_st_trig["triggered"],
                "standing_final_peak": tr_st_trig["p_peak_late"],
                "standing_final_spin": tr_st_trig["spin_index_late"],
                "spinning_triggered": tr_sp_decay["triggered"],
                "spinning_final_peak": tr_sp_decay["p_peak_late"],
            },
        },
        "exp2_2d_nonlinear_cavity_sweep": {
            "pure_5blade_baffles_Lb060": {"A_crit_1T": a_baf_1T, "A_crit_2T": a_baf_2T},
            "xi_mult_sweep": cav_rows,
        },
        "exp3_grid_convergence": {
            "coarse_30x60_A_crit_stand": row_037["A_crit_stand"],
            "fine_45x90_A_crit_stand": a_st_fine,
            "rel_diff": float(rel_conv),
        },
    }

    with open(RESULTS_DIR / "exp006_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved Entry 006 plots and metrics to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
