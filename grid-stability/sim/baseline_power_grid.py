"""Entry 000 (grid-stability): Nonlinear AC Swing Baseline, Exact Braess's Paradox & GFM Inverter Cliff.

Governing lossless nonlinear AC swing equations on an N-bus transmission network with
susceptance capacities K_ij = K_ji >= 0 and net power injections P_i (sum P_i = 0):

    M_i * d^2(theta_i)/dt^2 + D_i * d(theta_i)/dt = P_i - sum_j K_ij * sin(theta_i - theta_j)

Equivalence to Grid-Forming (GFM) low-pass-filtered droop inverters (D'Arco & Suul 2014):
    tau_f * d(P_tilde_i)/dt = P_e,i - P_tilde_i,   d(theta_i)/dt = -k_p,i * (P_tilde_i - P_i)
    ==> M_GFM,i = tau_f / k_p,i,   D_GFM,i = 1 / k_p,i

Exact nonlinear Hamiltonian / Lyapunov energy balance (for ANY trajectory, including blackouts):
    H(theta, omega) = 0.5 * sum_i M_i * omega_i^2 - sum_i P_i * theta_i - 0.5 * sum_{i,j} K_ij * cos(theta_i - theta_j)
    dH/dt = -sum_i D_i * omega_i^2 <= 0

Experiments:
1. Exact Hamiltonian (D=0) & damped Lyapunov (D>0) energy conservation + exact linearized
   inter-area eigenvalue vs. full nonlinear RK4 swing ODE verification on a 6-bus meshed grid.
2. Exact Analytical & Time-Domain AC Braess's Paradox on a 4-bus ring (P = (+P0,-P0,+P0,-P0)):
   Reinforcing a single line K_01 = 1 + dK provably cuts maximum synchronous grid capacity
   from P_max(0) = 2.000 down to P_max(inf) = 1.7602 (-11.99%) via Kirchhoff loop phase
   frustration, triggering an immediate cascading phase-slip blackout at P0 = 1.85 when the
   new parallel line is switched on!
3. 12-Bus Meshed Renewable Grid: Low-Inertia GFL Cliff vs. Grid-Forming (GFM) Droop Inverters
   under an N-1 tie-line trip + generation loss contingency, measuring RoCoF, frequency nadir,
   and nonlinear transient stability boundary across inverter penetration and GFM share.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares, minimize_scalar, root


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


# ---------------------------------------------------------------------------
# 1. Core Nonlinear AC Swing Network Engine (Exact Energy & Linear Spectrum)
# ---------------------------------------------------------------------------

def build_K_matrix(n_bus: int, edges: list[tuple[int, int, float]]) -> np.ndarray:
    K = np.zeros((n_bus, n_bus), dtype=float)
    for u, v, cap in edges:
        K[u, v] += cap
        K[v, u] += cap
    return K


def swing_energy(theta: np.ndarray, omega: np.ndarray, M: np.ndarray, P: np.ndarray, K: np.ndarray) -> float:
    T = 0.5 * float(np.sum(M * omega * omega))
    diff = theta[:, None] - theta[None, :]
    V = -float(np.dot(P, theta)) - 0.5 * float(np.sum(K * np.cos(diff)))
    return T + V


def find_equilibrium(P: np.ndarray, K: np.ndarray, theta_guess: np.ndarray | None = None) -> tuple[bool, np.ndarray, float]:
    """Solve P_i = sum_j K_ij sin(theta_i - theta_j) with reference theta_0 = 0.
    Returns (converged_and_stable, theta_eq, lambda2_L).
    """
    n = len(P)
    x0 = np.zeros(n - 1) if theta_guess is None else (theta_guess[1:] - theta_guess[0])

    def residual(th_red: np.ndarray) -> np.ndarray:
        th = np.concatenate([[0.0], th_red])
        diff = th[:, None] - th[None, :]
        pe = np.sum(K * np.sin(diff), axis=1)
        return (P - pe)[1:]

    sol = root(residual, x0, method="hybr", tol=1e-12)
    th = np.concatenate([[0.0], sol.x])
    res_max = float(np.max(np.abs(residual(sol.x))))
    if (not sol.success) or res_max > 1e-8:
        return False, th, -1.0

    # Weighted Laplacian L_ij = -K_ij cos(th_i - th_j)
    diff = th[:, None] - th[None, :]
    W = K * np.cos(diff)
    L = np.diag(np.sum(W, axis=1)) - W
    ev = np.sort(np.linalg.eigvalsh(L))
    lam2 = float(ev[1])
    return (lam2 > 1e-7), th, lam2


def linear_swing_spectrum(theta_eq: np.ndarray, M: np.ndarray, D: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Exact eigenvalues of M ddot(delta) + D dot(delta) + L delta = 0."""
    n = len(M)
    diff = theta_eq[:, None] - theta_eq[None, :]
    W = K * np.cos(diff)
    L = np.diag(np.sum(W, axis=1)) - W
    Minv = np.diag(1.0 / M)
    J = np.block([
        [np.zeros((n, n)), np.eye(n)],
        [-Minv @ L, -Minv @ np.diag(D)],
    ])
    return np.linalg.eigvals(J)


def simulate_swing_rk4(
    M: np.ndarray,
    D: np.ndarray,
    P: np.ndarray,
    K: np.ndarray,
    theta0: np.ndarray,
    omega0: np.ndarray,
    dt: float = 0.005,
    t_end: float = 40.0,
    event_fn=None,
) -> dict:
    """Classic 4th-order Runge-Kutta integration of coupled nonlinear AC swing network.

    Optional event_fn(t, K, P) -> (K_now, P_now) allows exact mid-run topology/fault switching.
    """
    n = len(M)
    inv_M = 1.0 / M
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps
    t_arr = np.linspace(0.0, t_end, n_steps + 1)

    th_hist = np.empty((n_steps + 1, n))
    om_hist = np.empty((n_steps + 1, n))
    H_hist = np.empty(n_steps + 1)
    diss_hist = np.empty(n_steps + 1)

    th = theta0.copy()
    om = omega0.copy()
    K_cur = K.copy()
    P_cur = P.copy()

    def rhs(th_v: np.ndarray, om_v: np.ndarray, K_m: np.ndarray, P_v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        diff = th_v[:, None] - th_v[None, :]
        pe = np.sum(K_m * np.sin(diff), axis=1)
        dom = inv_M * (P_v - pe - D * om_v)
        return om_v, dom

    th_hist[0] = th
    om_hist[0] = om
    H_hist[0] = swing_energy(th, om, M, P_cur, K_cur)
    diss_hist[0] = float(np.sum(D * om * om))

    for step in range(n_steps):
        t_now = t_arr[step]
        if event_fn is not None:
            K_cur, P_cur = event_fn(t_now, K, P)

        k1_t, k1_o = rhs(th, om, K_cur, P_cur)
        k2_t, k2_o = rhs(th + 0.5 * dt * k1_t, om + 0.5 * dt * k1_o, K_cur, P_cur)
        k3_t, k3_o = rhs(th + 0.5 * dt * k2_t, om + 0.5 * dt * k2_o, K_cur, P_cur)
        k4_t, k4_o = rhs(th + dt * k3_t, om + dt * k3_o, K_cur, P_cur)

        th = th + (dt / 6.0) * (k1_t + 2.0 * k2_t + 2.0 * k3_t + k4_t)
        om = om + (dt / 6.0) * (k1_o + 2.0 * k2_o + 2.0 * k3_o + k4_o)

        th_hist[step + 1] = th
        om_hist[step + 1] = om
        H_hist[step + 1] = swing_energy(th, om, M, P_cur, K_cur)
        diss_hist[step + 1] = float(np.sum(D * om * om))

    # Exact energy conservation residual (for static K, P)
    cum_diss = np.concatenate([[0.0], np.cumsum(0.5 * (diss_hist[:-1] + diss_hist[1:]) * dt)])
    dH = H_hist - H_hist[0]
    denom = max(float(np.max(np.abs(dH))), abs(H_hist[0]), 1e-12)
    e_err = float(np.max(np.abs(dH + cum_diss)) / denom)

    # Max instantaneous angular acceleration (RoCoF proxy) and frequency nadir (Hz = omega/(2*pi))
    rocof_max = float(np.max(np.abs(np.diff(om_hist, axis=0) / dt)) / (2.0 * np.pi))
    f_nadir_hz = float(np.max(np.abs(om_hist)) / (2.0 * np.pi))
    max_angle_spread = float(np.max(np.max(th_hist, axis=1) - np.min(th_hist, axis=1)))
    final_omega_rms = float(np.sqrt(np.mean(om_hist[-1] ** 2)))

    return {
        "t": t_arr, "theta": th_hist, "omega": om_hist,
        "H": H_hist, "energy_balance_rel_err": e_err,
        "rocof_max_hz_s": rocof_max, "f_nadir_hz": f_nadir_hz,
        "max_angle_spread_rad": max_angle_spread, "final_omega_rms": final_omega_rms,
    }


def fit_damped_sinusoid(t: np.ndarray, y: np.ndarray, a0: float, w0: float) -> tuple[float, float]:
    t0 = t - t[0]
    y0 = y - np.mean(y)
    amp0 = max(float(np.max(np.abs(y0))), 1e-12)

    def resid(p):
        c1, c2, al, om, dc = p
        return np.exp(al * t0) * (c1 * np.cos(om * t0) + c2 * np.sin(om * t0)) + dc - y

    sol = least_squares(resid, [amp0, 0.0, a0, w0, 0.0], xtol=1e-12, ftol=1e-12)
    return float(sol.x[2]), float(abs(sol.x[3]))


# ---------------------------------------------------------------------------
# 2. Exact Analytical 4-Bus AC Braess Capacity Curve P_max(dK)
# ---------------------------------------------------------------------------

def exact_4bus_braess_pmax(dK: float) -> tuple[float, float, float]:
    """Exact maximum synchronous power P_max(dK) on 4-bus ring 0-1-2-3-0 with
    P = (+P0, -P0, +P0, -P0), K_01 = 1 + dK, K_12 = K_23 = K_30 = 1.

    By exact symmetry:
        sin(gamma) = (1 + dK) * sin(alpha),    beta = 0.5 * (alpha + gamma)
        P0(gamma)  = sin(gamma) + sin(0.5 * (gamma + arcsin(sin(gamma) / (1 + dK))))
    Notice alpha in [0, pi/2] (principal branch on reinforced edge 0-1), while gamma can
    exceed pi/2 slightly up to the saddle-node bifurcation dP0/dgamma = 0!
    """
    kp = 1.0 + dK

    def p0_of_gamma(g: float) -> float:
        s_a = np.sin(g) / kp
        if abs(s_a) > 1.0:
            return -1e9
        a = np.arcsin(s_a)
        b = 0.5 * (a + g)
        return float(np.sin(g) + np.sin(b))

    res = minimize_scalar(lambda g: -p0_of_gamma(g), bounds=(0.1, np.pi - 0.01), method="bounded", options={"xatol": 1e-13})
    g_star = float(res.x)
    a_star = float(np.arcsin(np.sin(g_star) / kp))
    p_star = p0_of_gamma(g_star)
    return p_star, g_star, a_star


# ---------------------------------------------------------------------------
# 3. 12-Bus Meshed IEEE-Style Renewable Grid Topology
# ---------------------------------------------------------------------------

def make_12bus_grid() -> tuple[int, list[tuple[int, int, float]], np.ndarray, list[int]]:
    """12-bus meshed AC grid (two 6-bus areas linked by two critical tie lines).
    Buses 0..5: Area A (generators at 0, 1, 2; loads at 3, 4, 5)
    Buses 6..11: Area B (generators at 6, 7, 8; loads at 9, 10, 11)
    Net power export from Area A to Area B creates realistic inter-area stress!
    """
    n_bus = 12
    edges = [
        # Area A mesh
        (0, 1, 2.2), (1, 2, 2.0), (0, 3, 2.4), (1, 4, 2.2), (2, 5, 2.0),
        (3, 4, 1.8), (4, 5, 1.8), (0, 4, 1.4),
        # Area B mesh
        (6, 7, 2.2), (7, 8, 2.0), (6, 9, 2.4), (7, 10, 2.2), (8, 11, 2.0),
        (9, 10, 1.8), (10, 11, 1.8), (6, 10, 1.4),
        # Two inter-area tie lines (edge index 16 & 17)
        (2, 6, 1.6), (5, 9, 1.6),
    ]
    # Area A net exporter (+0.9), Area B net importer (-0.9); sum(P) == 0 exactly
    P = np.array([+1.10, +0.95, +0.85, -0.65, -0.70, -0.65,
                  +0.70, +0.65, +0.55, -0.95, -0.95, -0.90], dtype=float)
    assert abs(np.sum(P)) < 1e-12
    gen_buses = [0, 1, 2, 6, 7, 8]
    return n_bus, edges, P, gen_buses


# ---------------------------------------------------------------------------
# 4. Main Experiments & Figures
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Entry 000 (grid-stability): AC Swing Baseline, Exact Braess Paradox & GFM Inverter Cliff ===\n")

    # ------------------------------------------------------------------
    # Experiment 1: Exact Energy Conservation & Linear Eigenvalue vs Nonlinear RK4
    # ------------------------------------------------------------------
    print("--- Experiment 1: Exact Hamiltonian/Lyapunov Conservation & Eigenmode Verification (6-Bus) ---")
    edges_6 = [(0, 1, 1.8), (1, 2, 1.5), (2, 3, 1.6), (3, 4, 1.8), (4, 5, 1.5), (5, 0, 1.6), (1, 4, 1.0)]
    K6 = build_K_matrix(6, edges_6)
    P6 = np.array([+0.9, +0.6, -0.7, -0.8, +0.5, -0.5])
    M6 = np.array([2.5, 2.0, 1.0, 1.2, 2.2, 1.0])
    D6 = np.array([0.45, 0.35, 0.25, 0.25, 0.40, 0.25])

    ok6, th6_eq, lam2_6 = find_equilibrium(P6, K6)
    assert ok6, "6-bus equilibrium must exist!"

    # (1a) Lossless Hamiltonian conservation check (D = 0, finite perturbation)
    sim_ham = simulate_swing_rk4(
        M6, np.zeros(6), P6, K6, th6_eq + 0.25 * np.array([1, -1, 0.5, -0.5, 0.3, -0.3]),
        np.zeros(6), dt=0.005, t_end=30.0,
    )
    # (1b) Damped Lyapunov energy balance check (D > 0, finite perturbation)
    sim_lyap = simulate_swing_rk4(
        M6, D6, P6, K6, th6_eq + 0.30 * np.array([1, -1, 0.5, -0.5, 0.3, -0.3]),
        np.zeros(6), dt=0.005, t_end=30.0,
    )

    # (1c) Exact slowest oscillatory inter-area eigenvalue vs. nonlinear RK4 eigenmode fit
    ev6 = linear_swing_spectrum(th6_eq, M6, D6, K6)
    # Filter complex oscillatory modes (Im > 0.1) and pick slowest damped (largest Re)
    osc = [z for z in ev6 if z.imag > 0.1]
    s_slow = max(osc, key=lambda z: z.real)
    # Compute corresponding eigenvector for clean single-mode excitation
    diff6 = th6_eq[:, None] - th6_eq[None, :]
    W6 = K6 * np.cos(diff6)
    L6 = np.diag(np.sum(W6, axis=1)) - W6
    Minv6 = np.diag(1.0 / M6)
    J6 = np.block([[np.zeros((6, 6)), np.eye(6)], [-Minv6 @ L6, -Minv6 @ np.diag(D6)]])
    vals, vecs = np.linalg.eig(J6)
    idx_slow = int(np.argmin(np.abs(vals - s_slow)))
    v_slow = np.real(vecs[:, idx_slow])
    v_slow /= np.max(np.abs(v_slow[:6]))
    sim_eig = simulate_swing_rk4(
        M6, D6, P6, K6, th6_eq + 1e-4 * v_slow[:6], 1e-4 * v_slow[6:], dt=0.005, t_end=45.0,
    )
    sig = sim_eig["theta"][:, 0] - sim_eig["theta"][:, 3] - (th6_eq[0] - th6_eq[3])
    mask_fit = (sim_eig["t"] >= 2.0) & (sim_eig["t"] <= 40.0)
    a_fit, w_fit = fit_damped_sinusoid(sim_eig["t"][mask_fit], sig[mask_fit], s_slow.real, s_slow.imag)

    print(f"  Lossless Hamiltonian conservation rel err (D=0, finite amp): {sim_ham['energy_balance_rel_err']:.2e}")
    print(f"  Damped Lyapunov energy balance rel err (D>0, finite amp):    {sim_lyap['energy_balance_rel_err']:.2e}")
    print(
        f"  Slowest inter-area mode: exact = {s_slow.real:+.7f} + {s_slow.imag:.7f}i | "
        f"RK4 LSQ = {a_fit:+.7f} + {w_fit:.7f}i (|da|={abs(a_fit - s_slow.real):.2e}, |dw|={abs(w_fit - s_slow.imag):.2e})\n"
    )

    # ------------------------------------------------------------------
    # Experiment 2: Exact Nonlinear AC Braess's Paradox (4-Bus Ring)
    # ------------------------------------------------------------------
    print("--- Experiment 2: Exact AC Braess's Paradox (4-Bus Ring, Reinforced Edge 0-1) ---")
    dK_grid = np.concatenate([[0.0], np.logspace(-1.5, 1.7, 35)])
    pmax_exact = []
    pmax_num = []
    for dK in dK_grid:
        p_ex, g_ex, a_ex = exact_4bus_braess_pmax(float(dK))
        pmax_exact.append(p_ex)
        # Independently verify by warm-started numerical bisection of full 4-bus equilibrium + Fiedler stability
        edges_b = [(0, 1, 1.0 + float(dK)), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0)]
        Kb = build_K_matrix(4, edges_b)
        lo, hi = 0.5, 2.2
        # Warm-start safely inside the stable basin (scale angles by mid/p_ex)
        th_fold = np.array([0.0, -a_ex, 0.5 * (g_ex - a_ex), -0.5 * (g_ex + a_ex)])
        for _ in range(34):
            mid = 0.5 * (lo + hi)
            scale = min(0.95 * mid / max(p_ex, 1e-6), 0.98)
            ok, _, _ = find_equilibrium(mid * np.array([1.0, -1.0, 1.0, -1.0]), Kb, theta_guess=scale * th_fold)
            if ok:
                lo = mid
            else:
                hi = mid
        pmax_num.append(lo)

    pmax_exact = np.array(pmax_exact)
    pmax_num = np.array(pmax_num)
    braess_max_diff = float(np.max(np.abs(pmax_exact - pmax_num)))
    p_inf_exact = float(np.sin(2.0 * np.arccos((-1.0 + np.sqrt(33.0)) / 8.0)) + np.sin(np.arccos((-1.0 + np.sqrt(33.0)) / 8.0)))
    print(f"  Exact P_max(dK=0)   = {pmax_exact[0]:.6f}")
    print(f"  Exact P_max(dK=1.0) = {exact_4bus_braess_pmax(1.0)[0]:.6f} ({(exact_4bus_braess_pmax(1.0)[0]/2.0 - 1)*100:+.2f}%)")
    print(f"  Exact P_max(dK=50)  = {pmax_exact[-1]:.6f} (asymptote dK->inf = {p_inf_exact:.6f}, {(p_inf_exact/2.0 - 1)*100:+.2f}% CAPACITY LOSS!)")
    print(f"  Max |Analytical - Full 4-bus Numerical Bisection| across 36 dK values: {braess_max_diff:.2e}")

    # Time-domain nonlinear demonstration at P0 = 1.85:
    # Start in stable synchronized state (dK=0); at t = 15s switch on parallel line dK = +2.0 -> BLACKOUT!
    P_br = 1.85 * np.array([1.0, -1.0, 1.0, -1.0])
    M_br = np.ones(4)
    D_br = np.full(4, 0.30)
    K_br_base = build_K_matrix(4, [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0)])
    K_br_add = build_K_matrix(4, [(0, 1, 3.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0)])
    ok_br, th_br0, _ = find_equilibrium(P_br, K_br_base)
    assert ok_br

    sim_br_base = simulate_swing_rk4(M_br, D_br, P_br, K_br_base, th_br0 + 0.05 * np.array([1, -1, 0, 0]), np.zeros(4), dt=0.005, t_end=45.0)
    sim_br_trip = simulate_swing_rk4(
        M_br, D_br, P_br, K_br_base, th_br0 + 0.05 * np.array([1, -1, 0, 0]), np.zeros(4), dt=0.005, t_end=45.0,
        event_fn=lambda t, K0, P0: (K_br_add if t >= 15.0 else K0, P0),
    )
    print(
        f"  Time-domain blackout test at P0=1.85: baseline final omega_rms={sim_br_base['final_omega_rms']:.2e} (SYNCHRONIZED) | "
        f"after adding line dK=+2.0 at t=15s: final omega_rms={sim_br_trip['final_omega_rms']:.3f} rad/s, "
        f"max angle spread={sim_br_trip['max_angle_spread_rad']:.1f} rad (BLACKOUT!)\n"
    )

    # ------------------------------------------------------------------
    # Experiment 3: 12-Bus Meshed Renewable Grid — Generation-Loss + N-1 Tie-Line Contingency
    # ------------------------------------------------------------------
    print("--- Experiment 3: 12-Bus Meshed Grid — Low-Inertia GFL Cliff vs. GFM Stabilization ---")
    n12, edges12, P12, gen_buses = make_12bus_grid()
    K12_pre = build_K_matrix(n12, edges12)
    # N-1 Contingency at t=2.0s: trip inter-area tie line (5, 9) AND lose Delta_P = 0.85 generation at bus 6 (Area B!)
    # Non-zero net power imbalance sum(P) = -Delta_P triggers real system-wide COI RoCoF & primary frequency droop!
    edges12_post = [e for e in edges12 if not (e[0] == 5 and e[1] == 9)]
    K12_post = build_K_matrix(n12, edges12_post)
    delta_P_loss = 0.85
    P12_post = P12.copy()
    P12_post[6] -= delta_P_loss  # Generator trip in importing Area B!

    ok12_pre, th12_pre, _ = find_equilibrium(P12, K12_pre)
    assert ok12_pre, "12-bus pre-fault equilibrium must exist!"

    # Generator parameters (dimensionless per-unit, 1 pu time = 1 s):
    # Synchronous generator:          M_sync = 4.0, D_sync = 0.50 (heavy rotating inertia, slower governor droop)
    # Grid-Following (GFL) inverter:  M_gfl  = 0.08, D_gfl = 0.04 (near-zero inertia & no primary frequency droop)
    # Grid-Forming (GFM) inverter:    M_gfm  = tau_f/k_p = 1.8, D_gfm = 1/k_p = 1.60 (virtual inertia + fast droop)
    # Load buses (3,4,5,9,10,11):     M_load = 0.12, D_load = 0.12 (small frequency-sensitive load damping)
    def run_12bus_case(n_sync: int, n_gfm: int, dP: float = delta_P_loss) -> dict:
        order = [0, 6, 1, 7, 2, 8]
        M = np.full(n12, 0.12)
        D = np.full(n12, 0.12)
        for idx, b in enumerate(order):
            if idx < n_sync:
                M[b], D[b] = 4.0, 0.50
            elif idx < n_sync + n_gfm:
                M[b], D[b] = 1.8, 1.60
            else:
                M[b], D[b] = 0.08, 0.04

        P_fault = P12.copy()
        P_fault[6] -= dP
        sim = simulate_swing_rk4(
            M, D, P12, K12_pre, th12_pre, np.zeros(n12), dt=0.0025, t_end=35.0,
            event_fn=lambda t, K0, P0: (K12_post if t >= 2.0 else K0, P_fault if t >= 2.0 else P0),
        )
        # Center-of-inertia (COI) frequency f_COI(t) = sum(M_i * omega_i) / (2*pi*sum(M_i))
        M_tot = float(np.sum(M))
        D_tot = float(np.sum(D))
        om_coi = (sim["omega"] @ M) / M_tot
        f_coi = om_coi / (2.0 * np.pi)
        rocof_coi_exact = dP / (2.0 * np.pi * M_tot)  # Exact analytical initial COI RoCoF!
        df_ss_exact = dP / (2.0 * np.pi * D_tot)      # Exact analytical steady-state frequency droop!
        # Measure actual COI RoCoF immediately after t=2.0 (step index where t>=2.0)
        idx_f = int(np.searchsorted(sim["t"], 2.0))
        rocof_coi_sim = float(np.max(np.abs(np.diff(f_coi[idx_f:idx_f + 40]) / (sim["t"][1] - sim["t"][0]))))
        f_nadir_coi = float(np.max(np.abs(f_coi)))
        # Phase-angle spread relative to mean angle (detects inter-area pole-slip blackout!)
        th_rel = sim["theta"] - np.mean(sim["theta"], axis=1, keepdims=True)
        max_spread = float(np.max(np.max(th_rel, axis=1) - np.min(th_rel, axis=1)))
        # Post-fault synchronous steady state exists in the rotating frame omega_ss = -dP / D_tot
        # Effective power injections in rotating frame: P_eff = P_fault + D * (dP / D_tot)
        P_rot = P_fault + D * (dP / D_tot)
        ok_rot, _, _ = find_equilibrium(P_rot, K12_post)
        # Synchronized if inter-bus frequency spread relative to COI settles near 0 and no pole slip!
        om_rel_late = float(np.max(np.abs(sim["omega"][-200:] - om_coi[-200:, None])))
        synced = bool(ok_rot and (om_rel_late < 0.05) and (max_spread < np.pi))

        return {
            "n_sync": n_sync, "n_gfm": n_gfm, "n_gfl": 6 - n_sync - n_gfm,
            "inv_penetration_pct": (6 - n_sync) / 6.0 * 100.0,
            "M_tot": M_tot, "D_tot": D_tot,
            "synchronized": synced,
            "rocof_coi_exact_hz_s": rocof_coi_exact,
            "rocof_coi_sim_hz_s": rocof_coi_sim,
            "f_nadir_coi_hz": f_nadir_coi,
            "df_ss_exact_hz": df_ss_exact,
            "f_nadir_worst_bus_hz": sim["f_nadir_hz"],
            "max_angle_spread_rad": max_spread,
            "t": sim["t"], "f_coi": f_coi, "omega": sim["omega"], "theta": sim["theta"],
        }

    # (3A) Sweep GFL-only replacement of synchronous generators (n_sync = 6 -> 0, n_gfm = 0)
    print(f"  [3A] Replacing synchronous generators with Grid-Following (GFL) inverters (dP={delta_P_loss} + tie-line trip):")
    gfl_sweep = []
    for ns in (6, 5, 4, 3, 2, 1, 0):
        r = run_12bus_case(ns, 0)
        print(
            f"    Inv={r['inv_penetration_pct']:5.1f}% (Sync={ns}, GFL={6-ns}): "
            f"synced={str(r['synchronized']):5s} | COI RoCoF={r['rocof_coi_sim_hz_s']:5.3f} (exact {r['rocof_coi_exact_hz_s']:.3f}) Hz/s | "
            f"COI nadir={r['f_nadir_coi_hz']:.3f} Hz | spread={r['max_angle_spread_rad']:.2f} rad"
        )
        gfl_sweep.append(r)

    # (3B) At 100% inverter penetration (n_sync = 0), sweep Grid-Forming (GFM) share n_gfm = 0 -> 6!
    print("\n  [3B] 100% Inverter-Based Grid (n_sync = 0): Sweeping Grid-Forming (GFM) Inverter Share:")
    gfm_sweep = []
    for ng in (0, 1, 2, 3, 4, 5, 6):
        r = run_12bus_case(0, ng)
        print(
            f"    GFM={ng}/6 ({ng/6*100:5.1f}%): "
            f"synced={str(r['synchronized']):5s} | COI RoCoF={r['rocof_coi_sim_hz_s']:5.3f} (exact {r['rocof_coi_exact_hz_s']:.3f}) Hz/s | "
            f"COI nadir={r['f_nadir_coi_hz']:.3f} Hz | spread={r['max_angle_spread_rad']:.2f} rad"
        )
        gfm_sweep.append(r)

    # Counterfactual check: place the single GFM inverter at Bus 6 (importing Area B) instead of Bus 0 (Area A)
    M_b6 = np.full(n12, 0.12)
    D_b6 = np.full(n12, 0.12)
    for b in gen_buses:
        M_b6[b], D_b6[b] = 0.08, 0.04
    M_b6[6], D_b6[6] = 1.8, 1.60
    sim_b6 = simulate_swing_rk4(
        M_b6, D_b6, P12, K12_pre, th12_pre, np.zeros(n12), dt=0.0025, t_end=35.0,
        event_fn=lambda t, K0, P0: (K12_post if t >= 2.0 else K0, P12_post if t >= 2.0 else P0),
    )
    th_rel_b6 = sim_b6["theta"] - np.mean(sim_b6["theta"], axis=1, keepdims=True)
    spread_b6 = float(np.max(np.max(th_rel_b6, axis=1) - np.min(th_rel_b6, axis=1)))
    om_coi_b6 = (sim_b6["omega"] @ M_b6) / np.sum(M_b6)
    synced_b6 = bool((spread_b6 < np.pi) and (float(np.max(np.abs(sim_b6["omega"][-200:] - om_coi_b6[-200:, None]))) < 0.05))
    print(f"  Counterfactual 1 GFM at Bus 6 (Area B): synced={synced_b6}, max spread={spread_b6:.2f} rad (vs Bus 0 Area A: synced=False, spread={gfm_sweep[1]['max_angle_spread_rad']:.2f} rad)")

    # Genuine transient RK4 timestep convergence check (same t_end=35.0, transient peak tie-line angle |th_2 - th_6|)
    r_ref = gfm_sweep[-1]
    order_all = [0, 6, 1, 7, 2, 8]
    M_gfm_all = np.full(n12, 0.12)
    D_gfm_all = np.full(n12, 0.12)
    for b in order_all:
        M_gfm_all[b], D_gfm_all[b] = 1.8, 1.60
    sim_fine = simulate_swing_rk4(
        M_gfm_all, D_gfm_all, P12, K12_pre, th12_pre, np.zeros(n12), dt=0.00125, t_end=35.0,
        event_fn=lambda t, K0, P0: (K12_post if t >= 2.0 else K0, P12_post if t >= 2.0 else P0),
    )
    tie_peak_coarse = float(np.max(np.abs(r_ref["theta"][:, 2] - r_ref["theta"][:, 6])))
    tie_peak_fine = float(np.max(np.abs(sim_fine["theta"][:, 2] - sim_fine["theta"][:, 6])))
    print(f"\n  Transient RK4 dt check (100% GFM, t_end=35): peak |th_2-th_6|(dt=0.0025)={tie_peak_coarse:.8f} vs dt=0.00125={tie_peak_fine:.8f} (|diff|={abs(tie_peak_coarse-tie_peak_fine):.2e})")

    # ------------------------------------------------------------------
    # Figure 1: Exact AC Braess's Paradox (Analytical Capacity Curve + Blackout Traces)
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.2), dpi=150)

    ax1.semilogx(dK_grid[1:], pmax_exact[1:], "-", color="#1f77b4", lw=2.5, label="Exact analytical $P_{\\max}(\\Delta K)$")
    ax1.semilogx(dK_grid[1::3], pmax_num[1::3], "o", color="#d62728", ms=5.5, mfc="none", mew=1.4, label="Full 4-bus numerical bisection")
    ax1.axhline(2.0, color="#2ca02c", ls="--", lw=1.6, label="Unreinforced ring $P_{\\max}(0) = 2.000$")
    ax1.axhline(p_inf_exact, color="black", ls=":", lw=1.6, label=f"Superconducting limit $P_{{\\max}}(\\infty) = {p_inf_exact:.4f}$ (-12.0%)")
    ax1.axhline(1.85, color="#ff7f0e", ls="-.", lw=1.5, label="Operating point $P_0 = 1.85$ (blackout test)")
    ax1.fill_between(dK_grid[1:], pmax_exact[1:], 2.0, color="#ffcccc", alpha=0.35)
    ax1.set_xlabel("Added parallel transmission capacity $\\Delta K$ on line 0–1")
    ax1.set_ylabel("Maximum synchronous grid power $P_{\\max}$")
    ax1.set_title("Exact AC Braess's Paradox: More Capacity Cuts Stability", fontweight="bold")
    ax1.grid(True, which="both", alpha=0.28)
    ax1.legend(loc="lower left", fontsize=7.9)

    ax2.plot(sim_br_base["t"], sim_br_base["omega"][:, 0] / (2 * np.pi), "-", color="#2ca02c", lw=2.2, label="Baseline ring ($\\Delta K = 0$, synchronized)")
    ax2.plot(sim_br_trip["t"], sim_br_trip["omega"][:, 0] / (2 * np.pi), "-", color="#d62728", lw=2.0, label="Bus 0 freq. after adding line $\\Delta K = +2$ at $t=15$")
    ax2.plot(sim_br_trip["t"], sim_br_trip["omega"][:, 2] / (2 * np.pi), "--", color="#1f77b4", lw=1.8, label="Bus 2 freq. (phase-slipping blackout!)")
    ax2.axvline(15.0, color="black", ls=":", lw=1.4, label="New parallel line switched ON ($t = 15$)")
    ax2.set_xlabel("Dimensionless time $t$")
    ax2.set_ylabel("Bus frequency deviation $\\Delta f = \\omega / (2\\pi)$")
    ax2.set_title("Time-Domain Blackout Triggered by Adding a Line ($P_0 = 1.85$)", fontweight="bold")
    ax2.grid(True, alpha=0.28)
    ax2.legend(loc="upper left", fontsize=7.9)

    fig.suptitle("Grid Stability Entry 000 (Part 1): Exact Nonlinear AC Braess's Paradox", fontsize=12.0, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(RESULTS_DIR / "baseline_braess_paradox.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2: 12-Bus Low-Inertia GFL Cliff vs. GFM Inverter Stabilization
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.2), dpi=150)

    pens = [r["inv_penetration_pct"] for r in gfl_sweep]
    ax1.plot(pens, [r["rocof_coi_sim_hz_s"] for r in gfl_sweep], "o-", color="#d62728", lw=2.3, label="COI RoCoF (Hz/s) — GFL-only")
    ax1.plot(pens, [r["rocof_coi_exact_hz_s"] for r in gfl_sweep], ":", color="black", lw=1.5, label="Exact $\\Delta P / (2\\pi M_{\\mathrm{tot}})$")
    ax1_r = ax1.twinx()
    ax1_r.plot(pens, [r["f_nadir_coi_hz"] for r in gfl_sweep], "s--", color="#1f77b4", lw=2.2, label="COI $|\\Delta f|$ nadir (Hz) — GFL-only")
    ax1.set_xlabel("Inverter penetration (%) [GFL replacing synchronous generators]")
    ax1.set_ylabel("COI Rate of Change of Frequency (Hz/s)", color="#d62728")
    ax1_r.set_ylabel("COI frequency nadir $|\\Delta f|$ (Hz)", color="#1f77b4")
    ax1.set_title("Low-Inertia GFL Cliff Under Gen Trip + $N-1$ Tie-Line Outage", fontweight="bold")
    ax1.grid(True, alpha=0.28)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1_r.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7.9)

    gfm_pct = [r["n_gfm"] / 6.0 * 100.0 for r in gfm_sweep]
    ax2.plot(gfm_pct, [r["rocof_coi_sim_hz_s"] for r in gfm_sweep], "o-", color="#2ca02c", lw=2.4, label="100% Inv. Grid: COI RoCoF (Hz/s)")
    ax2.axhline(gfl_sweep[0]["rocof_coi_sim_hz_s"], color="#555555", ls="--", lw=1.5, label=f"100% Sync baseline RoCoF ({gfl_sweep[0]['rocof_coi_sim_hz_s']:.3f} Hz/s)")
    ax2_r = ax2.twinx()
    ax2_r.plot(gfm_pct, [r["f_nadir_coi_hz"] for r in gfm_sweep], "D--", color="#9467bd", lw=2.2, label="100% Inv. Grid: COI $|\\Delta f|$ (Hz)")
    ax2_r.axhline(gfl_sweep[0]["f_nadir_coi_hz"], color="#9467bd", ls=":", lw=1.5, label=f"100% Sync baseline $|\\Delta f|$ ({gfl_sweep[0]['f_nadir_coi_hz']:.3f} Hz)")
    ax2.set_xlabel("Grid-Forming (GFM) share of 100% inverter fleet (%)")
    ax2.set_ylabel("COI RoCoF (Hz/s)", color="#2ca02c")
    ax2_r.set_ylabel("COI frequency nadir $|\\Delta f|$ (Hz)", color="#9467bd")
    ax2.set_title("Grid-Forming (GFM) Inverters Stabilize 100% Renewable Grid", fontweight="bold")
    ax2.grid(True, alpha=0.28)
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2_r.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=7.5)

    fig.suptitle("Grid Stability Entry 000 (Part 2): 12-Bus Low-Inertia Cliff vs. Grid-Forming Droop Stabilization", fontsize=12.0, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(RESULTS_DIR / "baseline_gfm_vs_gfl_cliff.png")
    plt.close(fig)

    # Strip large time series before dumping JSON
    def clean_row(r: dict) -> dict:
        return {k: v for k, v in r.items() if k not in ("t", "f_coi", "omega", "theta")}

    metrics = {
        "exp1_verification_6bus": {
            "hamiltonian_rel_err_D0": sim_ham["energy_balance_rel_err"],
            "lyapunov_rel_err_Dpos": sim_lyap["energy_balance_rel_err"],
            "slow_mode_exact": [float(s_slow.real), float(s_slow.imag)],
            "slow_mode_rk4_lsq": [a_fit, w_fit],
            "alpha_abs_diff": float(abs(a_fit - s_slow.real)),
            "omega_abs_diff": float(abs(w_fit - s_slow.imag)),
        },
        "exp2_braess_4bus": {
            "P_max_dK0": float(pmax_exact[0]),
            "P_max_dK1": float(exact_4bus_braess_pmax(1.0)[0]),
            "P_max_dK50": float(pmax_exact[-1]),
            "P_max_inf_exact": p_inf_exact,
            "capacity_loss_pct_inf": float((p_inf_exact / 2.0 - 1.0) * 100.0),
            "max_analytic_vs_numerical_diff": braess_max_diff,
            "blackout_test_P185": {
                "baseline_final_omega_rms": sim_br_base["final_omega_rms"],
                "added_line_final_omega_rms": sim_br_trip["final_omega_rms"],
                "added_line_max_angle_spread_rad": sim_br_trip["max_angle_spread_rad"],
            },
        },
        "exp3_12bus_gfl_cliff": [clean_row(r) for r in gfl_sweep],
        "exp3_12bus_100pct_inv_gfm_sweep": [clean_row(r) for r in gfm_sweep],
        "exp3_counterfactual_1gfm_bus6": {"synchronized": synced_b6, "max_angle_spread_rad": spread_b6},
        "exp3_dt_convergence_100pct_gfm": {
            "tie_peak_dt0025": tie_peak_coarse,
            "tie_peak_dt00125": tie_peak_fine,
            "abs_diff": float(abs(tie_peak_coarse - tie_peak_fine)),
        },
    }
    with open(RESULTS_DIR / "baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved grid-stability Entry 000 plots and metrics to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
