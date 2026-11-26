import numpy as np
from scipy.optimize import minimize

from constants import (
    M_E_MEV,
    R_E_COMPTON_FM,
    R_CORE_VAC_FM,
)
from preprocessor import build_3d_grid, init_hopf_vortex_3d
from solution import relax_vortex_3d
from postprocessor import analyze_relaxed_vortex_3d

DEFAULT_TOPO_SEQUENCE = [
    (1, 2),
    (2, 3),
    (3, 4),
    (2, 5),
    (3, 5),
    (5, 8),
    (37, 73),
]
def evaluate_single_vortex_dynamics(
    m_tor,
    n_pol,
    medium,
    R0_init=R_E_COMPTON_FM,
    N_grid=32,
    dyn_steps=160,
):
    """
    Запускает 3D-динамику для одного узла (m_tor, n_pol) в заданной среде
    и проверяет прохождение buckling и динамическую устойчивость.
    """
    grid = build_3d_grid(R0_init, N=N_grid, box_scale=3.2)
    psi0 = init_hopf_vortex_3d(grid, R0_init, R0_init * 0.30, int(m_tor), int(n_pol))
    res = relax_vortex_3d(
        psi0,
        grid,
        medium,
        max_steps=int(dyn_steps),
        renormalize=False,
        damping_coeff=0.02,
    )
    obs = analyze_relaxed_vortex_3d(res["psi"], grid, medium, int(m_tor), int(n_pol))

    vol_scale = max((float(R0_init) / R_CORE_VAC_FM) ** 2, 1.0)
    mass_scaled = float(obs["mass_mev"]) / vol_scale
    survived = bool(
        res.get("dynamically_stable", False)
        and obs.get("buckling_ok", False)
        and obs["topology"]["topology_preserved"]
    )

    return {
        "topo": (int(m_tor), int(n_pol)),
        "survived": survived,
        "buckling_ok": bool(obs.get("buckling_ok", False)),
        "dynamically_stable": bool(res.get("dynamically_stable", False)),
        "topology_preserved": bool(obs["topology"]["topology_preserved"]),
        "chi_max": float(obs.get("chi_max", 0.0)),
        "mass_mev": mass_scaled,
        "R0_fm": float(obs["R0_fm"]),
        "g_factor": float(obs["g_factor"]),
        "steps_done": int(res["steps_done"]),
    }
