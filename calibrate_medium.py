import numpy as np
from scipy.optimize import minimize

from constants import (
    M_E_MEV,
    M_MU_MEV,
    R_E_COMPTON_FM,
    R_MU_COMPTON_FM,
    R_CORE_VAC_FM,
)
from preprocessor import build_3d_grid, init_hopf_vortex_3d
from solution import relax_vortex_3d
from postprocessor import analyze_relaxed_vortex_3d


def calibrate_medium_3d(
    base_medium,
    e_topo=(2, 3),
    mu_topo=(37, 73),
    N_grid=32,
    relax_steps=120,
    max_iter=40,
):
    """
    Калибрует параметры среды (alpha_nl, gamma_nl, beta_nl, eta_visc, delta_pack)
    под одновременное попадание 3D-узлов в массы и радиусы электрона и мюона.
    """
    grid_e = build_3d_grid(R_E_COMPTON_FM, N=N_grid, box_scale=3.2)
    grid_mu = build_3d_grid(R_MU_COMPTON_FM, N=N_grid, box_scale=3.2)

    psi0_e = init_hopf_vortex_3d(
        grid_e, R_E_COMPTON_FM, R_E_COMPTON_FM * 0.30, e_topo[0], e_topo[1]
    )
    psi0_mu = init_hopf_vortex_3d(
        grid_mu, R_MU_COMPTON_FM, R_MU_COMPTON_FM * 0.30, mu_topo[0], mu_topo[1]
    )

    keys = ["alpha_nl", "gamma_nl", "beta_nl", "eta_visc", "delta_pack"]
    x0 = np.log10([max(float(base_medium[k]), 1e-8) for k in keys])

    # Масштабный коэффициент приведения безразмерного 3D-интеграла к МэВ
    vol_scale_e = (R_E_COMPTON_FM / R_CORE_VAC_FM) ** 2
    vol_scale_mu = (R_MU_COMPTON_FM / R_CORE_VAC_FM) ** 2

    best_state = {"loss": float("inf"), "medium": dict(base_medium), "obs_e": None, "obs_mu": None}

    def loss_fn(log_params):
        vals = 10.0 ** np.clip(log_params, -7.0, 3.0)
        med = dict(base_medium)
        for k, v in zip(keys, vals):
            med[k] = float(v)

        res_e = relax_vortex_3d(psi0_e, grid_e, med, max_steps=relax_steps, norm_interval=1)
        res_mu = relax_vortex_3d(psi0_mu, grid_mu, med, max_steps=relax_steps, norm_interval=1)

        obs_e = analyze_relaxed_vortex_3d(res_e["psi"], grid_e, med, e_topo[0], e_topo[1])
        obs_mu = analyze_relaxed_vortex_3d(res_mu["psi"], grid_mu, med, mu_topo[0], mu_topo[1])

        m_e_est = obs_e["mass_mev"] / max(vol_scale_e, 1.0)
        m_mu_est = obs_mu["mass_mev"] / max(vol_scale_mu, 1e-3)

        if not np.isfinite(m_e_est) or not np.isfinite(m_mu_est) or m_e_est <= 0 or m_mu_est <= 0:
            return 1e6

        err_me = (np.log10(m_e_est / M_E_MEV)) ** 2
        err_mmu = (np.log10(m_mu_est / M_MU_MEV)) ** 2
        err_re = (np.log10(max(obs_e["R0_fm"], 1e-4) / R_E_COMPTON_FM)) ** 2
        err_rmu = (np.log10(max(obs_mu["R0_fm"], 1e-4) / R_MU_COMPTON_FM)) ** 2

        total_loss = float(err_me + err_mmu + 0.5 * (err_re + err_rmu))
        if total_loss < best_state["loss"]:
            obs_e_copy = dict(obs_e)
            obs_mu_copy = dict(obs_mu)
            obs_e_copy["mass_mev"] = float(m_e_est)
            obs_mu_copy["mass_mev"] = float(m_mu_est)
            best_state.update(
                {"loss": total_loss, "medium": med, "obs_e": obs_e_copy, "obs_mu": obs_mu_copy}
            )
        return total_loss

    minimize(loss_fn, x0, method="Nelder-Mead", options={"maxiter": int(max_iter), "xatol": 1e-3, "fatol": 1e-4})
    return best_state
