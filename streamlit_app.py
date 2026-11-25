import streamlit as st
import numpy as np
import plotly.graph_objects as go

from constants import DEFAULT_MEDIUM, DEFAULT_GRID, M_E_MEV, M_MU_MEV, R_E_COMPTON_FM, R_MU_COMPTON_FM
from preprocessor import build_3d_grid, init_hopf_vortex_3d, init_noise_field_3d
from solution import relax_vortex_3d
from postprocessor import analyze_relaxed_vortex_3d

st.set_page_config(page_title="Muon 3D Vortex Solver", layout="wide")
st.title("3D Soliton Relaxation Solver (dt = τ / 2)")

with st.sidebar:
    st.header("1. Топология и начальный масштаб")
    preset = st.selectbox("Пресет частицы", ["Электрон (2, 3)", "Мюон (37, 73)", "Мюон (73, 37)", "Свой узел"])
    if preset == "Электрон (2, 3)":
        def_m, def_n, def_r, def_s = 2, 3, float(R_E_COMPTON_FM), float(R_E_COMPTON_FM * 0.35)
    elif preset == "Мюон (37, 73)":
        def_m, def_n, def_r, def_s = 37, 73, float(R_MU_COMPTON_FM), float(R_MU_COMPTON_FM * 0.35)
    elif preset == "Мюон (73, 37)":
        def_m, def_n, def_r, def_s = 73, 37, float(R_MU_COMPTON_FM), float(R_MU_COMPTON_FM * 0.35)
    else:
        def_m, def_n, def_r, def_s = 2, 3, 100.0, 35.0

    col_m, col_n = st.columns(2)
    m_tor = col_m.number_input("m_tor", min_value=1, max_value=500, value=def_m, step=1)
    n_pol = col_n.number_input("n_pol", min_value=1, max_value=500, value=def_n, step=1)
    R0_init = st.number_input("Начальный R0 (fm)", min_value=0.1, max_value=10000.0, value=def_r, format="%.3f")
    sig_init = st.number_input("Начальная толщина σ (fm)", min_value=0.05, max_value=5000.0, value=def_s, format="%.3f")

    st.header("2. Среда и шаг dt = τ / 2")
    alpha_nl = st.number_input("alpha_nl", value=float(DEFAULT_MEDIUM["alpha_nl"]), format="%.6g")
    gamma_nl = st.number_input("gamma_nl", value=float(DEFAULT_MEDIUM["gamma_nl"]), format="%.6g")
    beta_nl = st.number_input("beta_nl", value=float(DEFAULT_MEDIUM["beta_nl"]), format="%.6g")
    eta_visc = st.number_input("eta_visc", value=float(DEFAULT_MEDIUM["eta_visc"]), format="%.6g")
    delta_pack = st.number_input("delta_pack", value=float(DEFAULT_MEDIUM["delta_pack"]), format="%.6g")
    tau = st.number_input("tau (шаг dt = tau / 2)", min_value=0.01, max_value=10.0, value=float(DEFAULT_MEDIUM["tau"]), format="%.3f")

    st.header("3. 3D-сетка (open BC)")
    N_grid = st.slider("Размер куба N³", min_value=24, max_value=80, value=int(DEFAULT_GRID["N"]), step=8)
    box_scale = st.slider("Размер бокса (× R0)", min_value=2.5, max_value=6.0, value=float(DEFAULT_GRID["box_scale"]), step=0.5)
    max_steps = st.slider("Макс. шагов релаксации", min_value=100, max_value=2000, value=int(DEFAULT_GRID["max_steps"]), step=100)

medium = {
    "alpha_nl": alpha_nl,
    "gamma_nl": gamma_nl,
    "beta_nl": beta_nl,
    "eta_visc": eta_visc,
    "delta_pack": delta_pack,
    "tau": tau,
}

col_btn1, col_btn2 = st.columns(2)
run_vortex = col_btn1.button("Запустить 3D-релаксацию узла", type="primary", use_container_width=True)
run_noise = col_btn2.button("Тест №1: Устойчивость шума (ψ ≈ 0)", use_container_width=True)
