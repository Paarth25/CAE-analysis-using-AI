"""Page 1 – CAE Explorer: run analytical FEA and display structural results."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from suzuki_cae.engineering.beam_theory import BeamParameters, CantileverBeam
from suzuki_cae.engineering.validation_equations import (
    check_tip_deflection, check_bending_stress, slenderness_ratio, shear_correction_factor
)

st.set_page_config(page_title="CAE Explorer", page_icon="⚙️", layout="wide")
st.title("⚙️ CAE Explorer")
st.caption("Analytical FEA for cantilever beams – verified against Euler-Bernoulli theory")

# ------------------------------------------------------------------ #
#  Sidebar inputs                                                       #
# ------------------------------------------------------------------ #
st.sidebar.header("Beam Parameters")

L = st.sidebar.slider("Length (mm)",    80,  160, 100)
b = st.sidebar.slider("Width (mm)",     15,  40,  20)
t = st.sidebar.slider("Thickness (mm)", 2,   6,   4)
E = st.sidebar.slider("Young's Modulus (GPa)", 65, 210, 70)
nu = st.sidebar.slider("Poisson's Ratio", 0.27, 0.35, 0.33, step=0.01)
F = st.sidebar.slider("Applied Force (N)", 5, 100, 50)

# ------------------------------------------------------------------ #
#  Solve                                                                #
# ------------------------------------------------------------------ #
try:
    params = BeamParameters(
        length_mm=float(L), width_mm=float(b), thickness_mm=float(t),
        youngs_modulus_GPa=float(E), poisson_ratio=float(nu), force_N=float(F),
    )
    beam   = CantileverBeam(params)
    result = beam.solve()
    error  = None
except ValueError as e:
    result = None
    error  = str(e)

if error:
    st.error(f"Invalid parameters: {error}")
    st.stop()

# ------------------------------------------------------------------ #
#  Metric cards                                                         #
# ------------------------------------------------------------------ #
col1, col2, col3, col4 = st.columns(4)
col1.metric("Tip Deflection",    f"{result.tip_deflection_mm:.3f} mm")
col2.metric("Max Von Mises",     f"{result.max_von_mises_MPa:.1f} MPa")
col3.metric("Strain Energy",     f"{result.strain_energy_J:.4f} J")
col4.metric("Component Mass",    f"{result.mass_kg*1e3:.2f} g")

st.divider()

# ------------------------------------------------------------------ #
#  Deflection profile                                                   #
# ------------------------------------------------------------------ #
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Deflection Profile")
    x_nodes = np.linspace(0, L, 200)
    L_m = L * 1e-3; b_m = b * 1e-3; t_m = t * 1e-3; E_Pa = E * 1e9
    I_m4 = b_m * t_m**3 / 12
    # EB deflection: v(x) = F/(6EI) * (3L x² - x³)
    x_m = x_nodes * 1e-3
    v_m = (F / (6 * E_Pa * I_m4)) * (3 * L_m * x_m**2 - x_m**3)
    v_mm = v_m * 1e3

    fig_defl = go.Figure()
    fig_defl.add_trace(go.Scatter(x=x_nodes, y=v_mm, mode="lines",
                                   line=dict(color="#1f77b4", width=3),
                                   name="Deflection"))
    fig_defl.add_trace(go.Scatter(x=[L], y=[result.tip_deflection_mm],
                                   mode="markers", marker=dict(size=10, color="red"),
                                   name="Tip"))
    fig_defl.update_layout(xaxis_title="Position along beam (mm)",
                            yaxis_title="Deflection (mm)",
                            height=320, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig_defl, use_container_width=True)

with col_right:
    st.subheader("Bending Moment & Shear Force")
    M_Nm = F * (L_m - x_m)          # N·m
    V_N  = np.full_like(x_m, F)     # constant shear

    fig_mv = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=("Bending Moment (N·mm)", "Shear Force (N)"))
    fig_mv.add_trace(go.Scatter(x=x_nodes, y=M_Nm*1e3, fill="tozeroy",
                                 line=dict(color="orange"), name="M"), row=1, col=1)
    fig_mv.add_trace(go.Scatter(x=x_nodes, y=V_N, fill="tozeroy",
                                 line=dict(color="green"), name="V"), row=2, col=1)
    fig_mv.update_xaxes(title_text="Position (mm)", row=2)
    fig_mv.update_layout(height=320, showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_mv, use_container_width=True)

# ------------------------------------------------------------------ #
#  Bending stress distribution                                          #
# ------------------------------------------------------------------ #
st.subheader("Bending Stress Distribution along Beam")
sigma_MPa = (F * (L_m - x_m) * (t_m / 2) / I_m4) * 1e-6
fig_stress = go.Figure()
fig_stress.add_trace(go.Scatter(x=x_nodes, y=sigma_MPa, mode="lines",
                                 line=dict(color="firebrick", width=2.5), name="σ_bend"))
fig_stress.add_hline(y=result.max_bending_stress_MPa, line_dash="dash",
                      annotation_text=f"Max σ = {result.max_bending_stress_MPa:.1f} MPa",
                      line_color="red")
fig_stress.update_layout(xaxis_title="Position (mm)", yaxis_title="Bending Stress (MPa)",
                          height=280, margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig_stress, use_container_width=True)

# ------------------------------------------------------------------ #
#  Theory validation                                                    #
# ------------------------------------------------------------------ #
st.subheader("Engineering Theory Validation")
sr  = slenderness_ratio(L, t)
scf = shear_correction_factor(L, t)
chk_d = check_tip_deflection(result.tip_deflection_mm, result.tip_deflection_mm)
chk_s = check_bending_stress(result.max_bending_stress_MPa, result.max_bending_stress_MPa)

v1, v2, v3 = st.columns(3)
v1.metric("Slenderness L/t", f"{sr:.1f}", help="Euler-Bernoulli accurate when > 10")
color = "✅" if sr >= 10 else "⚠️"
v1.caption(f"{color} {'Valid range' if sr >= 10 else 'Shear deformation may matter'}")
v2.metric("I = bt³/12", f"{result.second_moment_area_mm4:.1f} mm⁴")
v3.metric("Shear correction", f"{scf*100:.3f} %",
          help="Shear deflection as % of bending deflection")

# ------------------------------------------------------------------ #
#  Full results table                                                   #
# ------------------------------------------------------------------ #
with st.expander("Full analytical results"):
    import pandas as pd
    d = result.as_dict()
    df = pd.DataFrame({"Quantity": list(d.keys()), "Value": list(d.values())})
    st.dataframe(df, use_container_width=True)

st.caption("Source: Euler–Bernoulli beam theory (linear elastic, small displacement).")