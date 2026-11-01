"""Page 6 – Design Optimization: minimize mass subject to structural constraints."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from suzuki_cae.optimization.design_optimizer import DesignOptimizer
from suzuki_cae.engineering.beam_theory import BeamParameters, CantileverBeam, batch_analytical

st.set_page_config(page_title="Design Optimization", page_icon="🏆", layout="wide")
st.title("🏆 Design Optimization")
st.caption("Constrained mass minimisation using surrogate model + analytical verification")

# ------------------------------------------------------------------ #
#  Sidebar – constraints                                                #
# ------------------------------------------------------------------ #
st.sidebar.header("Engineering Constraints")
sigma_allow  = st.sidebar.slider("Max allowable stress (MPa)", 50, 400, 200)
delta_allow  = st.sidebar.slider("Max allowable deflection (mm)", 1.0, 20.0, 5.0, step=0.5)
t_min        = st.sidebar.slider("Min thickness (mm)", 1.5, 4.0, 2.0, step=0.5)
force_N      = st.sidebar.slider("Design load (N)", 5, 100, 50)

st.sidebar.header("Optimisation Settings")
popsize  = st.sidebar.slider("DE population size", 5, 25, 12)
maxiter  = st.sidebar.slider("Max iterations", 50, 300, 150)

# ------------------------------------------------------------------ #
#  Baseline design                                                      #
# ------------------------------------------------------------------ #
BASELINE = dict(length_mm=120.0, width_mm=25.0, thickness_mm=4.0,
                youngs_modulus_GPa=68.9, poisson_ratio=0.33, force_N=float(force_N))
try:
    bp_base = BeamParameters(**BASELINE)
    r_base  = CantileverBeam(bp_base).solve()
except ValueError as e:
    st.error(str(e)); st.stop()

col_base1, col_base2, col_base3, col_base4 = st.columns(4)
st.subheader("Baseline Design (L=120 mm, b=25 mm, t=4 mm, Al 6061)")
col_base1.metric("Baseline mass",       f"{r_base.mass_kg*1e3:.1f} g")
col_base2.metric("Baseline deflection", f"{r_base.tip_deflection_mm:.3f} mm",
                  delta=f"limit {delta_allow:.1f} mm",
                  delta_color="inverse" if r_base.tip_deflection_mm > delta_allow else "normal")
col_base3.metric("Baseline Von Mises",  f"{r_base.max_von_mises_MPa:.1f} MPa",
                  delta=f"limit {sigma_allow} MPa",
                  delta_color="inverse" if r_base.max_von_mises_MPa > sigma_allow else "normal")
col_base4.metric("Second moment I",     f"{r_base.second_moment_area_mm4:.1f} mm⁴")

st.divider()

# ------------------------------------------------------------------ #
#  Run optimisation                                                     #
# ------------------------------------------------------------------ #
st.subheader("Run Optimisation")
run_btn = st.button("🚀 Run Differential Evolution Optimisation", type="primary")

result_placeholder = st.empty()
log_placeholder    = st.empty()

if run_btn:
    with st.spinner("Optimising… (this may take 10–30 seconds)"):
        opt = DesignOptimizer(
            allowable_stress_MPa    = float(sigma_allow),
            allowable_deflection_mm = float(delta_allow),
            min_thickness_mm        = float(t_min),
            fixed_material          = (68.9, 0.33, 2700.0),
        )
        res = opt.optimise(force_N=float(force_N), seed=42,
                            popsize=popsize, maxiter=maxiter)

    st.session_state["opt_result"]   = res
    st.session_state["opt_baseline"] = r_base

if "opt_result" in st.session_state:
    res    = st.session_state["opt_result"]
    r_base = st.session_state["opt_baseline"]

    mass_saving = (r_base.mass_kg - res.verification_mass_kg) / r_base.mass_kg * 100

    st.subheader("Optimisation Result")
    status = "✅ Converged" if res.converged else "⚠️ Did not fully converge"
    st.info(f"{status} · {res.n_evaluations} surrogate evaluations")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Optimal mass",     f"{res.verification_mass_kg*1e3:.1f} g",
               delta=f"{-mass_saving:.1f}% vs baseline", delta_color="inverse")
    c2.metric("Verified deflection", f"{res.verification_tip_deflection_mm:.3f} mm",
               delta=f"limit {delta_allow:.1f} mm",
               delta_color="inverse" if res.verification_tip_deflection_mm > delta_allow else "normal")
    c3.metric("Verified Von Mises",  f"{res.verification_von_mises_MPa:.1f} MPa",
               delta=f"limit {sigma_allow} MPa",
               delta_color="inverse" if res.verification_von_mises_MPa > sigma_allow else "normal")
    c4.metric("Mass saving vs baseline", f"{mass_saving:.1f} %")

    # Optimal dimensions
    st.subheader("Optimal Dimensions")
    opt_df = pd.DataFrame({
        "Parameter":  ["Length (mm)", "Width (mm)", "Thickness (mm)",
                        "Young's Modulus (GPa)", "Poisson's ratio", "Force (N)"],
        "Baseline":   [BASELINE["length_mm"], BASELINE["width_mm"], BASELINE["thickness_mm"],
                        BASELINE["youngs_modulus_GPa"], BASELINE["poisson_ratio"], force_N],
        "Optimal":    [round(res.length_mm,2), round(res.width_mm,2), round(res.thickness_mm,2),
                        round(res.youngs_modulus_GPa,2), round(res.poisson_ratio,4), force_N],
    })
    st.dataframe(opt_df, use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------ #
    #  Mass vs constraint visualisation                                    #
    # ------------------------------------------------------------------ #
    st.divider()
    st.subheader("Mass–Deflection Trade-off (thickness sweep)")
    t_sweep = np.linspace(max(t_min, 2.0), 6.0, 80)
    masses, deflections, stresses = [], [], []
    L_opt = res.length_mm; b_opt = res.width_mm

    for t_val in t_sweep:
        try:
            bp = BeamParameters(L_opt, b_opt, t_val, 68.9, 0.33, float(force_N))
            r  = CantileverBeam(bp).solve()
            masses.append(r.mass_kg * 1e3)
            deflections.append(r.tip_deflection_mm)
            stresses.append(r.max_von_mises_MPa)
        except Exception:
            masses.append(np.nan); deflections.append(np.nan); stresses.append(np.nan)

    fig_trade = go.Figure()
    fig_trade.add_trace(go.Scatter(x=t_sweep, y=masses, name="Mass (g)",
        line=dict(color="#1f77b4", width=2), yaxis="y1"))
    fig_trade.add_trace(go.Scatter(x=t_sweep, y=deflections, name="Deflection (mm)",
        line=dict(color="firebrick", width=2, dash="dash"), yaxis="y2"))
    fig_trade.add_vline(x=res.thickness_mm, line_dash="dot", line_color="green",
                         annotation_text=f"Optimal t={res.thickness_mm:.2f} mm")
    fig_trade.add_hline(y=delta_allow, line_dash="dot", line_color="red",
                         annotation_text=f"δ limit={delta_allow}mm", yref="y2")
    fig_trade.update_layout(
        xaxis_title="Thickness (mm)",
        yaxis=dict(title="Mass (g)", color="#1f77b4"),
        yaxis2=dict(title="Deflection (mm)", color="firebrick",
                    overlaying="y", side="right"),
        height=360, margin=dict(l=0, r=60, t=10, b=0),
        legend=dict(x=0.01, y=0.99)
    )
    st.plotly_chart(fig_trade, use_container_width=True)

    # ------------------------------------------------------------------ #
    #  Surrogate vs verification comparison                                #
    # ------------------------------------------------------------------ #
    st.divider()
    st.subheader("Surrogate vs FEA Verification")
    verify_df = pd.DataFrame({
        "Metric":        ["Mass (kg)", "Tip Deflection (mm)", "Von Mises (MPa)"],
        "Surrogate":     [round(res.surrogate_mass_kg, 6),
                          round(res.tip_deflection_mm, 4),
                          round(res.max_von_mises_MPa, 2)],
        "Verified (FEA)":[round(res.verification_mass_kg, 6),
                          round(res.verification_tip_deflection_mm, 4),
                          round(res.verification_von_mises_MPa, 2)],
    })
    verify_df["|Error %|"] = abs(
        (verify_df["Surrogate"] - verify_df["Verified (FEA)"]) /
        verify_df["Verified (FEA)"] * 100
    ).round(3)
    st.dataframe(verify_df, use_container_width=True, hide_index=True)
    st.caption(
        "Verification uses the analytical solver (equivalent to CalculiX in production). "
        "Small discrepancies between surrogate and FEA are expected due to model error."
    )