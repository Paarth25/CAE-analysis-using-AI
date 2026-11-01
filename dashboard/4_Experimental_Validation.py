"""Page 4 – Experimental Validation: compare CAE, ML and experimental measurements."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

import pickle
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression as SkLinReg

from suzuki_cae.engineering.beam_theory import batch_analytical

st.set_page_config(page_title="Experimental Validation", page_icon="🧪", layout="wide")
st.title("🧪 Experimental Validation")

ROOT       = Path(__file__).parents[2]
EXP_PATH   = ROOT / "data" / "raw" / "experimental" / "synthetic_experimental.csv"
MODELS_DIR = ROOT / "models" / "saved"

st.warning(
    "⚠️ **Data provenance notice**: The experimental data shown here is "
    "**SYNTHETIC** – generated from the analytical model plus realistic "
    "measurement noise. It is NOT from physical experiments. "
    "Replace `data/raw/experimental/synthetic_experimental.csv` with real "
    "measurements to perform genuine validation."
)

# ------------------------------------------------------------------ #
#  Data loading                                                         #
# ------------------------------------------------------------------ #
st.subheader("Experimental Dataset")
uploaded = st.file_uploader("Upload experimental CSV (or use default synthetic data)",
                              type="csv")
if uploaded:
    df_exp = pd.read_csv(uploaded)
    st.success(f"Loaded {len(df_exp)} rows from uploaded file.")
elif EXP_PATH.exists():
    df_exp = pd.read_csv(EXP_PATH)
    st.info(f"Using synthetic experimental dataset ({len(df_exp)} records).")
else:
    st.error("No experimental data found. Run `generate_dataset.py` first.")
    st.stop()

with st.expander("Raw experimental data"):
    st.dataframe(df_exp, use_container_width=True)

# ------------------------------------------------------------------ #
#  Compute analytical and compute residuals                             #
# ------------------------------------------------------------------ #
# Analytical predictions for experimental specimens
ana_out = batch_analytical(
    df_exp["length_mm"].values,
    df_exp["width_mm"].values,
    df_exp["thickness_mm"].values,
    df_exp["youngs_modulus_GPa"].values,
    df_exp["poisson_ratio"].values,
    df_exp["force_N"].values,
)
df_exp["analytical_mm"]  = ana_out["tip_deflection_mm"]
df_exp["residual_mm"]    = df_exp["tip_deflection_mm"] - df_exp["analytical_mm"]
df_exp["residual_pct"]   = df_exp["residual_mm"] / df_exp["analytical_mm"] * 100

# ------------------------------------------------------------------ #
#  Summary metrics                                                      #
# ------------------------------------------------------------------ #
st.divider()
st.subheader("Validation Summary")

mean_err = df_exp["residual_mm"].abs().mean()
mean_pct = df_exp["residual_pct"].abs().mean()
bias     = df_exp["residual_mm"].mean()
std_res  = df_exp["residual_mm"].std()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Mean Abs Error", f"{mean_err:.4f} mm")
c2.metric("Mean Abs Error %", f"{mean_pct:.2f} %")
c3.metric("Bias (systematic)", f"{bias:+.4f} mm")
c4.metric("Residual Std Dev", f"{std_res:.4f} mm")

# ------------------------------------------------------------------ #
#  Measured vs Analytical scatter                                       #
# ------------------------------------------------------------------ #
st.divider()
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Measured vs Analytical Tip Deflection")
    fig = go.Figure()
    # Group by specimen
    for spec in df_exp["specimen_id"].unique():
        sub = df_exp[df_exp["specimen_id"] == spec]
        fig.add_trace(go.Scatter(
            x=sub["analytical_mm"], y=sub["tip_deflection_mm"],
            mode="markers", name=spec,
            error_y=dict(type="constant", value=sub["measurement_uncertainty_mm"].mean(),
                         visible=True, color="lightgray"),
            marker=dict(size=7)))
    lim = [df_exp["analytical_mm"].min() * 0.9, df_exp["analytical_mm"].max() * 1.1]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines",
        line=dict(color="black", dash="dash", width=1), name="Perfect agreement", showlegend=True))
    fig.update_layout(xaxis_title="Analytical (mm)", yaxis_title="Measured (mm)",
                       height=380, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.subheader("Residuals vs Force")
    fig2 = go.Figure()
    for spec in df_exp["specimen_id"].unique():
        sub = df_exp[df_exp["specimen_id"] == spec]
        fig2.add_trace(go.Scatter(
            x=sub["force_N"], y=sub["residual_mm"],
            mode="markers+lines", name=spec, marker=dict(size=6)))
    fig2.add_hline(y=0, line_dash="dash", line_color="black")
    fig2.add_hline(y=bias, line_dash="dot", line_color="orange",
                    annotation_text=f"Mean bias={bias:+.4f}mm")
    fig2.update_layout(xaxis_title="Force (N)", yaxis_title="Residual (mm)",
                        height=380, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig2, use_container_width=True)

# ------------------------------------------------------------------ #
#  Residual correction model                                            #
# ------------------------------------------------------------------ #
st.divider()
st.subheader("Residual Correction Model")
st.markdown("""
A simple linear correction is fitted: **Calibrated = Analytical + Correction(F, analytical)**

This accounts for systematic bias in the analytical model relative to the measurements
(e.g. fixture compliance, load application eccentricity, material property uncertainty).
""")

# Fit correction on 80% calibration, evaluate on 20% test
cal_mask = np.random.default_rng(0).random(len(df_exp)) < 0.8
df_cal  = df_exp[cal_mask]
df_test = df_exp[~cal_mask]

X_cal = df_cal[["force_N", "analytical_mm"]].values
y_cal = df_cal["residual_mm"].values
X_tst = df_test[["force_N", "analytical_mm"]].values
y_tst = df_test["residual_mm"].values

corr_model = SkLinReg().fit(X_cal, y_cal)
res_pred_cal  = corr_model.predict(X_cal)
res_pred_test = corr_model.predict(X_tst)

calib_test = df_test["analytical_mm"].values + res_pred_test
raw_err    = np.abs(y_tst).mean()
calib_err  = np.abs(df_test["tip_deflection_mm"].values - calib_test).mean()

c1, c2, c3 = st.columns(3)
c1.metric("Uncalibrated MAE (test)", f"{raw_err:.4f} mm")
c2.metric("Calibrated MAE (test)",   f"{calib_err:.4f} mm")
c3.metric("Correction coeff (F)",    f"{corr_model.coef_[0]:.6f}")

# ------------------------------------------------------------------ #
#  4-way comparison                                                     #
# ------------------------------------------------------------------ #
st.divider()
st.subheader("Four-Way Comparison by Force Level")
grouped = df_exp.groupby("force_N").agg(
    measured_mean=("tip_deflection_mm", "mean"),
    measured_std=("tip_deflection_mm", "std"),
    analytical_mean=("analytical_mm", "mean"),
).reset_index()

fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=grouped["force_N"], y=grouped["measured_mean"],
    mode="lines+markers", name="Measured (mean±std)",
    error_y=dict(array=grouped["measured_std"].values, visible=True),
    line=dict(color="#1f77b4"), marker=dict(size=8)))
fig4.add_trace(go.Scatter(x=grouped["force_N"], y=grouped["analytical_mean"],
    mode="lines+markers", name="Analytical", line=dict(color="orange", dash="dash"),
    marker=dict(size=8)))
fig4.update_layout(xaxis_title="Force (N)", yaxis_title="Tip Deflection (mm)",
                    height=340, margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig4, use_container_width=True)

st.caption(
    "Calibration uses 80% of experimental records; correction model is evaluated "
    "on the remaining 20% (independent test). "
    "Bias correction intercept: {:.4f} mm.".format(corr_model.intercept_)
)