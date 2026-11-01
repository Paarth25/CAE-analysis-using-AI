"""Page 5 – Uncertainty Analysis: prediction intervals and high-uncertainty detection."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

from suzuki_cae.data.preprocessing import DataPipeline, TARGET_COLUMNS
from suzuki_cae.data.generation import latin_hypercube_sample, PARAM_BOUNDS
from suzuki_cae.engineering.beam_theory import batch_analytical
from suzuki_cae.uncertainty.ensemble_uncertainty import EnsembleUQ

st.set_page_config(page_title="Uncertainty Analysis", page_icon="📐", layout="wide")
st.title("📐 Uncertainty Analysis")
st.caption("Bootstrap ensemble prediction intervals · High-uncertainty design detection")

ROOT      = Path(__file__).parents[2]
DATA_PATH = ROOT / "data" / "raw" / "cae" / "beam_dataset_analytical.csv"

TARGETS = {
    "tip_deflection_mm": ("Tip Deflection", "mm"),
    "max_von_mises_MPa": ("Max Von Mises Stress", "MPa"),
    "strain_energy_J":   ("Strain Energy", "J"),
}


@st.cache_resource
def build_uq_model(target: str, n_ensemble: int = 30):
    """Train bootstrap ensemble on training split."""
    if not DATA_PATH.exists():
        return None, None, None, None
    pipe = DataPipeline(test_size=0.15, val_size=0.15, seed=42)
    df   = pipe.load(DATA_PATH)
    df   = pipe.clean(df)
    df   = pipe.engineer_features(df)
    train, val, test = pipe.split(df)
    pipe.fit_scalers(train)

    feat_cols = [c for c in pipe.scaler_X.feature_names_in_ if c in train.columns]
    X_tr  = train[feat_cols].values
    y_tr  = train[target].values
    X_cal = val[feat_cols].values
    y_cal = val[target].values
    X_te  = test[feat_cols].values
    y_te  = test[target].values

    uq = EnsembleUQ(
        base_estimator=RandomForestRegressor(n_estimators=50, random_state=42),
        n_estimators=n_ensemble, alpha=0.05, seed=42
    )
    uq.fit(X_tr, y_tr)
    uq.calibrate(X_cal, y_cal)

    return uq, X_te, y_te, test


# ------------------------------------------------------------------ #
#  Controls                                                             #
# ------------------------------------------------------------------ #
target = st.selectbox("Target variable", list(TARGETS.keys()),
                       format_func=lambda x: TARGETS[x][0])
label, unit = TARGETS[target]
n_ensemble = st.sidebar.slider("Ensemble size", 10, 60, 30)

with st.spinner("Training ensemble (cached after first run)…"):
    uq, X_te, y_te, test_df = build_uq_model(target, n_ensemble)

if uq is None:
    st.warning("Training data not found. Run `generate_dataset.py` first.")
    st.stop()

# ------------------------------------------------------------------ #
#  Test-set prediction intervals                                        #
# ------------------------------------------------------------------ #
pi = uq.predict_interval(X_te)
coverage = float(np.mean((y_te >= pi.lower) & (y_te <= pi.upper)))
mean_width = float((pi.upper - pi.lower).mean())
flagged_n  = int(pi.flagged.sum())

st.divider()
st.subheader("Test-Set Prediction Intervals (95%)")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Empirical Coverage", f"{coverage*100:.1f} %", delta=f"{(coverage-0.95)*100:+.1f}% vs 95%")
c2.metric("Mean Interval Width", f"{mean_width:.4f} {unit}")
c3.metric("Flagged (high uncertainty)", str(flagged_n))
c4.metric("Calibration factor", f"{uq._calibration_factor:.3f}")

# Sort by actual value for a clean waterfall view
sort_idx = np.argsort(y_te)
y_sorted   = y_te[sort_idx]
mean_sorted = pi.mean[sort_idx]
lo_sorted   = pi.lower[sort_idx]
hi_sorted   = pi.upper[sort_idx]
flag_sorted = pi.flagged[sort_idx]

fig_pi = go.Figure()
fig_pi.add_trace(go.Scatter(
    x=list(range(len(y_sorted))),
    y=hi_sorted, mode="lines", line=dict(width=0), showlegend=False))
fig_pi.add_trace(go.Scatter(
    x=list(range(len(y_sorted))),
    y=lo_sorted, fill="tonexty", mode="lines", line=dict(width=0),
    fillcolor="rgba(31,119,180,0.20)", name="95% interval"))
fig_pi.add_trace(go.Scatter(
    x=list(range(len(y_sorted))), y=mean_sorted, mode="lines",
    line=dict(color="#1f77b4", width=1.5), name="Ensemble mean"))
fig_pi.add_trace(go.Scatter(
    x=list(range(len(y_sorted))), y=y_sorted, mode="markers",
    marker=dict(color=["red" if f else "black" for f in flag_sorted],
                size=4, opacity=0.7), name="Actual"))
fig_pi.update_layout(
    xaxis_title="Test sample (sorted by actual value)",
    yaxis_title=f"{label} ({unit})",
    height=380, margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig_pi, use_container_width=True)
st.caption("Red dots = samples flagged as high-uncertainty. Shaded band = 95% prediction interval.")

# ------------------------------------------------------------------ #
#  Uncertainty map over design space                                    #
# ------------------------------------------------------------------ #
st.divider()
st.subheader("Uncertainty Map – Length vs Force")
st.caption("Ensemble std across a 2D grid; other parameters at their midpoint values.")

n_grid = 30
L_vals = np.linspace(80, 160, n_grid)
F_vals = np.linspace(5, 100, n_grid)
LL, FF = np.meshgrid(L_vals, F_vals)

# Fixed midpoints for remaining parameters
b_mid  = (PARAM_BOUNDS["width_mm"][0]          + PARAM_BOUNDS["width_mm"][1]) / 2
t_mid  = (PARAM_BOUNDS["thickness_mm"][0]       + PARAM_BOUNDS["thickness_mm"][1]) / 2
E_mid  = (PARAM_BOUNDS["youngs_modulus_GPa"][0] + PARAM_BOUNDS["youngs_modulus_GPa"][1]) / 2
nu_mid = (PARAM_BOUNDS["poisson_ratio"][0]      + PARAM_BOUNDS["poisson_ratio"][1]) / 2

L_flat = LL.ravel(); F_flat = FF.ravel()
n_pts  = len(L_flat)
I_flat = b_mid * t_mid**3 / 12.0

grid_df = pd.DataFrame({
    "length_mm":          L_flat,
    "width_mm":           np.full(n_pts, b_mid),
    "thickness_mm":       np.full(n_pts, t_mid),
    "youngs_modulus_GPa": np.full(n_pts, E_mid),
    "poisson_ratio":      np.full(n_pts, nu_mid),
    "force_N":            F_flat,
    "I_mm4":              np.full(n_pts, I_flat),
    "EI":                 np.full(n_pts, E_mid * I_flat),
    "slenderness":        L_flat / t_mid,
    "FL3_EI":             F_flat * L_flat**3 / (E_mid * 1e3 * I_flat),
    "area_mm2":           np.full(n_pts, b_mid * t_mid),
    "aspect_ratio":       L_flat / b_mid,
})

feat_cols = [c for c in uq._ensemble[0].feature_names_in_ if c in grid_df.columns] \
    if hasattr(uq._ensemble[0], "feature_names_in_") else \
    ["length_mm","width_mm","thickness_mm","youngs_modulus_GPa",
     "poisson_ratio","force_N","I_mm4","EI","slenderness","FL3_EI","area_mm2","aspect_ratio"]

X_grid = grid_df[[c for c in feat_cols if c in grid_df.columns]].values
pi_grid = uq.predict_interval(X_grid)
std_map = pi_grid.std.reshape(n_grid, n_grid)

fig_hm = go.Figure(go.Heatmap(
    z=std_map, x=L_vals, y=F_vals,
    colorscale="YlOrRd", colorbar=dict(title=f"Std ({unit})")
))
fig_hm.update_layout(
    xaxis_title="Length (mm)", yaxis_title="Force (N)",
    height=380, margin=dict(l=0, r=0, t=10, b=0)
)
st.plotly_chart(fig_hm, use_container_width=True)
st.caption("High-uncertainty zones indicate where additional simulations would most improve the surrogate.")

# ------------------------------------------------------------------ #
#  Active learning recommendation                                       #
# ------------------------------------------------------------------ #
st.divider()
st.subheader("Active Learning Recommendations")
n_recommend = st.slider("Number of recommended new simulations", 3, 20, 5)

# Sample a large candidate pool and select highest-uncertainty points
cand_doe = latin_hypercube_sample(2000, PARAM_BOUNDS, seed=77)
I_cand = cand_doe["width_mm"] * cand_doe["thickness_mm"]**3 / 12.0
cand_doe["I_mm4"]      = I_cand
cand_doe["EI"]         = cand_doe["youngs_modulus_GPa"] * I_cand
cand_doe["slenderness"]= cand_doe["length_mm"] / cand_doe["thickness_mm"]
cand_doe["FL3_EI"]     = (cand_doe["force_N"] * cand_doe["length_mm"]**3
                           / (cand_doe["youngs_modulus_GPa"] * 1e3 * I_cand))
cand_doe["area_mm2"]   = cand_doe["width_mm"] * cand_doe["thickness_mm"]
cand_doe["aspect_ratio"]= cand_doe["length_mm"] / cand_doe["width_mm"]

X_cand = cand_doe[[c for c in feat_cols if c in cand_doe.columns]].values
pi_cand = uq.predict_interval(X_cand)

top_idx = np.argsort(pi_cand.std)[-n_recommend:][::-1]
recommended = cand_doe.iloc[top_idx][
    ["length_mm","width_mm","thickness_mm","youngs_modulus_GPa","poisson_ratio","force_N"]
].copy()
recommended.insert(0, "priority", range(1, n_recommend + 1))
recommended["pred_uncertainty"] = pi_cand.std[top_idx]
recommended = recommended.round(3)

st.dataframe(recommended, use_container_width=True, hide_index=True)
st.caption(
    f"Top {n_recommend} candidate designs where ensemble uncertainty is highest. "
    "Running CAE simulations at these points will most efficiently improve the surrogate."
)