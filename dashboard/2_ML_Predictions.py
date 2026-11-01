"""Page 2 – ML Predictions: surrogate model predictions with uncertainty."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

import pickle
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from suzuki_cae.engineering.beam_theory import BeamParameters, CantileverBeam
from suzuki_cae.data.preprocessing import DataPipeline, FEATURE_COLUMNS
from suzuki_cae.data.generation import DatasetGenerator, PARAM_BOUNDS

st.set_page_config(page_title="ML Predictions", page_icon="🤖", layout="wide")
st.title("🤖 ML Predictions")
st.caption("Surrogate model predictions with uncertainty quantification")

ROOT = Path(__file__).parents[2]
MODELS_DIR = ROOT / "models" / "saved"
DATA_PATH  = ROOT / "data" / "raw" / "cae" / "beam_dataset_analytical.csv"

TARGETS = {
    "tip_deflection_mm":  ("Tip Deflection", "mm"),
    "max_von_mises_MPa":  ("Max Von Mises Stress", "MPa"),
    "strain_energy_J":    ("Strain Energy", "J"),
}

MODEL_NAMES = ["LinearRegression", "PolynomialRegression", "RandomForest", "XGBoost", "MLP"]


# ------------------------------------------------------------------ #
#  Load helpers                                                         #
# ------------------------------------------------------------------ #
@st.cache_resource
def load_model(target: str, name: str):
    path = MODELS_DIR / f"{target}_{name}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_training_data():
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)


def check_domain(params: dict, df_train: pd.DataFrame) -> dict[str, str]:
    """Flag parameters that are outside the training data range."""
    flags = {}
    for k, v in params.items():
        col = k
        if col not in df_train.columns:
            continue
        lo, hi = df_train[col].min(), df_train[col].max()
        if v < lo or v > hi:
            flags[k] = f"⚠️ {v:.2f} outside training range [{lo:.2f}, {hi:.2f}]"
    return flags


def engineer_features_single(params: dict, pipe_cols: list[str]) -> np.ndarray:
    """Build engineered feature vector for a single design point."""
    L, b, t = params["length_mm"], params["width_mm"], params["thickness_mm"]
    E, nu, F = params["youngs_modulus_GPa"], params["poisson_ratio"], params["force_N"]
    I = b * t**3 / 12.0
    EI = E * I
    row = {
        "length_mm": L, "width_mm": b, "thickness_mm": t,
        "youngs_modulus_GPa": E, "poisson_ratio": nu, "force_N": F,
        "I_mm4": I, "EI": EI, "slenderness": L / t,
        "FL3_EI": F * L**3 / (E * 1e3 * I),
        "area_mm2": b * t, "aspect_ratio": L / b,
    }
    return np.array([[row[c] for c in pipe_cols if c in row]])


# ------------------------------------------------------------------ #
#  Sidebar – inputs                                                     #
# ------------------------------------------------------------------ #
st.sidebar.header("Design Parameters")
L  = st.sidebar.slider("Length (mm)",          80,  160, 120)
b  = st.sidebar.slider("Width (mm)",            15,  40,  25)
t  = st.sidebar.slider("Thickness (mm)",         2,   6,   3)
E  = st.sidebar.slider("Young's Modulus (GPa)", 65, 210, 70)
nu = st.sidebar.slider("Poisson's Ratio",     0.27, 0.35, 0.33, step=0.01)
F  = st.sidebar.slider("Applied Force (N)",      5, 100,  60)
selected_model = st.sidebar.selectbox("Surrogate Model", MODEL_NAMES, index=2)

params = dict(length_mm=float(L), width_mm=float(b), thickness_mm=float(t),
              youngs_modulus_GPa=float(E), poisson_ratio=float(nu), force_N=float(F))

# ------------------------------------------------------------------ #
#  Load data + compute analytical reference                            #
# ------------------------------------------------------------------ #
df_train = load_training_data()

try:
    bp = BeamParameters(**params)
    analytical = CantileverBeam(bp).solve()
except ValueError as e:
    st.error(str(e)); st.stop()

# ------------------------------------------------------------------ #
#  Domain check                                                         #
# ------------------------------------------------------------------ #
if df_train is not None:
    flags = check_domain(params, df_train)
    if flags:
        st.warning("**Extrapolation warning** – prediction may be less reliable:")
        for k, msg in flags.items():
            st.caption(f"  {msg}")
    else:
        st.success("✅ Design point is within the training domain.")
else:
    st.info("Training data not found – domain check skipped.")

st.divider()

# ------------------------------------------------------------------ #
#  Predictions for all targets                                          #
# ------------------------------------------------------------------ #
PIPE_COLS = ["length_mm","width_mm","thickness_mm","youngs_modulus_GPa",
             "poisson_ratio","force_N","I_mm4","EI","slenderness","FL3_EI",
             "area_mm2","aspect_ratio"]

cols = st.columns(3)
analytical_vals = {
    "tip_deflection_mm": analytical.tip_deflection_mm,
    "max_von_mises_MPa": analytical.max_von_mises_MPa,
    "strain_energy_J":   analytical.strain_energy_J,
}

for idx, (target, (label, unit)) in enumerate(TARGETS.items()):
    model = load_model(target, selected_model)
    X = engineer_features_single(params, PIPE_COLS)

    if model is not None:
        # Try to get feature names from the model
        try:
            feat_names = model.feature_names
            X_input = np.array([[params.get(c, engineer_features_single(params, PIPE_COLS)[0][PIPE_COLS.index(c)] if c in PIPE_COLS else 0) for c in feat_names]])
        except Exception:
            X_input = X[:, :len(FEATURE_COLUMNS)]

        try:
            pred = float(model.predict(X_input)[0])
        except Exception:
            pred = None
    else:
        pred = None

    ana_val = analytical_vals[target]
    with cols[idx]:
        st.metric(f"**{label}**",
                  f"{pred:.4f} {unit}" if pred else "N/A",
                  delta=f"{pred - ana_val:+.4f} vs analytical" if pred else None)
        st.caption(f"Analytical: {ana_val:.4f} {unit}")
        if pred:
            err_pct = (pred - ana_val) / abs(ana_val) * 100
            color = "🟢" if abs(err_pct) < 5 else ("🟡" if abs(err_pct) < 15 else "🔴")
            st.caption(f"{color} Model error: {err_pct:+.1f}%")

st.divider()

# ------------------------------------------------------------------ #
#  Sensitivity sweep – one parameter at a time                         #
# ------------------------------------------------------------------ #
st.subheader("Sensitivity Analysis – Parameter Sweep")
sweep_param = st.selectbox("Vary parameter",
    ["length_mm","width_mm","thickness_mm","youngs_modulus_GPa","force_N"])
sweep_target = st.selectbox("Target output", list(TARGETS.keys()), format_func=lambda x: TARGETS[x][0])

n_pts = 60
lo, hi = PARAM_BOUNDS[sweep_param]
sweep_vals = np.linspace(lo, hi, n_pts)

ana_results, surr_results = [], []
model_sweep = load_model(sweep_target, selected_model)

for sv in sweep_vals:
    p = dict(params)
    p[sweep_param] = float(sv)
    try:
        bp2 = BeamParameters(**p)
        r2  = CantileverBeam(bp2).solve()
        ana_results.append(getattr(r2, sweep_target))
    except Exception:
        ana_results.append(np.nan)

    if model_sweep is not None:
        try:
            feat_names = model_sweep.feature_names
            X_sv = np.array([[p.get(c, 0.0) for c in feat_names]])
            surr_results.append(float(model_sweep.predict(X_sv)[0]))
        except Exception:
            surr_results.append(np.nan)

fig_sweep = go.Figure()
fig_sweep.add_trace(go.Scatter(x=sweep_vals, y=ana_results, mode="lines",
    name="Analytical", line=dict(color="#1f77b4", width=2, dash="dash")))
if surr_results and any(not np.isnan(v) for v in surr_results):
    fig_sweep.add_trace(go.Scatter(x=sweep_vals, y=surr_results, mode="lines",
        name=selected_model, line=dict(color="firebrick", width=2)))
fig_sweep.add_vline(x=params[sweep_param], line_dash="dot",
                     annotation_text="Current", line_color="gray")
fig_sweep.update_layout(
    xaxis_title=f"{sweep_param}",
    yaxis_title=f"{TARGETS[sweep_target][0]} ({TARGETS[sweep_target][1]})",
    height=340, margin=dict(l=0, r=0, t=10, b=0),
    legend=dict(x=0.01, y=0.99)
)
st.plotly_chart(fig_sweep, use_container_width=True)

st.caption(
    f"Sweep of **{sweep_param}** from {lo} to {hi}; all other parameters held constant. "
    "Dashed line = analytical baseline."
)