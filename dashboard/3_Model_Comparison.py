"""Page 3 – Model Comparison: compare all surrogate models on the test set."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

import pickle
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from suzuki_cae.data.preprocessing import DataPipeline, FEATURE_COLUMNS, TARGET_COLUMNS
from suzuki_cae.engineering.beam_theory import batch_analytical
from suzuki_cae.models.model_training import compute_metrics

st.set_page_config(page_title="Model Comparison", page_icon="📊", layout="wide")
st.title("📊 Model Comparison")
st.caption("Test-set evaluation of all surrogate models vs. analytical baseline")

ROOT       = Path(__file__).parents[2]
MODELS_DIR = ROOT / "models" / "saved"
DATA_PATH  = ROOT / "data" / "raw" / "cae" / "beam_dataset_analytical.csv"
METRICS_DIR= ROOT / "results" / "model_metrics"

MODEL_NAMES = ["LinearRegression", "PolynomialRegression", "RandomForest", "XGBoost", "MLP"]
TARGETS = {
    "tip_deflection_mm": ("Tip Deflection", "mm"),
    "max_von_mises_MPa": ("Max Von Mises Stress", "MPa"),
    "strain_energy_J":   ("Strain Energy", "J"),
}


@st.cache_resource
def load_model(target: str, name: str):
    path = MODELS_DIR / f"{target}_{name}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data
def prepare_test_set(target: str):
    if not DATA_PATH.exists():
        return None, None, None
    pipe = DataPipeline(test_size=0.15, val_size=0.15, seed=42)
    df   = pipe.load(DATA_PATH)
    df   = pipe.clean(df)
    df   = pipe.engineer_features(df)
    train, val, test = pipe.split(df)
    pipe.fit_scalers(train)
    feat_cols = list(pipe.scaler_X.feature_names_in_)
    X_te = test[[c for c in feat_cols if c in test.columns]].values
    y_te = test[target].values
    return X_te, y_te, test


# ------------------------------------------------------------------ #
#  Target selector                                                      #
# ------------------------------------------------------------------ #
selected_target = st.selectbox(
    "Output variable",
    list(TARGETS.keys()),
    format_func=lambda x: TARGETS[x][0],
)
label, unit = TARGETS[selected_target]

# ------------------------------------------------------------------ #
#  Load pre-computed metrics CSV (fast path)                            #
# ------------------------------------------------------------------ #
metrics_csv = METRICS_DIR / f"{selected_target}_metrics.csv"
if metrics_csv.exists():
    df_metrics = pd.read_csv(metrics_csv)
else:
    st.warning("Metrics file not found. Run `scripts/train_models.py` first.")
    st.stop()

# ------------------------------------------------------------------ #
#  Metrics table                                                        #
# ------------------------------------------------------------------ #
st.subheader("Test-Set Metrics")
show_cols = ["label", "R2", "RMSE", "MAE", "MAPE_pct"]
df_show = df_metrics[show_cols].copy()
df_show.columns = ["Model", "R²", f"RMSE ({unit})", f"MAE ({unit})", "MAPE (%)"]
df_show = df_show.sort_values("R²", ascending=False)

def _colour_r2(v):
    c = "background-color: #d4f1c0" if v > 0.99 else (
        "background-color: #fff3cd" if v > 0.95 else "background-color: #fcd4d4")
    return c

styled = df_show.style.applymap(_colour_r2, subset=["R²"]).format(
    {"R²": "{:.4f}", f"RMSE ({unit})": "{:.4f}", f"MAE ({unit})": "{:.4f}", "MAPE (%)": "{:.2f}"}
)
st.dataframe(styled, use_container_width=True, hide_index=True)

st.divider()

# ------------------------------------------------------------------ #
#  Predicted vs Actual plots (load models & test set)                  #
# ------------------------------------------------------------------ #
X_te, y_te, test_df = prepare_test_set(selected_target)
if X_te is None:
    st.warning("Training data not found. Run generate_dataset.py first.")
    st.stop()

st.subheader("Predicted vs Actual (Test Set)")
cols = st.columns(3)
colours = px.colors.qualitative.Set2

for i, mname in enumerate(MODEL_NAMES):
    model = load_model(selected_target, mname)
    if model is None:
        continue
    try:
        feat_names = model.feature_names
        X_m = test_df[[c for c in feat_names if c in test_df.columns]].values
        y_pred = model.predict(X_m)
    except Exception as e:
        continue

    col = cols[i % 3]
    with col:
        metrics = compute_metrics(y_te, y_pred)
        lim = [min(y_te.min(), y_pred.min()), max(y_te.max(), y_pred.max())]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_te, y=y_pred, mode="markers",
            marker=dict(color=colours[i % len(colours)], size=5, opacity=0.7),
            name=mname))
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines",
            line=dict(color="black", dash="dash", width=1), showlegend=False))
        fig.update_layout(
            title=f"{mname}<br><sup>R²={metrics['R2']:.4f} | RMSE={metrics['RMSE']:.3f} {unit}</sup>",
            xaxis_title=f"Actual ({unit})", yaxis_title=f"Predicted ({unit})",
            height=300, margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------ #
#  Residual distributions                                               #
# ------------------------------------------------------------------ #
st.divider()
st.subheader("Residual Distributions (Predicted − Actual)")

fig_res = go.Figure()
for i, mname in enumerate(MODEL_NAMES):
    model = load_model(selected_target, mname)
    if model is None:
        continue
    try:
        feat_names = model.feature_names
        X_m = test_df[[c for c in feat_names if c in test_df.columns]].values
        y_pred = model.predict(X_m)
        residuals = y_pred - y_te
        fig_res.add_trace(go.Violin(y=residuals, name=mname,
            box_visible=True, meanline_visible=True,
            fillcolor=colours[i % len(colours)], opacity=0.6,
            line_color=colours[i % len(colours)]))
    except Exception:
        continue

fig_res.add_hline(y=0, line_dash="dash", line_color="black")
fig_res.update_layout(yaxis_title=f"Residual ({unit})", height=380,
                       margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig_res, use_container_width=True)

# ------------------------------------------------------------------ #
#  Bar chart: R² comparison across targets                             #
# ------------------------------------------------------------------ #
st.divider()
st.subheader("R² Across All Targets and Models")

r2_data = []
for tgt in TARGETS:
    csv = METRICS_DIR / f"{tgt}_metrics.csv"
    if csv.exists():
        df_t = pd.read_csv(csv)
        for _, row in df_t.iterrows():
            r2_data.append({"target": TARGETS[tgt][0], "model": row["label"].split("[")[0],
                             "R2": row["R2"]})

if r2_data:
    df_r2 = pd.DataFrame(r2_data)
    fig_bar = px.bar(df_r2, x="target", y="R2", color="model", barmode="group",
                     range_y=[0.8, 1.01],
                     color_discrete_sequence=px.colors.qualitative.Set2)
    fig_bar.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0),
                           yaxis_title="R²", xaxis_title="")
    st.plotly_chart(fig_bar, use_container_width=True)

st.caption("Analytical baseline shown for reference. Models trained with 70/15/15 leakage-resistant split.")