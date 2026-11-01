"""
Suzuki CAE AI Platform – Streamlit Dashboard
=============================================
Main entry point.  Run with:
    streamlit run app/dashboard.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import streamlit as st

st.set_page_config(
    page_title="Suzuki CAE AI Platform",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔧 Suzuki CAE AI Platform")
st.caption("AI-Driven Surrogate Modelling & Experimental Validation · Job No. 10")

st.markdown("""
This platform demonstrates an end-to-end engineering AI/ML workflow for structural
component analysis.  Use the sidebar to navigate between modules.

| Module | Description |
|--------|-------------|
| **CAE Explorer** | Run analytical FEA and visualise structural results |
| **ML Predictions** | Predict performance with surrogate models + uncertainty |
| **Model Comparison** | Compare all ML models on the test set |
| **Experimental Validation** | Upload or use synthetic experimental data |
| **Uncertainty Analysis** | Identify high-uncertainty regions of the design space |
| **Design Optimization** | Minimise mass subject to structural constraints |

---
*Data provenance: Analytical beam theory + LHS design-of-experiments.
Physical CAE (CalculiX) results require solver installation.
Any data labelled SYNTHETIC is not from physical experiments.*
""")