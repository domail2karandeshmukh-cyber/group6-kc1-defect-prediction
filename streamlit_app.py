
import json
import os
from pathlib import Path

import requests
import streamlit as st

API_BASE = os.environ.get("API_BASE_URL", "http://api:8000").rstrip("/")
DEFAULTS_PATH = Path(os.environ.get("ARTIFACT_DIR", "/app/artifacts")) / "streamlit_defaults.json"

FEATURE_NAMES = [
    "loc", "v(g)", "ev(g)", "iv(g)", "n", "v", "l", "d", "i", "e",
    "b", "t", "lOCode", "lOComment", "lOBlank", "locCodeAndComment",
    "uniq_Op", "uniq_Opnd", "total_Op", "total_Opnd", "branchCount"
]

FEATURE_GROUPS = {
    "Complexity": ["loc", "v(g)", "ev(g)", "iv(g)", "branchCount"],
    "Halstead metrics": ["n", "v", "l", "d", "i", "e", "b", "t"],
    "Code structure": ["lOCode", "lOComment", "lOBlank", "locCodeAndComment"],
    "Operators and operands": ["uniq_Op", "uniq_Opnd", "total_Op", "total_Opnd"],
}

if DEFAULTS_PATH.exists():
    with open(DEFAULTS_PATH, "r") as f:
        DEFAULTS = json.load(f)
else:
    DEFAULTS = {name: 0.0 for name in FEATURE_NAMES}

st.set_page_config(
    page_title="Group 6 - KC1 Defect Prediction",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    "<style>\n"
    ".block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1250px;}\n"
    "[data-testid=\"stMetric\"] {background: rgba(120,120,120,0.06); border: 1px solid rgba(120,120,120,0.18); padding: 14px; border-radius: 12px;}\n"
    ".small-note {font-size: 0.92rem; opacity: 0.78;}\n"
    ".result-box {padding: 1rem 1.1rem; border-radius: 12px; border: 1px solid rgba(120,120,120,0.22); margin-top: 0.8rem;}\n"
    "</style>",
    unsafe_allow_html=True
)

# Initialize widget state before controls are rendered.
for idx, feature in enumerate(FEATURE_NAMES):
    key = f"feature_{idx}"
    if key not in st.session_state:
        st.session_state[key] = float(DEFAULTS.get(feature, 0.0))

# API health check.
try:
    health_response = requests.get(f"{API_BASE}/health", timeout=5)
    health_response.raise_for_status()
    health_data = health_response.json()
except Exception as exc:
    st.error("FastAPI is not reachable. Start the API service first with Docker Compose.")
    st.code(str(exc))
    st.stop()

# Sidebar project information and controls.
st.sidebar.title("Group 6 Project")
st.sidebar.markdown("**Dataset:** NASA PROMISE KC1")
st.sidebar.markdown("**Task:** Software defect prediction")
st.sidebar.markdown("**Serving:** FastAPI REST API")
st.sidebar.markdown("**Tracking:** MLflow")
st.sidebar.markdown("**Interface:** Streamlit")
st.sidebar.divider()

if st.sidebar.button("Load KC1 sample", use_container_width=True):
    for idx, feature in enumerate(FEATURE_NAMES):
        st.session_state[f"feature_{idx}"] = float(DEFAULTS.get(feature, 0.0))

if st.sidebar.button("Clear inputs", use_container_width=True):
    for idx, feature in enumerate(FEATURE_NAMES):
        st.session_state[f"feature_{idx}"] = 0.0

st.sidebar.divider()
st.sidebar.caption("Prediction path")
st.sidebar.markdown("Streamlit → FastAPI `/predict` → MLflow-tracked model")

# Main dashboard.
st.title("KC1 Software Defect Prediction")
st.caption("Group 6 | Machine Learning | End-to-end model tracking, API serving and prediction interface")

m1, m2, m3, m4 = st.columns(4)
m1.metric("API status", health_data.get("status", "unknown"))
m2.metric("Model", health_data.get("model", "unknown"))
m3.metric("Threshold", health_data.get("threshold", "unknown"))
m4.metric("Input features", health_data.get("feature_count", 21))

st.divider()
st.subheader("KC1 code metrics")
st.markdown('<div class="small-note">Enter the 21 numerical metrics for a software module. A held-out KC1 sample is loaded by default.</div>', unsafe_allow_html=True)

payload = {}
tabs = st.tabs(list(FEATURE_GROUPS.keys()))

for tab, (group_name, features) in zip(tabs, FEATURE_GROUPS.items()):
    with tab:
        cols = st.columns(2)
        for position, feature in enumerate(features):
            idx = FEATURE_NAMES.index(feature)
            with cols[position % 2]:
                payload[feature] = st.number_input(
                    feature,
                    format="%.6f",
                    key=f"feature_{idx}",
                )

st.divider()
predict_clicked = st.button("Predict defect", type="primary", use_container_width=True)

if predict_clicked:
    try:
        response = requests.post(f"{API_BASE}/predict", json=payload, timeout=15)
        response.raise_for_status()
        result = response.json()

        label = result["label"]
        probability = float(result.get("defect_probability", 0.0))
        threshold = float(result.get("threshold", 0.5))

        st.subheader("Prediction result")
        if label == "Defective":
            st.error("Predicted class: Defective")
        else:
            st.success("Predicted class: Non-defective")

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Class", result["prediction"])
        r2.metric("Defect probability", f"{probability * 100:.1f}%")
        r3.metric("Threshold", f"{threshold:.2f}")
        r4.metric("Model", result["model"])

        st.progress(
            min(max(probability, 0.0), 1.0),
            text=f"Defect probability: {probability * 100:.1f}%"
        )

        if probability >= threshold:
            st.info(
                f"The predicted defect probability ({probability:.3f}) is at or above the decision threshold ({threshold:.2f}), so the module is classified as Defective."
            )
        else:
            st.info(
                f"The predicted defect probability ({probability:.3f}) is below the decision threshold ({threshold:.2f}), so the module is classified as Non-defective."
            )

        with st.expander("Technical API response"):
            st.json(result)

    except Exception as exc:
        st.error("Prediction request failed.")
        st.code(str(exc))

with st.expander("System architecture"):
    st.markdown("**KC1 metrics → Streamlit → FastAPI `/predict` → selected MLflow-tracked model → prediction response**")
