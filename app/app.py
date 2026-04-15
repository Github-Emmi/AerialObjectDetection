"""Aerial Object Classification & Detection — Streamlit App."""

import sys
from pathlib import Path

# Allow imports from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
# Allow imports from app/ directory (for api_client)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from components.classifier_ui import render_classifier
from components.detector_ui import render_detector

st.set_page_config(
    page_title="Aerial Bird vs Drone",
    page_icon="🛩️",
    layout="wide",
)

# ── Sidebar ─────────────────────────────────────────────────────────
st.sidebar.title("🛩️ Aerial Detection")
mode = st.sidebar.radio("Mode", ["Classification", "Detection"])

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Models trained on**  \n"
    "3,319 classification images  \n"
    "3,400 detection images  \n"
    "Classes: Bird · Drone"
)

# Show inference engine status
from api_client import get_api_url, is_api_available
api_url = get_api_url()
if api_url:
    if is_api_available():
        st.sidebar.success("⚡ GPU Server: Online")
    else:
        st.sidebar.error("🔴 GPU Server: Offline")
        st.sidebar.caption("Using local CPU inference")
else:
    st.sidebar.info("💻 Local inference mode")

# ── Main Area ───────────────────────────────────────────────────────
if mode == "Classification":
    render_classifier()
else:
    render_detector()
