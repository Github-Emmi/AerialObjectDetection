"""Aerial Object Classification & Detection — Streamlit App."""

import sys
from pathlib import Path

# Allow imports from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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

# ── Main Area ───────────────────────────────────────────────────────
if mode == "Classification":
    render_classifier()
else:
    render_detector()
