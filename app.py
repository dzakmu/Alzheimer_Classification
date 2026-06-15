import streamlit as st

from config import APP_CSS
from ui_components import render_sidebar, render_tab_classify, render_tab_cm


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroScan AI · Alzheimer Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject global CSS ──
st.markdown(APP_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR  (menghasilkan semua state widget)
# ─────────────────────────────────────────────
state = render_sidebar()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:1.5rem;">
    <h1>🧠 Alzheimer MRI Classification</h1>
    <p style="color:#2d4a63;font-size:0.88rem;margin-top:-4px;font-family:'DM Mono',monospace;
              letter-spacing:1px;text-transform:uppercase;">
        EfficientNetB0 · Grad-CAM · 4-Class Detection
    </p>
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TAB NAVIGATION
# ─────────────────────────────────────────────
tab_classify, tab_cm = st.tabs(["Classification", "Confusion Matrix"])

with tab_classify:
    render_tab_classify(state)

with tab_cm:
    render_tab_cm(state)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;">
    <span style="font-family:'DM Mono',monospace;font-size:0.65rem;letter-spacing:2px;
                 text-transform:uppercase;color:#141e2e;">
        NeuroScan AI · EfficientNetB0 · Grad-CAM
    </span>
    <span style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#141e2e;">
        For research &amp; educational use only
    </span>
</div>""", unsafe_allow_html=True)