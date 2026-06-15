import os
import numpy as np

# ── Class info ────────────────────────────────
CLASS_NAMES  = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"]
CLASS_COLORS = ["#fb923c", "#f87171", "#34d399", "#fbbf24"]
BAR_COLORS   = CLASS_COLORS

CLASS_META = {
    0: ("#fb923c", "badge-alert",   "Mild",     "⚠️"),
    1: ("#f87171", "badge-danger",  "Moderate", "🔴"),
    2: ("#34d399", "badge-safe",    "Normal",   "✅"),
    3: ("#fbbf24", "badge-warning", "Very Mild","🟡"),
}

# ── Image / model settings ─────────────────────
IMG_SIZE              = (224, 224)
BASE_MODEL_LAYER_NAME = "efficientnetb0"
LAST_CONV_LAYER_NAME  = "top_conv"

# _BASE = os.path.dirname(os.path.abspath(__file__))
# _CANDIDATES = [
#     os.path.join(_BASE, "effnet_best_weights_s1.keras"),
#     os.path.join(_BASE, "model", "effnet_best_weights_s1.keras"),
# ]
# MODEL_PATH = next((p for p in _CANDIDATES if os.path.exists(p)), _CANDIDATES[0])

HF_REPO_ID = "Jakmu/alzheimer-efficientnet"
HF_MODEL_FILE = "effnet_best_weights_s1.keras"

# ── Default confusion matrix ───────────────────
# Ganti dengan hasil evaluasi model kamu yang sesungguhnya
DEFAULT_CM = np.array([
    [179,  2, 10,  9],   # Mild Demented
    [  3, 45,  1,  3],   # Moderate Demented
    [  5,  0,627,  6],   # Non Demented
    [ 12,  1, 15,688],   # Very Mild Demented
])

# ── CSS ────────────────────────────────────────
APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #080c12; }
.main .block-container { padding: 2rem 2.5rem; max-width: 1400px; }

[data-testid="stSidebar"] { background: #0d1117 !important; border-right: 1px solid #1c2433; }
[data-testid="stSidebar"] .block-container { padding: 2rem 1.5rem; }

h1 { font-family: 'Syne', sans-serif !important; font-weight: 800 !important;
     font-size: 1.9rem !important; color: #e8f4fd !important;
     letter-spacing: -0.5px !important; line-height: 1.2 !important; }
h2, h3 { font-family: 'Syne', sans-serif !important; font-weight: 700 !important; color: #c9dff0 !important; }
p, li, span { color: #8aa0b8; }
code { font-family: 'DM Mono', monospace !important; font-size: 0.82rem !important; }

.card {
    background: linear-gradient(145deg, #0d1117 0%, #111827 100%);
    border: 1px solid #1c2d3f; border-radius: 14px;
    padding: 22px 24px; margin-bottom: 18px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
}
.card-glow {
    background: linear-gradient(145deg, #0d1117 0%, #0f1e2d 100%);
    border: 1px solid #1e4060; border-radius: 14px;
    padding: 22px 24px; margin-bottom: 18px;
    box-shadow: 0 0 30px rgba(56,189,248,0.06), 0 4px 24px rgba(0,0,0,0.4);
}
.label {
    font-family: 'DM Mono', monospace; font-size: 0.68rem; font-weight: 500;
    letter-spacing: 2.5px; text-transform: uppercase; color: #38bdf8;
    margin-bottom: 14px; display: flex; align-items: center; gap: 8px;
}
.label::after { content: ''; flex: 1; height: 1px; background: linear-gradient(90deg, #1e4060, transparent); }

.badge {
    display: inline-flex; align-items: center; gap: 7px;
    padding: 8px 18px; border-radius: 8px;
    font-family: 'Syne', sans-serif; font-weight: 700;
    font-size: 0.92rem; letter-spacing: 0.3px;
}
.badge-safe    { background: rgba(16,185,129,0.12); color: #34d399; border: 1px solid rgba(52,211,153,0.3); }
.badge-warning { background: rgba(251,191,36,0.10);  color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); }
.badge-danger  { background: rgba(248,113,113,0.10); color: #f87171; border: 1px solid rgba(248,113,113,0.3); }
.badge-alert   { background: rgba(251,146,60,0.10);  color: #fb923c; border: 1px solid rgba(251,146,60,0.3); }

.conf-row { margin-bottom: 12px; }
.conf-header { display: flex; justify-content: space-between; margin-bottom: 5px; align-items: baseline; }
.conf-cls  { font-size: 0.82rem; color: #8aa0b8; font-family: 'DM Sans', sans-serif; }
.conf-pct  { font-family: 'DM Mono', monospace; font-size: 0.82rem; font-weight: 500; }
.bar-bg    { background: #141e2e; border-radius: 4px; height: 6px; overflow: hidden; }
.bar-fill  { height: 100%; border-radius: 4px; transition: width 0.6s cubic-bezier(.4,0,.2,1); }

.big-conf  { font-family: 'Syne', sans-serif; font-size: 3.2rem; font-weight: 800;
             line-height: 1; letter-spacing: -2px; color: #38bdf8; }
.big-conf-label { font-family: 'DM Mono', monospace; font-size: 0.68rem; letter-spacing: 2px;
                  text-transform: uppercase; color: #3d5a73; margin-top: 4px; }

[data-testid="stFileUploader"] {
    background: #0d1117 !important; border: 1.5px dashed #1c3a52 !important;
    border-radius: 12px !important; padding: 8px !important;
}
.stButton > button {
    background: linear-gradient(135deg, #0369a1, #0ea5e9) !important;
    color: #fff !important; border: none !important; border-radius: 10px !important;
    padding: 0.65rem 1.5rem !important; font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important; font-size: 0.9rem !important;
    letter-spacing: 0.5px !important; width: 100% !important;
    box-shadow: 0 4px 14px rgba(14,165,233,0.25) !important;
}
.stButton > button:hover { transform: translateY(-1px) !important; }
hr { border: none; border-top: 1px solid #141e2e; margin: 1.5rem 0; }

[data-testid="stExpander"] {
    background: #0d1117 !important; border: 1px solid #1c2d3f !important; border-radius: 12px !important;
}
.cam-caption {
    text-align: center; font-family: 'DM Mono', monospace; font-size: 0.65rem;
    letter-spacing: 2px; text-transform: uppercase; color: #3d5a73; margin-top: 7px;
}
.empty-state { text-align: center; padding: 70px 24px; border: 1.5px dashed #141e2e; border-radius: 16px; }

.severity-item { display: flex; align-items: center; gap: 10px; padding: 8px 0; border-bottom: 1px solid #0f1923; }
.severity-dot  { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.severity-name { font-size: 0.82rem; color: #8aa0b8; }
.severity-desc { font-size: 0.72rem; color: #3d5a73; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #080c12; }
::-webkit-scrollbar-thumb { background: #1c2d3f; border-radius: 3px; }

[data-testid="stAlert"] {
    background: #0d1117 !important; border: 1px solid #1c3a52 !important;
    border-radius: 10px !important; color: #8aa0b8 !important;
}
.metric-chip {
    display: inline-block; padding: 4px 12px; border-radius: 6px;
    font-family: 'DM Mono', monospace; font-size: 0.72rem; font-weight: 500;
    background: rgba(56,189,248,0.08); color: #38bdf8;
    border: 1px solid rgba(56,189,248,0.2); margin: 3px 3px 3px 0;
}
.cm-info-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 10px 0; border-bottom: 1px solid #0f1923;
}
.cm-info-label { font-family: 'DM Mono', monospace; font-size: 0.72rem;
                 letter-spacing: 1.5px; text-transform: uppercase; color: #2d4a63; }
.cm-info-value { font-family: 'Syne', sans-serif; font-size: 0.92rem;
                 font-weight: 700; color: #c9dff0; }
</style>
"""