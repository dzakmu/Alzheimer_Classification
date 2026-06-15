import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from PIL import Image

from config import (
    CLASS_NAMES, CLASS_COLORS, CLASS_META,
    BASE_MODEL_LAYER_NAME, LAST_CONV_LAYER_NAME,
    MODEL_PATH, DEFAULT_CM, IMG_SIZE,
)
from model_utils import (
    load_model, make_gradcam_heatmap,
    preprocess, render_prob_bars, OVERLAY_FNS,
)
from cm_utils import (
    compute_metrics, plot_confusion_matrix,
    plot_per_class_metrics, cm_to_csv,
)


# ══════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════
def render_sidebar() -> dict:
    """
    Render sidebar lengkap.
    Returns dict berisi semua nilai widget yang dibutuhkan halaman utama.
    """
    with st.sidebar:
        # ── Branding ──
        st.markdown("""
        <div style="margin-bottom:2rem;">
            <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;
                        color:#e8f4fd;letter-spacing:-0.5px;">MRIScan</div>
            <div style="font-family:'DM Mono',monospace;font-size:0.65rem;letter-spacing:3px;
                        text-transform:uppercase;color:#1e4060;margin-top:2px;">
                Alzheimer MRI Classifier
            </div>
        </div>""", unsafe_allow_html=True)

        # ── Upload ──
        st.markdown('<div class="label">Input</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload Brain MRI", type=["jpg","jpeg","png"],
            label_visibility="collapsed",
        )
        st.markdown(
            '<p style="color:#2d4a63;font-size:0.75rem;margin-top:6px;">'
            'JPG / PNG · 224×224 px recommended</p>',
            unsafe_allow_html=True,
        )
        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Grad-CAM settings ──
        st.markdown('<div class="label">Grad-CAM Settings</div>', unsafe_allow_html=True)
        overlay_mode = st.selectbox(
            "Overlay Style", ["RAW","SMOOTH","CLEAN"],
            format_func=lambda x: {"RAW":"Raw","SMOOTH":"Smooth","CLEAN":"Clean"}[x],
        )
        alpha = st.slider("Overlay Intensity", 0.1, 0.9, 0.4, 0.05, format="%.2f")
        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Confusion Matrix settings ──
        st.markdown('<div class="label">Confusion Matrix</div>', unsafe_allow_html=True)
        cm_normalize = st.toggle("Normalize (row %)", value=False)
        cm_source    = st.radio(
            "Data source",
            ["Use default CM", "Input manual CM"],
            label_visibility="collapsed",
        )

        custom_cm = DEFAULT_CM.copy()
        if cm_source == "Input manual CM":
            st.markdown(
                '<p style="color:#2d4a63;font-size:0.74rem;margin-bottom:6px;">'
                'Enter 4×4 values (row = true, col = predicted):</p>',
                unsafe_allow_html=True,
            )
            rows = []
            for i, cls in enumerate(["Mild","Moderate","Non","VeryMild"]):
                row_str = st.text_input(
                    f"Row {cls}",
                    value=", ".join(str(v) for v in DEFAULT_CM[i]),
                    key=f"cm_row_{i}",
                )
                try:
                    parsed = [int(x.strip()) for x in row_str.split(",")]
                    rows.append(parsed if len(parsed) == 4 else list(DEFAULT_CM[i]))
                except Exception:
                    rows.append(list(DEFAULT_CM[i]))
            custom_cm = np.array(rows)

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Advanced layer names ──
        with st.expander("🔧 Advanced · Layer Names"):
            custom_base = st.text_input(
                "Backbone layer name", value=BASE_MODEL_LAYER_NAME,
                help="Nama sub-model backbone di dalam model kamu",
            )
            custom_conv = st.text_input(
                "Last conv layer name", value=LAST_CONV_LAYER_NAME,
                help="Nama Conv2D terakhir di dalam backbone",
            )

        analyze = st.button("🔍  Analyze MRI")
        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Class reference ──
        st.markdown('<div class="label">Class Reference</div>', unsafe_allow_html=True)
        for dot, name, desc in [
            ("#34d399","Non Demented",      "No cognitive decline"),
            ("#fbbf24","Very Mild Demented","Minimal memory loss"),
            ("#fb923c","Mild Demented",     "Noticeable impairment"),
            ("#f87171","Moderate Demented", "Significant decline"),
        ]:
            st.markdown(f"""
            <div class="severity-item">
                <div class="severity-dot" style="background:{dot}"></div>
                <div><div class="severity-name">{name}</div>
                     <div class="severity-desc">{desc}</div></div>
            </div>""", unsafe_allow_html=True)

        # ── Model status ──
        st.markdown("<hr>", unsafe_allow_html=True)
        model_found = os.path.exists(MODEL_PATH)
        dot_c = "#34d399" if model_found else "#f87171"
        dot_t = "Model loaded" if model_found else "Using random weights"
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;margin-top:8px;">
            <div style="width:7px;height:7px;border-radius:50%;background:{dot_c};
                        box-shadow:0 0 6px {dot_c};"></div>
            <span style="font-family:'DM Mono',monospace;font-size:0.68rem;color:#3d5a73;
                         letter-spacing:1px;text-transform:uppercase;">{dot_t}</span>
        </div>
        <div style="font-family:'DM Mono',monospace;font-size:0.62rem;color:#1c2d3f;
                    margin-top:6px;">{os.path.basename(MODEL_PATH)}</div>
        """, unsafe_allow_html=True)
        if not model_found:
            st.info("💡 Letakkan `effnet_best_weights_s1.keras` di folder yang sama dengan `app.py`, lalu restart.")

    return {
        "uploaded":     uploaded,
        "overlay_mode": overlay_mode,
        "alpha":        alpha,
        "cm_normalize": cm_normalize,
        "custom_cm":    custom_cm,
        "custom_base":  custom_base,
        "custom_conv":  custom_conv,
        "analyze":      analyze,
    }


# ══════════════════════════════════════════════
# TAB 1 — CLASSIFICATION
# ══════════════════════════════════════════════
def render_tab_classify(state: dict) -> None:
    """Render konten tab Classification."""
    uploaded     = state["uploaded"]
    analyze      = state["analyze"]
    overlay_mode = state["overlay_mode"]
    alpha        = state["alpha"]
    custom_base  = state["custom_base"]
    custom_conv  = state["custom_conv"]

    if uploaded is None:
        st.markdown("""
        <div class="empty-state">
            <div style="font-size:3.5rem;margin-bottom:16px;">🧠</div>
            <div style="color:#2d4a63;font-size:0.9rem;line-height:1.7;">
                Upload a brain MRI scan from the sidebar<br>
                to begin Alzheimer classification analysis
            </div>
        </div>""", unsafe_allow_html=True)
        return

    pil_img             = Image.open(uploaded)
    img_arr, img_tensor = preprocess(pil_img)

    # ── Preview sebelum analyze ──
    if not analyze:
        _render_preview(pil_img)
        return

    # ── Inference + Grad-CAM ──
    with st.spinner("Running inference + Grad-CAM…"):
        model    = load_model()
        probs    = model.predict(img_tensor, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        meta     = CLASS_META[pred_idx]
        try:
            heatmap    = make_gradcam_heatmap(img_tensor, model,
                                              custom_base, custom_conv, pred_idx)
            gradcam_ok = True
        except Exception as e:
            gradcam_ok  = False
            gradcam_err = str(e)

    _render_metric_cards(probs, pred_idx, meta)
    _render_bars_and_original(probs, img_arr)

    if gradcam_ok:
        _render_gradcam(img_arr, heatmap, overlay_mode, alpha)
    else:
        st.warning(f"Grad-CAM could not run: {gradcam_err}")
        st.info("💡 Cek nama layer di sidebar → **Advanced · Layer Names**.")
        st.image(img_arr.astype("uint8"), use_container_width=True)


# ── sub-render helpers ───────────────────────
def _render_preview(pil_img: Image.Image) -> None:
    col_prev, col_info = st.columns(2, gap="large")
    with col_prev:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="label">MRI Preview</div>', unsafe_allow_html=True)
        st.image(pil_img, use_container_width=True)
        st.markdown(
            f'<p style="color:#2d4a63;font-size:0.75rem;font-family:DM Mono,monospace;margin-top:8px;">'
            f'Original: {pil_img.size[0]}×{pil_img.size[1]}px → resized to 224×224</p>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)
    with col_info:
        st.markdown('<div class="card" style="height:100%;">', unsafe_allow_html=True)
        st.markdown('<div class="label">ℹHow it works</div>', unsafe_allow_html=True)
        st.markdown("""
        <ol style="padding-left:18px;color:#3d5a73;font-size:0.83rem;line-height:2;">
            <li>Upload a brain MRI image (axial view works best)</li>
            <li>The model preprocesses &amp; resizes to 224×224</li>
            <li>EfficientNetB0 extracts deep features</li>
            <li>Softmax outputs probabilities for 4 classes</li>
            <li>Grad-CAM highlights influential brain regions</li>
        </ol>
        <div style="margin-top:20px;padding:14px;background:#0d1117;border-radius:10px;
                    border:1px solid #141e2e;">
            <div style="font-family:'DM Mono',monospace;font-size:0.65rem;letter-spacing:2px;
                        text-transform:uppercase;color:#1e4060;margin-bottom:8px;">Note</div>
            <p style="font-size:0.78rem;color:#2d4a63;margin:0;">
                This tool is for research and educational purposes only.
                Always consult a qualified medical professional for diagnosis.
            </p>
        </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


def _render_metric_cards(probs: np.ndarray, pred_idx: int, meta: tuple) -> None:
    m1, m2, m3 = st.columns(3, gap="large")
    with m1:
        st.markdown(f"""
        <div class="card-glow">
            <div class="label">Diagnosis</div>
            <span class="badge {meta[1]}">{meta[3]} {CLASS_NAMES[pred_idx]}</span>
            <div style="margin-top:14px;">
                <div style="font-family:'DM Mono',monospace;font-size:0.65rem;letter-spacing:2px;
                            text-transform:uppercase;color:#1e4060;">Severity</div>
                <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;
                            color:#c9dff0;margin-top:3px;">{meta[2]}</div>
            </div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="card-glow">
            <div class="label">Confidence</div>
            <div class="big-conf">{probs[pred_idx]*100:.1f}
                <span style="font-size:1.6rem;color:#1e4060">%</span>
            </div>
            <div class="big-conf-label">Model confidence</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        second_idx = int(np.argsort(probs)[-2])
        st.markdown(f"""
        <div class="card">
            <div class="label">Runner-up</div>
            <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;
                        color:#c9dff0;margin-bottom:4px;">{CLASS_NAMES[second_idx]}</div>
            <div style="font-family:'DM Mono',monospace;font-size:1.4rem;font-weight:500;
                        color:#3d5a73;">{probs[second_idx]*100:.1f}%</div>
            <div style="font-family:'DM Mono',monospace;font-size:0.65rem;letter-spacing:2px;
                        text-transform:uppercase;color:#1c2d3f;margin-top:8px;">
                Δ {abs(probs[pred_idx]-probs[second_idx])*100:.1f}% gap
            </div>
        </div>""", unsafe_allow_html=True)


def _render_bars_and_original(probs: np.ndarray, img_arr: np.ndarray) -> None:
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="label">Class Probabilities</div>', unsafe_allow_html=True)
        st.markdown(render_prob_bars(probs), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="label">Original MRI</div>', unsafe_allow_html=True)
        st.image(img_arr.astype("uint8"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


def _render_gradcam(
    img_arr: np.ndarray,
    heatmap: np.ndarray,
    overlay_mode: str,
    alpha: float,
) -> None:
    overlay_fn = OVERLAY_FNS[overlay_mode]
    cam_img    = overlay_fn(img_arr, heatmap, alpha=alpha)
    hm_color   = cv2.cvtColor(
        cv2.applyColorMap(np.uint8(255 * cv2.resize(heatmap, IMG_SIZE)), cv2.COLORMAP_JET),
        cv2.COLOR_BGR2RGB,
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="label">🔬 Grad-CAM Visualization</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.image(img_arr.astype("uint8"), use_container_width=True)
        st.markdown('<div class="cam-caption">Original MRI</div>', unsafe_allow_html=True)
    with c2:
        st.image(hm_color, use_container_width=True)
        st.markdown('<div class="cam-caption">Activation Heatmap</div>', unsafe_allow_html=True)
    with c3:
        st.image(cam_img, use_container_width=True)
        st.markdown(f'<div class="cam-caption">Overlay · {overlay_mode}</div>', unsafe_allow_html=True)

    # Colorbar
    fig, ax = plt.subplots(figsize=(6, 0.35))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    ax.imshow(np.linspace(0, 1, 256).reshape(1, -1), aspect="auto", cmap="jet")
    ax.set_yticks([])
    ax.set_xticks([0, 128, 255])
    ax.set_xticklabels(["Low activation", "Mid", "High activation"],
                       color="#3d5a73", fontsize=7, fontfamily="monospace")
    ax.tick_params(colors="#1c2d3f", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    plt.tight_layout(pad=0)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("""<p style="color:#1c2d3f;font-size:0.75rem;font-family:'DM Mono',monospace;
        text-align:center;margin-top:8px;letter-spacing:1px;">
        RED → most influential brain regions &nbsp;|&nbsp; BLUE → low influence
    </p>""", unsafe_allow_html=True)

    with st.expander("Compare All Overlay Styles"):
        ea, eb, ec = st.columns(3, gap="medium")
        with ea:
            st.image(OVERLAY_FNS["RAW"](img_arr, heatmap, alpha), use_container_width=True)
            st.markdown('<div class="cam-caption">Raw</div>', unsafe_allow_html=True)
        with eb:
            st.image(OVERLAY_FNS["SMOOTH"](img_arr, heatmap, alpha), use_container_width=True)
            st.markdown('<div class="cam-caption">Smooth</div>', unsafe_allow_html=True)
        with ec:
            st.image(OVERLAY_FNS["CLEAN"](img_arr, heatmap, alpha), use_container_width=True)
            st.markdown('<div class="cam-caption">Clean</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 2 — CONFUSION MATRIX
# ══════════════════════════════════════════════
def render_tab_cm(state: dict) -> None:
    """Render konten tab Confusion Matrix."""
    custom_cm    = state["custom_cm"]
    cm_normalize = state["cm_normalize"]

    metrics = compute_metrics(custom_cm)
    total   = int(custom_cm.sum())
    correct = int(np.diag(custom_cm).sum())

    # ── Summary row ──
    st.markdown('<div class="card-glow">', unsafe_allow_html=True)
    st.markdown('<div class="label">Model Performance Summary</div>', unsafe_allow_html=True)
    chip_cols = st.columns(4, gap="medium")
    for col, (label, value, color) in zip(chip_cols, [
        ("Overall Accuracy", f"{metrics['accuracy']*100:.2f}%", "#38bdf8"),
        ("Total Samples",    f"{total:,}",                      "#34d399"),
        ("Correct Pred.",    f"{correct:,}",                    "#a78bfa"),
        ("Macro F1-Score",   f"{metrics['f1'].mean():.4f}",     "#fbbf24"),
    ]):
        with col:
            st.markdown(f"""
            <div style="text-align:center;">
                <div style="font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;
                            color:{color};letter-spacing:-1px;line-height:1;">{value}</div>
                <div style="font-family:'DM Mono',monospace;font-size:0.65rem;letter-spacing:2px;
                            text-transform:uppercase;color:#2d4a63;margin-top:5px;">{label}</div>
            </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── CM plot + per-class detail ──
    cm_col, metric_col = st.columns([1.1, 1], gap="large")
    with cm_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="label">Confusion Matrix</div>', unsafe_allow_html=True)
        title  = "Normalized (Row %)" if cm_normalize else "Confusion Matrix (Absolute)"
        fig_cm = plot_confusion_matrix(custom_cm, normalize=cm_normalize, title=title)
        st.pyplot(fig_cm, use_container_width=True)
        plt.close(fig_cm)
        st.markdown("""<p style="font-family:'DM Mono',monospace;font-size:0.68rem;color:#2d4a63;
                  text-align:center;letter-spacing:1px;margin-top:6px;">
            Diagonal boxes (colored border) = correct predictions
        </p>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with metric_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="label">Per-Class Metrics</div>', unsafe_allow_html=True)
        fig_bar = plot_per_class_metrics(metrics)
        st.pyplot(fig_bar, use_container_width=True)
        plt.close(fig_bar)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="label">Class Detail</div>', unsafe_allow_html=True)
        for i, (cls, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
            support = int(custom_cm[i].sum())
            st.markdown(f"""
            <div class="cm-info-row">
                <div>
                    <div style="display:flex;align-items:center;gap:7px;">
                        <div style="width:8px;height:8px;border-radius:50%;background:{color};"></div>
                        <span class="cm-info-label">{cls}</span>
                    </div>
                    <div style="margin-top:4px;display:flex;gap:6px;flex-wrap:wrap;">
                        <span class="metric-chip">P {metrics['precision'][i]:.3f}</span>
                        <span class="metric-chip">R {metrics['recall'][i]:.3f}</span>
                        <span class="metric-chip">F1 {metrics['f1'][i]:.3f}</span>
                    </div>
                </div>
                <div style="text-align:right;">
                    <div class="cm-info-value">{support}</div>
                    <div class="cm-info-label">samples</div>
                </div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Expanders ──
    with st.expander("Compare: Absolute vs Normalized"):
        ea, eb = st.columns(2, gap="large")
        with ea:
            f1 = plot_confusion_matrix(custom_cm, normalize=False, title="Absolute Counts")
            st.pyplot(f1, use_container_width=True)
            plt.close(f1)
        with eb:
            f2 = plot_confusion_matrix(custom_cm, normalize=True, title="Normalized (Row %)")
            st.pyplot(f2, use_container_width=True)
            plt.close(f2)

    with st.expander("Raw Matrix Values"):
        st.markdown('<div class="label" style="margin-bottom:10px;">4×4 Confusion Matrix</div>',
                    unsafe_allow_html=True)
        df_cm = pd.DataFrame(
            custom_cm,
            index=[f"True: {n}" for n in CLASS_NAMES],
            columns=[f"Pred: {n}" for n in CLASS_NAMES],
        )
        st.dataframe(
            df_cm.style.background_gradient(cmap="Blues", axis=None).format("{:,}"),
            use_container_width=True,
        )
        st.download_button(
            "⬇Download CSV",
            data=cm_to_csv(custom_cm),
            file_name="confusion_matrix.csv",
            mime="text/csv",
        )