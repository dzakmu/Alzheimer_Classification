import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import efficientnet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from PIL import Image
import streamlit as st
from huggingface_hub import hf_hub_download

# from config import (
#     IMG_SIZE, MODEL_PATH,
#     BASE_MODEL_LAYER_NAME, LAST_CONV_LAYER_NAME,
#     CLASS_NAMES, CLASS_META, BAR_COLORS,
# )

from config import (
    IMG_SIZE, 
    BASE_MODEL_LAYER_NAME, LAST_CONV_LAYER_NAME,
    CLASS_NAMES, CLASS_META, BAR_COLORS,
)

# ─────────────────────────────────────────────
# BUILD & LOAD MODEL
# ─────────────────────────────────────────────
def build_model(dropout_rate: float = 0.3) -> Model:
    """Bangun arsitektur EfficientNetB0 tanpa pre-trained weights."""
    IMG_SHAPE = IMG_SIZE + (3,)
    inputs = Input(shape=IMG_SHAPE)
    x      = efficientnet.preprocess_input(inputs)
    base   = efficientnet.EfficientNetB0(include_top=False, weights=None)
    base.trainable = False
    x       = base(x, training=False)
    x       = GlobalAveragePooling2D()(x)
    x       = Dropout(dropout_rate)(x)
    outputs = Dense(4, activation="softmax")(x)
    return Model(inputs, outputs)


# @st.cache_resource(show_spinner="Loading model weights…")
# def load_model() -> Model:
#     """Load model dari file .keras; fallback ke arsitektur kosong."""
#     import os
#     if os.path.exists(MODEL_PATH):
#         return tf.keras.models.load_model(MODEL_PATH)
#     return build_model()

from config import HF_REPO_ID, HF_MODEL_FILE

@st.cache_resource(show_spinner="Downloading model from Hugging Face...")
def load_model() -> Model:

    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_MODEL_FILE
    )

    return tf.keras.models.load_model(model_path)


# ─────────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────────
def make_gradcam_heatmap(
    img_array,
    model,
    base_model_name: str = BASE_MODEL_LAYER_NAME,
    last_conv_layer_name: str = LAST_CONV_LAYER_NAME,
    pred_index=None,
) -> np.ndarray:
    """
    Hasilkan Grad-CAM heatmap.

    Strategi:
      1. Ambil backbone via nama layer (base_model_name).
      2. Ambil conv terakhir via nama (last_conv_layer_name).
      3. Fallback otomatis: cari Conv2D terakhir jika nama tidak cocok.
    """
    try:
        base_model      = model.get_layer(base_model_name)
        last_conv_layer = base_model.get_layer(last_conv_layer_name)
        conv_model      = tf.keras.Model(base_model.inputs, last_conv_layer.output)

        cls_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
        x = cls_input
        base_idx = [l.name for l in model.layers].index(base_model.name)
        for layer in model.layers[base_idx + 1:]:
            x = layer(x)
        classifier_model = tf.keras.Model(cls_input, x)

    except (ValueError, AttributeError):
        # ── Fallback: cari Conv2D terakhir secara otomatis ──
        base_model = last_conv_layer = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                base_model = layer
                for sublayer in reversed(layer.layers):
                    if isinstance(sublayer, tf.keras.layers.Conv2D):
                        last_conv_layer = sublayer
                        break
                break

        if base_model is None or last_conv_layer is None:
            raise ValueError(
                f"Layer '{base_model_name}' atau conv layer '{last_conv_layer_name}' "
                "tidak ditemukan. Jalankan model.summary() untuk cek nama layer."
            )

        conv_model = tf.keras.Model(base_model.inputs, last_conv_layer.output)
        cls_input  = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
        x = cls_input
        base_idx = [l.name for l in model.layers].index(base_model.name)
        for layer in model.layers[base_idx + 1:]:
            x = layer(x)
        classifier_model = tf.keras.Model(cls_input, x)

    # ── Forward pass + gradient ──
    img_tensor = tf.cast(img_array, tf.float32)
    with tf.GradientTape() as tape:
        conv_outputs = conv_model(img_tensor)
        tape.watch(conv_outputs)
        preds = classifier_model(conv_outputs)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_score = preds[:, pred_index]

    grads        = tape.gradient(class_score, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap      = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap      = tf.squeeze(heatmap)
    heatmap      = tf.maximum(heatmap, 0)
    heatmap      = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


# ─────────────────────────────────────────────
# OVERLAY HELPERS
# ─────────────────────────────────────────────
def _to_rgb(img: np.ndarray) -> np.ndarray:
    img = img.numpy().astype("uint8") if hasattr(img, "numpy") else np.uint8(img)
    if len(img.shape) == 2 or img.shape[-1] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


def _colormap(heatmap: np.ndarray, size: tuple) -> np.ndarray:
    h = cv2.resize(heatmap, size)
    h = np.uint8(255 * h)
    c = cv2.applyColorMap(h, cv2.COLORMAP_JET)
    return cv2.cvtColor(c, cv2.COLOR_BGR2RGB)


def overlay_raw(img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    img = _to_rgb(img)
    return cv2.addWeighted(
        _colormap(heatmap, (img.shape[1], img.shape[0])), alpha,
        img, 1 - alpha, 0,
    )


def overlay_smooth(img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    img = _to_rgb(img)
    h   = cv2.GaussianBlur(cv2.resize(heatmap, (img.shape[1], img.shape[0])), (5, 5), 0)
    return cv2.addWeighted(
        _colormap(h, (img.shape[1], img.shape[0])), alpha,
        img, 1 - alpha, 0,
    )


def overlay_clean(img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    img = _to_rgb(img)
    h   = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    h   = np.maximum(h, 0)
    h  /= (h.max() + 1e-8)
    h   = cv2.GaussianBlur(h, (21, 21), 0)
    h[h < 0.4] = 0
    _, mask = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 10, 1, cv2.THRESH_BINARY)
    h *= mask
    return cv2.addWeighted(
        _colormap(h, (img.shape[1], img.shape[0])), alpha,
        img, 1 - alpha, 0,
    )


OVERLAY_FNS = {
    "RAW":    overlay_raw,
    "SMOOTH": overlay_smooth,
    "CLEAN":  overlay_clean,
}


# ─────────────────────────────────────────────
# PREPROCESSING & RENDER HELPERS
# ─────────────────────────────────────────────
def preprocess(pil_image: Image.Image):
    """Resize ke IMG_SIZE, return (np.array float32, tf.Tensor batch)."""
    img = pil_image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    return arr, tf.expand_dims(arr, axis=0)


def render_prob_bars(probs: np.ndarray) -> str:
    """Kembalikan HTML string untuk probability bar chart."""
    html = ""
    best = int(np.argmax(probs))
    for i, (cls, p) in enumerate(zip(CLASS_NAMES, probs)):
        pct    = p * 100
        color  = BAR_COLORS[i]
        active = "font-weight:600; color:#e8f4fd;" if i == best else ""
        html += f"""
        <div class="conf-row">
            <div class="conf-header">
                <span class="conf-cls" style="{active}">{cls}</span>
                <span class="conf-pct" style="color:{color}">{pct:.1f}%</span>
            </div>
            <div class="bar-bg">
                <div class="bar-fill" style="width:{pct:.1f}%; background:{color};"></div>
            </div>
        </div>"""
    return html