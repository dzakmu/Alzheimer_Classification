import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from config import CLASS_NAMES, CLASS_COLORS


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
def compute_metrics(cm: np.ndarray) -> dict:
    """
    Hitung metrik evaluasi dari confusion matrix (n×n).

    Returns dict:
        accuracy  : float
        precision : np.ndarray (per kelas)
        recall    : np.ndarray (per kelas)
        f1        : np.ndarray (per kelas)
    """
    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP

    precision = np.where((TP + FP) > 0, TP / (TP + FP), 0.0)
    recall    = np.where((TP + FN) > 0, TP / (TP + FN), 0.0)
    f1        = np.where(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall),
        0.0,
    )
    accuracy = float(TP.sum() / cm.sum())

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# ─────────────────────────────────────────────
# PLOT: CONFUSION MATRIX
# ─────────────────────────────────────────────
def plot_confusion_matrix(
    cm: np.ndarray,
    normalize: bool = False,
    title: str = "Confusion Matrix",
) -> plt.Figure:
    """
    Render confusion matrix dengan tema dark navy → electric blue.

    Args:
        cm        : array (4×4)
        normalize : True → tampilkan proporsi baris; False → tampilkan angka absolut
        title     : judul gambar
    """
    if normalize:
        cm_disp = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        fmt, vmax = ".2f", 1.0
    else:
        cm_disp = cm.astype(int)
        fmt, vmax = "d", int(cm.max())

    n      = cm.shape[0]
    labels = ["Mild\nDemented", "Moderate\nDemented", "Non\nDemented", "Very Mild\nDemented"]

    cmap = LinearSegmentedColormap.from_list(
        "neuroscan",
        ["#080c12", "#0c1e35", "#0d4a7a", "#0ea5e9", "#38bdf8"],
        N=256,
    )

    fig, ax = plt.subplots(figsize=(7, 5.5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    im = ax.imshow(cm_disp, interpolation="nearest", cmap=cmap,
                   vmin=0, vmax=vmax, aspect="auto")

    # ── Colorbar ──
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color="#3d5a73", labelsize=7)
    cbar.outline.set_edgecolor("#1c2d3f")
    cbar.ax.set_facecolor("#0d1117")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#3d5a73",
             fontfamily="monospace", fontsize=7)

    # ── Grid ──
    for i in range(n + 1):
        lw = 1.4 if i == 0 or i == n else 0.5
        ax.axhline(i - 0.5, color="#1c2d3f", linewidth=lw)
        ax.axvline(i - 0.5, color="#1c2d3f", linewidth=lw)

    # ── Teks dalam sel ──
    thresh = cm_disp.max() * 0.55
    for i in range(n):
        for j in range(n):
            val  = cm_disp[i, j]
            text = f"{val:{fmt}}" if fmt == "d" else f"{val:.2f}"
            col  = "#e8f4fd" if val > thresh else "#3d5a73"
            fw   = "bold" if i == j else "normal"
            ax.text(j, i, text, ha="center", va="center",
                    color=col, fontsize=9, fontweight=fw, fontfamily="monospace")

    # ── Diagonal highlight ──
    for k in range(n):
        rect = plt.Rectangle(
            (k - 0.5, k - 0.5), 1, 1,
            linewidth=1.8, edgecolor=CLASS_COLORS[k],
            facecolor="none", zorder=3,
        )
        ax.add_patch(rect)

    # ── Axis labels ──
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, color="#e8f4fd", fontsize=8, fontfamily="monospace")
    ax.set_yticklabels(labels, color="#e8f4fd", fontsize=8, fontfamily="monospace")
    ax.tick_params(colors="#e8f4fd", length=0)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1c2d3f")

    ax.set_xlabel("Predicted Label", color="#e8f4fd", fontsize=8,
                  fontfamily="monospace", labelpad=10)
    ax.set_ylabel("True Label", color="#e8f4fd", fontsize=8,
                  fontfamily="monospace", labelpad=10)

    fig.text(0.5, 0.97, title, ha="center", va="top",
             color="#c9dff0", fontsize=10, fontweight="bold", fontfamily="monospace")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ─────────────────────────────────────────────
# PLOT: PER-CLASS METRICS BAR CHART
# ─────────────────────────────────────────────
def plot_per_class_metrics(metrics: dict) -> plt.Figure:
    """Bar chart horizontal: Precision / Recall / F1 per kelas."""
    labels    = ["Mild\nDemented", "Moderate\nDemented", "Non\nDemented", "Very Mild\nDemented"]
    precision = metrics["precision"]
    recall    = metrics["recall"]
    f1        = metrics["f1"]

    x, w = np.arange(len(labels)), 0.25

    fig, ax = plt.subplots(figsize=(8, 3.8))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    ax.barh(x + w, precision, w, label="Precision", color="#38bdf8", alpha=0.85)
    ax.barh(x,     recall,    w, label="Recall",    color="#34d399", alpha=0.85)
    ax.barh(x - w, f1,        w, label="F1-Score",  color="#fbbf24", alpha=0.85)

    for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
        ax.text(p + 0.01, i + w, f"{p:.2f}", va="center", color="#38bdf8",
                fontsize=7.5, fontfamily="monospace")
        ax.text(r + 0.01, i,     f"{r:.2f}", va="center", color="#34d399",
                fontsize=7.5, fontfamily="monospace")
        ax.text(f + 0.01, i - w, f"{f:.2f}", va="center", color="#fbbf24",
                fontsize=7.5, fontfamily="monospace")

    ax.set_yticks(x)
    ax.set_yticklabels(labels, color="#e8f4fd", fontsize=8, fontfamily="monospace")
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Score", color="#e8f4fd", fontsize=8, fontfamily="monospace")
    ax.tick_params(colors="#e8f4fd", length=0)
    # ax.set_xticklabels(
    #     [f"{v:.1f}" for v in ax.get_xticks()],
    #     color="#e8f4fd", fontsize=7, fontfamily="monospace",
    # )
    ticks = ax.get_xticks()
    ax.set_xticks(ticks)

    ax.set_xticklabels(
        [f"{v:.1f}" for v in ticks],
        color="#e8f4fd",
        fontsize=7,
        fontfamily="monospace",
    )    
    for spine in ax.spines.values():
        spine.set_edgecolor("#1c2d3f")
    ax.xaxis.grid(True, color="#141e2e", linewidth=0.5, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=True,
              facecolor="#0d1117", edgecolor="#1c2d3f",
              labelcolor="#e8f4fd", fontsize=8)

    fig.text(0.5, 0.97, "Per-Class Metrics", ha="center", va="top",
             color="#c9dff0", fontsize=10, fontweight="bold", fontfamily="monospace")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# ─────────────────────────────────────────────
# EXPORT CSV
# ─────────────────────────────────────────────
def cm_to_csv(cm: np.ndarray) -> str:
    """Kembalikan string CSV dari confusion matrix."""
    buf = io.StringIO()
    buf.write("Class," + ",".join(CLASS_NAMES) + ",Total\n")
    for i, row in enumerate(cm):
        buf.write(f"{CLASS_NAMES[i]}," + ",".join(str(v) for v in row) + f",{row.sum()}\n")
    return buf.getvalue()