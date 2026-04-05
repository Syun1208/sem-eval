import os
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)


def plot_training_curves(history, save_dir):
    """Saves loss and accuracy curves as high-quality PNG files."""
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    TRAIN_COLOR = "#1E40AF"   # deep blue
    VAL_COLOR   = "#DC2626"   # vivid red
    BG_COLOR    = "#FFFFFF"
    SPINE_COLOR = "#1F2937"
    FONT_TITLE  = {"fontsize": 13, "fontweight": "bold", "color": SPINE_COLOR, "pad": 14}
    FONT_LABEL  = {"fontsize": 11, "color": "#374151"}
    FONT_TICK   = {"labelsize": 10, "colors": "#374151"}
    FONT_LEGEND = {"fontsize": 10, "framealpha": 0.0, "edgecolor": "none"}

    def _style(ax):
        ax.set_facecolor(BG_COLOR)
        ax.figure.patch.set_facecolor(BG_COLOR)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("bottom", "left"):
            ax.spines[spine].set_color(SPINE_COLOR)
            ax.spines[spine].set_linewidth(1.2)
        ax.tick_params(axis="both", **FONT_TICK)
        ax.grid(False)

    # ── Chart 1: Loss ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, history["train_loss"],
            label="Training Loss", color=TRAIN_COLOR,
            linewidth=2.5, marker="o", markersize=5, markerfacecolor="white",
            markeredgewidth=2, markeredgecolor=TRAIN_COLOR, zorder=3)
    ax.plot(epochs, history["val_loss"],
            label="Validation Loss", color=VAL_COLOR,
            linewidth=2.5, marker="s", markersize=5, markerfacecolor="white",
            markeredgewidth=2, markeredgecolor=VAL_COLOR, zorder=3)
    _style(ax)
    ax.set_title(
        "Training loss declines consistently while validation loss converges,\n"
        "indicating stable generalisation across epochs.",
        **FONT_TITLE
    )
    ax.set_xlabel("Epoch", **FONT_LABEL)
    ax.set_ylabel("Loss", **FONT_LABEL)
    ax.set_xticks(list(epochs))
    ax.legend(**FONT_LEGEND)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)

    # ── Chart 2: Accuracy ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    if "train_accuracy" in history:
        ax.plot(epochs, history["train_accuracy"],
                label="Training Accuracy", color=TRAIN_COLOR,
                linewidth=2.5, marker="o", markersize=5, markerfacecolor="white",
                markeredgewidth=2, markeredgecolor=TRAIN_COLOR, zorder=3)
    ax.plot(epochs, history["val_accuracy"],
            label="Validation Accuracy", color=VAL_COLOR,
            linewidth=2.5, marker="s", markersize=5, markerfacecolor="white",
            markeredgewidth=2, markeredgecolor=VAL_COLOR, zorder=3)
    _style(ax)
    ax.set_title(
        "Both training and validation accuracy rise steadily, reflecting\n"
        "the model's progressive improvement in distinguishing human from AI-generated text.",
        **FONT_TITLE
    )
    ax.set_xlabel("Epoch", **FONT_LABEL)
    ax.set_ylabel("Accuracy", **FONT_LABEL)
    ax.set_xticks(list(epochs))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))
    ax.legend(**FONT_LEGEND)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "accuracy_curve.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Training curves saved to {save_dir}/")


def plot_confusion_matrix(y_true, y_pred, label_names, save_path):
    """Saves a labelled confusion matrix heatmap as a PNG file."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)

    ticks = np.arange(len(label_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(label_names)
    ax.set_yticklabels(label_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    thresh = 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > thresh else "black",
                    fontsize=12)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Confusion matrix saved to {save_path}")
