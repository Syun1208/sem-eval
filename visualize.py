import os
import sys
import yaml
import torch
import logging
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.manifold import TSNE

import transformers.utils.import_utils
import transformers.modeling_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda *args, **kwargs: True
transformers.modeling_utils.check_torch_load_is_safe = lambda *args, **kwargs: True
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from models.model import HybridClassifier, TLModel, build_model, get_model_name, get_label_names
from dataset.dataset import AgnosticDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def setup_file_logging(log_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S"
    ))
    logging.getLogger().addHandler(fh)
    logger.info(f"Logging to file: {log_path}")

LABEL_NAMES = ["Human", "AI"]
LABEL_COLORS = ["#2563EB", "#DC2626"]


def extract_embeddings(model, dataloader, device, config):
    """Run forward pass and collect combined_features, labels, and optionally extra metadata."""
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            feats = batch["extra_features"].to(device, non_blocking=True)
            
            if config["model"]["model_type"] == "droiddetect":
                features = model(input_ids, mask, labels=None)
                combined_features = features["fused_embedding"]
            else:
                _, _, combined_features = model(input_ids, mask, feats, labels=None)

            all_embeddings.append(combined_features.cpu().numpy())
            all_labels.append(batch["labels"].numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return embeddings, labels


def plot_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    save_path: str,
    title: str = "t-SNE Visualization",
    label_names: list = None,
    perplexity: int = 30,
    n_iter: int = 1000,
    random_state: int = 42,
    languages: np.ndarray = None,
):
    """
    Apply t-SNE to embeddings and save a scatter plot.

    Args:
        embeddings:    2D array of shape [N, D].
        labels:        1D integer array of shape [N] — class indices.
        save_path:     Where to save the PNG.
        title:         Plot title.
        label_names:   Names for each class index (e.g. ["Human", "AI"]).
        perplexity:    t-SNE perplexity parameter.
        n_iter:        t-SNE number of iterations.
        random_state:  Random seed for reproducibility.
        languages:     Optional 1D string array [N] — if provided, marker shape
                       encodes the programming language.
    """
    if label_names is None:
        label_names = LABEL_NAMES

    logger.info(f"Running t-SNE on {len(embeddings)} samples (perplexity={perplexity}, n_iter={n_iter})...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=n_iter,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    coords = tsne.fit_transform(embeddings)  # [N, 2]
    logger.info("t-SNE done.")

    fig, ax = plt.subplots(figsize=(11, 8.5))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    unique_labels = sorted(np.unique(labels).tolist())
    colors = LABEL_COLORS if len(unique_labels) <= 2 else [
        cm.tab10(i / len(unique_labels)) for i in range(len(unique_labels))
    ]

    if languages is not None:
        unique_langs = sorted(set(languages.tolist()))
        markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "+"]
        lang_to_marker = {lang: markers[i % len(markers)] for i, lang in enumerate(unique_langs)}

        for lbl_idx, lbl in enumerate(unique_labels):
            lbl_name = label_names[lbl] if lbl < len(label_names) else str(lbl)
            color = colors[lbl_idx]
            mask_lbl = labels == lbl

            for lang in unique_langs:
                mask_lang = np.array(languages) == lang
                mask = mask_lbl & mask_lang
                if mask.sum() == 0:
                    continue
                ax.scatter(
                    coords[mask, 0], coords[mask, 1],
                    c=color,
                    marker=lang_to_marker[lang],
                    s=30,
                    alpha=0.75,
                    label=f"{lbl_name} / {lang}",
                    linewidths=0,
                    edgecolors="none",
                )
    else:
        markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "+"]
        for lbl_idx, lbl in enumerate(unique_labels):
            lbl_name = label_names[lbl] if lbl < len(label_names) else str(lbl)
            mask = labels == lbl
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=colors[lbl_idx],
                marker=markers[lbl_idx % len(markers)],
                s=25,
                alpha=0.75,
                label=lbl_name,
                linewidths=0,
                edgecolors="none",
            )

    # ── Legend ────────────────────────────────────────────────────────────────
    legend = ax.legend(
        loc="upper right", fontsize=9, markerscale=1.6,
        framealpha=0.9, edgecolor="#CCCCCC",
        facecolor="white", borderpad=0.8,
    )
    legend.get_frame().set_linewidth(0.8)

    # ── Title (wrap long single-line titles; preserve explicit newlines) ──────
    wrapped = title if "\n" in title else "\n".join(textwrap.wrap(title, width=90))
    ax.set_title(
        wrapped,
        fontsize=10.5,
        fontweight="bold",
        color="#1A1A2E",
        pad=12,
        linespacing=1.5,
    )

    # ── Axes labels & spines ──────────────────────────────────────────────────
    ax.set_xlabel("t-SNE Dimension 1", fontsize=10, color="#444444", labelpad=8)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=10, color="#444444", labelpad=8)
    ax.tick_params(colors="#666666", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#CCCCCC")
        spine.set_linewidth(0.8)
    ax.grid(True, linewidth=0.4, alpha=0.6, color="#CCCCCC", linestyle="--")

    fig.tight_layout(pad=1.5)
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"t-SNE plot saved to {save_path}")


def plot_training_curves(
    metrics_path: str,
    save_dir: str,
):
    """
    Read a metrics CSV and save three curve plots:
      - loss_curves.png  : train_loss, train_task_loss, train_supcon_loss, val_loss
      - accuracy_curve.png : val_accuracy
      - f1_curve.png       : val_f1_macro
    """
    df = pd.read_csv(metrics_path)
    os.makedirs(save_dir, exist_ok=True)
    epochs = df["epoch"].values

    # ── 1. Loss curves ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))

    loss_series = [
        ("train_loss",       "Total train loss",      "#2563EB", "-"),
        ("train_task_loss",  "Train task loss",       "#16A34A", "--"),
        ("train_supcon_loss","Train SupCon loss",     "#9333EA", "-."),
        ("val_loss",         "Validation loss",       "#DC2626", "-"),
    ]
    for col, label, color, ls in loss_series:
        if col in df.columns:
            ax.plot(epochs, df[col].values, label=label, color=color,
                    linestyle=ls, linewidth=1.8, marker="o", markersize=4)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and validation loss curves over epochs")
    ax.legend(fontsize=9, framealpha=0.8)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    path = os.path.join(save_dir, "loss_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Loss curves saved to {path}")

    # ── 2. Accuracy curve ─────────────────────────────────────────────────────
    if "val_accuracy" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(epochs, df["val_accuracy"].values,
                color="#2563EB", linewidth=1.8, marker="o", markersize=4,
                label="Validation accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Validation accuracy curve over epochs")
        ax.legend(fontsize=9, framealpha=0.8)
        ax.grid(True, linewidth=0.3, alpha=0.5)
        fig.tight_layout()
        path = os.path.join(save_dir, "accuracy_curve.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info(f"Accuracy curve saved to {path}")

    # ── 3. F1 curve ───────────────────────────────────────────────────────────
    if "val_f1_macro" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(epochs, df["val_f1_macro"].values,
                color="#DC2626", linewidth=1.8, marker="s", markersize=4,
                label="Validation F1 macro")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("F1 macro")
        ax.set_title("Validation F1 macro score curve over epochs")
        ax.legend(fontsize=9, framealpha=0.8)
        ax.grid(True, linewidth=0.3, alpha=0.5)
        fig.tight_layout()
        path = os.path.join(save_dir, "f1_curve.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info(f"F1 curve saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="t-SNE visualization of HybridClassifier embeddings")
    parser.add_argument("--config_dir", type=str, required=False, default="./config",
                        help="Path to config directory containing config.yaml.")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to saved checkpoint folder (contains config.yaml, model_state.bin, tokenizer).")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"],
                        help="Which data split to visualize: 'train' or 'val'.")
    parser.add_argument("--data_file", type=str, default=None,
                        help="Explicit path to a .parquet file. Overrides the split-based path from config.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output PNG path. Defaults to <checkpoint_dir>/tsne_<split>.png.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--perplexity", type=int, default=30)
    parser.add_argument("--n_iter", type=int, default=1000)
    parser.add_argument("--color_by_language", action="store_true",
                        help="If set and dataset has a 'language' column, use marker shape per language.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Randomly subsample at most N rows (useful for large train sets).")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--log_file", type=str, default="./visualize.log",
                        help="Optional path to a log file. If set, logs are written to both stdout and the file.")
    parser.add_argument("--metrics_file", type=str, default=None,
                        help="Path to a metrics CSV (epoch, train_loss, …). If set, training curves are plotted and the script exits.")
    parser.add_argument("--curves_output_dir", type=str, default=None,
                        help="Directory to save curve PNGs. Defaults to the directory of --metrics_file.")
    args = parser.parse_args()

    if args.log_file:
        setup_file_logging(args.log_file)

    if args.metrics_file:
        save_dir = args.curves_output_dir or os.path.dirname(os.path.abspath(args.metrics_file))
        plot_training_curves(args.metrics_file, save_dir)
        return

    config_path = args.config_dir
    if not os.path.exists(config_path):
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")


    if args.data_file:
        data_path = args.data_file
    else:
        perplexity_model_name_save = (
            config.get('data', {})
            .get('perplexity_model', 'model')
            .rsplit('/', 1)[-1]
            .replace('.', '_')
            .replace('-', '_')
            .lower()
        )
        data_dir = config["data"]["data_dir"]
        filename = f"train_processed_{perplexity_model_name_save}.parquet" if args.split == "train" else f"val_processed_{perplexity_model_name_save}.parquet"
        data_path = os.path.join(data_dir, filename)

    logger.info(f"Loading data from: {data_path}")
    df = pd.read_parquet(data_path)

    required = {"code", "label", "agnostic_features"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        logger.error(f"Parquet file missing columns: {missing}")
        sys.exit(1)

    df = df.dropna(subset=["label"]).reset_index(drop=True)

    if args.max_samples and len(df) > args.max_samples:
        logger.info(f"Subsampling {args.max_samples} rows from {len(df)}...")
        df = df.sample(n=args.max_samples, random_state=42).reset_index(drop=True)

    languages = None
    if args.color_by_language and "language" in df.columns:
        languages = df["language"].astype(str).values

    logger.info("Loading tokenizer and model...")
    hf_model_name = get_model_name(config)
    model_cfg = config.get("model", {})
    model_type = model_cfg.get("model_type", "hybrid")
    use_fast = model_type != "hybrid"
    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name, use_fast=use_fast)
    except Exception as e:
        print(f"[Tokenizer load failed] {e}")
        print("Retrying with slow tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name, use_fast=False)
    model = build_model(config)

    weights_path = os.path.join(args.checkpoint_dir, "model_state.bin")
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f"Missing keys (will use random init): {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys (ignored): {unexpected}")
    model.to(device)
    model.eval()

    max_length = config["data"]["max_length"]
    dataset = AgnosticDataset(df, tokenizer, max_length=max_length, is_train=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    emb_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "Task_A_Embeddings")
    os.makedirs(emb_dir, exist_ok=True)
    emb_path = os.path.join(emb_dir, f"embeddings_{args.split}_{config['model']['base_model'].split('/')[-1]}.npz")

    if os.path.exists(emb_path):
        logger.info(f"Embeddings already exist, loading from {emb_path}")
        data = np.load(emb_path)
        embeddings, labels = data["embeddings"], data["labels"]
    else:
        embeddings, labels = extract_embeddings(model, dataloader, device, config)
        np.savez_compressed(emb_path, embeddings=embeddings, labels=labels)
        logger.info(f"Embeddings saved to {emb_path}")


    output_path = args.output or os.path.join(args.checkpoint_dir, f"tsne_{args.split}.png")
    subtitle = {
        'val': 'validation set',
        'train': 'training set'
    }
    title = (
        f"t-SNE Embedding Visualization — {subtitle[args.split].title()}  ({len(labels):,} samples)\n"
        f"Model: {hf_model_name}   |   use_agnostic_features: {config['data']['use_agnostic_features']}"
    )

    plot_tsne(
        embeddings=embeddings,
        labels=labels,
        save_path=output_path,
        title=title,
        label_names=LABEL_NAMES,
        perplexity=args.perplexity,
        n_iter=args.n_iter,
        languages=languages,
    )


if __name__ == "__main__":
    main()
