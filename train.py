import os
import sys
import transformers.utils.import_utils
import transformers.modeling_utils

transformers.utils.import_utils.check_torch_load_is_safe = lambda *args, **kwargs: True
transformers.modeling_utils.check_torch_load_is_safe = lambda *args, **kwargs: True

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import yaml
import torch
import argparse
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from transformers import AutoTokenizer
from dotenv import load_dotenv
from comet_ml import Experiment
from sklearn.metrics import confusion_matrix
from pytorch_metric_learning import losses

from models.model import build_model, get_model_name, get_label_names
from dataset.dataset import load_data
from utils.utils import set_seed, evaluate_model
from utils import plot_confusion_matrix

torch.backends.cudnn.benchmark = True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("train.log", mode="a", encoding="utf-8")
    ]
)
logging.FileHandler("train.log", mode="w", encoding="utf-8")
logger = logging.getLogger(__name__)


class ConsoleUX:
    @staticmethod
    def print_banner(text):
        print(f"\n{'-'*60}\n{text.center(60)}\n{'-'*60}")

    @staticmethod
    def log_metrics(stage, metrics):
        log_str = f"[{stage}] "
        priority_keys = ["loss", "f1_macro", "acc", "task_loss", "supcon_loss"]

        for k in priority_keys:
            if k in metrics:
                log_str += f"{k}: {metrics[k]:.4f} | "

        for k, v in metrics.items():
            if k not in priority_keys and isinstance(v, float):
                log_str += f"{k}: {v:.4f} | "

        logger.info(log_str.strip(" | "))


def save_checkpoint(model, tokenizer, path, epoch, metrics, config,
                    optimizer=None, scheduler=None, scaler=None,
                    best_f1=None, patience_counter=None):
    os.makedirs(path, exist_ok=True)
    logger.info(f"Saving checkpoint to {path}...")

    tokenizer.save_pretrained(path)

    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(model_to_save.state_dict(), os.path.join(path, "model_state.bin"))

    with open(os.path.join(path, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    with open(os.path.join(path, "training_meta.yaml"), "w") as f:
        yaml.dump({"epoch": epoch, "metrics": metrics}, f)

    if optimizer is not None:
        training_state = {
            "epoch": epoch,
            "best_f1": best_f1,
            "patience_counter": patience_counter,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "scaler_state_dict": scaler.state_dict() if scaler else None,
        }
        torch.save(training_state, os.path.join(path, "training_state.bin"))


def load_checkpoint(resume_dir, model, optimizer=None, scheduler=None, scaler=None, device=None):
    model_path = os.path.join(resume_dir, "model_state.bin")
    state_path = os.path.join(resume_dir, "training_state.bin")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model_state.bin found in {resume_dir}")

    logger.info(f"Loading model weights from {model_path}")
    map_location = device if device else "cpu"
    state_dict = torch.load(model_path, map_location=map_location, weights_only=True)
    model_to_load = model.module if hasattr(model, 'module') else model
    model_to_load.load_state_dict(state_dict)

    start_epoch = 0
    best_f1 = 0.0
    patience_counter = 0

    if os.path.exists(state_path):
        logger.info(f"Loading training state from {state_path}")
        training_state = torch.load(state_path, map_location=map_location, weights_only=True)
        start_epoch = training_state["epoch"] + 1
        best_f1 = training_state.get("best_f1", 0.0)
        patience_counter = training_state.get("patience_counter", 0)
        if optimizer and training_state.get("optimizer_state_dict"):
            optimizer.load_state_dict(training_state["optimizer_state_dict"])
        if scheduler and training_state.get("scheduler_state_dict"):
            scheduler.load_state_dict(training_state["scheduler_state_dict"])
        if scaler and training_state.get("scaler_state_dict"):
            scaler.load_state_dict(training_state["scaler_state_dict"])
        logger.info(f"Resuming from epoch {start_epoch}, best_f1={best_f1:.4f}, patience={patience_counter}")
    else:
        logger.warning(f"No training_state.bin in {resume_dir}. Starting epoch 0 with loaded weights only.")

    return start_epoch, best_f1, patience_counter


def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, device,
                    epoch_idx, acc_steps=1, supcon_fn=None, model_type="hybrid", is_use_agnostic=False,
                    amp_dtype=torch.float16, amp_enabled=True):
    model.train()

    tracker = {"loss": 0.0, "task_loss": 0.0, "supcon_loss": 0.0, "correct": 0, "total": 0}
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(dataloader, desc=f"Train Ep {epoch_idx+1}", leave=False, dynamic_ncols=True)

    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with autocast(device_type='cuda', dtype=amp_dtype, enabled=amp_enabled):
            if model_type == "droiddetect":
                if is_use_agnostic:
                    extra_features = batch["extra_features"].float().to(device, non_blocking=True)
                    output = model(input_ids=input_ids, attention_mask=attention_mask, extra_features=extra_features, labels=labels)
                else:
                    output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                logits = output["logits"]
                task_loss = output.get("cross_entropy_loss", output["loss"])
                supcon_loss = output.get("contrastive_loss", torch.tensor(0.0, device=device))
                total_loss = output["loss"].mean() / acc_steps
            else:
                feats = batch["extra_features"].float().to(device, non_blocking=True)
                logits, task_loss, combined_features = model(input_ids, attention_mask, feats, labels=labels)
                supcon_loss = torch.tensor(0.0, device=device)
                if supcon_fn is not None:
                    features_norm = torch.nn.functional.normalize(combined_features, dim=1, p=2, eps=1e-12)
                    supcon_loss = supcon_fn(features_norm, labels)
                total_loss = (task_loss.mean() + 0.05 * supcon_loss) / acc_steps

        scaler.scale(total_loss).backward()

        if (step + 1) % acc_steps == 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        preds = logits.detach().argmax(dim=1)
        tracker["correct"] += (preds == labels).sum().item()
        tracker["total"] += labels.size(0)

        current_loss = total_loss.item() * acc_steps
        tracker["loss"] += current_loss
        tracker["task_loss"] += task_loss.mean().item()
        tracker["supcon_loss"] += supcon_loss.mean().item() if isinstance(supcon_loss, torch.Tensor) else float(supcon_loss)

        pbar.set_postfix({
            "Loss": f"{current_loss:.3f}",
            "Aux": f"{supcon_loss.mean().item():.3f}" if isinstance(supcon_loss, torch.Tensor) else "0.0"
        })

    num_batches = len(dataloader)
    result = {k: v / num_batches for k, v in tracker.items() if k not in ("correct", "total")}
    result["accuracy"] = tracker["correct"] / tracker["total"] if tracker["total"] > 0 else 0.0
    return result


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="SemEval Task A - Training")
    parser.add_argument("--config", type=str, default="./config/config_droiddetect.yaml")
    parser.add_argument("--result-dir", type=str, default=None)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--gpu-ids", type=str, default=None,
                        help="Comma-separated GPU IDs for DataParallel, e.g. '0,1,2,3'. Overrides --device-id.")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    ConsoleUX.print_banner("SemEval Task 13 - Subtask A [Generalization]")

    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Configuration:\n{config}")
    common_cfg = config.get("common", {})
    train_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    model_type = model_cfg.get("model_type", "hybrid")
    data_cfg = config.get("data", {})

    logger.info(f"Model type: {model_type}")
    logger.info(f"Using agnostic features: {data_cfg.get('use_agnostic_features', False)}")

    if args.result_dir:
        train_cfg["checkpoint_dir"] = args.result_dir

    set_seed(common_cfg.get("seed", 42))

    if args.gpu_ids is not None:
        gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
    else:
        gpu_ids = [args.device_id]

    device = torch.device(f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Primary device: {device} | GPUs: {gpu_ids}")

    api_key = os.getenv("COMET_API_KEY")
    experiment = None
    if api_key:
        try:
            experiment = Experiment(
                api_key=api_key,
                project_name=common_cfg.get("project_name", "semeval-task-a"),
                auto_metric_logging=False
            )
            experiment.log_parameters(config)
            experiment.add_tag(f"TaskA_{model_type}")
        except Exception as e:
            logger.warning(f"Comet Init Failed: {e}. Proceeding without it.")

    hf_model_name = get_model_name(config)
    use_fast = model_type != "hybrid"
    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name, use_fast=use_fast)
    except Exception as e:
        print(f"[Tokenizer load failed] {e}")
        print("Retrying with slow tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name, use_fast=False)
    tokenizer.model_max_length = data_cfg["max_length"]

    logger.info("Initializing Datasets...")
    train_ds, val_ds = load_data(config, tokenizer)

    train_dl = DataLoader(
        train_ds, batch_size=train_cfg["batch_size"], shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    val_dl = DataLoader(
        val_ds, batch_size=train_cfg["batch_size"], shuffle=False,
        num_workers=4, pin_memory=True
    )

    model = build_model(config)
    model = model.float().to(device)
    # model.to(device)

    if len(gpu_ids) > 1:
        logger.info(f"Wrapping model with DataParallel across GPUs: {gpu_ids}")
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    label_names = get_label_names()

    backbone_lr = float(train_cfg.get("backbone_lr", float(train_cfg["learning_rate"]) * 0.1))
    head_lr = float(train_cfg["learning_rate"])
    weight_decay = train_cfg.get("weight_decay", 0.01)

    m = model.module if hasattr(model, "module") else model
    if model_type == "droiddetect":
        param_groups = [
            {"params": m.text_encoder.parameters(), "lr": backbone_lr},
            {"params": m.text_projection.parameters(), "lr": head_lr},
            {"params": m.classifier.parameters(), "lr": head_lr},
        ]
        if data_cfg.get("use_agnostic_features", False):
            param_groups += [
                {"params": m.pooler.parameters(), "lr": head_lr},
                {"params": m.feature_encoder.parameters(), "lr": head_lr},
            ]
        optimizer = AdamW(param_groups, weight_decay=weight_decay)
    else:
        optimizer = AdamW([
            {"params": m.base_model.parameters(), "lr": backbone_lr},
            {"params": m.pooler.parameters(), "lr": head_lr},
            {"params": m.feature_encoder.parameters(), "lr": head_lr},
            {"params": m.classifier.parameters(), "lr": head_lr},
        ], weight_decay=weight_decay)

    supcon_fn = None
    if model_type == "hybrid" and train_cfg.get("use_supcon", False):
        supcon_temperature = train_cfg.get("supcon_temperature", 0.3)
        logger.info(f"Activating Supervised Contrastive Loss (temperature={supcon_temperature})...")
        supcon_fn = losses.SupConLoss(temperature=supcon_temperature).to(device)

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    amp_enabled = torch.cuda.is_available()
    logger.info(f"AMP dtype: {amp_dtype}, enabled: {amp_enabled}")
    scaler = GradScaler("cuda", enabled=(amp_dtype == torch.float16 and amp_enabled))
    acc_steps = train_cfg.get("gradient_accumulation_steps", 1)
    steps_per_epoch = len(train_dl) // acc_steps

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=steps_per_epoch * 3, T_mult=2, eta_min=1e-7
    )

    start_epoch = 0
    best_f1 = 0.0
    patience_counter = 0
    if args.resume:
        if not os.path.isdir(args.resume):
            logger.error(f"Resume path not found: {args.resume}")
            sys.exit(1)
        start_epoch, best_f1, patience_counter = load_checkpoint(
            args.resume, model, optimizer, scheduler, scaler, device
        )

    patience = train_cfg.get("early_stop_patience", 3)
    checkpoint_dir = train_cfg["checkpoint_dir"]

    os.makedirs(checkpoint_dir, exist_ok=True)
    csv_path = os.path.join(checkpoint_dir, "metrics.csv")
    metrics_rows = []
    logger.info(f"Metrics CSV: {csv_path}")
    logger.info(f"Starting Training for {train_cfg['num_epochs']} epochs...")

    for epoch in range(start_epoch, train_cfg["num_epochs"]):
        ConsoleUX.print_banner(f"Epoch {epoch+1}/{train_cfg['num_epochs']}")

        train_metrics = train_one_epoch(
            model, train_dl, optimizer, scheduler, scaler, device,
            epoch, acc_steps, supcon_fn, model_type=model_type,
            is_use_agnostic=data_cfg.get("use_agnostic_features", False),
            amp_dtype=amp_dtype, amp_enabled=amp_enabled,
        )
        ConsoleUX.log_metrics("Train", train_metrics)

        if experiment:
            experiment.log_metrics(train_metrics, prefix="Train", step=epoch)
            experiment.log_metric("lr", scheduler.get_last_lr()[0], step=epoch)

        val_metrics, val_preds, val_labels, report = evaluate_model(
            model, val_dl, device, label_names, model_type=model_type,
            use_agnostic_features=data_cfg.get("use_agnostic_features", False)
        )
        ConsoleUX.log_metrics("Val", val_metrics)
        logger.info(f"\n{report}")

        metrics_rows.append({
            "epoch":             epoch + 1,
            "train_loss":        round(train_metrics["loss"],        6),
            "train_task_loss":   round(train_metrics["task_loss"],   6),
            "train_supcon_loss": round(train_metrics["supcon_loss"], 6),
            "val_loss":          round(val_metrics["loss"],          6),
            "val_accuracy":      round(val_metrics["accuracy"],      6),
            "val_f1_macro":      round(val_metrics["f1_macro"],      6),
        })
        pd.DataFrame(metrics_rows).to_csv(csv_path, index=False)

        if experiment:
            experiment.log_metrics(val_metrics, prefix="Val", step=epoch)

        current_f1 = val_metrics["f1_macro"]

        if current_f1 > best_f1:
            best_f1 = current_f1
            patience_counter = 0
            logger.info(f"--> New Best F1: {best_f1:.4f}. Saving Model...")

            best_model_dir = os.path.join(checkpoint_dir, "best_model")
            save_checkpoint(model, tokenizer, best_model_dir, epoch, val_metrics, config)
            
            if model_type == "droiddetect":
                val_preds = [0 if p == 0 else 1 for p in val_preds]
                val_labels = [0 if l == 0 else 1 for l in val_labels]

            plot_confusion_matrix(
                val_labels, val_preds, label_names,
                os.path.join(best_model_dir, "confusion_matrix.png")
            )

            if experiment:
                cm = confusion_matrix(val_labels, val_preds)
                experiment.log_confusion_matrix(matrix=cm, labels=label_names, title="Best Model CM")
        else:
            patience_counter += 1
            logger.warning(f"--> No improvement. Patience: {patience_counter}/{patience}")

        last_ckpt_dir = os.path.join(checkpoint_dir, "last_checkpoint")
        save_checkpoint(
            model, tokenizer, last_ckpt_dir, epoch, val_metrics, config,
            optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            best_f1=best_f1, patience_counter=patience_counter,
        )

        if patience_counter >= patience:
            ConsoleUX.print_banner("EARLY STOPPING TRIGGERED")
            break

    if experiment:
        experiment.end()

    logger.info("Training Finished.")
