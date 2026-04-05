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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import transformers.utils.import_utils
import transformers.modeling_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda *args, **kwargs: True
transformers.modeling_utils.check_torch_load_is_safe = lambda *args, **kwargs: True
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from models.model import build_model, get_label_names
from dataset.dataset import AgnosticDataset, SimpleTextDataset
from dataset.preprocess_features import AgnosticFeatureExtractor
from utils import plot_confusion_matrix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("inference.log", mode="a", encoding="utf-8")
    ]
)
logging.FileHandler("inference.log", mode="w", encoding="utf-8")
logger = logging.getLogger(__name__)


def prepare_test_data(test_path, config, device):
    model_type = config.get("model", {}).get("model_type", "hybrid")
    logger.info(f"Checking data: {test_path}")
    df = pd.read_parquet(test_path)

    use_agnostic = config.get("data", {}).get("use_agnostic_features", False)
    if model_type == "droiddetect" and not use_agnostic:
        return df

    if 'agnostic_features' in df.columns:
        logger.info("Features already present in dataset.")
        return df
    
    perplexity_model_name_save = (
        config.get('data', {})
        .get('perplexity_model', 'model')
        .rsplit('/', 1)[-1]
        .replace('.', '_')
        .replace('-', '_')
        .lower()
    )
    cache_path = test_path.replace(".parquet", f"_processed_{perplexity_model_name_save}.parquet").replace("Task_A", "Task_A_Processed")
    if os.path.exists(cache_path):
        logger.info(f"Found cached processed file: {cache_path}")
        return pd.read_parquet(cache_path)

    logger.info("Initializing Feature Extractor (this takes time)...")
    extractor = AgnosticFeatureExtractor(config, str(device))

    features_list = []
    for code in tqdm(df['code'], desc="Feature Extraction"):
        try:
            features_list.append(extractor.extract_all(code))
        except Exception:
            features_list.append([0.0] * 11)

    df['agnostic_features'] = features_list
    logger.info(f"Saving processed test data to {cache_path}")
    df.to_parquet(cache_path)

    del extractor
    torch.cuda.empty_cache()

    return df


def run_inference(args):
    gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip()] if args.gpu_ids else [0]
    primary_id = gpu_ids[0]
    torch.cuda.set_device(primary_id)
    device = torch.device(f"cuda:{primary_id}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Inference Device: {device} | GPUs: {gpu_ids}")

    ckpt_config_path = os.path.join(args.checkpoint_dir, "config.yaml")
    config_path = ckpt_config_path if os.path.exists(ckpt_config_path) else args.config
    if not os.path.exists(config_path):
        logger.error(f"Config not found: tried {ckpt_config_path} and {args.config}")
        sys.exit(1)

    logger.info(f"Loading config from: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_type = config.get("model", {}).get("model_type", "hybrid")
    label_names = get_label_names()
    logger.info(f"Model type: {model_type} | Classes: {label_names}")

    test_df = prepare_test_data(args.test_file, config, device)

    has_labels = 'label' in test_df.columns and test_df['label'].notna().any()
    if has_labels:
        test_df = test_df.dropna(subset=['label']).reset_index(drop=True)
    else:
        test_df['label'] = 0

    logger.info("Loading Model & Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir, use_fast=(model_type != "hybrid"))

    model = build_model(config)
    state_dict = torch.load(os.path.join(args.checkpoint_dir, "model_state.bin"), map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    if len(gpu_ids) > 1 and torch.cuda.is_available():
        logger.info(f"Using DataParallel across GPUs: {gpu_ids}")
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    model.eval()

    max_length = config["data"]["max_length"]
    use_agnostic = config.get("data", {}).get("use_agnostic_features", False)
    if model_type == "droiddetect":
        dataset = SimpleTextDataset(test_df, tokenizer, max_length=max_length, is_train=False, is_use_agnostic=use_agnostic)
    else:
        dataset = AgnosticDataset(test_df, tokenizer, max_length=max_length, is_train=False)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    logger.info("Running Prediction...")
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inferencing"):
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)

            if model_type == "droiddetect":
                extra_features = batch["extra_features"].to(device) if "extra_features" in batch else None
                output = model(input_ids=input_ids, attention_mask=mask, extra_features=extra_features)
                logits = output["logits"] if isinstance(output, dict) else output[0]
            else:
                feats = batch["extra_features"].to(device)
                out = model(input_ids, mask, feats, labels=None)
                logits = out[0] if isinstance(out, (tuple, list)) else out

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            if has_labels:
                all_labels.extend(batch["labels"].numpy())

    if args.binary and len(label_names) > 2:
        logger.info("Remapping to binary: 0=HUMAN, 1=AI (MACHINE_*/ADVERSARIAL)")
        all_preds = [0 if p == 0 else 1 for p in all_preds]
        if has_labels:
            all_labels = [0 if l == 0 else 1 for l in all_labels]
        label_names = ["HUMAN_GENERATED", "AI_GENERATED"]

    if has_labels:
        print("\n" + "="*60)
        print("TEST SET EVALUATION REPORT".center(60))
        print("="*60)

        acc = accuracy_score(all_labels, all_preds)
        print(f"\nAccuracy: {acc:.4f}")

        unique = sorted(set(all_labels))
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds,
                                    target_names=[label_names[i] for i in unique], digits=4))

        print("\nConfusion Matrix:")
        print(confusion_matrix(all_labels, all_preds))

        cm_path = args.test_file.replace(".parquet", "_confusion_matrix.png")
        plot_confusion_matrix(all_labels, all_preds, label_names, cm_path)
        logger.info(f"Confusion matrix saved to {cm_path}")

        test_df['pred'] = all_preds
        errors = test_df[test_df['label'] != test_df['pred']]
        if not errors.empty:
            error_path = args.test_file.replace(".parquet", "_errors.csv")
            cols = ['code', 'label', 'pred']
            if 'language' in errors.columns:
                cols.append('language')
            errors[cols].head(100).to_csv(error_path, index=False)
            logger.info(f"Saved first {len(errors)} errors to {error_path}")

    else:
        print("\nInference Complete. Saving predictions...")
        test_df['prediction'] = all_preds
        test_df['prediction_label'] = [label_names[p] for p in all_preds]

        probs_array = np.array(all_probs)
        for i, name in enumerate(label_names):
            if i < probs_array.shape[1]:
                test_df[f'prob_{name.lower()}'] = probs_array[:, i]

        out_path = args.test_file.replace(".parquet", "_predictions.csv")
        save_cols = ['code', 'prediction', 'prediction_label'] + \
                    [f'prob_{n.lower()}' for n in label_names if f'prob_{n.lower()}' in test_df.columns]
        test_df[save_cols].to_csv(out_path, index=False)
        logger.info(f"Predictions saved to {out_path}")

    if "ID" in test_df.columns:
        submission_path = config["data"]["submission_path"]
        os.makedirs(os.path.dirname(submission_path), exist_ok=True)
        submission = pd.DataFrame({"ID": test_df["ID"].tolist(), "label": all_preds})
        submission.to_csv(config["data"]["submission_path"], index=False)
        print(f"Submission saved to {config['data']['submission_path']}")
        print(f"  Shape: {submission.shape}")
        print(f"  Label distribution:\n{submission['label'].value_counts()}")
        print(submission.head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config_hybrid.yaml")
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu_ids", type=str, default="0",
                        help="Comma-separated GPU IDs to use, e.g. '0,1,2,3'")
    parser.add_argument("--binary", action="store_true",
                        help="Remap multi-class predictions to binary: 0=HUMAN, 1=AI")
    args = parser.parse_args()

    run_inference(args)
