import torch
import random
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_model(model, dataloader, device, label_names=None, model_type="hybrid", use_agnostic_features=False):
    model.eval()
    preds_all = []
    labels_all = []
    total_loss = 0.0

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            if model_type == "droiddetect":
                if use_agnostic_features:
                    extra_features = batch["extra_features"].to(device, non_blocking=True)
                    output = model(input_ids=input_ids, attention_mask=mask, extra_features=extra_features)
                else:
                    output = model(input_ids=input_ids, attention_mask=mask)
                logits = output["logits"]
            else:
                feats = batch["extra_features"].to(device, non_blocking=True)
                logits, _, _ = model(input_ids, mask, feats, labels=None)

            total_loss += criterion(logits, labels).item()

            preds_all.extend(torch.argmax(logits, dim=1).cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    accuracy = accuracy_score(labels_all, preds_all)
    f1 = f1_score(labels_all, preds_all, average='macro')

    unique_labels = sorted(set(labels_all))
    target_names = [label_names[i] for i in unique_labels] if label_names else None
    report = classification_report(labels_all, preds_all, target_names=target_names, digits=4, zero_division=0)

    return {
        "loss": total_loss / len(dataloader),
        "accuracy": accuracy,
        "f1_macro": f1,
    }, preds_all, labels_all, report
