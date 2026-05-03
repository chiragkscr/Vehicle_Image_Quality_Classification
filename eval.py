import argparse
import os
import json
import torch
import numpy as np
from PIL import Image
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

from model import MultiLabelModel
from transforms import get_val_transforms


CLASSES = ["good", "blur", "low_light"]


def load_model(path, device):
    model = MultiLabelModel().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model



def normalize(name):
    return name.lower().replace("-", "_").replace(" ", "_")


def match_label(folder_name):
    name = normalize(folder_name)

    if "good" in name:
        return "good"
    elif "blur" in name:
        return "blur"
    elif "low" in name or "dark" in name:
        return "low_light"
    else:
        return None


def extract_label(path):
    parts = path.split(os.sep)

    
    for p in reversed(parts):
        label = match_label(p)
        if label is not None:
            return label

    return None



def get_images(root):
    data = []
    counts = {c: 0 for c in CLASSES}

    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            full_path = os.path.join(dirpath, f)
            label = extract_label(full_path)

            if label is None:
                continue

            data.append((full_path, label))
            counts[label] += 1

    print("\n[INFO] Dataset distribution:")
    for k, v in counts.items():
        print(f"  {k}: {v}")

    print(f"[INFO] Total images used: {len(data)}\n")

    return data



def predict(model, img, device, threshold):
    with torch.no_grad():
        out = model(img.to(device))
        probs = torch.sigmoid(out)[0].cpu().numpy()

    if probs.max() < threshold:
        return "good"
    else:
        return ["blur", "low_light"][int(np.argmax(probs))]



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out_cm", default="cm.png")
    parser.add_argument("--out_metrics", default="metrics.json")

    args = parser.parse_args()

    device = torch.device("cpu")
    transform = get_val_transforms()

    model = load_model(args.model, device)

    data = get_images(args.data)

    if len(data) == 0:
        raise RuntimeError("No images found. Check folder naming.")

    y_true, y_pred = [], []
    skipped = 0

    for path, true_label in data:
        try:
            image = Image.open(path).convert("RGB")
        except:
            skipped += 1
            continue

        img = transform(image).unsqueeze(0)
        pred_label = predict(model, img, device, args.threshold)

        y_true.append(true_label)
        y_pred.append(pred_label)

    print(f"[INFO] Used samples: {len(y_true)} | Skipped: {skipped}")

    if len(y_true) == 0:
        raise RuntimeError("All images failed.")


    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)

    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES,
                yticklabels=CLASSES)

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix (th={args.threshold})")
    plt.tight_layout()
    plt.savefig(args.out_cm)

    print(f"[INFO] Confusion matrix saved → {args.out_cm}")

    precision = precision_score(y_true, y_pred, labels=CLASSES, average=None, zero_division=0)
    recall    = recall_score(y_true, y_pred, labels=CLASSES, average=None, zero_division=0)
    f1        = f1_score(y_true, y_pred, labels=CLASSES, average=None, zero_division=0)

    metrics = {}

    for i, cls in enumerate(CLASSES):
        metrics[cls] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i])
        }


    metrics["overall"] = {
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    }


    with open(args.out_metrics, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"[INFO] Metrics saved → {args.out_metrics}")

    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, labels=CLASSES, zero_division=0))


if __name__ == "__main__":
    main()