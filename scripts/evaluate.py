#!/usr/bin/env python3
"""
Evaluate a trained behavior recognition model on labeled data and visualize metrics.

Usage:
    python scripts/evaluate.py --checkpoint outputs/checkpoints/last.ckpt --config configs/config.yaml --split val

This script intentionally refuses to run on unlabeled external splits. Evaluation is only
performed on datasets that include ground-truth annotations (train/val/test from the labeled data).
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    multilabel_confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
)
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.data.dataset import MABeDataModule, PrecomputedWindowDataset
from src.models.lightning_module import BehaviorRecognitionModule
from src.utils.metrics import per_class_statistics


def load_config(config_path: Path, overrides: Dict = None) -> Dict:
    """
    Load YAML config and resolve ${...} references via OmegaConf so derived paths work on Windows.
    """
    conf = OmegaConf.load(config_path)
    if overrides:
        for key, value in overrides.items():
            OmegaConf.update(conf, key, value, merge=True)
    return OmegaConf.to_container(conf, resolve=True)


def prepare_dataloader(config: Dict, split: str) -> Tuple[DataLoader, List[str]]:
    """
    Build a dataloader for a labeled split (train/val/test) using the labeled data.
    """
    if split not in ("train", "val", "test"):
        raise ValueError("Only 'train', 'val', and 'test' splits are supported for evaluation.")

    dm = MABeDataModule(
        data_dir=config["paths"]["data_dir"],
        behaviors=config.get("behaviors", None),
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        window_size=config["data"]["window_size"],
        stride=config["data"]["stride"],
        target_fps=config["data"]["target_fps"],
        val_split=config["data"].get("val_split", 0.2),
        test_split=config["data"].get("test_split", 0.1),
        tracking_cache_size=config["data"].get("tracking_cache_size", 4),
        annotation_cache_size=config["data"].get("annotation_cache_size", 8),
        use_precomputed=config["data"].get("use_precomputed", False),
        precomputed_dir=config["data"].get("precomputed_dir", None),
    )

    dm.setup(
        stage="fit",
        use_precomputed=config["data"].get("use_precomputed", False),
        precomputed_dir=config["data"].get("precomputed_dir", None),
    )

    if split == "val":
        dataset = dm.val_dataset
    elif split == "test":
        dataset = dm.test_dataset
    else:
        dataset = dm.train_dataset
        # Disable augmentation for evaluation.
        if hasattr(dataset, "augment"):
            dataset.augment = False
        if isinstance(dataset, PrecomputedWindowDataset):
            dataset.apply_augment = False

    if dataset is None:
        raise ValueError(f"No dataset found for split '{split}'.")

    # Guardrail: ensure annotations exist so we never evaluate on unlabeled/test data.
    if getattr(dataset, "annotation_dir", None) is None and not isinstance(dataset, PrecomputedWindowDataset):
        raise ValueError(f"Split '{split}' does not include annotations; refusing to evaluate unlabeled data.")

    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
        drop_last=False,
        prefetch_factor=config["training"].get("prefetch_factor", 4) if config["training"]["num_workers"] > 0 else None,
        persistent_workers=config["training"]["num_workers"] > 0,
    )

    behaviors = dataset.behaviors if hasattr(dataset, "behaviors") and dataset.behaviors else dm.behaviors
    if behaviors is None:
        raise ValueError("Behavior vocabulary is missing; ensure config.behaviors is set or training data is available.")

    return dataloader, behaviors


def collect_predictions(
    model: BehaviorRecognitionModule,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run model on dataloader and return flattened (frames x classes) arrays for labels and probabilities.
    """
    all_probs = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)
            mask = batch.get("valid_mask")
            if mask is not None:
                mask = mask.to(device)

            logits = model(features, mask)
            probs = torch.sigmoid(logits)

            # Use reshape to handle potential non-contiguous tensors from model outputs.
            probs_flat = probs.reshape(-1, probs.shape[-1])
            labels_flat = labels.reshape(-1, labels.shape[-1])

            if mask is not None:
                mask_flat = mask.reshape(-1) > 0.5
                probs_flat = probs_flat[mask_flat]
                labels_flat = labels_flat[mask_flat]

            all_probs.append(probs_flat.cpu())
            all_labels.append(labels_flat.cpu())

    y_probs = torch.cat(all_probs, dim=0).numpy()
    y_true = torch.cat(all_labels, dim=0).numpy()

    return y_true, y_probs


def compute_metric_summary(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    behaviors: List[str],
    threshold: float,
) -> Tuple[Dict, np.ndarray, np.ndarray, str]:
    """
    Compute macro/micro metrics, per-class stats, and confusion matrices.
    """
    binary_preds = (y_probs >= threshold).astype(int)

    macro_f1 = f1_score(y_true, binary_preds, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, binary_preds, average="micro", zero_division=0)
    macro_precision = precision_score(y_true, binary_preds, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, binary_preds, average="macro", zero_division=0)

    try:
        roc_auc_macro = roc_auc_score(y_true, y_probs, average="macro")
    except ValueError:
        roc_auc_macro = None

    try:
        roc_auc_micro = roc_auc_score(y_true, y_probs, average="micro")
    except ValueError:
        roc_auc_micro = None

    try:
        pr_auc_macro = average_precision_score(y_true, y_probs, average="macro")
        pr_auc_micro = average_precision_score(y_true, y_probs, average="micro")
    except ValueError:
        pr_auc_macro = None
        pr_auc_micro = None

    per_class_f1 = f1_score(y_true, binary_preds, average=None, zero_division=0)
    per_class_prec = precision_score(y_true, binary_preds, average=None, zero_division=0)
    per_class_rec = recall_score(y_true, binary_preds, average=None, zero_division=0)

    per_class_auc = []
    for idx in range(len(behaviors)):
        try:
            per_class_auc.append(roc_auc_score(y_true[:, idx], y_probs[:, idx]))
        except ValueError:
            per_class_auc.append(float("nan"))

    mcm = multilabel_confusion_matrix(y_true, binary_preds)

    # Confusion matrix for primary (argmax) label, skipping unlabeled frames
    labeled_mask = y_true.sum(axis=1) > 0
    dominant_cm = None
    if labeled_mask.any():
        true_primary = y_true[labeled_mask].argmax(axis=1)
        pred_primary = y_probs[labeled_mask].argmax(axis=1)
        dominant_cm = confusion_matrix(true_primary, pred_primary, labels=np.arange(len(behaviors)))

    class_report = classification_report(
        y_true,
        binary_preds,
        target_names=behaviors,
        zero_division=0,
    )

    per_class_stats = per_class_statistics(y_probs, y_true, behaviors, threshold)

    metrics = {
        "samples": int(y_true.shape[0]),
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_roc_auc": roc_auc_macro,
        "micro_roc_auc": roc_auc_micro,
        "macro_pr_auc": pr_auc_macro,
        "micro_pr_auc": pr_auc_micro,
        "threshold": threshold,
        "per_class": {},
    }

    for idx, name in enumerate(behaviors):
        metrics["per_class"][name] = {
            "f1": float(per_class_f1[idx]),
            "precision": float(per_class_prec[idx]),
            "recall": float(per_class_rec[idx]),
            "roc_auc": float(per_class_auc[idx]) if not np.isnan(per_class_auc[idx]) else None,
            "support": int(y_true[:, idx].sum()),
            **per_class_stats.get(name, {}),
        }

    return metrics, mcm, dominant_cm, class_report


def plot_per_class_confusion(conf_matrix: np.ndarray, behaviors: List[str], output_path: Path):
    """
    Visualize per-class TP/FP/FN/TN counts as a heatmap.
    """
    matrix = np.stack(
        [
            conf_matrix[:, 1, 1],  # TP
            conf_matrix[:, 0, 1],  # FP
            conf_matrix[:, 1, 0],  # FN
            conf_matrix[:, 0, 0],  # TN
        ],
        axis=1,
    )
    column_labels = ["TP", "FP", "FN", "TN"]

    fig_height = max(6, len(behaviors) * 0.25)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    im = ax.imshow(matrix, aspect="auto", cmap="Blues")
    ax.set_xticks(np.arange(len(column_labels)))
    ax.set_xticklabels(column_labels)
    ax.set_yticks(np.arange(len(behaviors)))
    ax.set_yticklabels(behaviors)
    ax.set_xlabel("Counts per class")
    ax.set_ylabel("Behavior")
    fig.colorbar(im, ax=ax, label="Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_primary_confusion(cm: np.ndarray, behaviors: List[str], output_path: Path):
    """
    Plot confusion matrix using the dominant label per frame (only frames with at least one label).
    """
    if cm is None:
        return

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation="nearest", cmap="OrRd")
    ax.set_title("Primary-label confusion (frames with at least one annotation)")
    ax.set_xticks(np.arange(len(behaviors)))
    ax.set_xticklabels(behaviors, rotation=90)
    ax.set_yticks(np.arange(len(behaviors)))
    ax.set_yticklabels(behaviors)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax, label="Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_per_class_f1(per_class_f1: Dict[str, Dict[str, float]], output_path: Path):
    """
    Plot per-class F1 scores as a horizontal bar chart.
    """
    labels = list(per_class_f1.keys())
    scores = [per_class_f1[name]["f1"] for name in labels]
    order = np.argsort(scores)
    labels = [labels[i] for i in order]
    scores = [scores[i] for i in order]

    fig_height = max(6, len(labels) * 0.25)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.barh(labels, scores, color="#1f77b4")
    ax.set_xlabel("F1 score")
    ax.set_xlim(0, 1)
    ax.set_title("Per-class F1 (sorted)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_roc_curves(y_true: np.ndarray, y_probs: np.ndarray, output_path: Path):
    """
    Plot micro and macro ROC curves.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    try:
        fpr_micro, tpr_micro, _ = roc_curve(y_true.ravel(), y_probs.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)
        ax.plot(fpr_micro, tpr_micro, label=f"micro-average (AUC = {roc_auc_micro:.3f})", color="deeppink", lw=2)
    except ValueError:
        fpr_micro = tpr_micro = None

    # Macro-average curve across valid classes
    fpr_dict = {}
    tpr_dict = {}
    for i in range(y_true.shape[1]):
        if np.unique(y_true[:, i]).size < 2:
            continue
        fpr_dict[i], tpr_dict[i], _ = roc_curve(y_true[:, i], y_probs[:, i])

    if fpr_dict:
        all_fpr = np.unique(np.concatenate(list(fpr_dict.values())))
        mean_tpr = np.zeros_like(all_fpr)
        for i in fpr_dict:
            mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
        mean_tpr /= len(fpr_dict)
        roc_auc_macro = auc(all_fpr, mean_tpr)
        ax.plot(all_fpr, mean_tpr, label=f"macro-average (AUC = {roc_auc_macro:.3f})", color="navy", lw=2)

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained model on labeled data.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.ckpt).")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Dataset split to evaluate.")
    parser.add_argument("--threshold", type=float, default=None, help="Override decision threshold for metrics.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to write evaluation artifacts. Defaults to <config.paths.output_dir>/eval.",
    )
    parser.add_argument("--device", type=str, default=None, help="Device to use (e.g., cuda, cpu). Auto-detected if unset.")
    return parser.parse_args()


def main():
    args = parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] Using {device}")

    threshold = args.threshold if args.threshold is not None else config.get("evaluation", {}).get("threshold", 0.5)

    dataloader, behaviors = prepare_dataloader(config, args.split)
    print(f"[data] Loaded {len(dataloader.dataset)} windows from '{args.split}' split with {len(behaviors)} classes.")

    print(f"[model] Loading checkpoint from {args.checkpoint}")
    # Torch 2.6 defaults to weights_only=True; explicitly allow legacy checkpoints and safe globals.
    safe_classes = [np.core.multiarray.scalar]
    try:
        torch.serialization.add_safe_globals(safe_classes)
    except Exception:
        pass

    try:
        safe_ctx = torch.serialization.safe_globals(safe_classes)  # type: ignore[attr-defined]
    except Exception:
        from contextlib import nullcontext
        safe_ctx = nullcontext()

    with safe_ctx:
        model = BehaviorRecognitionModule.load_from_checkpoint(
            args.checkpoint,
            map_location=device,
            weights_only=False,
            strict=True,
        )
    model.to(device)

    print("[eval] Running inference...")
    y_true, y_probs = collect_predictions(model, dataloader, device)

    metrics, conf_matrix, dominant_cm, class_report = compute_metric_summary(
        y_true,
        y_probs,
        behaviors,
        threshold,
    )

    output_root = Path(args.output_dir) if args.output_dir else Path(config["paths"]["output_dir"]) / "eval"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = output_root / f"{args.split}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics and reports
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(run_dir / "classification_report.txt", "w") as f:
        f.write(class_report)

    # Visualizations
    plot_per_class_confusion(conf_matrix, behaviors, run_dir / "confusion_per_class.png")
    plot_primary_confusion(dominant_cm, behaviors, run_dir / "confusion_primary_label.png")
    plot_per_class_f1(metrics["per_class"], run_dir / "per_class_f1.png")
    plot_roc_curves(y_true, y_probs, run_dir / "roc_curves.png")

    print(f"[done] Metrics and plots written to {run_dir}")
    print(f"macro F1: {metrics['macro_f1']:.4f} | micro F1: {metrics['micro_f1']:.4f}")
    if metrics["macro_roc_auc"] is not None:
        print(f"macro ROC-AUC: {metrics['macro_roc_auc']:.4f}")


if __name__ == "__main__":
    main()
