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
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
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
from src.utils.postprocessing import (
    aggregate_window_predictions,
    apply_nms,
    create_submission,
    extract_segments,
    merge_segments,
    BehaviorSegment,
)
from src.utils.kaggle_metric import mouse_fbeta


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


def _to_scalar(value: Any) -> Any:
    """Convert tensors/NumPy scalars to plain Python values."""
    if isinstance(value, torch.Tensor):
        value = value.item()
    if isinstance(value, np.generic):
        value = value.item()
    return value


def _format_mouse_id(mouse_id: Any, allow_self: bool = True) -> str:
    """Normalize mouse identifiers to submission format (mouseX or self)."""
    mouse_id = _to_scalar(mouse_id)
    if isinstance(mouse_id, str):
        cleaned = mouse_id.strip()
        if cleaned.lower().startswith("mouse"):
            return cleaned
        if cleaned.lstrip("-").isdigit():
            mouse_id = int(cleaned)
        else:
            return cleaned
    try:
        mouse_int = int(mouse_id)
    except (TypeError, ValueError):
        return str(mouse_id)

    if allow_self and mouse_int == -1:
        return "self"
    return f"mouse{mouse_int}"


def collect_predictions(
    model: BehaviorRecognitionModule,
    dataloader: DataLoader,
    device: torch.device,
    keep_windows: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Run model on dataloader and return flattened (frames x classes) arrays for labels and probabilities.
    Optionally collect window-level predictions with metadata for submission formatting.
    """
    all_probs = []
    all_labels = []
    window_predictions: List[Dict] = []

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

            if keep_windows:
                for i in range(features.shape[0]):
                    window_prob = probs[i].detach().cpu().numpy()
                    if mask is not None:
                        window_mask = mask[i].detach().cpu().numpy().reshape(-1, 1)
                        window_prob = window_prob * window_mask
                    window_predictions.append(
                        {
                            "video_id": _to_scalar(batch["video_id"][i]),
                            "agent_id": _to_scalar(batch["agent_id"][i]),
                            "target_id": _to_scalar(batch["target_id"][i]),
                            "start_frame": int(_to_scalar(batch["start_frame"][i])),
                            "probabilities": window_prob,
                        }
                    )

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

    return y_true, y_probs, window_predictions


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


def build_solution_df(config: Dict, split: str) -> Optional[pd.DataFrame]:
    """
    Build solution DataFrame from annotation files for Kaggle-compatible evaluation.

    Args:
        config: Configuration dictionary with paths
        split: Dataset split ('train', 'val', 'test')

    Returns:
        DataFrame with columns: video_id, agent_id, target_id, action,
        start_frame, stop_frame, lab_id, behaviors_labeled
        Returns None if annotations are not available.
    """
    data_dir = Path(config["paths"]["data_dir"])
    ann_dir = data_dir / "train_annotation"
    metadata_csv = data_dir / "train.csv"

    if not ann_dir.exists() or not metadata_csv.exists():
        return None

    metadata_df = pd.read_csv(metadata_csv)

    # Get video IDs for this split (we need to replicate the split logic)
    # For simplicity, we use all videos in metadata - caller should filter if needed
    all_rows = []

    for _, meta in metadata_df.iterrows():
        lab_id = meta["lab_id"]
        video_id = meta["video_id"]
        behaviors_labeled = meta.get("behaviors_labeled", "[]")

        ann_path = ann_dir / str(lab_id) / f"{video_id}.parquet"
        if not ann_path.exists():
            continue

        ann_df = pd.read_parquet(ann_path)
        for _, row in ann_df.iterrows():
            # Format agent/target as 'mouseX' to match submission format
            agent_id = f"mouse{row['agent_id']}"
            target_id = f"mouse{row['target_id']}" if row['target_id'] != row['agent_id'] else "self"

            all_rows.append({
                "video_id": video_id,
                "agent_id": agent_id,
                "target_id": target_id,
                "action": row["action"],
                "start_frame": row["start_frame"],
                "stop_frame": row["stop_frame"],
                "lab_id": lab_id,
                "behaviors_labeled": behaviors_labeled,
            })

    if not all_rows:
        return None

    return pd.DataFrame(all_rows)


def resolve_overlaps(segments: List[BehaviorSegment], min_duration: int = 5) -> List[BehaviorSegment]:
    """
    Resolve overlapping behaviors for the same agent-target pair.

    When two behaviors overlap for the same (video_id, agent_id, target_id),
    the latter behavior's start_frame is adjusted to begin right after the
    previous behavior's stop_frame.

    Args:
        segments: List of BehaviorSegment objects
        min_duration: Minimum duration for a valid segment

    Returns:
        List of BehaviorSegment objects with overlaps resolved
    """
    from collections import defaultdict

    # Group segments by (video_id, agent_id, target_id)
    groups = defaultdict(list)
    for seg in segments:
        key = (seg.video_id, seg.agent_id, seg.target_id)
        groups[key].append(seg)

    resolved_segments = []

    for key, group_segments in groups.items():
        # Sort by start_frame, then by stop_frame (to handle ties)
        group_segments.sort(key=lambda s: (s.start_frame, s.stop_frame))

        # Track the end of the last non-overlapping segment
        last_end = -1

        for seg in group_segments:
            new_start = seg.start_frame
            new_stop = seg.stop_frame

            # If this segment overlaps with the previous one, adjust start
            if new_start < last_end:
                new_start = last_end

            # Check if segment is still valid after adjustment
            if new_start < new_stop and (new_stop - new_start) >= min_duration:
                # Create a new segment with adjusted start
                resolved_seg = BehaviorSegment(
                    video_id=seg.video_id,
                    agent_id=seg.agent_id,
                    target_id=seg.target_id,
                    action=seg.action,
                    start_frame=new_start,
                    stop_frame=new_stop,
                    confidence=seg.confidence,
                )
                resolved_segments.append(resolved_seg)
                last_end = new_stop
            # If segment becomes invalid (too short or start >= stop), skip it

    return resolved_segments


def build_submission_rows(
    window_predictions: List[Dict],
    behaviors: List[str],
    threshold: float,
    min_duration: int,
    smoothing_kernel: int,
    nms_threshold: float,
    merge_gap: int = 5,
) -> List[Dict]:
    """
    Convert window-level predictions into submission-format rows.
    """
    aggregated = aggregate_window_predictions(window_predictions, overlap_strategy="average")
    all_segments: List[BehaviorSegment] = []

    for (video_id, agent_id, target_id), frame_probs in aggregated.items():
        raw_segments = extract_segments(
            frame_probs,
            behaviors,
            threshold=threshold,
            min_duration=min_duration,
            smoothing_kernel=smoothing_kernel,
        )
        merged = merge_segments(raw_segments, gap_threshold=merge_gap)
        final_segments = apply_nms(merged, iou_threshold=nms_threshold)

        # Format agent_id and target_id as strings (e.g., "mouse1", "mouse2", "self")
        formatted_agent = _format_mouse_id(agent_id, allow_self=False)
        formatted_target = _format_mouse_id(target_id, allow_self=True)

        for behavior, start, stop, conf in final_segments:
            all_segments.append(BehaviorSegment(
                video_id=int(_to_scalar(video_id)),
                agent_id=formatted_agent,
                target_id=formatted_target,
                action=behavior,
                start_frame=int(start),
                stop_frame=int(stop),
                confidence=float(conf),
            ))

    # Resolve overlapping behaviors for the same agent-target pair
    all_segments = resolve_overlaps(all_segments, min_duration=min_duration)

    # Sort and build rows directly from segments (not using create_submission)
    all_segments.sort(key=lambda s: (s.video_id, s.agent_id, s.target_id, s.start_frame))
    rows = []
    for row_id, seg in enumerate(all_segments):
        if seg.duration >= min_duration and seg.start_frame < seg.stop_frame:
            rows.append({
                'row_id': row_id,
                'video_id': seg.video_id,
                'agent_id': seg.agent_id,  # Now uses formatted string from segment
                'target_id': seg.target_id,  # Now uses formatted string from segment
                'action': seg.action,
                'start_frame': seg.start_frame,
                'stop_frame': seg.stop_frame,
            })

    return rows


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
    parser.add_argument(
        "--export_submission",
        action="store_true",
        help="Write submission-format predictions alongside metrics.",
    )
    parser.add_argument(
        "--submission_path",
        type=str,
        default=None,
        help="Optional custom path for submission CSV. Defaults to <output_dir>/submission.csv when export is enabled.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] Using {device}")

    threshold = args.threshold if args.threshold is not None else config.get("evaluation", {}).get("threshold", 0.5)
    export_submission = bool(args.export_submission or args.submission_path)

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
    y_true, y_probs, window_predictions = collect_predictions(
        model,
        dataloader,
        device,
        keep_windows=export_submission,
    )

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

    if export_submission:
        eval_cfg = config.get("evaluation", {})
        submission_rows = build_submission_rows(
            window_predictions,
            behaviors,
            threshold=threshold,
            min_duration=int(eval_cfg.get("min_duration", 2)),
            smoothing_kernel=int(eval_cfg.get("smoothing_kernel", 5)),
            nms_threshold=float(eval_cfg.get("nms_threshold", 0.3)),
        )
        submission_path = Path(args.submission_path) if args.submission_path else run_dir / "submission.csv"
        submission_df = pd.DataFrame(
            submission_rows,
            columns=["row_id", "video_id", "agent_id", "target_id", "action", "start_frame", "stop_frame"],
        )
        submission_df.to_csv(submission_path, index=False)
        print(f"[predictions] Submission-format predictions saved to {submission_path}")

        # Compute Kaggle-compatible segment-level F1 metric
        print("[kaggle] Computing Kaggle-compatible segment-level F1...")
        solution_df = build_solution_df(config, args.split)
        if solution_df is not None:
            print(f"[kaggle] Solution has {len(solution_df)} rows, {solution_df['video_id'].nunique()} videos")
            print(f"[kaggle] Submission has {len(submission_df)} rows, {submission_df['video_id'].nunique()} videos")

            # Debug: show sample video_ids and their types
            sol_videos = solution_df["video_id"].unique()[:5]
            sub_videos = submission_df["video_id"].unique()[:5]
            print(f"[kaggle] Solution video_ids (sample): {sol_videos} (type: {type(sol_videos[0]) if len(sol_videos) > 0 else 'N/A'})")
            print(f"[kaggle] Submission video_ids (sample): {sub_videos} (type: {type(sub_videos[0]) if len(sub_videos) > 0 else 'N/A'})")

            # Ensure consistent types for video_id
            solution_df["video_id"] = solution_df["video_id"].astype(int)
            submission_df["video_id"] = submission_df["video_id"].astype(int)

            # Filter solution to videos that are in the submission
            submission_videos = set(submission_df["video_id"].unique())
            solution_df = solution_df[solution_df["video_id"].isin(submission_videos)]
            print(f"[kaggle] After filtering: {len(solution_df)} solution rows for {len(submission_videos)} submission videos")

            if len(solution_df) > 0 and len(submission_df) > 0:
                # Debug: show sample rows
                print(f"[kaggle] Sample solution row: {solution_df.iloc[0].to_dict() if len(solution_df) > 0 else 'N/A'}")
                print(f"[kaggle] Sample submission row: {submission_df.iloc[0].to_dict() if len(submission_df) > 0 else 'N/A'}")

                try:
                    kaggle_f1 = mouse_fbeta(solution_df, submission_df, beta=1.0)
                    print(f"[kaggle] Segment-level F1 (Kaggle metric): {kaggle_f1:.4f}")

                    # Save to metrics
                    metrics["kaggle_f1"] = kaggle_f1
                    with open(run_dir / "metrics.json", "w") as f:
                        json.dump(metrics, f, indent=2)
                except Exception as e:
                    import traceback
                    print(f"[kaggle] Failed to compute Kaggle metric: {e}")
                    traceback.print_exc()
            else:
                print("[kaggle] No matching videos between solution and submission")
        else:
            print("[kaggle] Could not load solution data for Kaggle metric")

    print(f"[done] Metrics and plots written to {run_dir}")
    print(f"macro F1: {metrics['macro_f1']:.4f} | micro F1: {metrics['micro_f1']:.4f}")
    if metrics["macro_roc_auc"] is not None:
        print(f"macro ROC-AUC: {metrics['macro_roc_auc']:.4f}")


if __name__ == "__main__":
    main()
