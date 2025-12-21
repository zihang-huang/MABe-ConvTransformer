"""
Evaluation metrics for behavior recognition.
Includes both frame-level and segment-level metrics.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from sklearn.metrics import (
    precision_recall_fscore_support,
    average_precision_score,
    confusion_matrix
)


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    behavior_names: Optional[List[str]] = None,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute comprehensive frame-level metrics.

    Args:
        predictions: Predicted probabilities (n_frames, n_classes)
        targets: Ground truth labels (n_frames, n_classes) or (n_frames,)
        behavior_names: Names of behavior classes
        threshold: Threshold for binary predictions

    Returns:
        Dictionary of metrics
    """
    # Convert to binary predictions
    if predictions.ndim == 2:
        binary_preds = (predictions >= threshold).astype(int)
    else:
        binary_preds = predictions

    # Handle multi-label vs single-label
    if targets.ndim == 1:
        # Single-label: convert to multi-label format
        n_classes = predictions.shape[1] if predictions.ndim == 2 else int(targets.max()) + 1
        targets_ml = np.zeros((len(targets), n_classes))
        targets_ml[np.arange(len(targets)), targets.astype(int)] = 1
        targets = targets_ml

    metrics = {}

    # Overall metrics
    metrics['accuracy'] = (binary_preds == targets).mean()

    # Per-class metrics
    n_classes = targets.shape[1]

    precisions = []
    recalls = []
    f1s = []
    aps = []

    for c in range(n_classes):
        # Skip classes with no positive samples
        if targets[:, c].sum() == 0:
            continue

        tp = ((binary_preds[:, c] == 1) & (targets[:, c] == 1)).sum()
        fp = ((binary_preds[:, c] == 1) & (targets[:, c] == 0)).sum()
        fn = ((binary_preds[:, c] == 0) & (targets[:, c] == 1)).sum()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        # Average precision
        if predictions.ndim == 2:
            ap = average_precision_score(targets[:, c], predictions[:, c])
            aps.append(ap)

    # Macro averages
    metrics['macro_precision'] = np.mean(precisions) if precisions else 0
    metrics['macro_recall'] = np.mean(recalls) if recalls else 0
    metrics['macro_f1'] = np.mean(f1s) if f1s else 0

    if aps:
        metrics['mAP'] = np.mean(aps)

    # Per-class metrics
    if behavior_names:
        for c, name in enumerate(behavior_names):
            if c < len(precisions):
                metrics[f'{name}_f1'] = f1s[c]

    return metrics


def segment_f1_score(
    pred_segments: List[Tuple[str, int, int]],
    true_segments: List[Tuple[str, int, int]],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute segment-level F1 score using IoU matching.

    Args:
        pred_segments: List of (behavior, start, end) tuples
        true_segments: List of (behavior, start, end) tuples
        iou_threshold: IoU threshold for matching

    Returns:
        Dictionary with precision, recall, F1 per behavior and overall
    """
    # Group by behavior
    pred_by_behavior = defaultdict(list)
    true_by_behavior = defaultdict(list)

    for behavior, start, end in pred_segments:
        pred_by_behavior[behavior].append((start, end))

    for behavior, start, end in true_segments:
        true_by_behavior[behavior].append((start, end))

    all_behaviors = set(pred_by_behavior.keys()) | set(true_by_behavior.keys())

    metrics = {}
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for behavior in all_behaviors:
        preds = pred_by_behavior.get(behavior, [])
        trues = true_by_behavior.get(behavior, [])

        tp, fp, fn = match_segments(preds, trues, iou_threshold)

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        metrics[f'{behavior}_precision'] = precision
        metrics[f'{behavior}_recall'] = recall
        metrics[f'{behavior}_f1'] = f1

        total_tp += tp
        total_fp += fp
        total_fn += fn

    # Overall metrics
    metrics['precision'] = total_tp / (total_tp + total_fp + 1e-8)
    metrics['recall'] = total_tp / (total_tp + total_fn + 1e-8)
    metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (
        metrics['precision'] + metrics['recall'] + 1e-8
    )

    return metrics


def match_segments(
    pred_segments: List[Tuple[int, int]],
    true_segments: List[Tuple[int, int]],
    iou_threshold: float = 0.5
) -> Tuple[int, int, int]:
    """
    Match predicted and ground truth segments using IoU.

    Returns:
        (true_positives, false_positives, false_negatives)
    """
    if not pred_segments:
        return 0, 0, len(true_segments)
    if not true_segments:
        return 0, len(pred_segments), 0

    # Compute IoU matrix
    n_preds = len(pred_segments)
    n_trues = len(true_segments)
    iou_matrix = np.zeros((n_preds, n_trues))

    for i, (ps, pe) in enumerate(pred_segments):
        for j, (ts, te) in enumerate(true_segments):
            intersection = max(0, min(pe, te) - max(ps, ts))
            union = (pe - ps) + (te - ts) - intersection
            iou_matrix[i, j] = intersection / union if union > 0 else 0

    # Greedy matching
    matched_preds = set()
    matched_trues = set()

    while True:
        # Find best unmatched pair
        best_iou = 0
        best_i, best_j = -1, -1

        for i in range(n_preds):
            if i in matched_preds:
                continue
            for j in range(n_trues):
                if j in matched_trues:
                    continue
                if iou_matrix[i, j] > best_iou:
                    best_iou = iou_matrix[i, j]
                    best_i, best_j = i, j

        if best_iou < iou_threshold:
            break

        matched_preds.add(best_i)
        matched_trues.add(best_j)

    tp = len(matched_preds)
    fp = n_preds - tp
    fn = n_trues - len(matched_trues)

    return tp, fp, fn


def compute_edit_distance(
    pred_sequence: List[str],
    true_sequence: List[str]
) -> float:
    """
    Compute normalized edit distance between behavior sequences.
    Used for evaluating temporal consistency.
    """
    n, m = len(pred_sequence), len(true_sequence)

    if n == 0:
        return 1.0 if m > 0 else 0.0
    if m == 0:
        return 1.0

    # DP for edit distance
    dp = np.zeros((n + 1, m + 1))
    dp[0, :] = np.arange(m + 1)
    dp[:, 0] = np.arange(n + 1)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if pred_sequence[i-1] == true_sequence[j-1] else 1
            dp[i, j] = min(
                dp[i-1, j] + 1,      # deletion
                dp[i, j-1] + 1,      # insertion
                dp[i-1, j-1] + cost  # substitution
            )

    return dp[n, m] / max(n, m)


def compute_segmental_edit_distance(
    pred_segments: List[str],
    true_segments: List[str]
) -> float:
    """
    Compute edit distance on segment-level (collapsed) sequences.
    Removes consecutive duplicates before computing.
    """
    def collapse(seq):
        if not seq:
            return []
        collapsed = [seq[0]]
        for s in seq[1:]:
            if s != collapsed[-1]:
                collapsed.append(s)
        return collapsed

    pred_collapsed = collapse(pred_segments)
    true_collapsed = collapse(true_segments)

    return compute_edit_distance(pred_collapsed, true_collapsed)


def per_class_statistics(
    predictions: np.ndarray,
    targets: np.ndarray,
    behavior_names: List[str],
    threshold: float = 0.5
) -> Dict[str, Dict[str, float]]:
    """
    Compute detailed per-class statistics.

    Returns:
        Dictionary mapping behavior names to their statistics
    """
    binary_preds = (predictions >= threshold).astype(int)

    stats = {}

    for c, name in enumerate(behavior_names):
        class_stats = {}

        # Basic counts
        class_stats['n_true'] = int(targets[:, c].sum())
        class_stats['n_pred'] = int(binary_preds[:, c].sum())

        # TP, FP, FN, TN
        tp = ((binary_preds[:, c] == 1) & (targets[:, c] == 1)).sum()
        fp = ((binary_preds[:, c] == 1) & (targets[:, c] == 0)).sum()
        fn = ((binary_preds[:, c] == 0) & (targets[:, c] == 1)).sum()
        tn = ((binary_preds[:, c] == 0) & (targets[:, c] == 0)).sum()

        class_stats['tp'] = int(tp)
        class_stats['fp'] = int(fp)
        class_stats['fn'] = int(fn)
        class_stats['tn'] = int(tn)

        # Metrics
        class_stats['precision'] = tp / (tp + fp + 1e-8)
        class_stats['recall'] = tp / (tp + fn + 1e-8)
        class_stats['f1'] = 2 * class_stats['precision'] * class_stats['recall'] / (
            class_stats['precision'] + class_stats['recall'] + 1e-8
        )
        class_stats['specificity'] = tn / (tn + fp + 1e-8)

        stats[name] = class_stats

    return stats


def compute_confusion_matrix(
    predictions: np.ndarray,
    targets: np.ndarray,
    behavior_names: List[str]
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute multi-label confusion matrix.
    For single-label case, use sklearn's confusion_matrix.
    """
    if targets.ndim == 1 or (targets.ndim == 2 and targets.sum(axis=1).max() == 1):
        # Single-label case
        if targets.ndim == 2:
            targets = targets.argmax(axis=1)
        if predictions.ndim == 2:
            predictions = predictions.argmax(axis=1)

        cm = confusion_matrix(targets, predictions)
        return cm, behavior_names

    # Multi-label: return per-class confusion
    n_classes = len(behavior_names)
    cms = []

    for c in range(n_classes):
        binary_pred = (predictions[:, c] >= 0.5).astype(int)
        binary_true = targets[:, c].astype(int)
        cm = confusion_matrix(binary_true, binary_pred, labels=[0, 1])
        cms.append(cm)

    return np.array(cms), behavior_names
