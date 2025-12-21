"""
Post-processing utilities for converting frame-level predictions to behavior segments.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.ndimage import median_filter, uniform_filter1d


@dataclass
class BehaviorSegment:
    """Represents a detected behavior segment."""
    video_id: int
    agent_id: str
    target_id: str
    action: str
    start_frame: int
    stop_frame: int
    confidence: float = 1.0

    @property
    def duration(self) -> int:
        return self.stop_frame - self.start_frame

    def to_dict(self) -> Dict:
        return {
            'video_id': self.video_id,
            'agent_id': self.agent_id,
            'target_id': self.target_id,
            'action': self.action,
            'start_frame': self.start_frame,
            'stop_frame': self.stop_frame
        }


def extract_segments(
    frame_probs: np.ndarray,
    behavior_names: List[str],
    threshold: float = 0.5,
    min_duration: int = 5,
    smoothing_kernel: int = 5
) -> List[Tuple[str, int, int, float]]:
    """
    Convert frame-level probabilities to behavior segments.

    Args:
        frame_probs: Array of shape (n_frames, n_behaviors)
        behavior_names: List of behavior class names
        threshold: Probability threshold for detection
        min_duration: Minimum segment duration in frames
        smoothing_kernel: Size of median filter for smoothing

    Returns:
        List of (behavior, start_frame, stop_frame, confidence) tuples
    """
    n_frames, n_behaviors = frame_probs.shape
    segments = []

    for behavior_idx in range(n_behaviors):
        probs = frame_probs[:, behavior_idx]
        behavior = behavior_names[behavior_idx]

        # Apply temporal smoothing
        if smoothing_kernel > 1:
            probs = median_filter(probs, size=smoothing_kernel)

        # Threshold to binary
        binary = (probs >= threshold).astype(np.int32)

        # Find contiguous regions
        behavior_segments = find_contiguous_regions(binary)

        # Filter by duration and add confidence
        for start, end in behavior_segments:
            duration = end - start
            if duration >= min_duration:
                confidence = float(probs[start:end].mean())
                segments.append((behavior, start, end, confidence))

    return segments


def find_contiguous_regions(binary: np.ndarray) -> List[Tuple[int, int]]:
    """
    Find start and end indices of contiguous True regions.

    Args:
        binary: 1D binary array

    Returns:
        List of (start, end) tuples
    """
    regions = []

    # Find transitions
    diff = np.diff(np.concatenate([[0], binary, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    for start, end in zip(starts, ends):
        regions.append((int(start), int(end)))

    return regions


def merge_segments(
    segments: List[Tuple[str, int, int, float]],
    gap_threshold: int = 5,
    same_behavior_only: bool = True
) -> List[Tuple[str, int, int, float]]:
    """
    Merge nearby segments of the same behavior.

    Args:
        segments: List of (behavior, start, end, confidence) tuples
        gap_threshold: Maximum gap between segments to merge
        same_behavior_only: Only merge segments of same behavior

    Returns:
        Merged segment list
    """
    if not segments:
        return []

    # Sort by behavior, then start frame
    segments = sorted(segments, key=lambda x: (x[0], x[1]))

    merged = []
    current = list(segments[0])

    for behavior, start, end, conf in segments[1:]:
        if same_behavior_only and behavior != current[0]:
            merged.append(tuple(current))
            current = [behavior, start, end, conf]
        elif start - current[2] <= gap_threshold:
            # Merge: extend end, average confidence
            new_conf = (current[3] * (current[2] - current[1]) +
                       conf * (end - start)) / (end - current[1])
            current[2] = end
            current[3] = new_conf
        else:
            merged.append(tuple(current))
            current = [behavior, start, end, conf]

    merged.append(tuple(current))

    return merged


def apply_nms(
    segments: List[Tuple[str, int, int, float]],
    iou_threshold: float = 0.3
) -> List[Tuple[str, int, int, float]]:
    """
    Apply Non-Maximum Suppression to remove overlapping segments.

    Args:
        segments: List of (behavior, start, end, confidence) tuples
        iou_threshold: IoU threshold for suppression

    Returns:
        Filtered segment list
    """
    if not segments:
        return []

    # Sort by confidence (descending)
    segments = sorted(segments, key=lambda x: x[3], reverse=True)

    keep = []

    while segments:
        # Keep the highest confidence segment
        best = segments.pop(0)
        keep.append(best)

        # Remove overlapping segments
        remaining = []
        for seg in segments:
            iou = compute_segment_iou(
                (best[1], best[2]),
                (seg[1], seg[2])
            )
            if iou < iou_threshold:
                remaining.append(seg)

        segments = remaining

    return sorted(keep, key=lambda x: x[1])


def compute_segment_iou(
    seg1: Tuple[int, int],
    seg2: Tuple[int, int]
) -> float:
    """
    Compute Intersection over Union for two temporal segments.
    """
    start1, end1 = seg1
    start2, end2 = seg2

    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)

    if intersection_end <= intersection_start:
        return 0.0

    intersection = intersection_end - intersection_start
    union = (end1 - start1) + (end2 - start2) - intersection

    return intersection / union if union > 0 else 0.0


def resolve_overlapping_behaviors(
    segments: List[Tuple[str, int, int, float]],
    priority_order: Optional[List[str]] = None
) -> List[Tuple[str, int, int, float]]:
    """
    Resolve overlapping segments by keeping higher priority behaviors.

    Args:
        segments: List of (behavior, start, end, confidence) tuples
        priority_order: List of behaviors in priority order (higher first)

    Returns:
        Non-overlapping segment list
    """
    if not segments:
        return []

    n_frames = max(seg[2] for seg in segments) + 1

    # Create frame-level assignment
    frame_behavior = [None] * n_frames
    frame_confidence = np.zeros(n_frames)

    # Sort by priority if given, else by confidence
    if priority_order:
        priority_map = {b: i for i, b in enumerate(priority_order)}
        segments = sorted(
            segments,
            key=lambda x: priority_map.get(x[0], len(priority_order))
        )
    else:
        segments = sorted(segments, key=lambda x: x[3], reverse=True)

    # Assign frames
    for behavior, start, end, conf in segments:
        for f in range(start, end):
            if frame_behavior[f] is None or conf > frame_confidence[f]:
                frame_behavior[f] = behavior
                frame_confidence[f] = conf

    # Convert back to segments
    resolved = []
    current_behavior = None
    current_start = 0
    current_conf_sum = 0
    current_count = 0

    for f in range(n_frames):
        if frame_behavior[f] != current_behavior:
            if current_behavior is not None:
                avg_conf = current_conf_sum / current_count if current_count > 0 else 0
                resolved.append((current_behavior, current_start, f, avg_conf))

            current_behavior = frame_behavior[f]
            current_start = f
            current_conf_sum = frame_confidence[f] if frame_behavior[f] else 0
            current_count = 1 if frame_behavior[f] else 0
        else:
            current_conf_sum += frame_confidence[f]
            current_count += 1

    if current_behavior is not None:
        avg_conf = current_conf_sum / current_count if current_count > 0 else 0
        resolved.append((current_behavior, current_start, n_frames, avg_conf))

    return resolved


def create_submission(
    all_segments: Dict[Tuple[int, str, str], List[BehaviorSegment]],
    min_duration: int = 2
) -> List[Dict]:
    """
    Create submission-ready output from detected segments.

    Args:
        all_segments: Dict mapping (video_id, agent_id, target_id) to segments
        min_duration: Minimum segment duration to include

    Returns:
        List of submission rows
    """
    rows = []
    row_id = 0

    for (video_id, agent_id, target_id), segments in sorted(all_segments.items()):
        for seg in segments:
            if seg.duration >= min_duration and seg.start_frame < seg.stop_frame:
                rows.append({
                    'row_id': row_id,
                    'video_id': video_id,
                    'agent_id': agent_id,
                    'target_id': target_id,
                    'action': seg.action,
                    'start_frame': seg.start_frame,
                    'stop_frame': seg.stop_frame
                })
                row_id += 1

    return rows


def aggregate_window_predictions(
    window_predictions: List[Dict],
    overlap_strategy: str = 'average'
) -> Dict[Tuple[int, str, str], np.ndarray]:
    """
    Aggregate predictions from overlapping windows.

    Args:
        window_predictions: List of prediction dicts with 'probabilities', 'start_frame', etc.
        overlap_strategy: 'average', 'max', or 'first'

    Returns:
        Dict mapping (video_id, agent_id, target_id) to full-length prediction arrays
    """
    # Group by video and mouse pair
    grouped = {}

    for pred in window_predictions:
        key = (pred['video_id'], pred['agent_id'], pred['target_id'])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(pred)

    # Aggregate for each group
    aggregated = {}

    for key, preds in grouped.items():
        # Determine full sequence length
        max_end = max(p['start_frame'] + p['probabilities'].shape[0] for p in preds)
        n_classes = preds[0]['probabilities'].shape[-1]

        # Initialize accumulators
        sum_probs = np.zeros((max_end, n_classes), dtype=np.float32)
        counts = np.zeros(max_end, dtype=np.float32)

        for pred in preds:
            start = pred['start_frame']
            probs = pred['probabilities']
            end = start + len(probs)

            if overlap_strategy == 'average':
                sum_probs[start:end] += probs
                counts[start:end] += 1
            elif overlap_strategy == 'max':
                sum_probs[start:end] = np.maximum(sum_probs[start:end], probs)
                counts[start:end] = 1
            elif overlap_strategy == 'first':
                mask = counts[start:end] == 0
                sum_probs[start:end][mask] = probs[mask]
                counts[start:end][mask] = 1

        # Average where we have predictions
        mask = counts > 0
        sum_probs[mask] /= counts[mask, np.newaxis]

        aggregated[key] = sum_probs

    return aggregated
