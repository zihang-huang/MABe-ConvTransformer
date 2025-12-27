#!/usr/bin/env python3
"""
Precompute windowed tensors to disk to reduce training-time CPU load.

Writes sharded .pt files with features/labels/masks and a manifest per split.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import MABeDataModule  # noqa: E402


def load_config(config_path: Path, overrides: Optional[dict] = None) -> Dict:
    """
    Load YAML config and resolve ${...} references via OmegaConf so derived paths work on Windows.
    """
    conf = OmegaConf.load(config_path)
    if overrides:
        for key, value in overrides.items():
            OmegaConf.update(conf, key, value, merge=True)
    return OmegaConf.to_container(conf, resolve=True)


def flush_shard(
    split_dir: Path,
    shard_idx: int,
    buffer: Dict[str, List],
    manifest: Dict
):
    if len(buffer["metadata"]) == 0:
        return

    shard_path = split_dir / f"shard_{shard_idx:05d}.pt"
    shard = {
        "features": torch.stack(buffer["features"]),
        "labels": torch.stack(buffer["labels"]),
        "valid_mask": torch.stack(buffer["valid_mask"]),
        "metadata": buffer["metadata"],
    }
    torch.save(shard, shard_path)
    manifest["shards"].append({
        "path": shard_path.name,
        "num_samples": len(buffer["metadata"])
    })

    buffer["features"].clear()
    buffer["labels"].clear()
    buffer["valid_mask"].clear()
    buffer["metadata"].clear()


def compute_feature_stats(
    dataset,
    min_label_frames: int = 0,
    sample_ratio: float = 0.1
) -> Dict[str, np.ndarray]:
    """
    Compute per-feature mean and std using Welford's online algorithm.
    Uses sampling for efficiency on large datasets.

    Returns dict with 'mean' and 'std' arrays of shape (feature_dim,).
    """
    total = len(dataset)
    sample_indices = np.random.choice(total, size=max(1, int(total * sample_ratio)), replace=False)

    print(f"  Computing feature statistics from {len(sample_indices)} samples...")

    # Welford's online algorithm for numerical stability
    n = 0
    mean = None
    M2 = None  # Sum of squared differences from mean

    for i, idx in enumerate(sample_indices):
        if i % 500 == 0:
            print(f"    Stats: {i}/{len(sample_indices)}...", end="\r")

        sample = dataset[int(idx)]
        labels = sample["labels"]

        # Skip windows with too few labeled frames
        if min_label_frames > 0:
            label_coverage = (labels.sum(dim=-1) > 0).sum().item()
            if label_coverage < min_label_frames:
                continue

        features = sample["features"].numpy()  # (window_size, feature_dim)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Flatten window dimension to compute per-feature stats
        # Shape: (window_size, feature_dim) -> compute mean over axis 0
        batch_mean = features.mean(axis=0)  # (feature_dim,)

        if mean is None:
            mean = np.zeros_like(batch_mean)
            M2 = np.zeros_like(batch_mean)

        n += 1
        delta = batch_mean - mean
        mean += delta / n
        delta2 = batch_mean - mean
        M2 += delta * delta2

    print()  # Clear progress line

    if n < 2:
        raise ValueError("Not enough valid samples to compute statistics")

    std = np.sqrt(M2 / (n - 1))
    # Prevent division by zero - use 1.0 for constant features
    std = np.where(std < 1e-8, 1.0, std)

    print(f"  Feature stats: mean range [{mean.min():.2f}, {mean.max():.2f}], "
          f"std range [{std.min():.4f}, {std.max():.2f}]")

    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def process_split(
    name: str,
    dataset,
    out_root: Path,
    shard_size: int,
    min_label_frames: int = 0,
    feature_stats: Optional[Dict[str, np.ndarray]] = None
):
    """
    Process a dataset split into sharded files.

    Args:
        feature_stats: If provided, standardize features using these stats.
                      Should contain 'mean' and 'std' arrays.
    """
    if dataset is None or len(dataset) == 0:
        print(f"[skip] No samples for split '{name}'.")
        return None

    split_dir = out_root / name
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = split_dir / "manifest.json"

    # Peek first sample for metadata
    first_sample = dataset[0]
    window_size = first_sample["features"].shape[0]
    feature_dim = first_sample["features"].shape[-1]
    num_classes = first_sample["labels"].shape[-1]
    behaviors = getattr(dataset, "behaviors", None)

    manifest = {
        "split": name,
        "window_size": int(window_size),
        "feature_dim": int(feature_dim),
        "num_classes": int(num_classes),
        "behaviors": behaviors,
        "shards": []
    }

    # Store feature stats in manifest for reproducibility
    if feature_stats is not None:
        manifest["feature_mean"] = feature_stats["mean"].tolist()
        manifest["feature_std"] = feature_stats["std"].tolist()

    buffer = {
        "features": [],
        "labels": [],
        "valid_mask": [],
        "metadata": []
    }

    shard_idx = 0
    skipped = 0
    total = len(dataset)

    # Convert stats to tensors for efficient computation
    if feature_stats is not None:
        feat_mean = torch.from_numpy(feature_stats["mean"])
        feat_std = torch.from_numpy(feature_stats["std"])

    for idx in range(total):
        if idx % 1000 == 0:
            print(f"  [{name}] Processing {idx}/{total} (skipped: {skipped})...", end="\r")

        sample = dataset[idx]
        labels = sample["labels"]

        # Skip windows with too few labeled frames
        if min_label_frames > 0:
            label_coverage = (labels.sum(dim=-1) > 0).sum().item()
            if label_coverage < min_label_frames:
                skipped += 1
                continue

        # Sanitize NaN/Inf before standardization
        features = torch.nan_to_num(sample["features"], nan=0.0, posinf=0.0, neginf=0.0)
        labels = torch.nan_to_num(sample["labels"], nan=0.0, posinf=0.0, neginf=0.0)

        # Standardize features using training statistics
        if feature_stats is not None:
            features = (features - feat_mean) / feat_std
            # After standardization, values should be small, but clamp for safety
            features = torch.clamp(features, -100.0, 100.0)
        else:
            # Fallback: just clamp to float16 range
            features = torch.clamp(features, -65000.0, 65000.0)

        # Use float16 for storage efficiency
        buffer["features"].append(features.half().cpu())
        buffer["labels"].append(labels.half().cpu())
        buffer["valid_mask"].append(sample["valid_mask"].to(torch.uint8).cpu())
        buffer["metadata"].append({
            "video_id": sample["video_id"],
            "agent_id": int(sample["agent_id"]),
            "target_id": int(sample["target_id"]),
            "start_frame": int(sample["start_frame"])
        })

        if len(buffer["metadata"]) >= shard_size:
            flush_shard(split_dir, shard_idx, buffer, manifest)
            shard_idx += 1

    print()  # Clear progress line
    # Flush remainder
    flush_shard(split_dir, shard_idx, buffer, manifest)

    if skipped > 0:
        print(f"  [{name}] Skipped {skipped}/{total} windows with < {min_label_frames} labeled frames")

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total_samples = sum(s["num_samples"] for s in manifest["shards"])
    print(f"[done] {name}: {total_samples} samples -> {len(manifest['shards'])} shards at {split_dir}")

    return feature_stats


def main():
    parser = argparse.ArgumentParser(description="Precompute MABe windows to disk.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.")
    parser.add_argument("--output_dir", type=str, help="Override precomputed output directory.")
    parser.add_argument("--shard_size", type=int, default=512, help="Number of samples per shard file.")
    parser.add_argument("--stride", type=int, help="Override stride (higher = fewer windows, less overlap).")
    parser.add_argument("--min_label_frames", type=int, default=0,
                        help="Skip windows with fewer than N labeled frames (e.g., 30 to require ~1sec of labels).")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)

    out_dir = Path(args.output_dir) if args.output_dir else Path(config["data"].get("precomputed_dir", "precomputed"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Override stride if specified
    stride = args.stride if args.stride else config["data"]["stride"]
    window_size = config["data"]["window_size"]
    print(f"[info] Window size: {window_size}, Stride: {stride} (overlap: {max(0, window_size - stride)} frames)")

    # Build feature configuration from config
    features_cfg = config.get('features', {})
    use_engineered = features_cfg.get('use_engineered_features', False)
    feature_config = {
        'use_raw_coords': features_cfg.get('use_raw_coords', True),
        'use_single_mouse': features_cfg.get('use_single_mouse', True),
        'use_pairwise': features_cfg.get('use_pairwise', True),
        'use_temporal': features_cfg.get('use_temporal', True),
        'temporal_windows': features_cfg.get('temporal_windows', [5, 15, 30, 60]),
        'single_mouse': features_cfg.get('single_mouse', {}),
        'pair': features_cfg.get('pair', {})
    }

    if use_engineered:
        print(f"[info] Engineered features enabled: raw_coords={feature_config['use_raw_coords']}, "
              f"single_mouse={feature_config['use_single_mouse']}, "
              f"pairwise={feature_config['use_pairwise']}, "
              f"temporal={feature_config['use_temporal']}")

    dm = MABeDataModule(
        data_dir=config["paths"]["data_dir"],
        behaviors=config.get("behaviors", None),
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        window_size=window_size,
        stride=stride,
        target_fps=config["data"]["target_fps"],
        val_split=config["data"].get("val_split", 0.2),
        test_split=config["data"].get("test_split", 0),
        enable_test_split=config["data"].get("enable_test_split", True),
        tracking_cache_size=config["data"].get("tracking_cache_size", 4),
        annotation_cache_size=config["data"].get("annotation_cache_size", 8),
        # Engineered features
        use_engineered_features=use_engineered,
        feature_config=feature_config
    )

    # Build splits using the existing logic
    dm.setup(stage="fit", use_precomputed=False)

    # Step 1: Compute feature statistics from training set
    print("[step 1/3] Computing feature statistics from training set...")
    feature_stats = compute_feature_stats(
        dm.train_dataset,
        min_label_frames=args.min_label_frames,
        sample_ratio=0.1  # Use 10% of samples for efficiency
    )

    # Step 2: Process splits with standardization
    print("[step 2/3] Processing training split with standardization...")
    process_split("train", dm.train_dataset, out_dir, args.shard_size, args.min_label_frames, feature_stats)

    print("[step 3/3] Processing validation split with training statistics...")
    process_split("val", dm.val_dataset, out_dir, args.shard_size, args.min_label_frames, feature_stats)

    if dm.enable_test_split and dm.test_split > 0:
        dm.setup(stage="test", use_precomputed=False)
        print("[extra] Processing test split with training statistics...")
        process_split("test", dm.test_dataset, out_dir, args.shard_size, args.min_label_frames, feature_stats)
    else:
        print("[skip] Test split disabled; not precomputing test shards.")

    print(f"\nManifests written to {out_dir}.")
    print("Feature statistics saved in manifests for reproducibility.")
    print("Point config.data.precomputed_dir here and set use_precomputed=true to train from shards.")


if __name__ == "__main__":
    main()
