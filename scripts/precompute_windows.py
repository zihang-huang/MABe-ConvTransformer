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


def process_split(
    name: str,
    dataset,
    out_root: Path,
    shard_size: int
):
    if dataset is None or len(dataset) == 0:
        print(f"[skip] No samples for split '{name}'.")
        return

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

    buffer = {
        "features": [],
        "labels": [],
        "valid_mask": [],
        "metadata": []
    }

    shard_idx = 0
    for idx in range(len(dataset)):
        sample = dataset[idx]
        buffer["features"].append(sample["features"].half().cpu())
        buffer["labels"].append(sample["labels"].half().cpu())
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

    # Flush remainder
    flush_shard(split_dir, shard_idx, buffer, manifest)

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total_samples = sum(s["num_samples"] for s in manifest["shards"])
    print(f"[done] {name}: {total_samples} samples -> {len(manifest['shards'])} shards at {split_dir}")


def main():
    parser = argparse.ArgumentParser(description="Precompute MABe windows to disk.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.")
    parser.add_argument("--output_dir", type=str, help="Override precomputed output directory.")
    parser.add_argument("--shard_size", type=int, default=512, help="Number of samples per shard file.")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)

    out_dir = Path(args.output_dir) if args.output_dir else Path(config["data"].get("precomputed_dir", "precomputed"))
    out_dir.mkdir(parents=True, exist_ok=True)

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
    )

    # Build splits using the existing logic
    dm.setup(stage="fit", use_precomputed=False)
    process_split("train", dm.train_dataset, out_dir, args.shard_size)
    process_split("val", dm.val_dataset, out_dir, args.shard_size)

    dm.setup(stage="test", use_precomputed=False)
    process_split("test", dm.test_dataset, out_dir, args.shard_size)

    print(f"Manifests written to {out_dir}. Point config.data.precomputed_dir here and set use_precomputed=true to train from shards.")


if __name__ == "__main__":
    main()
