#!/usr/bin/env python3
"""
Inference script for MABe behavior recognition.

Usage:
    python scripts/inference.py --checkpoint path/to/checkpoint.ckpt --output submission.csv
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict

import pytorch_lightning as pl

from src.data.dataset import MABeDataModule, MABeDataset
from src.models.lightning_module import BehaviorRecognitionModule
from src.utils.postprocessing import (
    extract_segments,
    merge_segments,
    apply_nms,
    create_submission,
    aggregate_window_predictions,
    BehaviorSegment
)


def load_model(checkpoint_path: str, device: str = 'cuda') -> BehaviorRecognitionModule:
    """Load trained model from checkpoint."""
    model = BehaviorRecognitionModule.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to(device)
    return model


def run_inference(
    model: BehaviorRecognitionModule,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda'
) -> list:
    """Run inference on dataloader."""
    all_predictions = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running inference"):
            features = batch['features'].to(device)
            mask = batch.get('valid_mask')
            if mask is not None:
                mask = mask.to(device)

            # Get predictions
            predictions = model(features, mask)
            probs = torch.sigmoid(predictions).cpu().numpy()

            # Store predictions with metadata
            batch_size = features.shape[0]
            for i in range(batch_size):
                all_predictions.append({
                    'video_id': batch['video_id'][i].item() if torch.is_tensor(batch['video_id'][i]) else batch['video_id'][i],
                    'agent_id': batch['agent_id'][i].item() if torch.is_tensor(batch['agent_id'][i]) else batch['agent_id'][i],
                    'target_id': batch['target_id'][i].item() if torch.is_tensor(batch['target_id'][i]) else batch['target_id'][i],
                    'start_frame': batch['start_frame'][i].item() if torch.is_tensor(batch['start_frame'][i]) else batch['start_frame'][i],
                    'probabilities': probs[i]
                })

    return all_predictions


def predictions_to_segments(
    predictions: list,
    behaviors: list,
    threshold: float = 0.5,
    min_duration: int = 5,
    smoothing_kernel: int = 5,
    merge_gap: int = 5,
    nms_threshold: float = 0.3
) -> dict:
    """Convert raw predictions to behavior segments."""
    # Aggregate overlapping windows
    aggregated = aggregate_window_predictions(predictions, overlap_strategy='average')

    all_segments = {}

    for (video_id, agent_id, target_id), frame_probs in tqdm(
        aggregated.items(), desc="Extracting segments"
    ):
        # Extract segments for this video/pair
        raw_segments = extract_segments(
            frame_probs,
            behaviors,
            threshold=threshold,
            min_duration=min_duration,
            smoothing_kernel=smoothing_kernel
        )

        # Merge nearby segments
        merged = merge_segments(raw_segments, gap_threshold=merge_gap)

        # Apply NMS
        final_segments = apply_nms(merged, iou_threshold=nms_threshold)

        # Convert to BehaviorSegment objects
        segment_objects = []
        for behavior, start, end, conf in final_segments:
            segment_objects.append(BehaviorSegment(
                video_id=video_id,
                agent_id=f"mouse{agent_id}" if isinstance(agent_id, int) else agent_id,
                target_id=f"mouse{target_id}" if isinstance(target_id, int) and target_id >= 0 else ("self" if target_id == -1 else target_id),
                action=behavior,
                start_frame=start,
                stop_frame=end,
                confidence=conf
            ))

        all_segments[(video_id, agent_id, target_id)] = segment_objects

    return all_segments


def main(
    checkpoint_path: str,
    data_dir: str,
    output_path: str,
    threshold: float = 0.5,
    min_duration: int = 5,
    batch_size: int = 16,
    device: str = None
):
    """Main inference function."""
    # Auto-detect device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {checkpoint_path}...")
    model = load_model(checkpoint_path, device)

    # Get behaviors from model
    behaviors = model.behaviors

    # Initialize data module for test data
    print("Loading test data...")
    data_module = MABeDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=4
    )
    data_module.setup('test')

    if data_module.test_dataset is None:
        print("No test data found!")
        return

    test_loader = data_module.test_dataloader()

    # Run inference
    print("Running inference...")
    predictions = run_inference(model, test_loader, device)

    # Convert to segments
    print("Converting predictions to segments...")
    all_segments = predictions_to_segments(
        predictions,
        behaviors,
        threshold=threshold,
        min_duration=min_duration
    )

    # Create submission
    print("Creating submission file...")
    submission_rows = create_submission(all_segments, min_duration=2)

    # Save to CSV
    df = pd.DataFrame(submission_rows)
    df.to_csv(output_path, index=False)

    print(f"Submission saved to {output_path}")
    print(f"Total rows: {len(df)}")

    # Print summary statistics
    if len(df) > 0:
        print("\nSubmission summary:")
        print(f"  Videos: {df['video_id'].nunique()}")
        print(f"  Unique behaviors: {df['action'].nunique()}")
        print("\nBehavior counts:")
        print(df['action'].value_counts().head(10))

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference for MABe behavior recognition')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to data directory')
    parser.add_argument('--output', type=str, default='submission.csv',
                        help='Output submission file path')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Prediction threshold')
    parser.add_argument('--min_duration', type=int, default=5,
                        help='Minimum segment duration')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for inference')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    main(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_path=args.output,
        threshold=args.threshold,
        min_duration=args.min_duration,
        batch_size=args.batch_size,
        device=args.device
    )
