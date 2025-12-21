"""
Dataset and DataModule for MABe behavior recognition.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

try:
    import polars as pls
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

from .preprocessing import (
    CoordinateNormalizer,
    TemporalResampler,
    MissingDataHandler,
    BodyPartMapper
)


class MABeDataset(Dataset):
    """
    PyTorch Dataset for MABe behavior recognition.

    Loads tracking data and annotations, applies preprocessing,
    and returns fixed-length windows for training.
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        tracking_dir: Path,
        annotation_dir: Optional[Path] = None,
        behaviors: Optional[List[str]] = None,
        window_size: int = 512,
        stride: int = 256,
        target_fps: float = 30.0,
        normalize_coords: bool = True,
        augment: bool = False,
        is_train: bool = True
    ):
        """
        Args:
            metadata_df: DataFrame with video metadata (from train.csv/test.csv)
            tracking_dir: Path to tracking parquet files
            annotation_dir: Path to annotation parquet files (None for test)
            behaviors: List of behavior classes to predict
            window_size: Number of frames per sample
            stride: Stride for sliding window
            target_fps: Target frame rate for resampling
            normalize_coords: Whether to normalize coordinates
            augment: Whether to apply data augmentation
            is_train: Whether this is training data
        """
        self.tracking_dir = Path(tracking_dir)
        self.annotation_dir = Path(annotation_dir) if annotation_dir else None
        self.window_size = window_size
        self.stride = stride
        self.target_fps = target_fps
        self.normalize_coords = normalize_coords
        self.augment = augment
        self.is_train = is_train

        # Initialize preprocessors
        self.coord_normalizer = CoordinateNormalizer()
        self.temporal_resampler = TemporalResampler(target_fps)
        self.missing_handler = MissingDataHandler()
        self.bodypart_mapper = BodyPartMapper(use_core_only=False)

        # Parse metadata
        self.metadata_df = metadata_df.copy()
        self.video_ids = metadata_df['video_id'].unique().tolist()

        # Build behavior vocabulary
        if behaviors is None:
            self.behaviors = self._collect_behaviors()
        else:
            self.behaviors = behaviors
        self.behavior_to_idx = {b: i for i, b in enumerate(self.behaviors)}
        self.idx_to_behavior = {i: b for i, b in enumerate(self.behaviors)}
        self.num_classes = len(self.behaviors)

        # Build sample index (video_id, start_frame, mice_pair)
        self.samples = self._build_sample_index()

    def _collect_behaviors(self) -> List[str]:
        """Collect all unique behaviors from annotations."""
        if self.annotation_dir is None:
            return []

        behaviors = set()
        for _, row in self.metadata_df.iterrows():
            lab_id = row['lab_id']
            video_id = row['video_id']
            ann_path = self.annotation_dir / f"{lab_id}" / f"{video_id}.parquet"

            if ann_path.exists():
                ann_df = pd.read_parquet(ann_path)
                behaviors.update(ann_df['action'].unique())

        return sorted(list(behaviors))

    def _build_sample_index(self) -> List[Dict]:
        """Build index of (video_id, start_frame, agent, target) samples."""
        samples = []

        for _, row in self.metadata_df.iterrows():
            lab_id = row['lab_id']
            video_id = row['video_id']
            fps = row.get('frames per second', row.get('fps', 30))

            # Load tracking to get video length and mice
            track_path = self.tracking_dir / f"{lab_id}" / f"{video_id}.parquet"
            if not track_path.exists():
                continue

            track_df = pd.read_parquet(track_path)
            n_frames = track_df['video_frame'].max() + 1
            mice = track_df['mouse_id'].unique()

            # Adjust for resampling
            if fps != self.target_fps:
                duration = n_frames / fps
                n_frames = int(duration * self.target_fps)

            # Generate windows
            for start in range(0, max(1, n_frames - self.window_size + 1), self.stride):
                # For each agent-target pair
                for agent in mice:
                    for target in mice:
                        samples.append({
                            'lab_id': lab_id,
                            'video_id': video_id,
                            'start_frame': start,
                            'agent_id': agent,
                            'target_id': target,
                            'metadata': row.to_dict()
                        })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_info = self.samples[idx]

        # Load and preprocess tracking data
        features, valid_mask = self._load_tracking(sample_info)

        # Load annotations if available
        if self.annotation_dir is not None:
            labels = self._load_annotations(sample_info)
        else:
            labels = np.zeros((self.window_size, self.num_classes), dtype=np.float32)

        # Apply augmentation
        if self.augment and self.is_train:
            features, labels = self._augment(features, labels)

        return {
            'features': torch.from_numpy(features),
            'labels': torch.from_numpy(labels),
            'valid_mask': torch.from_numpy(valid_mask),
            'video_id': sample_info['video_id'],
            'agent_id': sample_info['agent_id'],
            'target_id': sample_info['target_id'],
            'start_frame': sample_info['start_frame']
        }

    def _load_tracking(self, sample_info: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess tracking data for a sample."""
        lab_id = sample_info['lab_id']
        video_id = sample_info['video_id']
        start_frame = sample_info['start_frame']
        agent_id = sample_info['agent_id']
        target_id = sample_info['target_id']
        metadata = sample_info['metadata']

        track_path = self.tracking_dir / f"{lab_id}" / f"{video_id}.parquet"
        track_df = pd.read_parquet(track_path)

        # Get FPS for resampling
        fps = metadata.get('frames per second', metadata.get('fps', 30))

        # Extract agent and target coordinates
        agent_coords = self._extract_mouse_coords(track_df, agent_id)
        target_coords = self._extract_mouse_coords(track_df, target_id)

        # Resample to target FPS
        if fps != self.target_fps:
            agent_coords = self.temporal_resampler(agent_coords, fps)
            target_coords = self.temporal_resampler(target_coords, fps)

        # Normalize coordinates
        if self.normalize_coords:
            agent_coords = self.coord_normalizer(agent_coords, metadata)
            target_coords = self.coord_normalizer(target_coords, metadata)

        # Handle missing data
        agent_coords, agent_valid = self.missing_handler.interpolate_missing(agent_coords)
        target_coords, target_valid = self.missing_handler.interpolate_missing(target_coords)

        # Extract window
        end_frame = start_frame + self.window_size
        agent_window = self._get_window(agent_coords, start_frame, end_frame)
        target_window = self._get_window(target_coords, start_frame, end_frame)

        # Concatenate agent and target features
        # Shape: (window_size, n_features)
        features = np.concatenate([
            agent_window.reshape(self.window_size, -1),
            target_window.reshape(self.window_size, -1)
        ], axis=-1)

        # Create validity mask
        agent_valid_window = self._get_window(agent_valid.astype(np.float32), start_frame, end_frame)
        target_valid_window = self._get_window(target_valid.astype(np.float32), start_frame, end_frame)
        valid_mask = (agent_valid_window.mean(axis=-1) > 0.5) & (target_valid_window.mean(axis=-1) > 0.5)

        return features.astype(np.float32), valid_mask.astype(np.float32)

    def _extract_mouse_coords(self, track_df: pd.DataFrame, mouse_id: int) -> np.ndarray:
        """Extract coordinates for a single mouse."""
        mouse_df = track_df[track_df['mouse_id'] == mouse_id].copy()

        # Pivot to get bodypart columns
        bodyparts = mouse_df['bodypart'].unique()
        n_frames = track_df['video_frame'].max() + 1

        coords = np.full((n_frames, len(bodyparts), 2), np.nan, dtype=np.float32)

        for i, bp in enumerate(bodyparts):
            bp_df = mouse_df[mouse_df['bodypart'] == bp].sort_values('video_frame')
            frames = bp_df['video_frame'].values
            coords[frames, i, 0] = bp_df['x'].values
            coords[frames, i, 1] = bp_df['y'].values

        return coords

    def _get_window(self, data: np.ndarray, start: int, end: int) -> np.ndarray:
        """Extract a window from data, padding if necessary."""
        n_frames = data.shape[0]

        # Handle edge cases
        if start < 0:
            pre_pad = -start
            start = 0
        else:
            pre_pad = 0

        if end > n_frames:
            post_pad = end - n_frames
            end = n_frames
        else:
            post_pad = 0

        window = data[start:end]

        # Pad if necessary
        if pre_pad > 0 or post_pad > 0:
            pad_width = [(pre_pad, post_pad)] + [(0, 0)] * (window.ndim - 1)
            window = np.pad(window, pad_width, mode='edge')

        return window

    def _load_annotations(self, sample_info: Dict) -> np.ndarray:
        """Load frame-level annotations for a sample."""
        lab_id = sample_info['lab_id']
        video_id = sample_info['video_id']
        start_frame = sample_info['start_frame']
        agent_id = sample_info['agent_id']
        target_id = sample_info['target_id']
        metadata = sample_info['metadata']

        ann_path = self.annotation_dir / f"{lab_id}" / f"{video_id}.parquet"

        # Initialize labels
        labels = np.zeros((self.window_size, self.num_classes), dtype=np.float32)

        if not ann_path.exists():
            return labels

        ann_df = pd.read_parquet(ann_path)

        # Filter for this agent-target pair
        pair_anns = ann_df[
            (ann_df['agent_id'] == f"mouse{agent_id}") &
            ((ann_df['target_id'] == f"mouse{target_id}") |
             (ann_df['target_id'] == 'self'))
        ]

        # Get FPS for frame mapping
        fps = metadata.get('frames per second', metadata.get('fps', 30))

        for _, row in pair_anns.iterrows():
            action = row['action']
            if action not in self.behavior_to_idx:
                continue

            action_idx = self.behavior_to_idx[action]

            # Convert frames if resampled
            ann_start = row['start_frame']
            ann_stop = row['stop_frame']

            if fps != self.target_fps:
                ann_start = int(ann_start * self.target_fps / fps)
                ann_stop = int(ann_stop * self.target_fps / fps)

            # Map to window coordinates
            window_start = max(0, ann_start - start_frame)
            window_end = min(self.window_size, ann_stop - start_frame)

            if window_start < window_end:
                labels[window_start:window_end, action_idx] = 1.0

        return labels

    def _augment(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation."""
        # Temporal jittering
        if np.random.random() < 0.3:
            shift = np.random.randint(-5, 6)
            features = np.roll(features, shift, axis=0)
            labels = np.roll(labels, shift, axis=0)

        # Random horizontal flip
        if np.random.random() < 0.5:
            # Flip x coordinates (every other feature in flattened coords)
            n_coords = features.shape[-1] // 2
            for i in range(0, features.shape[-1], 2):
                features[:, i] = -features[:, i]

        # Add small noise
        if np.random.random() < 0.3:
            noise = np.random.normal(0, 0.01, features.shape).astype(np.float32)
            features = features + noise

        return features, labels


class MABeDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for MABe dataset.
    Handles train/val/test splits and dataloader creation.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        behaviors: Optional[List[str]] = None,
        batch_size: int = 8,
        num_workers: int = 4,
        window_size: int = 512,
        stride: int = 256,
        target_fps: float = 30.0,
        val_split: float = 0.2,
        stratify_by_lab: bool = True
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.behaviors = behaviors
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.window_size = window_size
        self.stride = stride
        self.target_fps = target_fps
        self.val_split = val_split
        self.stratify_by_lab = stratify_by_lab

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for each stage."""
        train_csv = self.data_dir / 'train.csv'
        test_csv = self.data_dir / 'test.csv'

        if stage == 'fit' or stage is None:
            train_df = pd.read_csv(train_csv)

            # Split train/val
            if self.stratify_by_lab:
                train_split, val_split = self._stratified_split(train_df)
            else:
                train_split, val_split = self._random_split(train_df)

            self.train_dataset = MABeDataset(
                metadata_df=train_split,
                tracking_dir=self.data_dir / 'train_tracking',
                annotation_dir=self.data_dir / 'train_annotation',
                behaviors=self.behaviors,
                window_size=self.window_size,
                stride=self.stride,
                target_fps=self.target_fps,
                augment=True,
                is_train=True
            )

            self.val_dataset = MABeDataset(
                metadata_df=val_split,
                tracking_dir=self.data_dir / 'train_tracking',
                annotation_dir=self.data_dir / 'train_annotation',
                behaviors=self.behaviors or self.train_dataset.behaviors,
                window_size=self.window_size,
                stride=self.window_size,  # No overlap for validation
                target_fps=self.target_fps,
                augment=False,
                is_train=False
            )

            # Update behaviors from training data
            if self.behaviors is None:
                self.behaviors = self.train_dataset.behaviors

        if stage == 'test' or stage is None:
            if test_csv.exists():
                test_df = pd.read_csv(test_csv)

                self.test_dataset = MABeDataset(
                    metadata_df=test_df,
                    tracking_dir=self.data_dir / 'test_tracking',
                    annotation_dir=None,
                    behaviors=self.behaviors,
                    window_size=self.window_size,
                    stride=self.window_size // 2,
                    target_fps=self.target_fps,
                    augment=False,
                    is_train=False
                )

    def _stratified_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data stratified by lab_id."""
        train_dfs = []
        val_dfs = []

        for lab_id in df['lab_id'].unique():
            lab_df = df[df['lab_id'] == lab_id]
            n_val = max(1, int(len(lab_df) * self.val_split))

            indices = np.random.permutation(len(lab_df))
            val_idx = indices[:n_val]
            train_idx = indices[n_val:]

            val_dfs.append(lab_df.iloc[val_idx])
            train_dfs.append(lab_df.iloc[train_idx])

        return pd.concat(train_dfs), pd.concat(val_dfs)

    def _random_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Random train/val split."""
        n_val = int(len(df) * self.val_split)
        indices = np.random.permutation(len(df))

        val_df = df.iloc[indices[:n_val]]
        train_df = df.iloc[indices[n_val:]]

        return train_df, val_df

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    @property
    def num_classes(self) -> int:
        return len(self.behaviors) if self.behaviors else 0

    @property
    def feature_dim(self) -> int:
        if self.train_dataset is not None:
            sample = self.train_dataset[0]
            return sample['features'].shape[-1]
        return None
