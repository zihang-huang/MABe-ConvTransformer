"""
Dataset and DataModule for MABe behavior recognition.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import random
import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict

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
from ..features import CombinedFeatureExtractor


def flatten_behaviors_config(behaviors: Optional[Union[Dict[str, List[str]], List[str]]]):
    """
    Convert behaviors config into a flat list of class names.

    The YAML stores behaviors grouped under "self" and "pair". The datasets
    expect a flat list, so we normalize here and also handle legacy dicts
    loaded from manifests.
    """
    if behaviors is None:
        return None
    if isinstance(behaviors, dict):
        flat: List[str] = []
        for group in ("self", "pair"):
            vals = behaviors.get(group, [])
            if isinstance(vals, list):
                flat.extend(vals)
        # Pick up any other keys just in case.
        for key, vals in behaviors.items():
            if key in ("self", "pair"):
                continue
            if isinstance(vals, list):
                flat.extend(vals)
            else:
                flat.append(vals)
        return flat
    return behaviors


def augment_tensors(
    features: torch.Tensor,
    labels: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Lightweight on-the-fly augmentation for precomputed tensors.

    Mirrors the numpy-based augmentation in MABeDataset.
    """
    # Temporal jitter
    if random.random() < 0.3:
        shift = random.randint(-5, 5)
        features = torch.roll(features, shifts=shift, dims=0)
        labels = torch.roll(labels, shifts=shift, dims=0)

    # Horizontal flip (negate x coordinates at even indices)
    if random.random() < 0.5:
        x_indices = torch.arange(0, features.shape[-1], 2, device=features.device)
        features[:, x_indices] = -features[:, x_indices]

    # Additive noise
    if random.random() < 0.3:
        noise = torch.randn_like(features) * 0.01
        features = features + noise

    return features, labels


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
        is_train: bool = True,
        tracking_cache_size: int = 4,
        annotation_cache_size: int = 8,
        # Engineered feature settings
        use_engineered_features: bool = False,
        feature_config: Optional[Dict] = None
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
            tracking_cache_size: Max number of videos to keep cached for tracking data
            annotation_cache_size: Max number of videos to cache annotations for
            use_engineered_features: Whether to compute engineered features
            feature_config: Configuration dict for feature extraction
        """
        self.tracking_dir = Path(tracking_dir)
        self.annotation_dir = Path(annotation_dir) if annotation_dir else None
        self.window_size = window_size
        self.stride = stride
        self.target_fps = target_fps
        self.normalize_coords = normalize_coords
        self.augment = augment
        self.is_train = is_train
        self.tracking_cache_size = max(1, tracking_cache_size)
        self.annotation_cache_size = max(1, annotation_cache_size)
        self.use_engineered_features = use_engineered_features
        self.feature_config = feature_config or {}

        # Initialize preprocessors
        self.coord_normalizer = CoordinateNormalizer()
        self.temporal_resampler = TemporalResampler(target_fps)
        self.missing_handler = MissingDataHandler()
        self.bodypart_mapper = BodyPartMapper(use_core_only=False)

        # Initialize feature extractor if enabled
        self.feature_extractor: Optional[CombinedFeatureExtractor] = None
        if self.use_engineered_features:
            self.feature_extractor = CombinedFeatureExtractor(
                body_parts=self.bodypart_mapper.target_parts,
                fps=target_fps,
                use_raw_coords=self.feature_config.get('use_raw_coords', True),
                use_single_mouse_features=self.feature_config.get('use_single_mouse', True),
                use_pairwise_features=self.feature_config.get('use_pairwise', True),
                use_temporal_features=self.feature_config.get('use_temporal', True),
                temporal_windows=self.feature_config.get('temporal_windows', [5, 15, 30, 60]),
                single_mouse_config=self.feature_config.get('single_mouse', {}),
                pairwise_config=self.feature_config.get('pair', {})
            )

        # In-memory caches to reduce parquet read overhead
        self._tracking_cache: OrderedDict = OrderedDict()
        self._annotation_cache: OrderedDict = OrderedDict()

        # Parse metadata
        self.metadata_df = metadata_df.copy()
        self.video_ids = metadata_df['video_id'].unique().tolist()
        self._annotation_intervals_cache: Dict[Tuple[str, str], Dict[Tuple[int, int], List[Tuple[int, int]]]] = {}

        # Build behavior vocabulary
        if behaviors is None:
            self.behaviors = self._collect_behaviors()
        else:
            self.behaviors = flatten_behaviors_config(behaviors)
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
            fps = self._get_fps(row)
            ann_intervals = None

            if self.annotation_dir is not None:
                ann_intervals = self._get_annotation_intervals(lab_id, video_id, fps)
                if not ann_intervals:
                    # Skip videos with no annotations when labels are required.
                    continue

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
                        if ann_intervals is not None:
                            pair_intervals = ann_intervals.get((int(agent), int(target)), [])
                            if not self._window_overlaps(start, start + self.window_size, pair_intervals):
                                continue
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
            'start_frame': sample_info['start_frame'],
            'lab_id': sample_info['lab_id']
        }

    def _read_parquet(self, path: Path) -> pd.DataFrame:
        """Read parquet using polars when available (faster) or pandas fallback."""
        if HAS_POLARS:
            return pls.read_parquet(path).to_pandas()
        return pd.read_parquet(path)

    def _get_cached_tracking(self, lab_id: str, video_id: str, metadata: Dict):
        """Load and cache tracking data for a video to avoid repeated disk I/O per sample."""
        key = (lab_id, video_id)
        if key in self._tracking_cache:
            self._tracking_cache.move_to_end(key)
            return self._tracking_cache[key]

        track_path = self.tracking_dir / f"{lab_id}" / f"{video_id}.parquet"
        if not track_path.exists():
            raise FileNotFoundError(f"Tracking file not found: {track_path}")

        track_df = self._read_parquet(track_path)
        fps = self._get_fps(metadata)
        bodyparts = sorted(track_df['bodypart'].unique().tolist())

        coords_by_mouse = {}
        valid_by_mouse = {}
        for mouse_id in track_df['mouse_id'].unique():
            raw_coords = self._extract_mouse_coords(track_df, mouse_id, bodyparts)
            mapped_coords, mapped_parts, availability = self.bodypart_mapper.map_bodyparts(raw_coords, bodyparts)
            mapped_coords = self.bodypart_mapper.compute_derived_parts(mapped_coords, mapped_parts, availability)

            if fps != self.target_fps:
                mapped_coords = self.temporal_resampler(mapped_coords, fps)

            if self.normalize_coords:
                mapped_coords = self.coord_normalizer(mapped_coords, metadata)

            mapped_coords, valid_mask = self.missing_handler.interpolate_missing(mapped_coords)
            mapped_coords = np.nan_to_num(mapped_coords, nan=0.0)

            coords_by_mouse[mouse_id] = mapped_coords.astype(np.float32)
            valid_by_mouse[mouse_id] = valid_mask.astype(np.float32)

        cache_entry = {
            'coords_by_mouse': coords_by_mouse,
            'valid_by_mouse': valid_by_mouse
        }
        self._tracking_cache[key] = cache_entry
        if len(self._tracking_cache) > self.tracking_cache_size:
            self._tracking_cache.popitem(last=False)
        return cache_entry

    def _get_cached_annotations(self, lab_id: str, video_id: str):
        """Load and cache annotation parquet to reduce read overhead."""
        if self.annotation_dir is None:
            return None

        key = (lab_id, video_id)
        if key in self._annotation_cache:
            self._annotation_cache.move_to_end(key)
            return self._annotation_cache[key]

        ann_path = self.annotation_dir / f"{lab_id}" / f"{video_id}.parquet"
        if not ann_path.exists():
            ann_df = None
        else:
            ann_df = self._read_parquet(ann_path)

            # Normalize identifier columns to integer form for reliable filtering.
            ann_df = ann_df.copy()
            ann_df['agent_id_int'] = ann_df['agent_id'].apply(self._normalize_id)
            ann_df['target_id_int'] = ann_df.apply(
                lambda r: self._normalize_id(r['target_id'], agent_fallback=r['agent_id_int']),
                axis=1
            )
            ann_df['start_frame'] = ann_df['start_frame'].astype(int)
            ann_df['stop_frame'] = ann_df['stop_frame'].astype(int)
            ann_df = ann_df.dropna(subset=['agent_id_int', 'target_id_int'])

        self._annotation_cache[key] = ann_df
        if len(self._annotation_cache) > self.annotation_cache_size:
            self._annotation_cache.popitem(last=False)
        return ann_df

    def _load_tracking(self, sample_info: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess tracking data for a sample."""
        lab_id = sample_info['lab_id']
        video_id = sample_info['video_id']
        start_frame = sample_info['start_frame']
        agent_id = sample_info['agent_id']
        target_id = sample_info['target_id']
        metadata = sample_info['metadata']

        cache_entry = self._get_cached_tracking(lab_id, video_id, metadata)
        coords_by_mouse = cache_entry['coords_by_mouse']
        valid_by_mouse = cache_entry['valid_by_mouse']

        agent_coords = coords_by_mouse.get(agent_id)
        target_coords = coords_by_mouse.get(target_id)
        agent_valid = valid_by_mouse.get(agent_id)
        target_valid = valid_by_mouse.get(target_id)

        if agent_coords is None or target_coords is None:
            raise ValueError(f"Missing coordinates for agent {agent_id} or target {target_id} in {video_id}")

        # Extract window
        end_frame = start_frame + self.window_size
        agent_window = self._get_window(agent_coords, start_frame, end_frame)
        target_window = self._get_window(target_coords, start_frame, end_frame)

        # Extract features
        if self.feature_extractor is not None:
            # Use engineered features
            features, _ = self.feature_extractor.extract_features(
                agent_window,
                target_window,
                include_temporal=self.feature_config.get('use_temporal', True)
            )
        else:
            # Fallback to raw coordinates only
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

    def _extract_mouse_coords(
        self,
        track_df: pd.DataFrame,
        mouse_id: int,
        bodyparts: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """Extract raw coordinates for a single mouse keyed by bodypart."""
        mouse_df = track_df[track_df['mouse_id'] == mouse_id].copy()

        # Pivot to get bodypart columns
        if bodyparts is None:
            bodyparts = mouse_df['bodypart'].unique()
        n_frames = track_df['video_frame'].max() + 1

        coords = {}

        for bp in bodyparts:
            bp_df = mouse_df[mouse_df['bodypart'] == bp].sort_values('video_frame')
            frames = bp_df['video_frame'].values
            part_coords = np.full((n_frames, 2), np.nan, dtype=np.float32)
            part_coords[frames, 0] = bp_df['x'].values
            part_coords[frames, 1] = bp_df['y'].values
            coords[bp] = part_coords

        return coords

    @staticmethod
    def _normalize_id(raw_id, agent_fallback: Optional[int] = None) -> Optional[int]:
        """Convert agent/target id fields to integer form."""
        if isinstance(raw_id, str):
            lowered = raw_id.strip().lower()
            if lowered.startswith('mouse'):
                try:
                    return int(''.join(filter(str.isdigit, lowered)))
                except ValueError:
                    return None
            if lowered == 'self' and agent_fallback is not None:
                return agent_fallback
            try:
                return int(lowered)
            except ValueError:
                return None
        try:
            return int(raw_id)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _get_fps(metadata: Dict) -> float:
        """Robustly read fps from metadata with fallbacks."""
        return metadata.get('frames_per_second',
                            metadata.get('frames per second',
                                         metadata.get('fps', 30)))

    @staticmethod
    def _window_overlaps(start: int, end: int, intervals: List[Tuple[int, int]]) -> bool:
        """Return True if [start, end) overlaps any interval."""
        if not intervals:
            return False
        for s, e in intervals:
            if s < end and e > start:
                return True
            if s >= end:
                break
        return False

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

    def _get_annotation_intervals(
        self,
        lab_id: str,
        video_id: str,
        fps: float
    ) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
        """
        Build cached annotation intervals per (agent, target) pair at target FPS.

        Returns:
            Dict mapping (agent_id, target_id) -> list of (start, end) frame tuples.
        """
        cache_key = (lab_id, video_id)
        if cache_key in self._annotation_intervals_cache:
            return self._annotation_intervals_cache[cache_key]

        ann_df = self._get_cached_annotations(lab_id, video_id)
        if ann_df is None or len(ann_df) == 0:
            self._annotation_intervals_cache[cache_key] = {}
            return {}

        intervals: Dict[Tuple[int, int], List[Tuple[int, int]]] = defaultdict(list)
        for _, row in ann_df.iterrows():
            agent_id = row.get('agent_id_int')
            target_id = row.get('target_id_int')
            if agent_id is None or target_id is None:
                continue

            ann_start = int(row['start_frame'])
            ann_stop = int(row['stop_frame'])

            if fps != self.target_fps:
                ann_start = int(ann_start * self.target_fps / fps)
                ann_stop = int(ann_stop * self.target_fps / fps)

            if ann_stop <= ann_start:
                continue

            intervals[(agent_id, target_id)].append((ann_start, ann_stop))

        # Sort and merge overlapping intervals for efficient overlap checks
        merged_intervals: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for pair, spans in intervals.items():
            sorted_spans = sorted(spans, key=lambda x: x[0])
            merged: List[Tuple[int, int]] = []
            for span in sorted_spans:
                if not merged or span[0] > merged[-1][1]:
                    merged.append((span[0], span[1]))
                else:
                    last_start, last_end = merged[-1]
                    merged[-1] = (last_start, max(last_end, span[1]))
            merged_intervals[pair] = [(int(s), int(e)) for s, e in merged]

        self._annotation_intervals_cache[cache_key] = merged_intervals
        return merged_intervals

    def _load_annotations(self, sample_info: Dict) -> np.ndarray:
        """Load frame-level annotations for a sample."""
        lab_id = sample_info['lab_id']
        video_id = sample_info['video_id']
        start_frame = sample_info['start_frame']
        agent_id = sample_info['agent_id']
        target_id = sample_info['target_id']
        metadata = sample_info['metadata']

        # Initialize labels
        labels = np.zeros((self.window_size, self.num_classes), dtype=np.float32)

        ann_df = self._get_cached_annotations(lab_id, video_id)
        if ann_df is None:
            return labels

        # Filter for this agent-target pair
        pair_anns = ann_df[
            (ann_df['agent_id_int'] == int(agent_id)) &
            (ann_df['target_id_int'] == int(target_id))
        ]

        # Get FPS for frame mapping
        fps = self._get_fps(metadata)

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
        test_split: float = 0.1,
        enable_test_split: bool = True,
        stratify_by_lab: bool = True,
        prefetch_factor: int = 4,
        tracking_cache_size: int = 4,
        annotation_cache_size: int = 8,
        use_precomputed: bool = False,
        precomputed_dir: Optional[Union[str, Path]] = None,
        # Oversampling settings for rare behaviors
        oversample_rare: bool = False,
        rare_behaviors: Optional[List[str]] = None,
        oversample_factor: int = 10,
        # Engineered feature settings
        use_engineered_features: bool = False,
        feature_config: Optional[Dict] = None
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.behaviors = flatten_behaviors_config(behaviors)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.window_size = window_size
        self.stride = stride
        self.target_fps = target_fps
        self.val_split = val_split
        self.test_split = test_split
        self.enable_test_split = enable_test_split
        self.stratify_by_lab = stratify_by_lab
        self.prefetch_factor = prefetch_factor
        self.tracking_cache_size = tracking_cache_size
        self.annotation_cache_size = annotation_cache_size
        self.use_precomputed = use_precomputed
        self.precomputed_dir = Path(precomputed_dir) if precomputed_dir else None
        # Oversampling settings
        self.oversample_rare = oversample_rare
        self.rare_behaviors = rare_behaviors or ['submit', 'chaseattack']
        self.oversample_factor = oversample_factor
        # Engineered feature settings
        self.use_engineered_features = use_engineered_features
        self.feature_config = feature_config or {}

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self._train_split_df = None
        self._val_split_df = None
        self._test_split_df = None

    def setup(
        self,
        stage: Optional[str] = None,
        use_precomputed: Optional[bool] = None,
        precomputed_dir: Optional[Union[str, Path]] = None
    ):
        """Set up datasets for each stage."""
        if use_precomputed is None:
            use_precomputed = self.use_precomputed
        else:
            self.use_precomputed = use_precomputed

        if precomputed_dir is None:
            precomputed_dir = self.precomputed_dir
        else:
            self.precomputed_dir = Path(precomputed_dir)

        train_csv = self.data_dir / 'train.csv'

        if stage == 'fit' or stage is None:
            if use_precomputed and self.precomputed_dir:
                pretrain_manifest = self.precomputed_dir / 'train' / 'manifest.json'
                preval_manifest = self.precomputed_dir / 'val' / 'manifest.json'
                pretest_manifest = self.precomputed_dir / 'test' / 'manifest.json'
                if pretrain_manifest.exists() and preval_manifest.exists():
                    try:
                        print(f"[data] Using precomputed shards from {self.precomputed_dir}")
                        self.train_dataset = PrecomputedWindowDataset(
                            root=self.precomputed_dir,
                            split='train',
                            apply_augment=True,
                            oversample_rare=self.oversample_rare,
                            rare_behaviors=self.rare_behaviors,
                            oversample_factor=self.oversample_factor
                        )
                        self.val_dataset = PrecomputedWindowDataset(
                            root=self.precomputed_dir,
                            split='val',
                            apply_augment=False
                        )
                        if self.enable_test_split and self.test_split > 0 and pretest_manifest.exists():
                            self.test_dataset = PrecomputedWindowDataset(
                                root=self.precomputed_dir,
                                split='test',
                                apply_augment=False
                            )
                        self.behaviors = self.train_dataset.behaviors
                        # Behaviors provided by manifest; skip raw loading.
                        return
                    except ValueError as e:
                        print(f"[data] Precomputed shards invalid: {e}. Falling back to raw data.")

            train_df = pd.read_csv(train_csv)

            # Reuse cached split when available to keep consistency across stages
            if self._train_split_df is None:
                train_split, val_split, test_split = self._build_splits(train_df)
                self._train_split_df, self._val_split_df, self._test_split_df = train_split, val_split, test_split
            else:
                train_split, val_split, test_split = self._train_split_df, self._val_split_df, self._test_split_df

            self.train_dataset = MABeDataset(
                metadata_df=train_split,
                tracking_dir=self.data_dir / 'train_tracking',
                annotation_dir=self.data_dir / 'train_annotation',
                behaviors=self.behaviors,
                window_size=self.window_size,
                stride=self.stride,
                target_fps=self.target_fps,
                augment=True,
                is_train=True,
                tracking_cache_size=self.tracking_cache_size,
                annotation_cache_size=self.annotation_cache_size,
                use_engineered_features=self.use_engineered_features,
                feature_config=self.feature_config
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
                is_train=False,
                tracking_cache_size=self.tracking_cache_size,
                annotation_cache_size=self.annotation_cache_size,
                use_engineered_features=self.use_engineered_features,
                feature_config=self.feature_config
            )

            self.test_dataset = None
            if test_split is not None and len(test_split) > 0:
                self.test_dataset = MABeDataset(
                    metadata_df=test_split,
                    tracking_dir=self.data_dir / 'train_tracking',
                    annotation_dir=self.data_dir / 'train_annotation',
                    behaviors=self.behaviors or self.train_dataset.behaviors,
                    window_size=self.window_size,
                    stride=self.window_size,  # No overlap for held-out test
                    target_fps=self.target_fps,
                    augment=False,
                    is_train=False,
                    tracking_cache_size=self.tracking_cache_size,
                    annotation_cache_size=self.annotation_cache_size,
                    use_engineered_features=self.use_engineered_features,
                    feature_config=self.feature_config
                )

            # Update behaviors from training data
            if self.behaviors is None:
                self.behaviors = self.train_dataset.behaviors

        if stage == 'test' or stage is None:
            if not self.enable_test_split or self.test_split <= 0:
                self.test_dataset = None
                return

            if use_precomputed and self.precomputed_dir:
                pretest_manifest = self.precomputed_dir / 'test' / 'manifest.json'
                if pretest_manifest.exists():
                    try:
                        print(f"[data] Using precomputed test shards from {self.precomputed_dir}")
                        self.test_dataset = PrecomputedWindowDataset(
                            root=self.precomputed_dir,
                            split='test',
                            apply_augment=False
                        )
                        return
                    except ValueError as e:
                        print(f"[data] Precomputed test shards invalid: {e}. Falling back to raw data.")

            # If already created during fit, keep it; otherwise build from cached split
            if self._test_split_df is None and train_csv.exists():
                # Build splits lazily when only stage='test' is requested
                train_df = pd.read_csv(train_csv)
                train_split, val_split, test_split = self._build_splits(train_df)
                self._train_split_df, self._val_split_df, self._test_split_df = train_split, val_split, test_split
                # Preserve behaviors
                if self.behaviors is None and train_split is not None:
                    tmp_dataset = MABeDataset(
                        metadata_df=train_split,
                        tracking_dir=self.data_dir / 'train_tracking',
                        annotation_dir=self.data_dir / 'train_annotation',
                        behaviors=None,
                        window_size=self.window_size,
                        stride=self.stride,
                        target_fps=self.target_fps,
                        augment=False,
                        is_train=True,
                        tracking_cache_size=self.tracking_cache_size,
                        annotation_cache_size=self.annotation_cache_size,
                        use_engineered_features=self.use_engineered_features,
                        feature_config=self.feature_config
                    )
                    self.behaviors = tmp_dataset.behaviors

            if self.test_dataset is None and self._test_split_df is not None and len(self._test_split_df) > 0:
                self.test_dataset = MABeDataset(
                    metadata_df=self._test_split_df,
                    tracking_dir=self.data_dir / 'train_tracking',
                    annotation_dir=self.data_dir / 'train_annotation',
                    behaviors=self.behaviors,
                    window_size=self.window_size,
                    stride=self.window_size,
                    target_fps=self.target_fps,
                    augment=False,
                    is_train=False,
                    tracking_cache_size=self.tracking_cache_size,
                    annotation_cache_size=self.annotation_cache_size,
                    use_engineered_features=self.use_engineered_features,
                    feature_config=self.feature_config
                )

    def _stratified_split_three(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train/val/test stratified by lab_id."""
        train_dfs = []
        val_dfs = []
        test_dfs = []

        for lab_id in df['lab_id'].unique():
            lab_df = df[df['lab_id'] == lab_id]
            n = len(lab_df)
            n_val = max(1, int(n * self.val_split))
            n_test = max(1, int(n * self.test_split))
            if n_val + n_test >= n:
                # Ensure at least one sample remains for training when possible
                n_test = max(1, n - n_val - 1)
            indices = np.random.permutation(n)
            val_idx = indices[:n_val]
            test_idx = indices[n_val:n_val + n_test]
            train_idx = indices[n_val + n_test:]
            val_dfs.append(lab_df.iloc[val_idx])
            test_dfs.append(lab_df.iloc[test_idx])
            if len(train_idx) > 0:
                train_dfs.append(lab_df.iloc[train_idx])

        return pd.concat(train_dfs), pd.concat(val_dfs), pd.concat(test_dfs)

    def _random_split_three(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Random train/val/test split."""
        n = len(df)
        n_val = int(n * self.val_split)
        n_test = int(n * self.test_split)
        if n_val + n_test >= n:
            n_test = max(1, n - n_val - 1)
        indices = np.random.permutation(n)
        val_df = df.iloc[indices[:n_val]]
        test_df = df.iloc[indices[n_val:n_val + n_test]]
        train_df = df.iloc[indices[n_val + n_test:]]
        return train_df, val_df, test_df

    def _stratified_split_two(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train/val stratified by lab_id (no held-out test)."""
        train_dfs = []
        val_dfs = []

        for lab_id in df['lab_id'].unique():
            lab_df = df[df['lab_id'] == lab_id]
            n = len(lab_df)
            n_val = max(1, int(n * self.val_split))
            if n_val >= n and n > 1:
                n_val = n - 1  # leave at least one sample for training when possible
            indices = np.random.permutation(n)
            val_idx = indices[:n_val]
            train_idx = indices[n_val:]
            if n_val > 0:
                val_dfs.append(lab_df.iloc[val_idx])
            if len(train_idx) > 0:
                train_dfs.append(lab_df.iloc[train_idx])

        val_df = pd.concat(val_dfs) if val_dfs else pd.DataFrame(columns=df.columns)
        train_df = pd.concat(train_dfs) if train_dfs else pd.DataFrame(columns=df.columns)
        return train_df, val_df

    def _random_split_two(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Randomly split data into train/val without creating a held-out test set."""
        n = len(df)
        n_val = max(1, int(n * self.val_split))
        if n_val >= n and n > 1:
            n_val = n - 1
        indices = np.random.permutation(n)
        val_df = df.iloc[indices[:n_val]]
        train_df = df.iloc[indices[n_val:]]
        return train_df, val_df

    def _build_splits(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Build train/val (and optionally test) splits based on configuration.
        Returns (train_df, val_df, test_df_or_None).
        """
        test_enabled = self.enable_test_split and self.test_split > 0
        if self.stratify_by_lab:
            if test_enabled:
                return self._stratified_split_three(df)
            train_split, val_split = self._stratified_split_two(df)
            return train_split, val_split, None

        if test_enabled:
            return self._random_split_three(df)
        train_split, val_split = self._random_split_two(df)
        return train_split, val_split, None

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=self.num_workers > 0,
            drop_last=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=self.num_workers > 0
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=self.num_workers > 0
        )

    @property
    def num_classes(self) -> int:
        if self.behaviors:
            return len(self.behaviors)
        if self.train_dataset is not None and hasattr(self.train_dataset, 'num_classes'):
            return self.train_dataset.num_classes
        return 0

    @property
    def feature_dim(self) -> int:
        if self.train_dataset is not None:
            if hasattr(self.train_dataset, 'feature_dim') and self.train_dataset.feature_dim is not None:
                return self.train_dataset.feature_dim
            sample = self.train_dataset[0]
            return sample['features'].shape[-1]
        return None


class PrecomputedWindowDataset(Dataset):
    """
    Dataset for loading precomputed windows stored as sharded torch files.

    Supports oversampling of rare behaviors to address class imbalance.
    """

    def __init__(
        self,
        root: Union[str, Path],
        split: str = 'train',
        apply_augment: bool = False,
        oversample_rare: bool = False,
        rare_behaviors: Optional[List[str]] = None,
        oversample_factor: int = 10
    ):
        """
        Args:
            root: Root directory containing precomputed shards
            split: Data split ('train', 'val', 'test')
            apply_augment: Whether to apply data augmentation
            oversample_rare: Whether to oversample windows containing rare behaviors
            rare_behaviors: List of behavior names to oversample (default: submit, chaseattack)
            oversample_factor: How many times to repeat rare behavior windows
        """
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.apply_augment = apply_augment
        self.oversample_rare = oversample_rare and split == 'train'  # Only oversample training
        self.rare_behaviors = rare_behaviors or ['submit', 'chaseattack']
        self.oversample_factor = oversample_factor

        manifest_path = self.root / split / 'manifest.json'
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found for split '{split}': {manifest_path}")

        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        self.window_size = manifest['window_size']
        self.feature_dim = manifest['feature_dim']
        self.num_classes = manifest['num_classes']
        behaviors_raw = manifest.get('behaviors', [])
        self.behaviors = flatten_behaviors_config(behaviors_raw) or []
        if self.behaviors and self.num_classes != len(self.behaviors):
            raise ValueError(
                f"Manifest {manifest_path} lists {self.num_classes} classes but "
                f"{len(self.behaviors)} behaviors after flattening. "
                "Regenerate precomputed shards with the fixed behavior handling."
            )
        self.shards = manifest['shards']

        # Load all shards into memory up front to avoid disk I/O during training.
        self.shards_data: List[Dict[str, Any]] = []
        for shard_info in self.shards:
            shard_path = self.root / self.split / shard_info['path']
            shard = torch.load(shard_path, map_location='cpu')
            self.shards_data.append(shard)

        # Build cumulative counts for original indexing
        self._cum_counts = []
        total = 0
        for shard in self.shards:
            total += shard['num_samples']
            self._cum_counts.append(total)

        self._base_length = self._cum_counts[-1] if self._cum_counts else 0

        # Build oversampled index if enabled
        self._oversampled_indices: Optional[List[int]] = None
        if self.oversample_rare and self.shards_data:
            self._build_oversampled_indices()

        if self.shards_data:
            total_samples = sum(s['num_samples'] for s in self.shards)
            effective_samples = len(self)
            if self._oversampled_indices:
                print(f"[data] Loaded {len(self.shards_data)} precomputed shards for {split} "
                      f"({total_samples} samples, {effective_samples} after oversampling).")
            else:
                print(f"[data] Loaded {len(self.shards_data)} precomputed shards for {split} into memory ({total_samples} samples).")

    def _build_oversampled_indices(self):
        """Build expanded index list that includes duplicates for rare behavior samples."""
        # Find indices of rare behaviors
        rare_behavior_indices = set()
        for behavior_name in self.rare_behaviors:
            if behavior_name in self.behaviors:
                rare_behavior_indices.add(self.behaviors.index(behavior_name))

        if not rare_behavior_indices:
            print(f"[data] Warning: No rare behaviors found in {self.behaviors}")
            return

        # Build expanded index list
        indices = []
        rare_count = 0
        normal_count = 0

        for global_idx in range(self._base_length):
            shard_idx, local_idx = self._locate_shard_base(global_idx)
            shard = self.shards_data[shard_idx]
            labels = shard['labels'][local_idx]

            # Check if this window contains any rare behavior
            has_rare = False
            for behavior_idx in rare_behavior_indices:
                if labels[:, behavior_idx].sum() > 0:  # Has positive frames for this behavior
                    has_rare = True
                    break

            if has_rare:
                # Add multiple copies for rare behavior windows
                for _ in range(self.oversample_factor):
                    indices.append(global_idx)
                rare_count += 1
            else:
                indices.append(global_idx)
                normal_count += 1

        self._oversampled_indices = indices
        print(f"[data] Oversampling: {rare_count} rare windows x{self.oversample_factor}, "
              f"{normal_count} normal windows. Total: {len(indices)}")

    def __len__(self) -> int:
        if self._oversampled_indices is not None:
            return len(self._oversampled_indices)
        return self._base_length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Map through oversampled indices if enabled
        if self._oversampled_indices is not None:
            if idx < 0 or idx >= len(self._oversampled_indices):
                raise IndexError(idx)
            actual_idx = self._oversampled_indices[idx]
        else:
            actual_idx = idx

        shard_idx, local_idx = self._locate_shard_base(actual_idx)
        shard = self._load_shard(shard_idx)

        features = shard['features'][local_idx].float()
        labels = shard['labels'][local_idx].float()
        valid_mask = shard['valid_mask'][local_idx].float()
        metadata = shard['metadata'][local_idx]

        if self.apply_augment:
            features, labels = augment_tensors(features, labels)

        return {
            'features': features,
            'labels': labels,
            'valid_mask': valid_mask,
            'video_id': metadata['video_id'],
            'agent_id': metadata['agent_id'],
            'target_id': metadata['target_id'],
            'start_frame': metadata['start_frame']
        }

    def _locate_shard_base(self, idx: int) -> Tuple[int, int]:
        """Locate shard and local index for a base (non-oversampled) index."""
        if idx < 0 or idx >= self._base_length:
            raise IndexError(idx)
        for i, end in enumerate(self._cum_counts):
            if idx < end:
                start = 0 if i == 0 else self._cum_counts[i - 1]
                return i, idx - start
        raise IndexError(idx)

    def _locate_shard(self, idx: int) -> Tuple[int, int]:
        """Locate shard for external index (handles oversampling)."""
        if self._oversampled_indices is not None:
            if idx < 0 or idx >= len(self._oversampled_indices):
                raise IndexError(idx)
            actual_idx = self._oversampled_indices[idx]
            return self._locate_shard_base(actual_idx)
        return self._locate_shard_base(idx)

    def _load_shard(self, shard_idx: int) -> Dict[str, Any]:
        return self.shards_data[shard_idx]
