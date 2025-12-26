# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Kaggle competition solution for **MABe 2.0** (Mouse Action Behavior Experiment) - a temporal action detection task that identifies 30+ social and non-social mouse behaviors from pose estimation keypoints. The model outputs behavior segments with (agent_id, target_id, action, start_frame, stop_frame) for pairs of mice.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Precompute windowed tensors (run once before training to speed up data loading)
python scripts/precompute_windows.py --config configs/config.yaml --output_dir ./kaggle/input/MABe-mouse-behavior-detection/precomputed

# Train model
python scripts/train.py --config configs/config.yaml
python scripts/train.py --use_precomputed  # Use precomputed shards for faster I/O

# Resume training from checkpoint
python scripts/train.py --ckpt_path outputs/checkpoints/last.ckpt

# Evaluate on validation set
python scripts/evaluate.py --checkpoint outputs/checkpoints/last.ckpt --split val --export_submission

# Run inference on test data
python scripts/inference.py --checkpoint outputs/checkpoints/last.ckpt --data_dir ./kaggle/input/MABe-mouse-behavior-detection --output submission.csv
```

## Architecture

### Data Pipeline
- **Input**: Parquet files with mouse keypoint coordinates (x, y per bodypart per frame)
- **Preprocessing** (`src/data/preprocessing.py`): Coordinate normalization, temporal resampling to 30fps, body part mapping, missing data interpolation
- **Feature Engineering** (`src/features/`): Engineered features computed on-the-fly or during precomputation
- **Dataset** (`src/data/dataset.py`):
  - `MABeDataset`: Loads raw parquet files with caching, generates sliding windows over agent-target pairs
  - `PrecomputedWindowDataset`: Loads pre-sharded .pt files for faster training
  - `MABeDataModule`: Lightning DataModule handling train/val/test splits

### Engineered Features (`src/features/`)

When `features.use_engineered_features: true` in config, the pipeline extracts rich behavioral features beyond raw coordinates:

- **Single Mouse Features** (`single_mouse.py`):
  - Position: centroid, bounding box dimensions
  - Velocity: speed, heading direction, angular velocity
  - Acceleration: linear and angular
  - Pose: body orientation (nose-to-tail angle), body length/width
  - Posture: body curvature, nose-tail angle

- **Pairwise Features** (`pairwise.py`):
  - Relative position: distance, angle to target
  - Relative velocity: approach/retreat speed, tangential speed
  - Facing analysis: is agent facing target, mutual facing detection
  - Contact proxies: min body part distance, bounding box overlap
  - Nose-to-body distances for social behavior detection

- **Temporal Features** (`temporal.py`):
  - Rolling statistics (mean, std, min, max, range) over configurable windows
  - Motion energy aggregation
  - Rate of change (first derivative)
  - Change point detection scores

Feature extraction is controlled via `configs/config.yaml`:
```yaml
features:
  use_engineered_features: true  # Master toggle
  use_raw_coords: true           # Include raw keypoint coordinates
  use_single_mouse: true         # Velocity, acceleration, orientation
  use_pairwise: true             # Distance, facing, relative motion
  use_temporal: true             # Rolling statistics
  temporal_windows: [5, 15, 30, 60]  # Frames for rolling stats
```

With all features enabled, input dimension increases from ~44 (raw coords only) to ~450 features.

### Models (`src/models/`)
Two architectures available, selected via `config.model.name`:
- **MS-TCN++** (`mstcn.py`): Multi-stage temporal convolutional network with refinement stages
- **TCN-Transformer** (`tcn_transformer.py`): Hybrid combining TCN for local patterns + Transformer for long-range dependencies + pairwise interaction module for agent-target modeling

### Training (`src/models/lightning_module.py`)
- `BehaviorRecognitionModule`: Main Lightning module supporting multi-label classification
- Loss: Focal BCE + temporal smoothing loss
- Validation: Segment-level F1 using Kaggle's `mouse_fbeta` metric
- Class imbalance: pos_neg_ratio weighting, optional oversampling of rare behaviors (submit, chaseattack)

### Post-processing (`src/utils/postprocessing.py`)
Frame predictions → segments:
1. Aggregate overlapping window predictions (averaging)
2. Per-behavior thresholding + median smoothing
3. Merge nearby segments
4. Non-maximum suppression
5. Resolve overlaps for same agent-target pairs

### Evaluation Metric (`src/utils/kaggle_metric.py`)
- `mouse_fbeta`: Kaggle's segment-level F-beta score that only evaluates behaviors listed in each video's `behaviors_labeled` field

## Key Configuration (configs/config.yaml)

```yaml
data:
  window_size: 512       # Frames per training sample
  stride: 256            # Sliding window stride
  use_precomputed: true  # Use precomputed shards
  oversample_rare: true  # Oversample rare behaviors

evaluation:
  threshold: 0.30        # Prediction threshold
  min_duration: 5        # Minimum segment frames
```

## Behaviors

11 self-behaviors (e.g., selfgroom, rear, freeze) + 26 pair behaviors (e.g., attack, chase, sniff).
Test-evaluated behaviors: approach, attack, avoid, chase, chaseattack, submit, rear.

## Data Layout

```
kaggle/input/MABe-mouse-behavior-detection/
├── train.csv, test.csv           # Video metadata
├── train_tracking/, test_tracking/  # Keypoint parquets by lab_id/video_id
├── train_annotation/             # Behavior labels (train only)
└── precomputed/                  # Precomputed .pt shards
    ├── train/manifest.json
    └── val/manifest.json
```
