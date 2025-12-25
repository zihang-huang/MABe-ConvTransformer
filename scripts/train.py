#!/usr/bin/env python3
"""
Main training script for MABe behavior recognition.

Usage:
    python scripts/train.py --config configs/config.yaml
    python scripts/train.py model=tcn_transformer training.batch_size=16
"""

import os
import sys
from pathlib import Path
from typing import Optional, Union

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import torch
from torch.serialization import add_safe_globals
import pandas as pd
import numpy as np
from omegaconf import OmegaConf

# Allow legacy numpy scalar pickles when loading weights-only checkpoints (PyTorch 2.6 safety change)
add_safe_globals([np.core.multiarray.scalar])

from src.data.dataset import MABeDataModule
from src.models.lightning_module import BehaviorRecognitionModule, compute_class_weights

# Use Tensor Cores on supported GPUs; trades some precision for speed.
torch.set_float32_matmul_precision("high")


class TrainingHeartbeat(pl.Callback):
    """Print periodic progress so long data passes don't look stalled."""

    def __init__(self, log_every_n_steps: int = 100):
        super().__init__()
        self.log_every_n_steps = max(1, log_every_n_steps)

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.is_global_zero:
            print(f"Epoch {trainer.current_epoch + 1}/{trainer.max_epochs} starting...")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step + 1
        if trainer.is_global_zero and step % self.log_every_n_steps == 0:
            print(f"[progress] global step {step} (epoch {trainer.current_epoch + 1})", flush=True)


def load_config(config_path: Union[str, Path], overrides: Optional[dict] = None) -> dict:
    """
    Load configuration from YAML file and resolve ${...} references using OmegaConf.
    Overrides are applied with dot-notation keys so dependent paths stay in sync.
    """
    conf = OmegaConf.load(config_path)

    if overrides:
        for key, value in overrides.items():
            OmegaConf.update(conf, key, value, merge=True)

    return OmegaConf.to_container(conf, resolve=True)


def collect_class_counts_from_annotations(dataset, target_fps: float) -> Optional[np.ndarray]:
    """
    Approximate per-class frame counts directly from annotation parquet files.
    This keeps weighting logic lightweight while avoiding a full dataset pass.
    """
    if not hasattr(dataset, "metadata_df") or dataset.annotation_dir is None:
        return None

    counts = np.zeros(dataset.num_classes, dtype=np.float64)
    behavior_to_idx = dataset.behavior_to_idx
    ann_dir = dataset.annotation_dir

    for _, meta in dataset.metadata_df.iterrows():
        ann_path = ann_dir / f"{meta['lab_id']}" / f"{meta['video_id']}.parquet"
        if not ann_path.exists():
            continue

        ann_df = pd.read_parquet(ann_path)
        if ann_df.empty:
            continue

        fps = dataset._get_fps(meta.to_dict()) if hasattr(dataset, "_get_fps") else target_fps
        fps = fps if fps else target_fps
        scale = target_fps / fps if fps > 0 else 1.0

        for _, row in ann_df.iterrows():
            idx = behavior_to_idx.get(row['action'])
            if idx is None:
                continue
            start = int(row['start_frame'] * scale)
            stop = int(row['stop_frame'] * scale)
            if stop <= start:
                continue
            counts[idx] += max(1, stop - start)

    return counts


def collect_class_counts_from_precomputed(dataset) -> tuple:
    """
    Sum label activations across precomputed shards to estimate class frequency.
    Returns (counts, total_samples) tuple.
    """
    if not hasattr(dataset, "shards"):
        return None, 0

    counts = torch.zeros(dataset.num_classes, dtype=torch.float64)
    total_samples = 0

    if hasattr(dataset, "shards_data") and dataset.shards_data:
        loaded_shards = dataset.shards_data
    else:
        split_root = dataset.root / dataset.split
        loaded_shards = []
        for shard_info in dataset.shards:
            shard_path = split_root / shard_info['path']
            loaded_shards.append(torch.load(shard_path, map_location='cpu'))

    for shard in loaded_shards:
        counts += shard['labels'].sum(dim=(0, 1)).double()
        total_samples += shard['labels'].shape[0] * shard['labels'].shape[1]

    return counts.numpy(), total_samples


def build_class_weights(train_dataset, target_fps: float, weighting: str, beta: float):
    """
    Build class weights using the configured strategy.
    """
    if weighting == 'none' or train_dataset is None:
        return None

    counts, total_samples = collect_class_counts_from_precomputed(train_dataset)
    if counts is None:
        counts = collect_class_counts_from_annotations(train_dataset, target_fps)
        total_samples = None  # Will be estimated in compute_class_weights

    if counts is None or counts.sum() == 0:
        print("[warn] Unable to compute class counts for weighting; proceeding without.")
        return None

    weights = compute_class_weights(counts, method=weighting, beta=beta, is_counts=True, total_samples=total_samples)
    print(f"[info] Class weights computed (min={weights.min().item():.4f}, max={weights.max().item():.4f})")
    return weights


def setup_callbacks(config: dict) -> list:
    """Set up training callbacks."""
    callbacks = []

    # Determine which metric to use for checkpointing and early stopping
    eval_config = config.get('evaluation', {})
    use_segment_f1 = eval_config.get('segment_eval', True)

    if use_segment_f1:
        # Use Kaggle-compatible segment-level F1
        monitor_metric = 'val/segment_f1'
        filename_pattern = 'mabe-{epoch:02d}-{val_segment_f1:.4f}'
    else:
        # Use frame-level F1
        monitor_metric = 'val/f1'
        filename_pattern = 'mabe-{epoch:02d}-{val_f1:.4f}'

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['paths']['checkpoint_dir'],
        filename=filename_pattern,
        monitor=monitor_metric,
        mode='max',
        save_top_k=3,
        save_last=True
    )
    callbacks.append(checkpoint_callback)

    # Early stopping
    if config['training'].get('early_stopping'):
        # Override monitor if using segment-level F1
        early_stop_monitor = config['training']['early_stopping']['monitor']
        if use_segment_f1 and early_stop_monitor == 'val/f1':
            early_stop_monitor = 'val/segment_f1'

        early_stop = EarlyStopping(
            monitor=early_stop_monitor,
            mode=config['training']['early_stopping']['mode'],
            patience=config['training']['early_stopping']['patience'],
            verbose=True
        )
        callbacks.append(early_stop)

    # Learning rate monitor
    callbacks.append(LearningRateMonitor(logging_interval='epoch'))

    # Progress bar
    callbacks.append(RichProgressBar())
    callbacks.append(TrainingHeartbeat(
        log_every_n_steps=config['training'].get('progress_heartbeat_steps', 100)
    ))

    return callbacks


def setup_logger(config: dict):
    """Set up experiment logger."""
    if config['logging'].get('use_wandb', False):
        logger = WandbLogger(
            project=config['logging']['project'],
            name=config['logging'].get('run_name'),
            save_dir=config['paths']['output_dir']
        )
    else:
        logger = TensorBoardLogger(
            save_dir=config['paths']['output_dir'],
            name=config['logging']['project']
        )
    return logger


def main(config_path: str = None, ckpt_path: Optional[str] = None, **overrides):
    """Main training function."""
    # Load configuration
    if config_path is None:
        config_path = project_root / 'configs' / 'config.yaml'
    config = load_config(config_path, overrides)

    # Set random seed
    pl.seed_everything(42)

    # Create output directories
    os.makedirs(config['paths']['output_dir'], exist_ok=True)
    os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)

    # Initialize data module
    print("Initializing data module...")
    data_module = MABeDataModule(
        data_dir=config['paths']['data_dir'],
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        window_size=config['data']['window_size'],
        stride=config['data']['stride'],
        target_fps=config['data']['target_fps'],
        val_split=config['data'].get('val_split', 0.2),
        test_split=config['data'].get('test_split', 0.1),
        prefetch_factor=config['training'].get('prefetch_factor', 4),
        tracking_cache_size=config['data'].get('tracking_cache_size', 4),
        annotation_cache_size=config['data'].get('annotation_cache_size', 8),
        use_precomputed=config['data'].get('use_precomputed', False),
        precomputed_dir=config['data'].get('precomputed_dir')
    )

    # Setup data to get dimensions
    data_module.setup('fit')

    # Compute class weights if needed
    weighting = config['training'].get('class_weighting', 'none')
    beta = config['training'].get('effective_num_beta', 0.9999)
    class_weights = build_class_weights(
        data_module.train_dataset,
        target_fps=config['data']['target_fps'],
        weighting=weighting,
        beta=beta
    )

    # Initialize model
    print(f"Initializing {config['model']['name']} model...")

    model_config = config['model'][config['model']['name']]
    eval_config = config.get('evaluation', {})
    data_dir = Path(config['paths']['data_dir'])

    model = BehaviorRecognitionModule(
        model_name=config['model']['name'],
        input_dim=data_module.feature_dim,
        num_classes=data_module.num_classes,
        behaviors=data_module.behaviors,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        warmup_epochs=config['training']['scheduler']['warmup_epochs'],
        max_epochs=config['training']['max_epochs'],
        smoothing_weight=config['training']['loss']['smoothness_weight'],
        class_weights=class_weights,
        eval_threshold=eval_config.get('threshold', 0.5),
        use_focal_loss=config['training'].get('use_focal_loss', False),
        focal_gamma=config['training'].get('focal_gamma', 2.0),
        # Segment-level evaluation settings for Kaggle metric
        segment_eval=eval_config.get('segment_eval', True),
        min_segment_duration=eval_config.get('min_duration', 5),
        smoothing_kernel=eval_config.get('smoothing_kernel', 5),
        nms_threshold=eval_config.get('nms_threshold', 0.3),
        merge_gap=eval_config.get('merge_gap', 5),
        annotation_dir=str(data_dir / 'train_annotation'),
        metadata_csv=str(data_dir / 'train.csv'),
        **model_config
    )

    # Setup callbacks and logger
    callbacks = setup_callbacks(config)
    logger = setup_logger(config)

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator='auto',
        devices=1,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        precision='16-mixed' if torch.cuda.is_available() else 32,
        log_every_n_steps=10,
        val_check_interval=1.0,
        benchmark=True
    )

    # Train
    print("Starting training...")
    trainer.fit(model, data_module, ckpt_path=ckpt_path)

    # Test with best checkpoint
    print("Testing with best checkpoint...")
    if data_module.test_dataset is not None:
        trainer.test(model, data_module, ckpt_path='best')

    print("Training complete!")
    print(f"Best model saved to: {callbacks[0].best_model_path}")

    return model, trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MABe behavior recognition model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--data_dir', type=str, help='Override data directory')
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    parser.add_argument('--epochs', type=int, help='Override max epochs')
    parser.add_argument('--lr', type=float, help='Override learning rate')
    parser.add_argument('--model', type=str, choices=['mstcn', 'tcn_transformer'],
                        help='Override model type')
    parser.add_argument('--use_precomputed', action='store_true',
                        help='Use precomputed shards instead of raw parquet')
    parser.add_argument('--precomputed_dir', type=str, help='Directory with precomputed shards')
    parser.add_argument('--ckpt_path', type=str, help='Path to checkpoint to resume training from')

    args = parser.parse_args()

    # Build overrides
    overrides = {}
    if args.data_dir:
        overrides['paths.data_dir'] = args.data_dir
    if args.batch_size:
        overrides['training.batch_size'] = args.batch_size
    if args.epochs:
        overrides['training.max_epochs'] = args.epochs
    if args.lr:
        overrides['training.learning_rate'] = args.lr
    if args.model:
        overrides['model.name'] = args.model
    if args.use_precomputed:
        overrides['data.use_precomputed'] = True
    if args.precomputed_dir:
        overrides['data.precomputed_dir'] = args.precomputed_dir

    main(args.config, ckpt_path=args.ckpt_path, **overrides)
