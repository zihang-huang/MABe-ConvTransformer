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
import yaml
import torch

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


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_callbacks(config: dict) -> list:
    """Set up training callbacks."""
    callbacks = []

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['paths']['checkpoint_dir'],
        filename='mabe-{epoch:02d}-{val_f1:.4f}',
        monitor='val/f1',
        mode='max',
        save_top_k=3,
        save_last=True
    )
    callbacks.append(checkpoint_callback)

    # Early stopping
    if config['training'].get('early_stopping'):
        early_stop = EarlyStopping(
            monitor=config['training']['early_stopping']['monitor'],
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


def main(config_path: str = None, **overrides):
    """Main training function."""
    # Load configuration
    if config_path is None:
        config_path = project_root / 'configs' / 'config.yaml'
    config = load_config(config_path)

    # Apply overrides
    for key, value in overrides.items():
        keys = key.split('.')
        d = config
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = value

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
        prefetch_factor=config['training'].get('prefetch_factor', 4),
        tracking_cache_size=config['data'].get('tracking_cache_size', 4),
        annotation_cache_size=config['data'].get('annotation_cache_size', 8)
    )

    # Setup data to get dimensions
    data_module.setup('fit')

    # Compute class weights if needed
    class_weights = None
    if config['training'].get('class_weighting', 'none') != 'none':
        print("Computing class weights...")
        # This would require iterating through the dataset
        # For now, we'll skip this in the base implementation

    # Initialize model
    print(f"Initializing {config['model']['name']} model...")

    model_config = config['model'][config['model']['name']]
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
    trainer.fit(model, data_module)

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

    main(args.config, **overrides)
