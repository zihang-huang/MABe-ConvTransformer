"""
PyTorch Lightning module for behavior recognition training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

from .mstcn import MSTCN, MSTCNLoss
from .tcn_transformer import TCNTransformer, TCNTransformerLoss
from ..utils.postprocessing import (
    aggregate_window_predictions,
    extract_segments,
    merge_segments,
    apply_nms,
    BehaviorSegment,
)
from ..utils.kaggle_metric import mouse_fbeta


class BehaviorRecognitionModule(pl.LightningModule):
    """
    PyTorch Lightning module for training behavior recognition models.

    Supports:
    - MS-TCN++ and TCN-Transformer architectures
    - Multi-label behavior classification
    - Class imbalance handling
    - Domain adaptation (optional)
    """

    def __init__(
        self,
        model_name: str = 'mstcn',
        input_dim: int = 256,
        num_classes: int = 37,
        behaviors: Optional[List[str]] = None,
        learning_rate: float = 0.0005,
        weight_decay: float = 0.0001,
        warmup_epochs: int = 5,
        max_epochs: int = 100,
        smoothing_weight: float = 0.15,
        class_weights: Optional[torch.Tensor] = None,
        eval_threshold: float = 0.5,
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0,
        # Segment-level evaluation settings
        segment_eval: bool = True,
        min_segment_duration: int = 5,
        smoothing_kernel: int = 5,
        nms_threshold: float = 0.3,
        merge_gap: int = 5,
        annotation_dir: Optional[str] = None,
        metadata_csv: Optional[str] = None,
        # MS-TCN specific
        num_stages: int = 4,
        num_layers: int = 10,
        num_f_maps: int = 64,
        # TCN-Transformer specific
        tcn_channels: List[int] = [64, 128, 256],
        transformer_dim: int = 256,
        transformer_heads: int = 8,
        transformer_layers: int = 2,
        dropout: float = 0.5,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['annotation_dir', 'metadata_csv'])

        self.model_name = model_name
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.behaviors = behaviors or [f'behavior_{i}' for i in range(num_classes)]
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eval_threshold = eval_threshold

        # Build model
        if model_name == 'mstcn':
            self.model = MSTCN(
                input_dim=input_dim,
                num_classes=num_classes,
                num_stages=num_stages,
                num_layers=num_layers,
                num_f_maps=num_f_maps,
                dropout=dropout
            )
            self.criterion = MSTCNLoss(
                num_classes=num_classes,
                smoothing_weight=smoothing_weight,
                class_weights=class_weights,
                use_focal_loss=use_focal_loss,
                focal_gamma=focal_gamma
            )
        elif model_name == 'tcn_transformer':
            self.model = TCNTransformer(
                input_dim=input_dim,
                num_classes=num_classes,
                tcn_channels=tcn_channels,
                transformer_dim=transformer_dim,
                transformer_heads=transformer_heads,
                transformer_layers=transformer_layers,
                dropout=dropout
            )
            self.criterion = TCNTransformerLoss(
                num_classes=num_classes,
                smoothing_weight=smoothing_weight,
                class_weights=class_weights
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Metrics tracking
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)

        # Segment-level evaluation settings
        self.segment_eval = segment_eval
        self.min_segment_duration = min_segment_duration
        self.smoothing_kernel = smoothing_kernel
        self.nms_threshold = nms_threshold
        self.merge_gap = merge_gap
        self.annotation_dir = Path(annotation_dir) if annotation_dir else None
        self.metadata_csv = Path(metadata_csv) if metadata_csv else None
        self._solution_df = None  # Cached solution DataFrame

        # Storage for validation predictions (reset each epoch)
        self.val_predictions: List[Dict] = []

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass returning predictions."""
        if self.model_name == 'mstcn':
            predictions, _ = self.model(x, mask)
        else:
            predictions, _ = self.model(x, mask)
        return predictions

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        features = batch['features']
        labels = batch['labels']
        mask = batch.get('valid_mask', None)

        if self.model_name == 'mstcn':
            predictions, stage_outputs = self.model(features, mask)
            loss, loss_dict = self.criterion(predictions, stage_outputs, labels, mask)
        else:
            predictions, _ = self.model(features, mask)
            loss, loss_dict = self.criterion(predictions, labels, mask)

        # Log losses
        for key, value in loss_dict.items():
            self.log(f'train/{key}', value, prog_bar=(key == 'total_loss'))

        # Compute metrics
        with torch.no_grad():
            metrics = self._compute_metrics(predictions, labels, mask)
            for key, value in metrics.items():
                self.log(f'train/{key}', value)

        return loss

    def on_validation_epoch_start(self):
        """Clear validation predictions at the start of each epoch."""
        self.val_predictions = []

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict:
        """Validation step."""
        features = batch['features']
        labels = batch['labels']
        mask = batch.get('valid_mask', None)

        if self.model_name == 'mstcn':
            predictions, stage_outputs = self.model(features, mask)
            loss, loss_dict = self.criterion(predictions, stage_outputs, labels, mask)
        else:
            predictions, _ = self.model(features, mask)
            loss, loss_dict = self.criterion(predictions, labels, mask)

        # Log losses
        for key, value in loss_dict.items():
            self.log(f'val/{key}', value, prog_bar=(key == 'total_loss'))

        # Compute frame-level metrics
        metrics = self._compute_metrics(predictions, labels, mask)
        for key, value in metrics.items():
            self.log(f'val/{key}', value, prog_bar=(key in ['f1', 'accuracy']))
        # Provide slash-free alias so checkpoint filenames can include F1 without defaulting to 0
        if 'f1' in metrics:
            self.log('val_f1', metrics['f1'], prog_bar=True)

        # Store predictions for segment-level evaluation
        if self.segment_eval:
            probs = torch.sigmoid(predictions).detach().cpu().numpy()
            batch_size = probs.shape[0]
            for i in range(batch_size):
                self.val_predictions.append({
                    'probabilities': probs[i],
                    'video_id': self._to_scalar(batch['video_id'][i]),
                    'agent_id': self._to_scalar(batch['agent_id'][i]),
                    'target_id': self._to_scalar(batch['target_id'][i]),
                    'start_frame': self._to_scalar(batch['start_frame'][i]),
                    'lab_id': batch['lab_id'][i] if 'lab_id' in batch else None,
                })

        return {
            'loss': loss,
            'predictions': predictions.detach(),
            'labels': labels.detach()
        }

    def _to_scalar(self, value):
        """Convert tensor or numpy array to Python scalar."""
        if isinstance(value, torch.Tensor):
            return value.item() if value.numel() == 1 else value.cpu().numpy()
        elif isinstance(value, np.ndarray):
            return value.item() if value.size == 1 else value
        return value

    def on_validation_epoch_end(self):
        """Compute segment-level F1 at the end of each validation epoch."""
        if not self.segment_eval or not self.val_predictions:
            return

        try:
            segment_f1 = self._compute_segment_f1()
            if segment_f1 is not None:
                self.log('val/segment_f1', segment_f1, prog_bar=True, sync_dist=True)
                self.log('val_segment_f1', segment_f1, prog_bar=True, sync_dist=True)
        except Exception as e:
            # Log warning but don't crash training
            if self.trainer.is_global_zero:
                print(f"[warn] Segment F1 computation failed: {e}")

    def _compute_segment_f1(self) -> Optional[float]:
        """Compute Kaggle-compatible segment-level F1."""
        if not self.val_predictions:
            return None

        # Aggregate window predictions
        aggregated = aggregate_window_predictions(
            self.val_predictions, overlap_strategy='average'
        )

        # Convert to segments
        all_segments: List[BehaviorSegment] = []
        for (video_id, agent_id, target_id), frame_probs in aggregated.items():
            raw_segments = extract_segments(
                frame_probs,
                self.behaviors,
                threshold=self.eval_threshold,
                min_duration=self.min_segment_duration,
                smoothing_kernel=self.smoothing_kernel,
            )
            merged = merge_segments(raw_segments, gap_threshold=self.merge_gap)
            final_segments = apply_nms(merged, iou_threshold=self.nms_threshold)

            # Format agent_id and target_id as 'mouseX' or 'self'
            formatted_agent = self._format_mouse_id(agent_id, allow_self=False)
            formatted_target = self._format_mouse_id(target_id, allow_self=True)

            for behavior, start, stop, conf in final_segments:
                all_segments.append(BehaviorSegment(
                    video_id=int(self._to_scalar(video_id)),
                    agent_id=formatted_agent,
                    target_id=formatted_target,
                    action=behavior,
                    start_frame=int(start),
                    stop_frame=int(stop),
                    confidence=float(conf),
                ))

        # Resolve overlaps
        all_segments = self._resolve_overlaps(all_segments)

        if not all_segments:
            return 0.0

        # Build submission DataFrame
        submission_rows = []
        for row_id, seg in enumerate(all_segments):
            if seg.duration >= self.min_segment_duration:
                submission_rows.append({
                    'video_id': seg.video_id,
                    'agent_id': seg.agent_id,
                    'target_id': seg.target_id,
                    'action': seg.action,
                    'start_frame': seg.start_frame,
                    'stop_frame': seg.stop_frame,
                })
        submission_df = pd.DataFrame(submission_rows)

        # Load solution DataFrame
        solution_df = self._get_solution_df()
        if solution_df is None or solution_df.empty:
            return None

        # Filter solution to validation videos
        val_videos = set(submission_df['video_id'].unique())
        solution_df = solution_df[solution_df['video_id'].isin(val_videos)]

        if solution_df.empty or submission_df.empty:
            return 0.0

        # Compute Kaggle metric
        return mouse_fbeta(solution_df, submission_df, beta=1.0)

    def _format_mouse_id(self, mouse_id, allow_self: bool = False) -> str:
        """Format mouse ID as 'mouseX' or 'self'."""
        if isinstance(mouse_id, str):
            if mouse_id.startswith('mouse') or mouse_id == 'self':
                return mouse_id
            try:
                mouse_int = int(mouse_id)
            except ValueError:
                return mouse_id
        elif isinstance(mouse_id, (int, np.integer)):
            mouse_int = int(mouse_id)
        else:
            return str(mouse_id)

        # Check for self-behavior (agent_id == target_id)
        if allow_self and mouse_int == -1:
            return 'self'

        return f'mouse{mouse_int}'

    def _resolve_overlaps(self, segments: List[BehaviorSegment]) -> List[BehaviorSegment]:
        """Resolve overlapping behaviors for same agent-target pair."""
        from collections import defaultdict

        groups = defaultdict(list)
        for seg in segments:
            key = (seg.video_id, seg.agent_id, seg.target_id)
            groups[key].append(seg)

        resolved = []
        for key, group_segments in groups.items():
            group_segments.sort(key=lambda s: (s.start_frame, s.stop_frame))
            last_end = -1

            for seg in group_segments:
                new_start = max(seg.start_frame, last_end)
                new_stop = seg.stop_frame

                if new_start < new_stop and (new_stop - new_start) >= self.min_segment_duration:
                    resolved.append(BehaviorSegment(
                        video_id=seg.video_id,
                        agent_id=seg.agent_id,
                        target_id=seg.target_id,
                        action=seg.action,
                        start_frame=new_start,
                        stop_frame=new_stop,
                        confidence=seg.confidence,
                    ))
                    last_end = new_stop

        return resolved

    def _get_solution_df(self) -> Optional[pd.DataFrame]:
        """Load or return cached solution DataFrame for Kaggle metric."""
        if self._solution_df is not None:
            return self._solution_df

        if self.annotation_dir is None or self.metadata_csv is None:
            return None

        if not self.annotation_dir.exists() or not self.metadata_csv.exists():
            return None

        try:
            metadata_df = pd.read_csv(self.metadata_csv)
            all_rows = []

            for _, meta in metadata_df.iterrows():
                lab_id = meta['lab_id']
                video_id = meta['video_id']
                behaviors_labeled = meta.get('behaviors_labeled', '[]')

                ann_path = self.annotation_dir / str(lab_id) / f"{video_id}.parquet"
                if not ann_path.exists():
                    continue

                ann_df = pd.read_parquet(ann_path)
                for _, row in ann_df.iterrows():
                    agent_id = f"mouse{row['agent_id']}"
                    target_id = f"mouse{row['target_id']}" if row['target_id'] != row['agent_id'] else "self"

                    all_rows.append({
                        'video_id': video_id,
                        'agent_id': agent_id,
                        'target_id': target_id,
                        'action': row['action'],
                        'start_frame': row['start_frame'],
                        'stop_frame': row['stop_frame'],
                        'lab_id': lab_id,
                        'behaviors_labeled': behaviors_labeled,
                    })

            if all_rows:
                self._solution_df = pd.DataFrame(all_rows)
            return self._solution_df
        except Exception as e:
            print(f"[warn] Failed to load solution DataFrame: {e}")
            return None

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict:
        """Test step."""
        return self.validation_step(batch, batch_idx)

    def predict_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Prediction step for inference."""
        features = batch['features']
        mask = batch.get('valid_mask', None)

        predictions = self.forward(features, mask)

        # Convert to probabilities
        probs = torch.sigmoid(predictions)

        return {
            'predictions': predictions,
            'probabilities': probs,
            'video_id': batch['video_id'],
            'agent_id': batch['agent_id'],
            'target_id': batch['target_id'],
            'start_frame': batch['start_frame']
        }

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # Cosine annealing with warmup
        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return epoch / self.warmup_epochs
            else:
                progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }

    def _compute_metrics(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        # Convert to probabilities
        probs = torch.sigmoid(predictions)

        # Binary predictions
        preds = (probs > self.eval_threshold).float()

        mask_factor = None
        if mask is not None:
            mask = mask.unsqueeze(-1).expand_as(preds)
            preds = preds * mask
            labels = labels * mask
            mask_factor = mask
            n_valid = mask.sum().clamp_min(1e-8)
        else:
            n_valid = torch.tensor(preds.numel(), device=preds.device, dtype=preds.dtype)

        # Accuracy
        correct = (preds == labels).float()
        if mask_factor is not None:
            correct = correct * mask_factor
        accuracy = correct.sum() / n_valid

        # Per-class metrics
        if mask_factor is None:
            tp = (preds * labels).sum(dim=(0, 1))
            fp = (preds * (1 - labels)).sum(dim=(0, 1))
            fn = ((1 - preds) * labels).sum(dim=(0, 1))
            label_support = labels.sum(dim=(0, 1))
        else:
            tp = (preds * labels * mask_factor).sum(dim=(0, 1))
            fp = (preds * (1 - labels) * mask_factor).sum(dim=(0, 1))
            fn = ((1 - preds) * labels * mask_factor).sum(dim=(0, 1))
            label_support = (labels * mask_factor).sum(dim=(0, 1))

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        # Macro averages; ignore classes with no positives to avoid diluting with absent classes
        present_mask = label_support > 0
        if present_mask.any():
            macro_precision = precision[present_mask].mean()
            macro_recall = recall[present_mask].mean()
            macro_f1 = f1[present_mask].mean()
        else:
            macro_precision = precision.mean()
            macro_recall = recall.mean()
            macro_f1 = f1.mean()

        return {
            'accuracy': accuracy.item(),
            'precision': macro_precision.item(),
            'recall': macro_recall.item(),
            'f1': macro_f1.item()
        }


class DomainAdversarialModule(pl.LightningModule):
    """
    Domain-adversarial training for lab-invariant features.
    Uses gradient reversal layer to train domain-invariant encoder.
    """

    def __init__(
        self,
        base_module: BehaviorRecognitionModule,
        num_domains: int,
        domain_weight: float = 0.1
    ):
        super().__init__()

        self.base_module = base_module
        self.num_domains = num_domains
        self.domain_weight = domain_weight

        # Domain classifier
        hidden_dim = base_module.model.transformer.output_dim if hasattr(base_module.model, 'transformer') else 256
        self.domain_classifier = nn.Sequential(
            GradientReversalLayer(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_domains)
        )

    def forward(self, x, mask=None):
        return self.base_module(x, mask)

    def training_step(self, batch, batch_idx):
        features = batch['features']
        labels = batch['labels']
        domain_labels = batch['lab_id']  # Numeric lab IDs
        mask = batch.get('valid_mask', None)

        # Get model predictions and intermediate features
        if self.base_module.model_name == 'mstcn':
            predictions, stage_outputs = self.base_module.model(features, mask)
            loss, loss_dict = self.base_module.criterion(predictions, stage_outputs, labels, mask)
            # Use last stage features for domain classification
            domain_features = stage_outputs[-1].mean(dim=1)  # (batch, num_classes)
        else:
            predictions, intermediates = self.base_module.model(features, mask)
            loss, loss_dict = self.base_module.criterion(predictions, labels, mask)
            # Use transformer output for domain classification
            domain_features = intermediates[-1].mean(dim=1)  # (batch, hidden_dim)

        # Domain classification loss
        domain_logits = self.domain_classifier(domain_features)
        domain_loss = F.cross_entropy(domain_logits, domain_labels)

        total_loss = loss + self.domain_weight * domain_loss

        self.log('train/domain_loss', domain_loss)
        self.log('train/total_loss', total_loss)

        return total_loss


class GradientReversalLayer(torch.autograd.Function):
    """Gradient reversal layer for domain-adversarial training."""

    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    """Module wrapper for gradient reversal."""

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def compute_class_weights(
    labels: np.ndarray,
    method: str = 'effective_num',
    beta: float = 0.9999,
    is_counts: bool = False,
    total_samples: Optional[int] = None
) -> torch.Tensor:
    """
    Compute class weights for imbalanced data.

    For multi-label classification with BCE loss, this computes pos_weight
    which balances positive vs negative examples for each class.

    Args:
        labels: Array of labels or precomputed class counts (positive counts per class)
        method: 'inverse', 'effective_num', 'sqrt_inverse', or 'pos_neg_ratio'
        beta: Beta parameter for effective number weighting
        is_counts: Treat `labels` as already-aggregated class counts when True
        total_samples: Total number of samples (frames), needed for pos_neg_ratio method

    Returns:
        Class weight tensor (pos_weight for BCE loss)
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    label_array = np.asarray(labels)

    # Count samples per class
    if is_counts:
        class_counts = label_array.astype(np.float64)
    elif label_array.ndim > 1:
        class_counts = label_array.sum(axis=0)
    elif np.issubdtype(label_array.dtype, np.integer):
        class_counts = np.bincount(label_array.astype(np.int64))
    else:
        class_counts = label_array.astype(np.float64)

    class_counts = np.asarray(class_counts, dtype=np.float64)

    # Track which classes have actual positive examples
    has_positives = class_counts > 0

    # For classes with zero counts, use 1.0 as placeholder (will be set to neutral weight)
    safe_counts = np.where(has_positives, class_counts, 1.0)

    if method == 'pos_neg_ratio':
        # Compute pos_weight = num_negatives / num_positives for BCE
        # This properly balances positive vs negative examples
        if total_samples is None:
            # Estimate total samples from max count (assume at least one class has all positives)
            total_samples = max(safe_counts.max() * 2, safe_counts.sum())
        neg_counts = total_samples - safe_counts
        weights = neg_counts / (safe_counts + 1e-8)
        # Use sqrt to dampen extreme weights and prevent gradient explosion
        weights = np.sqrt(weights)
        # Clip to prevent extreme weights (max ~10x, not 100x)
        weights = np.clip(weights, 1.0, 10.0)
    elif method == 'inverse':
        weights = 1.0 / (safe_counts + 1)
    elif method == 'sqrt_inverse':
        weights = 1.0 / np.sqrt(safe_counts + 1)
    elif method == 'effective_num':
        # Effective number of samples
        effective_num = 1.0 - np.power(beta, safe_counts)
        weights = (1.0 - beta) / (effective_num + 1e-8)
        # Normalize only over classes that have positives
        if has_positives.any():
            weights_mean = weights[has_positives].mean()
            weights = weights / (weights_mean + 1e-8)
    else:
        weights = np.ones_like(class_counts, dtype=np.float32)

    # Set weight to 1.0 for classes with no positives (neutral, won't affect loss)
    weights = np.where(has_positives, weights, 1.0)

    return torch.tensor(weights, dtype=torch.float32)
