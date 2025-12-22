"""
PyTorch Lightning module for behavior recognition training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import numpy as np

from .mstcn import MSTCN, MSTCNLoss
from .tcn_transformer import TCNTransformer, TCNTransformerLoss


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
        self.save_hyperparameters()

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
                class_weights=class_weights
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

        # Compute metrics
        metrics = self._compute_metrics(predictions, labels, mask)
        for key, value in metrics.items():
            self.log(f'val/{key}', value, prog_bar=(key in ['f1', 'accuracy']))

        return {
            'loss': loss,
            'predictions': predictions.detach(),
            'labels': labels.detach()
        }

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
    is_counts: bool = False
) -> torch.Tensor:
    """
    Compute class weights for imbalanced data.

    Args:
        labels: Array of labels or precomputed class counts
        method: 'inverse', 'effective_num', or 'sqrt_inverse'
        beta: Beta parameter for effective number weighting
        is_counts: Treat `labels` as already-aggregated class counts when True

    Returns:
        Class weight tensor
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
    class_counts[class_counts < 1e-6] = 1e-6

    if method == 'inverse':
        weights = 1.0 / (class_counts + 1)
    elif method == 'sqrt_inverse':
        weights = 1.0 / np.sqrt(class_counts + 1)
    elif method == 'effective_num':
        # Effective number of samples
        effective_num = 1.0 - np.power(beta, class_counts)
        weights = (1.0 - beta) / (effective_num + 1e-8)
    else:
        weights = np.ones_like(class_counts, dtype=np.float32)

    # Normalize
    weights = weights / (weights.mean() + 1e-8)

    return torch.tensor(weights, dtype=torch.float32)
