"""
MS-TCN++ (Multi-Stage Temporal Convolutional Network) for action segmentation.

Based on:
"MS-TCN++: Multi-Stage Temporal Convolutional Network for Action Segmentation"
Li et al., TPAMI 2020

This is the state-of-the-art architecture for temporal action segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class DilatedResidualLayer(nn.Module):
    """
    Single dilated residual layer with dropout.
    Uses 1D convolutions for temporal modeling.
    """

    def __init__(
        self,
        dilation: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation // 2

        self.conv_dilated = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout(dropout)

        # Skip connection
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, seq_len)
        Returns:
            (batch, channels, seq_len)
        """
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return out + self.skip(x)


class SingleStageTCN(nn.Module):
    """
    Single stage of the TCN with dilated convolutions.
    Uses exponentially increasing dilation rates.
    """

    def __init__(
        self,
        in_channels: int,
        num_layers: int,
        num_f_maps: int,
        num_classes: int,
        kernel_size: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()

        self.conv_in = nn.Conv1d(in_channels, num_f_maps, 1)

        self.layers = nn.ModuleList([
            DilatedResidualLayer(
                dilation=2 ** i,
                in_channels=num_f_maps,
                out_channels=num_f_maps,
                kernel_size=kernel_size,
                dropout=dropout
            )
            for i in range(num_layers)
        ])

        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, seq_len)
        Returns:
            (batch, num_classes, seq_len)
        """
        out = self.conv_in(x)
        for layer in self.layers:
            out = layer(out)
        return self.conv_out(out)


class DualDilatedLayer(nn.Module):
    """
    Dual dilated layer from MS-TCN++.
    Combines two dilated convolutions with different dilation rates.
    """

    def __init__(
        self,
        dilation: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()

        padding1 = (kernel_size - 1) * dilation // 2
        padding2 = (kernel_size - 1) * (2 * dilation) // 2

        # First dilated conv
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding1, dilation=dilation
        )

        # Second dilated conv with doubled dilation
        self.conv2 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding2, dilation=2 * dilation
        )

        self.conv_fusion = nn.Conv1d(2 * out_channels, out_channels, 1)
        self.conv_out = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout(dropout)

        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = F.relu(self.conv1(x))
        out2 = F.relu(self.conv2(x))

        out = torch.cat([out1, out2], dim=1)
        out = self.conv_fusion(out)
        out = F.relu(out)
        out = self.conv_out(out)
        out = self.dropout(out)

        return out + self.skip(x)


class RefinementStage(nn.Module):
    """
    Refinement stage for MS-TCN++.
    Takes predictions from previous stage and refines them.
    """

    def __init__(
        self,
        num_layers: int,
        num_f_maps: int,
        num_classes: int,
        kernel_size: int = 3,
        dropout: float = 0.3,
        use_dual_dilated: bool = True
    ):
        super().__init__()

        self.conv_in = nn.Conv1d(num_classes, num_f_maps, 1)

        if use_dual_dilated:
            self.layers = nn.ModuleList([
                DualDilatedLayer(
                    dilation=2 ** i,
                    in_channels=num_f_maps,
                    out_channels=num_f_maps,
                    kernel_size=kernel_size,
                    dropout=dropout
                )
                for i in range(num_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                DilatedResidualLayer(
                    dilation=2 ** i,
                    in_channels=num_f_maps,
                    out_channels=num_f_maps,
                    kernel_size=kernel_size,
                    dropout=dropout
                )
                for i in range(num_layers)
            ])

        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_classes, seq_len) - previous stage predictions
        Returns:
            (batch, num_classes, seq_len) - refined predictions
        """
        out = self.conv_in(x)
        for layer in self.layers:
            out = layer(out)
        return self.conv_out(out)


class MSTCN(nn.Module):
    """
    MS-TCN++ (Multi-Stage Temporal Convolutional Network).

    A multi-stage architecture where:
    - First stage: Single-stage TCN on input features
    - Subsequent stages: Refinement stages on previous predictions

    The architecture uses dilated convolutions with exponentially
    increasing dilation rates to capture long-range temporal dependencies.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_stages: int = 4,
        num_layers: int = 10,
        num_f_maps: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.5
    ):
        """
        Args:
            input_dim: Number of input features
            num_classes: Number of behavior classes
            num_stages: Number of refinement stages
            num_layers: Number of layers per stage
            num_f_maps: Number of feature maps (channels)
            kernel_size: Convolution kernel size
            dropout: Dropout probability
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_stages = num_stages

        # First stage: process input features
        self.stage1 = SingleStageTCN(
            in_channels=input_dim,
            num_layers=num_layers,
            num_f_maps=num_f_maps,
            num_classes=num_classes,
            kernel_size=kernel_size,
            dropout=dropout
        )

        # Refinement stages
        self.stages = nn.ModuleList([
            RefinementStage(
                num_layers=num_layers,
                num_f_maps=num_f_maps,
                num_classes=num_classes,
                kernel_size=kernel_size,
                dropout=dropout,
                use_dual_dilated=True
            )
            for _ in range(num_stages - 1)
        ])

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through all stages.

        Args:
            x: Input features (batch, seq_len, input_dim)
            mask: Optional validity mask (batch, seq_len)

        Returns:
            final_output: Final predictions (batch, seq_len, num_classes)
            stage_outputs: List of outputs from each stage
        """
        # Convert from (batch, seq_len, features) to (batch, features, seq_len)
        x = x.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch, 1, seq_len)

        stage_outputs = []

        # First stage
        out = self.stage1(x)
        if mask is not None:
            out = out * mask
        stage_outputs.append(out)

        # Refinement stages
        for stage in self.stages:
            out = stage(F.softmax(out, dim=1))
            if mask is not None:
                out = out * mask
            stage_outputs.append(out)

        # Convert back to (batch, seq_len, num_classes)
        final_output = out.transpose(1, 2)
        stage_outputs = [s.transpose(1, 2) for s in stage_outputs]

        return final_output, stage_outputs


class MSTCNWithBoundary(nn.Module):
    """
    MS-TCN++ with additional boundary detection head.
    Predicts both action classes and segment boundaries.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_stages: int = 4,
        num_layers: int = 10,
        num_f_maps: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.5
    ):
        super().__init__()

        # Main TCN for action classification
        self.mstcn = MSTCN(
            input_dim=input_dim,
            num_classes=num_classes,
            num_stages=num_stages,
            num_layers=num_layers,
            num_f_maps=num_f_maps,
            kernel_size=kernel_size,
            dropout=dropout
        )

        # Boundary detection head
        self.boundary_head = nn.Sequential(
            nn.Conv1d(num_classes, num_f_maps, 1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(num_f_maps, 2, 1)  # Start and end boundary
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        Forward pass.

        Returns:
            final_output: Action predictions (batch, seq_len, num_classes)
            stage_outputs: List of stage outputs
            boundary_output: Boundary predictions (batch, seq_len, 2)
        """
        final_output, stage_outputs = self.mstcn(x, mask)

        # Boundary detection from final output
        final_transposed = final_output.transpose(1, 2)  # (batch, num_classes, seq_len)
        boundary_output = self.boundary_head(final_transposed)
        boundary_output = boundary_output.transpose(1, 2)  # (batch, seq_len, 2)

        return final_output, stage_outputs, boundary_output


class MSTCNLoss(nn.Module):
    """
    Combined loss for MS-TCN++ training.

    Includes:
    - Classification loss (cross-entropy or focal loss)
    - Temporal smoothing loss (T-MSE)
    - Optional boundary loss
    """

    def __init__(
        self,
        num_classes: int,
        smoothing_weight: float = 0.15,
        boundary_weight: float = 0.0,
        class_weights: Optional[torch.Tensor] = None,
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0
    ):
        super().__init__()

        self.num_classes = num_classes
        self.smoothing_weight = smoothing_weight
        self.boundary_weight = boundary_weight
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma

        if class_weights is not None:
            class_weights = class_weights.float()
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

    def forward(
        self,
        predictions: torch.Tensor,
        stage_outputs: List[torch.Tensor],
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        boundary_pred: Optional[torch.Tensor] = None,
        boundary_target: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss.

        Args:
            predictions: Final predictions (batch, seq_len, num_classes)
            stage_outputs: Outputs from each stage
            targets: Ground truth labels (batch, seq_len) or (batch, seq_len, num_classes)
            mask: Validity mask (batch, seq_len)
            boundary_pred: Boundary predictions (batch, seq_len, 2)
            boundary_target: Boundary ground truth (batch, seq_len, 2)

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual loss components
        """
        loss_dict = {}

        # Classification loss for each stage
        cls_loss = 0.0
        for i, stage_out in enumerate(stage_outputs):
            stage_loss = self._classification_loss(stage_out, targets, mask)
            cls_loss += stage_loss
            loss_dict[f'cls_loss_stage_{i}'] = stage_loss.item()

        cls_loss = cls_loss / len(stage_outputs)
        loss_dict['cls_loss'] = cls_loss.item()

        # Smoothing loss
        smooth_loss = 0.0
        for stage_out in stage_outputs:
            smooth_loss += self._smoothing_loss(stage_out, mask)
        smooth_loss = smooth_loss / len(stage_outputs)
        loss_dict['smooth_loss'] = smooth_loss.item()

        # Boundary loss
        boundary_loss = torch.tensor(0.0, device=predictions.device)
        if self.boundary_weight > 0 and boundary_pred is not None and boundary_target is not None:
            boundary_loss = F.binary_cross_entropy_with_logits(
                boundary_pred, boundary_target,
                reduction='none'
            )
            if mask is not None:
                boundary_loss = (boundary_loss * mask.unsqueeze(-1)).sum() / (mask.sum() + 1e-8)
            else:
                boundary_loss = boundary_loss.mean()
            loss_dict['boundary_loss'] = boundary_loss.item()

        # Total loss
        total_loss = cls_loss + self.smoothing_weight * smooth_loss + self.boundary_weight * boundary_loss
        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict

    def _classification_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute classification loss."""
        batch_size, seq_len, num_classes = predictions.shape

        # Handle multi-label targets
        if targets.dim() == 3:  # (batch, seq_len, num_classes)
            # Binary cross entropy for multi-label
            pos_weight = self.class_weights if self.class_weights is not None else None
            loss = F.binary_cross_entropy_with_logits(
                predictions, targets, reduction='none', pos_weight=pos_weight
            )
            loss = loss.mean(dim=-1)  # Average over classes
        else:  # (batch, seq_len)
            # Cross entropy for single-label
            predictions_flat = predictions.reshape(-1, num_classes)
            targets_flat = targets.reshape(-1)

            if self.use_focal_loss:
                loss = self._focal_loss(predictions_flat, targets_flat)
                loss = loss.reshape(batch_size, seq_len)
            else:
                loss = F.cross_entropy(
                    predictions_flat, targets_flat,
                    weight=self.class_weights,
                    reduction='none'
                )
                loss = loss.reshape(batch_size, seq_len)

        if mask is not None:
            loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        else:
            loss = loss.mean()

        return loss

    def _focal_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute focal loss for handling class imbalance."""
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.focal_gamma) * ce_loss

        if self.class_weights is not None:
            focal_loss = focal_loss * self.class_weights[targets]

        return focal_loss

    def _smoothing_loss(
        self,
        predictions: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Temporal smoothing loss (T-MSE).
        Encourages smooth predictions over time.
        """
        # Compute difference between consecutive frames
        log_probs = F.log_softmax(predictions, dim=-1)
        diff = log_probs[:, 1:, :] - log_probs[:, :-1, :]
        smooth_loss = torch.clamp(diff ** 2, min=0, max=16).mean(dim=-1)

        if mask is not None:
            mask_diff = mask[:, 1:] * mask[:, :-1]
            smooth_loss = (smooth_loss * mask_diff).sum() / (mask_diff.sum() + 1e-8)
        else:
            smooth_loss = smooth_loss.mean()

        return smooth_loss
