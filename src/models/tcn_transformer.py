"""
TCN + Transformer hybrid architecture for action segmentation.

Combines:
- TCN for local temporal pattern extraction
- Transformer for long-range dependencies
- Pairwise interaction module for multi-agent modeling
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class TemporalConvBlock(nn.Module):
    """
    Temporal convolution block with residual connection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            stride=1, padding=padding, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(dropout)

        # Skip connection
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = F.relu(out + residual)
        return self.dropout(out)


class TCNEncoder(nn.Module):
    """
    Multi-layer TCN encoder for local temporal feature extraction.
    """

    def __init__(
        self,
        input_dim: int,
        channels: List[int] = [64, 128, 256],
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_proj = nn.Conv1d(input_dim, channels[0], 1)

        layers = []
        in_channels = channels[0]

        for i, out_channels in enumerate(channels):
            # Exponentially increasing dilation
            dilation = 2 ** i
            layers.append(
                TemporalConvBlock(
                    in_channels, out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )
            in_channels = out_channels

        self.layers = nn.ModuleList(layers)
        self.output_dim = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            (batch, seq_len, output_dim)
        """
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        x = self.input_proj(x)

        for layer in self.layers:
            x = layer(x)

        return x.transpose(1, 2)  # (batch, seq_len, output_dim)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """
    Transformer encoder for long-range temporal dependencies.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 5000
    ):
        super().__init__()

        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_dim = d_model

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len) attention mask
        Returns:
            (batch, seq_len, d_model)
        """
        x = self.pos_encoder(x)

        # Convert mask to attention mask format
        if mask is not None:
            # TransformerEncoder expects True for masked positions
            attn_mask = (mask == 0)
        else:
            attn_mask = None

        return self.transformer(x, src_key_padding_mask=attn_mask)


class PairwiseInteractionModule(nn.Module):
    """
    Module for computing pairwise interactions between agent and target.
    Uses attention mechanism to model relationships.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.agent_proj = nn.Linear(input_dim, hidden_dim)
        self.target_proj = nn.Linear(input_dim, hidden_dim)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.output_dim = hidden_dim

    def forward(
        self,
        agent_features: torch.Tensor,
        target_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute pairwise interaction features.

        Args:
            agent_features: (batch, seq_len, input_dim)
            target_features: (batch, seq_len, input_dim)
            mask: (batch, seq_len)

        Returns:
            (batch, seq_len, hidden_dim)
        """
        agent_h = self.agent_proj(agent_features)
        target_h = self.target_proj(target_features)

        # Cross attention: agent attends to target
        attn_mask = None
        if mask is not None:
            attn_mask = (mask == 0)

        attended, _ = self.cross_attention(
            agent_h, target_h, target_h,
            key_padding_mask=attn_mask
        )

        # Fuse attended features with agent features
        combined = torch.cat([agent_h, attended], dim=-1)
        interaction = self.fusion(combined)

        return interaction


class TCNTransformer(nn.Module):
    """
    TCN + Transformer hybrid model for behavior recognition.

    Architecture:
    1. Per-mouse TCN encoder for local patterns
    2. Pairwise interaction module for agent-target modeling
    3. Transformer encoder for long-range dependencies
    4. Classification head
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        tcn_channels: List[int] = [64, 128, 256],
        tcn_kernel_size: int = 3,
        transformer_dim: int = 256,
        transformer_heads: int = 8,
        transformer_layers: int = 2,
        dropout: float = 0.3,
        use_pairwise: bool = True
    ):
        """
        Args:
            input_dim: Number of input features per mouse
            num_classes: Number of behavior classes
            tcn_channels: Channel sizes for TCN layers
            tcn_kernel_size: Kernel size for TCN
            transformer_dim: Transformer hidden dimension
            transformer_heads: Number of attention heads
            transformer_layers: Number of transformer layers
            dropout: Dropout probability
            use_pairwise: Whether to use pairwise interaction module
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.use_pairwise = use_pairwise

        # TCN encoder for each mouse
        # Assumes input is concatenated [agent_features, target_features]
        single_mouse_dim = input_dim // 2 if use_pairwise else input_dim

        self.agent_encoder = TCNEncoder(
            input_dim=single_mouse_dim,
            channels=tcn_channels,
            kernel_size=tcn_kernel_size,
            dropout=dropout
        )

        if use_pairwise:
            self.target_encoder = TCNEncoder(
                input_dim=single_mouse_dim,
                channels=tcn_channels,
                kernel_size=tcn_kernel_size,
                dropout=dropout
            )

            self.interaction_module = PairwiseInteractionModule(
                input_dim=tcn_channels[-1],
                hidden_dim=transformer_dim,
                dropout=dropout
            )

            # Project for transformer
            self.proj = nn.Linear(tcn_channels[-1] + transformer_dim, transformer_dim)
        else:
            self.proj = nn.Linear(tcn_channels[-1], transformer_dim)

        # Transformer for long-range dependencies
        self.transformer = TransformerEncoder(
            d_model=transformer_dim,
            nhead=transformer_heads,
            num_layers=transformer_layers,
            dim_feedforward=transformer_dim * 4,
            dropout=dropout
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(transformer_dim, num_classes)
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input features (batch, seq_len, input_dim)
            mask: Validity mask (batch, seq_len)

        Returns:
            predictions: (batch, seq_len, num_classes)
            intermediate: List of intermediate representations
        """
        intermediate = []

        if self.use_pairwise:
            # Split input into agent and target features
            single_dim = self.input_dim // 2
            agent_input = x[..., :single_dim]
            target_input = x[..., single_dim:]

            # Encode each mouse
            agent_encoded = self.agent_encoder(agent_input)
            target_encoded = self.target_encoder(target_input)
            intermediate.extend([agent_encoded, target_encoded])

            # Compute interaction features
            interaction = self.interaction_module(
                agent_encoded, target_encoded, mask
            )
            intermediate.append(interaction)

            # Combine encoded features
            combined = torch.cat([agent_encoded, interaction], dim=-1)
            features = self.proj(combined)
        else:
            # Single encoder path
            features = self.agent_encoder(x)
            features = self.proj(features)
            intermediate.append(features)

        # Transformer
        features = self.transformer(features, mask)
        intermediate.append(features)

        # Classification
        predictions = self.classifier(features)

        return predictions, intermediate


class TCNTransformerLoss(nn.Module):
    """
    Loss function for TCN-Transformer model.
    Supports both multi-label and single-label classification.
    """

    def __init__(
        self,
        num_classes: int,
        smoothing_weight: float = 0.1,
        class_weights: Optional[torch.Tensor] = None,
        multi_label: bool = True
    ):
        super().__init__()

        self.num_classes = num_classes
        self.smoothing_weight = smoothing_weight
        self.multi_label = multi_label

        if class_weights is not None:
            class_weights = class_weights.float()
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute loss.

        Args:
            predictions: (batch, seq_len, num_classes)
            targets: (batch, seq_len, num_classes) or (batch, seq_len)
            mask: (batch, seq_len)

        Returns:
            total_loss: Combined loss
            loss_dict: Component losses
        """
        loss_dict = {}

        # Classification loss
        if self.multi_label or targets.dim() == 3:
            # Binary cross entropy for multi-label
            pos_weight = self.class_weights if self.class_weights is not None else None
            # Clamp predictions to prevent numerical instability
            predictions_clamped = torch.clamp(predictions, -20.0, 20.0)
            cls_loss = F.binary_cross_entropy_with_logits(
                predictions_clamped, targets.float(),
                reduction='none',
                pos_weight=pos_weight
            )
            # Clamp loss values to prevent gradient explosion
            cls_loss = torch.clamp(cls_loss, 0.0, 100.0)
            cls_loss = cls_loss.mean(dim=-1)
        else:
            # Cross entropy for single-label
            batch, seq_len, num_cls = predictions.shape
            cls_loss = F.cross_entropy(
                predictions.reshape(-1, num_cls),
                targets.reshape(-1),
                weight=self.class_weights,
                reduction='none'
            ).reshape(batch, seq_len)

        if mask is not None:
            cls_loss = (cls_loss * mask).sum() / (mask.sum() + 1e-8)
        else:
            cls_loss = cls_loss.mean()

        loss_dict['cls_loss'] = cls_loss.item()

        # Smoothing loss
        probs = torch.sigmoid(predictions) if self.multi_label else F.softmax(predictions, dim=-1)
        diff = probs[:, 1:] - probs[:, :-1]
        smooth_loss = (diff ** 2).mean()
        loss_dict['smooth_loss'] = smooth_loss.item()

        total_loss = cls_loss + self.smoothing_weight * smooth_loss
        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict
