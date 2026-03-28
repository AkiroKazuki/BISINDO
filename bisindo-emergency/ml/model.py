"""
ST-GCN (Spatial-Temporal Graph Convolutional Network) for BISINDO gesture recognition.

Architecture: 10 ST-GCN blocks with channel config [3->64, 64^3, 64->128, 128^4, 128->256, 256^2]
followed by Global Average Pooling -> Dropout -> Linear(256, 3).

Output is raw logits (no softmax). Use softmax only during inference for confidence scores.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialGraphConv(nn.Module):
    """Spatial graph convolution layer.

    Performs graph convolution: H_out = A_hat @ H_in @ W
    Efficiently implemented using Conv2d with kernel_size=1 as the weight matrix.

    A_hat is received via forward() from the parent STGCN module, which registers
    it as a buffer. This ensures correct save/load behaviour with torch.save.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, C_in, T, V)
            A_hat: (V, V) normalized adjacency matrix

        Returns:
            (batch, C_out, T, V)
        """
        # Graph convolution: multiply feature with adjacency matrix
        # x shape: (B, C, T, V), A_hat shape: (V, V)
        # Equivalent to: einsum('vw, bctw -> bctv', A_hat, x)
        x = torch.einsum('vw,bctw->bctv', A_hat, x)

        # Apply learnable weight via 1x1 conv
        x = self.conv(x)

        return x


class TemporalConv(nn.Module):
    """Temporal convolution layer.

    Conv2d with kernel (9, 1) -- 9 on temporal axis, 1 on node axis.
    Padding (4, 0) preserves temporal length.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            channels, channels,
            kernel_size=(9, 1),
            padding=(4, 0)
        )
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, C, T, V)

        Returns:
            (batch, C, T, V) -- temporal length unchanged
        """
        return self.bn(self.conv(x))


class STGCNBlock(nn.Module):
    """Single ST-GCN block.

    Flow: SpatialGraphConv -> BN -> ReLU -> TemporalConv -> BN -> Residual -> ReLU

    A_hat is passed through forward() rather than stored, so that only the
    top-level STGCN module owns the buffer.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # Spatial graph convolution (A_hat passed via forward)
        self.spatial_conv = SpatialGraphConv(in_channels, out_channels)
        self.bn_spatial = nn.BatchNorm2d(out_channels)

        # Temporal convolution
        self.temporal_conv = TemporalConv(out_channels)

        # Residual connection
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, C_in, T, V)
            A_hat: (V, V) normalized adjacency matrix

        Returns:
            (batch, C_out, T, V)
        """
        # Save for residual
        res = self.residual(x)

        # Spatial -> BN -> ReLU
        x = F.relu(self.bn_spatial(self.spatial_conv(x, A_hat)))

        # Temporal -> BN
        x = self.temporal_conv(x)

        # Add residual + ReLU
        x = F.relu(x + res)

        return x


class STGCN(nn.Module):
    """ST-GCN model for BISINDO emergency gesture classification.

    10 ST-GCN blocks with channel progression:
    3->64, 64->64, 64->64, 64->128, 128->128, 128->128, 128->128, 128->256, 256->256, 256->256

    Output: Global Average Pooling -> Dropout(0.5) -> Linear(256, num_classes) -> logits

    A_hat is registered as a buffer here and passed to each block via forward().
    This is the single source of truth for the adjacency matrix.
    """

    # Channel configuration for each of the 10 blocks
    BLOCK_CONFIGS = [
        (3, 64),
        (64, 64),
        (64, 64),
        (64, 128),
        (128, 128),
        (128, 128),
        (128, 128),
        (128, 256),
        (256, 256),
        (256, 256),
    ]

    def __init__(self, num_classes: int, A: torch.Tensor):
        """
        Args:
            num_classes: Number of output classes (3 for BISINDO emergency).
            A: Normalized adjacency matrix of shape (75, 75) from graph.py.
        """
        super().__init__()

        # Register adjacency matrix as buffer (saved with model, not optimized)
        self.register_buffer('A_hat', A)

        # Build ST-GCN blocks (A_hat is NOT passed to __init__ anymore)
        self.blocks = nn.ModuleList()
        for in_ch, out_ch in self.BLOCK_CONFIGS:
            self.blocks.append(STGCNBlock(in_ch, out_ch))

        # Output layers
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 3, 60, 75) -- (batch, channels, time, vertices)

        Returns:
            (batch, num_classes) -- raw logits (no softmax)
        """
        # Pass through ST-GCN blocks, providing A_hat from our buffer
        for block in self.blocks:
            x = block(x, self.A_hat)

        # Global Average Pooling over temporal and node dimensions
        # x shape: (batch, 256, T, V) -> (batch, 256)
        x = x.mean(dim=-1).mean(dim=-1)

        # Dropout + classification
        x = self.dropout(x)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    from ml.graph import build_adjacency_matrix

    A = build_adjacency_matrix()
    model = STGCN(num_classes=3, A=A)

    # Test forward pass
    dummy_input = torch.randn(2, 3, 60, 75)
    output = model(dummy_input)

    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    assert output.shape == (2, 3), f"Expected (2, 3), got {output.shape}"
    print("\nModel shape test passed!")

    # Verify A_hat is in state_dict (buffer test)
    state = model.state_dict()
    assert 'A_hat' in state, "A_hat should be in state_dict as a registered buffer"
    print("Buffer registration test passed!")
