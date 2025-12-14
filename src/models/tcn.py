"""
Temporal Convolutional Network (TCN) Module

Features:
- Causal convolutions (no future leakage)
- Dilated convolutions for large receptive field
- Residual connections for gradient flow
- Faster than LSTM with comparable accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class CausalConv1d(nn.Module):
    """
    Causal 1D Convolution.
    
    Ensures that the output at time t only depends on inputs at time <= t.
    Uses left padding to achieve causality.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        bias: bool = True
    ):
        """
        Initialize causal convolution.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Kernel size
            dilation: Dilation factor
            bias: Whether to use bias
        """
        super().__init__()
        
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        # Calculate padding needed for causality
        self.padding = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=self.padding,
            bias=bias
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with causal convolution.
        
        Args:
            x: Input tensor (B, C, T)
            
        Returns:
            Output tensor (B, out_channels, T)
        """
        out = self.conv(x)
        
        # Remove future information (right side of output)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        
        return out


class TemporalBlock(nn.Module):
    """
    Single Temporal Block with residual connection.
    
    Structure:
    input -> Conv -> BN -> ReLU -> Dropout -> Conv -> BN -> + -> ReLU
              |                                            |
              +------------ Residual Connection -----------+
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2
    ):
        """
        Initialize temporal block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Kernel size
            dilation: Dilation factor
            dropout: Dropout probability
        """
        super().__init__()
        
        # First convolution
        self.conv1 = CausalConv1d(
            in_channels, out_channels, kernel_size, dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        # Second convolution
        self.conv2 = CausalConv1d(
            out_channels, out_channels, kernel_size, dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection (1x1 conv if channels don't match)
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.residual = nn.Identity()
        
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, T)
            
        Returns:
            Output tensor (B, out_channels, T)
        """
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        
        # Residual connection
        res = self.residual(x)
        
        # Add and activate
        out = self.relu(out + res)
        
        return out


class TCN(nn.Module):
    """
    Temporal Convolutional Network.
    
    Stack of temporal blocks with exponentially increasing dilation
    for capturing long-range temporal dependencies.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.2
    ):
        """
        Initialize TCN.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            output_dim: Output feature dimension
            num_layers: Number of temporal blocks
            kernel_size: Kernel size for convolutions
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Build layers
        layers = []
        
        for i in range(num_layers):
            # Exponentially increasing dilation
            dilation = 2 ** i
            
            # Input/output channels
            in_ch = input_dim if i == 0 else hidden_dim
            out_ch = output_dim if i == num_layers - 1 else hidden_dim
            
            layers.append(TemporalBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout
            ))
        
        self.network = nn.Sequential(*layers)
        
        # Calculate receptive field
        self.receptive_field = self._compute_receptive_field(
            num_layers, kernel_size
        )
    
    def _compute_receptive_field(
        self, 
        num_layers: int, 
        kernel_size: int
    ) -> int:
        """Compute the receptive field size."""
        rf = 1
        for i in range(num_layers):
            dilation = 2 ** i
            rf += 2 * (kernel_size - 1) * dilation
        return rf
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input sequence (B, T, C)
            
        Returns:
            Temporal features (B, T, output_dim)
        """
        # Transpose for Conv1d: (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)
        
        # Apply TCN
        out = self.network(x)
        
        # Transpose back: (B, C, T) -> (B, T, C)
        out = out.transpose(1, 2)
        
        return out
    
    def get_receptive_field(self) -> int:
        """Return the receptive field size in frames."""
        return self.receptive_field


class MultiScaleTCN(nn.Module):
    """
    Multi-scale TCN for capturing patterns at different temporal scales.
    
    Uses multiple TCN branches with different kernel sizes,
    then combines their outputs.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 4,
        kernel_sizes: List[int] = [3, 5, 7],
        dropout: float = 0.2
    ):
        """
        Initialize multi-scale TCN.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension per branch
            output_dim: Output feature dimension
            num_layers: Number of temporal blocks per branch
            kernel_sizes: List of kernel sizes for each branch
            dropout: Dropout probability
        """
        super().__init__()
        
        self.branches = nn.ModuleList()
        
        # Create a TCN branch for each kernel size
        for kernel_size in kernel_sizes:
            branch = TCN(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                num_layers=num_layers,
                kernel_size=kernel_size,
                dropout=dropout
            )
            self.branches.append(branch)
        
        # Fusion layer
        self.fusion = nn.Linear(hidden_dim * len(kernel_sizes), output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input sequence (B, T, C)
            
        Returns:
            Fused temporal features (B, T, output_dim)
        """
        # Run each branch
        branch_outputs = [branch(x) for branch in self.branches]
        
        # Concatenate along feature dimension
        combined = torch.cat(branch_outputs, dim=-1)
        
        # Fuse
        out = self.fusion(combined)
        
        return out


if __name__ == "__main__":
    # Test TCN
    B, T, C = 2, 30, 128
    
    # Single-scale TCN
    tcn = TCN(
        input_dim=C,
        hidden_dim=128,
        output_dim=128,
        num_layers=4,
        kernel_size=3
    )
    
    x = torch.randn(B, T, C)
    out = tcn(x)
    
    print(f"TCN Input shape: {x.shape}")
    print(f"TCN Output shape: {out.shape}")
    print(f"Receptive field: {tcn.get_receptive_field()} frames")
    
    # Count parameters
    num_params = sum(p.numel() for p in tcn.parameters())
    print(f"TCN parameters: {num_params:,}")
    
    # Multi-scale TCN
    ms_tcn = MultiScaleTCN(
        input_dim=C,
        hidden_dim=64,
        output_dim=128,
        num_layers=3,
        kernel_sizes=[3, 5, 7]
    )
    
    out_ms = ms_tcn(x)
    print(f"\nMulti-scale TCN Output shape: {out_ms.shape}")
    
    num_params_ms = sum(p.numel() for p in ms_tcn.parameters())
    print(f"Multi-scale TCN parameters: {num_params_ms:,}")
