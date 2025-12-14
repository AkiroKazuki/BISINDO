"""
Skeleton Graph Encoder Module

Encodes spatial relationships between body landmarks using
graph convolutions over the skeleton structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class GraphConv(nn.Module):
    """
    Graph Convolution Layer.
    
    Applies learnable transformation to node features
    based on graph adjacency structure.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adjacency_matrix: torch.Tensor,
        bias: bool = True
    ):
        """
        Initialize graph convolution layer.
        
        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension
            adjacency_matrix: Normalized adjacency matrix (V, V)
            bias: Whether to use bias
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Register adjacency matrix as buffer (not a parameter)
        self.register_buffer('A', adjacency_matrix)
        
        # Learnable weight matrix
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features (B, T, V, C)
            
        Returns:
            Output features (B, T, V, out_channels)
        """
        B, T, V, C = x.shape
        
        # Reshape for batch matrix multiplication
        # (B*T, V, C)
        x = x.view(B * T, V, C)
        
        # Graph convolution: A @ X @ W
        # Step 1: X @ W -> (B*T, V, out_channels)
        x = torch.matmul(x, self.weight)
        
        # Step 2: A @ (X @ W) -> (B*T, V, out_channels)
        x = torch.matmul(self.A, x)
        
        if self.bias is not None:
            x = x + self.bias
        
        # Reshape back
        x = x.view(B, T, V, self.out_channels)
        
        return x


class SkeletonGraphEncoder(nn.Module):
    """
    Skeleton Graph Encoder.
    
    Encodes spatial relationships between body landmarks using
    multiple graph convolution layers.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        adjacency_matrix: torch.Tensor,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        """
        Initialize skeleton graph encoder.
        
        Args:
            input_dim: Input feature dimension per vertex (e.g., 3 for x,y,vis)
            hidden_dim: Hidden dimension
            output_dim: Output feature dimension
            adjacency_matrix: Normalized adjacency matrix (V, V)
            num_layers: Number of graph conv layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Graph convolution layers
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        # First layer
        self.conv_layers.append(
            GraphConv(input_dim, hidden_dim, adjacency_matrix)
        )
        self.bn_layers.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.conv_layers.append(
                GraphConv(hidden_dim, hidden_dim, adjacency_matrix)
            )
            self.bn_layers.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        if num_layers > 1:
            self.conv_layers.append(
                GraphConv(hidden_dim, output_dim, adjacency_matrix)
            )
            self.bn_layers.append(nn.BatchNorm1d(output_dim))
        
        self.dropout = nn.Dropout(dropout)
        
        # Global pooling to aggregate vertex features
        self.pool_type = 'mean'  # Can be 'mean', 'max', or 'attention'
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input keypoints (B, T, V, C)
               B=batch, T=time, V=vertices, C=channels
               
        Returns:
            Encoded features (B, T, output_dim)
        """
        B, T, V, C = x.shape
        
        # Apply graph convolution layers
        for i, (conv, bn) in enumerate(zip(self.conv_layers, self.bn_layers)):
            x = conv(x)
            
            # Batch norm: reshape to (B*T*V, C) for batch norm
            x_shape = x.shape
            x = x.view(-1, x_shape[-1])
            x = bn(x)
            x = x.view(*x_shape)
            
            # Activation and dropout (except last layer)
            if i < len(self.conv_layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        
        # Global pooling over vertices
        if self.pool_type == 'mean':
            x = x.mean(dim=2)  # (B, T, output_dim)
        elif self.pool_type == 'max':
            x = x.max(dim=2)[0]  # (B, T, output_dim)
        
        return x
    
    def get_vertex_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get per-vertex features without pooling.
        
        Args:
            x: Input keypoints (B, T, V, C)
            
        Returns:
            Per-vertex features (B, T, V, output_dim)
        """
        B, T, V, C = x.shape
        
        for i, (conv, bn) in enumerate(zip(self.conv_layers, self.bn_layers)):
            x = conv(x)
            
            x_shape = x.shape
            x = x.view(-1, x_shape[-1])
            x = bn(x)
            x = x.view(*x_shape)
            
            if i < len(self.conv_layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        
        return x


def build_adjacency_matrix(
    num_pose: int = 33,
    num_hand: int = 21,
    normalize: bool = True
) -> torch.Tensor:
    """
    Build skeleton adjacency matrix.
    
    Args:
        num_pose: Number of pose landmarks
        num_hand: Number of hand landmarks (per hand)
        normalize: Whether to normalize the matrix
        
    Returns:
        Adjacency matrix as torch tensor
    """
    total = num_pose + 2 * num_hand
    adj = np.zeros((total, total))
    
    # Self-loops
    np.fill_diagonal(adj, 1)
    
    # Pose connections
    pose_connections = [
        (0, 1), (1, 2), (2, 3), (3, 7),
        (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10),
        (11, 12), (11, 23), (12, 24), (23, 24),
        (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
        (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
        (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
    ]
    
    for start, end in pose_connections:
        if start < num_pose and end < num_pose:
            adj[start, end] = 1
            adj[end, start] = 1
    
    # Hand connections
    hand_connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17),
    ]
    
    # Left hand
    offset = num_pose
    for start, end in hand_connections:
        adj[offset + start, offset + end] = 1
        adj[offset + end, offset + start] = 1
    
    # Right hand
    offset = num_pose + num_hand
    for start, end in hand_connections:
        adj[offset + start, offset + end] = 1
        adj[offset + end, offset + start] = 1
    
    # Connect hands to wrists
    adj[15, num_pose] = 1  # Left wrist to left hand
    adj[num_pose, 15] = 1
    adj[16, num_pose + num_hand] = 1  # Right wrist to right hand
    adj[num_pose + num_hand, 16] = 1
    
    if normalize:
        # Symmetric normalization: D^(-1/2) @ A @ D^(-1/2)
        degree = adj.sum(axis=1)
        degree = np.where(degree > 0, degree, 1)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
        adj = D_inv_sqrt @ adj @ D_inv_sqrt
    
    return torch.FloatTensor(adj)


if __name__ == "__main__":
    # Test graph encoder
    B, T, V, C = 2, 30, 75, 5  # 75 landmarks, 5 features (x, y, vis, dx, dy)
    
    # Build adjacency matrix
    adj = build_adjacency_matrix(num_pose=33, num_hand=21)
    print(f"Adjacency matrix shape: {adj.shape}")
    
    # Create encoder
    encoder = SkeletonGraphEncoder(
        input_dim=C,
        hidden_dim=64,
        output_dim=128,
        adjacency_matrix=adj,
        num_layers=2
    )
    
    # Test forward pass
    x = torch.randn(B, T, V, C)
    out = encoder(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    
    # Test vertex features
    vertex_out = encoder.get_vertex_features(torch.randn(B, T, V, C))
    print(f"Vertex features shape: {vertex_out.shape}")
    
    # Count parameters
    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"Number of parameters: {num_params:,}")
