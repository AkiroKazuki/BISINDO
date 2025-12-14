"""
Temporal Attention Module

Learns to focus on important frames in the sequence
for better sign recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class TemporalAttention(nn.Module):
    """
    Temporal Attention Module.
    
    Computes attention weights over temporal dimension to focus
    on the most important frames for classification.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 1,
        dropout: float = 0.1
    ):
        """
        Initialize attention module.
        
        Args:
            hidden_dim: Feature dimension
            num_heads: Number of attention heads (1 for single-head)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        if num_heads == 1:
            # Simple single-head attention
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1)
            )
        else:
            # Multi-head self-attention
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply temporal attention.
        
        Args:
            x: Input sequence (B, T, hidden_dim)
            mask: Optional attention mask (B, T)
            
        Returns:
            attended: Weighted sum of inputs (B, hidden_dim)
            attention_weights: Attention weights (B, T) for visualization
        """
        if self.num_heads == 1:
            return self._single_head_attention(x, mask)
        else:
            return self._multi_head_attention(x, mask)
    
    def _single_head_attention(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single-head attention mechanism."""
        B, T, C = x.shape
        
        # Compute attention scores
        scores = self.attention(x).squeeze(-1)  # (B, T)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax to get weights
        attention_weights = F.softmax(scores, dim=-1)  # (B, T)
        
        # Apply dropout
        attention_weights = self.dropout(attention_weights)
        
        # Weighted sum
        attended = torch.bmm(
            attention_weights.unsqueeze(1),  # (B, 1, T)
            x  # (B, T, C)
        ).squeeze(1)  # (B, C)
        
        return attended, attention_weights
    
    def _multi_head_attention(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Multi-head self-attention mechanism."""
        # Self-attention: query = key = value = x
        attn_output, attn_weights = self.attention(
            x, x, x,
            key_padding_mask=mask,
            need_weights=True,
            average_attn_weights=True
        )
        
        # Average over time dimension for final representation
        attended = attn_output.mean(dim=1)  # (B, hidden_dim)
        
        # Average attention weights over all positions
        attention_weights = attn_weights.mean(dim=1)  # (B, T)
        
        return attended, attention_weights


class SpatioTemporalAttention(nn.Module):
    """
    Spatio-Temporal Attention Module.
    
    Applies attention over both spatial (vertex) and temporal dimensions.
    Useful for understanding which body parts are important at which times.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_vertices: int = 75,
        dropout: float = 0.1
    ):
        """
        Initialize spatio-temporal attention.
        
        Args:
            hidden_dim: Feature dimension
            num_vertices: Number of body landmarks
            dropout: Dropout probability
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_vertices = num_vertices
        
        # Spatial attention (over vertices)
        self.spatial_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Temporal attention (over frames)
        self.temporal_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply spatio-temporal attention.
        
        Args:
            x: Input features (B, T, V, C)
            
        Returns:
            attended: Aggregated features (B, hidden_dim)
            spatial_weights: Spatial attention weights (B, V)
            temporal_weights: Temporal attention weights (B, T)
        """
        B, T, V, C = x.shape
        
        # Spatial attention: aggregate over vertices
        # Reshape for attention: (B*T, V, C)
        x_spatial = x.view(B * T, V, C)
        spatial_scores = self.spatial_attention(x_spatial).squeeze(-1)  # (B*T, V)
        spatial_weights = F.softmax(spatial_scores, dim=-1)
        spatial_weights = self.dropout(spatial_weights)
        
        # Weighted sum over vertices: (B*T, C)
        x_spatial_attended = torch.bmm(
            spatial_weights.unsqueeze(1), x_spatial
        ).squeeze(1)
        
        # Reshape back: (B, T, C)
        x_temporal = x_spatial_attended.view(B, T, C)
        
        # Temporal attention: aggregate over time
        temporal_scores = self.temporal_attention(x_temporal).squeeze(-1)  # (B, T)
        temporal_weights = F.softmax(temporal_scores, dim=-1)
        temporal_weights = self.dropout(temporal_weights)
        
        # Weighted sum over time: (B, C)
        attended = torch.bmm(
            temporal_weights.unsqueeze(1), x_temporal
        ).squeeze(1)
        
        # Average spatial weights over time for visualization
        spatial_weights_avg = spatial_weights.view(B, T, V).mean(dim=1)  # (B, V)
        
        return attended, spatial_weights_avg, temporal_weights


class PositionalEncoding(nn.Module):
    """
    Positional Encoding for transformer-style attention.
    
    Adds position information so the model knows the temporal order.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        max_len: int = 500,
        dropout: float = 0.1
    ):
        """
        Initialize positional encoding.
        
        Args:
            hidden_dim: Feature dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float() * 
            (-torch.log(torch.tensor(10000.0)) / hidden_dim)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, hidden_dim)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor (B, T, hidden_dim)
            
        Returns:
            Output with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


if __name__ == "__main__":
    # Test attention modules
    B, T, C = 2, 30, 128
    V = 75
    
    # Test temporal attention
    temporal_attn = TemporalAttention(hidden_dim=C)
    x = torch.randn(B, T, C)
    attended, weights = temporal_attn(x)
    
    print(f"Temporal Attention:")
    print(f"  Input shape: {x.shape}")
    print(f"  Attended shape: {attended.shape}")
    print(f"  Weights shape: {weights.shape}")
    print(f"  Weights sum: {weights.sum(dim=-1)}")  # Should be 1
    
    # Test spatio-temporal attention
    st_attn = SpatioTemporalAttention(hidden_dim=C, num_vertices=V)
    x_st = torch.randn(B, T, V, C)
    attended_st, spatial_w, temporal_w = st_attn(x_st)
    
    print(f"\nSpatio-Temporal Attention:")
    print(f"  Input shape: {x_st.shape}")
    print(f"  Attended shape: {attended_st.shape}")
    print(f"  Spatial weights shape: {spatial_w.shape}")
    print(f"  Temporal weights shape: {temporal_w.shape}")
    
    # Test positional encoding
    pe = PositionalEncoding(hidden_dim=C)
    x_pe = pe(x)
    print(f"\nPositional Encoding:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {x_pe.shape}")
