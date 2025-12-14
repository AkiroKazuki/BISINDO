"""
Full Sign Language Classifier Model

Architecture:
1. Skeleton Graph Encoder: Spatial feature extraction
2. TCN Backbone: Temporal feature extraction
3. Temporal Attention: Focus on key frames
4. Classification Head: Final prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import yaml

from .skeleton_graph import SkeletonGraphEncoder, build_adjacency_matrix
from .tcn import TCN
from .attention import TemporalAttention


class SignClassifier(nn.Module):
    """
    Full Sign Language Classifier.
    
    Combines skeleton graph encoding, temporal convolution,
    and attention for sign language recognition.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize full classifier.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        
        self.config = config
        
        # Extract configuration
        model_config = config.get('model', {})
        kp_config = config.get('keypoints', {})
        
        # Dimensions
        self.num_classes = model_config.get('num_classes', 10)
        self.hidden_dim = model_config.get('hidden_dim', 128)
        self.dropout = model_config.get('dropout', 0.3)
        self.use_attention = model_config.get('use_attention', True)
        
        # Keypoint dimensions
        num_pose = 33 if kp_config.get('use_pose', True) else 0
        num_hand = 21 if kp_config.get('use_hands', True) else 0
        self.num_vertices = num_pose + 2 * num_hand
        
        # Input dimension: x, y, visibility + velocity (dx, dy)
        self.input_dim = 5  # x, y, vis, dx, dy
        
        # Build adjacency matrix
        adj_matrix = build_adjacency_matrix(num_pose, num_hand)
        
        # 1. Skeleton Graph Encoder
        self.graph_encoder = SkeletonGraphEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim // 2,
            output_dim=self.hidden_dim,
            adjacency_matrix=adj_matrix,
            num_layers=2,
            dropout=self.dropout
        )
        
        # 2. TCN Backbone
        self.tcn = TCN(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            num_layers=model_config.get('num_layers', 4),
            kernel_size=model_config.get('kernel_size', 3),
            dropout=self.dropout
        )
        
        # 3. Temporal Attention
        if self.use_attention:
            self.attention = TemporalAttention(
                hidden_dim=self.hidden_dim,
                num_heads=1,
                dropout=self.dropout
            )
        
        # 4. Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, self.num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classifier weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input keypoints (B, T, V, C)
               B=batch, T=time, V=vertices, C=channels (5: x,y,vis,dx,dy)
               
        Returns:
            logits: Class logits (B, num_classes)
            attention_weights: Attention weights (B, T) if use_attention else None
        """
        # 1. Skeleton Graph Encoding
        # Input: (B, T, V, C) -> Output: (B, T, hidden_dim)
        x = self.graph_encoder(x)
        
        # 2. Temporal Convolution
        # Input: (B, T, hidden_dim) -> Output: (B, T, hidden_dim)
        x = self.tcn(x)
        
        # 3. Temporal Attention / Pooling
        if self.use_attention:
            # Input: (B, T, hidden_dim) -> Output: (B, hidden_dim), (B, T)
            x, attention_weights = self.attention(x)
        else:
            # Global average pooling
            x = x.mean(dim=1)  # (B, hidden_dim)
            attention_weights = None
        
        # 4. Classification
        logits = self.classifier(x)  # (B, num_classes)
        
        return logits, attention_weights
    
    def predict(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Single sample prediction with confidence.
        
        Args:
            x: Input keypoints (B, T, V, C) or (T, V, C)
            
        Returns:
            predicted_classes: Predicted class indices (B,)
            confidences: Prediction confidences (B,)
            attention_weights: Attention weights if available
        """
        # Add batch dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        self.eval()
        with torch.no_grad():
            logits, attention_weights = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            confidences, predicted_classes = probs.max(dim=-1)
        
        return predicted_classes, confidences, attention_weights
    
    def get_feature_maps(
        self, 
        x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Get intermediate feature maps for visualization.
        
        Args:
            x: Input keypoints (B, T, V, C)
            
        Returns:
            Dictionary of feature maps at each stage
        """
        features = {}
        
        # Graph features
        graph_features = self.graph_encoder(x)
        features['graph'] = graph_features
        
        # TCN features
        tcn_features = self.tcn(graph_features)
        features['tcn'] = tcn_features
        
        # Attention
        if self.use_attention:
            attended, weights = self.attention(tcn_features)
            features['attended'] = attended
            features['attention_weights'] = weights
        
        return features
    
    @classmethod
    def from_config(cls, config_path: str) -> 'SignClassifier':
        """
        Create model from config file.
        
        Args:
            config_path: Path to YAML config file
            
        Returns:
            SignClassifier instance
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(config)
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size_mb(self) -> float:
        """Get model size in megabytes."""
        param_size = sum(
            p.numel() * p.element_size() 
            for p in self.parameters()
        )
        buffer_size = sum(
            b.numel() * b.element_size() 
            for b in self.buffers()
        )
        return (param_size + buffer_size) / (1024 ** 2)


class LightweightSignClassifier(nn.Module):
    """
    Lightweight version of SignClassifier for faster inference.
    
    Uses smaller dimensions and fewer layers for
    real-time inference on edge devices.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize lightweight classifier.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        
        self.config = config
        model_config = config.get('model', {})
        
        # Smaller dimensions
        self.num_classes = model_config.get('num_classes', 10)
        self.hidden_dim = 64  # Smaller than full model
        
        # Input processing: flatten vertices
        num_vertices = 75  # 33 pose + 21*2 hands
        self.input_dim = num_vertices * 5  # x, y, vis, dx, dy
        
        # Simple input projection
        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)
        
        # Lightweight TCN (fewer layers, smaller kernel)
        self.tcn = TCN(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            num_layers=2,  # Fewer layers
            kernel_size=3,
            dropout=0.2
        )
        
        # Simple attention
        self.attention = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(self.hidden_dim // 4, 1)
        )
        
        # Classifier
        self.classifier = nn.Linear(self.hidden_dim, self.num_classes)
    
    def forward(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input keypoints (B, T, V, C)
            
        Returns:
            logits: Class logits (B, num_classes)
            attention_weights: Attention weights (B, T)
        """
        B, T, V, C = x.shape
        
        # Flatten vertices: (B, T, V*C)
        x = x.view(B, T, V * C)
        
        # Project to hidden dim
        x = self.input_proj(x)
        
        # TCN
        x = self.tcn(x)
        
        # Attention
        scores = self.attention(x).squeeze(-1)  # (B, T)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Weighted sum
        x = torch.bmm(attention_weights.unsqueeze(1), x).squeeze(1)
        
        # Classify
        logits = self.classifier(x)
        
        return logits, attention_weights


if __name__ == "__main__":
    # Test models
    config = {
        'model': {
            'num_classes': 10,
            'hidden_dim': 128,
            'num_layers': 4,
            'kernel_size': 3,
            'dropout': 0.3,
            'use_attention': True
        },
        'keypoints': {
            'use_pose': True,
            'use_hands': True
        }
    }
    
    B, T, V, C = 2, 30, 75, 5
    x = torch.randn(B, T, V, C)
    
    # Test full model
    print("Full SignClassifier:")
    model = SignClassifier(config)
    logits, attn = model(x)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Attention shape: {attn.shape if attn is not None else 'None'}")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Model size: {model.get_model_size_mb():.2f} MB")
    
    # Test prediction
    pred_class, conf, _ = model.predict(x)
    print(f"  Predictions: {pred_class}")
    print(f"  Confidences: {conf}")
    
    # Test lightweight model
    print("\nLightweight SignClassifier:")
    light_model = LightweightSignClassifier(config)
    logits_light, attn_light = light_model(x)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {logits_light.shape}")
    print(f"  Parameters: {sum(p.numel() for p in light_model.parameters()):,}")
    
    size_mb = sum(
        p.numel() * p.element_size() for p in light_model.parameters()
    ) / (1024 ** 2)
    print(f"  Model size: {size_mb:.2f} MB")
