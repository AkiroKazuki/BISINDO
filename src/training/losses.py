"""
Loss Functions for Sign Classification

Includes:
- Cross-entropy with label smoothing
- Focal loss for class imbalance
- Confidence penalty for calibration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Focuses training on hard examples by down-weighting
    the contribution of easy examples.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Reference: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Initialize focal loss.
        
        Args:
            alpha: Class weights tensor (num_classes,)
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self, 
        inputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Predicted logits (B, num_classes)
            targets: Ground truth labels (B,)
            
        Returns:
            Focal loss value
        """
        # Compute softmax probabilities
        p = F.softmax(inputs, dim=-1)
        
        # Get the probability of the correct class
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply class weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_weight = alpha_t * focal_weight
        
        # Compute focal loss
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """
    Cross-entropy with Label Smoothing.
    
    Prevents overconfident predictions by smoothing
    the target distribution.
    """
    
    def __init__(
        self,
        num_classes: int,
        smoothing: float = 0.1,
        reduction: str = 'mean'
    ):
        """
        Initialize label smoothing loss.
        
        Args:
            num_classes: Number of classes
            smoothing: Label smoothing factor (0 = no smoothing)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.reduction = reduction
        
        # Smoothed values
        self.confidence = 1.0 - smoothing
        self.smooth_value = smoothing / (num_classes - 1)
    
    def forward(
        self, 
        inputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute label smoothing loss.
        
        Args:
            inputs: Predicted logits (B, num_classes)
            targets: Ground truth labels (B,)
            
        Returns:
            Label smoothing loss value
        """
        # Create smoothed target distribution
        with torch.no_grad():
            smooth_targets = torch.full_like(inputs, self.smooth_value)
            smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        # Compute log softmax
        log_probs = F.log_softmax(inputs, dim=-1)
        
        # Compute cross-entropy with smoothed targets
        loss = -smooth_targets * log_probs
        loss = loss.sum(dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CombinedLoss(nn.Module):
    """
    Combined loss function for sign classification.
    
    Combines classification loss with optional regularization terms.
    """
    
    def __init__(
        self,
        num_classes: int,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1,
        attention_entropy_weight: float = 0.01,
        class_weights: Optional[torch.Tensor] = None
    ):
        """
        Initialize combined loss.
        
        Args:
            num_classes: Number of classes
            focal_gamma: Gamma for focal loss
            label_smoothing: Label smoothing factor
            attention_entropy_weight: Weight for attention entropy regularization
            class_weights: Optional class weights for imbalance
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.attention_entropy_weight = attention_entropy_weight
        
        # Main classification loss
        if focal_gamma > 0:
            self.cls_loss = FocalLoss(
                alpha=class_weights,
                gamma=focal_gamma
            )
        else:
            self.cls_loss = LabelSmoothingLoss(
                num_classes=num_classes,
                smoothing=label_smoothing
            )
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            logits: Predicted logits (B, num_classes)
            targets: Ground truth labels (B,)
            attention_weights: Optional attention weights (B, T)
            
        Returns:
            Total loss value
        """
        # Classification loss
        loss = self.cls_loss(logits, targets)
        
        # Attention entropy regularization
        # Encourages the model to focus on specific frames
        if attention_weights is not None and self.attention_entropy_weight > 0:
            # Compute entropy of attention distribution
            entropy = -(attention_weights * torch.log(attention_weights + 1e-10)).sum(dim=-1)
            entropy_loss = entropy.mean()
            loss = loss + self.attention_entropy_weight * entropy_loss
        
        return loss


if __name__ == "__main__":
    # Test losses
    B, num_classes = 8, 10
    
    logits = torch.randn(B, num_classes)
    targets = torch.randint(0, num_classes, (B,))
    attention = F.softmax(torch.randn(B, 30), dim=-1)
    
    # Test focal loss
    focal = FocalLoss(gamma=2.0)
    focal_loss = focal(logits, targets)
    print(f"Focal loss: {focal_loss.item():.4f}")
    
    # Test label smoothing
    smooth = LabelSmoothingLoss(num_classes=num_classes, smoothing=0.1)
    smooth_loss = smooth(logits, targets)
    print(f"Label smoothing loss: {smooth_loss.item():.4f}")
    
    # Test combined loss
    combined = CombinedLoss(
        num_classes=num_classes,
        focal_gamma=2.0,
        attention_entropy_weight=0.01
    )
    combined_loss = combined(logits, targets, attention)
    print(f"Combined loss: {combined_loss.item():.4f}")
    
    # Compare with standard cross-entropy
    ce_loss = F.cross_entropy(logits, targets)
    print(f"Standard CE loss: {ce_loss.item():.4f}")
