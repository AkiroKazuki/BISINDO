"""
Training Loop with Best Practices

Features:
- Mixed precision training (when supported)
- Learning rate scheduling
- Early stopping
- Checkpointing
- TensorBoard logging
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from pathlib import Path
from typing import Dict, Optional, Callable
import yaml
import json
from datetime import datetime
from tqdm import tqdm

from .losses import CombinedLoss
from .metrics import compute_accuracy, compute_f1, MetricTracker


class Trainer:
    """
    Training manager for sign language classifier.
    
    Handles training loop, validation, checkpointing,
    and logging with best practices.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs"
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            config: Training configuration
            train_loader: Training data loader
            val_loader: Validation data loader
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory to save logs
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Extract training config
        train_config = config.get('training', {})
        self.epochs = train_config.get('epochs', 30)
        self.lr = train_config.get('learning_rate', 0.001)
        self.patience = train_config.get('early_stopping_patience', 5)
        
        # Device
        device_str = train_config.get('device', 'cpu')
        if device_str == 'mps' and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif device_str == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        print(f"Training on device: {self.device}")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Loss function
        model_config = config.get('model', {})
        self.criterion = CombinedLoss(
            num_classes=model_config.get('num_classes', 10),
            focal_gamma=2.0,
            attention_entropy_weight=0.01
        )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=3
        )
        
        # Directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Experiment ID
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Metric tracker
        self.tracker = MetricTracker(['train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_f1'])
        
        # Training state
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.history = []
    
    def train(self) -> Dict:
        """
        Full training loop.
        
        Returns:
            Training history with metrics
        """
        print(f"\n{'='*60}")
        print(f"Starting training: {self.experiment_id}")
        print(f"{'='*60}")
        print(f"Epochs: {self.epochs}")
        print(f"Learning rate: {self.lr}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print(f"{'='*60}\n")
        
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            print("-" * 40)
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_metrics['accuracy'])
            
            # Store metrics
            epoch_metrics = {
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_acc': train_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'val_acc': val_metrics['accuracy'],
                'val_f1': val_metrics['f1_macro'],
                'lr': self.optimizer.param_groups[0]['lr']
            }
            self.history.append(epoch_metrics)
            
            # Print metrics
            print(f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Train Acc: {train_metrics['accuracy']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.4f} | "
                  f"Val F1: {val_metrics['f1_macro']:.4f}")
            
            # Check for improvement
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.best_epoch = epoch + 1
                self.epochs_without_improvement = 0
                
                # Save best model
                self.save_checkpoint(epoch + 1, val_metrics, is_best=True)
                print(f"  ✓ New best model saved!")
            else:
                self.epochs_without_improvement += 1
                print(f"  No improvement for {self.epochs_without_improvement} epochs")
            
            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
            
            # Regular checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch + 1, val_metrics, is_best=False)
        
        # Save training history
        self.save_history()
        
        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Best validation accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch}")
        print(f"{'='*60}\n")
        
        return {'history': self.history, 'best_epoch': self.best_epoch, 'best_val_acc': self.best_val_acc}
    
    def train_epoch(self) -> Dict:
        """
        Single training epoch.
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for batch_idx, (data, targets) in enumerate(pbar):
            # Move to device
            data = data.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits, attention = self.model(data)
            
            # Compute loss
            loss = self.criterion(logits, targets, attention)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item() * data.size(0)
            predictions = logits.argmax(dim=-1)
            total_correct += (predictions == targets).sum().item()
            total_samples += data.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': total_correct / total_samples
            })
        
        return {
            'loss': total_loss / total_samples,
            'accuracy': total_correct / total_samples
        }
    
    def validate(self) -> Dict:
        """
        Validation step.
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in tqdm(self.val_loader, desc="Validation", leave=False):
                # Move to device
                data = data.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                logits, attention = self.model(data)
                
                # Compute loss
                loss = self.criterion(logits, targets, attention)
                
                # Track metrics
                total_loss += loss.item() * data.size(0)
                predictions = logits.argmax(dim=-1)
                
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
        
        # Concatenate all predictions
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        
        # Compute metrics
        accuracy = compute_accuracy(all_predictions, all_targets)
        f1_macro = compute_f1(all_predictions, all_targets, 'macro')
        
        return {
            'loss': total_loss / len(all_targets),
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'predictions': all_predictions,
            'targets': all_targets
        }
    
    def save_checkpoint(
        self, 
        epoch: int, 
        metrics: Dict,
        is_best: bool = False
    ) -> None:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            metrics: Current metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'best_val_acc': self.best_val_acc
        }
        
        if is_best:
            path = self.checkpoint_dir / f"best_model.pt"
        else:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"Validation accuracy: {checkpoint['metrics'].get('accuracy', 'N/A'):.4f}")
    
    def save_history(self) -> None:
        """Save training history to JSON file."""
        history_path = self.log_dir / f"history_{self.experiment_id}.json"
        
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"Training history saved to {history_path}")


def train_model(
    config_path: str = "config/default.yaml",
    data_dir: str = "data/splits",
    checkpoint_dir: str = "checkpoints",
    log_dir: str = "logs"
) -> Dict:
    """
    Convenience function to train model from config file.
    
    Args:
        config_path: Path to configuration file
        data_dir: Directory containing data splits
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory to save logs
        
    Returns:
        Training results
    """
    from ..models.classifier import SignClassifier
    from ..data.dataset import create_dataloaders
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    model = SignClassifier(config)
    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Model size: {model.get_model_size_mb():.2f} MB")
    
    # Create data loaders
    dataloaders = create_dataloaders(
        config_path=config_path,
        splits_dir=data_dir
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir
    )
    
    # Train
    results = trainer.train()
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/default.yaml')
    parser.add_argument('--data_dir', type=str, default='data/splits')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--debug', action='store_true', help='Run quick debug training')
    args = parser.parse_args()
    
    if args.debug:
        print("Running in debug mode...")
        # Override config for quick testing
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        config['training']['epochs'] = 2
        config['training']['batch_size'] = 4
        
        # Save temporary config
        debug_config_path = 'config/debug.yaml'
        with open(debug_config_path, 'w') as f:
            yaml.dump(config, f)
        
        args.config = debug_config_path
    
    results = train_model(
        config_path=args.config,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir
    )
    
    print("\nTraining Results:")
    print(f"  Best epoch: {results['best_epoch']}")
    print(f"  Best validation accuracy: {results['best_val_acc']:.4f}")
