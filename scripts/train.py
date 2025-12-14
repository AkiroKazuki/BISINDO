#!/usr/bin/env python3
"""
Model Training Script

Train the sign language classifier model.

Usage:
    python scripts/train.py --config config/default.yaml
    python scripts/train.py --config config/default.yaml --debug
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import torch

from src.models.classifier import SignClassifier
from src.data.dataset import BISINDODataset, create_dataloaders
from src.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(
        description="Train BISINDO sign language classifier"
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/default.yaml',
        help='Path to config file'
    )
    
    parser.add_argument(
        '--data_dir', '-d',
        type=str,
        default='data/splits',
        help='Directory containing train/val/test splits'
    )
    
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='checkpoints',
        help='Directory to save checkpoints'
    )
    
    parser.add_argument(
        '--log_dir',
        type=str,
        default='logs',
        help='Directory to save logs'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        help='Path to checkpoint to resume from'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Run quick debug training (2 epochs)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        help='Override number of epochs'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        help='Override batch size'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        help='Override learning rate'
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply overrides
    if args.debug:
        print("Running in DEBUG mode")
        config['training']['epochs'] = 2
        config['training']['batch_size'] = 4
    
    if args.epochs:
        config['training']['epochs'] = args.epochs
    
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    
    if args.lr:
        config['training']['learning_rate'] = args.lr
    
    # Print configuration
    print("\n" + "="*60)
    print("BISINDO Sign Language Classifier Training")
    print("="*60)
    print(f"Config: {args.config}")
    print(f"Data directory: {args.data_dir}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Device: {config['training'].get('device', 'cpu')}")
    print("="*60 + "\n")
    
    # Check if data splits exist
    splits_dir = Path(args.data_dir)
    if not splits_dir.exists() or not (splits_dir / 'train.json').exists():
        print(f"Error: Data splits not found in {args.data_dir}")
        print("Run data preprocessing first to create train/val/test splits.")
        print("\nTo create splits from processed data:")
        print("  python -c \"from src.data.dataset import BISINDODataset; "
              "BISINDODataset.create_splits('data/processed', 'data/splits')\"")
        sys.exit(1)
    
    # Create model
    print("Creating model...")
    model = SignClassifier(config)
    
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Model size: {model.get_model_size_mb():.2f} MB")
    
    # Create data loaders
    print("\nLoading data...")
    dataloaders = create_dataloaders(
        config_path=args.config,
        splits_dir=args.data_dir,
        batch_size=config['training']['batch_size']
    )
    
    print(f"  Train batches: {len(dataloaders['train'])}")
    print(f"  Val batches: {len(dataloaders['val'])}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    print("\nStarting training...\n")
    results = trainer.train()
    
    # Print final results
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best epoch: {results['best_epoch']}")
    print(f"Best validation accuracy: {results['best_val_acc']:.4f}")
    print(f"Best model saved to: {args.checkpoint_dir}/best_model.pt")
    print("="*60 + "\n")
    
    # Evaluate on test set
    if 'test' in dataloaders:
        print("Evaluating on test set...")
        
        # Load best model
        trainer.load_checkpoint(f"{args.checkpoint_dir}/best_model.pt")
        
        # Evaluate on test set
        model.eval()
        
        from src.training.metrics import compute_all_metrics
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in dataloaders['test']:
                data = data.to(trainer.device)
                logits, _ = model(data)
                preds = logits.argmax(dim=-1)
                
                all_preds.append(preds.cpu())
                all_targets.append(targets)
        
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        
        test_metrics = compute_all_metrics(
            all_preds, all_targets,
            class_names=config['dataset']['classes'],
            num_classes=len(config['dataset']['classes'])
        )
        
        print(f"\nTest Results:")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  F1 (macro): {test_metrics['f1_macro']:.4f}")
        print(f"  Precision: {test_metrics['precision_macro']:.4f}")
        print(f"  Recall: {test_metrics['recall_macro']:.4f}")


if __name__ == "__main__":
    main()
