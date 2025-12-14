#!/usr/bin/env python3
"""
Model Evaluation Script

Evaluate trained model on test set and generate reports.

Usage:
    python scripts/evaluate.py --model checkpoints/best_model.pt
    python scripts/evaluate.py --model checkpoints/best_model.pt --visualize
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import yaml
import torch
import numpy as np
from tqdm import tqdm

from src.models.classifier import SignClassifier
from src.data.dataset import BISINDODataset
from src.training.metrics import compute_all_metrics
from src.utils.visualization import plot_confusion_matrix, plot_class_distribution


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained model"
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
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
        help='Directory containing data splits'
    )
    
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='Which split to evaluate on'
    )
    
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default='results',
        help='Directory to save results'
    )
    
    parser.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='Generate visualization plots'
    )
    
    parser.add_argument(
        '--per_condition',
        action='store_true',
        help='Report per lighting/occlusion condition'
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device_str = config.get('training', {}).get('device', 'cpu')
    if device_str == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif device_str == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"\nEvaluating on: {device}")
    
    # Load model
    print(f"Loading model: {args.model}")
    checkpoint = torch.load(args.model, map_location=device)
    
    model = SignClassifier(checkpoint.get('config', config))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Best val accuracy: {checkpoint.get('best_val_acc', 'unknown'):.4f}")
    
    # Load dataset
    split_file = Path(args.data_dir) / f"{args.split}.json"
    if not split_file.exists():
        print(f"Error: Split file not found: {split_file}")
        sys.exit(1)
    
    print(f"\nLoading {args.split} set...")
    dataset = BISINDODataset(
        split_file=str(split_file),
        config_path=args.config,
        augment=False
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    print(f"  Samples: {len(dataset)}")
    print(f"  Classes: {len(dataset.classes)}")
    
    # Evaluate
    print(f"\nEvaluating...")
    
    all_preds = []
    all_targets = []
    all_confidences = []
    sample_infos = []
    
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(tqdm(dataloader)):
            data = data.to(device)
            
            logits, attention = model(data)
            probs = torch.softmax(logits, dim=-1)
            confidences, preds = probs.max(dim=-1)
            
            all_preds.append(preds.cpu())
            all_targets.append(targets)
            all_confidences.append(confidences.cpu())
            
            # Collect sample info for per-condition analysis
            start_idx = batch_idx * dataloader.batch_size
            for i in range(len(targets)):
                if start_idx + i < len(dataset.samples):
                    sample_infos.append(dataset.samples[start_idx + i])
    
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_confidences = torch.cat(all_confidences)
    
    # Compute metrics
    class_names = config['dataset']['classes']
    metrics = compute_all_metrics(
        all_preds, all_targets,
        class_names=class_names,
        num_classes=len(class_names)
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print results
    print("\n" + "="*60)
    print(f"Evaluation Results on {args.split.upper()} set")
    print("="*60)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 (macro): {metrics['f1_macro']:.4f}")
    print(f"F1 (weighted): {metrics['f1_weighted']:.4f}")
    print(f"Precision: {metrics['precision_macro']:.4f}")
    print(f"Recall: {metrics['recall_macro']:.4f}")
    
    print("\nPer-class metrics:")
    print("-"*50)
    for class_name in class_names:
        if class_name in metrics['per_class']:
            cm = metrics['per_class'][class_name]
            print(f"  {class_name:15s} | P: {cm['precision']:.3f} | "
                  f"R: {cm['recall']:.3f} | F1: {cm['f1']:.3f} | "
                  f"N: {int(cm['support'])}")
    print("="*60)
    
    # Per-condition analysis
    if args.per_condition and sample_infos:
        print("\nPer-condition analysis:")
        print("-"*50)
        
        conditions = {}
        for i, (pred, target) in enumerate(zip(all_preds.numpy(), all_targets.numpy())):
            if i < len(sample_infos):
                filename = sample_infos[i].get('filename', '')
                parts = filename.replace('.npy', '').split('_')
                
                if len(parts) >= 4:
                    lighting = parts[-3]
                    occlusion = parts[-2]
                    cond_key = f"{lighting}_{occlusion}"
                    
                    if cond_key not in conditions:
                        conditions[cond_key] = {'correct': 0, 'total': 0}
                    
                    conditions[cond_key]['total'] += 1
                    if pred == target:
                        conditions[cond_key]['correct'] += 1
        
        for cond, stats in sorted(conditions.items()):
            acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            print(f"  {cond:20s} | Accuracy: {acc:.4f} | N: {stats['total']}")
    
    # Save results
    results = {
        'split': args.split,
        'model_path': str(args.model),
        'accuracy': float(metrics['accuracy']),
        'f1_macro': float(metrics['f1_macro']),
        'f1_weighted': float(metrics['f1_weighted']),
        'precision_macro': float(metrics['precision_macro']),
        'recall_macro': float(metrics['recall_macro']),
        'per_class': {
            k: {key: float(v) for key, v in vals.items()}
            for k, vals in metrics['per_class'].items()
        }
    }
    
    results_path = output_dir / f"results_{args.split}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Visualizations
    if args.visualize:
        import matplotlib.pyplot as plt
        
        print("\nGenerating visualizations...")
        
        # Confusion matrix
        fig = plot_confusion_matrix(
            all_targets.numpy(),
            all_preds.numpy(),
            class_names,
            normalize=True,
            save_path=str(output_dir / f"confusion_matrix_{args.split}.png")
        )
        plt.close()
        
        # Class distribution
        class_dist = dataset.get_class_distribution()
        fig = plot_class_distribution(
            class_dist,
            save_path=str(output_dir / f"class_distribution_{args.split}.png")
        )
        plt.close()
        
        # Confidence histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        
        correct_mask = (all_preds == all_targets).numpy()
        
        ax.hist(all_confidences.numpy()[correct_mask], bins=50, alpha=0.7, 
                label='Correct', color='green')
        ax.hist(all_confidences.numpy()[~correct_mask], bins=50, alpha=0.7, 
                label='Incorrect', color='red')
        
        ax.set_xlabel('Confidence', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Prediction Confidence Distribution', fontsize=14)
        ax.legend()
        
        plt.savefig(output_dir / f"confidence_histogram_{args.split}.png", 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to: {output_dir}")
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
