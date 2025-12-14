"""
PyTorch Dataset for BISINDO Sign Recognition

Features:
- Subject-disjoint train/val/test splits
- On-the-fly augmentation during training
- Balanced sampling option
- Support for different input modalities
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import yaml
from collections import Counter

from .preprocessing import KeypointPreprocessor
from .augmentor import KeypointAugmentor


class BISINDODataset(Dataset):
    """
    PyTorch Dataset for BISINDO sign language recognition.
    
    Loads preprocessed keypoint sequences and provides
    data augmentation for training.
    """
    
    def __init__(
        self,
        split_file: str,
        config_path: str = "config/default.yaml",
        augment: bool = False,
        preprocess: bool = True
    ):
        """
        Initialize dataset from split file.
        
        Args:
            split_file: Path to JSON file with sample paths and labels
            config_path: Path to configuration file
            augment: Whether to apply data augmentation
            preprocess: Whether to apply preprocessing
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load split file
        with open(split_file, 'r') as f:
            self.samples = json.load(f)
        
        # Class mapping
        self.classes = self.config['dataset']['classes']
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.idx_to_class = {i: c for i, c in enumerate(self.classes)}
        
        # Initialize preprocessor and augmentor
        self.preprocessor = KeypointPreprocessor(config_path)
        self.augmentor = KeypointAugmentor(config_path) if augment else None
        
        self.augment = augment
        self.preprocess = preprocess
        
        # Sequence length
        self.sequence_length = self.config['training']['sequence_length']
        
        print(f"Dataset loaded: {len(self.samples)} samples")
        print(f"Classes: {len(self.classes)}")
        print(f"Augmentation: {'enabled' if augment else 'disabled'}")
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Return preprocessed sample and label.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (keypoints tensor, label index)
        """
        sample = self.samples[idx]
        
        # Load keypoints
        keypoints = np.load(sample['path'])
        
        # Apply augmentation (during training)
        if self.augmentor is not None:
            keypoints = self.augmentor.augment(keypoints)
        
        # Apply preprocessing
        if self.preprocess:
            keypoints = self.preprocessor.preprocess(
                keypoints,
                normalize=True,
                compute_velocity=True,
                pad_sequence=True
            )
        else:
            # Just pad/truncate
            keypoints = self.preprocessor.pad_or_truncate(
                keypoints, self.sequence_length
            )
        
        # Convert to tensor
        keypoints_tensor = torch.FloatTensor(keypoints)
        
        # Get label
        label = self.class_to_idx[sample['class']]
        
        return keypoints_tensor, label
    
    def get_sample_info(self, idx: int) -> Dict:
        """
        Get metadata for a sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Sample metadata dictionary
        """
        return self.samples[idx]
    
    def get_class_distribution(self) -> Dict[str, int]:
        """
        Get class distribution in dataset.
        
        Returns:
            Dictionary mapping class names to counts
        """
        classes = [s['class'] for s in self.samples]
        return dict(Counter(classes))
    
    def get_balanced_sampler(self) -> WeightedRandomSampler:
        """
        Create balanced sampler for training.
        
        Returns:
            WeightedRandomSampler for balanced class sampling
        """
        class_counts = self.get_class_distribution()
        
        # Compute weights
        weights = []
        for sample in self.samples:
            class_name = sample['class']
            weight = 1.0 / class_counts[class_name]
            weights.append(weight)
        
        return WeightedRandomSampler(
            weights=weights,
            num_samples=len(self.samples),
            replacement=True
        )
    
    @staticmethod
    def create_splits(
        data_dir: str,
        output_dir: str,
        config_path: str = "config/default.yaml",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        random_seed: int = 42
    ) -> Dict[str, str]:
        """
        Create subject-disjoint train/val/test splits.
        
        Args:
            data_dir: Directory containing processed keypoints
            output_dir: Directory to save split files
            config_path: Path to configuration file
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            random_seed: Random seed for reproducibility
            
        Returns:
            Dictionary with paths to split files
        """
        np.random.seed(random_seed)
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        classes = config['dataset']['classes']
        
        # Collect all samples by subject
        data_path = Path(data_dir)
        subjects = {}
        
        for subject_dir in data_path.iterdir():
            if not subject_dir.is_dir():
                continue
            
            subject_id = subject_dir.name
            subjects[subject_id] = []
            
            for class_dir in subject_dir.iterdir():
                if not class_dir.is_dir():
                    continue
                
                class_name = class_dir.name
                if class_name not in classes:
                    continue
                
                for keypoint_file in class_dir.glob("*.npy"):
                    subjects[subject_id].append({
                        'path': str(keypoint_file),
                        'class': class_name,
                        'subject': subject_id,
                        'filename': keypoint_file.name
                    })
        
        # Split subjects or samples
        subject_list = list(subjects.keys())
        np.random.shuffle(subject_list)
        
        n_subjects = len(subject_list)
        
        # If fewer than 3 subjects, use sample-level splitting instead
        if n_subjects < 3:
            print(f"Only {n_subjects} subject(s) found. Using sample-level splitting...")
            
            # Collect all samples
            all_samples = []
            for sid in subject_list:
                all_samples.extend(subjects[sid])
            
            np.random.shuffle(all_samples)
            
            n_samples = len(all_samples)
            n_train = int(n_samples * train_ratio)
            n_val = int(n_samples * val_ratio)
            
            train_samples = all_samples[:n_train]
            val_samples = all_samples[n_train:n_train + n_val]
            test_samples = all_samples[n_train + n_val:]
            
            # Ensure at least 1 sample in each split if possible
            if len(train_samples) == 0 and len(all_samples) >= 1:
                train_samples = all_samples[:max(1, n_samples // 2)]
                remaining = all_samples[len(train_samples):]
                val_samples = remaining[:len(remaining) // 2]
                test_samples = remaining[len(remaining) // 2:]
            
            print(f"\nSample-level split:")
            print(f"  Train: {len(train_samples)} samples")
            print(f"  Val: {len(val_samples)} samples")
            print(f"  Test: {len(test_samples)} samples")
            
            train_subjects = subject_list
            val_subjects = subject_list
            test_subjects = subject_list
        else:
            # Subject-disjoint splitting (normal case)
            n_train = int(n_subjects * train_ratio)
            n_val = int(n_subjects * val_ratio)
            
            train_subjects = subject_list[:n_train]
            val_subjects = subject_list[n_train:n_train + n_val]
            test_subjects = subject_list[n_train + n_val:]
            
            print(f"Subject split:")
            print(f"  Train: {len(train_subjects)} subjects")
            print(f"  Val: {len(val_subjects)} subjects")
            print(f"  Test: {len(test_subjects)} subjects")
            
            # Collect samples for each split
            def collect_samples(subject_ids):
                samples = []
                for sid in subject_ids:
                    samples.extend(subjects[sid])
                return samples
            
            train_samples = collect_samples(train_subjects)
            val_samples = collect_samples(val_subjects)
            test_samples = collect_samples(test_subjects)
            
            print(f"\nSample counts:")
            print(f"  Train: {len(train_samples)}")
            print(f"  Val: {len(val_samples)}")
            print(f"  Test: {len(test_samples)}")
        
        # Save splits
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        splits = {
            'train': train_samples,
            'val': val_samples,
            'test': test_samples
        }
        
        split_files = {}
        for split_name, samples in splits.items():
            file_path = output_path / f"{split_name}.json"
            with open(file_path, 'w') as f:
                json.dump(samples, f, indent=2)
            split_files[split_name] = str(file_path)
            print(f"Saved {split_name} split to {file_path}")
        
        # Save split metadata
        metadata = {
            'train_subjects': train_subjects,
            'val_subjects': val_subjects,
            'test_subjects': test_subjects,
            'config_path': config_path,
            'random_seed': random_seed,
            'split_mode': 'sample' if n_subjects < 3 else 'subject'
        }
        
        metadata_path = output_path / "split_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return split_files


def create_dataloaders(
    config_path: str = "config/default.yaml",
    splits_dir: str = "data/splits",
    batch_size: Optional[int] = None,
    num_workers: int = 4
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for train/val/test sets.
    
    Args:
        config_path: Path to configuration file
        splits_dir: Directory containing split files
        batch_size: Batch size (overrides config if provided)
        num_workers: Number of data loading workers
        
    Returns:
        Dictionary with train/val/test DataLoaders
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if batch_size is None:
        batch_size = config['training']['batch_size']
    
    splits_path = Path(splits_dir)
    
    dataloaders = {}
    
    # Training set (with augmentation)
    train_dataset = BISINDODataset(
        split_file=str(splits_path / "train.json"),
        config_path=config_path,
        augment=True
    )
    
    # Use balanced sampler for training (if dataset is not empty)
    if len(train_dataset) > 0:
        train_sampler = train_dataset.get_balanced_sampler()
        dataloaders['train'] = DataLoader(
            train_dataset,
            batch_size=min(batch_size, len(train_dataset)),
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        dataloaders['train'] = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )
    
    # Validation set (no augmentation)
    val_dataset = BISINDODataset(
        split_file=str(splits_path / "val.json"),
        config_path=config_path,
        augment=False
    )
    
    dataloaders['val'] = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Test set (no augmentation)
    test_dataset = BISINDODataset(
        split_file=str(splits_path / "test.json"),
        config_path=config_path,
        augment=False
    )
    
    dataloaders['test'] = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloaders


if __name__ == "__main__":
    # Test dataset creation
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--output_dir', type=str, default='data/splits')
    parser.add_argument('--config', type=str, default='config/default.yaml')
    args = parser.parse_args()
    
    # Create splits
    splits = BISINDODataset.create_splits(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        config_path=args.config
    )
    
    # Test loading
    print("\nTesting dataset loading...")
    dataset = BISINDODataset(
        split_file=splits['train'],
        config_path=args.config,
        augment=True
    )
    
    # Get a sample
    keypoints, label = dataset[0]
    print(f"Sample shape: {keypoints.shape}")
    print(f"Label: {label} ({dataset.idx_to_class[label]})")
    
    # Check class distribution
    dist = dataset.get_class_distribution()
    print(f"\nClass distribution:")
    for cls, count in sorted(dist.items()):
        print(f"  {cls}: {count}")
