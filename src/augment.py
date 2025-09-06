#!/usr/bin/env python3
"""
Load-time data augmentation for the synthetic math dataset.

Provides utilities for rotating factor pairs and converting to ML-friendly formats.
"""

import random
from typing import Dict, Any
import torch
from datasets import Dataset


def rotate_pair(example: Dict[str, Any], probability: float = 0.5) -> Dict[str, Any]:
    """
    Randomly swap factor1 and factor2 with given probability.
    
    Args:
        example: Single dataset example
        probability: Probability of swapping (default: 0.5)
    
    Returns:
        Example with potentially swapped factors
    """
    if random.random() < probability:
        # Create swapped version
        swapped = {}
        for key, value in example.items():
            if key.startswith('factor1_'):
                new_key = key.replace('factor1_', 'factor2_')
                swapped[new_key] = value
            elif key.startswith('factor2_'):
                new_key = key.replace('factor2_', 'factor1_')  
                swapped[new_key] = value
            else:
                swapped[key] = value
        return swapped
    return example


def create_rotation_transform(seed: int = None, probability: float = 0.5):
    """
    Create a rotation transform function with fixed seed.
    
    Args:
        seed: Random seed for reproducibility
        probability: Probability of rotation
    
    Returns:
        Transform function
    """
    if seed is not None:
        random.seed(seed)
    
    def transform(example):
        return rotate_pair(example, probability)
    
    return transform


def to_pytorch_tensors(batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Convert batch to PyTorch tensors.
    
    Args:
        batch: Batch from HuggingFace dataset
    
    Returns:
        Dictionary with PyTorch tensors
    """
    tensor_batch = {}
    
    for key, values in batch.items():
        if key.endswith('_bits'):
            # Convert bit arrays to uint8 tensors
            tensor_batch[key] = torch.tensor(values, dtype=torch.uint8)
        elif key.endswith('_dec'):
            # Keep decimal strings as-is (or convert to int if needed)
            tensor_batch[key] = values  # or torch.tensor([int(v) for v in values])
        elif key.endswith(('_popcount', '_msb_index', '_bit_length', '_trailing_zeros')):
            # Convert numeric features to tensors
            tensor_batch[key] = torch.tensor(values, dtype=torch.uint8)
        elif key.startswith(('pair_', 'product_is_', 'factor1_is_', 'factor2_is_')):
            # Convert boolean features
            tensor_batch[key] = torch.tensor(values, dtype=torch.bool)
        else:
            # Keep other fields as-is
            tensor_batch[key] = values
    
    return tensor_batch


def create_ml_dataset(dataset_path: str, 
                     apply_rotation: bool = True,
                     rotation_probability: float = 0.5,
                     seed: int = None) -> Dataset:
    """
    Load dataset with ML-friendly preprocessing.
    
    Args:
        dataset_path: Path to the parquet file
        apply_rotation: Whether to apply random rotation
        rotation_probability: Probability of rotation per example
        seed: Random seed for reproducibility
    
    Returns:
        Processed HuggingFace Dataset
    """
    from datasets import load_dataset
    
    # Load dataset
    dataset = load_dataset('parquet', data_files=dataset_path)['train']
    
    # Apply rotation if requested
    if apply_rotation:
        rotation_fn = create_rotation_transform(seed, rotation_probability)
        dataset = dataset.map(rotation_fn)
    
    # Convert to tensors
    dataset = dataset.with_transform(to_pytorch_tensors)
    
    return dataset


def example_usage():
    """Example of how to use the augmentation functions."""
    # Example: Load 12-bit dataset with rotation
    dataset = create_ml_dataset(
        'data/prime_products_12bit.parquet',
        apply_rotation=True,
        rotation_probability=0.5,
        seed=42
    )
    
    # Create DataLoader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Iterate through batches
    for batch in dataloader:
        factor1_bits = batch['factor1_bits']  # Shape: [32, N]
        factor2_bits = batch['factor2_bits']  # Shape: [32, N] 
        product_bits = batch['product_bits']  # Shape: [32, 2N]
        
        # Your ML model training code here
        print(f"Batch shape: {factor1_bits.shape}")
        break


if __name__ == "__main__":
    example_usage()
