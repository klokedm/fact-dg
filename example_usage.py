#!/usr/bin/env python3
"""
Example usage of the synthetic math dataset generator.

This script demonstrates how to generate and use the dataset for small bit counts.
"""

import os
import sys
import subprocess
from datasets import load_dataset


def run_generation_example():
    """Generate a small dataset for testing."""
    print("=== Generating 10-bit Prime Products Dataset ===")
    print("This will create a small dataset for testing purposes.\n")
    
    # Run the generator
    cmd = [sys.executable, "src/generate.py", "10", "--output-dir", "example_data"]
    
    try:
        subprocess.run(cmd, check=True)
        print("\n✓ Dataset generation completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Generation failed: {e}")
        return False
    except FileNotFoundError:
        print("❌ Could not find the generator script. Make sure you're in the project root.")
        return False
    
    return True


def explore_dataset():
    """Load and explore the generated dataset."""
    dataset_path = "example_data/prime_products_10bit.parquet"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at {dataset_path}")
        return
    
    print("\n=== Exploring the Generated Dataset ===")
    
    # Load dataset
    dataset = load_dataset('parquet', data_files=dataset_path)['train']
    
    print(f"Dataset size: {len(dataset):,} rows")
    print(f"Dataset features: {list(dataset.features.keys())}")
    
    # Show first example
    print("\n--- First Example ---")
    example = dataset[0]
    
    print(f"Factor 1: {example['factor1_dec']} (decimal)")
    print(f"Factor 1 bits: {''.join(map(str, example['factor1_bits']))}")
    print(f"Factor 1 popcount: {example['factor1_popcount']}")
    
    print(f"Factor 2: {example['factor2_dec']} (decimal)")  
    print(f"Factor 2 bits: {''.join(map(str, example['factor2_bits']))}")
    print(f"Factor 2 popcount: {example['factor2_popcount']}")
    
    print(f"Product: {example['product_dec']} (decimal)")
    print(f"Product bits: {''.join(map(str, example['product_bits']))}")
    print(f"Product popcount: {example['product_popcount']}")
    
    print(f"Is same pair: {example['pair_is_same']}")
    print(f"Product is square: {example['product_is_square']}")
    
    # Show some statistics
    print("\n--- Dataset Statistics ---")
    
    # Count perfect squares
    squares = sum(1 for x in dataset if x['product_is_square'])
    print(f"Perfect squares: {squares:,}")
    
    # Count unique factors
    all_factors = set()
    for x in dataset:
        all_factors.add(x['factor1_dec'])
        all_factors.add(x['factor2_dec'])
    print(f"Unique prime factors: {len(all_factors)}")
    
    # Show file size
    file_size_mb = os.path.getsize(dataset_path) / (1024**2)
    print(f"Compressed file size: {file_size_mb:.2f} MB")


def demonstrate_rotation():
    """Demonstrate the rotation augmentation."""
    dataset_path = "example_data/prime_products_10bit.parquet"
    
    if not os.path.exists(dataset_path):
        print("❌ Dataset not found. Run generation first.")
        return
    
    print("\n=== Demonstrating Rotation Augmentation ===")
    
    # Import augmentation functions
    sys.path.append('src')
    from augment import create_ml_dataset
    
    # Load with rotation
    dataset = create_ml_dataset(
        dataset_path,
        apply_rotation=True,
        rotation_probability=1.0,  # Always rotate for demo
        seed=42
    )
    
    print("Original vs Rotated examples:")
    
    # Load original for comparison
    original_dataset = load_dataset('parquet', data_files=dataset_path)['train']
    
    # Show a few examples
    for i in range(3):
        orig = original_dataset[i]
        rotated = dataset[i]
        
        print(f"\nExample {i+1}:")
        print(f"Original:  {orig['factor1_dec']} × {orig['factor2_dec']}")
        print(f"Rotated:   {rotated['factor1_dec']} × {rotated['factor2_dec']}")


def main():
    """Main demonstration function."""
    print("Synthetic Math Dataset Generator - Example Usage")
    print("=" * 50)
    
    # Step 1: Generate dataset
    if not run_generation_example():
        return
    
    # Step 2: Explore dataset
    explore_dataset()
    
    # Step 3: Demonstrate rotation
    demonstrate_rotation()
    
    print("\n" + "=" * 50)
    print("✓ Example completed successfully!")
    print("\nNext steps:")
    print("1. Try generating larger datasets: python src/generate.py 12")
    print("2. Use the dataset in your ML experiments")
    print("3. Implement your mathematical reasoning models")


if __name__ == "__main__":
    main()
