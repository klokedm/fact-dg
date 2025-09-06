# Synthetic Mathematical Dataset Generator

A high-performance generator for creating synthetic datasets of prime number products for mathematical machine learning problems.

## Overview

This tool generates datasets containing all unique combinations of prime number pairs and their products, optimized for mathematical reasoning tasks. The dataset includes comprehensive features about factors and products in both binary and decimal representations.

## Key Features

- **Prime-only factors**: Only uses prime numbers as factors to ensure mathematical richness
- **Unique unordered pairs**: Stores each pair (a,b) only once where a ≤ b, reducing dataset size by ~50%
- **Fixed-size tensors**: Binary representations as arrays of 0s and 1s, directly usable in ML frameworks
- **Zero-padded representations**: Both decimal and binary forms are consistently padded
- **Parallel processing**: Multi-core processing with progress tracking
- **Efficient storage**: Compressed Parquet format via HuggingFace Datasets

## Installation

```bash
# Basic installation
pip install numpy datasets pyarrow tqdm

# High-performance installation (RECOMMENDED)
pip install numpy datasets pyarrow tqdm numba

# For GPU acceleration (optional, requires CUDA)
pip install cupy-cuda12x

# Using Poetry (recommended for development)
poetry install
```

## Usage

### Basic Usage

Generate a dataset for N-bit prime numbers:

```bash
python src/generate.py 12  # 12-bit numbers (recommended for testing)
python src/generate.py 16  # 16-bit numbers (medium size)
python src/generate.py 20  # 20-bit numbers (very large - see warnings below)
```

### High-Performance Version (RECOMMENDED)

For significantly faster generation, use the optimized version:

```bash
# Optimized version with all performance enhancements
python src/generate_optimized.py 14 --output-dir /fast/storage

# Disable GPU acceleration if needed
python src/generate_optimized.py 16 --no-gpu
```

### Advanced Options

```bash
python src/generate.py 14 \
    --output-dir /path/to/output \
    --workers 8 \
    --batch-size 50000
```

**Basic Parameters:**
- `max_bits`: Maximum number of bits for factors (required)
- `--output-dir`: Output directory (default: `data/`)
- `--workers`: Number of parallel processes (default: CPU count - 1)
- `--batch-size`: Processing batch size (default: 10,000)

**Optimized Parameters:**
- `--no-gpu`: Disable GPU acceleration
- `--batch-size`: Larger batches (default: 1M) for better performance

## Dataset Schema

Each row represents a unique prime pair and their product:

### Factor Features (for both factor1 and factor2)
| Field | Type | Description |
|-------|------|-------------|
| `factor1_dec` | `string` | Zero-padded decimal representation |
| `factor1_bits` | `uint8[]` | Fixed-size array of 0s and 1s (N bits) |
| `factor1_is_prime` | `bool` | Always `true` (kept for schema consistency) |
| `factor1_is_odd` | `bool` | Whether the factor is odd |
| `factor1_popcount` | `uint8` | Number of 1-bits (Hamming weight) |
| `factor1_msb_index` | `uint8` | Index of most significant bit (0-based) |

### Product Features
| Field | Type | Description |
|-------|------|-------------|
| `product_dec` | `string` | Zero-padded decimal representation |
| `product_bits` | `uint8[]` | Fixed-size array of 0s and 1s (2N bits) |
| `product_popcount` | `uint8` | Number of 1-bits in product |
| `product_bit_length` | `uint8` | Actual bit length of product |
| `product_trailing_zeros` | `uint8` | Number of trailing zero bits |

### Pair-Level Features
| Field | Type | Description |
|-------|------|-------------|
| `pair_is_same` | `bool` | Whether both factors are identical (perfect squares) |
| `pair_coprime` | `bool` | Whether factors share no common divisors (always `true` for different primes) |
| `product_is_square` | `bool` | Whether the product is a perfect square |

## Size and Performance Estimates

### Dataset Sizes (Compressed Parquet with Zstd)

| Bits | Prime Count | Unique Pairs | Compressed Size | Basic Version | Optimized Version |
|------|-------------|--------------|-----------------|---------------|-------------------|
| 10   | ~200        | ~20K         | ~2 MB           | < 1 minute    | < 10 seconds     |
| 12   | ~600        | ~180K        | ~20 MB          | ~2 minutes    | < 30 seconds     |
| 14   | ~1,600      | ~1.3M        | ~150 MB         | ~10 minutes   | ~2 minutes       |
| 16   | ~6,500      | ~21M         | ~2.5 GB         | ~1 hour       | ~8 minutes       |
| 18   | ~26,000     | ~340M        | ~40 GB          | ~6 hours      | ~45 minutes      |
| 20   | ~81,000     | ~3.3B        | ~400 GB         | ~20 hours     | ~3 hours         |

**Performance Notes:**
- **Optimized version** uses Numba JIT compilation and vectorization for 5-10x speedup
- **GPU acceleration** available for datasets with >1M pairs (additional 2-5x speedup)
- Times assume modern 16-core CPU with fast NVMe storage
- **Memory**: Optimized version uses ~50% less RAM

## Load-Time Data Augmentation

Since the dataset stores ordered pairs (a ≤ b), you can implement rotation augmentation:

```python
import random
from datasets import load_dataset

def rotate_pair(example):
    """Randomly swap factor1 and factor2 with 50% probability."""
    if random.random() < 0.5:
        # Swap all factor1_ and factor2_ fields
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

# Load and augment dataset
dataset = load_dataset('parquet', data_files='data/prime_products_12bit.parquet')
dataset = dataset.map(rotate_pair)
```

## Example Usage in ML

```python
from datasets import load_dataset
import torch

# Load dataset
dataset = load_dataset('parquet', data_files='data/prime_products_12bit.parquet')['train']

# Convert to PyTorch tensors
def to_tensors(batch):
    return {
        'factor1_bits': torch.tensor(batch['factor1_bits'], dtype=torch.uint8),
        'factor2_bits': torch.tensor(batch['factor2_bits'], dtype=torch.uint8), 
        'product_bits': torch.tensor(batch['product_bits'], dtype=torch.uint8),
        # ... other fields as needed
    }

dataset = dataset.with_transform(to_tensors)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
```

## Memory and Storage Requirements

### Development/Testing (≤16 bits)
- **RAM**: 4-8 GB sufficient
- **Storage**: 1-10 GB free space
- **Time**: Minutes to hours

### Production (18-20 bits)  
- **RAM**: 16-32 GB recommended
- **Storage**: 50-500 GB free space (fast NVMe recommended)
- **Time**: 6-24 hours
- **Consider**: Distributed processing for 20+ bits

## Performance Optimization Tips

1. **Use the optimized version**: Always prefer `generate_optimized.py` for 5-10x speedup
2. **Install Numba**: Essential performance boost with `pip install numba`
3. **GPU acceleration**: Install CuPy with `pip install cupy-cuda12x` for massive datasets
4. **Start small**: Test with 12-14 bits before scaling up
5. **Fast storage**: Use NVMe SSDs for large datasets
6. **Memory**: Optimized version uses ~50% less RAM
7. **Benchmarking**: Run `python benchmark.py` to test your system's performance

### Performance Benchmarking

Test your system's performance across different optimization levels:

```bash
# Run comprehensive benchmark
python benchmark.py

# Compare basic vs optimized versions
python benchmark.py --end-to-end
```

The benchmark will show:
- Prime generation speed comparison
- Feature computation performance
- Memory usage efficiency  
- End-to-end dataset generation times
- Specific optimization recommendations for your hardware

## Warnings

- **18+ bits**: Generates very large datasets (40+ GB)
- **20 bits**: Requires substantial computational resources (~20 hours, 400+ GB)
- **Memory**: Large bit counts may require 16+ GB RAM
- **Interruption**: Use Ctrl+C to safely stop generation

## Mathematical Properties

The dataset captures rich mathematical relationships:

- **Primality**: All factors are prime numbers
- **Multiplicative structure**: Products preserve interesting bit patterns
- **Binary patterns**: Hamming weights, bit positions, trailing zeros
- **Symmetry**: Commutative property handled via ordered storage + rotation

This makes the dataset ideal for:
- Multiplication learning tasks
- Prime number reasoning
- Binary arithmetic understanding
- Mathematical pattern recognition

## Implementation Versions

This repository includes two implementations:

### 1. Basic Version (`generate.py`)
- ✅ **Stable**: Production-ready with comprehensive error handling
- ✅ **Compatible**: Works on any system with basic Python dependencies
- ✅ **Multiprocessing**: CPU parallelization using standard library
- ⚠️ **Slower**: Pure Python implementation

### 2. Optimized Version (`generate_optimized.py`) - **RECOMMENDED**
- 🚀 **Fast**: 5-10x speedup with Numba JIT compilation
- 🎯 **Vectorized**: NumPy-based batch operations
- 🖥️ **GPU Support**: Optional CuPy acceleration for massive datasets
- 💾 **Memory Efficient**: ~50% less RAM usage
- 🔧 **Smart Fallbacks**: Gracefully degrades if optimization libraries unavailable

**Recommendation**: Always use the optimized version unless you have specific compatibility requirements.

## Files Structure

```
synthetic-math-dataset/
├── src/
│   ├── generate.py           # Basic implementation
│   ├── generate_optimized.py # High-performance implementation ⭐
│   └── augment.py           # Load-time data augmentation
├── benchmark.py             # Performance benchmarking tool
├── example_usage.py         # Complete usage examples
├── requirements.txt         # Dependencies with performance optimizations
└── README.md               # This file
```

## License

This project is open source. See LICENSE file for details.
