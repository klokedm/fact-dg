#!/usr/bin/env python3
"""
Optimized Synthetic Math Dataset Generator

High-performance version using Numba JIT, NumPy vectorization, 
and optional GPU acceleration for maximum throughput.
"""

import argparse
import os
import sys
import math
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from datasets import Dataset, Features, Value, Sequence
from tqdm import tqdm

# Performance optimizations
try:
    from numba import jit, prange, types
    from numba.typed import List as NumbaList
    NUMBA_AVAILABLE = True
except ImportError:
    print("WARNING: Numba not available. Install with: pip install numba")
    NUMBA_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = cp.cuda.is_available()
    if CUPY_AVAILABLE:
        print(f"✓ CUDA available: {cp.cuda.get_device_name()}")
except ImportError:
    CUPY_AVAILABLE = False

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False


# ============================================================================
# OPTIMIZED PRIME GENERATION
# ============================================================================

if NUMBA_AVAILABLE:
    @jit(nopython=True, cache=True)
    def fast_sieve_of_eratosthenes(limit: int) -> np.ndarray:
        """Ultra-fast JIT-compiled prime sieve with vectorized operations."""
        if limit < 2:
            return np.array([], dtype=np.int32)
        
        # Use numpy boolean array for efficiency
        prime = np.ones(limit + 1, dtype=np.bool_)
        prime[0] = prime[1] = False
        
        # Optimized sieve with vectorized multiple marking
        sqrt_limit = int(np.sqrt(limit)) + 1
        for p in range(2, sqrt_limit):
            if prime[p]:
                # Vectorized marking of multiples
                start = p * p
                prime[start::p] = False
        
        # Extract prime indices efficiently
        return np.where(prime)[0].astype(np.int32)

    @jit(nopython=True, cache=True)  
    def fast_int_to_bits_batch(numbers: np.ndarray, bit_length: int) -> np.ndarray:
        """Fast batch conversion of integers to bit arrays."""
        n = len(numbers)
        result = np.zeros((n, bit_length), dtype=np.uint8)
        
        for idx in prange(n):  # Parallel loop
            num = numbers[idx]
            for bit_pos in range(bit_length):
                result[idx, bit_length - 1 - bit_pos] = (num >> bit_pos) & 1
        
        return result
    
    @jit(nopython=True, cache=True)
    def fast_popcount_batch(numbers: np.ndarray) -> np.ndarray:
        """Fast batch popcount using bit manipulation."""
        result = np.zeros(len(numbers), dtype=np.uint8)
        for i in prange(len(numbers)):
            num = numbers[i]
            count = 0
            while num:
                count += num & 1
                num >>= 1
            result[i] = count
        return result
    
    @jit(nopython=True, cache=True)
    def fast_msb_index_batch(numbers: np.ndarray) -> np.ndarray:
        """Fast batch MSB index calculation."""
        result = np.zeros(len(numbers), dtype=np.uint8)
        for i in prange(len(numbers)):
            num = numbers[i]
            if num == 0:
                result[i] = 0
            else:
                msb = 0
                while (1 << (msb + 1)) <= num:
                    msb += 1
                result[i] = msb
        return result
    
    @jit(nopython=True, cache=True)
    def fast_trailing_zeros_batch(numbers: np.ndarray) -> np.ndarray:
        """Fast batch trailing zeros calculation."""
        result = np.zeros(len(numbers), dtype=np.uint8)
        for i in prange(len(numbers)):
            num = numbers[i]
            if num == 0:
                result[i] = 0
            else:
                count = 0
                while (num & (1 << count)) == 0:
                    count += 1
                result[i] = count
        return result

else:
    # Fallback implementations without Numba
    def fast_sieve_of_eratosthenes(limit: int) -> np.ndarray:
        """Numpy-optimized sieve fallback."""
        if limit < 2:
            return np.array([], dtype=np.int32)
        
        prime = np.ones(limit + 1, dtype=bool)
        prime[0] = prime[1] = False
        
        for p in range(2, int(np.sqrt(limit)) + 1):
            if prime[p]:
                prime[p*p::p] = False
        
        return np.where(prime)[0].astype(np.int32)
    
    def fast_int_to_bits_batch(numbers: np.ndarray, bit_length: int) -> np.ndarray:
        """Vectorized bit conversion fallback."""
        n = len(numbers)
        result = np.zeros((n, bit_length), dtype=np.uint8)
        
        for i, num in enumerate(numbers):
            bits = np.array([(num >> j) & 1 for j in range(bit_length)[::-1]], dtype=np.uint8)
            result[i] = bits
        
        return result
    
    def fast_popcount_batch(numbers: np.ndarray) -> np.ndarray:
        return np.array([bin(x).count('1') for x in numbers], dtype=np.uint8)
    
    def fast_msb_index_batch(numbers: np.ndarray) -> np.ndarray:
        return np.array([x.bit_length() - 1 if x > 0 else 0 for x in numbers], dtype=np.uint8)
    
    def fast_trailing_zeros_batch(numbers: np.ndarray) -> np.ndarray:
        result = np.zeros(len(numbers), dtype=np.uint8)
        for i, num in enumerate(numbers):
            if num == 0:
                result[i] = 0
            else:
                result[i] = (num & -num).bit_length() - 1
        return result


# ============================================================================
# OPTIMIZED FEATURE COMPUTATION
# ============================================================================

class OptimizedFeatureComputer:
    """Vectorized feature computation for maximum performance."""
    
    def __init__(self, max_bits: int):
        self.max_bits = max_bits
        self.factor_bin_len = max_bits
        self.product_bin_len = max_bits * 2
        
        # Pre-calculate padding lengths
        max_factor = 2**max_bits - 1
        max_product = max_factor * max_factor
        self.factor_dec_len = len(str(max_factor))
        self.product_dec_len = len(str(max_product))
    
    def compute_prime_features(self, primes: np.ndarray) -> Dict[str, np.ndarray]:
        """Batch compute all prime features using vectorized operations."""
        print("Computing prime features...")
        
        # Decimal representations (vectorized string formatting)
        decimals = np.array([f'{p:0{self.factor_dec_len}d}' for p in primes])
        
        # Binary bit arrays (vectorized)
        bits = fast_int_to_bits_batch(primes, self.factor_bin_len)
        
        # Mathematical features (all vectorized)
        is_prime = np.ones(len(primes), dtype=bool)  # All are primes
        is_odd = (primes % 2 == 1).astype(bool)
        popcount = fast_popcount_batch(primes)
        msb_index = fast_msb_index_batch(primes)
        
        return {
            'decimals': decimals,
            'bits': bits,
            'is_prime': is_prime,
            'is_odd': is_odd,
            'popcount': popcount,
            'msb_index': msb_index
        }
    
    def compute_product_features(self, products: np.ndarray) -> Dict[str, np.ndarray]:
        """Batch compute all product features."""
        # Decimal representations
        decimals = np.array([f'{p:0{self.product_dec_len}d}' for p in products])
        
        # Binary bit arrays
        bits = fast_int_to_bits_batch(products, self.product_bin_len)
        
        # Mathematical features
        popcount = fast_popcount_batch(products)
        bit_length = np.array([p.bit_length() for p in products], dtype=np.uint8)
        trailing_zeros = fast_trailing_zeros_batch(products)
        
        return {
            'decimals': decimals,
            'bits': bits,
            'popcount': popcount,
            'bit_length': bit_length,
            'trailing_zeros': trailing_zeros
        }


# ============================================================================
# GPU-ACCELERATED PAIR GENERATION (OPTIONAL)
# ============================================================================

def gpu_generate_pair_indices(num_primes: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate all unique pair indices on GPU if available."""
    if not CUPY_AVAILABLE:
        return cpu_generate_pair_indices(num_primes)
    
    print("Using GPU acceleration for pair generation...")
    
    # Generate triangle indices on GPU
    total_pairs = num_primes * (num_primes + 1) // 2
    
    # Efficient GPU-based index generation
    i_indices = cp.zeros(total_pairs, dtype=cp.int32)
    j_indices = cp.zeros(total_pairs, dtype=cp.int32)
    
    # Fill indices efficiently on GPU
    idx = 0
    for i in range(num_primes):
        count = num_primes - i
        i_indices[idx:idx+count] = i
        j_indices[idx:idx+count] = cp.arange(i, num_primes)
        idx += count
    
    return cp.asnumpy(i_indices), cp.asnumpy(j_indices)


def cpu_generate_pair_indices(num_primes: int) -> Tuple[np.ndarray, np.ndarray]:
    """CPU fallback for pair index generation."""
    total_pairs = num_primes * (num_primes + 1) // 2
    i_indices = np.zeros(total_pairs, dtype=np.int32)
    j_indices = np.zeros(total_pairs, dtype=np.int32)
    
    idx = 0
    for i in range(num_primes):
        count = num_primes - i
        i_indices[idx:idx+count] = i
        j_indices[idx:idx+count] = np.arange(i, num_primes)
        idx += count
    
    return i_indices, j_indices


# ============================================================================
# MAIN OPTIMIZED GENERATION FUNCTION
# ============================================================================

def generate_dataset_optimized(max_bits: int, output_dir: str = "data", 
                             batch_size: int = 1000000, use_gpu: bool = True):
    """Optimized dataset generation with all performance enhancements."""
    print(f"🚀 Generating OPTIMIZED dataset for {max_bits}-bit prime pairs...")
    
    # Show optimization status
    print(f"Numba JIT: {'✓' if NUMBA_AVAILABLE else '❌'}")
    print(f"GPU (CuPy): {'✓' if (CUPY_AVAILABLE and use_gpu) else '❌'}")
    
    # Generate primes with optimized sieve
    max_factor = 2**max_bits - 1
    print("Generating prime numbers...")
    primes = fast_sieve_of_eratosthenes(max_factor)
    num_primes = len(primes)
    print(f"Found {num_primes} prime numbers up to {max_factor}")
    
    # Calculate total pairs
    total_pairs = num_primes * (num_primes + 1) // 2
    print(f"Total unique pairs to generate: {total_pairs:,}")
    
    # Pre-compute all prime features (vectorized)
    feature_computer = OptimizedFeatureComputer(max_bits)
    prime_features = feature_computer.compute_prime_features(primes)
    
    # Generate pair indices (GPU-accelerated if available)
    if use_gpu and CUPY_AVAILABLE and total_pairs > 1000000:
        i_indices, j_indices = gpu_generate_pair_indices(num_primes)
    else:
        i_indices, j_indices = cpu_generate_pair_indices(num_primes)
    
    # Compute products (vectorized)
    print("Computing products...")
    products = primes[i_indices] * primes[j_indices]
    
    # Compute product features (vectorized)
    print("Computing product features...")
    product_features = feature_computer.compute_product_features(products)
    
    # Create dataset features schema
    features = create_dataset_features_optimized(max_bits)
    
    # Generate dataset in optimized batches
    def optimized_data_generator():
        """Memory-efficient batch generator."""
        pbar = tqdm(total=total_pairs, desc="Generating dataset", unit="pairs")
        
        for start_idx in range(0, total_pairs, batch_size):
            end_idx = min(start_idx + batch_size, total_pairs)
            batch_indices = slice(start_idx, end_idx)
            
            # Extract batch data
            batch_i = i_indices[batch_indices]
            batch_j = j_indices[batch_indices]
            
            # Generate batch records
            batch_records = []
            for idx in range(len(batch_i)):
                i, j = batch_i[idx], batch_j[idx]
                
                record = {
                    # Factor 1 features
                    'factor1_dec': prime_features['decimals'][i],
                    'factor1_bits': prime_features['bits'][i].tolist(),
                    'factor1_is_prime': prime_features['is_prime'][i],
                    'factor1_is_odd': prime_features['is_odd'][i],
                    'factor1_popcount': prime_features['popcount'][i],
                    'factor1_msb_index': prime_features['msb_index'][i],
                    
                    # Factor 2 features
                    'factor2_dec': prime_features['decimals'][j],
                    'factor2_bits': prime_features['bits'][j].tolist(),
                    'factor2_is_prime': prime_features['is_prime'][j],
                    'factor2_is_odd': prime_features['is_odd'][j],
                    'factor2_popcount': prime_features['popcount'][j],
                    'factor2_msb_index': prime_features['msb_index'][j],
                    
                    # Product features
                    'product_dec': product_features['decimals'][start_idx + idx],
                    'product_bits': product_features['bits'][start_idx + idx].tolist(),
                    'product_popcount': product_features['popcount'][start_idx + idx],
                    'product_bit_length': product_features['bit_length'][start_idx + idx],
                    'product_trailing_zeros': product_features['trailing_zeros'][start_idx + idx],
                    
                    # Pair-level features
                    'pair_is_same': i == j,
                    'pair_coprime': i != j,
                    'product_is_square': i == j,
                }
                
                batch_records.append(record)
            
            # Yield batch
            for record in batch_records:
                yield record
                pbar.update(1)
        
        pbar.close()
    
    # Create dataset from generator
    print("Creating optimized dataset...")
    dataset = Dataset.from_generator(optimized_data_generator, features=features)
    
    # Save with maximum compression
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"prime_products_{max_bits}bit_optimized.parquet")
    
    print(f"Saving to {output_path}...")
    dataset.to_parquet(output_path, compression="zstd", compression_level=6)
    
    # Show results
    file_size_mb = os.path.getsize(output_path) / (1024**2)
    print(f"✅ Dataset saved successfully!")
    print(f"Rows: {len(dataset):,}")
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"Compression ratio: {(total_pairs * 150 / (1024**2)) / file_size_mb:.1f}:1")


def create_dataset_features_optimized(max_bits: int) -> Features:
    """Create optimized Arrow schema."""
    factor_bin_len = max_bits
    product_bin_len = max_bits * 2
    
    return Features({
        # Factor 1
        'factor1_dec': Value('string'),
        'factor1_bits': Sequence(Value('uint8'), length=factor_bin_len),
        'factor1_is_prime': Value('bool'),
        'factor1_is_odd': Value('bool'),
        'factor1_popcount': Value('uint8'),
        'factor1_msb_index': Value('uint8'),
        
        # Factor 2
        'factor2_dec': Value('string'),
        'factor2_bits': Sequence(Value('uint8'), length=factor_bin_len),
        'factor2_is_prime': Value('bool'),
        'factor2_is_odd': Value('bool'),
        'factor2_popcount': Value('uint8'),
        'factor2_msb_index': Value('uint8'),
        
        # Product
        'product_dec': Value('string'),
        'product_bits': Sequence(Value('uint8'), length=product_bin_len),
        'product_popcount': Value('uint8'),
        'product_bit_length': Value('uint8'),
        'product_trailing_zeros': Value('uint8'),
        
        # Pair-level
        'pair_is_same': Value('bool'),
        'pair_coprime': Value('bool'),
        'product_is_square': Value('bool')
    })


def main():
    """Main entry point for optimized generator."""
    parser = argparse.ArgumentParser(description="OPTIMIZED Synthetic Math Dataset Generator")
    parser.add_argument("max_bits", type=int, help="Maximum number of bits")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=1000000, help="Processing batch size")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    
    args = parser.parse_args()
    
    if args.max_bits < 1 or args.max_bits > 32:
        print("Error: max_bits must be between 1 and 32")
        sys.exit(1)
    
    use_gpu = not args.no_gpu
    
    try:
        generate_dataset_optimized(
            max_bits=args.max_bits,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            use_gpu=use_gpu
        )
    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
