#!/usr/bin/env python3
"""
Synthetic Math Dataset Generator

Generates a dataset of all unique prime number pairs and their products
for mathematical machine learning problems.
"""

import argparse
import multiprocessing as mp
import os
import sys
from typing import List, Dict, Any, Iterator, Tuple
import numpy as np
from datasets import Dataset, Features, Value, Sequence
from tqdm import tqdm
import pyarrow as pa
import math


def sieve_of_eratosthenes(limit: int) -> List[int]:
    """Generate all prime numbers up to limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    
    # Initialize boolean array "prime[0..limit]" and set all entries as true
    prime = [True] * (limit + 1)
    prime[0] = prime[1] = False  # 0 and 1 are not prime
    
    p = 2
    while p * p <= limit:
        if prime[p]:
            # Update all multiples of p
            for i in range(p * p, limit + 1, p):
                prime[i] = False
        p += 1
    
    # Collect all prime numbers
    return [i for i in range(2, limit + 1) if prime[i]]


def int_to_padded_bits(num: int, bit_length: int) -> List[int]:
    """Convert integer to padded binary array of 0s and 1s."""
    binary_str = format(num, f'0{bit_length}b')
    return [int(bit) for bit in binary_str]


def calculate_padding_lengths(max_bits: int) -> Tuple[int, int, int, int]:
    """Calculate padding lengths for decimal and binary representations."""
    max_factor = 2**max_bits - 1
    max_product = max_factor * max_factor
    
    # Decimal padding lengths
    factor_dec_len = len(str(max_factor))
    product_dec_len = len(str(max_product))
    
    # Binary padding lengths (already known but calculated for consistency)
    factor_bin_len = max_bits
    product_bin_len = max_bits * 2
    
    return factor_dec_len, product_dec_len, factor_bin_len, product_bin_len


def create_prime_features(prime: int, max_bits: int, factor_dec_len: int, factor_bin_len: int) -> Dict[str, Any]:
    """Create feature dictionary for a prime number."""
    binary_bits = int_to_padded_bits(prime, factor_bin_len)
    
    return {
        'dec': f'{prime:0{factor_dec_len}d}',  # Zero-padded decimal string
        'bits': binary_bits,  # List of 0s and 1s
        'is_prime': True,  # Always true for primes
        'is_odd': prime % 2 == 1,
        'popcount': bin(prime).count('1'),
        'msb_index': prime.bit_length() - 1 if prime > 0 else 0
    }


def create_product_features(product: int, product_dec_len: int, product_bin_len: int) -> Dict[str, Any]:
    """Create feature dictionary for a product."""
    binary_bits = int_to_padded_bits(product, product_bin_len)
    
    return {
        'dec': f'{product:0{product_dec_len}d}',  # Zero-padded decimal string
        'bits': binary_bits,  # List of 0s and 1s
        'popcount': bin(product).count('1'),
        'bit_length': product.bit_length(),
        'trailing_zeros': (product & -product).bit_length() - 1 if product > 0 else 0
    }


def generate_pair_data(prime1: int, prime2: int, max_bits: int, 
                      factor_dec_len: int, product_dec_len: int,
                      factor_bin_len: int, product_bin_len: int) -> Dict[str, Any]:
    """Generate a complete data record for a prime pair."""
    # Ensure ordering: prime1 <= prime2 for unique unordered pairs
    if prime1 > prime2:
        prime1, prime2 = prime2, prime1
    
    # Calculate product
    product = prime1 * prime2
    
    # Generate features for both factors
    factor1_features = create_prime_features(prime1, max_bits, factor_dec_len, factor_bin_len)
    factor2_features = create_prime_features(prime2, max_bits, factor_dec_len, factor_bin_len)
    
    # Generate product features
    product_features = create_product_features(product, product_dec_len, product_bin_len)
    
    # Create the complete record
    record = {}
    
    # Factor 1 features
    for key, value in factor1_features.items():
        record[f'factor1_{key}'] = value
    
    # Factor 2 features
    for key, value in factor2_features.items():
        record[f'factor2_{key}'] = value
    
    # Product features
    for key, value in product_features.items():
        record[f'product_{key}'] = value
    
    # Pair-level features
    record['pair_is_same'] = prime1 == prime2
    record['pair_coprime'] = prime1 != prime2  # Different primes are always coprime
    record['product_is_square'] = prime1 == prime2
    
    return record


def create_dataset_features(max_bits: int) -> Features:
    """Create the Arrow/Parquet schema for the dataset."""
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


def generate_batch_worker(args):
    """Worker function for multiprocessing batch generation."""
    primes, start_idx, end_idx, max_bits, padding_lengths, batch_size = args
    
    factor_dec_len, product_dec_len, factor_bin_len, product_bin_len = padding_lengths
    batch = []
    
    for i in range(start_idx, min(end_idx, len(primes))):
        for j in range(i, len(primes)):  # j >= i ensures unique unordered pairs
            prime1, prime2 = primes[i], primes[j]
            
            record = generate_pair_data(
                prime1, prime2, max_bits,
                factor_dec_len, product_dec_len,
                factor_bin_len, product_bin_len
            )
            
            batch.append(record)
            
            if len(batch) >= batch_size:
                yield batch
                batch = []
    
    # Yield remaining records
    if batch:
        yield batch


def calculate_total_pairs(num_primes: int) -> int:
    """Calculate total number of unique unordered pairs."""
    return num_primes * (num_primes + 1) // 2


def generate_dataset(max_bits: int, output_dir: str = "data", 
                    num_workers: int = None, batch_size: int = 10000):
    """Generate the complete dataset."""
    print(f"Generating dataset for {max_bits}-bit prime pairs...")
    
    # Calculate maximum values and padding lengths
    max_factor = 2**max_bits - 1
    padding_lengths = calculate_padding_lengths(max_bits)
    factor_dec_len, product_dec_len, factor_bin_len, product_bin_len = padding_lengths
    
    print(f"Maximum factor: {max_factor}")
    print(f"Factor decimal padding: {factor_dec_len} digits")
    print(f"Product decimal padding: {product_dec_len} digits")
    print(f"Factor binary padding: {factor_bin_len} bits")
    print(f"Product binary padding: {product_bin_len} bits")
    
    # Generate primes
    print("Generating prime numbers...")
    primes = sieve_of_eratosthenes(max_factor)
    num_primes = len(primes)
    print(f"Found {num_primes} prime numbers up to {max_factor}")
    
    # Calculate total number of pairs
    total_pairs = calculate_total_pairs(num_primes)
    print(f"Total unique pairs to generate: {total_pairs:,}")
    
    # Estimate dataset size
    estimated_size_gb = (total_pairs * 200) / (1024**3)  # Rough estimate: 200 bytes per row
    print(f"Estimated uncompressed size: {estimated_size_gb:.2f} GB")
    
    # Set up multiprocessing
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
    print(f"Using {num_workers} worker processes")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset features schema
    features = create_dataset_features(max_bits)
    
    # Generate data in batches using a generator
    def data_generator():
        """Generator function for batch processing with progress tracking."""
        # Calculate work distribution for workers
        chunk_size = max(1, num_primes // num_workers)
        
        # Create progress bar
        pbar = tqdm(total=total_pairs, desc="Generating pairs", unit="pairs")
        
        # Process in chunks
        for start_idx in range(0, num_primes, chunk_size):
            end_idx = min(start_idx + chunk_size, num_primes)
            
            # Calculate pairs in this chunk
            chunk_pairs = 0
            for i in range(start_idx, end_idx):
                chunk_pairs += len(primes) - i
            
            # Generate batches for this chunk
            worker_args = (primes, start_idx, end_idx, max_bits, padding_lengths, batch_size)
            
            for batch in generate_batch_worker(worker_args):
                for record in batch:
                    yield record
                    pbar.update(1)
        
        pbar.close()
    
    # Create dataset from generator
    print("Creating dataset...")
    dataset = Dataset.from_generator(
        data_generator,
        features=features
    )
    
    # Save to parquet with compression
    output_path = os.path.join(output_dir, f"prime_products_{max_bits}bit.parquet")
    print(f"Saving dataset to {output_path}...")
    
    dataset.to_parquet(
        output_path,
        compression="zstd"
    )
    
    print(f"Dataset saved successfully!")
    print(f"Final dataset size: {len(dataset):,} rows")
    
    # Show file size
    if os.path.exists(output_path):
        file_size_mb = os.path.getsize(output_path) / (1024**2)
        print(f"Compressed file size: {file_size_mb:.2f} MB")
        compression_ratio = (estimated_size_gb * 1024) / file_size_mb
        print(f"Compression ratio: {compression_ratio:.1f}:1")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic mathematical dataset with prime number products"
    )
    parser.add_argument(
        "max_bits",
        type=int,
        help="Maximum number of bits for factors (e.g., 20 for 20-bit numbers)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for the dataset (default: data)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: CPU count - 1)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Batch size for processing (default: 10000)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.max_bits < 1 or args.max_bits > 32:
        print("Error: max_bits must be between 1 and 32")
        sys.exit(1)
    
    # Show warning for large datasets
    if args.max_bits >= 18:
        max_factor = 2**args.max_bits - 1
        primes_estimate = max_factor / math.log(max_factor)  # Prime number theorem approximation
        pairs_estimate = primes_estimate * (primes_estimate + 1) // 2
        
        print(f"WARNING: {args.max_bits}-bit dataset will be very large!")
        print(f"Estimated prime count: ~{int(primes_estimate):,}")
        print(f"Estimated pair count: ~{int(pairs_estimate):,}")
        print(f"Consider testing with smaller bit counts first (12-16 bits)")
        
        response = input("Continue? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)
    
    try:
        generate_dataset(
            max_bits=args.max_bits,
            output_dir=args.output_dir,
            num_workers=args.workers,
            batch_size=args.batch_size
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
