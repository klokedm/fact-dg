#!/usr/bin/env python3
"""
ULTRA FAST Synthetic Math Dataset Generator - Fixed Performance Issues

This version eliminates multiprocessing bottlenecks by having workers
write directly to separate parquet files that get merged.
"""

import argparse
import os
import sys
import math
import multiprocessing as mp
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from datasets import Dataset, Features, Value, Sequence
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

# HuggingFace Hub integration
try:
    from huggingface_hub import HfApi, upload_file
    from datasets import load_dataset
    HF_HUB_AVAILABLE = True
except ImportError:
    print("WARNING: HuggingFace Hub not available. Install with: pip install huggingface_hub")
    HF_HUB_AVAILABLE = False

# Performance optimizations
try:
    from numba import jit, prange
    import numba
    NUMBA_AVAILABLE = True
    print(f"✅ Numba {numba.__version__} available")
except ImportError:
    print("⚠️ Numba not available. Install with: pip install numba")
    NUMBA_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

def get_memory_usage() -> str:
    """Get current memory usage if psutil is available."""
    if not PSUTIL_AVAILABLE:
        return "Memory monitoring unavailable"
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024**2)
        return f"{memory_mb:.1f} MB"
    except:
        return "Memory monitoring error"

def calc_bit_length(num) -> int:
    """Calculate bit length for any integer type (Python int, numpy int32, etc)."""
    if num == 0:
        return 0
    
    # Convert to Python int to ensure bit_length() works
    if hasattr(num, 'item'):  # numpy scalar
        num = int(num.item())
    else:
        num = int(num)
    
    # Use Python's built-in bit_length for regular integers
    return num.bit_length()

# Import optimized functions from the original script
if NUMBA_AVAILABLE:
    @jit(nopython=True)
    def fast_sieve_of_eratosthenes(max_num: int) -> np.ndarray:
        """Optimized sieve using Numba JIT."""
        if max_num < 2:
            # Create a properly typed empty array for Numba
            return np.empty(0, dtype=np.int32)

        sieve = np.ones(max_num + 1, dtype=np.uint8)
        sieve[0] = sieve[1] = 0

        for i in range(2, int(math.sqrt(max_num)) + 1):
            if sieve[i]:
                for j in prange(i*i, max_num + 1, i):
                    sieve[j] = 0

        return np.where(sieve)[0].astype(np.int32)

    @jit(nopython=True)
    def compute_features_fast(numbers: np.ndarray, max_bits: int, bits_array: np.ndarray, popcount: np.ndarray, msb_index: np.ndarray):
        """Fast vectorized feature computation - modifies arrays in place."""
        n = len(numbers)

        for i in prange(n):
            num = numbers[i]

            # Convert to binary
            for bit_pos in range(max_bits):
                if num & (1 << bit_pos):
                    bits_array[i, max_bits - 1 - bit_pos] = 1
                    popcount[i] += 1

            # MSB index - Optimized Numba-compatible bit length calculation
            if num > 0:
                # Fast bit length using bit manipulation
                temp_num = num
                bit_len = 0
                # Optimized: use powers of 2 to find bit length faster
                if temp_num >= (1 << 16):
                    bit_len += 16
                    temp_num >>= 16
                if temp_num >= (1 << 8):
                    bit_len += 8
                    temp_num >>= 8
                if temp_num >= (1 << 4):
                    bit_len += 4
                    temp_num >>= 4
                if temp_num >= (1 << 2):
                    bit_len += 2
                    temp_num >>= 2
                if temp_num >= (1 << 1):
                    bit_len += 1
                    temp_num >>= 1
                if temp_num >= 1:
                    bit_len += 1
                msb_index[i] = bit_len - 1
            else:
                msb_index[i] = 0
else:
    # Fallback non-JIT versions
    def fast_sieve_of_eratosthenes(max_num: int) -> np.ndarray:
        if max_num < 2:
            return np.array([], dtype=np.int32)

        sieve = np.ones(max_num + 1, dtype=bool)
        sieve[0] = sieve[1] = False

        for i in range(2, int(math.sqrt(max_num)) + 1):
            if sieve[i]:
                sieve[i*i::i] = False

        return np.where(sieve)[0].astype(np.int32)

    def compute_features_fast(numbers: np.ndarray, max_bits: int, bits_array: np.ndarray, popcount: np.ndarray, msb_index: np.ndarray):
        """Fallback non-JIT version - modifies arrays in place."""
        n = len(numbers)

        for i, num in enumerate(numbers):
            bit_str = format(num, f'0{max_bits}b')
            bits_array[i] = [int(b) for b in bit_str]

        popcount[:] = np.sum(bits_array, axis=1, dtype=np.uint8)
        msb_index[:] = np.array([calc_bit_length(num) - 1 if num > 0 else 0
                                for num in numbers], dtype=np.uint8)

def create_parquet_schema(max_bits: int) -> pa.Schema:
    """Create Arrow schema for parquet files."""
    factor_bin_len = max_bits
    product_bin_len = max_bits * 2
    
    return pa.schema([
        # Factor 1
        pa.field('factor1_dec', pa.string()),
        pa.field('factor1_bits', pa.list_(pa.uint8(), list_size=factor_bin_len)),
        pa.field('factor1_is_prime', pa.bool_()),
        pa.field('factor1_is_odd', pa.bool_()),
        pa.field('factor1_popcount', pa.uint8()),
        pa.field('factor1_msb_index', pa.uint8()),
        
        # Factor 2
        pa.field('factor2_dec', pa.string()),
        pa.field('factor2_bits', pa.list_(pa.uint8(), list_size=factor_bin_len)),
        pa.field('factor2_is_prime', pa.bool_()),
        pa.field('factor2_is_odd', pa.bool_()),
        pa.field('factor2_popcount', pa.uint8()),
        pa.field('factor2_msb_index', pa.uint8()),
        
        # Product
        pa.field('product_dec', pa.string()),
        pa.field('product_bits', pa.list_(pa.uint8(), list_size=product_bin_len)),
        pa.field('product_popcount', pa.uint8()),
        pa.field('product_bit_length', pa.uint8()),
        pa.field('product_trailing_zeros', pa.uint8()),
        
        # Additional fields
        pa.field('a_dec', pa.string()),
        pa.field('a_bits', pa.list_(pa.uint8(), list_size=product_bin_len)),
        pa.field('b_dec', pa.string()),
        pa.field('b_bits', pa.list_(pa.uint8(), list_size=product_bin_len)),
        pa.field('pair_is_same', pa.bool_()),
        pa.field('pair_coprime', pa.bool_()),
        pa.field('product_is_square', pa.bool())
    ])

def generate_chunk_fast(args) -> str:
    """
    FAST worker function - writes directly to parquet file.
    Returns the path to the generated chunk file.
    """
    chunk_pairs, primes, prime_features, max_bits, chunk_id, temp_dir = args
    
    if not chunk_pairs:
        return None
    
    # Generate records for this chunk
    records = []
    
    for i, j in chunk_pairs:
        # Get prime values
        p1, p2 = primes[i], primes[j]
        product = p1 * p2
        
        # Compute product features
        product_bits = format(product, f'0{max_bits*2}b')
        product_bit_array = [int(b) for b in product_bits]
        product_popcount = sum(product_bit_array)
        product_bit_length = calc_bit_length(product)
        product_trailing_zeros = calc_bit_length(product & -product) - 1 if product > 0 else 0
        
        # a and b for p² = b² - a² formula  
        a = abs(p2 - p1) // 2
        b = (p1 + p2) // 2
        
        a_bits = format(a, f'0{max_bits*2}b')
        b_bits = format(b, f'0{max_bits*2}b')
        
        record = {
            # Factor 1 features
            'factor1_dec': str(p1),
            'factor1_bits': prime_features['bits'][i].tolist(),
            'factor1_is_prime': True,
            'factor1_is_odd': bool(p1 % 2),
            'factor1_popcount': int(prime_features['popcount'][i]),
            'factor1_msb_index': int(prime_features['msb_index'][i]),
            
            # Factor 2 features
            'factor2_dec': str(p2),
            'factor2_bits': prime_features['bits'][j].tolist(),
            'factor2_is_prime': True,
            'factor2_is_odd': bool(p2 % 2),
            'factor2_popcount': int(prime_features['popcount'][j]),
            'factor2_msb_index': int(prime_features['msb_index'][j]),
            
            # Product features
            'product_dec': str(product),
            'product_bits': product_bit_array,
            'product_popcount': product_popcount,
            'product_bit_length': product_bit_length,
            'product_trailing_zeros': product_trailing_zeros,
            
            # a and b values
            'a_dec': str(a),
            'a_bits': [int(bit) for bit in a_bits],
            'b_dec': str(b), 
            'b_bits': [int(bit) for bit in b_bits],
            
            # Pair-level features
            'pair_is_same': i == j,
            'pair_coprime': math.gcd(p1, p2) == 1,
            'product_is_square': i == j
        }
        
        records.append(record)
    
    # Write chunk directly to parquet file
    chunk_file = os.path.join(temp_dir, f"chunk_{chunk_id:04d}.parquet")
    
    # Create table and write
    table = pa.Table.from_pylist(records, schema=create_parquet_schema(max_bits))
    pq.write_table(table, chunk_file, compression='zstd', compression_level=22)
    
    return chunk_file

def merge_parquet_files(chunk_files: List[str], output_path: str, compression_level: int = 22):
    """Efficiently merge parquet chunk files."""
    print("Merging parquet chunks...")
    
    # Filter out None values (empty chunks)
    valid_chunks = [f for f in chunk_files if f is not None and os.path.exists(f)]
    
    if not valid_chunks:
        raise ValueError("No valid chunk files to merge")
    
    # Read all chunks and combine
    tables = []
    for chunk_file in tqdm(valid_chunks, desc="Reading chunks"):
        table = pq.read_table(chunk_file)
        tables.append(table)
    
    # Concatenate all tables
    print("Combining tables...")
    combined_table = pa.concat_tables(tables)
    
    # Write final file
    print(f"Writing final parquet file: {output_path}")
    pq.write_table(combined_table, output_path, compression='zstd', 
                   compression_level=compression_level)
    
    # Clean up chunk files
    for chunk_file in valid_chunks:
        try:
            os.remove(chunk_file)
        except:
            pass  # Ignore cleanup errors

def upload_to_huggingface(output_path: str, repo_name: str, max_bits: int, is_private: bool = True) -> bool:
    """Upload dataset to HuggingFace Hub."""
    if not HF_HUB_AVAILABLE:
        print("ERROR: HuggingFace Hub not available.")
        return False
    
    if not repo_name:
        print("WARNING: No repo-name provided. Skipping upload.")
        return False
        
    try:
        print(f"\nUploading dataset to HuggingFace Hub: {repo_name}")
        
        dataset = load_dataset('parquet', data_files=output_path)
        print(f"Dataset info: {len(dataset['train']):,} rows")
        
        dataset.push_to_hub(
            repo_name, 
            commit_message=f"Add {max_bits}-bit prime factorization dataset",
            private=is_private
        )
        
        visibility = "private" if is_private else "public"
        print(f"✅ SUCCESS: Dataset uploaded as {visibility} to https://huggingface.co/datasets/{repo_name}")
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to upload to HuggingFace Hub: {e}")
        return False

def generate_dataset_fast(max_bits: int, output_dir: str = "data",
                         repo_name: str = None, num_workers: int = None,
                         compression_level: int = 22, no_upload: bool = False, 
                         is_public: bool = False):
    """FAST dataset generation - no multiprocessing bottlenecks."""
    
    print(f"🚀 FAST Synthetic Math Dataset Generator")
    print(f"Generating dataset for {max_bits}-bit prime pairs...")
    print(f"Target repository: {repo_name}")
    print(f"Compression level: {compression_level}")
    if NUMBA_AVAILABLE:
        try:
            import numba
            print(f"Numba JIT: ✅ ({numba.__version__})")
        except:
            print("Numba JIT: ✅")
    else:
        print("Numba JIT: ❌")
    
    # Set up workers
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
    print(f"Using {num_workers} worker processes")
    
    # Generate primes
    max_factor = 2**max_bits - 1
    print("Generating prime numbers...")
    primes = fast_sieve_of_eratosthenes(max_factor)
    num_primes = len(primes)
    print(f"Found {num_primes:,} primes up to {max_factor}")
    
    # Calculate total pairs  
    total_pairs = num_primes * (num_primes + 1) // 2
    print(f"Total pairs to generate: {total_pairs:,}")
    
    # Compute prime features once
    print("Computing prime features...")
    n_primes = len(primes)
    bits_array = np.zeros((n_primes, max_bits), dtype=np.uint8)
    popcount_array = np.zeros(n_primes, dtype=np.uint8)
    msb_index_array = np.zeros(n_primes, dtype=np.uint8)

    compute_features_fast(primes, max_bits, bits_array, popcount_array, msb_index_array)

    prime_features = {
        'bits': bits_array,
        'popcount': popcount_array,
        'msb_index': msb_index_array
    }
    print(f"Memory usage: {get_memory_usage()}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = os.path.join(output_dir, f"temp_chunks_{max_bits}bit")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Split work into chunks
    pairs_per_chunk = max(10000, total_pairs // (num_workers * 4))
    
    print(f"Creating work chunks ({pairs_per_chunk:,} pairs per chunk)...")
    
    worker_args = []
    chunk_id = 0
    pair_idx = 0
    
    # Generate all (i,j) pairs and split into chunks  
    for i in range(num_primes):
        chunk_pairs = []
        for j in range(i, num_primes):
            chunk_pairs.append((i, j))
            pair_idx += 1
            
            # Create chunk when full
            if len(chunk_pairs) >= pairs_per_chunk:
                worker_args.append((chunk_pairs, primes, prime_features, max_bits, chunk_id, temp_dir))
                chunk_pairs = []
                chunk_id += 1
        
        # Add remaining pairs in chunk
        if chunk_pairs:
            worker_args.append((chunk_pairs, primes, prime_features, max_bits, chunk_id, temp_dir))
            chunk_id += 1
    
    print(f"Created {len(worker_args)} chunks")
    
    # Process chunks in parallel
    print("Processing chunks in parallel...")
    with mp.Pool(num_workers) as pool:
        chunk_files = list(tqdm(
            pool.imap(generate_chunk_fast, worker_args),
            total=len(worker_args),
            desc="Processing chunks"
        ))
    
    # Merge all chunk files
    output_path = os.path.join(output_dir, f"prime_products_{max_bits}bit_fast.parquet")
    merge_parquet_files(chunk_files, output_path, compression_level)
    
    # Clean up temp directory
    try:
        os.rmdir(temp_dir)
    except:
        pass
    
    # Show results
    file_size_mb = os.path.getsize(output_path) / (1024**2)
    print(f"\n🎉 SUCCESS!")
    print(f"Dataset: {output_path}")
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"Total records: {total_pairs:,}")
    print(f"Final memory: {get_memory_usage()}")
    
    # Upload to HuggingFace if requested
    if repo_name and not no_upload:
        upload_success = upload_to_huggingface(output_path, repo_name, max_bits, is_private=not is_public)
        if upload_success:
            print("🚀 Upload completed successfully!")
        else:
            print("⚠️  Upload failed.")
    elif no_upload:
        print("📁 Upload skipped (--no-upload)")
    else:
        print("📁 No repo-name provided")

def main():
    parser = argparse.ArgumentParser(description="ULTRA FAST Synthetic Math Dataset Generator")
    parser.add_argument("max_bits", type=int, help="Maximum number of bits")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory")
    parser.add_argument("--repo-name", type=str, help="HuggingFace repository name")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes")
    parser.add_argument("--compression-level", type=int, default=22, help="Compression level (1-22)")
    parser.add_argument("--no-upload", action="store_true", help="Skip HuggingFace upload")
    parser.add_argument("--public", action="store_true", help="Make dataset public (default: private)")
    parser.add_argument("--no-numba", action="store_true", help="Disable Numba JIT compilation (use fallback)")
    parser.add_argument("--test", action="store_true", help="Run a quick test with small dataset (8-bit)")

    args = parser.parse_args()

    # Handle test mode
    if args.test:
        print("🧪 TEST MODE: Generating small 8-bit dataset...")
        args.max_bits = 8
        args.workers = 2
        args.no_upload = True
    
    if args.max_bits < 1 or args.max_bits > 32:
        print("Error: max_bits must be between 1 and 32")
        sys.exit(1)
        
    # Show dataset size warning
    if args.max_bits >= 16:
        max_factor = 2**args.max_bits - 1
        primes_estimate = max_factor / math.log(max_factor)
        pairs_estimate = primes_estimate * (primes_estimate + 1) // 2
        print(f"📊 Estimated {int(pairs_estimate):,} pairs for {args.max_bits}-bit dataset")
    
    # Handle --no-numba option
    if args.no_numba:
        NUMBA_AVAILABLE = False
        print("⚠️ Numba JIT disabled via --no-numba")

    try:
        generate_dataset_fast(
            max_bits=args.max_bits,
            output_dir=args.output_dir,
            repo_name=args.repo_name,
            num_workers=args.workers,
            compression_level=args.compression_level,
            no_upload=args.no_upload,
            is_public=args.public
        )
    except KeyboardInterrupt:
        print("\n❌ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
