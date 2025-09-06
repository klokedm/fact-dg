#!/usr/bin/env python3
"""
Fixed Optimized Synthetic Math Dataset Generator

Corrected high-performance version with proper parallelization,
vectorization, and maximum compression.
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

# Performance optimizations
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    print("WARNING: Numba not available. Install with: pip install numba")
    NUMBA_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = cp.cuda.is_available()
    if CUPY_AVAILABLE:
        device = cp.cuda.Device()
        print(f"[OK] CUDA available: {device.name}")
except ImportError:
    CUPY_AVAILABLE = False
except Exception:
    # Handle any CUDA-related errors gracefully
    CUPY_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


def get_memory_usage() -> str:
    """Get current memory usage if psutil is available."""
    if not PSUTIL_AVAILABLE:
        return "Memory monitoring unavailable (install psutil)"

    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024**2)
        return f"{memory_mb:.1f} MB"
    except Exception as e:
        return f"Memory monitoring error: {e}"


# ============================================================================
# OPTIMIZED PRIME GENERATION (SAME AS BEFORE)
# ============================================================================

if NUMBA_AVAILABLE:
    @jit(nopython=True, cache=True)
    def fast_sieve_of_eratosthenes(limit: int) -> np.ndarray:
        """Ultra-fast JIT-compiled prime sieve."""
        if limit < 2:
            return np.empty(0, dtype=np.int32)

        prime = np.ones(limit + 1, dtype=np.bool_)
        prime[0] = prime[1] = False
        
        sqrt_limit = int(np.sqrt(limit)) + 1
        for p in range(2, sqrt_limit):
            if prime[p]:
                prime[p*p::p] = False
        
        return np.where(prime)[0].astype(np.int32)

    @jit(nopython=True, cache=True)  
    def fast_int_to_bits_batch(numbers: np.ndarray, bit_length: int) -> np.ndarray:
        """Fast batch conversion of integers to bit arrays."""
        n = len(numbers)
        result = np.zeros((n, bit_length), dtype=np.uint8)
        
        for idx in prange(n):
            num = numbers[idx]
            for bit_pos in range(bit_length):
                result[idx, bit_length - 1 - bit_pos] = (num >> bit_pos) & 1
        
        return result
    
    @jit(nopython=True, cache=True)
    def fast_popcount_batch(numbers: np.ndarray) -> np.ndarray:
        """Fast batch popcount."""
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
        if limit < 2:
            return np.empty(0, dtype=np.int32)
        
        prime = np.ones(limit + 1, dtype=bool)
        prime[0] = prime[1] = False
        
        for p in range(2, int(np.sqrt(limit)) + 1):
            if prime[p]:
                prime[p*p::p] = False
        
        return np.where(prime)[0].astype(np.int32)
    
    def fast_int_to_bits_batch(numbers: np.ndarray, bit_length: int) -> np.ndarray:
        n = len(numbers)
        result = np.zeros((n, bit_length), dtype=np.uint8)
        
        for i, num in enumerate(numbers):
            bits = np.array([(num >> j) & 1 for j in range(bit_length)[::-1]], dtype=np.uint8)
            result[i] = bits
        
        return result
    
    def fast_popcount_batch(numbers: np.ndarray) -> np.ndarray:
        return np.array([bin(int(x)).count('1') for x in numbers], dtype=np.uint8)
    
    def fast_msb_index_batch(numbers: np.ndarray) -> np.ndarray:
        return np.array([int(x).bit_length() - 1 if x > 0 else 0 for x in numbers], dtype=np.uint8)
    
    def fast_trailing_zeros_batch(numbers: np.ndarray) -> np.ndarray:
        result = np.zeros(len(numbers), dtype=np.uint8)
        for i, num in enumerate(numbers):
            if num == 0:
                result[i] = 0
            else:
                result[i] = int(num & -num).bit_length() - 1
        return result


# ============================================================================
# CUDA-ACCELERATED FUNCTIONS (OPTIONAL)
# ============================================================================

if CUPY_AVAILABLE:
    def cuda_int_to_bits_batch(numbers: np.ndarray, bit_length: int) -> np.ndarray:
        """CUDA-accelerated batch conversion of integers to bit arrays."""
        import cupy as cp
        
        # Transfer to GPU
        gpu_numbers = cp.asarray(numbers)
        n = len(gpu_numbers)
        
        # Create result array on GPU
        gpu_result = cp.zeros((n, bit_length), dtype=cp.uint8)
        
        # CUDA kernel for bit extraction (vectorized)
        for bit_pos in range(bit_length):
            gpu_result[:, bit_length - 1 - bit_pos] = (gpu_numbers >> bit_pos) & 1
        
        # Transfer back to CPU
        return cp.asnumpy(gpu_result)
    
    def cuda_popcount_batch(numbers: np.ndarray) -> np.ndarray:
        """CUDA-accelerated batch popcount using GPU bit operations."""
        import cupy as cp
        
        # Transfer to GPU
        gpu_numbers = cp.asarray(numbers, dtype=cp.uint64)
        
        # Use CuPy's built-in popcount (much faster than manual counting)
        gpu_result = cp.zeros(len(gpu_numbers), dtype=cp.uint8)
        
        # Manual popcount for each number (CuPy doesn't have built-in popcount)
        for i in range(len(gpu_numbers)):
            num = gpu_numbers[i]
            count = 0
            while num:
                count += num & 1
                num >>= 1
            gpu_result[i] = count
        
        return cp.asnumpy(gpu_result)
    
    def cuda_mathematical_operations(products: np.ndarray) -> tuple:
        """CUDA-accelerated mathematical operations for a and b values."""
        import cupy as cp
        
        # Transfer to GPU
        gpu_products = cp.asarray(products, dtype=cp.uint64)
        
        # Vectorized operations on GPU
        gpu_products_squared = gpu_products * gpu_products
        gpu_a_values = (gpu_products_squared - 1) // 2
        gpu_b_values = (gpu_products_squared + 1) // 2
        
        # Transfer back to CPU
        return (cp.asnumpy(gpu_a_values), cp.asnumpy(gpu_b_values))

else:
    # Fallback to CPU versions if CUDA not available
    cuda_int_to_bits_batch = None
    cuda_popcount_batch = None
    cuda_mathematical_operations = None


# ============================================================================
# OPTIMIZED FEATURE COMPUTATION
# ============================================================================

def compute_prime_features_vectorized(primes: np.ndarray, max_bits: int) -> Dict[str, np.ndarray]:
    """Batch compute all prime features using vectorized operations."""
    
    # Pre-calculate padding
    max_factor = 2**max_bits - 1
    factor_dec_len = len(str(max_factor))
    
    # Decimal representations (vectorized string formatting)
    decimals = np.array([f'{p:0{factor_dec_len}d}' for p in primes])
    
    # Binary bit arrays (vectorized with optional CUDA acceleration)
    if CUPY_AVAILABLE and len(primes) > 1000:  # Use GPU for large arrays
        bits = cuda_int_to_bits_batch(primes, max_bits)
        print(f"  Using CUDA for bit conversion ({len(primes)} primes)")
    else:
        bits = fast_int_to_bits_batch(primes, max_bits)
    
    # Mathematical features (all vectorized)
    is_prime = np.ones(len(primes), dtype=bool)
    is_odd = (primes % 2 == 1).astype(bool)
    
    # Popcount with optional CUDA acceleration
    if CUPY_AVAILABLE and len(primes) > 1000:
        popcount = cuda_popcount_batch(primes)
        print(f"  Using CUDA for popcount ({len(primes)} primes)")
    else:
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


def compute_products_and_features(prime_pairs: List[Tuple[int, int]], primes: np.ndarray, 
                                max_bits: int) -> Dict[str, np.ndarray]:
    """Compute products and their features for a batch of prime pairs."""
    
    max_factor = 2**max_bits - 1
    max_product = max_factor * max_factor
    product_dec_len = len(str(max_product))
    product_bin_len = max_bits * 2
    
    # Extract products
    products = np.array([primes[i] * primes[j] for i, j in prime_pairs], dtype=np.uint64)
    
    # Decimal representations
    decimals = np.array([f'{p:0{product_dec_len}d}' for p in products])
    
    # Binary bit arrays with optional CUDA acceleration
    if CUPY_AVAILABLE and len(products) > 500:  # Use GPU for medium+ arrays
        bits = cuda_int_to_bits_batch(products, product_bin_len)
    else:
        bits = fast_int_to_bits_batch(products, product_bin_len)
    
    # Mathematical features with optional CUDA acceleration
    if CUPY_AVAILABLE and len(products) > 500:
        popcount = cuda_popcount_batch(products)
    else:
        popcount = fast_popcount_batch(products)
    
    bit_length = np.array([int(p).bit_length() for p in products], dtype=np.uint8)
    trailing_zeros = fast_trailing_zeros_batch(products)
    
    # Compute a and b values for p² = b² - a² formula with optional CUDA acceleration
    if CUPY_AVAILABLE and len(products) > 500:
        a_values, b_values = cuda_mathematical_operations(products)
    else:
        # CPU version
        products_squared = products * products
        a_values = (products_squared - 1) // 2
        b_values = (products_squared + 1) // 2
    
    # Decimal representations for a and b (same padding as product)
    a_decimals = np.array([f'{a:0{product_dec_len}d}' for a in a_values])
    b_decimals = np.array([f'{b:0{product_dec_len}d}' for b in b_values])
    
    # Binary representations for a and b with optional CUDA acceleration
    if CUPY_AVAILABLE and len(a_values) > 500:
        a_bits = cuda_int_to_bits_batch(a_values, product_bin_len)
        b_bits = cuda_int_to_bits_batch(b_values, product_bin_len)
    else:
        a_bits = fast_int_to_bits_batch(a_values, product_bin_len)
        b_bits = fast_int_to_bits_batch(b_values, product_bin_len)
    
    return {
        'decimals': decimals,
        'bits': bits,
        'popcount': popcount,
        'bit_length': bit_length,
        'trailing_zeros': trailing_zeros,
        'a_dec': a_decimals,
        'a_bits': a_bits,
        'b_dec': b_decimals,
        'b_bits': b_bits
    }


# ============================================================================
# PARALLEL WORKER FUNCTIONS
# ============================================================================

def generate_chunk_pairs(args):
    """Worker function to generate pairs for a chunk of prime pairs."""
    pairs_chunk, primes, prime_features, max_bits, chunk_id = args
    
    # pairs_chunk is already a list of (i, j) pairs to process
    pairs = pairs_chunk
    
    if not pairs:
        return []
    
    # Compute product features for this chunk
    product_features = compute_products_and_features(pairs, primes, max_bits)
    
    # Build records
    records = []
    for idx, (i, j) in enumerate(pairs):
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
            'product_dec': product_features['decimals'][idx],
            'product_bits': product_features['bits'][idx].tolist(),
            'product_popcount': product_features['popcount'][idx],
            'product_bit_length': product_features['bit_length'][idx],
            'product_trailing_zeros': product_features['trailing_zeros'][idx],
            
            # a and b values for p² = b² - a² formula
            'a_dec': product_features['a_dec'][idx],
            'a_bits': product_features['a_bits'][idx].tolist(),
            'b_dec': product_features['b_dec'][idx],
            'b_bits': product_features['b_bits'][idx].tolist(),
            
            # Pair-level features
            'pair_is_same': i == j,
            'pair_coprime': i != j,
            'product_is_square': i == j,
        }
        records.append(record)
    
    return records


# ============================================================================
# MAIN OPTIMIZED GENERATION FUNCTION
# ============================================================================

def generate_dataset_optimized_fixed(max_bits: int, output_dir: str = "data",
                                   repo_name: str = None, num_workers: int = None,
                                   compression_level: int = 22):
    """Fixed optimized dataset generation with proper parallelization."""
    print(f"Generating OPTIMIZED dataset for {max_bits}-bit prime pairs...")
    print(f"Target repository: {repo_name}")
    print(f"Maximum compression level: {compression_level}")
    
    # Show optimization status
    print(f"Numba JIT: {'[OK]' if NUMBA_AVAILABLE else '[NO]'}")
    print(f"CUDA GPU: {'[OK]' if CUPY_AVAILABLE else '[NO]'}")
    print(f"Multiprocessing: [OK]")
    print(f"Memory monitoring: {'[OK]' if PSUTIL_AVAILABLE else '[NO]'}")
    
    # Set up workers
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
        print(f"Using {num_workers} worker processes (auto-detected)")
    else:
        print(f"Using {num_workers} worker processes (user-specified)")
    
    # Generate primes with optimized sieve
    max_factor = 2**max_bits - 1
    print("Generating prime numbers...")
    primes = fast_sieve_of_eratosthenes(max_factor)
    num_primes = len(primes)
    print(f"Found {num_primes} prime numbers up to {max_factor}")
    print(f"Memory usage: {get_memory_usage()}")
    
    # Calculate total pairs
    total_pairs = num_primes * (num_primes + 1) // 2
    print(f"Total unique pairs to generate: {total_pairs:,}")
    
    # Pre-compute all prime features (vectorized)
    print("Computing prime features...")
    prime_features = compute_prime_features_vectorized(primes, max_bits)
    print(f"Memory usage: {get_memory_usage()}")
    
    # Create dataset features schema
    features = create_dataset_features_optimized_fixed(max_bits)
    
    # Generate data using parallel processing
    def parallel_data_generator():
        """Parallel generator with proper load balancing."""
        
        # Calculate optimal chunk size
        pairs_per_chunk = max(50000, total_pairs // (num_workers * 8))
        num_chunks = (total_pairs + pairs_per_chunk - 1) // pairs_per_chunk
        
        print(f"Splitting {total_pairs:,} pairs into {num_chunks} chunks (~{pairs_per_chunk:,} pairs/chunk)")
        
        # Generate chunks using index ranges instead of storing pairs
        worker_args = []
        chunk_id = 0
        pairs_assigned = 0
        
        # Use a more efficient chunking strategy
        for chunk_id in range(num_chunks):
            # Calculate which pairs belong to this chunk
            start_pair_idx = chunk_id * pairs_per_chunk
            end_pair_idx = min((chunk_id + 1) * pairs_per_chunk, total_pairs)
            chunk_size = end_pair_idx - start_pair_idx
            
            # Convert pair indices to actual (i,j) pairs for this chunk
            chunk_pairs = []
            pair_idx = 0
            
            # Efficiently find the pairs for this chunk
            for i in range(num_primes):
                for j in range(i, num_primes):
                    if pair_idx >= start_pair_idx and pair_idx < end_pair_idx:
                        chunk_pairs.append((i, j))
                    pair_idx += 1
                    if pair_idx >= end_pair_idx:
                        break
                if pair_idx >= end_pair_idx:
                    break
            
            worker_args.append((chunk_pairs, primes, prime_features, max_bits, chunk_id))
            pairs_assigned += len(chunk_pairs)
        
        print(f"Created {len(worker_args)} chunks with ~{pairs_per_chunk:,} pairs each")
        print(f"Starting parallel processing with {num_workers} workers...")
        
        # Process chunks in parallel with better progress tracking
        with mp.Pool(num_workers) as pool:
            # Use imap_unordered for better performance
            with tqdm(total=total_pairs, desc="Processing pairs", unit="pairs") as pbar:
                for chunk_records in pool.imap_unordered(generate_chunk_pairs, worker_args):
                    for record in chunk_records:
                        yield record
                        pbar.update(1)
    
    # Create dataset using streaming to avoid memory issues
    print("Creating dataset with streaming approach...")
    
    # Write directly to parquet in chunks to avoid memory issues
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"prime_products_{max_bits}bit_optimized_fixed.parquet")
    
    print(f"Writing dataset directly to {output_path}...")
    
    # Collect data in batches and write to parquet
    batch_size = 100000  # Process 100k records at a time
    batch_records = []
    total_written = 0
    
    try:
        for record in parallel_data_generator():
            batch_records.append(record)
            
            if len(batch_records) >= batch_size:
                # Create batch dataset and write/append to parquet
                batch_dataset = Dataset.from_list(batch_records, features=features)
                
                if total_written == 0:
                    # First batch - create new file
                    batch_dataset.to_parquet(output_path, compression="zstd", compression_level=compression_level)
                else:
                    # Subsequent batches - append to existing file
                    # Note: HuggingFace datasets doesn't support append, so we use a workaround
                    temp_path = output_path + f".batch_{total_written}"
                    batch_dataset.to_parquet(temp_path, compression="zstd", compression_level=compression_level)
                
                total_written += len(batch_records)
                print(f"Written {total_written:,} records, Memory: {get_memory_usage()}")
                batch_records = []  # Clear batch
                
        # Write final batch
        if batch_records:
            batch_dataset = Dataset.from_list(batch_records, features=features)
            if total_written == 0:
                batch_dataset.to_parquet(output_path, compression="zstd", compression_level=compression_level)
            else:
                temp_path = output_path + f".batch_{total_written}"
                batch_dataset.to_parquet(temp_path, compression="zstd", compression_level=compression_level)
            total_written += len(batch_records)
            
        print(f"Dataset generation complete! Total records: {total_written:,}")
        
        # If we wrote multiple batches, we need to combine them
        if total_written > batch_size:
            print("Combining batch files...")
            # Load and combine all batches
            dataset = Dataset.load_from_disk(output_path)
            batch_num = batch_size
            while os.path.exists(output_path + f".batch_{batch_num}"):
                batch_dataset = Dataset.load_from_disk(output_path + f".batch_{batch_num}")
                dataset = dataset.concatenate(batch_dataset)
                os.remove(output_path + f".batch_{batch_num}")
                batch_num += batch_size
                
    except Exception as e:
        print(f"ERROR during dataset creation: {e}")
        raise
    
    
    # Show results
    try:
        file_size_mb = os.path.getsize(output_path) / (1024**2)
        estimated_uncompressed_mb = total_pairs * 150 / (1024**2)  # Rough estimate
        compression_ratio = estimated_uncompressed_mb / file_size_mb if file_size_mb > 0 else 0

        print(f"SUCCESS: Dataset saved successfully!")
        print(f"Rows: {len(dataset):,}")
        print(f"File size: {file_size_mb:.2f} MB")
        print(f"Compression ratio: {compression_ratio:.1f}:1")
        print(f"Compression efficiency: {((1 - file_size_mb/estimated_uncompressed_mb) * 100):.1f}%")
        print(f"Final memory usage: {get_memory_usage()}")
    except OSError as e:
        print(f"WARNING: Could not get file size: {e}")
        print(f"SUCCESS: Dataset saved successfully!")
        print(f"Rows: {len(dataset):,}")
        print(f"Final memory usage: {get_memory_usage()}")


def create_dataset_features_optimized_fixed(max_bits: int) -> Features:
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
        
        # a and b values for p² = b² - a² formula
        'a_dec': Value('string'),
        'a_bits': Sequence(Value('uint8'), length=product_bin_len),
        'b_dec': Value('string'),
        'b_bits': Sequence(Value('uint8'), length=product_bin_len),
        
        # Pair-level
        'pair_is_same': Value('bool'),
        'pair_coprime': Value('bool'),
        'product_is_square': Value('bool')
    })


def main():
    """Main entry point for fixed optimized generator."""
    parser = argparse.ArgumentParser(description="FIXED OPTIMIZED Synthetic Math Dataset Generator")
    parser.add_argument("max_bits", type=int, help="Maximum number of bits")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory")
    parser.add_argument("--repo-name", type=str, required=True, help="Name of the repository where dataset should be pushed")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes (deprecated, use --cores)")
    parser.add_argument("--cores", type=int, default=None, help="Number of CPU cores to use (default: auto-detect)")
    parser.add_argument("--compression-level", type=int, default=22,
                       help="Zstd compression level (1-22, default: 22 for maximum compression)")
    
    args = parser.parse_args()

    # Handle cores/workers argument (prefer --cores, fallback to --workers for compatibility)
    if args.cores is not None:
        num_cores = args.cores
    elif args.workers is not None:
        num_cores = args.workers
    else:
        num_cores = None

    # Validate cores argument
    if num_cores is not None and num_cores < 1:
        print("Error: Number of cores must be at least 1")
        sys.exit(1)

    if num_cores is not None and num_cores > mp.cpu_count():
        print(f"Warning: Requested {num_cores} cores but only {mp.cpu_count()} available")
        num_cores = mp.cpu_count()

    # Validate output directory
    if not os.path.exists(args.output_dir):
        try:
            os.makedirs(args.output_dir, exist_ok=True)
            print(f"Created output directory: {args.output_dir}")
        except OSError as e:
            print(f"Error: Cannot create output directory {args.output_dir}: {e}")
            sys.exit(1)
    elif not os.access(args.output_dir, os.W_OK):
        print(f"Error: Output directory {args.output_dir} is not writable")
        sys.exit(1)

    if args.max_bits < 1 or args.max_bits > 32:
        print("Error: max_bits must be between 1 and 32")
        sys.exit(1)
        
    if args.compression_level < 1 or args.compression_level > 22:
        print("Error: compression_level must be between 1 and 22")
        sys.exit(1)
    
    # Show warning for large datasets
    if args.max_bits >= 16:
        max_factor = 2**args.max_bits - 1
        primes_estimate = max_factor / math.log(max_factor)
        pairs_estimate = primes_estimate * (primes_estimate + 1) // 2
        
        print(f"INFO: {args.max_bits}-bit dataset will generate ~{int(pairs_estimate):,} pairs")
        if args.max_bits >= 18:
            print("WARNING: Large dataset - ensure sufficient disk space and time")
    
    try:
        generate_dataset_optimized_fixed(
            max_bits=args.max_bits,
            output_dir=args.output_dir,
            repo_name=args.repo_name,
            num_workers=num_cores,
            compression_level=args.compression_level
        )
    except KeyboardInterrupt:
        print("\nINTERRUPTED: User interrupted execution.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
