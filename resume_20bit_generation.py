#!/usr/bin/env python3
"""
Resume 20-bit dataset generation from where it was interrupted.
Only generates missing chunks to avoid redoing completed work.
"""

import os
import sys
import multiprocessing as mp
import argparse
import glob
import re
from pathlib import Path
from typing import Set, List
from tqdm import tqdm

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def get_existing_chunk_numbers(chunks_dir: str) -> Set[int]:
    """Get set of chunk numbers that already exist."""
    
    if not os.path.exists(chunks_dir):
        return set()
    
    chunk_files = glob.glob(os.path.join(chunks_dir, "chunk_*.parquet"))
    chunk_numbers = set()
    
    chunk_pattern = re.compile(r'chunk_(\d+)\.parquet')
    
    for file_path in chunk_files:
        filename = os.path.basename(file_path)
        match = chunk_pattern.match(filename)
        if match:
            chunk_numbers.add(int(match.group(1)))
    
    return chunk_numbers

def calculate_missing_chunks(existing_chunks: Set[int], total_pairs: int, pairs_per_chunk: int) -> List[int]:
    """Calculate which chunk numbers are missing."""
    
    expected_total_chunks = (total_pairs + pairs_per_chunk - 1) // pairs_per_chunk
    all_expected_chunks = set(range(expected_total_chunks))
    
    missing_chunks = sorted(list(all_expected_chunks - existing_chunks))
    return missing_chunks

def generate_missing_chunk_args(missing_chunks: List[int], primes, prime_features, 
                              max_bits: int, temp_dir: str, pairs_per_chunk: int,
                              num_primes: int, lazy_features: bool = False) -> List[tuple]:
    """Generate worker arguments for missing chunks only."""
    
    worker_args = []
    
    for chunk_id in missing_chunks:
        start_idx = chunk_id * pairs_per_chunk
        end_idx = min(start_idx + pairs_per_chunk, num_primes * (num_primes + 1) // 2)
        
        if start_idx >= end_idx:
            continue  # Skip invalid chunks
        
        worker_args.append((
            start_idx, end_idx,           
            primes, prime_features,       
            max_bits, chunk_id, temp_dir, 
            num_primes, lazy_features     
        ))
    
    return worker_args

def resume_generation(max_bits: int = 20, 
                     chunks_dir: str = "data/temp_chunks_20bit",
                     num_workers: int = None,
                     force_chunk_size: int = None,
                     max_memory_gb: float = None,
                     lazy_features: bool = True):
    """Resume generation for missing chunks only."""
    
    print(f"🔄 Resuming 20-bit dataset generation...")
    print(f"Chunks directory: {chunks_dir}")
    
    # Import required functions
    try:
        from generate_fast import (
            fast_sieve_of_eratosthenes, 
            generate_chunk_fast,
            get_memory_usage
        )
        
        # Try to import numba functions
        try:
            from generate_fast import compute_features_fast
            import numba
            print(f"Numba JIT: ✅ ({numba.__version__})")
        except ImportError:
            print("⚠️ Numba not available - using slower computation")
            compute_features_fast = None
    
    except ImportError as e:
        print(f"❌ Could not import generation functions: {e}")
        sys.exit(1)
    
    # Set up workers
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
    print(f"Using {num_workers} worker processes")
    
    # Generate primes (same as original)
    max_factor = 2**max_bits - 1
    print("Generating prime numbers...")
    primes = fast_sieve_of_eratosthenes(max_factor)
    num_primes = len(primes)
    total_pairs = num_primes * (num_primes + 1) // 2
    
    print(f"Found {num_primes:,} primes up to {max_factor}")
    print(f"Total pairs: {total_pairs:,}")
    print(f"Memory usage: {get_memory_usage()}")
    
    # Check existing chunks
    print("Scanning existing chunks...")
    existing_chunks = get_existing_chunk_numbers(chunks_dir)
    print(f"Found {len(existing_chunks):,} existing chunks")
    
    if not existing_chunks:
        print("❌ No existing chunks found. Use the original generation script instead.")
        sys.exit(1)
    
    # Determine chunk size from existing chunks
    if force_chunk_size is not None:
        pairs_per_chunk = force_chunk_size
        print(f"Using forced chunk size: {pairs_per_chunk:,}")
    else:
        # Try to determine chunk size from existing files
        try:
            chunk_files = glob.glob(os.path.join(chunks_dir, "chunk_*.parquet"))
            if chunk_files:
                import pyarrow.parquet as pq
                sample_table = pq.read_table(chunk_files[0])
                pairs_per_chunk = len(sample_table)
                print(f"Detected chunk size: {pairs_per_chunk:,} pairs per chunk")
            else:
                # Fallback to memory-based calculation
                pairs_per_chunk = 243_000  # Conservative default for 20-bit
                print(f"Using default chunk size: {pairs_per_chunk:,}")
        except Exception as e:
            print(f"Could not determine chunk size: {e}")
            pairs_per_chunk = 243_000  # Conservative fallback
            print(f"Using fallback chunk size: {pairs_per_chunk:,}")
    
    # Calculate missing chunks
    missing_chunks = calculate_missing_chunks(existing_chunks, total_pairs, pairs_per_chunk)
    
    if not missing_chunks:
        print("✅ All chunks are already generated!")
        print("💡 Use combine_and_upload_chunks.py to create final parquet file")
        return True
    
    print(f"📋 Missing chunks: {len(missing_chunks):,}")
    print(f"   First missing: {min(missing_chunks):,}")
    print(f"   Last missing: {max(missing_chunks):,}")
    
    # Calculate completion percentage
    expected_total_chunks = (total_pairs + pairs_per_chunk - 1) // pairs_per_chunk
    completion_pct = (len(existing_chunks) / expected_total_chunks) * 100
    print(f"📊 Current completion: {completion_pct:.1f}%")
    
    # Estimate time remaining
    remaining_pairs = len(missing_chunks) * pairs_per_chunk
    print(f"⏱️  Remaining pairs to generate: ~{remaining_pairs:,}")
    
    # Compute prime features (same logic as original)
    if lazy_features:
        print("Using lazy prime features computation (saves memory)")
        prime_features = None
    else:
        print("Computing prime features...")
        try:
            import numpy as np
            
            n_primes = len(primes)
            bits_array = np.zeros((n_primes, max_bits), dtype=np.uint8)
            popcount_array = np.zeros(n_primes, dtype=np.uint8)  
            msb_index_array = np.zeros(n_primes, dtype=np.uint8)

            if compute_features_fast:
                compute_features_fast(primes, max_bits, bits_array, popcount_array, msb_index_array)
            else:
                # Fallback Python implementation
                for i, prime in enumerate(primes):
                    binary = format(prime, f'0{max_bits}b')
                    bits_array[i] = [int(b) for b in binary]
                    popcount_array[i] = sum(bits_array[i])
                    msb_index_array[i] = max_bits - 1 - binary.find('1')

            prime_features = {
                'bits': bits_array,
                'popcount': popcount_array,
                'msb_index': msb_index_array
            }
        except Exception as e:
            print(f"Error computing prime features: {e}")
            print("Falling back to lazy features")
            prime_features = None
            lazy_features = True
    
    print(f"Memory usage: {get_memory_usage()}")
    
    # Create temp directory if needed
    os.makedirs(chunks_dir, exist_ok=True)
    
    # Generate worker arguments for missing chunks only
    print(f"Preparing work for {len(missing_chunks):,} missing chunks...")
    worker_args = generate_missing_chunk_args(
        missing_chunks, primes, prime_features, 
        max_bits, chunks_dir, pairs_per_chunk, 
        num_primes, lazy_features
    )
    
    if not worker_args:
        print("✅ No work needed - all chunks exist!")
        return True
    
    print(f"Generated {len(worker_args):,} work items")
    
    # Process missing chunks in parallel
    print("🚀 Processing missing chunks in parallel...")
    
    failed_chunks = []
    completed_chunks = 0
    
    try:
        with mp.Pool(num_workers) as pool:
            with tqdm(total=len(worker_args), desc="Generating missing chunks", unit="chunks") as pbar:
                
                # Use imap for better progress tracking
                for result in pool.imap(generate_chunk_fast, worker_args):
                    if result:
                        completed_chunks += 1
                        pbar.set_postfix({'completed': completed_chunks, 'failed': len(failed_chunks)})
                    else:
                        failed_chunks.append(result)
                    
                    pbar.update(1)
    
    except KeyboardInterrupt:
        print("\n⏹️  Generation interrupted by user")
        print(f"📊 Progress: {completed_chunks:,} chunks completed")
        return False
    
    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        return False
    
    # Report results
    print(f"\n✅ Resume generation completed!")
    print(f"   📄 Processed: {len(worker_args):,} chunks")
    print(f"   ✅ Completed: {completed_chunks:,} chunks") 
    print(f"   ❌ Failed: {len(failed_chunks):,} chunks")
    
    if failed_chunks:
        print(f"⚠️  Some chunks failed. You can run this script again to retry.")
    
    # Check final completion status
    final_existing = get_existing_chunk_numbers(chunks_dir)
    final_completion = (len(final_existing) / expected_total_chunks) * 100
    print(f"📊 Final completion: {final_completion:.1f}%")
    
    if final_completion >= 99.9:
        print("🎉 Dataset generation complete!")
        print("💡 Next steps:")
        print("   1. Run: python3 combine_and_upload_chunks.py --repo username/math-20")
        print("   2. Or run: python3 analyze_20bit_chunks.py")
    else:
        print("⏳ Generation partially complete")
        print("💡 Run this script again to complete remaining chunks")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Resume 20-bit dataset generation from interruption",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Resume with default settings
  python3 resume_20bit_generation.py
  
  # Resume with specific memory limit
  python3 resume_20bit_generation.py --max-memory 16
  
  # Resume with custom chunk size
  python3 resume_20bit_generation.py --chunk-size 200000
  
  # Resume with more workers
  python3 resume_20bit_generation.py --workers 8
        """
    )
    
    parser.add_argument("--chunks-dir", default="data/temp_chunks_20bit",
                       help="Directory containing existing chunks")
    parser.add_argument("--workers", type=int,
                       help="Number of worker processes")
    parser.add_argument("--chunk-size", type=int, 
                       help="Force specific chunk size (pairs per chunk)")
    parser.add_argument("--max-memory", type=float,
                       help="Maximum memory usage in GB")
    parser.add_argument("--no-lazy", action="store_true",
                       help="Disable lazy features (uses more memory)")
    
    args = parser.parse_args()
    
    # Resume generation
    success = resume_generation(
        max_bits=20,
        chunks_dir=args.chunks_dir,
        num_workers=args.workers,
        force_chunk_size=args.chunk_size,
        max_memory_gb=args.max_memory,
        lazy_features=not args.no_lazy
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
