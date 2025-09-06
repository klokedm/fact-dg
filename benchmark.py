#!/usr/bin/env python3
"""
Benchmark script to compare performance of different optimization levels.

This script helps you understand the performance impact of various optimizations.
"""

import time
import sys
import os
import subprocess
from typing import Dict, Any
import numpy as np

# Try to import optimization libraries
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = cp.cuda.is_available()
except ImportError:
    CUPY_AVAILABLE = False


def benchmark_prime_generation(max_bits: int) -> Dict[str, float]:
    """Benchmark different prime generation methods."""
    max_num = 2**max_bits - 1
    print(f"\n📊 Benchmarking prime generation up to {max_num:,}")
    
    results = {}
    
    # Basic Python sieve
    start_time = time.time()
    primes_basic = basic_sieve(max_num)
    results['basic_python'] = time.time() - start_time
    print(f"Basic Python: {results['basic_python']:.3f}s ({len(primes_basic)} primes)")
    
    # NumPy optimized sieve
    start_time = time.time()
    primes_numpy = numpy_sieve(max_num)
    results['numpy_optimized'] = time.time() - start_time
    print(f"NumPy optimized: {results['numpy_optimized']:.3f}s ({len(primes_numpy)} primes)")
    
    # Numba JIT sieve (if available)
    if NUMBA_AVAILABLE:
        from src.generate_optimized import fast_sieve_of_eratosthenes
        
        # First call (includes compilation time)
        start_time = time.time()
        primes_numba = fast_sieve_of_eratosthenes(max_num)
        compilation_time = time.time() - start_time
        
        # Second call (JIT compiled)
        start_time = time.time()
        primes_numba = fast_sieve_of_eratosthenes(max_num)
        results['numba_jit'] = time.time() - start_time
        
        print(f"Numba JIT (1st call): {compilation_time:.3f}s (includes compilation)")
        print(f"Numba JIT (compiled): {results['numba_jit']:.3f}s ({len(primes_numba)} primes)")
        
        # Verify results match
        assert np.array_equal(primes_basic, primes_numba), "Numba results don't match!"
    
    return results


def basic_sieve(limit: int):
    """Basic Python implementation for comparison."""
    if limit < 2:
        return []
    
    prime = [True] * (limit + 1)
    prime[0] = prime[1] = False
    
    p = 2
    while p * p <= limit:
        if prime[p]:
            for i in range(p * p, limit + 1, p):
                prime[i] = False
        p += 1
    
    return [i for i in range(2, limit + 1) if prime[i]]


def numpy_sieve(limit: int):
    """NumPy optimized implementation."""
    if limit < 2:
        return np.array([], dtype=int)
    
    prime = np.ones(limit + 1, dtype=bool)
    prime[0] = prime[1] = False
    
    for p in range(2, int(np.sqrt(limit)) + 1):
        if prime[p]:
            prime[p*p::p] = False
    
    return np.where(prime)[0]


def benchmark_feature_computation(primes, max_bits: int) -> Dict[str, float]:
    """Benchmark feature computation methods."""
    print(f"\n📊 Benchmarking feature computation for {len(primes)} primes")
    
    results = {}
    
    # Basic Python approach
    start_time = time.time()
    basic_features = compute_features_basic(primes, max_bits)
    results['basic_python'] = time.time() - start_time
    print(f"Basic Python: {results['basic_python']:.3f}s")
    
    # NumPy vectorized approach
    start_time = time.time()
    numpy_features = compute_features_numpy(primes, max_bits)
    results['numpy_vectorized'] = time.time() - start_time
    print(f"NumPy vectorized: {results['numpy_vectorized']:.3f}s")
    
    # Numba JIT approach (if available)
    if NUMBA_AVAILABLE:
        from src.generate_optimized import fast_popcount_batch, fast_int_to_bits_batch
        
        start_time = time.time()
        
        # First call (includes compilation)
        popcount = fast_popcount_batch(np.array(primes))
        bits = fast_int_to_bits_batch(np.array(primes), max_bits)
        compilation_time = time.time() - start_time
        
        # Second call (compiled)
        start_time = time.time()
        popcount = fast_popcount_batch(np.array(primes))
        bits = fast_int_to_bits_batch(np.array(primes), max_bits)
        results['numba_jit'] = time.time() - start_time
        
        print(f"Numba JIT (1st call): {compilation_time:.3f}s (includes compilation)")
        print(f"Numba JIT (compiled): {results['numba_jit']:.3f}s")
    
    return results


def compute_features_basic(primes, max_bits: int):
    """Basic Python feature computation."""
    features = []
    for prime in primes:
        features.append({
            'popcount': bin(prime).count('1'),
            'bits': [(prime >> i) & 1 for i in range(max_bits-1, -1, -1)],
            'msb': prime.bit_length() - 1
        })
    return features


def compute_features_numpy(primes, max_bits: int):
    """NumPy vectorized feature computation."""
    primes = np.array(primes)
    
    # Vectorized popcount
    popcount = np.array([bin(p).count('1') for p in primes])
    
    # Vectorized MSB
    msb = np.array([p.bit_length() - 1 for p in primes])
    
    return {'popcount': popcount, 'msb': msb}


def benchmark_memory_usage():
    """Benchmark memory usage of different approaches."""
    print(f"\n💾 Memory usage comparison:")
    
    # Small test case
    test_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    
    # Python lists vs NumPy arrays
    python_size = sys.getsizeof(test_primes) + sum(sys.getsizeof(p) for p in test_primes)
    numpy_size = np.array(test_primes).nbytes
    
    print(f"Python list: {python_size} bytes")
    print(f"NumPy array: {numpy_size} bytes")
    print(f"Memory efficiency: {python_size/numpy_size:.1f}x improvement with NumPy")


def run_end_to_end_benchmark():
    """Run a complete end-to-end benchmark."""
    print("🏁 End-to-end dataset generation benchmark")
    print("=" * 50)
    
    test_bits = [8, 10, 12]  # Small sizes for benchmarking
    
    for bits in test_bits:
        print(f"\n🎯 Testing {bits}-bit dataset generation...")
        
        # Basic version
        start_time = time.time()
        try:
            result = subprocess.run([
                sys.executable, "src/generate.py", str(bits),
                "--output-dir", "benchmark_data_basic"
            ], capture_output=True, text=True, timeout=60)
            basic_time = time.time() - start_time
            basic_success = result.returncode == 0
        except subprocess.TimeoutExpired:
            basic_time = float('inf')
            basic_success = False
        
        # Optimized version (if available)
        optimized_time = float('inf')
        optimized_success = False
        
        if os.path.exists("src/generate_optimized.py") and NUMBA_AVAILABLE:
            start_time = time.time()
            try:
                result = subprocess.run([
                    sys.executable, "src/generate_optimized.py", str(bits),
                    "--output-dir", "benchmark_data_optimized"
                ], capture_output=True, text=True, timeout=60)
                optimized_time = time.time() - start_time
                optimized_success = result.returncode == 0
            except subprocess.TimeoutExpired:
                optimized_time = float('inf')
                optimized_success = False
        
        # Show results
        print(f"Basic version: {'✓' if basic_success else '❌'} {basic_time:.1f}s")
        print(f"Optimized version: {'✓' if optimized_success else '❌'} {optimized_time:.1f}s")
        
        if basic_success and optimized_success and basic_time > 0:
            speedup = basic_time / optimized_time
            print(f"🚀 Speedup: {speedup:.1f}x faster")
        
        # Check file sizes
        basic_file = f"benchmark_data_basic/prime_products_{bits}bit.parquet"
        opt_file = f"benchmark_data_optimized/prime_products_{bits}bit_optimized.parquet"
        
        if os.path.exists(basic_file) and os.path.exists(opt_file):
            basic_size = os.path.getsize(basic_file) / 1024
            opt_size = os.path.getsize(opt_file) / 1024
            print(f"File sizes: Basic {basic_size:.1f}KB, Optimized {opt_size:.1f}KB")


def show_optimization_recommendations():
    """Show specific optimization recommendations."""
    print("\n🔧 OPTIMIZATION RECOMMENDATIONS")
    print("=" * 50)
    
    print(f"✅ NumPy: Always available")
    print(f"{'✅' if NUMBA_AVAILABLE else '❌'} Numba JIT: {'Available' if NUMBA_AVAILABLE else 'Install with: pip install numba'}")
    print(f"{'✅' if CUPY_AVAILABLE else '❌'} CuPy (GPU): {'Available' if CUPY_AVAILABLE else 'Install with: pip install cupy-cuda12x'}")
    
    print(f"\n📈 Expected Performance Improvements:")
    print(f"• Basic → NumPy vectorized: 3-5x faster")
    print(f"• NumPy → Numba JIT: 10-50x faster")
    print(f"• CPU → GPU (large datasets): 5-20x faster")
    
    print(f"\n⚡ Installation Commands:")
    if not NUMBA_AVAILABLE:
        print(f"pip install numba  # Essential for performance")
    if not CUPY_AVAILABLE:
        print(f"pip install cupy-cuda12x  # For GPU acceleration")


def main():
    """Main benchmark function."""
    print("🚀 Synthetic Math Dataset Generator - Performance Benchmark")
    print("=" * 60)
    
    # Show system info
    print(f"Python version: {sys.version}")
    print(f"NumPy available: {'✅' if True else '❌'}")
    print(f"Numba available: {'✅' if NUMBA_AVAILABLE else '❌'}")
    print(f"CuPy/CUDA available: {'✅' if CUPY_AVAILABLE else '❌'}")
    
    # Run benchmarks
    test_bits = 12  # Small enough to run quickly
    
    # Benchmark 1: Prime generation
    prime_results = benchmark_prime_generation(test_bits)
    
    # Generate primes for feature benchmarking
    primes = basic_sieve(2**test_bits - 1)
    
    # Benchmark 2: Feature computation
    feature_results = benchmark_feature_computation(primes, test_bits)
    
    # Benchmark 3: Memory usage
    benchmark_memory_usage()
    
    # Benchmark 4: End-to-end (optional)
    response = input("\n❓ Run end-to-end benchmark? (takes 1-2 minutes) [y/N]: ")
    if response.lower() == 'y':
        run_end_to_end_benchmark()
    
    # Show recommendations
    show_optimization_recommendations()
    
    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()
