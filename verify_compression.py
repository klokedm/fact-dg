#!/usr/bin/env python3
"""
Compression verification script

Tests different compression levels and implementations to verify
optimal settings are being used.
"""

import os
import sys
import time
import subprocess
from typing import Dict, List


def test_compression_levels(test_bits: int = 10) -> Dict[int, Dict[str, float]]:
    """Test different compression levels on a small dataset."""
    print(f"🧪 Testing compression levels with {test_bits}-bit dataset...")
    
    results = {}
    
    # Test different compression levels
    levels_to_test = [1, 3, 6, 9, 15, 19, 22]
    
    for level in levels_to_test:
        print(f"\nTesting compression level {level}...")
        
        # Generate test dataset with specific compression level
        output_dir = f"compression_test_{level}"
        start_time = time.time()
        
        try:
            # Use the basic version with specified compression level
            cmd = [
                sys.executable, "-c", f"""
import sys
sys.path.append('src')
from generate import *
import os

# Monkey-patch to use specific compression level
original_to_parquet = Dataset.to_parquet

def patched_to_parquet(self, path, **kwargs):
    kwargs['compression'] = 'zstd'
    kwargs['compression_level'] = {level}
    return original_to_parquet(self, path, **kwargs)

Dataset.to_parquet = patched_to_parquet

# Generate small dataset
generate_dataset({test_bits}, "{output_dir}", num_workers=1, batch_size=1000)
"""
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            generation_time = time.time() - start_time
            
            if result.returncode == 0:
                # Check file size
                parquet_file = os.path.join(output_dir, f"prime_products_{test_bits}bit.parquet")
                if os.path.exists(parquet_file):
                    file_size_kb = os.path.getsize(parquet_file) / 1024
                    results[level] = {
                        'generation_time': generation_time,
                        'file_size_kb': file_size_kb,
                        'success': True
                    }
                    print(f"  ✓ Level {level}: {file_size_kb:.1f} KB in {generation_time:.1f}s")
                else:
                    results[level] = {'success': False, 'error': 'File not found'}
                    print(f"  ❌ Level {level}: File not generated")
            else:
                results[level] = {'success': False, 'error': result.stderr}
                print(f"  ❌ Level {level}: Generation failed")
                
        except subprocess.TimeoutExpired:
            results[level] = {'success': False, 'error': 'Timeout'}
            print(f"  ❌ Level {level}: Timeout")
        except Exception as e:
            results[level] = {'success': False, 'error': str(e)}
            print(f"  ❌ Level {level}: {e}")
    
    return results


def compare_implementations(test_bits: int = 10) -> Dict[str, Dict[str, float]]:
    """Compare different implementations."""
    print(f"\n🔄 Comparing implementations with {test_bits}-bit dataset...")
    
    implementations = [
        ("Basic", "src/generate.py"),
        ("Optimized (original)", "src/generate_optimized.py"),
        ("Optimized (fixed)", "src/generate_optimized_fixed.py")
    ]
    
    results = {}
    
    for name, script_path in implementations:
        if not os.path.exists(script_path):
            print(f"  ❌ {name}: Script not found - {script_path}")
            continue
            
        print(f"\nTesting {name}...")
        output_dir = f"impl_test_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}"
        
        start_time = time.time()
        
        try:
            cmd = [sys.executable, script_path, str(test_bits), "--output-dir", output_dir]
            if "optimized" in script_path:
                cmd.extend(["--compression-level", "22"])
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            generation_time = time.time() - start_time
            
            if result.returncode == 0:
                # Find the parquet file
                parquet_files = [f for f in os.listdir(output_dir) if f.endswith('.parquet')]
                if parquet_files:
                    parquet_file = os.path.join(output_dir, parquet_files[0])
                    file_size_kb = os.path.getsize(parquet_file) / 1024
                    
                    results[name] = {
                        'generation_time': generation_time,
                        'file_size_kb': file_size_kb,
                        'success': True
                    }
                    print(f"  ✓ {name}: {file_size_kb:.1f} KB in {generation_time:.1f}s")
                else:
                    results[name] = {'success': False, 'error': 'No parquet file found'}
                    print(f"  ❌ {name}: No output file")
            else:
                error_msg = result.stderr[-200:] if result.stderr else result.stdout[-200:]
                results[name] = {'success': False, 'error': error_msg}
                print(f"  ❌ {name}: Failed - {error_msg[:100]}...")
                
        except subprocess.TimeoutExpired:
            results[name] = {'success': False, 'error': 'Timeout'}
            print(f"  ❌ {name}: Timeout")
        except Exception as e:
            results[name] = {'success': False, 'error': str(e)}
            print(f"  ❌ {name}: {e}")
    
    return results


def show_compression_analysis(compression_results: Dict[int, Dict[str, float]]):
    """Show compression level analysis."""
    print("\n📊 COMPRESSION ANALYSIS")
    print("=" * 50)
    
    successful_results = {k: v for k, v in compression_results.items() if v.get('success', False)}
    
    if not successful_results:
        print("❌ No successful compression tests")
        return
    
    print(f"{'Level':<6} {'Size (KB)':<12} {'Time (s)':<10} {'Efficiency':<12}")
    print("-" * 50)
    
    # Sort by compression level
    sorted_results = sorted(successful_results.items())
    
    # Find the largest file size (lowest compression) for efficiency calculation
    max_size = max(v['file_size_kb'] for v in successful_results.values())
    
    best_compression = min(successful_results.items(), key=lambda x: x[1]['file_size_kb'])
    fastest_generation = min(successful_results.items(), key=lambda x: x[1]['generation_time'])
    
    for level, data in sorted_results:
        size_kb = data['file_size_kb']
        time_s = data['generation_time']
        efficiency = ((max_size - size_kb) / max_size * 100) if max_size > 0 else 0
        
        marker = ""
        if level == best_compression[0]:
            marker += " 🏆 Best compression"
        if level == fastest_generation[0]:
            marker += " ⚡ Fastest"
        
        print(f"{level:<6} {size_kb:<12.1f} {time_s:<10.1f} {efficiency:<12.1f}%{marker}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"• Best compression: Level {best_compression[0]} ({best_compression[1]['file_size_kb']:.1f} KB)")
    print(f"• Fastest generation: Level {fastest_generation[0]} ({fastest_generation[1]['generation_time']:.1f}s)")
    
    # Sweet spot recommendation
    sweet_spot = None
    for level, data in sorted_results:
        if data['generation_time'] < fastest_generation[1]['generation_time'] * 1.5:
            if sweet_spot is None or data['file_size_kb'] < sweet_spot[1]['file_size_kb']:
                sweet_spot = (level, data)
    
    if sweet_spot:
        print(f"• Recommended: Level {sweet_spot[0]} (good balance of size and speed)")


def show_implementation_analysis(impl_results: Dict[str, Dict[str, float]]):
    """Show implementation comparison analysis."""
    print("\n🚀 IMPLEMENTATION COMPARISON")
    print("=" * 50)
    
    successful_results = {k: v for k, v in impl_results.items() if v.get('success', False)}
    
    if not successful_results:
        print("❌ No successful implementation tests")
        return
    
    print(f"{'Implementation':<20} {'Size (KB)':<12} {'Time (s)':<10} {'Speedup':<10}")
    print("-" * 60)
    
    # Find baseline (basic implementation) for speedup calculation
    baseline_time = successful_results.get('Basic', {}).get('generation_time', None)
    
    for name, data in successful_results.items():
        size_kb = data['file_size_kb']
        time_s = data['generation_time']
        
        speedup_str = ""
        if baseline_time and baseline_time > 0 and name != 'Basic':
            speedup = baseline_time / time_s
            speedup_str = f"{speedup:.1f}x"
        elif name == 'Basic':
            speedup_str = "baseline"
        
        print(f"{name:<20} {size_kb:<12.1f} {time_s:<10.1f} {speedup_str:<10}")
    
    # Find best implementation
    if successful_results:
        fastest = min(successful_results.items(), key=lambda x: x[1]['generation_time'])
        smallest = min(successful_results.items(), key=lambda x: x[1]['file_size_kb'])
        
        print(f"\n🏆 BEST RESULTS:")
        print(f"• Fastest: {fastest[0]} ({fastest[1]['generation_time']:.1f}s)")
        print(f"• Smallest: {smallest[0]} ({smallest[1]['file_size_kb']:.1f} KB)")


def cleanup_test_directories():
    """Clean up test directories."""
    print("\n🧹 Cleaning up test directories...")
    
    test_dirs = [d for d in os.listdir('.') if d.startswith(('compression_test_', 'impl_test_'))]
    
    for test_dir in test_dirs:
        try:
            import shutil
            shutil.rmtree(test_dir)
            print(f"  ✓ Removed {test_dir}")
        except Exception as e:
            print(f"  ❌ Failed to remove {test_dir}: {e}")


def main():
    """Main verification function."""
    print("🔍 Compression and Implementation Verification")
    print("=" * 60)
    
    # Test compression levels
    compression_results = test_compression_levels()
    show_compression_analysis(compression_results)
    
    # Compare implementations
    impl_results = compare_implementations()
    show_implementation_analysis(impl_results)
    
    # Cleanup
    response = input("\n❓ Clean up test directories? [Y/n]: ")
    if response.lower() != 'n':
        cleanup_test_directories()
    
    print("\n✅ Verification complete!")


if __name__ == "__main__":
    main()
