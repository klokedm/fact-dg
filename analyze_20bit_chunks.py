#!/usr/bin/env python3
"""
Analyze incomplete 20-bit dataset chunks to determine what's missing
and provide options for resumption or partial upload.
"""

import os
import glob
from pathlib import Path
import re
from typing import List, Set, Tuple, Optional
import pandas as pd

def analyze_chunks_directory(chunks_dir: str) -> dict:
    """Analyze the chunks directory to understand the current state."""
    
    if not os.path.exists(chunks_dir):
        return {"error": f"Directory not found: {chunks_dir}"}
    
    # Get all chunk files
    chunk_files = glob.glob(os.path.join(chunks_dir, "chunk_*.parquet"))
    chunk_files.sort()
    
    if not chunk_files:
        return {"error": f"No chunk files found in {chunks_dir}"}
    
    # Extract chunk numbers
    chunk_numbers = []
    chunk_pattern = re.compile(r'chunk_(\d+)\.parquet')
    
    for file_path in chunk_files:
        filename = os.path.basename(file_path)
        match = chunk_pattern.match(filename)
        if match:
            chunk_numbers.append(int(match.group(1)))
    
    chunk_numbers.sort()
    
    # Analyze chunk file sizes to estimate pairs per chunk
    file_sizes = []
    for file_path in chunk_files[:100]:  # Sample first 100 files
        try:
            size = os.path.getsize(file_path)
            file_sizes.append(size)
        except OSError:
            continue
    
    # Calculate statistics
    min_chunk = min(chunk_numbers)
    max_chunk = max(chunk_numbers)
    total_chunks_found = len(chunk_numbers)
    expected_chunks_in_range = max_chunk - min_chunk + 1
    missing_chunks = expected_chunks_in_range - total_chunks_found
    
    # Find missing chunk numbers
    all_expected = set(range(min_chunk, max_chunk + 1))
    found_chunks = set(chunk_numbers)
    missing_chunk_numbers = sorted(list(all_expected - found_chunks))
    
    # Estimate pairs per chunk by examining a sample chunk
    pairs_per_chunk = None
    total_pairs_estimate = None
    if chunk_files:
        try:
            sample_file = chunk_files[0]
            import pyarrow.parquet as pq
            table = pq.read_table(sample_file)
            pairs_per_chunk = len(table)
            total_pairs_estimate = pairs_per_chunk * total_chunks_found
        except Exception as e:
            print(f"Could not read sample chunk: {e}")
    
    # Calculate expected total chunks for 20-bit dataset
    expected_total_pairs = 3_364_091_325  # From calculation above
    expected_total_chunks = None
    completion_percentage = None
    
    if pairs_per_chunk:
        expected_total_chunks = (expected_total_pairs + pairs_per_chunk - 1) // pairs_per_chunk
        completion_percentage = (total_pairs_estimate / expected_total_pairs) * 100
    
    return {
        "chunks_dir": chunks_dir,
        "total_files": total_chunks_found,
        "chunk_range": (min_chunk, max_chunk),
        "missing_in_range": missing_chunks,
        "missing_chunk_numbers": missing_chunk_numbers,
        "file_sizes": {
            "min": min(file_sizes) if file_sizes else 0,
            "max": max(file_sizes) if file_sizes else 0,
            "avg": sum(file_sizes) / len(file_sizes) if file_sizes else 0
        },
        "pairs_per_chunk": pairs_per_chunk,
        "total_pairs_found": total_pairs_estimate,
        "expected_total_pairs": expected_total_pairs,
        "expected_total_chunks": expected_total_chunks,
        "completion_percentage": completion_percentage,
        "gaps": len(missing_chunk_numbers) > 0
    }

def print_analysis_report(analysis: dict):
    """Print a comprehensive analysis report."""
    
    if "error" in analysis:
        print(f"❌ Error: {analysis['error']}")
        return
    
    print("📊 20-bit Dataset Chunk Analysis Report")
    print("=" * 50)
    
    print(f"📁 Chunks directory: {analysis['chunks_dir']}")
    print(f"📄 Total chunk files found: {analysis['total_files']:,}")
    print(f"🔢 Chunk range: {analysis['chunk_range'][0]:,} to {analysis['chunk_range'][1]:,}")
    print(f"❓ Missing chunks in range: {analysis['missing_in_range']:,}")
    
    if analysis['pairs_per_chunk']:
        print(f"👥 Pairs per chunk: {analysis['pairs_per_chunk']:,}")
        print(f"📈 Total pairs found: {analysis['total_pairs_found']:,}")
        print(f"🎯 Expected total pairs: {analysis['expected_total_pairs']:,}")
        print(f"📊 Expected total chunks: {analysis['expected_total_chunks']:,}")
        print(f"✅ Completion: {analysis['completion_percentage']:.1f}%")
    
    print(f"\n💾 File size stats:")
    print(f"   Min: {analysis['file_sizes']['min']/1024/1024:.1f} MB")
    print(f"   Max: {analysis['file_sizes']['max']/1024/1024:.1f} MB")
    print(f"   Avg: {analysis['file_sizes']['avg']/1024/1024:.1f} MB")
    
    if analysis['gaps']:
        print(f"\n⚠️  Found gaps in chunk numbering!")
        missing = analysis['missing_chunk_numbers']
        if len(missing) <= 20:
            print(f"   Missing chunks: {missing}")
        else:
            print(f"   Missing chunks: {missing[:10]} ... {missing[-10:]} ({len(missing):,} total)")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    
    if analysis['completion_percentage'] and analysis['completion_percentage'] > 80:
        print("   ✅ Dataset is >80% complete - consider uploading partial dataset")
        print("   📤 Use combine_and_upload_chunks.py to create partial dataset")
        print("   🔄 Or use resume_20bit_generation.py to complete remaining chunks")
    elif analysis['completion_percentage'] and analysis['completion_percentage'] > 50:
        print("   ⏳ Dataset is >50% complete - can resume generation")
        print("   🔄 Use resume_20bit_generation.py to complete missing chunks")
        print("   📤 Or upload partial dataset if time/resources are limited")
    else:
        print("   🔄 Dataset <50% complete - recommend resuming generation")
        print("   🔄 Use resume_20bit_generation.py to continue")
        print("   🗑️  Or consider restarting with optimized settings")

def generate_missing_chunks_list(analysis: dict, output_file: str = "missing_chunks_20bit.txt"):
    """Generate a list of missing chunk numbers for resume script."""
    
    if "error" in analysis or not analysis.get('missing_chunk_numbers'):
        print("No missing chunks to write")
        return
    
    missing = analysis['missing_chunk_numbers']
    expected_total_chunks = analysis.get('expected_total_chunks', analysis['chunk_range'][1] + 1)
    
    # Also include chunks beyond the current max that haven't been started
    max_found = analysis['chunk_range'][1]
    if expected_total_chunks and expected_total_chunks > max_found:
        missing.extend(range(max_found + 1, expected_total_chunks))
    
    with open(output_file, 'w') as f:
        for chunk_num in missing:
            f.write(f"{chunk_num}\n")
    
    print(f"📝 Written {len(missing):,} missing chunk numbers to {output_file}")
    return missing

def main():
    chunks_dir = "data/temp_chunks_20bit"
    
    print("🔍 Analyzing 20-bit dataset chunks...")
    analysis = analyze_chunks_directory(chunks_dir)
    
    print_analysis_report(analysis)
    
    # Generate missing chunks list
    if "error" not in analysis:
        missing_chunks = generate_missing_chunks_list(analysis)
        
        print(f"\n📋 Next steps:")
        print(f"   1. Run: python3 resume_20bit_generation.py (to complete missing chunks)")
        print(f"   2. Run: python3 combine_and_upload_chunks.py (to upload partial dataset)")
        print(f"   3. Check: ls -la data/temp_chunks_20bit/ | wc -l (to monitor progress)")

if __name__ == "__main__":
    main()
