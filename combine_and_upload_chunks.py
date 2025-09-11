#!/usr/bin/env python3
"""
Combine existing 20-bit chunks into a single parquet file and upload to HuggingFace.
This allows uploading partial datasets while generation continues.
"""

import os
import sys
import glob
import argparse
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import tempfile
import shutil

def get_chunk_files(chunks_dir: str, limit: int = None) -> list:
    """Get sorted list of chunk files."""
    chunk_files = glob.glob(os.path.join(chunks_dir, "chunk_*.parquet"))
    
    # Sort by chunk number, not lexicographically
    def extract_chunk_num(filepath):
        filename = os.path.basename(filepath)
        try:
            return int(filename.replace('chunk_', '').replace('.parquet', ''))
        except ValueError:
            return 0
    
    chunk_files.sort(key=extract_chunk_num)
    
    if limit:
        chunk_files = chunk_files[:limit]
    
    return chunk_files

def estimate_output_size(chunk_files: list, sample_size: int = 10) -> dict:
    """Estimate the final parquet file size and record count."""
    
    if not chunk_files:
        return {"error": "No chunk files provided"}
    
    # Sample a few files to estimate
    sample_files = chunk_files[:min(sample_size, len(chunk_files))]
    
    total_sample_size = 0
    total_sample_records = 0
    
    for file_path in sample_files:
        try:
            # Get file size
            file_size = os.path.getsize(file_path)
            total_sample_size += file_size
            
            # Get record count
            table = pq.read_table(file_path)
            total_sample_records += len(table)
            
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")
            continue
    
    if total_sample_records == 0:
        return {"error": "Could not read any sample files"}
    
    # Estimate totals
    avg_size_per_file = total_sample_size / len(sample_files)
    avg_records_per_file = total_sample_records / len(sample_files)
    
    estimated_total_size = avg_size_per_file * len(chunk_files)
    estimated_total_records = int(avg_records_per_file * len(chunk_files))
    
    return {
        "chunk_files": len(chunk_files),
        "estimated_size_mb": estimated_total_size / (1024 * 1024),
        "estimated_size_gb": estimated_total_size / (1024 * 1024 * 1024),
        "estimated_records": estimated_total_records,
        "avg_records_per_chunk": int(avg_records_per_file),
        "sample_files_used": len(sample_files)
    }

def combine_chunks_to_parquet(chunk_files: list, output_path: str, 
                             compression: str = 'zstd', compression_level: int = 22) -> bool:
    """Combine chunk files into a single optimized parquet file."""
    
    print(f"📦 Combining {len(chunk_files):,} chunks into {output_path}")
    
    # Read schema from first file
    try:
        first_table = pq.read_table(chunk_files[0])
        schema = first_table.schema
        print(f"📋 Schema: {len(schema)} columns, {len(first_table):,} records in first chunk")
    except Exception as e:
        print(f"❌ Error reading first chunk: {e}")
        return False
    
    # Set up parquet writer with high compression
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Use streaming approach to avoid loading all data into memory
        with pq.ParquetWriter(output_path, schema, compression=compression, 
                             compression_level=compression_level) as writer:
            
            total_records = 0
            
            with tqdm(chunk_files, desc="Combining chunks", unit="chunks") as pbar:
                for chunk_file in pbar:
                    try:
                        # Read chunk
                        table = pq.read_table(chunk_file)
                        
                        # Write to combined file
                        writer.write_table(table)
                        
                        total_records += len(table)
                        pbar.set_postfix({
                            'records': f"{total_records:,}",
                            'size': f"{os.path.getsize(output_path) / (1024*1024):.1f}MB"
                        })
                        
                    except Exception as e:
                        print(f"❌ Error processing {chunk_file}: {e}")
                        continue
            
        print(f"✅ Successfully combined {len(chunk_files):,} chunks")
        print(f"   📄 Total records: {total_records:,}")
        print(f"   💾 Output size: {os.path.getsize(output_path) / (1024*1024*1024):.2f} GB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating combined parquet: {e}")
        return False

def upload_to_huggingface(parquet_path: str, repo_name: str, is_public: bool = False, 
                         use_xet: bool = True) -> bool:
    """Upload the combined parquet file to HuggingFace."""
    
    # Import upload script functionality
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    
    try:
        from upload_to_hf import upload_dataset
        
        print(f"🚀 Uploading to HuggingFace Hub: {repo_name}")
        
        success = upload_dataset(
            parquet_path=parquet_path,
            repo_name=repo_name,
            private=not is_public,
            use_xet=use_xet
        )
        
        return success
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Combine and upload existing 20-bit dataset chunks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze what we have
  python3 combine_and_upload_chunks.py --analyze-only
  
  # Combine first 10,000 chunks for testing
  python3 combine_and_upload_chunks.py --limit 10000 --output data/partial_20bit_test.parquet
  
  # Combine all chunks and upload
  python3 combine_and_upload_chunks.py --repo username/math-20-partial --public
  
  # Combine without uploading
  python3 combine_and_upload_chunks.py --no-upload --output data/partial_20bit.parquet
        """
    )
    
    parser.add_argument("--chunks-dir", default="data/temp_chunks_20bit",
                       help="Directory containing chunk files")
    parser.add_argument("--output", 
                       default="data/prime_products_20bit_partial.parquet",
                       help="Output parquet file path")
    parser.add_argument("--repo", type=str, 
                       help="HuggingFace repository name (e.g., username/dataset-name)")
    parser.add_argument("--public", action="store_true",
                       help="Make HuggingFace dataset public")
    parser.add_argument("--no-upload", action="store_true",
                       help="Skip HuggingFace upload, just create parquet file")
    parser.add_argument("--limit", type=int,
                       help="Limit number of chunks to process (for testing)")
    parser.add_argument("--analyze-only", action="store_true",
                       help="Only analyze chunks, don't combine or upload")
    parser.add_argument("--compression", default="zstd",
                       choices=["snappy", "gzip", "brotli", "zstd"],
                       help="Compression algorithm")
    parser.add_argument("--compression-level", type=int, default=22,
                       help="Compression level (higher = smaller file)")
    parser.add_argument("--no-xet", action="store_true",
                       help="Disable Xet for upload (use standard HF)")
    
    args = parser.parse_args()
    
    # Check if chunks directory exists
    if not os.path.exists(args.chunks_dir):
        print(f"❌ Chunks directory not found: {args.chunks_dir}")
        sys.exit(1)
    
    # Get chunk files
    print(f"📁 Scanning chunks directory: {args.chunks_dir}")
    chunk_files = get_chunk_files(args.chunks_dir, args.limit)
    
    if not chunk_files:
        print(f"❌ No chunk files found in {args.chunks_dir}")
        sys.exit(1)
    
    print(f"📄 Found {len(chunk_files):,} chunk files")
    
    if args.limit:
        print(f"⚠️  Limited to first {args.limit:,} chunks for processing")
    
    # Estimate output size
    print(f"🔍 Estimating output size...")
    estimate = estimate_output_size(chunk_files)
    
    if "error" in estimate:
        print(f"❌ Error estimating size: {estimate['error']}")
        sys.exit(1)
    
    print(f"📊 Estimation:")
    print(f"   Chunks: {estimate['chunk_files']:,}")
    print(f"   Records: ~{estimate['estimated_records']:,}")
    print(f"   Size: ~{estimate['estimated_size_gb']:.2f} GB")
    print(f"   Avg per chunk: {estimate['avg_records_per_chunk']:,} records")
    
    if args.analyze_only:
        print("✅ Analysis complete (--analyze-only specified)")
        sys.exit(0)
    
    # Ask for confirmation if file will be large
    if estimate['estimated_size_gb'] > 10:
        print(f"⚠️  Output file will be very large (~{estimate['estimated_size_gb']:.1f} GB)")
        response = input("Continue? (y/N): ").strip().lower()
        if response != 'y':
            print("Cancelled")
            sys.exit(0)
    
    # Combine chunks
    success = combine_chunks_to_parquet(
        chunk_files, 
        args.output, 
        compression=args.compression,
        compression_level=args.compression_level
    )
    
    if not success:
        print("❌ Failed to combine chunks")
        sys.exit(1)
    
    # Upload if requested
    if not args.no_upload:
        if not args.repo:
            print("❌ --repo required for upload (or use --no-upload)")
            sys.exit(1)
        
        upload_success = upload_to_huggingface(
            args.output, 
            args.repo, 
            is_public=args.public,
            use_xet=not args.no_xet
        )
        
        if upload_success:
            print("🎉 Upload completed successfully!")
        else:
            print("💥 Upload failed!")
            sys.exit(1)
    
    print(f"✅ Process completed!")
    print(f"   📄 Output file: {args.output}")
    if not args.no_upload and args.repo:
        print(f"   🌐 HuggingFace: https://huggingface.co/datasets/{args.repo}")

if __name__ == "__main__":
    main()
