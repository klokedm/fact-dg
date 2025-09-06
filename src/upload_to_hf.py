#!/usr/bin/env python3
"""
Standalone HuggingFace Dataset Upload Script

Uploads pre-generated parquet files to HuggingFace Hub while keeping
all cache and temporary files in /workspace to avoid permission issues.
"""

import argparse
import os
import sys
from pathlib import Path

# Force HuggingFace to use workspace for all cache/temp files
WORKSPACE_HF_CACHE = "/workspace/.cache/huggingface"
os.environ["HF_HOME"] = WORKSPACE_HF_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = WORKSPACE_HF_CACHE
os.environ["HF_DATASETS_CACHE"] = os.path.join(WORKSPACE_HF_CACHE, "datasets")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(WORKSPACE_HF_CACHE, "transformers")

# Create cache directories
os.makedirs(WORKSPACE_HF_CACHE, exist_ok=True)
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)

def copy_hf_token_to_workspace():
    """Copy HuggingFace token from default location to workspace."""
    default_token_path = os.path.expanduser("~/.cache/huggingface/token")
    workspace_token_path = os.path.join(WORKSPACE_HF_CACHE, "token")
    
    # If workspace token already exists, use it
    if os.path.exists(workspace_token_path):
        return True
    
    # Try to copy from default location
    if os.path.exists(default_token_path):
        try:
            import shutil
            shutil.copy2(default_token_path, workspace_token_path)
            print(f"✅ Copied HuggingFace token to workspace")
            return True
        except Exception as e:
            print(f"⚠️ Could not copy token: {e}")
            return False
    
    return False

# Try to copy existing token to workspace
copy_hf_token_to_workspace()

try:
    from datasets import load_dataset
    from huggingface_hub import HfApi, login
    HF_AVAILABLE = True
except ImportError:
    print("ERROR: HuggingFace libraries not available")
    print("Install with: pip install datasets huggingface_hub")
    sys.exit(1)

def get_file_size_mb(file_path: str) -> float:
    """Get file size in MB."""
    try:
        return os.path.getsize(file_path) / (1024**2)
    except OSError:
        return 0.0

def check_authentication():
    """Check if HuggingFace authentication is working."""
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        print(f"✅ Authenticated as: {user_info['name']}")
        print(f"   Email: {user_info.get('email', 'Not provided')}")
        print(f"   Token location: {os.path.join(WORKSPACE_HF_CACHE, 'token')}")
        return True
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return False

def login_to_workspace():
    """Interactive login that saves token to workspace."""
    try:
        from huggingface_hub import login
        print("🔑 Logging into HuggingFace...")
        print("This will store your token in /workspace/.cache/huggingface/token")
        login()
        print("✅ Login successful!")
        return check_authentication()
    except Exception as e:
        print(f"❌ Login failed: {e}")
        return False

def detect_dataset_info(parquet_path: str):
    """Extract dataset info from filename."""
    filename = os.path.basename(parquet_path)
    
    # Extract bit size from filename
    if "16bit" in filename:
        bits = 16
    elif "18bit" in filename:
        bits = 18
    elif "20bit" in filename:
        bits = 20
    else:
        # Try to parse from filename pattern
        parts = filename.split("_")
        for part in parts:
            if "bit" in part and part.replace("bit", "").isdigit():
                bits = int(part.replace("bit", ""))
                break
        else:
            bits = "unknown"
    
    return {
        "bits": bits,
        "filename": filename,
        "description": f"Synthetic math dataset with {bits}-bit prime factorization pairs"
    }

def upload_dataset(parquet_path: str, repo_name: str, private: bool = True, 
                  token: str = None, chunk_size: int = 10000):
    """
    Upload parquet dataset to HuggingFace Hub with proper workspace handling.
    
    Args:
        parquet_path: Path to the parquet file
        repo_name: HuggingFace repository name (e.g., "username/dataset-name")  
        private: Whether to make the dataset private
        token: HuggingFace token (if not already logged in)
        chunk_size: Chunk size for processing (smaller = less memory)
    """
    
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    
    print(f"🚀 HuggingFace Dataset Uploader")
    print(f"Source file: {parquet_path}")
    print(f"File size: {get_file_size_mb(parquet_path):.2f} MB")
    print(f"Target repository: {repo_name}")
    print(f"Privacy: {'Private' if private else 'Public'}")
    print(f"HF Cache directory: {WORKSPACE_HF_CACHE}")
    print(f"Datasets Cache: {os.environ['HF_DATASETS_CACHE']}")
    
    # Detect dataset information
    dataset_info = detect_dataset_info(parquet_path)
    print(f"Detected: {dataset_info['description']}")
    
    # Login if token provided
    if token:
        print("Logging in with provided token...")
        login(token=token)
    
    try:
        print(f"\n📖 Loading dataset from {parquet_path}...")
        print("This may take a few minutes for large files...")
        
        # Load dataset with workspace caching
        dataset = load_dataset(
            'parquet', 
            data_files=parquet_path,
            cache_dir=os.environ["HF_DATASETS_CACHE"],
            # Use streaming for very large datasets to save memory
            streaming=False  # Set to True for enormous datasets
        )
        
        print(f"✅ Dataset loaded successfully")
        print(f"   - Total rows: {len(dataset['train']):,}")
        print(f"   - Features: {len(dataset['train'].features)}")
        print(f"   - Sample features: {list(dataset['train'].features.keys())[:5]}...")
        
        print(f"\n☁️  Uploading to HuggingFace Hub: {repo_name}")
        print("This will take a while for large datasets...")
        
        # Upload with progress tracking
        dataset.push_to_hub(
            repo_name,
            private=private,
            commit_message=f"Add {dataset_info['bits']}-bit prime factorization dataset",
            # Use smaller chunks for better progress tracking
            max_shard_size="500MB"  # Smaller shards for better upload tracking
        )
        
        visibility = "private" if private else "public"
        print(f"✅ SUCCESS: Dataset uploaded as {visibility}")
        print(f"   📡 URL: https://huggingface.co/datasets/{repo_name}")
        print(f"   📊 Rows: {len(dataset['train']):,}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: Upload failed")
        print(f"Error details: {e}")
        
        # Handle specific authentication errors
        if "401" in str(e) or "Unauthorized" in str(e):
            print(f"\n🔑 Authentication Error:")
            print(f"   This looks like a token/login issue. Try:")
            print(f"   1. python3 src/upload_to_hf.py --check-auth")
            print(f"   2. python3 src/upload_to_hf.py --login")
            print(f"   3. Or use: --token YOUR_HF_TOKEN")
        
        # Provide helpful debugging info
        print(f"\n🔍 Debug Information:")
        print(f"   - Working directory: {os.getcwd()}")
        print(f"   - HF Cache dir exists: {os.path.exists(WORKSPACE_HF_CACHE)}")
        print(f"   - HF Cache writable: {os.access(WORKSPACE_HF_CACHE, os.W_OK)}")
        print(f"   - Parquet file size: {get_file_size_mb(parquet_path):.1f} MB")
        
        # Check if it's a space issue
        try:
            import shutil
            free_space_gb = shutil.disk_usage("/workspace")[2] / (1024**3)
            print(f"   - Free space in /workspace: {free_space_gb:.1f} GB")
        except:
            print(f"   - Could not check free space")
            
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Upload parquet datasets to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check if you're logged in
  python3 upload_to_hf.py --check-auth
  
  # Login to HuggingFace (first time setup)
  python3 upload_to_hf.py --login
  
  # List available datasets
  python3 upload_to_hf.py --list

  # Upload private dataset
  python3 upload_to_hf.py data/prime_products_18bit_fast.parquet username/math-18

  # Upload public dataset  
  python3 upload_to_hf.py data/prime_products_16bit_fast.parquet username/math-16 --public
  
  # Upload with specific token
  python3 upload_to_hf.py data/prime_products_20bit_fast.parquet username/math-20 --token YOUR_TOKEN
        """
    )
    
    parser.add_argument("parquet_file", nargs="?", help="Path to parquet file to upload")
    parser.add_argument("repo_name", nargs="?", help="HuggingFace repository name (e.g., username/dataset-name)")
    parser.add_argument("--public", action="store_true", help="Make dataset public (default: private)")
    parser.add_argument("--token", type=str, help="HuggingFace token (if not logged in)")
    parser.add_argument("--list", action="store_true", help="List available parquet files in data/")
    parser.add_argument("--chunk-size", type=int, default=10000, help="Processing chunk size")
    parser.add_argument("--login", action="store_true", help="Login to HuggingFace (stores token in workspace)")
    parser.add_argument("--check-auth", action="store_true", help="Check authentication status")
    
    args = parser.parse_args()
    
    # Check authentication status
    if args.check_auth:
        print("🔍 Checking HuggingFace authentication...")
        if check_authentication():
            print("🎉 Ready to upload datasets!")
        else:
            print("💡 Run with --login to authenticate")
        return
    
    # Login to HuggingFace
    if args.login:
        login_to_workspace()
        return
    
    # List available datasets
    if args.list:
        print("📁 Available parquet files in data/:")
        data_dir = Path("data")
        if data_dir.exists():
            parquet_files = list(data_dir.glob("*.parquet"))
            if parquet_files:
                for file_path in sorted(parquet_files):
                    size_mb = get_file_size_mb(str(file_path))
                    info = detect_dataset_info(str(file_path))
                    print(f"   - {file_path.name} ({size_mb:.1f} MB) - {info['description']}")
            else:
                print("   No parquet files found")
        else:
            print("   data/ directory not found")
        return
    
    # Validate arguments
    if not args.parquet_file or not args.repo_name:
        parser.print_help()
        print("\n❌ ERROR: Both parquet_file and repo_name are required")
        sys.exit(1)
    
    # Check if file exists
    if not os.path.exists(args.parquet_file):
        print(f"❌ ERROR: File not found: {args.parquet_file}")
        
        # Suggest available files
        data_dir = Path("data")
        if data_dir.exists():
            parquet_files = list(data_dir.glob("*.parquet"))
            if parquet_files:
                print("\n💡 Available files:")
                for file_path in sorted(parquet_files):
                    print(f"   {file_path}")
        sys.exit(1)
    
    # Validate repo name format
    if "/" not in args.repo_name:
        print(f"❌ ERROR: Repository name must be in format 'username/dataset-name'")
        print(f"   Got: {args.repo_name}")
        sys.exit(1)
    
    try:
        success = upload_dataset(
            parquet_path=args.parquet_file,
            repo_name=args.repo_name,
            private=not args.public,
            token=args.token,
            chunk_size=args.chunk_size
        )
        
        if success:
            print(f"\n🎉 Upload completed successfully!")
            sys.exit(0)
        else:
            print(f"\n💥 Upload failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n⏹️  Upload interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
