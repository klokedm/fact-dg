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
from typing import Tuple

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

# Check for Xet support
try:
    import hf_xet
    XET_AVAILABLE = True
except ImportError:
    XET_AVAILABLE = False

def get_file_size_mb(file_path: str) -> float:
    """Get file size in MB."""
    try:
        return os.path.getsize(file_path) / (1024**2)
    except OSError:
        return 0.0

def check_xet_availability():
    """Check if Xet is available and working."""
    if not XET_AVAILABLE:
        return False, "hf_xet not installed"
    
    try:
        # Try to import and test basic Xet functionality
        import hf_xet
        return True, "hf_xet available"
    except Exception as e:
        return False, f"hf_xet error: {e}"

def install_xet_support():
    """Install Xet support for HuggingFace Hub."""
    print("🔧 Installing Xet support...")
    try:
        import subprocess
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-U", "huggingface_hub[hf_xet]"
        ], capture_output=True, text=True, check=True)
        
        print("✅ Xet support installed successfully!")
        print("ℹ️  Please restart the script to use Xet features")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Xet support: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error installing Xet: {e}")
        return False

def check_repo_xet_enabled(repo_name: str):
    """Check if a repository has Xet enabled."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        
        # Try to get repository info
        repo_info = api.repo_info(repo_id=repo_name, repo_type="dataset")
        
        # Check if Xet is enabled (this is heuristic-based since there's no direct API)
        # Xet-enabled repos often have certain characteristics
        xet_indicators = [
            hasattr(repo_info, 'xet_enabled') and repo_info.xet_enabled,
            # Add other heuristics as they become available
        ]
        
        return any(xet_indicators), repo_info
    except Exception as e:
        # If repo doesn't exist yet, we can't determine Xet status
        return None, f"Cannot check repo status: {e}"

def print_xet_status():
    """Print current Xet availability and status."""
    print("🔍 Xet Support Status:")
    
    xet_available, xet_msg = check_xet_availability()
    if xet_available:
        print(f"   ✅ Xet: {xet_msg}")
    else:
        print(f"   ❌ Xet: {xet_msg}")
        print("   💡 Install with: pip install -U 'huggingface_hub[hf_xet]'")
        print("   📖 Or use --install-xet flag")
    
    return xet_available

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

def clean_hf_cache(cache_dir: str = None) -> Tuple[bool, str]:
    """Clean HuggingFace cache to free up disk space."""
    if cache_dir is None:
        cache_dir = os.environ.get("HF_DATASETS_CACHE", WORKSPACE_HF_CACHE)
    
    try:
        import shutil
        
        if not os.path.exists(cache_dir):
            return True, "Cache directory doesn't exist - nothing to clean"
        
        # Calculate current cache size
        def get_dir_size(path):
            total = 0
            try:
                for dirpath, dirnames, filenames in os.walk(path):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        if os.path.exists(fp):
                            total += os.path.getsize(fp)
            except (OSError, PermissionError):
                pass
            return total
        
        cache_size_mb = get_dir_size(cache_dir) / (1024**2)
        
        # Remove cache contents
        for item in os.listdir(cache_dir):
            item_path = os.path.join(cache_dir, item)
            try:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.unlink(item_path)
            except (OSError, PermissionError) as e:
                print(f"⚠️ Could not remove {item_path}: {e}")
        
        return True, f"Freed {cache_size_mb:.1f} MB from cache"
        
    except Exception as e:
        return False, f"Cache cleanup failed: {e}"

def create_dataset_card(dataset_info: dict, parquet_path: str) -> str:
    """Create a basic dataset card for direct upload."""
    filename = os.path.basename(parquet_path)
    file_size_mb = get_file_size_mb(parquet_path)
    
    return f"""---
license: mit
dataset_info:
  features:
  - name: input
    dtype: string
  - name: output  
    dtype: string
  config_name: default
  splits:
  - name: train
    num_examples: unknown
task_categories:
- text-generation
- question-answering
language:
- en
tags:
- math
- prime-factorization
- synthetic
size_categories:
- 100M<n<1B
---

# {dataset_info['description']}

This dataset contains {dataset_info['bits']}-bit prime factorization pairs for mathematical training.

## Dataset Details

- **Original file:** {filename}  
- **File size:** {file_size_mb:.1f} MB
- **Format:** Parquet
- **Columns:** input (composite number), output (prime factors)
- **Upload method:** Direct upload (preserves original data exactly)

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("{{repo_name}}")
```

## Data Format

Each row contains:
- `input`: A composite number (product of two primes)
- `output`: The prime factorization in the format "p * q"
"""

def direct_upload_parquet(parquet_path: str, repo_name: str, private: bool = True, 
                         token: str = None) -> bool:
    """Upload parquet file directly using HuggingFace Hub API."""
    try:
        from huggingface_hub import HfApi, login
        
        # Login if token provided
        if token:
            login(token=token)
        
        # Initialize API
        api = HfApi()
        
        print(f"\n📁 Creating repository: {repo_name}")
        
        # Create repository
        try:
            api.create_repo(
                repo_id=repo_name,
                repo_type="dataset",
                private=private,
                exist_ok=True  # Don't fail if repo already exists
            )
            print(f"✅ Repository created/verified")
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"✅ Repository already exists, proceeding with upload")
            else:
                raise e
        
        # Get dataset info for metadata
        dataset_info = detect_dataset_info(parquet_path)
        
        # Create dataset card
        dataset_card = create_dataset_card(dataset_info, parquet_path).replace("{{repo_name}}", repo_name)
        
        print(f"\n☁️  Uploading files to {repo_name}...")
        
        # Upload parquet file
        filename = os.path.basename(parquet_path)
        print(f"   📄 Uploading {filename}...")
        
        api.upload_file(
            path_or_fileobj=parquet_path,
            path_in_repo=f"train-00000-of-00001.parquet",  # Standard HF naming
            repo_id=repo_name,
            repo_type="dataset",
            commit_message=f"Add {dataset_info['bits']}-bit prime factorization dataset"
        )
        print(f"   ✅ Parquet file uploaded")
        
        # Upload dataset card/README
        print(f"   📄 Uploading README.md...")
        api.upload_file(
            path_or_fileobj=dataset_card.encode('utf-8'),
            path_in_repo="README.md",
            repo_id=repo_name,
            repo_type="dataset",
            commit_message="Add dataset card"
        )
        print(f"   ✅ Dataset card uploaded")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Direct upload failed: {e}")
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
                  token: str = None, chunk_size: int = 10000, use_xet: bool = None,
                  streaming: bool = False, clean_cache: bool = False, direct_upload: bool = False):
    """
    Upload parquet dataset to HuggingFace Hub with proper workspace handling.
    
    Args:
        parquet_path: Path to the parquet file
        repo_name: HuggingFace repository name (e.g., "username/dataset-name")  
        private: Whether to make the dataset private
        token: HuggingFace token (if not already logged in)
        chunk_size: Chunk size for processing (smaller = less memory)
        use_xet: Whether to use Xet for upload (None = auto-detect)
        streaming: Use streaming mode to avoid disk space issues (recommended for large files)
        clean_cache: Clean HuggingFace cache before upload to free space
        direct_upload: Upload parquet file directly without datasets library (fastest, minimal resources)
    """
    
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    
    print(f"🚀 HuggingFace Dataset Uploader")
    print(f"Source file: {parquet_path}")
    print(f"File size: {get_file_size_mb(parquet_path):.2f} MB")
    print(f"Target repository: {repo_name}")
    print(f"Privacy: {'Private' if private else 'Public'}")
    
    # Determine upload mode
    if direct_upload:
        upload_mode = "Direct Upload"
        mode_desc = "(fastest, minimal resources, no processing)"
    elif streaming:
        upload_mode = "Streaming"
        mode_desc = "(saves disk space, processes through datasets library)"
    else:
        upload_mode = "Standard"
        mode_desc = "(creates cache files, full validation)"
    
    print(f"Upload mode: {upload_mode} {mode_desc}")
    
    if not direct_upload:
        print(f"HF Cache directory: {WORKSPACE_HF_CACHE}")
        print(f"Datasets Cache: {os.environ['HF_DATASETS_CACHE']}")
    
    # Clean cache if requested
    if clean_cache:
        print(f"🧹 Cleaning HuggingFace cache...")
        cache_success, cache_msg = clean_hf_cache()
        if cache_success:
            print(f"✅ {cache_msg}")
        else:
            print(f"❌ {cache_msg}")
    
    # Detect dataset information
    dataset_info = detect_dataset_info(parquet_path)
    print(f"Detected: {dataset_info['description']}")
    
    # Check Xet availability and determine usage
    xet_available, xet_msg = check_xet_availability()
    
    if use_xet is None:
        # Auto-detect: use Xet if available
        use_xet = xet_available
    elif use_xet and not xet_available:
        print(f"❌ Xet requested but not available: {xet_msg}")
        print("💡 Install with: pip install -U 'huggingface_hub[hf_xet]'")
        return False
    
    print(f"Upload method: {'Xet-enabled' if use_xet else 'Standard'} HuggingFace Hub")
    
    if use_xet:
        print(f"✅ Using Xet for efficient chunk-based deduplication")
        print(f"   - Faster uploads for large files")
        print(f"   - Better resumption on interruption") 
    else:
        print(f"ℹ️  Using standard HuggingFace upload")
        if xet_available:
            print(f"   💡 Add --xet flag to enable Xet uploads")
        else:
            print(f"   💡 Install hf_xet for faster uploads: pip install -U 'huggingface_hub[hf_xet]'")
    
    # Check if target repo is Xet-enabled (informational)
    if use_xet:
        try:
            repo_xet_status, repo_info = check_repo_xet_enabled(repo_name)
            if repo_xet_status is True:
                print(f"✅ Target repository is Xet-enabled")
            elif repo_xet_status is False:
                print(f"⚠️  Target repository may not be Xet-enabled - upload may fall back to standard method")
            # else: repo doesn't exist yet, which is fine
        except Exception:
            pass  # Don't fail on repo status check
    
    # Login if token provided
    if token:
        print("Logging in with provided token...")
        login(token=token)
    
    # Handle direct upload (bypass datasets library entirely)
    if direct_upload:
        print(f"\n⚡ Using direct upload (fastest method)")
        print(f"   - No data processing or validation")
        print(f"   - Minimal memory and disk usage")
        print(f"   - Preserves original parquet format exactly")
        
        success = direct_upload_parquet(
            parquet_path=parquet_path,
            repo_name=repo_name,
            private=private,
            token=token
        )
        
        if success:
            visibility = "private" if private else "public"
            print(f"\n✅ SUCCESS: Dataset uploaded as {visibility} via direct upload")
            print(f"   📡 URL: https://huggingface.co/datasets/{repo_name}")
            print(f"   📊 File: {os.path.basename(parquet_path)} ({get_file_size_mb(parquet_path):.1f} MB)")
            print(f"   ⚡ Total time: seconds (no processing required)")
            
        return success
    
    # Standard/streaming upload using datasets library
    try:
        if streaming:
            print(f"\n📖 Loading dataset in streaming mode from {parquet_path}...")
            print("Streaming mode: minimal disk space usage, direct upload")
            
            # Load dataset in streaming mode - saves massive amounts of disk space
            dataset = load_dataset(
                'parquet', 
                data_files=parquet_path,
                streaming=True  # This avoids creating cache files!
            )
            
            print(f"✅ Dataset loaded in streaming mode")
            print(f"   - Features: {len(dataset['train'].features)}")
            print(f"   - Sample features: {list(dataset['train'].features.keys())[:5]}...")
            # Note: Can't get exact row count in streaming mode without iterating
            
        else:
            print(f"\n📖 Loading dataset from {parquet_path}...")
            print("This may take a few minutes for large files...")
            print("⚠️  Standard mode creates cache files (2-3x file size)")
            
            # Load dataset with workspace caching
            dataset = load_dataset(
                'parquet', 
                data_files=parquet_path,
                cache_dir=os.environ["HF_DATASETS_CACHE"],
                streaming=False
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
        
        # Show row count if available (not available in streaming mode)
        if not streaming:
            print(f"   📊 Rows: {len(dataset['train']):,}")
        else:
            print(f"   📊 Rows: Large dataset (exact count not shown in streaming mode)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: Upload failed")
        print(f"Error details: {e}")
        
        # Print full traceback for detailed debugging
        import traceback
        print(f"\n🔍 Full Error Traceback:")
        print("=" * 60)
        traceback.print_exc()
        print("=" * 60)
        
        # Handle specific error types
        error_str = str(e).lower()
        
        # Authentication errors
        if "401" in error_str or "unauthorized" in error_str:
            print(f"\n🔑 Authentication Error:")
            print(f"   This looks like a token/login issue. Try:")
            print(f"   1. python3 src/upload_to_hf.py --check-auth")
            print(f"   2. python3 src/upload_to_hf.py --login")
            print(f"   3. Or use: --token YOUR_HF_TOKEN")
        
        # Memory errors
        elif "memory" in error_str or "out of memory" in error_str:
            print(f"\n🧠 Memory Error:")
            print(f"   The dataset is too large to load into memory. Try:")
            print(f"   1. Use a machine with more RAM")
            print(f"   2. Split the parquet file into smaller chunks")
            print(f"   3. Enable streaming mode (dataset loading)")
        
        # Disk space/quota errors
        elif ("no space" in error_str or "disk full" in error_str or 
              "disk quota exceeded" in error_str or "errno 122" in error_str):
            print(f"\n💾 Disk Quota/Space Error:")
            print(f"   Hit disk quota limit during dataset processing. Try:")
            print(f"   1. Use streaming mode: --streaming (avoids cache files)")
            print(f"   2. Clean cache first: --clean-cache")
            print(f"   3. Use both: --streaming --clean-cache")
            print(f"   4. Contact admin to increase disk quota")
            print(f"   💡 The HuggingFace datasets library creates cache files 2-3x larger than the original")
        
        # Network/connection errors
        elif "connection" in error_str or "network" in error_str or "timeout" in error_str:
            print(f"\n🌐 Network Error:")
            print(f"   Network connection issue. Try:")
            print(f"   1. Check your internet connection")
            print(f"   2. Retry the upload (it may resume from where it left off)")
            print(f"   3. Use smaller chunk sizes: --chunk-size 5000")
        
        # Dataset loading/parsing errors
        elif "generating" in error_str or "dataset" in error_str:
            print(f"\n📊 Dataset Loading Error:")
            print(f"   Issue loading or parsing the parquet file. Try:")
            print(f"   1. Check if the parquet file is corrupted")
            print(f"   2. Verify the file format and schema")
            print(f"   3. Try loading a smaller sample first")
            print(f"   4. Check if you have permission to read the file")
        
        # Repository errors
        elif "repo" in error_str or "repository" in error_str:
            print(f"\n📦 Repository Error:")
            print(f"   Issue with the target repository. Try:")
            print(f"   1. Check if the repository name format is correct (username/dataset-name)")
            print(f"   2. Verify you have write access to this repository")
            print(f"   3. Check if the repository already exists and has conflicts")
        
        # Provide helpful debugging info
        print(f"\n🔍 Debug Information:")
        print(f"   - Working directory: {os.getcwd()}")
        print(f"   - HF Cache dir exists: {os.path.exists(WORKSPACE_HF_CACHE)}")
        print(f"   - HF Cache writable: {os.access(WORKSPACE_HF_CACHE, os.W_OK)}")
        print(f"   - Parquet file exists: {os.path.exists(parquet_path)}")
        print(f"   - Parquet file readable: {os.access(parquet_path, os.R_OK)}")
        print(f"   - Parquet file size: {get_file_size_mb(parquet_path):.1f} MB")
        
        # Check if it's a space issue
        try:
            import shutil
            total, used, free = shutil.disk_usage("/workspace")
            free_space_gb = free / (1024**3)
            total_space_gb = total / (1024**3)
            used_space_gb = used / (1024**3)
            print(f"   - Free space in /workspace: {free_space_gb:.1f} GB")
            print(f"   - Total space in /workspace: {total_space_gb:.1f} GB")
            print(f"   - Used space in /workspace: {used_space_gb:.1f} GB")
        except Exception as disk_e:
            print(f"   - Could not check disk space: {disk_e}")
        
        # Check parquet file details
        try:
            import pandas as pd
            print(f"   - Attempting to read parquet file with pandas...")
            df = pd.read_parquet(parquet_path, engine='pyarrow')
            print(f"   - Parquet file readable: {len(df):,} rows, {len(df.columns)} columns")
            print(f"   - Columns: {list(df.columns)}")
            print(f"   - Memory usage: {df.memory_usage(deep=True).sum() / (1024**2):.1f} MB")
        except Exception as pd_e:
            print(f"   - Could not read parquet with pandas: {pd_e}")
        
        # Check Python and library versions
        try:
            import sys
            import datasets
            import huggingface_hub
            print(f"   - Python version: {sys.version}")
            print(f"   - datasets version: {datasets.__version__}")
            print(f"   - huggingface_hub version: {huggingface_hub.__version__}")
        except Exception as version_e:
            print(f"   - Could not get version info: {version_e}")
            
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Upload parquet datasets to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check if you're logged in and Xet status
  python3 upload_to_hf.py --check-auth
  
  # Login to HuggingFace (first time setup)
  python3 upload_to_hf.py --login
  
  # Install Xet support for faster uploads
  python3 upload_to_hf.py --install-xet
  
  # Check Xet availability
  python3 upload_to_hf.py --check-xet
  
  # List available datasets
  python3 upload_to_hf.py --list
  
  # Clean HuggingFace cache to free disk space
  python3 upload_to_hf.py --clean-cache

  # Upload private dataset (auto-detects Xet)
  python3 upload_to_hf.py data/prime_products_18bit_fast.parquet username/math-18

  # Upload with Xet enabled (faster, better resumption)
  python3 upload_to_hf.py data/prime_products_18bit_fast.parquet username/math-18 --xet

  # Upload without Xet (standard method)
  python3 upload_to_hf.py data/prime_products_18bit_fast.parquet username/math-18 --no-xet

  # Upload with direct mode (fastest, no processing, minimal resources)
  python3 upload_to_hf.py data/prime_products_20bit_fast.parquet username/math-20 --direct

  # Upload with streaming mode (saves disk space, avoids quota issues)
  python3 upload_to_hf.py data/prime_products_20bit_fast.parquet username/math-20 --streaming

  # Clean cache and upload with direct mode (recommended for large files)
  python3 upload_to_hf.py data/prime_products_20bit_fast.parquet username/math-20 --direct --clean-cache

  # Upload public dataset with direct upload
  python3 upload_to_hf.py data/prime_products_16bit_fast.parquet username/math-16 --public --direct
  
  # Upload with specific token and direct mode
  python3 upload_to_hf.py data/prime_products_20bit_fast.parquet username/math-20 --token YOUR_TOKEN --direct
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
    parser.add_argument("--xet", action="store_true", help="Force enable Xet for uploads (faster, better resumption)")
    parser.add_argument("--no-xet", action="store_true", help="Force disable Xet, use standard uploads")
    parser.add_argument("--check-xet", action="store_true", help="Check Xet availability and status")
    parser.add_argument("--install-xet", action="store_true", help="Install Xet support for HuggingFace Hub")
    parser.add_argument("--streaming", action="store_true", help="Use streaming mode (saves disk space, avoids quota issues)")
    parser.add_argument("--clean-cache", action="store_true", help="Clean HuggingFace cache before upload to free space")
    parser.add_argument("--direct", action="store_true", help="Direct upload (fastest, minimal resources, no processing)")
    
    args = parser.parse_args()
    
    # Handle conflicting flags
    if args.xet and args.no_xet:
        parser.error("Cannot use both --xet and --no-xet flags")
    
    # Direct upload conflicts with other processing modes
    if args.direct and args.streaming:
        parser.error("Cannot use both --direct and --streaming (direct upload doesn't use datasets library)")
    
    if args.direct and (args.xet or args.no_xet):
        parser.error("Xet flags not applicable with --direct upload (bypasses datasets library)")
    
    # Check Xet status
    if args.check_xet:
        print_xet_status()
        return
    
    # Install Xet support
    if args.install_xet:
        success = install_xet_support()
        if success:
            print("🎉 Xet support ready! You can now use --xet flag for faster uploads")
        return
    
    # Check authentication status
    if args.check_auth:
        print("🔍 Checking HuggingFace authentication...")
        if check_authentication():
            print("🎉 Ready to upload datasets!")
        else:
            print("💡 Run with --login to authenticate")
        
        # Also show Xet status
        print()
        print_xet_status()
        return
    
    # Login to HuggingFace
    if args.login:
        login_to_workspace()
        return
    
    # Clean cache only
    if args.clean_cache and not args.parquet_file:
        print("🧹 Cleaning HuggingFace cache...")
        cache_success, cache_msg = clean_hf_cache()
        if cache_success:
            print(f"✅ {cache_msg}")
        else:
            print(f"❌ {cache_msg}")
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
    
    # Determine Xet preference
    use_xet = None  # Auto-detect by default
    if args.xet:
        use_xet = True
    elif args.no_xet:
        use_xet = False
    
    try:
        success = upload_dataset(
            parquet_path=args.parquet_file,
            repo_name=args.repo_name,
            private=not args.public,
            token=args.token,
            chunk_size=args.chunk_size,
            use_xet=use_xet,
            streaming=args.streaming,
            clean_cache=args.clean_cache,
            direct_upload=args.direct
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
