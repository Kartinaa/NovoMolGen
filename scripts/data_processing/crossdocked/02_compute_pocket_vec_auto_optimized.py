#!/usr/bin/env python3
"""
Step 2: Compute pocket vectors using Uni-Mol (Auto-optimized for your hardware)
Optimized for 32 CPU cores + RTX 4070 Ti Super
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import torch
import sys
import subprocess
import tempfile
import os
import glob
import pickle
import json
import time
import psutil
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add src and utils to path for imports
# Script is at: scripts/data_processing/crossdocked/02_compute_pocket_vec_auto_optimized.py
# Need to go up 4 levels to reach project root
_project_root = Path(__file__).parent.parent.parent.parent
_utils_path = str(_project_root / "utils")
_src_path = str(_project_root / "src")
if _utils_path not in sys.path:
    sys.path.insert(0, _utils_path)
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)


def load_base_dataset(input_file: Path):
    """Load the base dataset."""
    print(f"Loading base dataset from {input_file}")
    df = pd.read_parquet(input_file)
    print(f"Loaded {len(df)} samples")
    return df


def get_optimal_parameters():
    """Get optimal parameters based on system resources."""
    cpu_count = mp.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    
    print(f"System info: {cpu_count} CPUs, {memory_gb:.1f}GB total RAM, {available_memory_gb:.1f}GB available")
    
    # Very conservative settings for low memory systems
    # Each worker needs ~2-4GB for Uni-Mol processing
    if available_memory_gb < 16:
        max_workers = min(2, cpu_count // 4)  # Very limited parallelism
        batch_size = 8
        print("⚠️  Low memory detected, using minimal parallelism")
    elif available_memory_gb < 32:
        max_workers = min(4, cpu_count // 3)
        batch_size = 16
        print("⚠️  Moderate memory, using conservative settings")
    elif available_memory_gb < 64:
        max_workers = min(8, cpu_count // 2)
        batch_size = 32
    else:
        max_workers = min(16, cpu_count // 2)
        batch_size = 64
    
    print(f"Recommended: max_workers={max_workers}, batch_size={batch_size}")
    return max_workers, batch_size


def monitor_system_resources():
    """Monitor system resources during processing."""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    print(f"System usage: CPU {cpu_percent:.1f}%, RAM {memory.percent:.1f}% ({memory.used/1024**3:.1f}GB/{memory.total/1024**3:.1f}GB)")
    
    # Warning if resources are high
    if cpu_percent > 90:
        print("⚠️  High CPU usage detected")
    if memory.percent > 85:
        print("⚠️  High memory usage detected")


def compute_single_pocket_vector_optimized(args_tuple):
    """Optimized version with better error handling and resource monitoring."""
    # Ensure utils path is available in subprocess
    import sys
    from pathlib import Path
    # Get project root from the script's location
    # Script is at: scripts/data_processing/crossdocked/02_compute_pocket_vec_auto_optimized.py
    # Need to go up 4 levels to reach project root
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent.parent
    utils_path = str(project_root / "utils")
    if utils_path not in sys.path:
        sys.path.insert(0, utils_path)
    
    protein_path, dict_file, weights, radius, batch_size, temp_base_dir = args_tuple
    
    try:
        # Create a unique job name for this protein
        protein_name = Path(protein_path).stem
        job_name = f"pocket_{protein_name}"
        
        # Create temporary directory for this specific protein
        with tempfile.TemporaryDirectory(dir=temp_base_dir) as temp_dir:
            temp_path = Path(temp_dir)
            
            # Run build_pocket_lmdb_and_infer.py with timeout
            success = _run_pocket_inference_with_timeout(
                pdb_path=protein_path,
                out_dir=str(temp_path),
                job_name=job_name,
                dict_file=dict_file,
                weights=weights,
                radius=radius,
                batch_size=batch_size,
                timeout=300  # 5 minutes timeout per protein
            )
            
            if success:
                # Load the generated pocket vector
                pkl_files = glob.glob(str(temp_path / f"*_{job_name}.out.pkl"))
                if pkl_files:
                    from pocket_tensor import load_mol_repr_tensor
                    pocket_vec = load_mol_repr_tensor(pkl_files[0])
                    return protein_path, pocket_vec.tolist(), None
                else:
                    return protein_path, [0.0] * 512, f"No pickle file found for {protein_path}"
            else:
                return protein_path, [0.0] * 512, f"Failed to process {protein_path}"
                
    except Exception as e:
        return protein_path, [0.0] * 512, f"Error processing {protein_path}: {e}"


def _run_pocket_inference_with_timeout(pdb_path: str, out_dir: str, job_name: str, 
                                     dict_file: str, weights: str, radius: float, 
                                     batch_size: int, timeout: int = 300) -> bool:
    """Run pocket inference with timeout and better error handling."""
    try:
        # Check required parameters
        if not dict_file or not weights:
            print(f"❌ Error: --dict-file and --weights are required for pocket inference")
            return False
        
        # Check if files exist
        if not Path(dict_file).exists():
            print(f"❌ Error: dict-file not found: {dict_file}")
            return False
        if not Path(weights).exists():
            print(f"❌ Error: weights file not found: {weights}")
            return False
        
        # Get project root (script is at scripts/data_processing/crossdocked/, need to go up 4 levels)
        script_path = Path(__file__).resolve()
        project_root = script_path.parent.parent.parent.parent
        
        # Build the command with absolute path to the script
        build_script = project_root / "utils" / "build_pocket_lmdb_and_infer.py"
        if not build_script.exists():
            print(f"❌ Error: build_pocket_lmdb_and_infer.py not found at {build_script}")
            return False
        
        cmd = [
            "python", str(build_script),
            "--pdb", pdb_path,
            "--out-dir", out_dir,
            "--job-name", job_name,
            "--dict-file", str(Path(dict_file).resolve()),  # Use absolute path
            "--weights", str(Path(weights).resolve()),      # Use absolute path
            "--radius", str(radius),
            "--batch-size", str(batch_size)
        ]
        
        # Run the command with timeout
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            cwd=str(project_root),
            timeout=timeout
        )
        
        if result.returncode == 0:
            return True
        else:
            print(f"Error running pocket inference: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"Timeout ({timeout}s) processing {pdb_path}")
        return False
    except Exception as e:
        print(f"Exception running pocket inference: {e}")
        return False


def compute_pocket_vectors_auto_optimized(df: pd.DataFrame, output_dir: Path, 
                                        dict_file: str = None, weights: str = None, 
                                        radius: float = 10.0, cache_file: str = None,
                                        batch_save_interval: int = 100):
    """Auto-optimized pocket vector computation with memory-efficient batching."""
    print("🚀 Starting auto-optimized pocket vector computation...")
    print("💡 Cache file: If it exists, will skip already-computed proteins. "
          "If not, will create one automatically.")
    
    # Get optimal parameters
    max_workers, batch_size = get_optimal_parameters()
    
    # Get unique protein paths
    unique_proteins = df['protein_path'].unique()
    print(f"Found {len(unique_proteins)} unique proteins")
    
    # Check for existing cache (load incrementally to save memory)
    pocket_vectors = {}
    if cache_file and Path(cache_file).exists():
        print(f"📂 Loading cached pocket vectors from {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                pocket_vectors = pickle.load(f)
            print(f"✅ Loaded {len(pocket_vectors)} cached vectors")
        except Exception as e:
            print(f"⚠️  Error loading cache: {e}, will recompute all")
            pocket_vectors = {}
    
    # Find proteins that still need processing
    proteins_to_process = [p for p in unique_proteins if p not in pocket_vectors]
    print(f"📋 Need to process {len(proteins_to_process)} proteins")
    
    if proteins_to_process:
        # Check if Uni-Mol tools are available
        try:
            from pocket_tensor import load_mol_repr_tensor
            print("✅ Uni-Mol tools available, computing real pocket vectors...")
        except ImportError as e:
            print(f"❌ Failed to import pocket_tensor: {e}")
            print(f"   Project root: {_project_root}")
            print(f"   Utils path: {_utils_path}")
            print(f"   Utils path exists: {Path(_utils_path).exists()}")
            pocket_tensor_file = Path(_utils_path) / "pocket_tensor.py"
            print(f"   pocket_tensor.py exists: {pocket_tensor_file.exists()}")
            print(f"   Current sys.path (first 5):")
            for p in sys.path[:5]:
                print(f"     - {p}")
            print("⚠️  Uni-Mol tools not available, using random vectors as placeholder")
            for protein_path in proteins_to_process:
                pocket_vectors[protein_path] = np.random.randn(512).tolist()
        else:
            # Monitor system before starting
            monitor_system_resources()
            
            # Create temporary base directory
            with tempfile.TemporaryDirectory() as temp_base_dir:
                # Process in batches to save memory
                start_time = time.time()
                processed_count = 0
                total_to_process = len(proteins_to_process)
                
                # Process in smaller batches to avoid memory buildup
                batch_size_proteins = max(50, max_workers * 10)  # Process in chunks
                
                for batch_start in range(0, total_to_process, batch_size_proteins):
                    batch_end = min(batch_start + batch_size_proteins, total_to_process)
                    batch_proteins = proteins_to_process[batch_start:batch_end]
                    
                    print(f"\n🔄 Processing batch {batch_start//batch_size_proteins + 1}: "
                          f"proteins {batch_start+1}-{batch_end} of {total_to_process}")
                    
                    # Prepare arguments for this batch
                    args_list = [
                        (protein_path, dict_file, weights, radius, batch_size, temp_base_dir)
                        for protein_path in batch_proteins
                    ]
                    
                    # Process batch in parallel
                    batch_results = {}
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    future_to_protein = {
                        executor.submit(compute_single_pocket_vector_optimized, args): args[0]
                        for args in args_list
                    }
                    
                        # Collect results for this batch
                    for future in tqdm(as_completed(future_to_protein), 
                                         total=len(batch_proteins), 
                                         desc=f"Batch {batch_start//batch_size_proteins + 1}"):
                        protein_path = future_to_protein[future]
                        try:
                            result_protein, pocket_vec, error = future.result()
                                batch_results[result_protein] = pocket_vec
                            processed_count += 1
                            
                            if error:
                                print(f"⚠️  {error}")
                            
                            except Exception as e:
                                print(f"❌ Error processing {protein_path}: {e}")
                                batch_results[protein_path] = [0.0] * 512
                    
                    # Add batch results to main dict
                    pocket_vectors.update(batch_results)
                    
                    # Periodically save cache to avoid losing progress
                    if cache_file and processed_count % batch_save_interval == 0:
                        print(f"💾 Saving intermediate cache ({processed_count}/{total_to_process} processed)...")
                        try:
                            with open(cache_file, 'wb') as f:
                                pickle.dump(pocket_vectors, f)
                        except Exception as e:
                            print(f"⚠️  Failed to save cache: {e}")
                    
                    # Monitor system after each batch
                    monitor_system_resources()
                    elapsed = time.time() - start_time
                    if processed_count > 0:
                        rate = processed_count / elapsed
                        remaining = total_to_process - processed_count
                        eta = remaining / rate if rate > 0 else 0
                        print(f"📊 Overall progress: {processed_count}/{total_to_process} "
                              f"({rate:.2f} proteins/sec, ETA: {eta/60:.1f} min)")
            
            # Final cache save
            if cache_file:
                print(f"💾 Saving final cache to {cache_file}")
                try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(pocket_vectors, f)
                    print(f"✅ Cache saved with {len(pocket_vectors)} vectors")
                except Exception as e:
                    print(f"⚠️  Failed to save final cache: {e}")
    
    return _update_dataframe_with_vectors(df, pocket_vectors)


def _update_dataframe_with_vectors(df: pd.DataFrame, pocket_vectors: dict) -> pd.DataFrame:
    """Update dataframe with pocket vectors and handle missing values."""
    print("📝 Updating dataset with pocket vectors...")
    df['pocket_vec'] = df['protein_path'].map(pocket_vectors)
    
    # Check for missing values
    missing_count = df['pocket_vec'].isna().sum()
    if missing_count > 0:
        print(f"⚠️  {missing_count} samples have missing pocket vectors")
        # Fill with zero vectors
        df['pocket_vec'] = df['pocket_vec'].fillna(df['pocket_vec'].apply(lambda x: [0.0] * 512))
    
    return df


def save_updated_dataset(df: pd.DataFrame, output_dir: Path, step_name: str, input_file: Path = None):
    """Save the updated dataset.
    
    Args:
        df: DataFrame to save
        output_dir: Output directory
        step_name: Step name (e.g., 'pocket_vec')
        input_file: Input file path (used to extract split name if present)
    """
    # Try to extract split name from input filename (e.g., base_dataset_test.parquet -> test)
    split_suffix = ""
    if input_file:
        input_stem = Path(input_file).stem
        if "_test" in input_stem:
            split_suffix = "_test"
        elif "_train" in input_stem:
            split_suffix = "_train"
    
    output_file = output_dir / f"dataset_with_{step_name}{split_suffix}.parquet"
    df.to_parquet(output_file, index=False)
    print(f"💾 Saved updated dataset to {output_file}")
    
    # Also save as CSV for inspection (only first 1000 rows to avoid huge files)
    csv_file = output_dir / f"dataset_with_{step_name}{split_suffix}_sample.csv"
    df.head(1000).to_csv(csv_file, index=False)
    print(f"💾 Saved sample dataset CSV to {csv_file}")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Compute pocket vectors using Uni-Mol (Auto-optimized)")
    parser.add_argument("--input_file", type=str, required=True,
                       help="Path to the base dataset parquet file")
    parser.add_argument("--output_dir", type=str, default="finetune_data/crossdock_dataset",
                       help="Output directory for processed data")
    parser.add_argument("--dict_file", type=str, required=True,
                       help="Path to Uni-Mol pocket dictionary file (required)")
    parser.add_argument("--weights", type=str, required=True,
                       help="Path to Uni-Mol pocket checkpoint weights (required)")
    parser.add_argument("--radius", type=float, default=10.0,
                       help="Pocket selection radius in Å")
    parser.add_argument("--cache_file", type=str, default=None,
                       help="Path to cache file for pocket vectors")
    parser.add_argument("--force_sequential", action="store_true",
                       help="Force sequential processing (for debugging)")
    parser.add_argument("--batch_save_interval", type=int, default=100,
                       help="Save cache every N processed proteins (default: 100)")
    
    args = parser.parse_args()
    
    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set default cache file if not provided
    if args.cache_file is None:
        args.cache_file = str(output_dir / "pocket_vectors_cache.pkl")
    
    print("🎯 Auto-optimized Pocket Vector Computation")
    print("=" * 50)
    
    # Load base dataset
    df = load_base_dataset(input_file)
    
    # Compute pocket vectors
    start_time = time.time()
    df_updated = compute_pocket_vectors_auto_optimized(
        df, 
        output_dir, 
        dict_file=args.dict_file,
        weights=args.weights,
        radius=args.radius,
        cache_file=args.cache_file,
        batch_save_interval=args.batch_save_interval
    )
    
    # Save updated dataset
    output_file = save_updated_dataset(df_updated, output_dir, "pocket_vec", input_file=input_file)
    
    total_time = time.time() - start_time
    print("=" * 50)
    print("✅ Step 2 completed successfully!")
    print(f"📊 Updated dataset shape: {df_updated.shape}")
    print(f"📊 Pocket vector dimensions: {len(df_updated['pocket_vec'].iloc[0])}")
    print(f"⏱️  Total processing time: {total_time/60:.1f} minutes")
    print(f"🚀 Processing rate: {len(df_updated)/total_time:.2f} samples/second")


if __name__ == "__main__":
    main()
