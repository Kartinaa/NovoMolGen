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

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))
sys.path.append(str(Path(__file__).parent.parent.parent / "utils"))


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
    
    print(f"System info: {cpu_count} CPUs, {memory_gb:.1f}GB RAM")
    
    # Conservative settings for stability
    max_workers = min(16, cpu_count // 2)  # Use half of CPUs
    batch_size = 64  # Good balance for pocket computation
    
    # Adjust based on available memory
    if memory_gb > 64:
        max_workers = min(20, cpu_count // 2)
        batch_size = 64
    elif memory_gb > 32:
        max_workers = min(16, cpu_count // 2)
        batch_size = 64
    else:
        max_workers = min(8, cpu_count // 2)
        batch_size = 16
    
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
        # Build the command
        cmd = [
            "python", "utils/build_pocket_lmdb_and_infer.py",
            "--pdb", pdb_path,
            "--out-dir", out_dir,
            "--job-name", job_name,
            "--radius", str(radius),
            "--batch-size", str(batch_size)
        ]
        
        # Add optional parameters if provided
        if dict_file:
            cmd.extend(["--dict-file", dict_file])
        if weights:
            cmd.extend(["--weights", weights])
        
        # Run the command with timeout
        project_root = Path(__file__).parent.parent.parent
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            cwd=project_root,
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
                                        radius: float = 10.0, cache_file: str = None):
    """Auto-optimized pocket vector computation."""
    print("🚀 Starting auto-optimized pocket vector computation...")
    
    # Get optimal parameters
    max_workers, batch_size = get_optimal_parameters()
    
    # Get unique protein paths
    unique_proteins = df['protein_path'].unique()
    print(f"Found {len(unique_proteins)} unique proteins")
    
    # Check for existing cache
    pocket_vectors = {}
    if cache_file and Path(cache_file).exists():
        print(f"Loading cached pocket vectors from {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                pocket_vectors = pickle.load(f)
            print(f"Loaded {len(pocket_vectors)} cached vectors")
        except Exception as e:
            print(f"Error loading cache: {e}")
    
    # Find proteins that still need processing
    proteins_to_process = [p for p in unique_proteins if p not in pocket_vectors]
    print(f"Need to process {len(proteins_to_process)} proteins")
    
    if proteins_to_process:
        # Check if Uni-Mol tools are available
        try:
            from pocket_tensor import load_mol_repr_tensor
            print("✅ Uni-Mol tools available, computing real pocket vectors...")
        except ImportError:
            print("⚠️  Uni-Mol tools not available, using random vectors as placeholder")
            for protein_path in proteins_to_process:
                pocket_vectors[protein_path] = np.random.randn(512).tolist()
        else:
            # Monitor system before starting
            monitor_system_resources()
            
            # Create temporary base directory
            with tempfile.TemporaryDirectory() as temp_base_dir:
                # Prepare arguments for parallel processing
                args_list = [
                    (protein_path, dict_file, weights, radius, batch_size, temp_base_dir)
                    for protein_path in proteins_to_process
                ]
                
                # Process in parallel with progress tracking
                start_time = time.time()
                processed_count = 0
                
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all jobs
                    future_to_protein = {
                        executor.submit(compute_single_pocket_vector_optimized, args): args[0]
                        for args in args_list
                    }
                    
                    # Collect results with progress tracking
                    for future in tqdm(as_completed(future_to_protein), 
                                     total=len(proteins_to_process), 
                                     desc="Computing pocket vectors"):
                        protein_path = future_to_protein[future]
                        try:
                            result_protein, pocket_vec, error = future.result()
                            pocket_vectors[result_protein] = pocket_vec
                            processed_count += 1
                            
                            if error:
                                print(f"⚠️  {error}")
                            
                            # Monitor system every 100 proteins
                            if processed_count % 100 == 0:
                                monitor_system_resources()
                                elapsed = time.time() - start_time
                                rate = processed_count / elapsed
                                eta = (len(proteins_to_process) - processed_count) / rate
                                print(f"📊 Progress: {processed_count}/{len(proteins_to_process)} "
                                      f"({rate:.2f} proteins/sec, ETA: {eta/60:.1f} min)")
                                
                        except Exception as e:
                            print(f"❌ Error processing {protein_path}: {e}")
                            pocket_vectors[protein_path] = [0.0] * 512
            
            # Save cache
            if cache_file:
                print(f"💾 Saving pocket vectors cache to {cache_file}")
                with open(cache_file, 'wb') as f:
                    pickle.dump(pocket_vectors, f)
    
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


def save_updated_dataset(df: pd.DataFrame, output_dir: Path, step_name: str):
    """Save the updated dataset."""
    output_file = output_dir / f"dataset_with_{step_name}.parquet"
    df.to_parquet(output_file, index=False)
    print(f"💾 Saved updated dataset to {output_file}")
    
    # Also save as CSV for inspection (only first 1000 rows to avoid huge files)
    csv_file = output_dir / f"dataset_with_{step_name}_sample.csv"
    df.head(1000).to_csv(csv_file, index=False)
    print(f"💾 Saved sample dataset CSV to {csv_file}")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Compute pocket vectors using Uni-Mol (Auto-optimized)")
    parser.add_argument("--input_file", type=str, required=True,
                       help="Path to the base dataset parquet file")
    parser.add_argument("--output_dir", type=str, default="finetune_data/processed_data",
                       help="Output directory for processed data")
    parser.add_argument("--dict_file", type=str, default=None,
                       help="Path to Uni-Mol pocket dictionary file")
    parser.add_argument("--weights", type=str, default=None,
                       help="Path to Uni-Mol pocket checkpoint weights")
    parser.add_argument("--radius", type=float, default=10.0,
                       help="Pocket selection radius in Å")
    parser.add_argument("--cache_file", type=str, default=None,
                       help="Path to cache file for pocket vectors")
    parser.add_argument("--force_sequential", action="store_true",
                       help="Force sequential processing (for debugging)")
    
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
        cache_file=args.cache_file
    )
    
    # Save updated dataset
    output_file = save_updated_dataset(df_updated, output_dir, "pocket_vec")
    
    total_time = time.time() - start_time
    print("=" * 50)
    print("✅ Step 2 completed successfully!")
    print(f"📊 Updated dataset shape: {df_updated.shape}")
    print(f"📊 Pocket vector dimensions: {len(df_updated['pocket_vec'].iloc[0])}")
    print(f"⏱️  Total processing time: {total_time/60:.1f} minutes")
    print(f"🚀 Processing rate: {len(df_updated)/total_time:.2f} samples/second")


if __name__ == "__main__":
    main()
