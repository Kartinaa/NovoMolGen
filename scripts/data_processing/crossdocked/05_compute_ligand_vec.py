#!/usr/bin/env python3
"""
Step 5: Compute ligand vectors (Optimized with batch processing)
This script loads the dataset and computes ligand_vec for all ligands using UniMol.
Uses batch processing for maximum efficiency - collects all SMILES first, then processes in batches.

Key optimizations:
- Model initialized only once
- All SMILES collected before processing
- Batch processing for embeddings (much faster than individual processing)
- Error handling: raises error if any ligand fails (no filling missing values)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import torch
import sys
from rdkit import Chem

# Add paths for imports
# Script is at: scripts/data_processing/crossdocked/05_compute_ligand_vec.py
# Need to go up 3 levels to reach project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
utils_path = str(PROJECT_ROOT / "utils")
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

def load_dataset(input_file: Path):
    """Load the dataset."""
    print(f"Loading dataset from {input_file}")
    df = pd.read_parquet(input_file)
    print(f"Loaded {len(df)} samples")
    return df

def sdf_to_smiles(sdf_path: str) -> str:
    """Convert SDF file to SMILES string."""
    try:
        mol = Chem.MolFromMolFile(sdf_path)
        if mol is None:
            return None
        # Remove hydrogens before generating SMILES
        mol = Chem.RemoveHs(mol)
        smiles = Chem.MolToSmiles(mol)
        return smiles
    except Exception as e:
        print(f"Error converting SDF to SMILES for {sdf_path}: {e}")
        return None

def compute_ligand_vectors(df: pd.DataFrame, output_dir: Path, batch_size: int = 32, 
                           allow_failures: bool = False, resume_from: int = 0):
    """Compute ligand vectors for all ligands using UniMol with batch processing.
    
    Args:
        df: DataFrame with ligand_path column
        output_dir: Output directory for saving progress
        batch_size: Batch size for processing (smaller = less memory)
        allow_failures: If True, skip failed batches instead of raising error
        resume_from: Resume from this batch index (for recovery)
    """
    print("Computing ligand vectors...")
    print(f"Found {len(df)} ligands to process")
    print(f"Batch size: {batch_size}")
    if allow_failures:
        print("⚠️  Failure tolerance enabled: failed batches will be skipped")
    
    try:
        # Import UniMolRepr
        from unimol_tools import UniMolRepr
        
        # Step 1: Convert all SDF files to SMILES first (batch processing)
        print("Step 1: Converting SDF files to SMILES...")
        ligand_path_to_smiles = {}
        failed_paths = []
        
        for ligand_path in tqdm(df['ligand_path'], desc="Converting SDF to SMILES"):
            smiles = sdf_to_smiles(ligand_path)
            if smiles is None:
                failed_paths.append(ligand_path)
            else:
                ligand_path_to_smiles[ligand_path] = smiles
        
        # Check for failed conversions
        if failed_paths:
            error_msg = (
                f"ERROR: Failed to convert {len(failed_paths)} SDF files to SMILES!\n"
                f"All SDF files must be convertible to SMILES.\n"
                f"First 10 failed paths:\n{chr(10).join(failed_paths[:10])}"
            )
            raise ValueError(error_msg)
        
        print(f"✓ Successfully converted {len(ligand_path_to_smiles)} SDF files to SMILES")
        
        # Step 2: Initialize UniMolRepr model (only once)
        print("\nStep 2: Initializing UniMolRepr model...")
        clf = UniMolRepr(
            data_type='molecule',
            remove_hs=False,
            model_name='unimolv2',
            model_size='1.1B',
        )
        print("✓ UniMolRepr model initialized successfully")
        
        # Step 3: Collect all SMILES in order
        print("\nStep 3: Preparing SMILES batch...")
        ligand_paths_ordered = df['ligand_path'].tolist()
        all_smiles = [ligand_path_to_smiles[ligand_path] for ligand_path in ligand_paths_ordered]
        print(f"✓ Collected {len(all_smiles)} SMILES for batch processing")
        
        # Step 4: Process all SMILES at once in batches
        print(f"\nStep 4: Computing ligand vectors (batch size: {batch_size})...")
        all_vectors = []
        num_batches = (len(all_smiles) + batch_size - 1) // batch_size
        failed_batches = []
        
        # Skip batches before resume_from
        start_idx = resume_from * batch_size if resume_from > 0 else 0
        if resume_from > 0:
            print(f"Resuming from batch {resume_from} (index {start_idx})...")
            # Load previous vectors if resuming
            progress_file = output_dir / "ligand_vec_progress.npy"
            if progress_file.exists():
                print(f"Loading previous progress from {progress_file}")
                all_vectors = list(np.load(progress_file, allow_pickle=True))
                print(f"Loaded {len(all_vectors)} vectors from previous run")
        
        for i in tqdm(range(start_idx, len(all_smiles), batch_size), 
                     total=num_batches - resume_from,
                     desc="Computing embeddings"):
            batch_smiles = all_smiles[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            try:
                # Clear GPU cache before each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Compute embeddings for the batch
                unimol_repr = clf.get_repr(batch_smiles, return_atomic_reprs=True)
                
                # Get molecular representations from cls_repr
                if 'cls_repr' in unimol_repr:
                    batch_vectors = unimol_repr['cls_repr']
                    # Convert to numpy arrays and move to CPU immediately
                    for vec in batch_vectors:
                        if isinstance(vec, torch.Tensor):
                            vec_np = vec.detach().cpu().numpy().astype(np.float32)
                        else:
                            vec_np = np.array(vec, dtype=np.float32)
                        all_vectors.append(vec_np)
                    
                    # Clear GPU cache after successful batch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    error_msg = f"ERROR: No 'cls_repr' found in UniMol output for batch starting at index {i}"
                    raise ValueError(error_msg)
                    
            except torch.cuda.OutOfMemoryError as e:
                error_msg = f"CUDA OOM for batch {batch_num}/{num_batches} (index {i})"
                print(f"⚠️  {error_msg}")
                failed_batches.append((i, batch_num, str(e)))
                
                # Clear cache and try smaller batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                if allow_failures:
                    # Fill with zero vectors for failed batch
                    print(f"   Filling batch with zero vectors (failure tolerance enabled)")
                    zero_vec = np.zeros(512, dtype=np.float32)  # UniMol typically returns 512-dim
                    for _ in range(len(batch_smiles)):
                        all_vectors.append(zero_vec.copy())
                else:
                    # Try with smaller batch size
                    print(f"   Retrying with smaller batch size ({batch_size // 2})...")
                    try:
                        # Process in smaller chunks
                        chunk_size = max(1, batch_size // 4)
                        for chunk_start in range(0, len(batch_smiles), chunk_size):
                            chunk_smiles = batch_smiles[chunk_start:chunk_start + chunk_size]
                            chunk_repr = clf.get_repr(chunk_smiles, return_atomic_reprs=True)
                            if 'cls_repr' in chunk_repr:
                                for vec in chunk_repr['cls_repr']:
                                    if isinstance(vec, torch.Tensor):
                                        vec_np = vec.detach().cpu().numpy().astype(np.float32)
                                    else:
                                        vec_np = np.array(vec, dtype=np.float32)
                                    all_vectors.append(vec_np)
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                    except Exception as e2:
                        if allow_failures:
                            print(f"   Still failed, filling with zero vectors")
                            zero_vec = np.zeros(512, dtype=np.float32)
                            for _ in range(len(batch_smiles)):
                                all_vectors.append(zero_vec.copy())
                        else:
                            raise RuntimeError(f"Failed even with smaller batch: {e2}")
                    
            except Exception as e:
                error_msg = f"Error for batch {batch_num}/{num_batches} (index {i}): {e}"
                print(f"⚠️  {error_msg}")
                failed_batches.append((i, batch_num, str(e)))
                
                if allow_failures:
                    # Fill with zero vectors
                    print(f"   Filling batch with zero vectors (failure tolerance enabled)")
                    zero_vec = np.zeros(512, dtype=np.float32)
                    for _ in range(len(batch_smiles)):
                        all_vectors.append(zero_vec.copy())
                else:
                import traceback
                traceback.print_exc()
                    raise RuntimeError(f"Failed to compute embeddings for batch {batch_num}/{num_batches}: {e}")
            
            # Save progress periodically (every 50 batches)
            if (batch_num % 50 == 0) and len(all_vectors) > 0:
                progress_file = output_dir / "ligand_vec_progress.npy"
                np.save(progress_file, np.array(all_vectors, dtype=object), allow_pickle=True)
                print(f"   Progress saved: {len(all_vectors)}/{len(all_smiles)} vectors computed")
        
        # Report failures
        if failed_batches:
            print(f"\n⚠️  {len(failed_batches)} batches failed:")
            for idx, batch_num, err in failed_batches[:10]:
                print(f"   Batch {batch_num} (index {idx}): {err[:80]}...")
            if len(failed_batches) > 10:
                print(f"   ... and {len(failed_batches) - 10} more")
        
        # Verify we got all vectors
        if len(all_vectors) != len(ligand_paths_ordered):
            error_msg = (
                f"ERROR: Mismatch in vector count!\n"
                f"Expected {len(ligand_paths_ordered)} vectors, got {len(all_vectors)}"
            )
            raise ValueError(error_msg)
        
        # Step 5: Create mapping from ligand_path to vector
        print("\nStep 5: Mapping vectors to ligand paths...")
        ligand_vectors = {
            path: vec for path, vec in zip(ligand_paths_ordered, all_vectors)
        }
        print(f"✓ Mapped {len(ligand_vectors)} vectors to ligand paths")
        
    except ImportError as e:
        error_msg = f"ERROR: Failed to import UniMol tools: {e}\nCannot compute ligand vectors without UniMol."
        raise ImportError(error_msg)
    except Exception as e:
        error_msg = f"ERROR: Failed to compute ligand vectors: {e}"
        import traceback
        traceback.print_exc()
        raise RuntimeError(error_msg)
    
    # Update dataframe with ligand vectors
    print("\nStep 6: Updating dataset with ligand vectors...")
    df['ligand_vec'] = df['ligand_path'].map(ligand_vectors)
    
    # Check for missing values - raise error if any missing (no filling)
    missing_count = df['ligand_vec'].isna().sum()
    if missing_count > 0:
        missing_paths = df[df['ligand_vec'].isna()]['ligand_path'].unique()[:10]
        error_msg = (
            f"ERROR: {missing_count} samples have missing ligand vectors!\n"
            f"All ligand vectors must be computed successfully.\n"
            f"First 10 missing ligand paths:\n{chr(10).join(missing_paths)}"
        )
        raise ValueError(error_msg)
    
    print(f"✓ All {len(df)} ligand vectors computed successfully")
    
    return df

def save_updated_dataset(df: pd.DataFrame, output_dir: Path, step_name: str, input_file: Path = None):
    """Save the updated dataset.
    
    Args:
        df: DataFrame to save
        output_dir: Output directory
        step_name: Step name (e.g., 'ligand_vec')
        input_file: Input file path (used to extract split name if present)
    """
    # Try to extract split name from input filename (e.g., dataset_with_ifp_meta_test.parquet -> test)
    split_suffix = ""
    if input_file:
        input_stem = Path(input_file).stem
        if "_test" in input_stem:
            split_suffix = "_test"
        elif "_train" in input_stem:
            split_suffix = "_train"
    
    output_file = output_dir / f"dataset_with_{step_name}_ligand_vec{split_suffix}.parquet"
    df.to_parquet(output_file, index=False)
    print(f"Saved updated dataset to {output_file}")
    
    # Save only one row as CSV sample for verification
    csv_file = output_dir / f"dataset_with_{step_name}_ligand_vec{split_suffix}_sample.csv"
    if len(df) > 0:
        # Save first row as sample
        sample_df = df.iloc[[0]].copy()
        # Convert numpy array to list for CSV serialization
        if 'ligand_vec' in sample_df.columns:
            sample_df['ligand_vec'] = sample_df['ligand_vec'].apply(
                lambda x: x.tolist() if isinstance(x, np.ndarray) else x
            )
        sample_df.to_csv(csv_file, index=False)
        print(f"Saved sample dataset CSV (1 row) to {csv_file}")
        # Print vector info for verification
        if 'ligand_vec' in sample_df.columns:
            vec = df['ligand_vec'].iloc[0]
            if isinstance(vec, np.ndarray):
                print(f"Sample ligand_vec shape: {vec.shape}, dtype: {vec.dtype}, first 5 values: {vec[:5]}")
            else:
                print(f"Sample ligand_vec length: {len(vec)}, first 5 values: {vec[:5]}")
    else:
        print("Warning: No data to save as CSV sample")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Compute ligand vectors")
    parser.add_argument("--input_file", type=str, required=True,
                       help="Path to the dataset parquet file")
    parser.add_argument("--output_dir", type=str, default="finetune_data/crossdock_dataset",
                       help="Output directory for processed data")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for processing (default: 32, reduce if OOM errors occur)")
    parser.add_argument("--allow_failures", action="store_true",
                       help="Allow failed batches (fill with zero vectors instead of raising error)")
    parser.add_argument("--resume_from", type=int, default=0,
                       help="Resume from this batch index (for recovery after interruption)")
    
    args = parser.parse_args()
    
    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    df = load_dataset(input_file)
    
    # Compute ligand vectors
    df_updated = compute_ligand_vectors(
        df, 
        output_dir, 
        args.batch_size,
        allow_failures=args.allow_failures,
        resume_from=args.resume_from
    )
    
    # Save updated dataset
    output_file = save_updated_dataset(df_updated, output_dir, "ligand_vec", input_file)
    
    print("Step 5 completed successfully!")
    print(f"Updated dataset shape: {df_updated.shape}")
    if len(df_updated) > 0:
        sample_vec = df_updated['ligand_vec'].iloc[0]
        if isinstance(sample_vec, np.ndarray):
            print(f"Ligand vector dimensions: {sample_vec.shape[0] if len(sample_vec.shape) > 0 else len(sample_vec)}")
        else:
            print(f"Ligand vector dimensions: {len(sample_vec)}")
    else:
        print("Warning: No samples in dataset")

if __name__ == "__main__":
    main()

