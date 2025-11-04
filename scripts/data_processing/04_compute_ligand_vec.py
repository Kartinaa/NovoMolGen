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
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "utils"))

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

def compute_ligand_vectors(df: pd.DataFrame, output_dir: Path, batch_size: int = 32):
    """Compute ligand vectors for all ligands using UniMol with batch processing."""
    print("Computing ligand vectors...")
    print(f"Found {len(df)} ligands to process")
    
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
        
        for i in tqdm(range(0, len(all_smiles), batch_size), 
                     total=num_batches,
                     desc="Computing embeddings"):
            batch_smiles = all_smiles[i:i + batch_size]
            
            try:
                # Compute embeddings for the batch
                unimol_repr = clf.get_repr(batch_smiles, return_atomic_reprs=True)
                
                # Get molecular representations from cls_repr
                if 'cls_repr' in unimol_repr:
                    batch_vectors = unimol_repr['cls_repr']
                    # Convert to numpy arrays
                    for vec in batch_vectors:
                        if isinstance(vec, torch.Tensor):
                            vec_np = vec.detach().cpu().numpy().astype(np.float32)
                        else:
                            vec_np = np.array(vec, dtype=np.float32)
                        all_vectors.append(vec_np)
                else:
                    error_msg = f"ERROR: No 'cls_repr' found in UniMol output for batch starting at index {i}"
                    raise ValueError(error_msg)
                    
            except Exception as e:
                error_msg = (
                    f"ERROR: Failed to compute embeddings for batch {i//batch_size + 1}/{num_batches}\n"
                    f"Batch contains SMILES: {batch_smiles[:3]}...\n"
                    f"Error: {e}"
                )
                import traceback
                traceback.print_exc()
                raise RuntimeError(error_msg)
        
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

def save_updated_dataset(df: pd.DataFrame, output_dir: Path, step_name: str):
    """Save the updated dataset."""
    output_file = output_dir / f"dataset_with_{step_name}_ligand_vec.parquet"
    df.to_parquet(output_file, index=False)
    print(f"Saved updated dataset to {output_file}")
    
    # Save only one row as CSV sample for verification
    csv_file = output_dir / f"dataset_with_{step_name}_sample.csv"
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
    parser.add_argument("--output_dir", type=str, default="processed_data",
                       help="Output directory for processed data")
    parser.add_argument("--batch_size", type=int, default=128,
                       help="Batch size for processing (recommended: 64-256 for large datasets)")
    
    args = parser.parse_args()
    
    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    df = load_dataset(input_file)
    
    # Compute ligand vectors
    df_updated = compute_ligand_vectors(df, output_dir, args.batch_size)
    
    # Save updated dataset
    output_file = save_updated_dataset(df_updated, output_dir, "ligand_vec")
    
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

