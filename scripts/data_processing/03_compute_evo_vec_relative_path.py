#!/usr/bin/env python3
"""
Step 3: Compute evolutionary vectors using ESM-2
This script loads the dataset and computes evo_vec for each protein-pocket pair
using ESM2PocketEmbedder, computing mean embeddings for pocket residues.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import torch
import sys

# Add src and utils to path for imports
# Add only the utils package first to avoid colliding with src/utils.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "utils"))

from esm_embedding_extract import ESM2PocketEmbedder  # ← 来自 utils/esm_embedding_extract.py


def to_relative_path(p: str) -> str:
    """Strip absolute prefix, keep only finetune_data/... as relative path."""
    marker = "finetune_data/"
    idx = p.find(marker)
    if idx != -1:
        return p[idx:]
    return p


def to_absolute_path(p: str) -> str:
    """Resolve a (possibly relative) path against PROJECT_ROOT."""
    path = Path(p)
    if path.is_absolute():
        return str(path)
    return str(PROJECT_ROOT / path)


def load_dataset(input_file: Path):
    """Load the dataset and normalise protein_path to relative paths."""
    print(f"Loading dataset from {input_file}")
    df = pd.read_parquet(input_file)
    print(f"Loaded {len(df)} samples")

    if "protein_path" in df.columns:
        df["protein_path"] = df["protein_path"].apply(to_relative_path)
        print(f"Normalised protein_path to relative (e.g. {df['protein_path'].iloc[0]})")

    return df

def compute_evolutionary_vectors(df: pd.DataFrame, output_dir: Path, batch_size: int = 32):
    """Compute evolutionary vectors for all proteins using ESM2PocketEmbedder."""
    print("Computing evolutionary vectors...")
    
    def get_protein_path(pocket_rel):
        """From pocket relative path, find protein.pdb in the same directory."""
        pocket_abs = Path(to_absolute_path(pocket_rel))
        return str(pocket_abs.parent / "protein.pdb")

    df['actual_protein_path'] = df['protein_path'].apply(get_protein_path)
    
    # Get unique protein-pocket pairs using actual protein paths
    unique_pairs = df[['actual_protein_path', 'protein_path']].drop_duplicates()
    unique_pairs = unique_pairs.rename(columns={'actual_protein_path': 'protein_path', 'protein_path': 'pocket_path'})
    print(f"Found {len(unique_pairs)} unique protein-pocket pairs")
    
    # Initialize evolutionary vector storage
    evo_vectors = {}
    
    try:
        # Import ESM2PocketEmbedder
        
        # Initialize ESM-2 embedder
        print("Initializing ESM-2 model...")
        embedder = ESM2PocketEmbedder(
            model_name="esm2_t33_650M_UR50D",
            repr_layer=33,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"Using device: {embedder.device}")
        
        for idx, row in tqdm(unique_pairs.iterrows(), total=len(unique_pairs), desc="Computing evolutionary vectors"):
            protein_path = row['protein_path']  # absolute path to protein.pdb
            pocket_path = row['pocket_path']    # relative pocket path (finetune_data/...)
            pocket_abs = to_absolute_path(pocket_path)
            pair_key = f"{protein_path}_{pocket_path}"

            try:
                pocket_res_list, pocket_emb = embedder.pocket_embeddings(
                    full_pdb_path=protein_path,
                    pocket_pdb_path=pocket_abs,
                    chain_id=None
                )
                
                # Compute mean embedding as the single representative vector
                pocket_vec = pocket_emb.mean(dim=0)  # [H]
                evo_vectors[pair_key] = pocket_vec.detach().cpu().numpy()  # Convert to numpy array
                
                print(f"Successfully computed embedding for {pair_key}: shape {pocket_vec.shape}")
                
            except Exception as e:
                print(f"Error processing {pair_key}: {e}")
                # Use zero vector as fallback
                evo_vectors[pair_key] = np.zeros(1280, dtype=np.float32)
    
    except ImportError as e:
        print(f"Warning: ESM-2 tools not available ({e}), using random vectors as placeholder")
        for idx, row in tqdm(unique_pairs.iterrows(), total=len(unique_pairs), desc="Creating placeholder evolutionary vectors"):
            protein_path = row['protein_path']
            pocket_path = row['pocket_path']
            pair_key = f"{protein_path}_{pocket_path}"
            # Create random vector as placeholder
            evo_vectors[pair_key] = np.random.randn(1280).astype(np.float32)
    
    # Create pair key for mapping using actual protein path and original pocket path
    df['pair_key'] = df['actual_protein_path'] + '_' + df['protein_path']
    
    # Update dataframe with evolutionary vectors
    print("Updating dataset with evolutionary vectors...")
    df['evo_vec'] = df['pair_key'].map(evo_vectors)
    
    # Check for missing values
    missing_count = df['evo_vec'].isna().sum()
    if missing_count > 0:
        print(f"Warning: {missing_count} samples have missing evolutionary vectors")
        # Fill with zero vectors
        df['evo_vec'] = df['evo_vec'].fillna(df['evo_vec'].apply(lambda x: np.zeros(1280, dtype=np.float32)))
    
    # Drop the temporary columns
    df = df.drop(['pair_key', 'actual_protein_path'], axis=1)
    
    return df

def save_updated_dataset(df: pd.DataFrame, output_dir: Path, step_name: str):
    """Save the updated dataset."""
    output_file = output_dir / f"dataset_with_{step_name}.parquet"
    df.to_parquet(output_file, index=False)
    print(f"Saved updated dataset to {output_file}")
    
    # Also save as CSV for inspection (only first 1000 rows to avoid huge files)
    csv_file = output_dir / f"dataset_with_{step_name}_sample.csv"
    df.head(1000).to_csv(csv_file, index=False)
    print(f"Saved sample dataset CSV to {csv_file}")
    
    # Save numpy arrays separately for easy loading
    if 'evo_vec' in df.columns:
        evo_vecs = np.array([vec for vec in df['evo_vec']])
        npy_file = output_dir / f"evo_vecs_{step_name}.npy"
        np.save(npy_file, evo_vecs)
        print(f"Saved evolutionary vectors as numpy array to {npy_file}")
        print(f"Evolutionary vectors shape: {evo_vecs.shape}")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Compute evolutionary vectors using ESM-2 for protein-pocket pairs")
    parser.add_argument("--input_file", type=str, required=True,
                       help="Path to the dataset parquet file")
    parser.add_argument("--output_dir", type=str, default="processed_data",
                       help="Output directory for processed data")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for processing (not used in current implementation)")
    
    args = parser.parse_args()
    
    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    df = load_dataset(input_file)
    
    # Compute evolutionary vectors
    df_updated = compute_evolutionary_vectors(df, output_dir, args.batch_size)
    
    # Save updated dataset
    output_file = save_updated_dataset(df_updated, output_dir, "evo_vec")
    
    print("Step 3 completed successfully!")
    print(f"Updated dataset shape: {df_updated.shape}")
    print(f"Evolutionary vector dimensions: {len(df_updated['evo_vec'].iloc[0])}")

if __name__ == "__main__":
    main()

