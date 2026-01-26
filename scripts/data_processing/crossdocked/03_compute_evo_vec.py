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
import re
import urllib.request
import time

# Add src and utils to path for imports
# Add only the utils package first to avoid colliding with src/utils.py
# Script is at: scripts/data_processing/crossdocked/03_compute_evo_vec.py
# Need to go up 3 levels to reach project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
utils_path = str(PROJECT_ROOT / "utils")
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from esm_embedding_extract import ESM2PocketEmbedder  # ← 来自 utils/esm_embedding_extract.py



def load_dataset(input_file: Path):
    """Load the dataset."""
    print(f"Loading dataset from {input_file}")
    df = pd.read_parquet(input_file)
    print(f"Loaded {len(df)} samples")
    return df

def extract_pdb_id_from_path(pocket_path: str) -> str:
    """Extract PDB ID from pocket PDB file path.
    
    Example: 
        /path/to/1m4n_A_rec_1m7y_ppg_lig_tt_min_0_pocket10.pdb -> 1m4n
    """
    pocket_path = Path(pocket_path)
    filename = pocket_path.stem  # Get filename without extension
    # Extract first 4 characters (PDB ID is always 4 characters)
    # PDB IDs are alphanumeric, typically starting with a digit
    match = re.match(r'^([0-9a-zA-Z]{4})', filename)
    if match:
        return match.group(1).upper()
    else:
        raise ValueError(f"Could not extract PDB ID from filename: {filename}")

def download_pdb_file(pdb_id: str, output_dir: Path, max_retries: int = 3) -> Path:
    """Download PDB file from RCSB PDB database.
    
    Args:
        pdb_id: 4-character PDB ID (e.g., '1M4N')
        output_dir: Directory to save the downloaded PDB file
        max_retries: Maximum number of retry attempts
    
    Returns:
        Path to the downloaded PDB file
    """
    pdb_id = pdb_id.upper()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{pdb_id}.pdb"
    
    # If file already exists, return it
    if output_file.exists():
        return output_file
    
    # RCSB PDB download URL
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    
    for attempt in range(max_retries):
        try:
            urllib.request.urlretrieve(url, output_file)
            
            # Verify file was downloaded (should be > 0 bytes)
            if output_file.stat().st_size > 0:
                return output_file
            else:
                output_file.unlink()  # Remove empty file
                
        except Exception as e:
            if output_file.exists():
                output_file.unlink()  # Remove partial file
            
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                time.sleep(wait_time)
    
    raise FileNotFoundError(f"Failed to download {pdb_id}.pdb after {max_retries} attempts")

def get_protein_path(pocket_path: str, download_if_missing: bool = True) -> str:
    """Get full protein PDB path, downloading if necessary.
    
    First tries to find protein.pdb in the same directory as pocket_path.
    If not found, extracts PDB ID from pocket filename and looks for {PDB_ID}.pdb.
    If still not found and download_if_missing=True, downloads from RCSB PDB.
    
    Args:
        pocket_path: Path to pocket PDB file
        download_if_missing: Whether to download PDB file if not found locally
    
    Returns:
        Path to full protein PDB file (may not exist if download failed)
    """
    pocket_path = Path(pocket_path)
    
    # First, try to find protein.pdb in the same directory
    local_protein_path = pocket_path.parent / "protein.pdb"
    if local_protein_path.exists():
        return str(local_protein_path)
    
    # If not found, extract PDB ID and look for {PDB_ID}.pdb in the same directory
    try:
        pdb_id = extract_pdb_id_from_path(str(pocket_path))
        pdb_file = pocket_path.parent / f"{pdb_id.upper()}.pdb"
        
        if pdb_file.exists():
            return str(pdb_file)
        elif download_if_missing:
            # Try to download
            try:
                print(f"📥 Downloading {pdb_id}.pdb from RCSB PDB...")
                downloaded_file = download_pdb_file(pdb_id, pocket_path.parent)
                print(f"✅ Downloaded {pdb_id}.pdb to {downloaded_file}")
                return str(downloaded_file)
            except Exception as e:
                print(f"⚠️  Failed to download {pdb_id}.pdb: {e}")
                return str(pdb_file)  # Return expected path even if download failed
        else:
            # Return the expected path even if it doesn't exist (will be handled by error handling)
            return str(pdb_file)
        
    except Exception as e:
        print(f"⚠️  Error extracting PDB ID from {pocket_path}: {e}")
        # Fallback: return a non-existent path (will be handled by error handling)
        return str(local_protein_path)

def compute_evolutionary_vectors(df: pd.DataFrame, output_dir: Path, batch_size: int = 32):
    """Compute evolutionary vectors for all proteins using ESM2PocketEmbedder.
    
    Args:
        df: DataFrame with 'protein_path' column (pocket PDB paths)
        output_dir: Output directory for results
        batch_size: Batch size (not used in current implementation)
    """
    print("Computing evolutionary vectors...")
    
    # Add actual protein_path column (will download PDB files if needed)
    print("Extracting PDB IDs and preparing protein paths...")
    df['actual_protein_path'] = df['protein_path'].apply(lambda x: get_protein_path(x, download_if_missing=True))
    
    # Get unique protein-pocket pairs using actual protein paths
    unique_pairs = df[['actual_protein_path', 'protein_path']].drop_duplicates()
    unique_pairs = unique_pairs.rename(columns={'actual_protein_path': 'protein_path', 'protein_path': 'pocket_path'})
    print(f"Found {len(unique_pairs)} unique protein-pocket pairs")
    
    # Initialize evolutionary vector storage
    evo_vectors = {}
    
    # Statistics tracking
    stats = {
        'total': len(unique_pairs),
        'success': 0,
        'zero_vector': 0,
        'missing_file': 0,
        'chain_error': 0,
        'other_error': 0
    }
    
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
            protein_path = row['protein_path']  # This is now the actual protein.pdb path
            pocket_path = row['pocket_path']    # This is the original protein_path column (pocket path)
            pair_key = f"{protein_path}_{pocket_path}"
            
            try:
                # Check if protein file exists
                if not Path(protein_path).exists():
                    stats['missing_file'] += 1
                    evo_vectors[pair_key] = np.zeros(1280, dtype=np.float32)
                    continue
                
                # Try to extract chain ID from pocket filename
                # Example: 3iwy_C_rec_... -> chain_id = 'C'
                chain_id = None
                pocket_filename = Path(pocket_path).stem
                # Look for pattern like "_C_rec_" or "_A_rec_" in filename
                chain_match = re.search(r'_([A-Z])_rec_', pocket_filename)
                if chain_match:
                    chain_id = chain_match.group(1)
                
                # Try with detected chain_id, or None (which will use first chain)
                success = False
                last_error = None
                
                # First try with detected chain_id (or None)
                try:
                    pocket_res_list, pocket_emb = embedder.pocket_embeddings(
                        full_pdb_path=protein_path,
                        pocket_pdb_path=pocket_path,
                        chain_id=chain_id
                    )
                    success = True
                except (ValueError, KeyError) as e:
                    last_error = e
                    # If chain_id was specified but failed, try without chain_id (first chain)
                    if chain_id is not None:
                        try:
                pocket_res_list, pocket_emb = embedder.pocket_embeddings(
                    full_pdb_path=protein_path,
                    pocket_pdb_path=pocket_path,
                    chain_id=None
                )
                            success = True
                        except (ValueError, KeyError) as e2:
                            last_error = e2
                
                if not success:
                    raise last_error
                
                # Compute mean embedding as the single representative vector
                pocket_vec = pocket_emb.mean(dim=0)  # [H]
                evo_vectors[pair_key] = pocket_vec.detach().cpu().numpy()  # Convert to numpy array
                stats['success'] += 1
                
                if idx % 100 == 0:  # Print progress every 100 proteins
                    print(f"Processed {idx+1}/{len(unique_pairs)}: {stats['success']} success, {stats['zero_vector']} failed")
                
            except Exception as e:
                error_msg = str(e)
                stats['zero_vector'] += 1
                
                # Categorize errors
                if "'" in error_msg:
                    stats['chain_error'] += 1
                elif "No standard amino acids" in error_msg:
                    stats['chain_error'] += 1
                else:
                    stats['other_error'] += 1
                
                # Only print detailed error for non-standard cases
                # Skip common errors that are expected for some PDB files
                if "No standard amino acids" not in error_msg and "'" not in error_msg:
                    # The "'B'" type errors are KeyError for missing chains, which is common
                    print(f"❌ Error processing {pair_key}: {e}")
                # Use zero vector as fallback
                evo_vectors[pair_key] = np.zeros(1280, dtype=np.float32)
    
    except ImportError as e:
        print(f"Warning: ESM-2 tools not available ({e}), using random vectors as placeholder")
        for idx, row in tqdm(unique_pairs.iterrows(), total=len(unique_pairs), desc="Creating placeholder evolutionary vectors"):
            protein_path = row['protein_path']  # This is now the actual protein.pdb path
            pocket_path = row['pocket_path']    # This is the original protein_path column (pocket path)
            pair_key = f"{protein_path}_{pocket_path}"
            # Create random vector as placeholder
            evo_vectors[pair_key] = np.random.randn(1280).astype(np.float32)
        stats['zero_vector'] = len(unique_pairs)  # All are placeholders
        stats['success'] = 0
    
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
    
    # Print statistics
    print("\n" + "=" * 60)
    print("📊 Processing Statistics:")
    print("=" * 60)
    print(f"Total unique protein-pocket pairs: {stats['total']}")
    print(f"✅ Successfully processed: {stats['success']} ({stats['success']/stats['total']*100:.2f}%)")
    print(f"❌ Failed (using zero vector): {stats['zero_vector']} ({stats['zero_vector']/stats['total']*100:.2f}%)")
    print(f"   - Missing PDB files: {stats['missing_file']}")
    print(f"   - Chain errors (missing chain/no amino acids): {stats['chain_error']}")
    print(f"   - Other errors: {stats['other_error']}")
    print("=" * 60 + "\n")
    
    return df

def save_updated_dataset(df: pd.DataFrame, output_dir: Path, step_name: str, input_file: Path = None):
    """Save the updated dataset.
    
    Args:
        df: DataFrame to save
        output_dir: Output directory
        step_name: Step name (e.g., 'evo_vec')
        input_file: Input file path (used to extract split name if present)
    """
    # Try to extract split name from input filename (e.g., dataset_with_pocket_vec_test.parquet -> test)
    split_suffix = ""
    if input_file:
        input_stem = Path(input_file).stem
        if "_test" in input_stem:
            split_suffix = "_test"
        elif "_train" in input_stem:
            split_suffix = "_train"
    
    output_file = output_dir / f"dataset_with_{step_name}{split_suffix}.parquet"
    df.to_parquet(output_file, index=False)
    print(f"Saved updated dataset to {output_file}")
    
    # Also save as CSV for inspection (only first 1000 rows to avoid huge files)
    csv_file = output_dir / f"dataset_with_{step_name}{split_suffix}_sample.csv"
    df.head(1000).to_csv(csv_file, index=False)
    print(f"Saved sample dataset CSV to {csv_file}")
    
    # Save numpy arrays separately for easy loading
    if 'evo_vec' in df.columns:
        evo_vecs = np.array([vec for vec in df['evo_vec']])
        npy_file = output_dir / f"evo_vecs_{step_name}{split_suffix}.npy"
        np.save(npy_file, evo_vecs)
        print(f"Saved evolutionary vectors as numpy array to {npy_file}")
        print(f"Evolutionary vectors shape: {evo_vecs.shape}")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Compute evolutionary vectors using ESM-2 for protein-pocket pairs")
    parser.add_argument("--input_file", type=str, required=True,
                       help="Path to the dataset parquet file")
    parser.add_argument("--output_dir", type=str, default="finetune_data/crossdock_dataset",
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
    output_file = save_updated_dataset(df_updated, output_dir, "evo_vec", input_file=input_file)
    
    print("Step 3 completed successfully!")
    print(f"Updated dataset shape: {df_updated.shape}")
    print(f"Evolutionary vector dimensions: {len(df_updated['evo_vec'].iloc[0])}")

if __name__ == "__main__":
    main()

