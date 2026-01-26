#!/usr/bin/env python3
"""
Create base dataset for CrossDocked from split_by_name.pt file.
The .pt file contains a dict with 'train' and 'test' keys, where values are
lists of tuples (receptor_relative_path, ligand_relative_path).
"""

import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import torch

from rdkit import Chem

def sdf_to_smiles(sdf_path: Path) -> str | None:
    """Extract SMILES from SDF file (without standardization)."""
    if not sdf_path.exists():
        return None
    
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    mol = next(iter(suppl), None)
    
    if mol is None:
        return None
    
    try:
        return Chem.MolToSmiles(mol)
    except Exception as e:
        print(f"Error extracting SMILES from {sdf_path}: {e}")
        return None

def load_split_data(pt_file: Path, split: str = 'train'):
    """Load split data from .pt file.
    
    Args:
        pt_file: Path to .pt file
        split: Which split to load ('train' or 'test')
    """
    print(f"Loading split data from {pt_file}")
    data = torch.load(pt_file, map_location='cpu')
    
    if split not in data:
        raise ValueError(f"'{split}' key not found in {pt_file}. Available keys: {list(data.keys())}")
    
    split_data = data[split]
    print(f"Found {len(split_data)} {split} samples")
    
    return split_data

def build_base_df_from_pt(pt_file: Path, base_dir: Path, split: str = 'train'):
    """Build base dataframe from .pt file.
    
    Args:
        pt_file: Path to .pt file
        base_dir: Base directory for file paths
        split: Which split to process ('train' or 'test')
    """
    # Load split data
    split_data = load_split_data(pt_file, split=split)
    
    rows = []
    skipped = 0
    
    for i, (receptor_rel_path, ligand_rel_path) in enumerate(tqdm(split_data, desc=f"Processing {split} samples")):
        # Build absolute paths
        receptor_path = base_dir / receptor_rel_path
        ligand_path = base_dir / ligand_rel_path
        
        # Check if files exist
        if not receptor_path.exists():
            print(f"Warning: Receptor file not found: {receptor_path}, skip")
            skipped += 1
            continue
        
        if not ligand_path.exists():
            print(f"Warning: Ligand file not found: {ligand_path}, skip")
            skipped += 1
            continue

        # Extract SMILES from ligand
        smi = sdf_to_smiles(ligand_path)
        if smi is None:
            print(f"Warning: cannot extract SMILES from {ligand_path}, skip")
            skipped += 1
            continue

        rows.append({
            "id": f"crossdock_{split}_{i}",
            "split": split,
            "protein_path": str(receptor_path.resolve()),  # Absolute path to pocket PDB
            "ligand_path": str(ligand_path.resolve()),     # Absolute path to ligand SDF
            "standardize_smi": smi,  # Note: column name kept for compatibility, but value is not standardized
            "pocket_vec": None,  # Will be computed later
            "evo_vec": None,    # Will be computed later
            "ifp": None,        # Will be computed later
            "ligand_vec": None, # Will be computed later
        })
    
    df = pd.DataFrame(rows)
    print(f"Built base dataframe with {len(df)} rows (skipped {skipped} invalid samples)")
    return df

def main():
    parser = argparse.ArgumentParser(description="Create base dataset from CrossDocked split_by_name.pt file")
    parser.add_argument(
        "--pt_file",
        type=str,
        required=True,
        help="Path to split_by_name.pt file",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/home/yang2531/Documents/Project/Structure_safe/finetune_data/crossdock_dataset/crossdocked_pocket10",
        help="Base directory to prepend to relative paths",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="finetune_data/crossdock_dataset",
        help="Where to save base_dataset.parquet",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Which split to process (default: train)",
    )
    args = parser.parse_args()

    pt_file = Path(args.pt_file)
    if not pt_file.exists():
        raise FileNotFoundError(f"PT file not found: {pt_file}")
    
    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build base dataframe
    df = build_base_df_from_pt(pt_file, base_dir, split=args.split)
    
    # Save as Parquet (with split name in filename)
    out_path = out_dir / f"base_dataset_{args.split}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"✅ Saved {args.split} dataset to {out_path}")
    
    # Also save as CSV for inspection (first 1000 rows)
    csv_path = out_dir / f"base_dataset_{args.split}_sample.csv"
    df.head(1000).to_csv(csv_path, index=False)
    print(f"✅ Saved sample CSV to {csv_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("Dataset Summary:")
    print(f"  Total samples: {len(df)}")
    print(f"  Split: {df['split'].value_counts().to_dict()}")
    print(f"  Columns: {df.columns.tolist()}")
    print("="*50)

if __name__ == "__main__":
    main()