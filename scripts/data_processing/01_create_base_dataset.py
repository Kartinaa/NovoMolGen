#!/usr/bin/env python3
"""
Step 1: Create base dataset with SMILES and metadata
This script loads the pickle files and CSV, creates a base Parquet dataset with all necessary metadata.
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

def load_data_splits(data_dir: Path):
    """Load training and validation data splits."""
    print("Loading data splits...")
    
    # Load training data
    with open(data_dir / "training_set.pkl", 'rb') as f:
        training_data = pickle.load(f)
    
    # Load validation data  
    with open(data_dir / "validation_set.pkl", 'rb') as f:
        validation_data = pickle.load(f)
    
    print(f"Training samples: {len(training_data)}")
    print(f"Validation samples: {len(validation_data)}")
    
    return training_data, validation_data

def load_standardize_mapping(data_dir: Path):
    """Load the standardize SMILES mapping."""
    print("Loading standardize SMILES mapping...")
    
    df = pd.read_csv(data_dir / "bdnv2_standardize.csv")
    print(f"Standardize mapping shape: {df.shape}")
    
    # Create a lookup dictionary for faster access
    mapping = {}
    for _, row in df.iterrows():
        key = (row['protein_path'], row['ligand_path'])
        mapping[key] = row['standardize_smi']
    
    print(f"Created mapping for {len(mapping)} protein-ligand pairs")
    return mapping

def create_base_dataset(training_data, validation_data, standardize_mapping, output_dir: Path):
    """Create base dataset with all metadata."""
    print("Creating base dataset...")
    
    all_data = []
    
    # Process training data
    print("Processing training data...")
    for i, (protein_path, ligand_path) in enumerate(tqdm(training_data)):
        # Get standardize SMILES
        key = (protein_path, ligand_path)
        if key in standardize_mapping:
            standardize_smi = standardize_mapping[key]
        else:
            print(f"Warning: No standardize SMILES found for {key}")
            continue
            
        all_data.append({
            'id': f"train_{i}",
            'split': 'train',
            'protein_path': protein_path,
            'ligand_path': ligand_path,
            'standardize_smi': standardize_smi,
            'pocket_vec': None,  # Will be computed later
            'evo_vec': None,     # Will be computed later
            'ifp': None,         # Will be computed later
            'ligand_vec': None,  # Will be computed later
        })
    
    # Process validation data
    print("Processing validation data...")
    for i, (protein_path, ligand_path) in enumerate(tqdm(validation_data)):
        # Get standardize SMILES
        key = (protein_path, ligand_path)
        if key in standardize_mapping:
            standardize_smi = standardize_mapping[key]
        else:
            print(f"Warning: No standardize SMILES found for {key}")
            continue
            
        all_data.append({
            'id': f"val_{i}",
            'split': 'validation',
            'protein_path': protein_path,
            'ligand_path': ligand_path,
            'standardize_smi': standardize_smi,
            'pocket_vec': None,  # Will be computed later
            'evo_vec': None,     # Will be computed later
            'ifp': None,         # Will be computed later
            'ligand_vec': None,  # Will be computed later
        })
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    print(f"Created dataset with {len(df)} samples")
    print(f"Training samples: {len(df[df['split'] == 'train'])}")
    print(f"Validation samples: {len(df[df['split'] == 'validation'])}")
    
    # Save as Parquet
    output_file = output_dir / "base_dataset.parquet"
    df.to_parquet(output_file, index=False)
    print(f"Saved base dataset to {output_file}")
    
    # Also save as CSV for inspection
    csv_file = output_dir / "base_dataset.csv"
    df.to_csv(csv_file, index=False)
    print(f"Saved base dataset CSV to {csv_file}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Create base dataset with SMILES and metadata")
    parser.add_argument("--data_dir", type=str, default="finetune_data", 
                       help="Directory containing the pickle files and CSV")
    parser.add_argument("--output_dir", type=str, default="finetune_data/processed_data", 
                       help="Output directory for processed data")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    training_data, validation_data = load_data_splits(data_dir)
    standardize_mapping = load_standardize_mapping(data_dir)
    
    # Create base dataset
    df = create_base_dataset(training_data, validation_data, standardize_mapping, output_dir)
    
    print("Step 1 completed successfully!")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

if __name__ == "__main__":
    main()
