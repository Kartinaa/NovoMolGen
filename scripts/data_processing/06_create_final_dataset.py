#!/usr/bin/env python3
"""
Step 6: Create final dataset for HuggingFace upload
This script creates the final dataset in HuggingFace datasets format.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import torch
from datasets import Dataset, DatasetDict
import json

def load_final_dataset(input_file: Path):
    """Load the final dataset with all condition features."""
    print(f"Loading final dataset from {input_file}")
    df = pd.read_parquet(input_file)
    print(f"Loaded {len(df)} samples")
    return df

def create_huggingface_dataset(df: pd.DataFrame, output_dir: Path):
    """Create HuggingFace dataset format."""
    print("Creating HuggingFace dataset format...")
    
    # Convert to HuggingFace datasets format
    train_df = df[df['split'] == 'train'].copy()
    val_df = df[df['split'] == 'validation'].copy()
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    # Create datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    # Create dataset dictionary
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })
    
    # Save dataset
    dataset_path = output_dir / "final_dataset"
    dataset_dict.save_to_disk(str(dataset_path))
    print(f"Saved HuggingFace dataset to {dataset_path}")
    
    return dataset_dict

def create_metadata_file(df: pd.DataFrame, output_dir: Path):
    """Create metadata file with dataset information."""
    metadata = {
        "dataset_name": "bdnv2_finetune_dataset",
        "description": "BDNV2 dataset with condition features for NovoMolGen finetuning",
        "total_samples": len(df),
        "training_samples": len(df[df['split'] == 'train']),
        "validation_samples": len(df[df['split'] == 'validation']),
        "features": {
            "standardize_smi": "Standardized SMILES string",
            "pocket_vec": "Uni-Mol pocket embedding (512 dimensions)",
            "evo_vec": "ESM-2 evolutionary embedding (1280 dimensions)",
            "ifp": "Interaction fingerprint (16384 dimensions)",
            "ligand_vec": "Ligand molecular representation (1536 dimensions)"
        },
        "condition_feature_dimensions": {
            "pocket_vec": 512,
            "evo_vec": 1280,
            "ifp": 16384,
            "ligand_vec": 1536
        }
    }
    
    metadata_file = output_dir / "dataset_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_file}")
    
    return metadata

def create_sample_data(df: pd.DataFrame, output_dir: Path, num_samples: int = 5):
    """Create sample data for inspection."""
    print(f"Creating sample data with {num_samples} samples...")
    
    # Sample from both splits
    train_samples = df[df['split'] == 'train'].head(num_samples)
    val_samples = df[df['split'] == 'validation'].head(num_samples)
    
    sample_data = {
        "training_samples": train_samples.to_dict('records'),
        "validation_samples": val_samples.to_dict('records')
    }
    
    sample_file = output_dir / "sample_data.json"
    with open(sample_file, 'w') as f:
        json.dump(sample_data, f, indent=2)
    print(f"Saved sample data to {sample_file}")

def main():
    parser = argparse.ArgumentParser(description="Create final dataset for HuggingFace upload")
    parser.add_argument("--input_file", type=str, required=True,
                       help="Path to the final dataset parquet file")
    parser.add_argument("--output_dir", type=str, default="processed_data",
                       help="Output directory for processed data")
    parser.add_argument("--num_samples", type=int, default=5,
                       help="Number of samples to include in sample data")
    
    args = parser.parse_args()
    
    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load final dataset
    df = load_final_dataset(input_file)
    
    # Create HuggingFace dataset
    dataset_dict = create_huggingface_dataset(df, output_dir)
    
    # Create metadata
    metadata = create_metadata_file(df, output_dir)
    
    # Create sample data
    create_sample_data(df, output_dir, args.num_samples)
    
    print("Step 6 completed successfully!")
    print(f"Final dataset created with {len(df)} samples")
    print(f"Training samples: {len(df[df['split'] == 'train'])}")
    print(f"Validation samples: {len(df[df['split'] == 'validation'])}")
    print(f"Dataset saved to: {output_dir / 'final_dataset'}")

if __name__ == "__main__":
    main()

