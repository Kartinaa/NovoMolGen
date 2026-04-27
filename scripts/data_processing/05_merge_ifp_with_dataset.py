#!/usr/bin/env python3
"""
Step 5: Merge IFP matrix with dataset
This script merges the IFP matrix file with the original dataset parquet.

Memory-efficient approach:
- Uses memmap to read IFP matrix (no full load)
- Processes in chunks to stay within memory limits
- Option to keep IFP separate (recommended) or embed in parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import sys
from typing import Optional

# Constants
IFP_DIM = 16384


def load_ifp_matrix(matrix_file: Path, packed_bits: bool = False, num_rows: Optional[int] = None):
    """
    Load IFP matrix as memmap (read-only, memory-efficient).
    
    Args:
        matrix_file: Path to the .npy matrix file
        packed_bits: Whether matrix uses packed bits
        num_rows: Number of rows (if None, will calculate from file size)
    
    Returns:
        memmap array and shape info
    """
    if not matrix_file.exists():
        raise FileNotFoundError(f"IFP matrix file not found: {matrix_file}")
    
    # Calculate number of rows from file size if not provided
    if num_rows is None:
        file_size = matrix_file.stat().st_size
        
        if packed_bits:
            cols = IFP_DIM // 8
            element_size = np.dtype(np.uint8).itemsize
            row_size = cols * element_size
        else:
            element_size = np.dtype(np.float32).itemsize
            row_size = IFP_DIM * element_size
        
        # Use integer division to avoid overflow
        num_rows = int(file_size // row_size)
        
        print(f"File size: {file_size / 1e9:.2f} GB")
        print(f"Row size: {row_size} bytes")
        print(f"Calculated matrix size: {num_rows} rows")
    
    # Create memmap with explicit shape
    if packed_bits:
        cols = IFP_DIM // 8
        X = np.memmap(matrix_file, mode="r", dtype=np.uint8, shape=(num_rows, cols))
    else:
        X = np.memmap(matrix_file, mode="r", dtype=np.float32, shape=(num_rows, IFP_DIM))
    
    print(f"Loaded IFP matrix: {matrix_file}")
    print(f"  Shape: {X.shape}")
    print(f"  Dtype: {X.dtype}")
    print(f"  File size: {matrix_file.stat().st_size / 1e9:.2f} GB")
    
    return X


def merge_ifp_separate_mode(
    dataset_file: Path,
    ifp_matrix_file: Path,
    ifp_meta_file: Optional[Path],
    output_file: Path,
    packed_bits: bool = False,
):
    """
    Merge mode: Keep IFP separate, add ifp_row_idx to dataset.
    This is the recommended mode for large datasets.
    """
    print("=" * 60)
    print("Merging in SEPARATE mode (IFP stays in external matrix)")
    print("=" * 60)
    
    # Load dataset
    print(f"\nLoading dataset from {dataset_file}...")
    df = pd.read_parquet(dataset_file)
    print(f"Loaded {len(df)} rows")
    
    # Load IFP meta if exists (should have ifp_row_idx)
    if ifp_meta_file and ifp_meta_file.exists():
        print(f"\nLoading IFP metadata from {ifp_meta_file}...")
        df_ifp_meta = pd.read_parquet(ifp_meta_file)
        print(f"IFP metadata has {len(df_ifp_meta)} rows")
        
        # Merge on common columns (protein_path, ligand_path)
        merge_cols = ['protein_path', 'ligand_path']
        if all(col in df.columns and col in df_ifp_meta.columns for col in merge_cols):
            print(f"Merging on columns: {merge_cols}")
            df = df.merge(
                df_ifp_meta[merge_cols + ['ifp_row_idx']],
                on=merge_cols,
                how='left',
                validate='one_to_one'
            )
            
            # Check for missing ifp_row_idx
            missing = df['ifp_row_idx'].isna().sum()
            if missing > 0:
                print(f"⚠️  Warning: {missing} rows have missing ifp_row_idx")
        else:
            print("⚠️  Warning: Cannot merge - missing common columns")
            # Add sequential ifp_row_idx if not present
            if 'ifp_row_idx' not in df.columns:
                df['ifp_row_idx'] = np.arange(len(df), dtype=np.int64)
                print("Added sequential ifp_row_idx")
    else:
        # Add sequential ifp_row_idx if not present
        if 'ifp_row_idx' not in df.columns:
            print("Adding sequential ifp_row_idx...")
            df['ifp_row_idx'] = np.arange(len(df), dtype=np.int64)
    
    # Load IFP matrix with explicit row count
    print(f"\nLoading IFP matrix...")
    ifp_matrix = load_ifp_matrix(ifp_matrix_file, packed_bits, num_rows=len(df))
    
    # Verify IFP matrix size matches
    if len(ifp_matrix) != len(df):
        print(f"⚠️  Warning: IFP matrix has {len(ifp_matrix)} rows, dataset has {len(df)} rows")
        print("  This might indicate a mismatch. Proceeding anyway...")
    
    # Save merged dataset
    print(f"\nSaving merged dataset to {output_file}...")
    df.to_parquet(output_file, index=False)
    print(f"✓ Saved merged dataset")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  IFP matrix: {ifp_matrix_file} (external, {ifp_matrix_file.stat().st_size / 1e9:.2f} GB)")
    
    return df, ifp_matrix_file


def merge_ifp_embed_mode(
    dataset_file: Path,
    ifp_matrix_file: Path,
    ifp_meta_file: Optional[Path],
    output_file: Path,
    packed_bits: bool = False,
    chunk_size: int = 1000,
):
    """
    Merge mode: Embed IFP vectors into parquet (WARNING: creates large file).
    Only use if you have enough disk space and need all data in one file.
    """
    print("=" * 60)
    print("Merging in EMBED mode (IFP vectors embedded in parquet)")
    print("⚠️  WARNING: This will create a very large parquet file!")
    print("=" * 60)
    
    # Load dataset
    print(f"\nLoading dataset from {dataset_file}...")
    df = pd.read_parquet(dataset_file)
    print(f"Loaded {len(df)} rows")
    
    # Load IFP matrix as memmap with explicit row count
    print(f"\nLoading IFP matrix...")
    ifp_matrix = load_ifp_matrix(ifp_matrix_file, packed_bits, num_rows=len(df))
    
    # Verify sizes match
    if len(ifp_matrix) != len(df):
        raise ValueError(
            f"Size mismatch: IFP matrix has {len(ifp_matrix)} rows, "
            f"dataset has {len(df)} rows"
        )
    
    # Process in chunks to avoid memory issues
    print(f"\nEmbedding IFP vectors in chunks of {chunk_size}...")
    ifp_vectors_list = []
    
    for start in tqdm(range(0, len(df), chunk_size), desc="Loading IFP chunks"):
        end = min(len(df), start + chunk_size)
        chunk = ifp_matrix[start:end]
        
        if packed_bits:
            # Unpack bits for this chunk
            unpacked_chunk = []
            for row in chunk:
                bits = np.unpackbits(row, bitorder="little")
                unpacked_chunk.append(bits[:IFP_DIM].astype(np.float32))
            ifp_vectors_list.extend(unpacked_chunk)
        else:
            # Convert to list of arrays
            for row in chunk:
                ifp_vectors_list.append(np.array(row, copy=True))
    
    # Add IFP vectors to dataframe
    print("Adding IFP vectors to dataframe...")
    df['ifp'] = ifp_vectors_list
    
    # Save merged dataset
    print(f"\nSaving merged dataset to {output_file}...")
    print("⚠️  This may take a while and create a large file...")
    df.to_parquet(output_file, index=False)
    
    file_size_gb = output_file.stat().st_size / 1e9
    print(f"✓ Saved merged dataset")
    print(f"  Rows: {len(df)}")
    print(f"  File size: {file_size_gb:.2f} GB")
    print(f"  Columns: {list(df.columns)}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Merge IFP matrix with dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Separate mode (recommended - keeps IFP in external file)
  python 05_merge_ifp_with_dataset.py \\
    --dataset finetune_data/processed_data/dataset_with_ligand_vec.parquet \\
    --ifp_matrix finetune_data/processed_data/ifp_float32.npy \\
    --output finetune_data/processed_data/dataset_with_ifp.parquet \\
    --mode separate
  
  # Embed mode (WARNING: creates very large file)
  python 05_merge_ifp_with_dataset.py \\
    --dataset finetune_data/processed_data/dataset_with_ligand_vec.parquet \\
    --ifp_matrix finetune_data/processed_data/ifp_float32.npy \\
    --output finetune_data/processed_data/dataset_with_ifp_embedded.parquet \\
    --mode embed
        """
    )
    parser.add_argument("--dataset", type=str, required=True,
                       help="Path to dataset parquet file")
    parser.add_argument("--ifp_matrix", type=str, required=True,
                       help="Path to IFP matrix .npy file")
    parser.add_argument("--ifp_meta", type=str, default=None,
                       help="Path to IFP metadata parquet (optional, if available)")
    parser.add_argument("--output", type=str, required=True,
                       help="Output parquet file path")
    parser.add_argument("--mode", type=str, default="separate",
                       choices=["separate", "embed"],
                       help="Merge mode: 'separate' (recommended) or 'embed' (creates large file)")
    parser.add_argument("--packed_bits", action="store_true",
                       help="IFP matrix uses packed bits format")
    parser.add_argument("--chunk_size", type=int, default=1000,
                       help="Chunk size for embed mode (default: 1000)")
    
    args = parser.parse_args()
    
    dataset_file = Path(args.dataset)
    ifp_matrix_file = Path(args.ifp_matrix)
    ifp_meta_file = Path(args.ifp_meta) if args.ifp_meta else None
    output_file = Path(args.output)
    
    # Verify input files exist
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
    if not ifp_matrix_file.exists():
        raise FileNotFoundError(f"IFP matrix file not found: {ifp_matrix_file}")
    
    # Create output directory
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Merge based on mode
    if args.mode == "separate":
        df, matrix_path = merge_ifp_separate_mode(
            dataset_file,
            ifp_matrix_file,
            ifp_meta_file,
            output_file,
            args.packed_bits,
        )
        print("\n" + "=" * 60)
        print("✅ Merge completed successfully!")
        print("=" * 60)
        print(f"📊 Summary:")
        print(f"   Output file: {output_file}")
        print(f"   IFP matrix: {matrix_path} (external)")
        print(f"   Total rows: {len(df)}")
        print(f"   Mode: Separate (IFP loaded on-demand via ifp_row_idx)")
        print(f"\n💡 To load IFP for a row:")
        print(f"   from scripts.data_processing.04_1_compute_ifp import load_ifp_row")
        print(f"   ifp = load_ifp_row('{matrix_path}', row_idx, packed_bits={args.packed_bits})")
        
    else:  # embed mode
        df = merge_ifp_embed_mode(
            dataset_file,
            ifp_matrix_file,
            ifp_meta_file,
            output_file,
            args.packed_bits,
            args.chunk_size,
        )
        print("\n" + "=" * 60)
        print("✅ Merge completed successfully!")
        print("=" * 60)
        print(f"📊 Summary:")
        print(f"   Output file: {output_file}")
        print(f"   File size: {output_file.stat().st_size / 1e9:.2f} GB")
        print(f"   Total rows: {len(df)}")
        print(f"   Mode: Embedded (all IFP vectors in parquet)")


if __name__ == "__main__":
    main()

