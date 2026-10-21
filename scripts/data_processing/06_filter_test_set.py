#!/usr/bin/env python3
"""
Filter test set from dataset
This script reads a CSV file with test set information and filters out
those entries from the parquet dataset based on protein_path matching.

CSV format:
- Target_ChEMBLID: e.g., 'CHEMBL5755'
- Molecule ChEMBLID: e.g., 'CHEMBL2017005'

Filters parquet rows where protein_path contains 'target_CHEMBLxxx/CHEMBLxx'
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm


def load_test_set_patterns(csv_file: Path) -> set:
    """
    Load test set patterns from CSV file.
    
    Returns:
        Set of patterns like 'target_CHEMBL5755/CHEMBL2017005'
    """
    print(f"Loading test set CSV from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    print(f"Loaded {len(df)} rows from CSV")
    print(f"Columns: {list(df.columns)}")
    
    # Check required columns
    target_col = None
    molecule_col = None
    
    # Try to find the columns (case-insensitive, handle spaces)
    for col in df.columns:
        col_lower = col.lower().strip()
        if 'target' in col_lower and 'chembl' in col_lower:
            target_col = col
        elif 'molecule' in col_lower and 'chembl' in col_lower:
            molecule_col = col
    
    if target_col is None or molecule_col is None:
        # Try exact matches
        if 'Target_ChEMBLID' in df.columns:
            target_col = 'Target_ChEMBLID'
        elif 'target_chemblid' in df.columns:
            target_col = 'target_chemblid'
        
        if 'Molecule ChEMBLID' in df.columns:
            molecule_col = 'Molecule ChEMBLID'
        elif 'molecule_chemblid' in df.columns:
            molecule_col = 'molecule_chemblid'
    
    if target_col is None:
        raise ValueError(
            f"Could not find Target_ChEMBLID column. Available columns: {list(df.columns)}"
        )
    if molecule_col is None:
        raise ValueError(
            f"Could not find Molecule ChEMBLID column. Available columns: {list(df.columns)}"
        )
    
    print(f"Using columns: '{target_col}' and '{molecule_col}'")
    
    # Create patterns
    patterns = set()
    skipped = 0
    
    for idx, row in df.iterrows():
        target_id = str(row[target_col]).strip()
        molecule_id = str(row[molecule_col]).strip()
        
        # Skip if either is NaN or empty
        if pd.isna(row[target_col]) or pd.isna(row[molecule_col]) or target_id == 'nan' or molecule_id == 'nan':
            skipped += 1
            continue
        
        # Create pattern: target_CHEMBLxxx/CHEMBLxx
        pattern = f"target_{target_id}/{molecule_id}"
        patterns.add(pattern)
    
    print(f"Created {len(patterns)} unique patterns")
    if skipped > 0:
        print(f"Skipped {skipped} rows with missing values")
    
    # Show first few patterns
    print(f"\nSample patterns (first 5):")
    for i, pattern in enumerate(list(patterns)[:5]):
        print(f"  {i+1}. {pattern}")
    
    return patterns


def filter_dataset(
    input_parquet: Path,
    output_parquet: Path,
    test_patterns: set,
    protein_path_column: str = "protein_path",
    save_filtered: Path | None = None,
):
    """
    Filter dataset by removing rows where protein_path contains any test pattern.
    
    Args:
        input_parquet: Input parquet file
        output_parquet: Output parquet file (filtered dataset)
        test_patterns: Set of patterns to filter out
        protein_path_column: Name of the column containing protein paths
        save_filtered: Optional path to save filtered-out rows (test set)
    
    Returns:
        df_filtered: Filtered dataframe
        original_count: Original row count
        filtered_count: Number of rows filtered out
        df_removed: Dataframe of removed rows (if save_filtered is provided)
    """
    print(f"\nLoading dataset from {input_parquet}...")
    df = pd.read_parquet(input_parquet)
    print(f"Loaded {len(df)} rows")
    
    if protein_path_column not in df.columns:
        raise ValueError(
            f"Column '{protein_path_column}' not found. Available columns: {list(df.columns)}"
        )
    
    print(f"\nFiltering rows where '{protein_path_column}' contains test set patterns...")
    
    # Convert protein_path to string for pattern matching
    protein_paths = df[protein_path_column].astype(str)
    
    # Create a mask for rows to keep (True = keep, False = filter out)
    # Initialize as all True (keep all rows)
    mask = pd.Series([True] * len(df), index=df.index)
    
    filtered_count = 0
    for pattern in tqdm(test_patterns, desc="Checking patterns"):
        # Check if protein_path contains this pattern
        contains_pattern = protein_paths.str.contains(pattern, regex=False, na=False)
        count = contains_pattern.sum()
        if count > 0:
            filtered_count += count
            # Mark matching rows as False (to filter out)
            mask = mask & ~contains_pattern
    
    # Apply filter
    df_filtered = df[mask].copy()
    df_removed = df[~mask].copy()  # Rows that were filtered out
    
    # Calculate actual filtered count
    actual_filtered = len(df_removed)
    
    print(f"\nFiltering results:")
    print(f"  Original rows: {len(df)}")
    print(f"  Filtered out: {actual_filtered} rows")
    print(f"  Remaining rows: {len(df_filtered)}")
    if len(df) > 0:
        print(f"  Reduction: {100 * (actual_filtered / len(df)):.2f}%")
    
    # Save filtered dataset (training set)
    print(f"\nSaving filtered dataset (training set) to {output_parquet}...")
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_parquet(output_parquet, index=False)
    print(f"✓ Saved filtered dataset ({len(df_filtered)} rows)")
    
    # Save removed rows (test set) if requested
    if save_filtered is not None:
        print(f"\nSaving removed rows (test set) to {save_filtered}...")
        save_filtered.parent.mkdir(parents=True, exist_ok=True)
        df_removed.to_parquet(save_filtered, index=False)
        print(f"✓ Saved test set ({len(df_removed)} rows)")
    
    return df_filtered, len(df), actual_filtered, df_removed


def main():
    parser = argparse.ArgumentParser(
        description="Filter test set from dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (only save training set)
  python 06_filter_test_set.py \\
    --csv finetune_data/training_test_spliting/test_set_info.csv \\
    --input finetune_data/processed_data/dataset_with_ifp.parquet \\
    --output finetune_data/processed_data/dataset_filtered.parquet
  
  # Save both training and test sets
  python 06_filter_test_set.py \\
    --csv finetune_data/training_test_spliting/test_set_info.csv \\
    --input finetune_data/processed_data/dataset_with_ifp.parquet \\
    --output finetune_data/processed_data/dataset_train.parquet \\
    --save_filtered finetune_data/processed_data/dataset_test.parquet
  
  # Custom column name
  python 06_filter_test_set.py \\
    --csv test_set_info.csv \\
    --input dataset.parquet \\
    --output dataset_train.parquet \\
    --save_filtered dataset_test.parquet \\
    --protein_col "protein_path"
        """
    )
    parser.add_argument("--csv", type=str, required=True,
                       help="Path to CSV file with test set information")
    parser.add_argument("--input", type=str, required=True,
                       help="Path to input parquet file")
    parser.add_argument("--output", type=str, required=True,
                       help="Path to output parquet file (filtered)")
    parser.add_argument("--protein_col", type=str, default="protein_path",
                       help="Name of the column containing protein paths (default: protein_path)")
    parser.add_argument("--save_filtered", type=str, default=None,
                       help="Optional: Path to save filtered-out rows (test set) as parquet")
    
    args = parser.parse_args()
    
    csv_file = Path(args.csv)
    input_parquet = Path(args.input)
    output_parquet = Path(args.output)
    save_filtered = Path(args.save_filtered) if args.save_filtered else None
    
    # Verify input files exist
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    if not input_parquet.exists():
        raise FileNotFoundError(f"Input parquet file not found: {input_parquet}")
    
    print("=" * 60)
    print("Filtering Test Set from Dataset")
    print("=" * 60)
    
    # Load test set patterns
    test_patterns = load_test_set_patterns(csv_file)
    
    # Filter dataset
    df_filtered, original_count, filtered_count, df_removed = filter_dataset(
        input_parquet,
        output_parquet,
        test_patterns,
        args.protein_col,
        save_filtered,
    )
    
    print("\n" + "=" * 60)
    print("✅ Filtering completed successfully!")
    print("=" * 60)
    print(f"📊 Summary:")
    print(f"   Test patterns: {len(test_patterns)}")
    print(f"   Input file: {input_parquet}")
    print(f"   Training set (filtered): {output_parquet}")
    print(f"     Rows: {len(df_filtered)}")
    if save_filtered:
        print(f"   Test set (removed): {save_filtered}")
        print(f"     Rows: {len(df_removed)}")
    print(f"   Original rows: {original_count}")
    print(f"   Removed: {filtered_count} rows")
    if original_count > 0:
        print(f"   Reduction: {100 * (filtered_count / original_count):.2f}%")


if __name__ == "__main__":
    main()

