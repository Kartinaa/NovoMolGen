#!/usr/bin/env python3
"""
Step 4: Remove samples with zero evolutionary vectors (failed processing)
This script identifies and removes samples where evo_vec is a zero vector,
indicating that the evolutionary vector computation failed.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def is_zero_vector(vec, tolerance=1e-6):
    """Check if a vector is effectively zero (all values close to 0)."""
    # Handle None
    if vec is None:
        return True
    
    # Convert to numpy array for easier handling
    if isinstance(vec, (list, np.ndarray)):
        vec = np.array(vec)
    else:
        # Try to convert other types
        try:
            vec = np.array(vec)
        except:
            return False
    
    # Check if array is empty
    if vec.size == 0:
        return True
    
    # Check for NaN values (if any NaN, consider it invalid/zero)
    if np.isnan(vec).any():
        return True
    
    # Check if all values are close to zero
    return np.allclose(vec, 0.0, atol=tolerance)


def remove_zero_vectors(input_file: Path, output_file: Path = None):
    """Remove samples with zero evolutionary vectors from the dataset.
    
    Args:
        input_file: Path to input parquet file with evo_vec column
        output_file: Path to output file (default: same as input with '_cleaned' suffix)
    
    Returns:
        Cleaned DataFrame and statistics
    """
    print(f"Loading dataset from {input_file}")
    df = pd.read_parquet(input_file)
    print(f"Loaded {len(df)} samples")
    
    if 'evo_vec' not in df.columns:
        raise ValueError("Dataset does not contain 'evo_vec' column")
    
    # Identify zero vectors
    print("\nIdentifying zero vectors...")
    df['is_zero_vec'] = df['evo_vec'].apply(is_zero_vector)
    zero_vec_count = df['is_zero_vec'].sum()
    
    print(f"Found {zero_vec_count} samples with zero vectors ({zero_vec_count/len(df)*100:.2f}%)")
    
    # Show some statistics
    if zero_vec_count > 0:
        print("\nSample indices with zero vectors:")
        zero_indices = df[df['is_zero_vec']].index[:10].tolist()  # Show first 10
        print(f"  {zero_indices}")
        if zero_vec_count > 10:
            print(f"  ... and {zero_vec_count - 10} more")
    
    # Remove zero vectors
    print(f"\nRemoving {zero_vec_count} samples with zero vectors...")
    df_cleaned = df[~df['is_zero_vec']].copy()
    
    # Drop the temporary flag column
    df_cleaned = df_cleaned.drop(['is_zero_vec'], axis=1)
    
    print(f"Remaining samples: {len(df_cleaned)} ({len(df_cleaned)/len(df)*100:.2f}%)")
    
    # Save cleaned dataset
    if output_file is None:
        # Create output filename by adding '_cleaned' before .parquet
        output_file = input_file.parent / f"{input_file.stem}_cleaned.parquet"
    
    print(f"\nSaving cleaned dataset to {output_file}")
    df_cleaned.to_parquet(output_file, index=False)
    print(f"✅ Saved {len(df_cleaned)} samples to {output_file}")
    
    # Also save statistics
    stats = {
        'original_count': len(df),
        'removed_count': zero_vec_count,
        'remaining_count': len(df_cleaned),
        'removal_rate': zero_vec_count / len(df) * 100
    }
    
    print("\n" + "=" * 60)
    print("📊 Cleaning Statistics:")
    print("=" * 60)
    print(f"Original samples: {stats['original_count']}")
    print(f"Removed (zero vectors): {stats['removed_count']} ({stats['removal_rate']:.2f}%)")
    print(f"Remaining samples: {stats['remaining_count']} ({100-stats['removal_rate']:.2f}%)")
    print("=" * 60)
    
    return df_cleaned, stats


def main():
    parser = argparse.ArgumentParser(
        description="Remove samples with zero evolutionary vectors from dataset"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to input parquet file with evo_vec column"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to output cleaned parquet file (default: input_file with '_cleaned' suffix)"
    )
    
    args = parser.parse_args()
    
    input_file = Path(args.input_file)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    output_file = Path(args.output_file) if args.output_file else None
    
    # Remove zero vectors
    df_cleaned, stats = remove_zero_vectors(input_file, output_file)
    
    print("\n✅ Step 4 completed successfully!")
    print(f"Cleaned dataset shape: {df_cleaned.shape}")


if __name__ == "__main__":
    main()

