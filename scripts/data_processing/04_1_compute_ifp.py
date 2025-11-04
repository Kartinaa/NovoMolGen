#!/usr/bin/env python3
"""
Step 4: Compute interaction fingerprints (IFP) - Memory-efficient version
This script loads the dataset and computes IFP for each protein-ligand pair using PLEC.

Memory-efficient features:
- IFP vectors stored in separate .npy memmap file (not in DataFrame)
- DataFrame contains only metadata + ifp_row_idx
- Chunked processing to stay within memory limits
- Optional packed bits mode for reduced storage
- No Python lists - all NumPy arrays
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import sys
import oddt as od
from oddt import fingerprints

# Add paths for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "utils"))

# Constants
IFP_DIM = 16384


def load_dataset(input_file: Path):
    """Load the dataset."""
    print(f"Loading dataset from {input_file}")
    df = pd.read_parquet(input_file)
    print(f"Loaded {len(df)} samples")
    return df


def compute_interaction_fingerprints(
    df: pd.DataFrame,
    output_dir: Path,
    batch_size: int = 1024,
    matrix_file: Path | None = None,
    packed_bits: bool = False,
):
    """
    Compute PLEC IFPs and write them into a single on-disk matrix:
      - float32 memmap of shape (N, 16384) when packed_bits=False
      - packed uint8 memmap with np.packbits per row when packed_bits=True
    
    The dataframe gets a new column 'ifp_row_idx' (0..N-1). No object columns.
    
    Returns:
        df: Updated DataFrame with ifp_row_idx column
        matrix_file: Path to the saved IFP matrix file
    """
    N = len(df)
    D = IFP_DIM
    
    if N == 0:
        raise ValueError("Empty dataset")
    
    print("Computing interaction fingerprints using PLEC...")
    print(f"Processing {N} protein-ligand pairs")
    print(f"Mode: {'packed bits' if packed_bits else 'float32'}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    if matrix_file is None:
        matrix_file = output_dir / ("ifp_packed.npy" if packed_bits else "ifp_float32.npy")
    
    print(f"IFP matrix will be saved to: {matrix_file}")
    
    # Allocate memmap
    if packed_bits:
        # packed bits: 16384 bits -> 2048 bytes; shape (N, 2048), dtype=uint8
        cols = D // 8
        X = np.memmap(matrix_file, mode="w+", dtype=np.uint8, shape=(N, cols))
        print(f"Allocated packed matrix: shape ({N}, {cols}), dtype=uint8")
    else:
        X = np.memmap(matrix_file, mode="w+", dtype=np.float32, shape=(N, D))
        print(f"Allocated float32 matrix: shape ({N}, {D}), dtype=float32")
    
    errors = []
    
    # Create an index column to avoid dict lookups
    df = df.reset_index(drop=True).copy()
    df["ifp_row_idx"] = np.arange(N, dtype=np.int64)
    
    def calc_ifp_plec_numpy(prot_path: str, lig_path: str) -> np.ndarray:
        """Compute IFP and return as NumPy array."""
        try:
            protein = next(od.toolkit.readfile('pdb', prot_path))
            protein.protein = True
            
            ligand = next(od.toolkit.readfile('sdf', lig_path))
            
            ifp = od.fingerprints.PLEC(
                protein, ligand,
                depth_ligand=1,
                depth_protein=5,
                distance_cutoff=4.5,
                size=D,
                count_bits=True,
                sparse=False,
                ignore_hoh=True,
            )
            
            # ODDT returns numpy-like; ensure contiguous float32
            arr = np.asarray(ifp, dtype=np.float32)
            if arr.shape != (D,):
                raise ValueError(f"IFP dim mismatch: {arr.shape} != ({D},)")
            return arr
        except Exception as e:
            raise RuntimeError(f"{prot_path} or {lig_path}: {e}")
    
    # Chunked compute using itertuples for efficiency
    print(f"\nProcessing in chunks of {batch_size}...")
    row_iterator = df.itertuples(index=False)
    
    for start in tqdm(range(0, N, batch_size), desc="Computing IFP", 
                     total=(N + batch_size - 1) // batch_size):
        end = min(N, start + batch_size)
        
        # Process this chunk - iterate only over rows in current chunk
        for i in range(start, end):
            row = next(row_iterator)
            prot_path = row.protein_path
            lig_path = row.ligand_path
            
            try:
                arr = calc_ifp_plec_numpy(prot_path, lig_path)
                
                if packed_bits:
                    # Convert counts to boolean bits, then pack
                    bits = (arr > 0).astype(np.uint8)
                    packed = np.packbits(bits, bitorder="little")
                    # Ensure correct length (packbits might return slightly different length)
                    target_len = D // 8
                    if len(packed) < target_len:
                        packed = np.pad(packed, (0, target_len - len(packed)), 'constant')
                    elif len(packed) > target_len:
                        packed = packed[:target_len]
                    X[i, :] = packed
                else:
                    X[i, :] = arr.astype(np.float32, copy=False)
                    
            except Exception as e:
                error_msg = str(e)
                errors.append((i, prot_path, lig_path, error_msg))
                print(f"ERROR: IFP failed for row {i}: {error_msg}")
                # Fill with NaN or zeros to keep shape consistent
                if packed_bits:
                    X[i, :] = 0
                else:
                    X[i, :] = np.nan
        
        # Flush periodically after each chunk
        X.flush()
    
    # Final flush
    X.flush()
    del X  # Close memmap
    
    if errors:
        # Count failures and raise with a short preview
        failed_count = len(errors)
        preview = "\n".join([
            f"  idx={idx} {prot} | {lig} | {msg}" 
            for idx, prot, lig, msg in errors[:20]
        ])
        if failed_count > 20:
            preview += f"\n  ... and {failed_count - 20} more failures"
        
        error_msg = (
            f"ERROR: IFP failed for {failed_count} / {N} pairs.\n"
            f"All IFP calculations must succeed.\n"
            f"Examples:\n{preview}"
        )
        raise ValueError(error_msg)
    
    print(f"✓ IFP matrix written to {matrix_file}")
    print(f"  Shape: ({N}, {D if not packed_bits else D//8})")
    print(f"  Dtype: {'uint8 (packed)' if packed_bits else 'float32'}")
    
    return df, matrix_file


def save_updated_dataset(df: pd.DataFrame, output_dir: Path, step_name: str):
    """
    Save metadata-only parquet (no 16k-dim vectors). Includes 'ifp_row_idx'.
    """
    output_file = output_dir / f"dataset_with_{step_name}_meta.parquet"
    df.to_parquet(output_file, index=False)
    print(f"✓ Saved metadata parquet to {output_file}")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    
    # Write a tiny sample CSV without the huge vector
    csv_file = output_dir / f"dataset_with_{step_name}_sample.csv"
    if len(df) > 0:
        sample_df = df.iloc[[0]].copy()
        # Remove any object columns that might contain vectors
        sample_df.to_csv(csv_file, index=False)
        print(f"✓ Saved sample row (metadata only) to {csv_file}")
    else:
        print("Warning: No data to save as CSV sample")
    
    return output_file


def load_ifp_row(matrix_file: str | Path, row_idx: int, packed_bits: bool = False) -> np.ndarray:
    """
    Load a single IFP row from the matrix file.
    
    Args:
        matrix_file: Path to the .npy matrix file
        row_idx: Row index (0-based)
        packed_bits: Whether the matrix uses packed bits
    
    Returns:
        NumPy array of shape (16384,) with dtype float32 or uint8
    """
    matrix_file = Path(matrix_file)
    
    if not matrix_file.exists():
        raise FileNotFoundError(f"Matrix file not found: {matrix_file}")
    
    if packed_bits:
        cols = IFP_DIM // 8
        X = np.memmap(matrix_file, mode="r", dtype=np.uint8, shape=(-1, cols))
        row = X[row_idx]
        bits = np.unpackbits(row, bitorder="little")
        # Return as uint8 (0/1) for consistency
        return bits[:IFP_DIM].astype(np.uint8)
    else:
        X = np.memmap(matrix_file, mode="r", dtype=np.float32, shape=(-1, IFP_DIM))
        # Return a copy to avoid memmap issues
        return np.array(X[row_idx], copy=True, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Compute interaction fingerprints (memory-efficient)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard float32 mode
  python 04_1_compute_ifp.py --input_file data.parquet --output_dir output
  
  # Packed bits mode (reduces storage ~4x)
  python 04_1_compute_ifp.py --input_file data.parquet --output_dir output --packed_bits
  
  # Custom matrix file location
  python 04_1_compute_ifp.py --input_file data.parquet --matrix_file /path/to/ifp.npy
        """
    )
    parser.add_argument("--input_file", type=str, required=True,
                       help="Path to the dataset parquet file")
    parser.add_argument("--output_dir", type=str, default="processed_data",
                       help="Output directory for processed data")
    parser.add_argument("--batch_size", type=int, default=1024,
                       help="Batch size for chunked processing (default: 1024)")
    parser.add_argument("--matrix_file", type=str, default=None,
                       help="Custom path for IFP matrix file (default: auto-generated in output_dir)")
    parser.add_argument("--packed_bits", action="store_true",
                       help="Use packed bits format (reduces storage but requires unpacking)")
    
    args = parser.parse_args()
    
    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    df = load_dataset(input_file)
    
    # Compute interaction fingerprints
    matrix_file = Path(args.matrix_file) if args.matrix_file else None
    df_updated, matrix_path = compute_interaction_fingerprints(
        df,
        output_dir,
        batch_size=args.batch_size,
        matrix_file=matrix_file,
        packed_bits=args.packed_bits,
    )
    
    # Save updated dataset (metadata only)
    output_file = save_updated_dataset(df_updated, output_dir, "ifp")
    
    # Print summary
    print("\n" + "=" * 60)
    print("✅ Step 4 completed successfully!")
    print("=" * 60)
    print(f"📊 Summary:")
    print(f"   Total pairs processed: {len(df_updated)}")
    print(f"   IFP matrix file: {matrix_path}")
    print(f"   Matrix shape: ({len(df_updated)}, {IFP_DIM if not args.packed_bits else IFP_DIM//8})")
    print(f"   Matrix dtype: {'uint8 (packed)' if args.packed_bits else 'float32'}")
    print(f"   Metadata parquet: {output_file}")
    print(f"   Mode: {'Packed bits (4x smaller)' if args.packed_bits else 'Float32 (standard)'}")
    
    # Quick verification
    if len(df_updated) > 0:
        sample_idx = df_updated['ifp_row_idx'].iloc[0]
        try:
            sample_vec = load_ifp_row(matrix_path, sample_idx, args.packed_bits)
            print(f"\n✓ Verification:")
            print(f"   Sample row 0 IFP shape: {sample_vec.shape}")
            print(f"   Sample row 0 IFP dtype: {sample_vec.dtype}")
            if args.packed_bits:
                print(f"   Sample row 0 bits set: {np.sum(sample_vec > 0)} / {len(sample_vec)}")
            else:
                print(f"   Sample row 0 range: [{sample_vec.min():.3f}, {sample_vec.max():.3f}]")
        except Exception as e:
            print(f"⚠️  Warning: Could not verify sample row: {e}")


if __name__ == "__main__":
    main()
