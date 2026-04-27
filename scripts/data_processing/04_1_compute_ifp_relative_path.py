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
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Union
import oddt as od
from oddt import fingerprints

# Add paths for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "utils"))

# Constants
IFP_DIM = 16384


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
    """Load the dataset and normalise path columns to relative paths."""
    print(f"Loading dataset from {input_file}")
    df = pd.read_parquet(input_file)
    print(f"Loaded {len(df)} samples")

    for col in ("protein_path", "ligand_path"):
        if col in df.columns:
            df[col] = df[col].apply(to_relative_path)
            print(f"Normalised {col} to relative (e.g. {df[col].iloc[0]})")

    return df


def _worker_compute_ifp(args):
    """
    Worker function for parallel IFP computation (fork-safe).
    On Linux, ProcessPoolExecutor uses fork so oddt.toolkit is inherited from
    the parent process — no re-initialization needed.
    Returns (row_idx, arr_bytes_or_None, error_msg_or_None).
    """
    row_idx, prot_path, lig_path, D, packed_bits = args
    try:
        import oddt as od
        import numpy as np

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

        arr = np.asarray(ifp, dtype=np.float32)
        if arr.shape != (D,):
            raise ValueError(f"IFP dim mismatch: {arr.shape} != ({D},)")

        if packed_bits:
            bits   = (arr > 0).astype(np.uint8)
            packed = np.packbits(bits, bitorder="little")
            tlen   = D // 8
            packed = packed[:tlen] if len(packed) >= tlen else np.pad(packed, (0, tlen - len(packed)))
            return row_idx, packed.tobytes(), None
        else:
            return row_idx, arr.tobytes(), None

    except Exception as e:
        return row_idx, None, str(e)


def compute_interaction_fingerprints(
    df: pd.DataFrame,
    output_dir: Path,
    batch_size: int = 1024,
    matrix_file: Optional[Path] = None,
    packed_bits: bool = False,
    n_workers: int = 1,
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
    print(f"Workers: {n_workers}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if matrix_file is None:
        matrix_file = output_dir / ("ifp_packed.npy" if packed_bits else "ifp_float32.npy")

    print(f"IFP matrix will be saved to: {matrix_file}")

    # Allocate memmap
    if packed_bits:
        cols = D // 8
        X = np.memmap(matrix_file, mode="w+", dtype=np.uint8, shape=(N, cols))
        print(f"Allocated packed matrix: shape ({N}, {cols}), dtype=uint8")
    else:
        X = np.memmap(matrix_file, mode="w+", dtype=np.float32, shape=(N, D))
        print(f"Allocated float32 matrix: shape ({N}, {D}), dtype=float32")

    # Create an index column
    df = df.reset_index(drop=True).copy()
    df["ifp_row_idx"] = np.arange(N, dtype=np.int64)

    # Build job list
    jobs = [
        (i, to_absolute_path(row.protein_path), to_absolute_path(row.ligand_path), D, packed_bits)
        for i, row in enumerate(df.itertuples(index=False))
    ]

    errors = []

    if n_workers <= 1:
        # Direct single-process path — avoids subprocess wrapper / re-import issues
        for i, row in enumerate(tqdm(df.itertuples(index=False), total=N, desc="Computing IFP")):
            prot_path = to_absolute_path(row.protein_path)
            lig_path  = to_absolute_path(row.ligand_path)
            try:
                protein = next(od.toolkit.readfile('pdb', prot_path))
                protein.protein = True
                ligand  = next(od.toolkit.readfile('sdf', lig_path))

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
                arr = np.asarray(ifp, dtype=np.float32)
                if arr.shape != (D,):
                    raise ValueError(f"IFP dim mismatch: {arr.shape} != ({D},)")

                if packed_bits:
                    bits   = (arr > 0).astype(np.uint8)
                    packed = np.packbits(bits, bitorder="little")
                    tlen   = D // 8
                    packed = packed[:tlen] if len(packed) >= tlen else np.pad(packed, (0, tlen - len(packed)))
                    X[i, :] = packed
                else:
                    X[i, :] = arr

            except Exception as e:
                errors.append((i, prot_path, lig_path, str(e)))
                X[i, :] = 0 if packed_bits else np.nan

            if i % batch_size == 0:
                X.flush()
    else:
        # Parallel path
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_worker_compute_ifp, job): job for job in jobs}
            completed = 0
            with tqdm(total=N, desc="Computing IFP") as pbar:
                for future in as_completed(futures):
                    job = futures[future]
                    try:
                        row_idx, data, err = future.result()
                    except Exception as exc:
                        row_idx = job[0]
                        errors.append((row_idx, job[1], job[2], str(exc)))
                        X[row_idx, :] = 0 if packed_bits else np.nan
                    else:
                        if err is not None:
                            errors.append((row_idx, job[1], job[2], err))
                            X[row_idx, :] = 0 if packed_bits else np.nan
                        else:
                            dtype = np.uint8 if packed_bits else np.float32
                            X[row_idx, :] = np.frombuffer(data, dtype=dtype)
                    completed += 1
                    if completed % batch_size == 0:
                        X.flush()
                    pbar.update(1)

    X.flush()
    del X  # Close memmap

    if errors:
        failed_count = len(errors)
        preview = "\n".join([
            f"  idx={idx} {prot} | {lig} | {msg}"
            for idx, prot, lig, msg in errors[:20]
        ])
        if failed_count > 20:
            preview += f"\n  ... and {failed_count - 20} more failures"
        raise ValueError(
            f"ERROR: IFP failed for {failed_count} / {N} pairs.\n"
            f"All IFP calculations must succeed.\n"
            f"Examples:\n{preview}"
        )

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


def load_ifp_row(matrix_file: Union[str, Path], row_idx: int, packed_bits: bool = False) -> np.ndarray:
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
    parser.add_argument("--n_workers", type=int, default=1,
                       help="Number of parallel worker processes (default: 1)")
    
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
        n_workers=args.n_workers,
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
