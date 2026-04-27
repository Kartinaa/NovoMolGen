#!/usr/bin/env python3
"""
Split a parquet file (separate mode, ifp_row_idx) into train_part_*.parquet shards
+ validation.parquet, filling in IFP vectors from an external .npy memmap file.

Usage:
    python scripts/data_processing/shard_parquet_to_train_data.py \
        --input  finetune_data/processed_data_scaffold_splitting/dataset_with_ifp.parquet \
        --ifp_npy finetune_data/processed_data_scaffold_splitting/ifp_float32.npy \
        --output_dir finetune_data/processed_data_scaffold_splitting/train_data/ \
        --rows_per_shard 10000 \
        --split_col split
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

IFP_DIM = 16384


def main():
    parser = argparse.ArgumentParser(
        description="Shard parquet (separate-mode) into train_part_*.parquet + validation.parquet"
    )
    parser.add_argument("--input",         type=str, required=True,
                        help="Input parquet file (separate mode, has ifp_row_idx)")
    parser.add_argument("--ifp_npy",       type=str, required=True,
                        help="Path to ifp_float32.npy memmap file")
    parser.add_argument("--output_dir",    type=str, required=True,
                        help="Output directory")
    parser.add_argument("--rows_per_shard",type=int, default=10000,
                        help="Rows per shard (default: 10000)")
    parser.add_argument("--num_shards",    type=int, default=None,
                        help="Number of shards (overrides --rows_per_shard)")
    parser.add_argument("--split_col",     type=str, default="split",
                        help="Column name for train/validation split (default: split)")
    parser.add_argument("--train_val",     type=str, default="train")
    parser.add_argument("--val_val",       type=str, default="validation")
    parser.add_argument("--no_split_col",  action="store_true",
                        help="No split column — treat everything as train")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load metadata parquet (small, no huge list columns) ──────────────────
    print(f"Loading metadata parquet: {args.input} ...")
    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df):,} rows  |  columns: {list(df.columns)}")

    # ── Memory-map IFP matrix ─────────────────────────────────────────────────
    print(f"Memory-mapping IFP matrix: {args.ifp_npy} ...")
    ifp = np.memmap(args.ifp_npy, dtype=np.float32, mode="r", shape=(len(df), IFP_DIM))
    print(f"IFP matrix shape: {ifp.shape}")

    # ── Split train / validation ──────────────────────────────────────────────
    has_split = (not args.no_split_col) and (args.split_col in df.columns)
    if has_split:
        train_df = df[df[args.split_col] == args.train_val].reset_index(drop=True)
        val_df   = df[df[args.split_col] == args.val_val].reset_index(drop=True)
        print(f"train: {len(train_df):,}  |  validation: {len(val_df):,}")
    else:
        if not args.no_split_col:
            print(f"[warn] Column '{args.split_col}' not found — treating all rows as train")
        train_df = df.reset_index(drop=True)
        val_df   = None

    # ── Write validation ──────────────────────────────────────────────────────
    if val_df is not None and len(val_df) > 0:
        print("Writing validation.parquet ...")
        val_ifp_rows = val_df["ifp_row_idx"].values.astype(int)
        val_df = val_df.copy()
        val_df["ifp"] = [ifp[r].tolist() for r in tqdm(val_ifp_rows, desc="validation IFP")]
        val_df = val_df.drop(columns=["ifp_row_idx"], errors="ignore")
        val_path = output_dir / "validation.parquet"
        val_df.to_parquet(val_path, index=False)
        print(f"✓ Saved {val_path}  ({len(val_df):,} rows)")

    # ── Shard train ───────────────────────────────────────────────────────────
    n = len(train_df)
    if args.num_shards is not None:
        shard_size = math.ceil(n / args.num_shards)
    else:
        shard_size = args.rows_per_shard
    num_shards = math.ceil(n / shard_size)
    print(f"\nSharding {n:,} train rows → {num_shards} files ({shard_size:,} rows each) ...")

    for i in tqdm(range(num_shards), desc="Writing shards"):
        start = i * shard_size
        end   = min(n, start + shard_size)
        shard = train_df.iloc[start:end].copy()

        # Fill IFP from memmap
        ifp_indices = shard["ifp_row_idx"].values.astype(int)
        shard["ifp"] = [ifp[r].tolist() for r in ifp_indices]
        shard = shard.drop(columns=["ifp_row_idx"], errors="ignore")

        shard_path = output_dir / f"train_part_{i:04d}.parquet"
        shard.to_parquet(shard_path, index=False)

    print(f"\n✓ Done.  {num_shards} train shards written to {output_dir}")


if __name__ == "__main__":
    main()
