#!/usr/bin/env python3
"""
Compute QED and SA for generated molecules stored in JSON files.

Supports all JSON formats produced by this project:
  - generate_from_base_model.py     : {"generated_molecules": [...]}
  - generate_from_full_model.py     : {"generated_molecules": [...]}
  - lead_optimization_from_valset   : {"results": [{"generated_molecules": [...]}, ...]}

Usage:
    # Single file
    python scripts/validation_scripts/compute_qed_sa.py \
        --json outputs/generated_molecules/base_pretrained/generated_validation.json \
        --name base_pretrained

    # Compare multiple files
    python scripts/validation_scripts/compute_qed_sa.py \
        --json outputs/.../base_pretrained/generated_validation.json \
                outputs/.../v1_normal/generated_validation.json \
                outputs/.../v4_normal/generated_validation.json \
        --name base_pretrained v1_normal v4_normal


    # CSV (training set)
    python scripts/validation_scripts/compute_qed_sa.py \
        --csv finetune_data/bdnv2_standardize.csv \
        --smiles_col standardize_smi \
        --name train_set
"""

import argparse
import json
import sys
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, QED

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from rdkit.Chem import RDConfig
    import os
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer
    _SA_AVAILABLE = True
except Exception:
    _SA_AVAILABLE = False
    print("[WARN] sascorer not found via RDConfig; trying direct import...")
    try:
        import sascorer
        _SA_AVAILABLE = True
    except ImportError:
        print("[WARN] sascorer unavailable — SA scores will be skipped.")


def extract_smiles(data: dict) -> list:
    if "generated_molecules" in data and isinstance(data["generated_molecules"], list):
        return data["generated_molecules"]
    if "results" in data:
        out = []
        for entry in data["results"]:
            out.extend(entry.get("generated_molecules", []))
        return out
    return []


def compute_scores(smiles_list: list, desc: str = "Computing") -> pd.DataFrame:
    rows = []
    for smi in tqdm(smiles_list, desc=desc, mininterval=2.0):
        if not isinstance(smi, str):
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        row = {"SMILES": Chem.MolToSmiles(mol), "QED": QED.qed(mol)}
        if _SA_AVAILABLE:
            row["SA"] = sascorer.calculateScore(mol)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, name: str) -> dict:
    n = len(df)
    summary = {"name": name, "n_valid": n}
    for col in ["QED", "SA"]:
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        summary[f"{col}_mean"]   = round(float(vals.mean()), 4)
        summary[f"{col}_median"] = round(float(vals.median()), 4)
        summary[f"{col}_std"]    = round(float(vals.std()), 4)
        if col == "QED":
            summary["QED_ge0.5"] = int((vals >= 0.5).sum())
            summary["QED_ge0.5_pct"] = round(float((vals >= 0.5).mean()) * 100, 2)
        if col == "SA":
            summary["SA_le3"] = int((vals <= 3).sum())
            summary["SA_le3_pct"] = round(float((vals <= 3).mean()) * 100, 2)
    return summary


def print_comparison_table(summaries: list) -> None:
    metrics = ["n_valid", "QED_mean", "QED_median", "QED_std",
               "QED_ge0.5_pct", "SA_mean", "SA_median", "SA_le3_pct"]
    metrics = [m for m in metrics if any(m in s for s in summaries)]

    col_w = max(len(s["name"]) for s in summaries) + 2
    header = f"{'Metric':<22}" + "".join(f"{s['name']:>{col_w}}" for s in summaries)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for m in metrics:
        row = f"{m:<22}"
        for s in summaries:
            val = s.get(m, "-")
            row += f"{str(val):>{col_w}}"
        print(row)
    print("=" * len(header))


def process_one(path, name, is_csv=False, smiles_col="standardize_smi") -> tuple:
    if is_csv:
        print(f"\nProcessing CSV: {path}  [{name}]")
        df_raw = pd.read_csv(path)
        smiles_list = df_raw[smiles_col].dropna().tolist()
    else:
        print(f"\nProcessing: {path}  [{name}]")
        with open(path) as f:
            data = json.load(f)
        smiles_list = extract_smiles(data)

    print(f"  SMILES loaded: {len(smiles_list)}")
    df = compute_scores(smiles_list, desc=f"  {name}")
    print(f"  Valid molecules: {len(df)}")
    s = summarize(df, name)

    for col in ["QED", "SA"]:
        if f"{col}_mean" in s:
            print(f"  {col}: mean={s[f'{col}_mean']:.4f}  median={s[f'{col}_median']:.4f}  std={s[f'{col}_std']:.4f}")
    if "QED_ge0.5_pct" in s:
        print(f"  QED≥0.5: {s['QED_ge0.5']} / {s['n_valid']}  ({s['QED_ge0.5_pct']:.1f}%)")
    if "SA_le3_pct" in s:
        print(f"  SA≤3:    {s['SA_le3']} / {s['n_valid']}  ({s['SA_le3_pct']:.1f}%)")
    return s, df


def main():
    parser = argparse.ArgumentParser(description="Compute QED/SA from JSON or CSV files")
    parser.add_argument("--json",       nargs="*", default=[],
                        help="Path(s) to generated JSON file(s)")
    parser.add_argument("--csv",        nargs="*", default=[],
                        help="Path(s) to CSV file(s) with SMILES column")
    parser.add_argument("--smiles_col", type=str, default="standardize_smi",
                        help="Column name for SMILES in CSV (default: standardize_smi)")
    parser.add_argument("--name",       nargs="*", default=None,
                        help="Display name(s) for each input (JSON first, then CSV)")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Save per-molecule scores to this CSV (single-input mode only)")
    args = parser.parse_args()

    if not args.json and not args.csv:
        parser.error("Provide at least one --json or --csv file.")

    all_paths = list(args.json) + list(args.csv)
    is_csv    = [False] * len(args.json) + [True] * len(args.csv)
    auto_names = [Path(p).stem for p in all_paths]
    names = list(args.name or [])
    if len(names) < len(all_paths):
        names += auto_names[len(names):]

    all_summaries, all_dfs = [], []
    for path, name, csv_flag in zip(all_paths, names, is_csv):
        s, df = process_one(path, name, is_csv=csv_flag, smiles_col=args.smiles_col)
        all_summaries.append(s)
        all_dfs.append((path, df))

    if len(all_summaries) > 1:
        print_comparison_table(all_summaries)

    for (path, df), s in zip(all_dfs, all_summaries):
        if args.output_csv and len(all_paths) == 1:
            df.to_csv(args.output_csv, index=False)
            print(f"  Per-molecule scores saved → {args.output_csv}")
        out = Path(path).parent / "qed_sa_summary.json"
        with open(out, "w") as f:
            json.dump(s, f, indent=2)
        print(f"Summary saved → {out}")


if __name__ == "__main__":
    main()
