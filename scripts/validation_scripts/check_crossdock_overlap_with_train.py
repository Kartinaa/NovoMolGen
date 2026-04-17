#!/usr/bin/env python3
"""
Check how many SMILES and scaffolds from CrossDocked test targets overlap
with the training set.

Input:
  --json_file   crossdock2020_duplicated_uniprotId_map_smiles_in_trainset.json
                  {uniprot: {smiles: [...], scaffold: [...]}, ...}
  --train_pkl   training_set.pkl  (list of (protein_path, ligand_path) tuples)
  --bdnv2_csv   bdnv2_standardize.csv  (protein_path, standardize_smi)

Output (--output_dir):
  summary.csv          — per-target counts
  in_train_smiles.json — {uniprot: [smiles that ARE in train]}
  not_in_train_smiles.json
  in_train_scaffolds.json
  not_in_train_scaffolds.json
"""

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')


def canonicalize(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def murcko_scaffold(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaf, isomericSmiles=True)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file", type=str,
        default="/home/yang2531/Documents/Project/StructureSAFE_benchmarking/MolGenBench/sup_info/"
                "crossdock2020_duplicated_uniprotId_map_smiles_in_trainset.json")
    parser.add_argument("--train_pkl", type=str,
        default="finetune_data/training_set.pkl")
    parser.add_argument("--bdnv2_csv", type=str,
        default="finetune_data/bdnv2_standardize.csv")
    parser.add_argument("--output_dir", type=str,
        default="outputs/crossdock_train_overlap")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Build training SMILES & scaffold sets ─────────────────────────────
    print("Loading training_set.pkl ...")
    with open(args.train_pkl, "rb") as f:
        train_pairs = pickle.load(f)

    train_protein_paths = set(p for p, _ in train_pairs)
    print(f"  {len(train_protein_paths):,} unique protein paths in training set")

    print(f"Loading {args.bdnv2_csv} ...")
    df = pd.read_csv(args.bdnv2_csv, usecols=["protein_path", "standardize_smi"])
    df_train = df[df["protein_path"].isin(train_protein_paths)]
    print(f"  {len(df_train):,} rows matched to training set")

    print("Canonicalizing training SMILES ...")
    train_smi_set = set()
    for smi in tqdm(df_train["standardize_smi"], desc="  train SMILES"):
        c = canonicalize(smi)
        if c:
            train_smi_set.add(c)
    print(f"  {len(train_smi_set):,} unique canonical training SMILES")

    print("Computing training scaffolds ...")
    train_scaffold_set = set()
    for smi in tqdm(train_smi_set, desc="  train scaffolds"):
        s = murcko_scaffold(smi)
        if s:
            train_scaffold_set.add(s)
    print(f"  {len(train_scaffold_set):,} unique training scaffolds")

    # ── 2. Load JSON ─────────────────────────────────────────────────────────
    print(f"\nLoading {args.json_file} ...")
    with open(args.json_file) as f:
        target_data = json.load(f)
    print(f"  {len(target_data)} targets in JSON")

    # ── 3. Per-target overlap check ──────────────────────────────────────────
    summary_rows = []
    in_smi  = {}
    out_smi = {}
    in_scaf  = {}
    out_scaf = {}

    for uniprot, info in tqdm(target_data.items(), desc="Checking targets"):
        raw_smiles   = info.get("smiles", [])
        raw_scaffolds = info.get("scaffold", [])

        # --- SMILES ---
        can_smiles = [canonicalize(s) for s in raw_smiles]
        can_smiles_valid = [(raw, c) for raw, c in zip(raw_smiles, can_smiles) if c]
        n_smi_total = len(can_smiles_valid)

        smi_in  = [raw for raw, c in can_smiles_valid if c in train_smi_set]
        smi_out = [raw for raw, c in can_smiles_valid if c not in train_smi_set]

        # --- Scaffolds (from JSON, canonicalized) ---
        can_scafs = [canonicalize(s) for s in raw_scaffolds]
        can_scafs_valid = [(raw, c) for raw, c in zip(raw_scaffolds, can_scafs) if c]
        n_scaf_total = len(can_scafs_valid)

        scaf_in  = [raw for raw, c in can_scafs_valid if c in train_scaffold_set]
        scaf_out = [raw for raw, c in can_scafs_valid if c not in train_scaffold_set]

        in_smi[uniprot]   = smi_in
        out_smi[uniprot]  = smi_out
        in_scaf[uniprot]  = scaf_in
        out_scaf[uniprot] = scaf_out

        summary_rows.append({
            "uniprot":          uniprot,
            "n_smiles_total":   n_smi_total,
            "n_smiles_in_train":  len(smi_in),
            "n_smiles_not_in_train": len(smi_out),
            "frac_smiles_in_train":  round(len(smi_in) / n_smi_total, 4) if n_smi_total else 0,
            "n_scaffolds_total":  n_scaf_total,
            "n_scaffolds_in_train": len(scaf_in),
            "n_scaffolds_not_in_train": len(scaf_out),
            "frac_scaffolds_in_train": round(len(scaf_in) / n_scaf_total, 4) if n_scaf_total else 0,
        })

    # ── 4. Save outputs ───────────────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows).sort_values("frac_smiles_in_train", ascending=False)
    summary_csv = out_dir / "summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    with open(out_dir / "in_train_smiles.json",  "w") as f: json.dump(in_smi,   f, indent=2)
    with open(out_dir / "not_in_train_smiles.json", "w") as f: json.dump(out_smi, f, indent=2)
    with open(out_dir / "in_train_scaffolds.json", "w") as f: json.dump(in_scaf,  f, indent=2)
    with open(out_dir / "not_in_train_scaffolds.json", "w") as f: json.dump(out_scaf, f, indent=2)

    # ── 5. Print summary ──────────────────────────────────────────────────────
    total_smi   = summary_df["n_smiles_total"].sum()
    total_in_smi = summary_df["n_smiles_in_train"].sum()
    total_scaf  = summary_df["n_scaffolds_total"].sum()
    total_in_scaf = summary_df["n_scaffolds_in_train"].sum()

    print(f"\n{'='*55}")
    print(f"Targets checked: {len(summary_df)}")
    print(f"SMILES   : {total_in_smi:,} / {total_smi:,} in train "
          f"({100*total_in_smi/total_smi:.1f}%)")
    print(f"Scaffolds: {total_in_scaf:,} / {total_scaf:,} in train "
          f"({100*total_in_scaf/total_scaf:.1f}%)")
    print(f"\nTop 10 targets by SMILES overlap:")
    print(summary_df[["uniprot","n_smiles_total","n_smiles_in_train","frac_smiles_in_train"]].head(10).to_string(index=False))
    print(f"\nOutputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
