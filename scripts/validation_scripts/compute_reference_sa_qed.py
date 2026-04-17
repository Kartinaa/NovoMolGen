#!/usr/bin/env python3
"""
Compute mean SA and QED for all ligands belonging to test-set targets,
by looking up target ChEMBL IDs from test_set_info.csv in bdnv2_standardize.csv.

Usage:
    python scripts/validation_scripts/compute_reference_sa_qed.py \
        --test_set_csv finetune_data/training_test_spliting/test_set_info.csv \
        --bdnv2_csv finetune_data/bdnv2_standardize.csv
"""

import argparse
import sys
from pathlib import Path

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import pandas as pd
from rdkit import Chem, RDLogger
from tqdm import tqdm

from src.eval.components.moses import SA, QED

RDLogger.DisableLog('rdApp.*')


def main():
    parser = argparse.ArgumentParser(
        description="Compute mean SA and QED for all ligands of test-set targets in bdnv2"
    )
    parser.add_argument("--test_set_csv", type=str,
                        default="finetune_data/training_test_spliting/test_set_info.csv",
                        help="Path to test_set_info.csv (contains 'Target ChEMBLID' column)")
    parser.add_argument("--bdnv2_csv", type=str,
                        default="finetune_data/bdnv2_standardize.csv",
                        help="Path to bdnv2_standardize.csv")
    args = parser.parse_args()

    # --- Load test-set target ChEMBL IDs ---
    test_df = pd.read_csv(args.test_set_csv)
    test_target_ids = set(test_df['Target ChEMBLID'].dropna())
    print(f"Test-set targets: {len(test_target_ids)}")

    # --- Load bdnv2 and extract target ChEMBL ID from protein_path ---
    # protein_path format: .../target_CHEMBLXXXX/CHEMBLYYY/protein_10.0A.pdb
    print(f"Loading {args.bdnv2_csv} ...")
    df = pd.read_csv(args.bdnv2_csv)
    df['target_chembl'] = df['protein_path'].str.extract(r'target_(CHEMBL\d+)', expand=False)

    df_val = df[df['target_chembl'].isin(test_target_ids)].copy()
    n_targets_found = df_val['target_chembl'].nunique()
    print(f"Targets found in bdnv2: {n_targets_found}/{len(test_target_ids)}")
    print(f"Total ligands for those targets: {len(df_val)}")

    if df_val.empty:
        print("ERROR: No ligands found. Check that target IDs match.")
        sys.exit(1)

    # --- Compute SA and QED ---
    sa_scores, qed_scores = [], []
    failed = 0

    for smi in tqdm(df_val['standardize_smi'], desc="Computing SA/QED"):
        try:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                failed += 1
                continue
            sa_scores.append(SA(mol))
            qed_scores.append(QED(mol))
        except Exception:
            failed += 1

    n = len(sa_scores)
    if n == 0:
        print("ERROR: No valid molecules found.")
        sys.exit(1)

    print(f"\nResults ({n} ligands, {failed} failed/invalid):")
    print(f"  SA  — mean: {sum(sa_scores)/n:.4f}  min: {min(sa_scores):.4f}  max: {max(sa_scores):.4f}")
    print(f"  QED — mean: {sum(qed_scores)/n:.4f}  min: {min(qed_scores):.4f}  max: {max(qed_scores):.4f}")


if __name__ == "__main__":
    main()
