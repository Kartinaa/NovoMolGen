#!/usr/bin/env python3
"""
Create base dataset for MolGenBench_Version1 so that it can go through
02_compute_pocket_vec_auto_optimized.py, 03_compute_evo_vec.py, 04_compute_ligand_vec.py,
04_1_compute_ifp.py, 05_merge_ifp_with_dataset.py, and finally be used by generate_from_full_model.py.
"""

import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import molvs

from rdkit import Chem
from rdkit.Chem import AllChem

def sdf_to_standardized_smiles(sdf_path: Path) -> str | None:
    standardizer = molvs.standardize.Standardizer()
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    mol_original = next(iter(suppl), None)
    try:
        mol = standardizer.standardize(mol_original)
        mol = standardizer.fragment_parent(mol, skip_standardize=True)
        mol = standardizer.tautomer_parent(mol, skip_standardize=True)
        mol = standardizer.isotope_parent(mol, skip_standardize=True)
        mol = standardizer.charge_parent(mol, skip_standardize=True)
        return Chem.MolToSmiles(mol)
    except Exception as e:
        print(f"Error standardizing {sdf_path}: {e}")
        return Chem.MolToSmiles(mol_original)

def build_base_df(root_dir: Path):
    rows = []
    subdirs = sorted([p for p in root_dir.iterdir() if p.is_dir()])
    print(f"Found {len(subdirs)} subfolders under {root_dir}")
    for i, sub in enumerate(tqdm(subdirs, desc="Scanning folders")):
        pdb_files = list(sub.glob("*_pocket10.pdb"))
        sdf_files = list(sub.glob("*_lig.sdf"))
        if len(pdb_files) != 1 or len(sdf_files) != 1:
            print(f"Warning: {sub} does not have exactly one *_pocket10.pdb and *_lig.sdf, skip")
            continue
        pdb_path = pdb_files[0].resolve()
        sdf_path = sdf_files[0].resolve()

        smi = sdf_to_standardized_smiles(sdf_path)
        if smi is None:
            print(f"Warning: cannot extract SMILES from {sdf_path}, skip")
            continue

        rows.append({
            "id": f"bench_{i}",
            "split": "validation",              # 或者 'benchmark'
            "protein_path": str(pdb_path),      # 對應原來的 pocket 路徑
            "ligand_path": str(sdf_path),
            "standardize_smi": smi,
            "pocket_vec": None,
            "evo_vec": None,
            "ifp": None,
            "ligand_vec": None,
        })
    df = pd.DataFrame(rows)
    print(f"Built base dataframe with {len(df)} rows")
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root folder of MolGenBench_Version1",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="finetune_data/molgenbench_dataset",
        help="Where to save base_dataset.parquet",
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_base_df(root_dir)
    out_path = out_dir / "base_dataset.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved base dataset to {out_path}")

if __name__ == "__main__":
    main()