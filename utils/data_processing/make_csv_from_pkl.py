#!/usr/bin/env python3
"""
Load a pickle file containing a list of (protein_path, ligand_path.sdf) tuples,
extract a SMILES string for each ligand (first valid molecule in SDF), and write a CSV with:
  smiles, protein_path, ligand_path

Usage:
  python make_csv_from_pkl.py \
    --pkl /abs/path/to/finetune_data/protein_extraction_10.0A_index.pkl \
    --out /abs/path/to/output.csv

Assumes RDKit is available.
Rows without a valid SMILES are skipped.
"""

import argparse
import os
import pickle
import csv
from rdkit import Chem


def extract_first_smiles_from_sdf(sdf_path):
    try:
        supplier = Chem.SDMolSupplier(sdf_path)
    except Exception:
        return None
    if supplier is None:
        return None
    for mol in supplier:
        if mol is None:
            continue
        try:
            smi = Chem.MolToSmiles(mol)
        except Exception:
            smi = None
        if smi:
            return smi
    return None


def load_tuple_list(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    result = []
    for item in data:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            prot, lig = item[0], item[1]
            if isinstance(prot, str) and isinstance(lig, str):
                result.append((prot, lig))
    return result


def write_csv(rows, out_csv):
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or '.', exist_ok=True)
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['smiles', 'protein_path', 'ligand_path'])
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description='Convert (protein, ligand.sdf) pickle to CSV with SMILES')
    parser.add_argument('--pkl', required=True, help='Path to pickle containing list of (protein_path, ligand_path) tuples')
    parser.add_argument('--out', required=True, help='Output CSV path')
    args = parser.parse_args()

    rows = []
    for prot_path, lig_path in load_tuple_list(args.pkl):
        smi = extract_first_smiles_from_sdf(lig_path)
        if smi:
            rows.append((smi, prot_path, lig_path))

    write_csv(rows, args.out)
    print(f"Wrote {len(rows)} rows to {args.out}.")


if __name__ == '__main__':
    main()


