#!/usr/bin/env python3
"""
Filter a pickle of (receptor_path, ligand_path) tuples to keep only entries
where both target and molecule ChEMBLIDs from CSV are present in the paths.

Assumptions:
- The pickle contains an iterable of 2-tuples: (receptor_path: str, ligand_path: str)
- The CSV has columns 'Target ChEMBLID' and 'Molecule ChEMBLID'
- Target folders in paths are named 'target_<Target ChEMBLID>'
- Molecule files/folders in paths contain '<Molecule ChEMBLID>'
- Also verifies that the filtered paths actually exist on disk

Example:
  python utils/filter_pkl_by_chemblids.py \
    --pkl-in /home/yang2531/Documents/Project/Structure_safe/finetune_data/protein_extraction_10.0A_index.pkl \
    --csv /home/yang2531/Documents/Project/Structure_safe/test_set_info.csv \
    --pkl-out /home/yang2531/Documents/Project/Structure_safe/finetune_data/protein_extraction_10.0A_index.filtered.pkl
"""
import argparse
import os
import pickle
from typing import List, Set, Tuple

import pandas as pd


Tuple2 = Tuple[str, str]


def load_pickle_tuples(path: str) -> List[Tuple2]:
    """Load pickle and validate it contains 2-tuples of strings."""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    if not isinstance(data, (list, tuple)):
        raise TypeError('Pickle must contain a list/tuple of 2-tuples')
    items: List[Tuple2] = []
    for i, item in enumerate(data):
        if not (isinstance(item, (list, tuple)) and len(item) == 2):
            raise TypeError(f'Item {i} is not a 2-tuple: {type(item)}')
        r, l = item
        if not isinstance(r, str) or not isinstance(l, str):
            raise TypeError(f'Item {i} elements must be strings, got {type(r)}, {type(l)}')
        items.append((r, l))
    print(f"Loaded {len(items)} tuples from pickle")
    return items


def load_chembl_pairs(csv_path: str) -> List[Tuple[str, str]]:
    """Load CSV and extract (target_chembl, molecule_chembl) pairs."""
    df = pd.read_csv(csv_path, dtype=str)
    
    # Handle potential whitespace in column names
    df.columns = [c.strip() for c in df.columns]
    
    required_cols = ['Target ChEMBLID', 'Molecule ChEMBLID']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"CSV must contain columns: {missing_cols}")
    
    # Clean and filter valid pairs
    pairs = []
    for _, row in df.iterrows():
        target = str(row['Target ChEMBLID']).strip()
        molecule = str(row['Molecule ChEMBLID']).strip()
        
        if target and molecule and target != 'nan' and molecule != 'nan':
            pairs.append((target, molecule))
    
    print(f"Loaded {len(pairs)} valid ChEMBL ID pairs from CSV")
    return pairs


def path_contains_chembl(path: str, chembl_id: str) -> bool:
    """Check if a path contains the given ChEMBL ID."""
    # Normalize path separators and check for the ID
    normalized_path = path.replace('\\', '/').lower()
    return chembl_id.lower() in normalized_path


def path_contains_target_chembl(path: str, target_chembl: str) -> bool:
    """Check if a path contains target_<chembl_id> folder."""
    target_folder = f"target_{target_chembl}"
    return path_contains_chembl(path, target_folder)


def filter_tuples_by_chembl_pairs(
    items: List[Tuple2], 
    chembl_pairs: List[Tuple[str, str]]
) -> List[Tuple2]:
    """Filter tuples to keep only those matching ChEMBL pairs from CSV."""
    kept: List[Tuple2] = []
    
    for receptor_path, ligand_path in items:
        # Check if this tuple matches any of the ChEMBL pairs
        for target_chembl, molecule_chembl in chembl_pairs:
            # Check if receptor path contains target_<chembl> and ligand path contains molecule_<chembl>
            if (path_contains_target_chembl(receptor_path, target_chembl) and 
                path_contains_chembl(ligand_path, molecule_chembl)):
                kept.append((receptor_path, ligand_path))
                break  # Found a match, no need to check other pairs
    
    print(f"Found {len(kept)} tuples matching ChEMBL pairs")
    return kept


def verify_paths_exist(items: List[Tuple2]) -> List[Tuple2]:
    """Verify that both receptor and ligand paths exist on disk."""
    existing: List[Tuple2] = []
    missing_count = 0
    
    for receptor_path, ligand_path in items:
        if os.path.exists(receptor_path) and os.path.exists(ligand_path):
            existing.append((receptor_path, ligand_path))
        else:
            missing_count += 1
            if missing_count <= 5:  # Show first 5 missing for debugging
                if not os.path.exists(receptor_path):
                    print(f"Missing receptor: {receptor_path}")
                if not os.path.exists(ligand_path):
                    print(f"Missing ligand: {ligand_path}")
    
    if missing_count > 5:
        print(f"... and {missing_count - 5} more missing paths")
    
    print(f"Verified {len(existing)} tuples have existing paths ({missing_count} missing)")
    return existing


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl-in', required=True, help='Input pickle path (list of (receptor, ligand) tuples)')
    ap.add_argument('--csv', required=True, help='CSV path with Target ChEMBLID and Molecule ChEMBLID columns')
    ap.add_argument('--pkl-out', required=True, help='Output pickle path for filtered tuples')
    ap.add_argument('--skip-verify', action='store_true', help='Skip path existence verification')
    args = ap.parse_args()

    # Load data
    items = load_pickle_tuples(args.pkl_in)
    chembl_pairs = load_chembl_pairs(args.csv)

    # Filter by ChEMBL pairs
    before = len(items)
    filtered = filter_tuples_by_chembl_pairs(items, chembl_pairs)
    after_chembl = len(filtered)
    removed_chembl = before - after_chembl

    # Verify paths exist
    if args.skip_verify:
        final_items = filtered
        after_verify = len(final_items)
        missing_paths = 0
    else:
        final_items = verify_paths_exist(filtered)
        after_verify = len(final_items)
        missing_paths = after_chembl - after_verify

    # Save filtered pickle
    os.makedirs(os.path.dirname(os.path.abspath(args.pkl_out)) or '.', exist_ok=True)
    with open(args.pkl_out, 'wb') as f:
        pickle.dump(final_items, f)

    print(f"\nSummary:")
    print(f"  Original tuples: {before}")
    print(f"  After ChEMBL filtering: {after_chembl} (removed {removed_chembl})")
    if not args.skip_verify:
        print(f"  After path verification: {after_verify} (removed {missing_paths} with missing paths)")
    print(f"  Final kept: {after_verify}")
    print(f"  Wrote filtered pickle to {args.pkl_out}")


if __name__ == '__main__':
    main()
