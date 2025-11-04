#!/usr/bin/env python3
"""
Filter a pickle of (receptor_path, ligand_path) tuples by excluding entries
whose receptor path contains a target folder present in a CSV test set.

Assumptions:
- The pickle contains an iterable of 2-tuples: (receptor_path: str, ligand_path: str)
- The CSV has a column named 'Target ChEMBLID'
- For each value in that column, the corresponding target folder name in paths
  is 'target_<value>'

Example:
  python utils/filter_pkl_by_testset.py \
    --pkl-in /home/yang2531/Documents/Project/Structure_safe/protein_extraction_10.0A_index.pkl \
    --csv /home/yang2531/Documents/Project/Structure_safe/test_set_info.csv \
    --pkl-out /home/yang2531/Documents/Project/Structure_safe/protein_extraction_10.0A_index.filtered.pkl
"""
import argparse
import os
import pickle
from typing import Iterable, List, Sequence, Tuple

import pandas as pd


Tuple2 = Tuple[str, str]


def load_pickle_tuples(path: str) -> List[Tuple2]:
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
    print(len(items))
    return items


def build_test_targets(csv_path: str) -> set:
    df = pd.read_csv(csv_path, dtype=str)
    if 'Target ChEMBLID' not in df.columns:
        # Try to handle near-matches by stripping whitespace in headers
        df.columns = [c.strip() for c in df.columns]
    if 'Target ChEMBLID' not in df.columns:
        raise KeyError("CSV must contain column 'Target ChEMBLID'")
    # Clean values and prefix
    vals = (
        df['Target ChEMBLID']
        .astype(str)
        .str.strip()
        .replace({'': pd.NA})
        .dropna()
        .tolist()
    )
    return {f'target_{v}' for v in vals}


def receptor_contains_any_target_folder(receptor_path: str, targets: set) -> bool:
    # Split into path segments and check exact segment match
    # Also consider both os.sep and '/' to be safe
    segments = []
    for seg in receptor_path.replace('\\', '/').split('/'):
        s = seg.strip()
        if s:
            segments.append(s)
    return any(seg in targets for seg in segments)


def filter_tuples(
    items: Sequence[Tuple2], targets: set
) -> List[Tuple2]:
    kept: List[Tuple2] = []
    for receptor_path, ligand_path in items:
        if receptor_contains_any_target_folder(receptor_path, targets):
            continue
        kept.append((receptor_path, ligand_path))
    print(len(kept))
    return kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl-in', required=True, help='Input pickle path (list of (receptor, ligand) tuples)')
    ap.add_argument('--csv', required=True, help='CSV path with column "Target ChEMBLID"')
    ap.add_argument('--pkl-out', required=True, help='Output pickle path for filtered tuples')
    args = ap.parse_args()

    items = load_pickle_tuples(args.pkl_in)
    targets = build_test_targets(args.csv)

    before = len(items)
    filtered = filter_tuples(items, targets)
    after = len(filtered)
    removed = before - after

    os.makedirs(os.path.dirname(os.path.abspath(args.pkl_out)) or '.', exist_ok=True)
    with open(args.pkl_out, 'wb') as f:
        pickle.dump(filtered, f)

    print(f'Read {before} tuples, removed {removed} (matched test targets), kept {after}.')
    print(f'Wrote filtered pickle to {args.pkl_out}')


if __name__ == '__main__':
    main()


