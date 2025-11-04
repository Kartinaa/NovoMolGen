import argparse
import os
import pickle
from typing import List, Tuple, Any


def load_pickle(pickle_path: str) -> Any:


    with open(pickle_path, "rb") as f:
        return pickle.load(f)


def extract_protein_paths(items: Any) -> List[str]:
    """
    The pickle is expected to be a list of tuples where the first element
    is a protein path. We collect unique paths while preserving order.
    """
    seen = set()
    paths: List[str] = []
    for item in items:
        if isinstance(item, tuple) and len(item) >= 1:
            path = item[0]
            if isinstance(path, str) and path not in seen:
                seen.add(path)
                paths.append(path)
    return paths


def _filter_lines_remove_hetatm_and_orphan_ter(lines: List[str]) -> List[str]:
    """
    Remove all HETATM lines. Also remove orphan/redundant TER lines so that:
    - TER is only kept if the previous kept line is an ATOM line
    - Leading TER is removed
    - Consecutive TER are collapsed/removed (since previous kept won't be ATOM)
    """
    filtered: List[str] = []
    prev_kept_was_atom = False

    for ln in lines:
        if ln.startswith("HETATM"):
            # drop all HETATM
            continue
        if ln.startswith("TER"):
            # keep TER only if previous kept was ATOM
            if prev_kept_was_atom:
                filtered.append(ln)
                # After TER, we consider chain ended; next TER should not be kept
                prev_kept_was_atom = False
            # else: orphan TER -> drop
            continue

        # Keep all other lines as-is
        filtered.append(ln)
        # Update prev flag for ATOM lines only
        prev_kept_was_atom = ln.startswith("ATOM")

    return filtered


def remove_hetatm_inplace(pdb_path: str) -> bool:
    """
    Remove all lines starting with 'HETATM' from the PDB file in place.
    Returns True if file modified, False if no changes were needed.
    """
    if not os.path.isfile(pdb_path):
        return False

    try:
        with open(pdb_path, "r") as f:
            lines = f.readlines()
    except (OSError, UnicodeDecodeError):
        return False

    filtered = _filter_lines_remove_hetatm_and_orphan_ter(lines)

    if len(filtered) == len(lines):
        return False

    # Write back only if changed
    with open(pdb_path, "w") as f:
        f.writelines(filtered)
    return True


def main():
    parser = argparse.ArgumentParser(description="Remove HETATM records from PDBs listed in a pickle of tuples.")
    parser.add_argument("pickle", help="Path to pickle file (list of tuples, first elem = protein path)")
    parser.add_argument("--dry-run", action="store_true", help="Do not modify files; just report")
    args = parser.parse_args()

    items = load_pickle(args.pickle)
    pdb_paths = extract_protein_paths(items)

    total = len(pdb_paths)
    modified = 0
    missing = 0

    for pdb_path in pdb_paths:
        if not os.path.isfile(pdb_path):
            missing += 1
            print(f"MISSING: {pdb_path}")
            continue

        if args.dry_run:
            # Check if filtering would change the file
            try:
                with open(pdb_path, "r") as f:
                    original = f.readlines()
                filtered = _filter_lines_remove_hetatm_and_orphan_ter(original)
                if len(filtered) != len(original) or any(a != b for a, b in zip(filtered, original)):
                    print(f"WOULD MODIFY: {pdb_path}")
            except (OSError, UnicodeDecodeError):
                print(f"SKIP (unreadable): {pdb_path}")
            continue

        changed = remove_hetatm_inplace(pdb_path)
        if changed:
            modified += 1
            print(f"MODIFIED: {pdb_path}")

    print(f"Processed: {total}, Modified: {modified}, Missing: {missing}")


if __name__ == "__main__":
    main()


