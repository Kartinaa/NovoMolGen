#!/usr/bin/env python3
"""
Per-protein test-set recovery rate for generated molecules.

For each test-set protein, computes how many of its known actives (from
bdnv2_standardize.csv) appear among the molecules generated conditioned on
that specific protein.

Data linkage:
  validation_set.pkl[i] → protein_path → target_CHEMBLXXX
  condition_info[i]['validation_index'] == i → generated molecules for CHEMBLXXX
  bdnv2_standardize.csv  target_CHEMBLXXX → all known actives for that protein

Comparison uses InChI key connectivity layer (first 14 chars).

Supported input formats:
  - generated_validation.json  with 'condition_info' list (per-protein conditioning)
  - lead_optimization_valset_results.json  with 'results' list

Usage:
    # Single file (per-protein + aggregate)
    python scripts/validation_scripts/check_test_set_recovery.py \
        --json outputs/.../cond/generated_validation.json

    # Multiple files comparison
    python scripts/validation_scripts/check_test_set_recovery.py \
        --json outputs/.../v1/cond/generated_validation.json \
                outputs/.../v4/cond/generated_validation.json \
        --name v1_cond v4_cond

    # Lead-opt format (auto-detected)
    python scripts/validation_scripts/check_test_set_recovery.py \
        --json outputs/.../lead_optimization_valset_results.json

    # Show per-protein detail
    python scripts/validation_scripts/check_test_set_recovery.py \
        --json ... --show_per_protein

    # Show which actives were found
    python scripts/validation_scripts/check_test_set_recovery.py \
        --json ... --show_hits
"""

import argparse
import ast
import json
import pickle
import re
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem.inchi import MolToInchiKey
from tqdm import tqdm

DEFAULT_TEST_CSV   = "finetune_data/training_test_spliting/test_set_info.csv"
DEFAULT_BDN_CSV    = "finetune_data/bdnv2_standardize.csv"
DEFAULT_VAL_PKL    = "finetune_data/validation_set.pkl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def connectivity_key(smi: str) -> str | None:
    if not isinstance(smi, str):
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    k = MolToInchiKey(mol)
    return k[:14] if k else None


def extract_target(protein_path: str) -> str | None:
    m = re.search(r"target_(CHEMBL\d+)", protein_path)
    return m.group(1) if m else None


def parse_smiles_field(val) -> list[str]:
    """condition_info generated_molecules can be a list or a string repr of list."""
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            result = ast.literal_eval(val)
            return result if isinstance(result, list) else []
        except Exception:
            return []
    return []


# ---------------------------------------------------------------------------
# Build reference: target_chembl → frozenset of connectivity keys
# ---------------------------------------------------------------------------

def build_target_actives(
    test_csv: str = DEFAULT_TEST_CSV,
    bdn_csv: str = DEFAULT_BDN_CSV,
) -> tuple[dict[str, frozenset], set[str]]:
    """
    Returns:
        target_to_keys : target_chembl → frozenset of connectivity keys (all known actives)
        test_targets   : set of Target ChEMBLIDs in the test set (exactly 228)
    """
    # Test-set targets (by Target ChEMBLID — the correct column)
    test_df = pd.read_csv(test_csv)
    test_df.columns = test_df.columns.str.strip()
    test_targets = set(test_df["Target ChEMBLID"].dropna().astype(str))
    print(f"  Test-set proteins (Target ChEMBLID): {len(test_targets)}")

    # All bdnv2 ligands, extract target ID from path
    print(f"  Loading {bdn_csv} …")
    bdn = pd.read_csv(bdn_csv)
    bdn["target_chembl"] = bdn["protein_path"].str.extract(r"target_(CHEMBL\d+)/")

    # Keep only test-set targets
    bdn_test = bdn[bdn["target_chembl"].isin(test_targets)].copy()
    print(f"  bdnv2 rows for test proteins: {len(bdn_test)} "
          f"({bdn_test['standardize_smi'].nunique()} unique SMILES)")

    # Build per-target active key sets
    target_to_keys: dict[str, set] = {}
    for _, row in tqdm(bdn_test.iterrows(), total=len(bdn_test),
                       desc="  indexing known actives", leave=False):
        tgt = row["target_chembl"]
        k   = connectivity_key(row["standardize_smi"])
        if k:
            target_to_keys.setdefault(tgt, set()).add(k)

    target_to_keys_frozen = {t: frozenset(ks) for t, ks in target_to_keys.items()}
    total_unique = sum(len(ks) for ks in target_to_keys_frozen.values())
    print(f"  Proteins with actives indexed: {len(target_to_keys_frozen)} "
          f"| total unique keys: {total_unique}")
    return target_to_keys_frozen, test_targets


# ---------------------------------------------------------------------------
# Build val_index → target mapping from validation_set.pkl
# ---------------------------------------------------------------------------

def build_valindex_map(val_pkl: str = DEFAULT_VAL_PKL) -> dict[int, str]:
    with open(val_pkl, "rb") as f:
        vs = pickle.load(f)
    mapping = {}
    for i, entry in enumerate(vs):
        protein_path = entry[0] if isinstance(entry, (list, tuple)) else entry
        tgt = extract_target(str(protein_path))
        if tgt:
            mapping[i] = tgt
    print(f"  validation_set.pkl: {len(mapping)}/228 indices mapped to target")
    return mapping


# ---------------------------------------------------------------------------
# Per-protein recovery from condition_info format
# ---------------------------------------------------------------------------

def evaluate_per_protein(
    condition_info: list,
    valindex_map: dict[int, str],
    target_to_keys: dict[str, frozenset],
    name: str = "",
    show_per_protein: bool = False,
    show_hits: bool = False,
) -> dict:
    per_protein = []
    unmapped = 0

    for entry in condition_info:
        val_idx = int(entry["validation_index"])
        gen_smiles = parse_smiles_field(entry["generated_molecules"])
        target = valindex_map.get(val_idx)

        if target is None or target not in target_to_keys:
            unmapped += 1
            continue

        known_keys = target_to_keys[target]
        n_known    = len(known_keys)

        # InChI keys of generated molecules for this protein
        gen_keys = set()
        for smi in gen_smiles:
            k = connectivity_key(smi)
            if k:
                gen_keys.add(k)

        hit_keys = gen_keys & known_keys
        n_hits   = len(hit_keys)

        per_protein.append({
            "val_idx":    val_idx,
            "target":     target,
            "n_generated": len(gen_smiles),
            "n_gen_unique": len(gen_keys),
            "n_known":    n_known,
            "n_hits":     n_hits,
            "recovery_pct": round(n_hits / n_known * 100, 4) if n_known else 0.0,
        })

    # Aggregate
    proteins_with_hits = sum(1 for r in per_protein if r["n_hits"] > 0)
    total_hits   = sum(r["n_hits"]   for r in per_protein)
    total_known  = sum(r["n_known"]  for r in per_protein)
    total_gen    = sum(r["n_generated"] for r in per_protein)
    mean_recovery = (sum(r["recovery_pct"] for r in per_protein) / len(per_protein)
                     if per_protein else 0.0)

    result = {
        "name":               name,
        "n_proteins":         len(per_protein),
        "proteins_with_hits": proteins_with_hits,
        "total_generated":    total_gen,
        "total_known_actives": total_known,
        "total_hits":         total_hits,
        "mean_recovery_pct":  round(mean_recovery, 4),
        "macro_recovery_pct": round(total_hits / total_known * 100, 4) if total_known else 0.0,
        "unmapped_proteins":  unmapped,
        "per_protein":        per_protein,
    }

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  Proteins evaluated  : {len(per_protein)}  (unmapped: {unmapped})")
    print(f"  Proteins with hits  : {proteins_with_hits}")
    print(f"  Total generated     : {total_gen:,}")
    print(f"  Total known actives : {total_known:,}")
    print(f"  Total hits          : {total_hits}")
    print(f"  Mean per-protein recovery : {mean_recovery:.4f}%")
    print(f"  Macro recovery (hits/known): {result['macro_recovery_pct']:.4f}%")

    if show_per_protein:
        print(f"\n  {'idx':>4}  {'Target':<16}  {'gen':>5}  {'known':>6}  {'hits':>5}  {'recov%':>7}")
        print(f"  {'-'*54}")
        for r in sorted(per_protein, key=lambda x: -x["n_hits"]):
            print(f"  {r['val_idx']:>4}  {r['target']:<16}  "
                  f"{r['n_gen_unique']:>5}  {r['n_known']:>6}  "
                  f"{r['n_hits']:>5}  {r['recovery_pct']:>7.4f}%")

    return result


# ---------------------------------------------------------------------------
# Lead-opt format: each entry has reference_smiles + val_idx
# ---------------------------------------------------------------------------

def evaluate_lead_opt(
    data: dict,
    valindex_map: dict[int, str],
    target_to_keys: dict[str, frozenset],
    name: str = "",
    show_per_protein: bool = False,
    show_hits: bool = False,
) -> dict:
    # Group by val_idx
    from collections import defaultdict
    groups: dict[int, list[str]] = defaultdict(list)
    for entry in data["results"]:
        val_idx    = int(entry.get("val_idx", -1))
        gen_smiles = entry.get("generated_molecules", [])
        groups[val_idx].extend(gen_smiles)

    condition_info = [
        {"validation_index": vi, "generated_molecules": smiles}
        for vi, smiles in groups.items()
    ]
    return evaluate_per_protein(condition_info, valindex_map, target_to_keys,
                                name=name, show_per_protein=show_per_protein,
                                show_hits=show_hits)


# ---------------------------------------------------------------------------
# Flat format (no condition_info): aggregate only, no per-protein breakdown
# ---------------------------------------------------------------------------

def evaluate_flat(
    generated_smiles: list[str],
    target_to_keys: dict[str, frozenset],
    name: str = "",
) -> dict:
    all_known_keys: frozenset = frozenset().union(*target_to_keys.values())
    gen_keys = set()
    for smi in tqdm(generated_smiles, desc=f"  indexing {name}", leave=False):
        k = connectivity_key(smi)
        if k:
            gen_keys.add(k)

    hits = gen_keys & all_known_keys
    result = {
        "name":            name,
        "n_generated":     len(generated_smiles),
        "n_gen_unique":    len(gen_keys),
        "total_known_actives": len(all_known_keys),
        "total_hits":      len(hits),
        "macro_recovery_pct": round(len(hits) / len(all_known_keys) * 100, 4) if all_known_keys else 0.0,
        "note": "flat format — no per-protein breakdown",
    }
    print(f"\n{'='*60}")
    print(f"  {name}  [flat / no condition_info]")
    print(f"  Generated : {len(generated_smiles):,}  ({len(gen_keys):,} unique)")
    print(f"  Known actives (all test proteins): {len(all_known_keys):,}")
    print(f"  Hits      : {len(hits)}")
    print(f"  Macro recovery: {result['macro_recovery_pct']:.4f}%")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Per-protein test-set active recovery in generated molecules"
    )
    parser.add_argument("--json",            nargs="+", required=True)
    parser.add_argument("--name",            nargs="*", default=None)
    parser.add_argument("--test_csv",        default=DEFAULT_TEST_CSV)
    parser.add_argument("--bdn_csv",         default=DEFAULT_BDN_CSV)
    parser.add_argument("--val_pkl",         default=DEFAULT_VAL_PKL,
                        help=f"validation_set.pkl path (default: {DEFAULT_VAL_PKL})")
    parser.add_argument("--show_per_protein", action="store_true",
                        help="Print per-protein recovery table")
    parser.add_argument("--show_hits",       action="store_true",
                        help="Print which actives were found (verbose)")
    parser.add_argument("--output_json",     default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("Building test-protein known actives …")
    target_to_keys, test_targets = build_target_actives(args.test_csv, args.bdn_csv)

    print("\nBuilding validation_index → target mapping …")
    valindex_map = build_valindex_map(args.val_pkl)

    names = list(args.name or [])
    for p in args.json:
        if len(names) < len(args.json):
            names.append(Path(p).parent.name + "/" + Path(p).stem)

    all_results = []
    for json_path, name in zip(args.json, names):
        print(f"\nProcessing: {json_path}")
        with open(json_path) as f:
            data = json.load(f)

        if "results" in data and "generated_molecules" not in data:
            # Lead-opt format
            result = evaluate_lead_opt(data, valindex_map, target_to_keys,
                                       name=name,
                                       show_per_protein=args.show_per_protein,
                                       show_hits=args.show_hits)
        elif "condition_info" in data:
            # Per-protein conditioning format
            result = evaluate_per_protein(data["condition_info"], valindex_map,
                                          target_to_keys, name=name,
                                          show_per_protein=args.show_per_protein,
                                          show_hits=args.show_hits)
        else:
            # Flat list — no per-protein info
            gen_smiles = data.get("generated_molecules", [])
            result = evaluate_flat(gen_smiles, target_to_keys, name=name)

        per_protein_detail = result.pop("per_protein", [])
        all_results.append(result)

        # Save per-protein CSV alongside the input JSON if requested
        if args.output_json and per_protein_detail:
            pp_df = pd.DataFrame(per_protein_detail)
            pp_path = Path(args.output_json).with_suffix("") / f"{name.replace('/','_')}_per_protein.csv"
            pp_path.parent.mkdir(parents=True, exist_ok=True)
            pp_df.to_csv(pp_path, index=False)
            print(f"  Per-protein table → {pp_path}")

    # Aggregate comparison table
    if len(all_results) > 1:
        metrics = ["total_generated", "total_known_actives", "total_hits",
                   "proteins_with_hits", "mean_recovery_pct", "macro_recovery_pct"]
        metrics = [m for m in metrics if any(m in r for r in all_results)]
        col_w   = max(len(r["name"]) for r in all_results) + 2
        header  = f"{'Metric':<28}" + "".join(f"{r['name']:>{col_w}}" for r in all_results)
        print("\n" + "=" * len(header))
        print(header)
        print("-" * len(header))
        for m in metrics:
            row = f"{m:<28}"
            for r in all_results:
                row += f"{str(r.get(m, '-')):>{col_w}}"
            print(row)
        print("=" * len(header))

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved → {args.output_json}")


if __name__ == "__main__":
    main()
