#!/usr/bin/env python3
"""
Evaluate per-group metrics for lead_optimization_valset_results.json output.

Reads the "results" list from lead_optimization_from_valset.py output and
computes per-group and overall metrics:
  - SA, QED (drug-likeness)
  - Unique count / ratio
  - Novelty (fraction distinct from reference_smiles)
  - Tanimoto similarity to reference_smiles (mean, max)
  - Overall generation stats (validity, fragment rate, etc.)

Usage:
    python scripts/validation_scripts/evaluate_lead_opt_metrics.py \
        --input_json outputs/lead_opt_valset/.../lead_optimization_valset_results.json \
        --output_json outputs/lead_opt_valset/.../lead_opt_metrics.json \
        --n_jobs 16
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors
from tqdm import tqdm
from multiprocess.pool import Pool

from src.eval.components.moses import SA, QED
from src.eval.utils import get_mol


# ---------------------------------------------------------------------------
# Per-molecule worker (called in parallel)
# ---------------------------------------------------------------------------

def _process_smiles(smiles: str) -> Optional[Dict[str, Any]]:
    mol = get_mol(smiles)
    if mol is None:
        return None
    try:
        sa_score  = SA(mol)
        qed_score = QED(mol)
        canonical = Chem.MolToSmiles(mol)
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    except Exception:
        return None
    return {"sa": sa_score, "qed": qed_score, "canonical": canonical, "fp": fp}


def _process_batch(smiles_list: List[str], n_jobs: int) -> List[Optional[Dict]]:
    if n_jobs <= 1:
        return [_process_smiles(s) for s in tqdm(smiles_list, desc="Processing", unit="mol")]
    with Pool(n_jobs) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(_process_smiles, smiles_list, chunksize=64),
                total=len(smiles_list),
                desc="Processing",
                unit="mol",
            )
        )
    return results


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_lead_opt(
    input_json_path: str,
    output_json_path: str,
    logger: logging.Logger,
    n_jobs: int = 16,
) -> Dict[str, Any]:

    logger.info(f"Loading: {input_json_path}")
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "results" not in data:
        raise ValueError("JSON must contain a 'results' key (lead_optimization_valset_results.json format)")

    results_list = data["results"]
    logger.info(f"Found {len(results_list)} groups")

    # ------------------------------------------------------------------
    # Flatten all generated molecules for batch processing
    # ------------------------------------------------------------------
    all_smiles:    List[str] = []
    group_of:      List[int] = []   # which group each molecule belongs to
    ref_smiles_of: List[str] = []   # reference smiles for that group

    groups_meta = []
    for g, entry in enumerate(results_list):
        gen_mols = entry.get("generated_molecules", [])
        ref      = entry.get("reference_smiles", "")
        groups_meta.append({
            "val_idx":          entry.get("val_idx", g),
            "reference_smiles": ref,
            "removed_fragment": entry.get("removed_fragment", ""),
            "num_safe_parts":   entry.get("num_safe_parts", 0),
            "conversion_stats": entry.get("conversion_stats", {}),
            "num_generated":    len(gen_mols),
        })
        for smi in gen_mols:
            all_smiles.append(smi)
            group_of.append(g)
            ref_smiles_of.append(ref)

    num_groups = len(groups_meta)
    logger.info(f"Total molecules to evaluate: {len(all_smiles)}")

    # ------------------------------------------------------------------
    # Pre-compute reference fingerprints (one per group)
    # ------------------------------------------------------------------
    ref_fps: List[Optional[DataStructs.ExplicitBitVect]] = []
    ref_canonicals: List[Optional[str]] = []
    for meta in groups_meta:
        ref_mol = get_mol(meta["reference_smiles"])
        if ref_mol is not None:
            try:
                ref_fps.append(
                    rdMolDescriptors.GetMorganFingerprintAsBitVect(ref_mol, radius=2, nBits=2048)
                )
                ref_canonicals.append(Chem.MolToSmiles(ref_mol))
            except Exception:
                ref_fps.append(None)
                ref_canonicals.append(None)
        else:
            ref_fps.append(None)
            ref_canonicals.append(None)

    # ------------------------------------------------------------------
    # Parallel SA / QED / canonicalization
    # ------------------------------------------------------------------
    logger.info(f"Computing SA, QED, fingerprints with {n_jobs} workers...")
    processed = _process_batch(all_smiles, n_jobs)

    # Note: imap_unordered returns results out of order.
    # Re-map by (smiles → result) is unreliable if duplicates exist,
    # so we ran imap and just accumulated — but order is lost.
    # Use imap (ordered) instead for correctness.
    if n_jobs > 1:
        # Re-run with ordered imap to preserve group_of alignment
        with Pool(n_jobs) as pool:
            processed = list(
                tqdm(
                    pool.imap(_process_smiles, all_smiles, chunksize=64),
                    total=len(all_smiles),
                    desc="Processing (ordered)",
                    unit="mol",
                )
            )

    # ------------------------------------------------------------------
    # Aggregate per group
    # ------------------------------------------------------------------
    g_sa:       List[List[float]] = [[] for _ in range(num_groups)]
    g_qed:      List[List[float]] = [[] for _ in range(num_groups)]
    g_unique:   List[set]         = [set() for _ in range(num_groups)]
    g_fps:      List[list]        = [[] for _ in range(num_groups)]

    for i, res in enumerate(processed):
        g = group_of[i]
        if res is None:
            continue
        g_sa[g].append(res["sa"])
        g_qed[g].append(res["qed"])
        g_unique[g].add(res["canonical"])
        g_fps[g].append(res["fp"])

    # ------------------------------------------------------------------
    # Per-group metrics
    # ------------------------------------------------------------------
    per_group = []
    agg = {k: [] for k in [
        "sa_mean", "qed_mean", "unique_ratio", "unique_count",
        "novelty", "tanimoto_to_ref_mean", "tanimoto_to_ref_max",
        "validity_rate",
    ]}

    for g, meta in enumerate(groups_meta):
        valid_cnt = len(g_sa[g])
        total_cnt = meta["num_generated"]   # already-valid molecules in JSON
        unique_cnt = len(g_unique[g])

        sa_mean  = sum(g_sa[g])  / valid_cnt if valid_cnt else 0.0
        qed_mean = sum(g_qed[g]) / valid_cnt if valid_cnt else 0.0
        unique_ratio = unique_cnt / total_cnt if total_cnt else 0.0

        # Novelty: fraction not identical to reference
        ref_can = ref_canonicals[g]
        if ref_can is not None and valid_cnt > 0:
            novel_cnt = sum(1 for can in g_unique[g] if can != ref_can)
            novelty = novel_cnt / unique_cnt if unique_cnt else 0.0
        else:
            novelty = None

        # Tanimoto similarity to reference
        ref_fp = ref_fps[g]
        if ref_fp is not None and g_fps[g]:
            sims = []
            for fp in g_fps[g]:
                try:
                    sims.append(DataStructs.TanimotoSimilarity(fp, ref_fp))
                except Exception:
                    pass
            tanimoto_mean = sum(sims) / len(sims) if sims else None
            tanimoto_max  = max(sims) if sims else None
        else:
            tanimoto_mean = None
            tanimoto_max  = None

        # Overall validity from conversion_stats
        cs = meta["conversion_stats"]
        total_safe = cs.get("total_safe_generated", 0)
        success    = cs.get("conversion_success", total_cnt)
        validity_rate = success / total_safe if total_safe > 0 else None

        gm = {
            "val_idx":             meta["val_idx"],
            "reference_smiles":    meta["reference_smiles"],
            "removed_fragment":    meta["removed_fragment"],
            "num_safe_parts":      meta["num_safe_parts"],
            "num_generated":       total_cnt,
            "valid_count":         valid_cnt,
            "sa_mean":             round(sa_mean, 4),
            "qed_mean":            round(qed_mean, 4),
            "unique_count":        unique_cnt,
            "unique_ratio":        round(unique_ratio, 4),
            "novelty":             round(novelty, 4) if novelty is not None else None,
            "tanimoto_to_ref_mean": round(tanimoto_mean, 4) if tanimoto_mean is not None else None,
            "tanimoto_to_ref_max":  round(tanimoto_max,  4) if tanimoto_max  is not None else None,
            "validity_rate":        round(validity_rate, 6) if validity_rate is not None else None,
            "conversion_stats":    cs,
        }
        per_group.append(gm)

        if valid_cnt > 0:
            agg["sa_mean"].append(sa_mean)
            agg["qed_mean"].append(qed_mean)
            agg["unique_ratio"].append(unique_ratio)
            agg["unique_count"].append(unique_cnt)
        if novelty is not None:
            agg["novelty"].append(novelty)
        if tanimoto_mean is not None:
            agg["tanimoto_to_ref_mean"].append(tanimoto_mean)
        if tanimoto_max is not None:
            agg["tanimoto_to_ref_max"].append(tanimoto_max)
        if validity_rate is not None:
            agg["validity_rate"].append(validity_rate)

    # ------------------------------------------------------------------
    # Overall stats
    # ------------------------------------------------------------------
    def _avg(lst):
        return round(sum(lst) / len(lst), 4) if lst else None

    # Global unique across all groups
    global_unique: set = set()
    for s in g_unique:
        global_unique.update(s)
    total_valid_all = sum(len(g_sa[g]) for g in range(num_groups))

    # Overall conversion stats summed
    total_safe_all   = sum(gm["conversion_stats"].get("total_safe_generated", 0) for gm in per_group)
    success_all      = sum(gm["conversion_stats"].get("conversion_success", 0)   for gm in per_group)
    fragments_all    = sum(gm["conversion_stats"].get("conversion_fragments", 0) for gm in per_group)
    empty_all        = sum(gm["conversion_stats"].get("conversion_empty", 0)     for gm in per_group)
    duplicates_all   = sum(gm["conversion_stats"].get("duplicates_skipped", 0)   for gm in per_group)

    overall = {
        "num_groups":              len(per_group),
        "sa_mean":                 _avg(agg["sa_mean"]),
        "qed_mean":                _avg(agg["qed_mean"]),
        "unique_count_mean":       _avg(agg["unique_count"]),
        "unique_ratio_mean":       _avg(agg["unique_ratio"]),
        "novelty_mean":            _avg(agg["novelty"]),
        "tanimoto_to_ref_mean":    _avg(agg["tanimoto_to_ref_mean"]),
        "tanimoto_to_ref_max_mean": _avg(agg["tanimoto_to_ref_max"]),
        "validity_rate_mean":      _avg(agg["validity_rate"]),
        "global_unique_count":     len(global_unique),
        "global_unique_rate":      round(len(global_unique) / total_valid_all, 4) if total_valid_all else 0.0,
        "total_safe_generated":    total_safe_all,
        "total_conversion_success": success_all,
        "total_conversion_fragments": fragments_all,
        "total_conversion_empty":  empty_all,
        "total_duplicates_skipped": duplicates_all,
        "overall_validity_rate":   round(success_all / total_safe_all, 6) if total_safe_all else None,
    }

    output = {
        "input_json":      str(input_json_path),
        "overall_metrics": overall,
        "per_group_metrics": per_group,
    }

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_path = Path(output_json_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved to: {out_path}")

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("Lead Optimization Evaluation Summary")
    logger.info("=" * 60)
    logger.info(f"Groups evaluated          : {overall['num_groups']}")
    logger.info(f"SA mean                   : {overall['sa_mean']}")
    logger.info(f"QED mean                  : {overall['qed_mean']}")
    logger.info(f"Unique count (mean/group) : {overall['unique_count_mean']}")
    logger.info(f"Unique ratio (mean/group) : {overall['unique_ratio_mean']}")
    logger.info(f"Novelty (mean/group)      : {overall['novelty_mean']}")
    logger.info(f"Tanimoto to ref (mean)    : {overall['tanimoto_to_ref_mean']}")
    logger.info(f"Tanimoto to ref (max avg) : {overall['tanimoto_to_ref_max_mean']}")
    logger.info(f"Global unique count       : {overall['global_unique_count']}")
    logger.info(f"Overall validity rate     : {overall['overall_validity_rate']}")
    logger.info("=" * 60)

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate lead optimization results (lead_optimization_valset_results.json)"
    )
    parser.add_argument("--input_json",  required=True,
                        help="Path to lead_optimization_valset_results.json")
    parser.add_argument("--output_json", default=None,
                        help="Path to save metrics JSON (default: same dir as input, lead_opt_metrics.json)")
    parser.add_argument("--n_jobs", type=int, default=16,
                        help="Parallel workers for RDKit (default: 16)")
    parser.add_argument("--log_level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=getattr(logging, args.log_level),
    )
    logger = logging.getLogger("eval_lead_opt")

    input_path = Path(args.input_json)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1

    output_path = args.output_json or str(input_path.parent / "lead_opt_metrics.json")

    try:
        evaluate_lead_opt(
            input_json_path=str(input_path),
            output_json_path=output_path,
            logger=logger,
            n_jobs=args.n_jobs,
        )
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
