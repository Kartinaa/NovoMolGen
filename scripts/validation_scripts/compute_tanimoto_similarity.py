#!/usr/bin/env python3
"""
Compare two sets of generated molecules via Tanimoto similarity.

Computes:
  1. Nearest-neighbour (NN) Tanimoto: for each molecule in A, max sim to all of B (and vice versa)
  2. Internal diversity: mean pairwise sim within each set (sampled if large)
  3. Prints summary statistics and saves a histogram

Usage:
    python scripts/validation_scripts/compute_tanimoto_similarity.py \
        --file_a outputs/.../cond/generated_validation.txt \
        --file_b outputs/.../uncond/generated_validation.txt \
        --label_a "cond" --label_b "uncond" \
        --output_dir outputs/tanimoto_analysis/
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm


def load_smiles(path: str) -> list[str]:
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    return lines


def smiles_to_fps(smiles_list: list[str], radius: int = 2, nbits: int = 2048):
    """Return (valid_smiles, fps) with invalid entries dropped."""
    valid, fps = [], []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits))
            valid.append(smi)
    return valid, fps


def nn_tanimoto(fps_query: list, fps_ref: list, desc: str = "") -> np.ndarray:
    """For each query fp, return the max Tanimoto similarity to all ref fps."""
    sims = np.zeros(len(fps_query), dtype=np.float32)
    for i, fp in enumerate(tqdm(fps_query, desc=desc, leave=False)):
        bulk = DataStructs.BulkTanimotoSimilarity(fp, fps_ref)
        sims[i] = max(bulk) if bulk else 0.0
    return sims


def internal_diversity(fps: list, sample_n: int = 2000, seed: int = 42) -> float:
    """Estimate mean pairwise Tanimoto within a set (sampled for speed)."""
    if len(fps) <= 1:
        return float("nan")
    rng = random.Random(seed)
    if len(fps) > sample_n:
        fps = rng.sample(fps, sample_n)
    total, count = 0.0, 0
    for i in range(len(fps)):
        bulk = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        total += sum(bulk)
        count += len(bulk)
    return total / count if count > 0 else float("nan")


def print_stats(label: str, arr: np.ndarray):
    print(f"  {label}:")
    print(f"    n={len(arr)}  mean={arr.mean():.4f}  median={np.median(arr):.4f}"
          f"  std={arr.std():.4f}  min={arr.min():.4f}  max={arr.max():.4f}")
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
        pct = (arr >= thr).mean() * 100
        print(f"    ≥ {thr:.1f}: {pct:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_a", required=True)
    parser.add_argument("--file_b", required=True)
    parser.add_argument("--label_a", default="A")
    parser.add_argument("--label_b", default="B")
    parser.add_argument("--radius", type=int, default=2, help="Morgan fingerprint radius")
    parser.add_argument("--nbits", type=int, default=2048)
    parser.add_argument("--output_dir", default=None, help="Directory to save JSON results")
    parser.add_argument("--diversity_sample", type=int, default=2000,
                        help="Max molecules sampled for internal diversity (expensive)")
    args = parser.parse_args()

    print(f"Loading {args.label_a} from: {args.file_a}")
    smiles_a = load_smiles(args.file_a)
    print(f"Loading {args.label_b} from: {args.file_b}")
    smiles_b = load_smiles(args.file_b)

    print(f"\nComputing fingerprints (Morgan r={args.radius}, {args.nbits} bits)...")
    valid_a, fps_a = smiles_to_fps(smiles_a, args.radius, args.nbits)
    valid_b, fps_b = smiles_to_fps(smiles_b, args.radius, args.nbits)
    print(f"  {args.label_a}: {len(smiles_a)} loaded → {len(valid_a)} valid")
    print(f"  {args.label_b}: {len(smiles_b)} loaded → {len(valid_b)} valid")

    # ── Nearest-neighbour Tanimoto ──────────────────────────────────────
    print(f"\nNearest-neighbour Tanimoto ({args.label_a} → {args.label_b})...")
    nn_a2b = nn_tanimoto(fps_a, fps_b, desc=f"{args.label_a}→{args.label_b}")

    print(f"Nearest-neighbour Tanimoto ({args.label_b} → {args.label_a})...")
    nn_b2a = nn_tanimoto(fps_b, fps_a, desc=f"{args.label_b}→{args.label_a}")

    print(f"\n{'='*55}")
    print("Nearest-neighbour Tanimoto summary")
    print(f"{'='*55}")
    print_stats(f"{args.label_a} → {args.label_b} (how well uncond covers cond)", nn_a2b)
    print_stats(f"{args.label_b} → {args.label_a} (how well cond covers uncond)", nn_b2a)

    # ── Internal diversity ──────────────────────────────────────────────
    print(f"\nInternal diversity (sampled ≤{args.diversity_sample} pairs each)...")
    div_a = internal_diversity(fps_a, args.diversity_sample)
    div_b = internal_diversity(fps_b, args.diversity_sample)
    print(f"  {args.label_a} internal mean pairwise sim: {div_a:.4f}  (diversity = {1-div_a:.4f})")
    print(f"  {args.label_b} internal mean pairwise sim: {div_b:.4f}  (diversity = {1-div_b:.4f})")

    # ── Save results ────────────────────────────────────────────────────
    results = {
        "file_a": args.file_a, "label_a": args.label_a,
        "file_b": args.file_b, "label_b": args.label_b,
        "valid_a": len(valid_a), "valid_b": len(valid_b),
        "nn_a2b": {
            "mean": float(nn_a2b.mean()), "median": float(np.median(nn_a2b)),
            "std": float(nn_a2b.std()), "min": float(nn_a2b.min()), "max": float(nn_a2b.max()),
        },
        "nn_b2a": {
            "mean": float(nn_b2a.mean()), "median": float(np.median(nn_b2a)),
            "std": float(nn_b2a.std()), "min": float(nn_b2a.min()), "max": float(nn_b2a.max()),
        },
        "internal_diversity": {
            args.label_a: {"mean_sim": float(div_a), "diversity": float(1 - div_a)},
            args.label_b: {"mean_sim": float(div_b), "diversity": float(1 - div_b)},
        },
    }

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        json_path = out / f"tanimoto_{args.label_a}_vs_{args.label_b}.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {json_path}")

        # Save raw NN arrays for downstream plotting
        np.save(out / f"nn_{args.label_a}2{args.label_b}.npy", nn_a2b)
        np.save(out / f"nn_{args.label_b}2{args.label_a}.npy", nn_b2a)
        print(f"NN arrays saved to: {out}/nn_*.npy")

    print(f"\n{'='*55}")
    print("Done.")


if __name__ == "__main__":
    main()
