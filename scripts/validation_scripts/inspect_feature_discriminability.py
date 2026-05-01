#!/usr/bin/env python3
"""
診斷 condition 特征 (pocket_vec / evo_vec / ifp / ligand_vec) 在數據集上的區分度，
以及它們經過 protein_encoder / ligand_encoder 後是否仍然區分。

核心問題：
  - 如果輸入特征本身就近乎一樣 (pairwise cos sim 接近 1)
    → 是 *特征問題*，需要換更強的表示 (ESM-3 / 結構直接輸入)
  - 如果輸入區分度高但 encoder 輸出很相似
    → 是 *encoder collapse*，需要加 contrastive loss / 改架構
  - 如果輸入和 encoder 輸出都區分度高
    → 問題在下游（cross-attn / gate / 訓練不足）

用法：
  python scripts/validation_scripts/inspect_feature_discriminability.py \
    --model_path outputs/.../full_model \
    --validation_set finetune_data/processed_data/hf_bdnv2_dataset/validation
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple

import torch
import numpy as np
from datasets import load_from_disk

# 把 src 加到路径
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from models.modeling_novomolgen_infonce_120225_v4 import NovoMolGen, NovoMolGenConfig  # type: ignore


def setup_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("inspect_feature_discriminability")
    logger.setLevel(getattr(logging, level.upper()))
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(h)
    return logger


def cosine_pairwise_stats(features: torch.Tensor) -> dict:
    """Compute pairwise cosine similarity statistics for [N, D] tensor."""
    feats = features.float()
    norms = feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    normed = feats / norms
    sim = normed @ normed.T  # [N, N]

    n = sim.size(0)
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    pairs = sim[mask]

    pairs_np = pairs.numpy()
    return {
        "n_pairs": int(pairs.numel()),
        "mean":    float(np.mean(pairs_np)),
        "std":     float(np.std(pairs_np)),
        "min":     float(np.min(pairs_np)),
        "p10":     float(np.percentile(pairs_np, 10)),
        "median":  float(np.median(pairs_np)),
        "p90":     float(np.percentile(pairs_np, 90)),
        "max":     float(np.max(pairs_np)),
    }


def effective_rank(features: torch.Tensor, threshold: float = 0.99) -> Tuple[int, int]:
    """
    Effective rank via SVD: 留多少個奇異值才能解釋 99% 的能量。
    Returns: (effective_rank_99, total_dim)
    """
    feats = features.float()
    feats_centered = feats - feats.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(feats_centered)
    s2 = s ** 2
    cumsum = torch.cumsum(s2, dim=0) / s2.sum()
    eff_rank = int((cumsum < threshold).sum().item()) + 1
    return eff_rank, feats.size(1)


def log_stats(logger: logging.Logger, name: str, stats: dict, eff_rank_info=None):
    logger.info(f"\n  ── {name} ──")
    logger.info(f"    pairwise cos sim: mean={stats['mean']:+.4f}  std={stats['std']:.4f}")
    logger.info(f"    distribution    : min={stats['min']:+.4f}  p10={stats['p10']:+.4f}  median={stats['median']:+.4f}  p90={stats['p90']:+.4f}  max={stats['max']:+.4f}")
    logger.info(f"    n_pairs         : {stats['n_pairs']}")
    if eff_rank_info is not None:
        eff, total = eff_rank_info
        logger.info(f"    effective rank  : {eff} / {total} ({100*eff/total:.1f}% of full dim explains 99% variance)")


def diagnose(
    raw_stats: dict, enc_stats: dict, name_raw: str, name_enc: str, logger: logging.Logger
):
    """Diagnostic verdict comparing raw input vs encoder output similarity."""
    logger.info(f"\n  → {name_raw}  →  {name_enc}")
    logger.info(f"    raw pairwise sim mean    : {raw_stats['mean']:+.4f}")
    logger.info(f"    encoded pairwise sim mean: {enc_stats['mean']:+.4f}")
    delta = enc_stats["mean"] - raw_stats["mean"]
    logger.info(f"    Δ (encoded - raw)        : {delta:+.4f}")

    if raw_stats["mean"] > 0.95:
        verdict = (
            "⚠ INPUT FEATURE PROBLEM: 原始特征已經高度相似 (mean > 0.95)，"
            "encoder 怎麼學都救不回來。建議換更強的表示。"
        )
    elif enc_stats["mean"] > 0.9 and raw_stats["mean"] < 0.7:
        verdict = (
            "⚠ ENCODER COLLAPSE: 原始特征區分度好，但 encoder 把它們壓到了一起。"
            "建議加 contrastive loss 或重新設計 encoder。"
        )
    elif enc_stats["mean"] > 0.85:
        verdict = (
            "△ MODERATE COLLAPSE: encoder 輸出仍然偏相似，可能訓練不足 "
            "或表示能力有限。"
        )
    else:
        verdict = "✓ 表示空間區分度合理。問題可能在下游 (cross-attn / gate / 訓練步數)。"

    logger.info(f"    verdict: {verdict}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to full_model directory")
    parser.add_argument("--validation_set", type=str, required=True,
                        help="Path to HF dataset with condition features")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Optional cap on samples (default: all)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for encoder forward (default 64)")
    parser.add_argument("--log_level", type=str, default="INFO")
    args = parser.parse_args()

    logger = setup_logging(args.log_level)

    # ── Load model (only need encoders) ─────────────────────────────────────
    logger.info("=" * 70)
    logger.info("Feature Discriminability Inspector")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Dataset: {args.validation_set}")

    config = NovoMolGenConfig.from_pretrained(args.model_path)
    if getattr(config, "train_new_modules_only", False):
        config.train_new_modules_only = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = NovoMolGen.from_pretrained(
        args.model_path, config=config, torch_dtype=model_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    ).to(device).eval()
    if torch.cuda.is_available() and model_dtype == torch.bfloat16:
        model = model.to(torch.bfloat16)

    # ── Load data ───────────────────────────────────────────────────────────
    ds = load_from_disk(args.validation_set)
    n_total = len(ds)
    n = n_total if args.max_samples is None else min(n_total, args.max_samples)
    logger.info(f"Using {n} / {n_total} samples")

    pocket_list, evo_list, ifp_list, lig_list = [], [], [], []
    for i in range(n):
        s = ds[i]
        pocket_list.append(s["pocket_vec"])
        evo_list.append(s["evo_vec"])
        ifp_list.append(s["ifp"])
        lig_list.append(s["ligand_vec"])

    pocket = torch.tensor(pocket_list, dtype=torch.float32)
    evo    = torch.tensor(evo_list,    dtype=torch.float32)
    ifp    = torch.tensor(ifp_list,    dtype=torch.float32)
    lig    = torch.tensor(lig_list,    dtype=torch.float32)
    logger.info(f"  pocket_vec shape: {tuple(pocket.shape)}")
    logger.info(f"  evo_vec    shape: {tuple(evo.shape)}")
    logger.info(f"  ifp        shape: {tuple(ifp.shape)}")
    logger.info(f"  ligand_vec shape: {tuple(lig.shape)}")

    # ── Check duplicate ligand_path / unique-target count if available ──────
    if "ligand_path" in ds.column_names:
        paths = [ds[i]["ligand_path"] for i in range(n)]
        # Extract target name (the parent's parent of ligand.sdf)
        targets = []
        for p in paths:
            if p:
                parts = Path(p).parts
                tgt = next((x for x in parts if x.startswith("target_")), None)
                if tgt is None and len(parts) >= 3:
                    tgt = parts[-3]
                targets.append(tgt)
        n_unique_targets = len(set([t for t in targets if t]))
        logger.info(f"  unique targets in this set: {n_unique_targets} / {n}")
        if n_unique_targets < n:
            logger.info(f"  → 同一個 target 的多個 ligand 會共享 pocket_vec，"
                        f"這會拉高 pocket 的 pairwise 相似度。後續結果需要結合這個事實理解。")

    # ── Stage 1: Raw input feature similarity ──────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 1: Raw input feature pairwise similarity")
    logger.info("=" * 70)

    raw_stats = {}
    for name, feat in [("pocket_vec", pocket), ("evo_vec", evo),
                       ("ifp", ifp), ("ligand_vec", lig)]:
        s = cosine_pairwise_stats(feat)
        eff = effective_rank(feat)
        log_stats(logger, name, s, eff)
        raw_stats[name] = s

    # ── Stage 2: After encoder ─────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 2: After protein_encoder / ligand_encoder")
    logger.info("=" * 70)

    protein_cond_chunks, mu_chunks = [], []
    with torch.no_grad():
        for i in range(0, n, args.batch_size):
            j = min(i + args.batch_size, n)
            p_b = pocket[i:j].to(device).to(model_dtype)
            e_b = evo[i:j].to(device).to(model_dtype)
            f_b = ifp[i:j].to(device).to(model_dtype)
            l_b = lig[i:j].to(device).to(model_dtype)

            protein_cond = model.protein_encoder(p_b, e_b)
            mu, _, _     = model.ligand_encoder(f_b, l_b, return_logvar=True)
            protein_cond_chunks.append(protein_cond.float().cpu())
            mu_chunks.append(mu.float().cpu())

    protein_cond = torch.cat(protein_cond_chunks, dim=0)
    mu           = torch.cat(mu_chunks, dim=0)

    enc_stats = {}
    for name, feat in [("protein_condition (after protein_encoder)", protein_cond),
                       ("mu (after ligand_encoder)", mu)]:
        s = cosine_pairwise_stats(feat)
        eff = effective_rank(feat)
        log_stats(logger, name, s, eff)
        enc_stats[name] = s

    # ── Stage 3: Diagnosis ─────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 3: Diagnosis")
    logger.info("=" * 70)

    # Combine pocket+evo into a notion of "raw protein"; we use evo as reference
    # since pocket might be duplicated within targets.
    # We compare *each* raw input to its corresponding encoded output.
    diagnose(raw_stats["pocket_vec"], enc_stats["protein_condition (after protein_encoder)"],
             "pocket_vec", "protein_condition", logger)
    diagnose(raw_stats["evo_vec"], enc_stats["protein_condition (after protein_encoder)"],
             "evo_vec", "protein_condition", logger)
    diagnose(raw_stats["ifp"], enc_stats["mu (after ligand_encoder)"],
             "ifp", "mu", logger)
    diagnose(raw_stats["ligand_vec"], enc_stats["mu (after ligand_encoder)"],
             "ligand_vec", "mu", logger)

    # ── Comparison: protein vs ligand discriminability ─────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("Cross-comparison")
    logger.info("=" * 70)
    p_cond_mean = enc_stats["protein_condition (after protein_encoder)"]["mean"]
    mu_mean     = enc_stats["mu (after ligand_encoder)"]["mean"]
    logger.info(f"  protein_condition mean pairwise sim: {p_cond_mean:+.4f}")
    logger.info(f"  mu                mean pairwise sim: {mu_mean:+.4f}")
    logger.info(f"  Δ (protein - ligand)               : {p_cond_mean - mu_mean:+.4f}")
    if p_cond_mean - mu_mean > 0.1:
        logger.info("  → protein_condition 的區分度顯著低於 mu，"
                    "這正好解釋了 shuffle protein 對 loss 幾乎無影響、"
                    "而 shuffle ligand 有 ~3% 影響的現象。")
    elif p_cond_mean - mu_mean < -0.1:
        logger.info("  → mu 區分度反而低於 protein_condition，"
                    "和 shuffle 測試結果可能不一致。")
    else:
        logger.info("  → 兩個 encoder 的區分度差不多。")

    logger.info("\nDone.")


if __name__ == "__main__":
    sys.exit(main() or 0)
