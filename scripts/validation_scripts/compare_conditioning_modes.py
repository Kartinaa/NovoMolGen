#!/usr/bin/env python3
"""
Compare SA / QED / validity / uniqueness across three conditioning modes:

  full        — protein + ligand conditioning (normal)
  protein_only— protein only (ligand → null_ligand_emb)
  zero_cond   — all condition vectors zeroed (true unconditional baseline)

If protein_only ≈ zero_cond  → protein conditioning is not contributing
If full        ≈ protein_only → ligand conditioning is not contributing

Usage:
    python scripts/validation_scripts/compare_conditioning_modes.py \
        --model_path outputs/models/.../checkpoint-XXXX/full_model \
        --validation_set finetune_data/processed_data/hf_bdnv2_dataset/validation \
        --num_samples_per_target 20 \
        --num_targets 30 \
        --batch_size 20 \
        --convert_smiles_to_safe
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from datasets import load_from_disk, load_dataset, Features, Value, Sequence
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from models.modeling_novomolgen_infonce_120225_v4 import NovoMolGen, NovoMolGenConfig  # type: ignore
from data_loader.utils import safe_to_smiles
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("compare_cond_modes")
    logger.setLevel(getattr(logging, level.upper()))
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(h)
    return logger


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_path: str, logger: logging.Logger):
    mp = Path(model_path)
    config = NovoMolGenConfig.from_pretrained(str(mp))
    if getattr(config, "train_new_modules_only", False):
        config.train_new_modules_only = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = NovoMolGen.from_pretrained(
        str(mp), config=config, torch_dtype=model_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    ).to(device)
    if torch.cuda.is_available():
        model = model.to(model_dtype)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(str(mp))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Model on {device}, dtype={next(model.parameters()).dtype}")
    return model, tokenizer, device


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset_safe(path: str, logger: logging.Logger):
    logger.info(f"Loading dataset from: {path}")
    try:
        return load_from_disk(path)
    except Exception as e:
        logger.warning(f"load_from_disk failed ({e}), trying arrow fallback...")
        arrow_files = sorted(Path(path).glob("*.arrow"))
        if not arrow_files:
            raise ValueError(f"No arrow files found in {path}")
        features = Features({
            "SMILES":      Value("string"),
            "pocket_vec":  Sequence(Value("float64")),
            "evo_vec":     Sequence(Value("float32")),
            "ifp":         Sequence(Value("float32")),
            "ligand_vec":  Sequence(Value("float32")),
        })
        return load_dataset("arrow", data_files=[str(f) for f in arrow_files],
                            features=features, split="train")


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def _generate_for_sample(
    model, tokenizer, device, model_dtype,
    pocket_vec: torch.Tensor,  # [1, D]
    evo_vec: torch.Tensor,
    ifp: torch.Tensor,
    ligand_vec: torch.Tensor,
    mode: str,                 # "full" | "protein_only" | "zero_cond"
    num_samples: int,
    batch_size: int,
    max_length: int,
    temperature: float,
    top_k: int,
    top_p: float,
    max_retries: int,
    convert_to_safe: bool,
) -> List[str]:
    """Generate `num_samples` valid SMILES for one validation sample under `mode`."""
    results: List[str] = []
    attempts = 0
    max_total = max_retries * max(1, num_samples // batch_size + 1)

    while len(results) < num_samples and attempts < max_total:
        needed = num_samples - len(results)
        bs = min(batch_size, needed)

        # Build condition tensors for this mode
        pv = pocket_vec.repeat(bs, 1)
        ev = evo_vec.repeat(bs, 1)

        if mode == "protein_only":
            ifp_b = None
            lig_b = None
        elif mode == "zero_cond":
            pv = torch.zeros_like(pv)
            ev = torch.zeros_like(ev)
            ifp_b = torch.zeros(bs, ifp.shape[-1], dtype=model_dtype, device=device)
            lig_b = torch.zeros(bs, ligand_vec.shape[-1], dtype=model_dtype, device=device)
        else:  # full
            ifp_b = ifp.repeat(bs, 1)
            lig_b = ligand_vec.repeat(bs, 1)

        input_ids = torch.tensor(
            [[tokenizer.bos_token_id]], device=device
        ).repeat(bs, 1)

        with torch.no_grad():
            autocast_on = device.type == "cuda" and model_dtype == torch.bfloat16
            ctx = torch.autocast("cuda", dtype=model_dtype) if autocast_on else torch.inference_mode()
            with ctx:
                seqs = model.generate_with_condition(
                    input_ids=input_ids,
                    pocket_vec=pv,
                    evo_vec=ev,
                    ifp=ifp_b,
                    ligand_vec=lig_b,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    eos_token_id=tokenizer.eos_token_id,
                )

        seqs = model._filter_tokens_after_eos(seqs, eos_id=tokenizer.eos_token_id)
        decoded = tokenizer.batch_decode(seqs, skip_special_tokens=True)
        decoded = [s.replace(" ", "") for s in decoded]

        for safe_str in decoded:
            if len(results) >= num_samples:
                break
            try:
                smi = safe_to_smiles(safe_str)
                if smi and "." not in smi:
                    results.append(smi)
            except Exception:
                pass

        attempts += 1

    return results


# ---------------------------------------------------------------------------
# Property computation
# ---------------------------------------------------------------------------

def compute_properties(smiles_list: List[str]) -> Dict:
    """Compute validity, SA, QED, uniqueness for a list of SMILES."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    # Use the project's eval module which bundles sascorer
    try:
        from eval.components.moses import SA as SA_fn, QED as QED_fn  # type: ignore
        has_sa = True
    except ImportError:
        has_sa = False
        SA_fn = None
        QED_fn = None

    valid, sa_scores, qed_scores, canonical_set = [], [], [], set()
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
        except Exception:
            continue
        canon = Chem.MolToSmiles(mol)
        canonical_set.add(canon)
        valid.append(canon)
        if QED_fn is not None:
            try:
                qed_scores.append(QED_fn(mol))
            except Exception:
                pass
        if SA_fn is not None:
            try:
                sa_scores.append(SA_fn(mol))
            except Exception:
                pass

    n_total = len(smiles_list)
    n_valid = len(valid)
    return {
        "n_total":    n_total,
        "n_valid":    n_valid,
        "validity":   n_valid / n_total if n_total > 0 else 0.0,
        "n_unique":   len(canonical_set),
        "uniqueness": len(canonical_set) / n_valid if n_valid > 0 else 0.0,
        "sa_mean":    sum(sa_scores) / len(sa_scores) if sa_scores else float("nan"),
        "qed_mean":   sum(qed_scores) / len(qed_scores) if qed_scores else float("nan"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare SA/QED/validity across full / protein_only / zero_cond generation"
    )
    parser.add_argument("--model_path",        required=True)
    parser.add_argument("--validation_set",    required=True)
    parser.add_argument("--num_samples_per_target", type=int, default=20,
                        help="Molecules to generate per validation sample (default 20)")
    parser.add_argument("--num_targets",       type=int, default=30,
                        help="How many validation samples to use (default 30)")
    parser.add_argument("--batch_size",        type=int, default=20)
    parser.add_argument("--max_length",        type=int, default=64)
    parser.add_argument("--temperature",       type=float, default=1.0)
    parser.add_argument("--top_k",             type=int, default=50)
    parser.add_argument("--top_p",             type=float, default=0.95)
    parser.add_argument("--max_retries",       type=int, default=10)
    parser.add_argument("--modes",             nargs="+",
                        default=["full", "protein_only", "zero_cond"],
                        help="Which modes to run (any subset of: full protein_only zero_cond)")
    parser.add_argument("--convert_smiles_to_safe", action="store_true",
                        help="Convert SMILES column to SAFE before tokenization (not used in generation, "
                             "but noted for dataset compatibility)")
    parser.add_argument("--log_level",         default="INFO")
    args = parser.parse_args()

    logger = setup_logging(args.log_level)
    logger.info("=" * 65)
    logger.info("Conditioning Mode Comparison")
    logger.info(f"  Modes:   {args.modes}")
    logger.info(f"  Targets: {args.num_targets}")
    logger.info(f"  Samples per target: {args.num_samples_per_target}")
    logger.info("=" * 65)

    model, tokenizer, device = load_model_and_tokenizer(args.model_path, logger)
    model_dtype = next(model.parameters()).dtype
    ds = load_dataset_safe(args.validation_set, logger)
    logger.info(f"Dataset: {len(ds)} samples")

    num_targets = min(args.num_targets, len(ds))

    # Collect all generated SMILES per mode
    all_smiles: Dict[str, List[str]] = {m: [] for m in args.modes}

    for idx in tqdm(range(num_targets), desc="Targets"):
        sample = ds[idx]
        pocket_vec  = torch.tensor([sample["pocket_vec"]], dtype=model_dtype, device=device)
        evo_vec     = torch.tensor([sample["evo_vec"]],    dtype=model_dtype, device=device)
        ifp         = torch.tensor([sample["ifp"]],        dtype=model_dtype, device=device)
        ligand_vec  = torch.tensor([sample["ligand_vec"]], dtype=model_dtype, device=device)

        for mode in args.modes:
            smiles = _generate_for_sample(
                model=model, tokenizer=tokenizer, device=device, model_dtype=model_dtype,
                pocket_vec=pocket_vec, evo_vec=evo_vec, ifp=ifp, ligand_vec=ligand_vec,
                mode=mode,
                num_samples=args.num_samples_per_target,
                batch_size=args.batch_size,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                max_retries=args.max_retries,
                convert_to_safe=args.convert_smiles_to_safe,
            )
            all_smiles[mode].extend(smiles)

    # Compute properties per mode
    logger.info("\n" + "=" * 65)
    logger.info("Results")
    logger.info("=" * 65)
    header = f"{'Mode':<14} {'N_total':>8} {'validity%':>10} {'unique%':>9} {'SA_mean':>9} {'QED_mean':>9}"
    logger.info(header)
    logger.info("-" * 65)

    results = {}
    for mode in args.modes:
        props = compute_properties(all_smiles[mode])
        results[mode] = props
        sa_str  = f"{props['sa_mean']:.3f}"  if props['sa_mean'] == props['sa_mean']  else "  N/A  "
        qed_str = f"{props['qed_mean']:.3f}" if props['qed_mean'] == props['qed_mean'] else "  N/A  "
        logger.info(
            f"{mode:<14} {props['n_total']:>8} "
            f"{props['validity']*100:>9.1f}% "
            f"{props['uniqueness']*100:>8.1f}% "
            f"{sa_str:>9} "
            f"{qed_str:>9}"
        )

    logger.info("=" * 65)

    # Interpretation
    logger.info("\nInterpretation:")
    modes = args.modes
    if "full" in modes and "protein_only" in modes:
        dv = results["full"]["validity"] - results["protein_only"]["validity"]
        dq = results["full"]["qed_mean"] - results["protein_only"]["qed_mean"]
        ds_a = results["full"]["sa_mean"]  - results["protein_only"]["sa_mean"]
        logger.info(f"  full vs protein_only → validity Δ={dv*100:+.1f}%  QED Δ={dq:+.4f}  SA Δ={ds_a:+.4f}")
        if abs(dv) < 0.02 and abs(dq) < 0.01:
            logger.info("  → Ligand conditioning has NO measurable effect on validity/QED")
        else:
            logger.info("  → Ligand conditioning IS changing validity/QED")

    if "protein_only" in modes and "zero_cond" in modes:
        dv = results["protein_only"]["validity"] - results["zero_cond"]["validity"]
        dq = results["protein_only"]["qed_mean"] - results["zero_cond"]["qed_mean"]
        ds_a = results["protein_only"]["sa_mean"]  - results["zero_cond"]["sa_mean"]
        logger.info(f"  protein_only vs zero_cond → validity Δ={dv*100:+.1f}%  QED Δ={dq:+.4f}  SA Δ={ds_a:+.4f}")
        if abs(dv) < 0.02 and abs(dq) < 0.01:
            logger.info("  → Protein conditioning has NO measurable effect on validity/QED")
        else:
            logger.info("  → Protein conditioning IS changing validity/QED")

    if "full" in modes and "zero_cond" in modes:
        dv = results["full"]["validity"] - results["zero_cond"]["validity"]
        dq = results["full"]["qed_mean"] - results["zero_cond"]["qed_mean"]
        logger.info(f"  full vs zero_cond → validity Δ={dv*100:+.1f}%  QED Δ={dq:+.4f}")
        if abs(dv) < 0.02 and abs(dq) < 0.01:
            logger.info("  → Overall conditioning has NO measurable effect on validity/QED")
        else:
            logger.info("  → Overall conditioning IS changing validity/QED")

    return 0


if __name__ == "__main__":
    sys.exit(main())
