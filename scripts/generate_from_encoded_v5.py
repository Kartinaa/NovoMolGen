#!/usr/bin/env python3
"""
Generate molecules from encoded condition vectors using the v5 CFG/Tanimoto model
(multi-token conditioning: n_protein_tokens + n_ligand_tokens, default 4 + 4).

Loads pre-computed condition vectors (pocket_vec, evo_vec, ifp, ligand_vec) from
a directory produced by the encoding pipeline (e.g. p53/encode_p53/), optionally
seeded with a SAFE-string prefix for lead-optimization subtasks (linker design,
scaffold hopping, fragment growing, etc.).

Compatible with models trained with CFG + Tanimoto (e.g. the paper's checkpoint
050826_SAFEGen_lr5e-5_cfg_tani0.1_attnall_new_aug_drop0.5_scaffold_split_v5).

Usage — de novo, ligand-aware (paper's ligand-aware setting, s=1.5):
    python scripts/generate_from_encoded_v5.py \
        --model_path outputs/models/050826_SAFEGen_.../checkpoint-33110/full_model \
        --encoded_dir p53/encode_p53 \
        --num_samples 10000 \
        --cfg_scale 1.5 \
        --output_dir p53/generated

Usage — SAFE-prefix lead optimization:
    python scripts/generate_from_encoded_v5.py \
        --model_path .../full_model \
        --encoded_dir p53/encode_p53 \
        --safe_prefix 'c1ccc(-c2nc3ccccc3s2)cc1.' \
        --num_samples 500 \
        --output_dir p53/generated_scaffold_hop

Usage — pocket-only (paper's pocket-conditioned setting, s=1):
    python scripts/generate_from_encoded_v5.py \
        --model_path .../full_model \
        --encoded_dir p53/encode_p53 \
        --protein_only --cfg_scale 1.0 \
        --num_samples 10000 \
        --output_dir p53/generated_de_novo
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.modeling_novomolgen_infonce_120225_v5 import NovoMolGen, NovoMolGenConfig
from data_loader.molecule_tokenizer import MoleculeTokenizer
from data_loader.utils import safe_to_smiles
from transformers import AutoTokenizer


# ── helpers ─────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("generate_from_encoded_v4")
    logger.setLevel(getattr(logging, log_level.upper()))
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
    return logger


# ── model / condition loading ────────────────────────────────────────────────

def load_model_and_tokenizer(model_path: str, logger: logging.Logger):
    model_path = Path(model_path)
    logger.info(f"Loading model from: {model_path}")

    config = NovoMolGenConfig.from_pretrained(str(model_path))
    logger.info(
        f"  enable_cross_attn={config.enable_cross_attn}, "
        f"cross_layers={len(config.cross_layers)} layers, "
        f"tanimoto_weight={getattr(config, 'tanimoto_weight', 0.0)}"
    )
    if getattr(config, "train_new_modules_only", False):
        logger.info("  Overriding train_new_modules_only -> False for generation.")
        config.train_new_modules_only = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = NovoMolGen.from_pretrained(
        str(model_path),
        config=config,
        torch_dtype=model_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = model.to(device)
    if torch.cuda.is_available():
        model = model.to(torch.bfloat16)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    model.eval()

    tokenizer_path = model_path / "tokenizer.json"
    if tokenizer_path.exists():
        try:
            mol_tokenizer = MoleculeTokenizer.load(str(tokenizer_path))
            tokenizer = mol_tokenizer.get_pretrained()
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"  Model on {device} | dtype {next(model.parameters()).dtype}")
    return model, tokenizer, device


def load_conditions_from_dir(
    encoded_dir: Path, use_ligand: bool, logger: logging.Logger
) -> Dict[str, np.ndarray]:
    """Load pocket_vec, evo_vec, and (if use_ligand) ifp, ligand_vec .npy files."""
    required = ["pocket_vec.npy", "evo_vec.npy"]
    if use_ligand:
        required += ["ifp.npy", "ligand_vec.npy"]

    conds: Dict[str, np.ndarray] = {}
    for fname in required:
        p = encoded_dir / fname
        if not p.exists():
            raise FileNotFoundError(f"Missing condition vector: {p}")
        arr = np.load(str(p))
        conds[p.stem] = arr
        logger.info(f"  Loaded {p.name}: shape={arr.shape}, dtype={arr.dtype}")
    return conds


# ── generation ──────────────────────────────────────────────────────────────

def generate_from_prefix(
    model,
    tokenizer,
    conditions: Dict[str, np.ndarray],
    num_samples: int,
    device: torch.device,
    logger: logging.Logger,
    safe_prefix: Optional[str] = None,
    keep_trailing_dot: bool = True,
    batch_size: int = 200,
    max_length: int = 64,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    max_retries: int = 50,
    use_ligand: bool = True,
    cfg_scale: float = 1.5,
    dedupe: bool = True,
) -> Tuple[List[str], Dict[str, int]]:
    """Generate `num_samples` valid SMILES conditioned on encoded vectors and (optionally) a SAFE prefix."""
    model_dtype = next(model.parameters()).dtype

    # ── condition tensors (batch of 1 → repeated per sample step) ──
    def _to_dev(v):
        return torch.tensor([v], dtype=torch.float32).to(device).to(model_dtype)

    pocket_vec = _to_dev(conditions["pocket_vec"])
    evo_vec    = _to_dev(conditions["evo_vec"])
    if use_ligand:
        ifp        = _to_dev(conditions["ifp"])
        ligand_vec = _to_dev(conditions["ligand_vec"])
    else:
        ifp = ligand_vec = None

    # ── prefix ids (or BOS-only for de novo) ──
    if safe_prefix:
        prefix_input = safe_prefix if keep_trailing_dot else safe_prefix.rstrip(".")
        prefix_ids = tokenizer(
            prefix_input, return_tensors="pt", add_special_tokens=True
        )["input_ids"].to(device)
        # Strip trailing EOS if the tokenizer added one
        if prefix_ids[0, -1] == tokenizer.eos_token_id:
            prefix_ids = prefix_ids[:, :-1]
        logger.info(
            f"  SAFE prefix: {prefix_input!r} ({prefix_ids.shape[1]} tokens)"
        )
        if prefix_ids.shape[1] >= max_length:
            raise ValueError(
                f"Prefix ({prefix_ids.shape[1]} tokens) already fills max_length={max_length}; nothing to generate."
            )
    else:
        prefix_ids = torch.tensor(
            [[tokenizer.bos_token_id]], device=device
        )
        logger.info("  No SAFE prefix — de novo generation from BOS.")

    stats = {
        "total_safe_generated":  0,
        "conversion_failed":     0,
        "conversion_empty":      0,
        "conversion_fragments":  0,
        "conversion_success":    0,
        "duplicates_skipped":    0,
    }

    generated: List[str] = []
    seen: set = set()
    attempt = 0
    max_attempts = max(max_retries, num_samples * 5 // batch_size + 50)

    pbar = tqdm(total=num_samples, desc="Generating", unit="mol")
    while len(generated) < num_samples and attempt < max_attempts:
        needed = num_samples - len(generated)
        cur_bs = min(batch_size, max(needed, batch_size))

        ids_batch    = prefix_ids.repeat(cur_bs, 1)
        pocket_batch = pocket_vec.repeat(cur_bs, 1)
        evo_batch    = evo_vec.repeat(cur_bs, 1)
        ifp_batch    = ifp.repeat(cur_bs, 1) if use_ligand else None
        lig_batch    = ligand_vec.repeat(cur_bs, 1) if use_ligand else None

        gen_kwargs = dict(
            input_ids=ids_batch,
            pocket_vec=pocket_batch, evo_vec=evo_batch,
            ifp=ifp_batch, ligand_vec=lig_batch,
            append=True,
            max_length=max_length,
            temperature=temperature, top_k=top_k, top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            cfg_scale=cfg_scale,
        )

        with torch.no_grad():
            if device.type == "cuda" and model_dtype == torch.bfloat16:
                with torch.autocast(device_type="cuda", dtype=model_dtype):
                    seqs = model.generate_with_condition(**gen_kwargs)
            else:
                seqs = model.generate_with_condition(**gen_kwargs)

        if isinstance(seqs, dict):
            seqs = seqs.get("sequences", seqs.get("full_safe"))
        seqs = model._filter_tokens_after_eos(seqs, eos_id=tokenizer.eos_token_id)
        decoded = [s.replace(" ", "") for s in tokenizer.batch_decode(seqs, skip_special_tokens=True)]
        stats["total_safe_generated"] += len(decoded)

        added = 0
        for safe_str in decoded:
            if len(generated) >= num_samples:
                break
            try:
                smi = safe_to_smiles(safe_str)
            except Exception:
                stats["conversion_failed"] += 1
                continue
            if not smi:
                stats["conversion_empty"] += 1
                continue
            if "." in smi:
                stats["conversion_fragments"] += 1
                continue
            if dedupe and smi in seen:
                stats["duplicates_skipped"] += 1
                continue
            seen.add(smi)
            generated.append(smi)
            stats["conversion_success"] += 1
            added += 1

        pbar.update(added)
        attempt += 1

    pbar.close()
    if len(generated) < num_samples:
        logger.warning(
            f"Only produced {len(generated)}/{num_samples} valid SMILES after {attempt} attempts. Stats: {stats}"
        )
    return generated, stats


# ── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate molecules from encoded vectors (+ optional SAFE prefix) using the v4 CFG model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to a v4 CFG checkpoint (e.g. .../checkpoint-33110/full_model)")
    parser.add_argument("--encoded_dir", type=str, required=True,
                        help="Directory containing pocket_vec.npy, evo_vec.npy, ifp.npy, ligand_vec.npy")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_name", type=str, default="generated_molecules.smi",
                        help="Filename for the .smi output (written under --output_dir)")

    # SAFE prefix (optional; empty ⇒ de novo)
    parser.add_argument("--safe_prefix", type=str, default=None,
                        help="Optional SAFE string prefix. Trailing '.' controls attachment behavior — see --strip_trailing_dot.")
    parser.add_argument("--strip_trailing_dot", action="store_true",
                        help="Strip the trailing '.' from the SAFE prefix before tokenizing (default: keep).")

    # Generation params
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--batch_size",  type=int, default=200)
    parser.add_argument("--max_length",  type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k",       type=int, default=50)
    parser.add_argument("--top_p",       type=float, default=0.95)
    parser.add_argument("--max_retries", type=int, default=200)
    parser.add_argument("--no_dedupe",   action="store_true",
                        help="Keep duplicate SMILES in the output (default: dedupe).")

    # Conditioning mode
    parser.add_argument("--protein_only", action="store_true",
                        help="Drop ifp + ligand_vec and generate with null ligand embedding "
                             "(paper's pocket-conditioned setting). Forces effective cfg_scale=1.0.")
    parser.add_argument("--cfg_scale", type=float, default=1.5,
                        help="Classifier-free guidance scale. Paper: s=1 for pocket-only, s=1.5 for ligand-aware. "
                             ">1.0 doubles per-step compute (halve --batch_size if you OOM).")

    parser.add_argument("--seed",       type=int, default=2025)
    parser.add_argument("--log_level",  type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    set_seed(args.seed)
    logger = setup_logging(args.log_level)

    # Resolve mode: protein_only forces cfg_scale=1.0 (see reference script rationale —
    # under protein_only the ligand token is already null_ligand_emb, so CFG's uncond
    # branch would be identical to the cond branch → guidance amplifies nothing).
    use_ligand = not args.protein_only
    effective_cfg = 1.0 if args.protein_only else float(args.cfg_scale)
    if args.protein_only and args.cfg_scale != 1.0:
        logger.warning(
            f"--protein_only forces cfg_scale=1.0 (requested {args.cfg_scale}). "
            "Under protein-only, cond ≡ uncond, so CFG would waste 2× compute for no effect."
        )

    encoded_dir = Path(args.encoded_dir)
    if not encoded_dir.is_dir():
        raise FileNotFoundError(f"--encoded_dir not found: {encoded_dir}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("StructureSAFE generation (v4 / CFG)")
    logger.info("=" * 70)
    logger.info(f"Model         : {args.model_path}")
    logger.info(f"Encoded dir   : {encoded_dir}")
    logger.info(f"Output        : {output_dir / args.output_name}")
    logger.info(f"Num samples   : {args.num_samples}")
    logger.info(f"Mode          : use_ligand={use_ligand}, cfg_scale={effective_cfg}")
    logger.info(f"Sampling      : T={args.temperature}, top_k={args.top_k}, top_p={args.top_p}, max_length={args.max_length}")
    if args.safe_prefix:
        logger.info(f"SAFE prefix   : {args.safe_prefix!r}")
    else:
        logger.info("SAFE prefix   : (none, de novo)")

    # Load conditions
    logger.info("\nLoading condition vectors...")
    conditions = load_conditions_from_dir(encoded_dir, use_ligand, logger)

    # Load model
    logger.info("\nLoading model...")
    model, tokenizer, device = load_model_and_tokenizer(args.model_path, logger)

    # Generate
    logger.info("\nGenerating...")
    generated, stats = generate_from_prefix(
        model=model,
        tokenizer=tokenizer,
        conditions=conditions,
        num_samples=args.num_samples,
        device=device,
        logger=logger,
        safe_prefix=args.safe_prefix,
        keep_trailing_dot=not args.strip_trailing_dot,
        batch_size=args.batch_size,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_retries=args.max_retries,
        use_ligand=use_ligand,
        cfg_scale=effective_cfg,
        dedupe=not args.no_dedupe,
    )

    # Write .smi
    out_smi = output_dir / args.output_name
    with open(out_smi, "w") as f:
        for smi in generated:
            f.write(f"{smi}\n")
    logger.info(f"\n✓ Wrote {len(generated)} SMILES to {out_smi}")
    logger.info(f"  Stats: {stats}")


if __name__ == "__main__":
    main()
