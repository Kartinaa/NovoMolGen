#!/usr/bin/env python3
"""
Generate molecules from a v4 model checkpoint for each MolGenBench validation sample.

v4 differences vs dropout_v2:
  - Deterministic ligand encoder (mu only, no VAE reparameterization)
  - CFG support: --cfg_scale > 1.0 enables classifier-free guidance
  - Protein-only mode: --protein_only uses null_ligand_emb instead of ligand features

Usage:
    python scripts/data_processing/molgenbench/generate_from_full_model_v4.py \
        --model_path outputs/models/042725_.../checkpoint-30500/full_model \
        --validation_sets finetune_data/processed_data/hf_molgenbench_dataset/validation \
        --num_samples_per_val 50 \
        --output_dir outputs/generated_molecules_v4

    # Protein-only generation (no ligand condition):
    python scripts/data_processing/molgenbench/generate_from_full_model_v4.py \
        --model_path outputs/models/042725_.../checkpoint-30500/full_model \
        --validation_sets finetune_data/processed_data/hf_molgenbench_dataset/validation \
        --num_samples_per_val 50 \
        --output_dir outputs/generated_molecules_v4_protein_only \
        --protein_only

    # CFG generation (cfg_scale > 1.0):
    python scripts/data_processing/molgenbench/generate_from_full_model_v4.py \
        --model_path outputs/models/042725_.../checkpoint-30500/full_model \
        --validation_sets finetune_data/processed_data/hf_molgenbench_dataset/validation \
        --num_samples_per_val 50 \
        --output_dir outputs/generated_molecules_v4_cfg2 \
        --cfg_scale 2.0
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import random
import torch
import yaml
from datasets import load_from_disk
from tqdm import tqdm
from rdkit import Chem

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT / "src"))

from models.modeling_novomolgen_infonce_120225_v4 import NovoMolGen, NovoMolGenConfig  # type: ignore
from data_loader.molecule_tokenizer import MoleculeTokenizer
from data_loader.utils import safe_to_smiles
from transformers import AutoTokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("generate_v4")
    logger.setLevel(getattr(logging, log_level.upper()))
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
    return logger


def load_model_and_tokenizer(model_path: str, logger: logging.Logger):
    model_path = Path(model_path)
    logger.info(f"Loading model from: {model_path}")

    config = NovoMolGenConfig.from_pretrained(str(model_path))
    logger.info(f"Model config: enable_cross_attn={config.enable_cross_attn}, cross_layers={config.cross_layers}")
    if getattr(config, "train_new_modules_only", False):
        logger.info("Overriding config.train_new_modules_only=True -> False for generation.")
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

    if torch.cuda.is_available() and model_dtype == torch.bfloat16:
        model = model.to(torch.bfloat16)
        param_dtypes = set(p.dtype for p in model.parameters())
        if torch.float32 in param_dtypes:
            logger.warning(f"Some parameters still float32: {param_dtypes}. Converting...")
            model = model.to(torch.bfloat16)
        logger.info(f"Model converted to bfloat16. Parameter dtypes: {param_dtypes}")

    model.eval()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    logger.info(f"Model loaded on {device} with dtype {next(model.parameters()).dtype}")

    tokenizer_path = model_path / "tokenizer.json"
    if tokenizer_path.exists():
        try:
            mol_tokenizer = MoleculeTokenizer.load(str(tokenizer_path))
            tokenizer = mol_tokenizer.get_pretrained()
            logger.info(f"Loaded tokenizer as MoleculeTokenizer from {tokenizer_path}")
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            logger.info(f"Loaded tokenizer as AutoTokenizer from {model_path}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        logger.info(f"Loaded tokenizer from {model_path}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, device


def load_validation_sets(validation_set_paths: Union[str, List[str]], logger: logging.Logger):
    if isinstance(validation_set_paths, str):
        validation_set_paths = [validation_set_paths]

    validation_sets = {}
    for val_path in validation_set_paths:
        val_path = Path(val_path)
        if val_path.exists():
            dataset = load_from_disk(str(val_path))
            name = val_path.name
            validation_sets[name] = dataset
            logger.info(f"Loaded validation set '{name}': {len(dataset)} samples")
            if len(dataset) > 0:
                sample = dataset[0]
                has_conditions = all(k in sample for k in ["pocket_vec", "evo_vec", "ifp", "ligand_vec"])
                logger.info(f"  Condition features available: {has_conditions}")
        else:
            logger.warning(f"Validation set path does not exist: {val_path}")

    return validation_sets


def generate_molecules_for_validation_set(
    model: NovoMolGen,
    tokenizer,
    validation_set,
    num_samples: int,
    device: torch.device,
    logger: logging.Logger,
    batch_size: int = 50,
    max_length: int = 64,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    max_retries: int = 10,
    ifp_ablation: str = "none",
    cond_ablation: str = "none",
    keep_invalid: bool = False,
    cfg_scale: float = 1.0,
    protein_only: bool = False,
):
    all_generated = []
    all_condition_info = []

    conversion_stats = {
        "total_safe_generated": 0,
        "conversion_failed": 0,
        "conversion_empty": 0,
        "conversion_fragments": 0,
        "conversion_success": 0,
    }

    if len(validation_set) == 0:
        logger.warning("Validation set is empty")
        return all_generated, all_condition_info, conversion_stats

    sample = validation_set[0]
    has_conditions = all(k in sample for k in ["pocket_vec", "evo_vec", "ifp", "ligand_vec"])

    if not has_conditions:
        logger.warning("Condition features not found; generating unconditionally")
        attempt = 0
        max_total_attempts = max_retries * (num_samples // batch_size + 1)
        while len(all_generated) < num_samples and attempt < max_total_attempts:
            needed = num_samples - len(all_generated)
            current_batch_size = min(batch_size, needed)
            with torch.no_grad():
                outputs = model.sample(
                    tokenizer=tokenizer,
                    batch_size=current_batch_size,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    device=device,
                )
                safe_strings = outputs[model.mol_type]
                conversion_stats["total_safe_generated"] += len(safe_strings)
                for safe_str in safe_strings:
                    if len(all_generated) >= num_samples:
                        break
                    try:
                        smiles = safe_to_smiles(safe_str)
                        if keep_invalid:
                            if smiles is None or smiles == "":
                                conversion_stats["conversion_empty"] += 1
                            else:
                                conversion_stats["conversion_success"] += 1
                            all_generated.append(smiles)
                        else:
                            if smiles and '.' not in smiles:
                                all_generated.append(smiles)
                                conversion_stats["conversion_success"] += 1
                            elif not smiles:
                                conversion_stats["conversion_empty"] += 1
                            else:
                                conversion_stats["conversion_fragments"] += 1
                    except Exception:
                        conversion_stats["conversion_failed"] += 1
            attempt += 1
        if len(all_generated) < num_samples:
            logger.warning(f"Only generated {len(all_generated)}/{num_samples} molecules after {attempt} attempts")
    else:
        logger.info(f"Generating {num_samples} molecules for each of {len(validation_set)} samples")
        logger.info(f"  Total molecules to generate: {len(validation_set) * num_samples}")
        if protein_only:
            logger.info("  protein_only=True: ligand features replaced with null_ligand_emb")
        if cfg_scale > 1.0:
            logger.info(f"  CFG enabled: cfg_scale={cfg_scale}")
        if cond_ablation != "none":
            logger.info(f"  Global condition ablation: {cond_ablation}")

        model_dtype = next(model.parameters()).dtype

        for val_idx in tqdm(range(len(validation_set)), desc="Processing validation samples"):
            val_sample = validation_set[val_idx]
            ligand_path = val_sample.get("ligand_path", None)

            pocket_vec = torch.tensor([val_sample["pocket_vec"]], dtype=torch.float32).to(device).to(model_dtype)
            evo_vec    = torch.tensor([val_sample["evo_vec"]],    dtype=torch.float32).to(device).to(model_dtype)
            ifp        = torch.tensor([val_sample["ifp"]],        dtype=torch.float32).to(device).to(model_dtype)
            ligand_vec = torch.tensor([val_sample["ligand_vec"]], dtype=torch.float32).to(device).to(model_dtype)

            samples_generated_for_this_val = []
            attempt = 0
            max_total_attempts = max_retries * (num_samples // batch_size + 1)

            while len(samples_generated_for_this_val) < num_samples and attempt < max_total_attempts:
                needed = num_samples - len(samples_generated_for_this_val)
                current_batch_size = min(batch_size, needed)

                pocket_vec_batch = pocket_vec.repeat(current_batch_size, 1)
                evo_vec_batch    = evo_vec.repeat(current_batch_size, 1)

                # protein_only: pass None for ifp/ligand_vec → model uses null_ligand_emb
                if protein_only:
                    ifp_batch        = None
                    ligand_vec_batch = None
                else:
                    ifp_batch        = ifp.repeat(current_batch_size, 1)
                    ligand_vec_batch = ligand_vec.repeat(current_batch_size, 1)

                    # Global condition ablation
                    if cond_ablation == "zero":
                        pocket_vec_batch = torch.zeros_like(pocket_vec_batch)
                        evo_vec_batch    = torch.zeros_like(evo_vec_batch)
                        ifp_batch        = torch.zeros_like(ifp_batch)
                        ligand_vec_batch = torch.zeros_like(ligand_vec_batch)
                    elif cond_ablation == "shuffle":
                        perm = torch.randperm(current_batch_size, device=device)
                        pocket_vec_batch = pocket_vec_batch[perm]
                        evo_vec_batch    = evo_vec_batch[perm]
                        ifp_batch        = ifp_batch[perm]
                        ligand_vec_batch = ligand_vec_batch[perm]

                    # IFP-only ablation (when no global ablation)
                    if cond_ablation == "none":
                        if ifp_ablation == "zero":
                            ifp_batch = torch.zeros_like(ifp_batch)
                        elif ifp_ablation == "shuffle":
                            perm_ifp = torch.randperm(current_batch_size, device=device)
                            ifp_batch = ifp_batch[perm_ifp]

                with torch.no_grad():
                    input_ids = torch.tensor(
                        [[tokenizer.bos_token_id]], device=device
                    ).repeat(current_batch_size, 1)

                    autocast_enabled = device.type == "cuda" and model_dtype == torch.bfloat16
                    if autocast_enabled:
                        with torch.autocast(device_type="cuda", dtype=model_dtype, enabled=True):
                            generated_sequences = model.generate_with_condition(
                                input_ids=input_ids,
                                pocket_vec=pocket_vec_batch,
                                evo_vec=evo_vec_batch,
                                ifp=ifp_batch,
                                ligand_vec=ligand_vec_batch,
                                max_length=max_length,
                                temperature=temperature,
                                top_k=top_k,
                                top_p=top_p,
                                eos_token_id=tokenizer.eos_token_id,
                                cfg_scale=cfg_scale,
                            )
                    else:
                        generated_sequences = model.generate_with_condition(
                            input_ids=input_ids,
                            pocket_vec=pocket_vec_batch,
                            evo_vec=evo_vec_batch,
                            ifp=ifp_batch,
                            ligand_vec=ligand_vec_batch,
                            max_length=max_length,
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                            eos_token_id=tokenizer.eos_token_id,
                            cfg_scale=cfg_scale,
                        )

                if isinstance(generated_sequences, dict):
                    sequences = generated_sequences.get('sequences', generated_sequences.get('full_safe'))
                else:
                    sequences = generated_sequences

                sequences = model._filter_tokens_after_eos(sequences, eos_id=tokenizer.eos_token_id)
                decoded_strings = tokenizer.batch_decode(sequences, skip_special_tokens=True)
                decoded_strings = [s.replace(" ", "") for s in decoded_strings]
                conversion_stats["total_safe_generated"] += len(decoded_strings)

                for safe_str in decoded_strings:
                    if len(samples_generated_for_this_val) >= num_samples:
                        break
                    try:
                        smiles = safe_to_smiles(safe_str)
                        if keep_invalid:
                            if smiles is None or smiles == "":
                                conversion_stats["conversion_empty"] += 1
                            else:
                                conversion_stats["conversion_success"] += 1
                            samples_generated_for_this_val.append(smiles)
                        else:
                            if smiles and '.' not in smiles:
                                samples_generated_for_this_val.append(smiles)
                                conversion_stats["conversion_success"] += 1
                            elif not smiles:
                                conversion_stats["conversion_empty"] += 1
                            else:
                                conversion_stats["conversion_fragments"] += 1
                    except Exception:
                        conversion_stats["conversion_failed"] += 1

                attempt += 1

            if len(samples_generated_for_this_val) < num_samples:
                logger.warning(
                    f"  Sample {val_idx}: Only {len(samples_generated_for_this_val)}/{num_samples} "
                    f"molecules after {attempt} attempts"
                )

            all_generated.extend(samples_generated_for_this_val)
            all_condition_info.append({
                "validation_index": int(val_idx),
                "ligand_path": ligand_path,
                "generated_molecules": samples_generated_for_this_val,
            })

        logger.info(f"\n  Overall conversion statistics:")
        logger.info(f"    Total SAFE generated: {conversion_stats['total_safe_generated']}")
        logger.info(f"    Successful conversions: {conversion_stats['conversion_success']}")
        logger.info(f"    Failed (exception): {conversion_stats['conversion_failed']}")
        logger.info(f"    Empty: {conversion_stats['conversion_empty']}")
        logger.info(f"    Fragments: {conversion_stats['conversion_fragments']}")
        total_failed = sum([
            conversion_stats['conversion_failed'],
            conversion_stats['conversion_empty'],
            conversion_stats['conversion_fragments'],
        ])
        if conversion_stats['total_safe_generated'] > 0:
            success_rate = conversion_stats['conversion_success'] / conversion_stats['total_safe_generated'] * 100
            logger.info(f"    Success rate: {success_rate:.2f}%")

    return all_generated, all_condition_info, conversion_stats


def main():
    parser = argparse.ArgumentParser(description="Generate molecules from v4 model for MolGenBench")
    parser.add_argument("--model_path",          type=str, required=True)
    parser.add_argument("--validation_sets",     type=str, nargs="+", required=True)
    parser.add_argument("--num_samples_per_val", type=int, default=50)
    parser.add_argument("--output_dir",          type=str, required=True)
    parser.add_argument("--batch_size",          type=int, default=10)
    parser.add_argument("--max_length",          type=int, default=64)
    parser.add_argument("--temperature",         type=float, default=1.0)
    parser.add_argument("--top_k",               type=int, default=50)
    parser.add_argument("--top_p",               type=float, default=0.95)
    parser.add_argument("--log_level",           type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--max_retries",         type=int, default=10)
    parser.add_argument("--seed",                type=int, default=2025)
    parser.add_argument("--keep_invalid",        action="store_true")
    parser.add_argument(
        "--ifp_ablation", type=str, default="none", choices=["none", "zero", "shuffle"],
    )
    parser.add_argument(
        "--cond_ablation", type=str, default="none", choices=["none", "zero", "shuffle"],
        help="Global ablation for all condition features (pocket_vec, evo_vec, ifp, ligand_vec)",
    )
    # v4-specific
    parser.add_argument(
        "--cfg_scale", type=float, default=1.0,
        help="Classifier-free guidance scale (default 1.0 = no CFG; >1.0 enables CFG)",
    )
    parser.add_argument(
        "--protein_only", action="store_true",
        help="Generate with protein condition only (ligand replaced with null_ligand_emb)",
    )

    args = parser.parse_args()
    set_seed(args.seed)

    logger = setup_logging(args.log_level)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Molecule Generation (v4 model) — MolGenBench")
    logger.info("=" * 60)
    logger.info(f"Model path:        {args.model_path}")
    logger.info(f"Validation sets:   {args.validation_sets}")
    logger.info(f"Samples per val:   {args.num_samples_per_val}")
    logger.info(f"Output dir:        {output_dir}")
    logger.info(f"cfg_scale:         {args.cfg_scale}")
    logger.info(f"protein_only:      {args.protein_only}")
    logger.info(f"cond_ablation:     {args.cond_ablation}")
    logger.info(f"ifp_ablation:      {args.ifp_ablation}")
    logger.info(f"keep_invalid:      {args.keep_invalid}")

    model, tokenizer, device = load_model_and_tokenizer(args.model_path, logger)
    validation_sets = load_validation_sets(args.validation_sets, logger)

    if not validation_sets:
        logger.error("No validation sets loaded. Exiting.")
        return

    all_results = {}

    for val_name, val_dataset in validation_sets.items():
        logger.info(f"\nProcessing validation set: {val_name} ({len(val_dataset)} samples)")

        generated_molecules, condition_info, conversion_stats = generate_molecules_for_validation_set(
            model=model,
            tokenizer=tokenizer,
            validation_set=val_dataset,
            num_samples=args.num_samples_per_val,
            device=device,
            logger=logger,
            batch_size=args.batch_size,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_retries=args.max_retries,
            ifp_ablation=args.ifp_ablation,
            cond_ablation=args.cond_ablation,
            keep_invalid=args.keep_invalid,
            cfg_scale=args.cfg_scale,
            protein_only=args.protein_only,
        )

        logger.info(f"  Generated {len(generated_molecules)} valid SMILES")

        results = {
            "validation_set": val_name,
            "num_samples": len(generated_molecules),
            "generated_molecules": generated_molecules,
            "condition_info": condition_info,
            "conversion_stats": conversion_stats,
            "generation_params": {
                "batch_size":    args.batch_size,
                "max_length":    args.max_length,
                "temperature":   args.temperature,
                "top_k":         args.top_k,
                "top_p":         args.top_p,
                "cfg_scale":     args.cfg_scale,
                "protein_only":  args.protein_only,
                "ifp_ablation":  args.ifp_ablation,
                "cond_ablation": args.cond_ablation,
            },
        }

        output_file = output_dir / f"generated_{val_name}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"  Saved JSON to: {output_file}")

        txt_file = output_dir / f"generated_{val_name}.txt"
        with open(txt_file, "w", encoding="utf-8") as f:
            for mol in generated_molecules:
                f.write(f"{mol}\n")
        logger.info(f"  Saved TXT to: {txt_file}")

        all_results[val_name] = results

    summary = {
        "model_path": str(args.model_path),
        "validation_sets": args.validation_sets,
        "num_samples_per_val": args.num_samples_per_val,
        "cfg_scale": args.cfg_scale,
        "protein_only": args.protein_only,
        "total_generated": sum(len(r["generated_molecules"]) for r in all_results.values()),
        "results": {
            name: {
                "num_samples": len(r["generated_molecules"]),
                "conversion_stats": r.get("conversion_stats", {}),
            }
            for name, r in all_results.items()
        },
    }

    summary_file = output_dir / "generation_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"\nSaved summary to: {summary_file}")

    logger.info("\n" + "=" * 60)
    logger.info("Generation completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
