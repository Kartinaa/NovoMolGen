#!/usr/bin/env python3
"""
Lead optimization: Generate molecules from a trained model by continuing from given SAFE strings.
Unlike generate_from_full_model.py which generates from scratch, this script continues generation
from provided SAFE prefixes, enabling lead optimization scenarios.

Usage:
    python scripts/lead_optimization_from_full_model_xdock.py \
        --model_path outputs/model/checkpoint/full_model \
        --validation_sets finetune_data/validation \
        --prefix_safe "C1CCCCC1" \
        --num_samples_per_val 50 \
        --output_dir outputs/lead_optimization
    
    # Or use different prefix for each validation sample from a file:
    python scripts/lead_optimization_from_full_model_xdock.py \
        --model_path outputs/model/checkpoint/full_model \
        --validation_sets finetune_data/validation \
        --prefix_file prefixes.txt \
        --num_samples_per_val 50 \
        --output_dir outputs/lead_optimization
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
from datasets import load_from_disk, load_dataset, Features, Value, Sequence
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.modeling_novomolgen_infonce_120225 import NovoMolGen, NovoMolGenConfig
from data_loader.molecule_tokenizer import MoleculeTokenizer
from data_loader.utils import safe_to_smiles
from transformers import AutoTokenizer


def set_seed(seed: int) -> None:
    """Set random seed for reproducible generation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging."""
    logger = logging.getLogger("lead_optimization")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def load_model_and_tokenizer(model_path: str, logger: logging.Logger):
    """Load model and tokenizer from full_model directory."""
    model_path = Path(model_path)
    
    logger.info(f"Loading model from: {model_path}")
    
    # Load config
    config = NovoMolGenConfig.from_pretrained(str(model_path))
    logger.info(f"Model config: enable_cross_attn={config.enable_cross_attn}, cross_layers={config.cross_layers}")
    
    if getattr(config, "train_new_modules_only", False):
        logger.info("Overriding config.train_new_modules_only=True -> False for generation (no freezing logic needed).")
        config.train_new_modules_only = False
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = NovoMolGen.from_pretrained(
        str(model_path),
        config=config,
        torch_dtype=model_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = model.to(device)
    
    # Convert to bfloat16 if on CUDA
    if torch.cuda.is_available() and model_dtype == torch.bfloat16:
        model = model.to(torch.bfloat16)
        param_dtypes = set(p.dtype for p in model.parameters())
        if torch.float32 in param_dtypes:
            logger.warning(f"Some parameters are still float32: {param_dtypes}. Converting all to bfloat16...")
            model = model.to(torch.bfloat16)
        logger.info(f"Model converted to bfloat16. Parameter dtypes: {param_dtypes}")
    
    model.eval()
    
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    final_model_dtype = next(model.parameters()).dtype
    logger.info(f"Model loaded on {device} with dtype {final_model_dtype}")
    
    # Load tokenizer
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
    """Load validation sets from disk."""
    if isinstance(validation_set_paths, str):
        validation_set_paths = [validation_set_paths]
    
    validation_sets = {}
    for val_path in validation_set_paths:
        val_path = Path(val_path)
        if val_path.exists():
            try:
                dataset = load_from_disk(str(val_path))
            except (TypeError, AttributeError, ValueError) as e:
                logger.warning(
                    f"Failed to load validation set from {val_path} using load_from_disk: {e}\n"
                    f"Attempting fallback: loading from arrow files with explicit features..."
                )
                try:
                    arrow_files = sorted(val_path.glob("*.arrow"))
                    if arrow_files:
                        features = Features({
                            'SMILES': Value('string'),
                            'pocket_vec': Sequence(Value('float64')),
                            'evo_vec': Sequence(Value('float32')),
                            'ifp': Sequence(Value('float32')),
                            'ligand_vec': Sequence(Value('float32')),
                        })
                        dataset = load_dataset(
                            "arrow",
                            data_files=[str(f) for f in arrow_files],
                            features=features,
                            split="train"
                        )
                        logger.info(f"Successfully loaded validation set using arrow fallback")
                    else:
                        raise ValueError(f"No arrow files found in {val_path}")
                except Exception as e2:
                    logger.error(
                        f"Fallback also failed: {e2}\n"
                        f"Skipping validation set: {val_path}"
                    )
                    continue
            
            name = val_path.name
            validation_sets[name] = dataset
            logger.info(f"Loaded validation set '{name}': {len(dataset)} samples")
            
            if len(dataset) > 0:
                sample = dataset[0]
                has_conditions = all(key in sample for key in ["pocket_vec", "evo_vec", "ifp", "ligand_vec"])
                logger.info(f"  Condition features available: {has_conditions}")
        else:
            logger.warning(f"Validation set path does not exist: {val_path}")
    
    return validation_sets


def load_prefix_safe_strings(
    prefix_safe: Optional[str] = None,
    prefix_file: Optional[str] = None,
    num_samples: int = 1,
    logger: logging.Logger = None,
) -> List[str]:
    """Load SAFE prefix strings from command line or file.
    
    Args:
        prefix_safe: Single SAFE string to use for all samples
        prefix_file: Path to file containing SAFE strings (one per line)
        num_samples: Expected number of samples (for validation)
        logger: Logger instance
    
    Returns:
        List of SAFE prefix strings
    """
    if prefix_safe is not None:
        # Use the same prefix for all samples
        logger.info(f"Using single SAFE prefix for all samples: {prefix_safe}")
        return [prefix_safe] * num_samples
    elif prefix_file is not None:
        # Load prefixes from file
        prefix_file = Path(prefix_file)
        if not prefix_file.exists():
            raise FileNotFoundError(f"Prefix file not found: {prefix_file}")
        
        with open(prefix_file, "r", encoding="utf-8") as f:
            prefixes = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Loaded {len(prefixes)} SAFE prefixes from {prefix_file}")
        
        if len(prefixes) != num_samples:
            logger.warning(
                f"Number of prefixes ({len(prefixes)}) does not match "
                f"number of validation samples ({num_samples}). "
                f"Will cycle/truncate as needed."
            )
        
        return prefixes
    else:
        raise ValueError("Either --prefix_safe or --prefix_file must be provided")


def generate_molecules_with_prefix(
    model: NovoMolGen,
    tokenizer,
    validation_set,
    prefix_safe_strings: List[str],
    num_samples: int,
    device: torch.device,
    logger: logging.Logger,
    batch_size: int = 50,
    sample_posterior: bool = False,
    max_length: int = 64,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    max_retries: int = 10,
    ifp_ablation: str = "none",
    cond_ablation: str = "none",
    append_mode: bool = True,
):
    """Generate molecules by continuing from SAFE prefix strings.
    
    Args:
        prefix_safe_strings: List of SAFE strings to use as prefixes for each validation sample
        append_mode: If True, append to prefix; if False, replace after prefix
        Other args same as generate_molecules_for_validation_set
    
    Returns:
        tuple: (all_generated, all_condition_info, conversion_stats)
    """
    all_generated = []
    all_condition_info = []
    
    conversion_stats = {
        "total_safe_generated": 0,
        "conversion_failed": 0,
        "conversion_empty": 0,
        "conversion_fragments": 0,
        "conversion_success": 0,
        "duplicates_skipped": 0,
    }
    
    if len(validation_set) == 0:
        logger.warning("Validation set is empty")
        return all_generated, all_condition_info, conversion_stats
    
    sample = validation_set[0]
    has_conditions = all(key in sample for key in ["pocket_vec", "evo_vec", "ifp", "ligand_vec"])
    
    if not has_conditions:
        logger.error("This script requires condition features in validation set")
        return all_generated, all_condition_info, conversion_stats
    
    # Generate with condition features
    logger.info(f"Generating {num_samples} molecules for each of {len(validation_set)} samples in validation set")
    logger.info(f"  Total molecules to generate: {len(validation_set) * num_samples}")
    logger.info(f"  Using SAFE prefix strings for lead optimization")
    logger.info(f"  Append mode: {append_mode}")
    
    if ifp_ablation != "none":
        logger.info(f"  IFP ablation mode: '{ifp_ablation}'")
    if cond_ablation != "none":
        logger.info(f"  Global condition ablation mode: '{cond_ablation}'")
    
    model_dtype = next(model.parameters()).dtype
    
    # Iterate over each sample in validation set
    for val_idx in tqdm(range(len(validation_set)), desc="Processing validation samples"):
        val_sample = validation_set[val_idx]
        
        # Get prefix SAFE string for this sample (cycle if needed)
        prefix_safe = prefix_safe_strings[val_idx % len(prefix_safe_strings)]
        
        # Extract condition features
        pocket_vec = torch.tensor([val_sample["pocket_vec"]], dtype=torch.float32).to(device).to(model_dtype)
        evo_vec = torch.tensor([val_sample["evo_vec"]], dtype=torch.float32).to(device).to(model_dtype)
        ifp = torch.tensor([val_sample["ifp"]], dtype=torch.float32).to(device).to(model_dtype)
        ligand_vec = torch.tensor([val_sample["ligand_vec"]], dtype=torch.float32).to(device).to(model_dtype)
        
        # Tokenize prefix SAFE string
        # The tokenizer's pre_tokenizer (regex-based) will automatically split the string,
        # so we don't need to manually add spaces
        prefix_encoding = tokenizer(
            prefix_safe,
            return_tensors="pt",
            add_special_tokens=True,  # Include BOS
        )
        prefix_ids = prefix_encoding["input_ids"].to(device)
        
        logger.debug(f"Sample {val_idx}: Prefix SAFE = '{prefix_safe}', tokenized length = {prefix_ids.shape[1]}")
        
        # Generate num_samples molecules for this validation sample
        samples_generated_for_this_val = []
        unique_smiles_set = set()  # Track unique SMILES for deduplication per sample
        attempt = 0
        # Increase max attempts to ensure we get enough unique molecules
        max_total_attempts = max_retries * (num_samples * 3 // batch_size + 1)
        
        while len(samples_generated_for_this_val) < num_samples and attempt < max_total_attempts:
            needed = num_samples - len(samples_generated_for_this_val)
            current_batch_size = min(batch_size, needed)
            
            # Repeat prefix_ids and condition features for batch
            input_ids = prefix_ids.repeat(current_batch_size, 1)
            pocket_vec_batch = pocket_vec.repeat(current_batch_size, 1)
            evo_vec_batch = evo_vec.repeat(current_batch_size, 1)
            ifp_batch = ifp.repeat(current_batch_size, 1)
            ligand_vec_batch = ligand_vec.repeat(current_batch_size, 1)
            
            # Global condition ablation
            if cond_ablation == "zero":
                pocket_vec_batch = torch.zeros_like(pocket_vec_batch)
                evo_vec_batch = torch.zeros_like(evo_vec_batch)
                ifp_batch = torch.zeros_like(ifp_batch)
                ligand_vec_batch = torch.zeros_like(ligand_vec_batch)
            elif cond_ablation == "shuffle":
                perm = torch.randperm(current_batch_size, device=device)
                pocket_vec_batch = pocket_vec_batch[perm]
                evo_vec_batch = evo_vec_batch[perm]
                ifp_batch = ifp_batch[perm]
                ligand_vec_batch = ligand_vec_batch[perm]
            
            # IFP-only ablation
            if cond_ablation == "none":
                if ifp_ablation == "zero":
                    ifp_batch = torch.zeros_like(ifp_batch)
                elif ifp_ablation == "shuffle":
                    perm_ifp = torch.randperm(current_batch_size, device=device)
                    ifp_batch = ifp_batch[perm_ifp]
            
            # Generate using condition features, continuing from prefix
            with torch.no_grad():
                autocast_enabled = device.type == "cuda" and model_dtype == torch.bfloat16
                if autocast_enabled:
                    with torch.autocast(device_type="cuda", dtype=model_dtype, enabled=True):
                        generated_sequences = model.generate_with_condition(
                            input_ids=input_ids,
                            pocket_vec=pocket_vec_batch,
                            evo_vec=evo_vec_batch,
                            ifp=ifp_batch,
                            ligand_vec=ligand_vec_batch,
                            sample_posterior=sample_posterior,
                            append=append_mode,
                            max_length=max_length,
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                            eos_token_id=tokenizer.eos_token_id,
                        )
                else:
                    generated_sequences = model.generate_with_condition(
                        input_ids=input_ids,
                        pocket_vec=pocket_vec_batch,
                        evo_vec=evo_vec_batch,
                        ifp=ifp_batch,
                        ligand_vec=ligand_vec_batch,
                        sample_posterior=sample_posterior,
                        append=append_mode,
                        max_length=max_length,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        eos_token_id=tokenizer.eos_token_id,
                    )
            
            # Decode sequences
            if isinstance(generated_sequences, dict):
                sequences = generated_sequences.get('sequences', generated_sequences.get('full_safe'))
            else:
                sequences = generated_sequences
            
            # Filter EOS tokens
            sequences = model._filter_tokens_after_eos(sequences, eos_id=tokenizer.eos_token_id)
            
            # Decode to strings (SAFE format)
            decoded_strings = tokenizer.batch_decode(sequences, skip_special_tokens=True)
            decoded_strings = [s.replace(" ", "") for s in decoded_strings]
            conversion_stats["total_safe_generated"] += len(decoded_strings)
            
            # Convert SAFE to SMILES
            for safe_str in decoded_strings:
                if len(samples_generated_for_this_val) >= num_samples:
                    break
                try:
                    smiles = safe_to_smiles(safe_str)
                    if smiles and '.' not in smiles:
                        # Deduplication: only add if not already seen for this sample
                        if smiles not in unique_smiles_set:
                            unique_smiles_set.add(smiles)
                            samples_generated_for_this_val.append(smiles)
                            conversion_stats["conversion_success"] += 1
                        else:
                            conversion_stats["duplicates_skipped"] += 1
                    elif not smiles:
                        conversion_stats["conversion_empty"] += 1
                        logger.debug(f"Failed to convert SAFE to SMILES (empty): {safe_str[:50]}...")
                    else:
                        conversion_stats["conversion_fragments"] += 1
                        logger.debug(f"Failed to convert SAFE to SMILES (fragments): {safe_str[:50]}...")
                except Exception as e:
                    conversion_stats["conversion_failed"] += 1
                    logger.debug(f"Error converting SAFE to SMILES: {safe_str[:50]}... Error: {e}")
            
            attempt += 1
            if attempt % 10 == 0:
                logger.debug(f"  Sample {val_idx}: Generated {len(samples_generated_for_this_val)}/{num_samples} valid molecules (attempt {attempt})")
        
        if len(samples_generated_for_this_val) < num_samples:
            logger.warning(f"  Sample {val_idx}: Only generated {len(samples_generated_for_this_val)}/{num_samples} valid molecules after {attempt} attempts")
        
        # Store results
        all_generated.extend(samples_generated_for_this_val)
        all_condition_info.append({
            "validation_index": int(val_idx),
            "prefix_safe": prefix_safe,
            "generated_molecules": samples_generated_for_this_val,
        })
    
    # Log conversion statistics
    logger.info(f"\n  Overall conversion statistics:")
    logger.info(f"    Total SAFE generated: {conversion_stats['total_safe_generated']}")
    logger.info(f"    Successful conversions: {conversion_stats['conversion_success']}")
    logger.info(f"    Failed conversions (exceptions): {conversion_stats['conversion_failed']}")
    logger.info(f"    Empty conversions: {conversion_stats['conversion_empty']}")
    logger.info(f"    Fragment conversions: {conversion_stats['conversion_fragments']}")
    if conversion_stats['duplicates_skipped'] > 0:
        logger.info(f"    Duplicates skipped: {conversion_stats['duplicates_skipped']}")
    total_failed = (conversion_stats['conversion_failed'] + 
                   conversion_stats['conversion_empty'] + 
                   conversion_stats['conversion_fragments'])
    if conversion_stats['total_safe_generated'] > 0:
        failure_rate = (total_failed / conversion_stats['total_safe_generated']) * 100
        success_rate = (conversion_stats['conversion_success'] / conversion_stats['total_safe_generated']) * 100
        logger.info(f"    Success rate: {success_rate:.2f}% ({conversion_stats['conversion_success']}/{conversion_stats['total_safe_generated']})")
        logger.info(f"    Failure rate: {failure_rate:.2f}% ({total_failed}/{conversion_stats['total_safe_generated']})")
    
    return all_generated, all_condition_info, conversion_stats


def main():
    parser = argparse.ArgumentParser(
        description="Lead optimization: Generate molecules by continuing from SAFE prefixes"
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to full_model directory")
    parser.add_argument("--validation_sets", type=str, nargs="+", required=True, 
                       help="Path(s) to validation set(s)")
    parser.add_argument("--prefix_safe", type=str, default=None,
                       help="Single SAFE string to use as prefix for all samples")
    parser.add_argument("--prefix_file", type=str, default=None,
                       help="Path to file containing SAFE prefixes (one per line, one for each validation sample)")
    parser.add_argument("--num_samples_per_val", type=int, default=50, 
                       help="Number of molecules to generate per validation sample")
    parser.add_argument("--output_dir", type=str, required=True, 
                       help="Output directory for generated molecules")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for generation")
    parser.add_argument("--max_length", type=int, default=64, 
                       help="Maximum generation length (including prefix)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p (nucleus) sampling")
    parser.add_argument("--log_level", type=str, default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--sample_posterior", action="store_true", 
                       help="Sample from posterior for ligand latent")
    parser.add_argument("--max_retries", type=int, default=10, 
                       help="Maximum generation attempts per batch if conversion fails")
    parser.add_argument("--append_mode", action="store_true", default=True,
                       help="Append to prefix (default: True). If False, replace after prefix")
    parser.add_argument(
        "--ifp_ablation",
        type=str,
        default="none",
        choices=["none", "zero", "shuffle"],
        help="Ablation mode for ligand IFP condition",
    )
    parser.add_argument(
        "--cond_ablation",
        type=str,
        default="none",
        choices=["none", "zero", "shuffle"],
        help="Global ablation mode for ALL condition features",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Random seed for generation (default: 2025)",
    )
    
    args = parser.parse_args()
    
    # Validate prefix arguments
    if args.prefix_safe is None and args.prefix_file is None:
        parser.error("Either --prefix_safe or --prefix_file must be provided")
    if args.prefix_safe is not None and args.prefix_file is not None:
        parser.error("Cannot specify both --prefix_safe and --prefix_file")
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup
    logger = setup_logging(args.log_level)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Lead Optimization: Molecule Generation from SAFE Prefixes")
    logger.info("=" * 60)
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Validation sets: {args.validation_sets}")
    logger.info(f"Prefix SAFE: {args.prefix_safe if args.prefix_safe else f'from file: {args.prefix_file}'}")
    logger.info(f"Number of samples per validation sample: {args.num_samples_per_val}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Append mode: {args.append_mode}")
    
    # Load model and tokenizer
    logger.info("\n" + "=" * 60)
    logger.info("Loading model and tokenizer...")
    logger.info("=" * 60)
    model, tokenizer, device = load_model_and_tokenizer(args.model_path, logger)
    
    # Load validation sets
    logger.info("\n" + "=" * 60)
    logger.info("Loading validation sets...")
    logger.info("=" * 60)
    validation_sets = load_validation_sets(args.validation_sets, logger)
    
    if not validation_sets:
        logger.error("No validation sets loaded. Exiting.")
        return
    
    # Generate molecules for each validation set
    logger.info("\n" + "=" * 60)
    logger.info("Generating molecules...")
    logger.info("=" * 60)
    
    all_results = {}
    
    for val_name, val_dataset in validation_sets.items():
        logger.info(f"\nProcessing validation set: {val_name}")
        logger.info(f"  Dataset size: {len(val_dataset)}")
        
        # Load prefix SAFE strings
        prefix_safe_strings = load_prefix_safe_strings(
            prefix_safe=args.prefix_safe,
            prefix_file=args.prefix_file,
            num_samples=len(val_dataset),
            logger=logger,
        )
        
        generated_molecules, condition_info, conversion_stats = generate_molecules_with_prefix(
            model=model,
            tokenizer=tokenizer,
            validation_set=val_dataset,
            prefix_safe_strings=prefix_safe_strings,
            num_samples=args.num_samples_per_val,
            device=device,
            logger=logger,
            batch_size=args.batch_size,
            sample_posterior=args.sample_posterior,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_retries=args.max_retries,
            ifp_ablation=args.ifp_ablation,
            cond_ablation=args.cond_ablation,
            append_mode=args.append_mode,
        )
        
        logger.info(f"  Generated {len(generated_molecules)} valid SMILES molecules")
        
        # Save results
        results = {
            "validation_set": val_name,
            "num_samples": len(generated_molecules),
            "generated_molecules": generated_molecules,
            "condition_info": condition_info,
            "conversion_stats": conversion_stats,
            "generation_params": {
                "prefix_safe": args.prefix_safe,
                "prefix_file": args.prefix_file,
                "append_mode": args.append_mode,
                "batch_size": args.batch_size,
                "max_length": args.max_length,
                "temperature": args.temperature,
                "sample_posterior": args.sample_posterior,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "ifp_ablation": args.ifp_ablation,
                "cond_ablation": args.cond_ablation,
            }
        }
        
        # Save as JSON
        output_file = output_dir / f"lead_opt_{val_name}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"  Saved results to: {output_file}")
        
        # Save as text file
        txt_file = output_dir / f"lead_opt_{val_name}.txt"
        with open(txt_file, "w", encoding="utf-8") as f:
            for mol in generated_molecules:
                f.write(f"{mol}\n")
        logger.info(f"  Saved molecules to: {txt_file}")
        
        all_results[val_name] = results
    
    # Save summary
    summary = {
        "model_path": str(args.model_path),
        "validation_sets": args.validation_sets,
        "prefix_safe": args.prefix_safe,
        "prefix_file": args.prefix_file,
        "num_samples_per_val": args.num_samples_per_val,
        "total_generated": sum(len(r["generated_molecules"]) for r in all_results.values()),
        "results": {
            name: {
                "num_samples": len(r["generated_molecules"]),
                "conversion_stats": r.get("conversion_stats", {})
            } 
            for name, r in all_results.items()
        }
    }
    
    summary_file = output_dir / "lead_optimization_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"\nSaved summary to: {summary_file}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Lead optimization completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

