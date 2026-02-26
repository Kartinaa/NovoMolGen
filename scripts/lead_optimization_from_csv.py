#!/usr/bin/env python3
"""
Lead optimization from CSV: Generate molecules from breakpoint SAFE strings in CSV file.
Matches breakpoints to validation set conditions by UniProt_ID.

Usage:
    python scripts/lead_optimization_from_csv.py \
        --model_path outputs/model/checkpoint/full_model \
        --validation_set finetune_data/molgenbench_dataset/validation \
        --csv_file finetune_data/molgenbench_dataset/h2l_scaffold/h2l_breakpoint_scaffolds_with_safe.csv \
        --num_samples_per_breakpoint 50 \
        --output_dir outputs/lead_optimization_csv
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

import numpy as np
import random
import torch
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
    logger = logging.getLogger("lead_optimization_csv")
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


def load_validation_set(validation_set_path: str, logger: logging.Logger):
    """Load validation set from disk."""
    val_path = Path(validation_set_path)
    
    if not val_path.exists():
        raise FileNotFoundError(f"Validation set path does not exist: {val_path}")
    
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
                    'ligand_path': Value('string'),  # Required for extracting UniProt_ID
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
            raise RuntimeError(f"Fallback also failed: {e2}")
    
    logger.info(f"Loaded validation set: {len(dataset)} samples")
    
    if len(dataset) > 0:
        sample = dataset[0]
        has_conditions = all(key in sample for key in ["pocket_vec", "evo_vec", "ifp", "ligand_vec"])
        has_ligand_path = "ligand_path" in sample
        logger.info(f"  Condition features available: {has_conditions}")
        logger.info(f"  Ligand path available: {has_ligand_path}")
        
        if not has_conditions:
            raise ValueError("Validation set must contain condition features: pocket_vec, evo_vec, ifp, ligand_vec")
        if not has_ligand_path:
            raise ValueError("Validation set must contain 'ligand_path' for extracting UniProt_ID")
    
    return dataset


def extract_uniprot_id_from_path(ligand_path: str) -> str:
    """Extract UniProt_ID from ligand_path.
    
    Example: /path/to/O14757/O14757_lig.sdf -> O14757
    
    The file name format is {UniProt_ID}_lig.sdf, so we extract the part before '_lig.sdf'.
    """
    filename = Path(ligand_path).name  # Get filename from path
    # Remove '_lig.sdf' suffix to get UniProt_ID
    if filename.endswith('_lig.sdf'):
        uniprot_id = filename.replace('_lig.sdf', '')
        return uniprot_id
    else:
        raise ValueError(f"Unexpected filename format: {filename} (expected *_lig.sdf)")


def create_uniprot_to_conditions_mapping(validation_set, logger: logging.Logger) -> Dict[str, Dict]:
    """Create mapping from UniProt_ID to condition vectors.
    
    Returns:
        Dict mapping UniProt_ID -> {pocket_vec, evo_vec, ifp, ligand_vec}
    """
    mapping = {}
    
    for idx, sample in enumerate(validation_set):
        ligand_path = sample.get('ligand_path', '')
        try:
            uniprot_id = extract_uniprot_id_from_path(ligand_path)
            if uniprot_id not in mapping:
                mapping[uniprot_id] = {
                    'pocket_vec': sample['pocket_vec'],
                    'evo_vec': sample['evo_vec'],
                    'ifp': sample['ifp'],
                    'ligand_vec': sample['ligand_vec'],
                    'index': idx,
                    'ligand_path': ligand_path,
                }
                logger.debug(f"Mapped {uniprot_id} to validation sample {idx}")
            else:
                logger.warning(f"Duplicate UniProt_ID found: {uniprot_id} (using first occurrence)")
        except ValueError as e:
            logger.warning(f"Skipping sample {idx}: {e}")
    
    logger.info(f"Created mapping for {len(mapping)} unique UniProt_IDs")
    return mapping


def load_breakpoint_csv(csv_file: str, logger: logging.Logger) -> pd.DataFrame:
    """Load breakpoint CSV file.
    
    Expected columns: UniProt_ID, SeriseID, Breakpoint_SAFE
    """
    csv_path = Path(csv_file)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    df = pd.read_csv(csv_file)
    logger.info(f"Loaded CSV file: {csv_file}")
    logger.info(f"  Total rows: {len(df)}")
    logger.info(f"  Columns: {df.columns.tolist()}")
    
    # Validate required columns
    required_cols = ['UniProt_ID', 'SeriseID', 'Breakpoint_SAFE']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV file missing required columns: {missing_cols}")
    
    logger.info(f"  Unique UniProt_IDs: {df['UniProt_ID'].nunique()}")
    logger.info(f"  Unique SeriseIDs: {df['SeriseID'].nunique()}")
    
    return df


def generate_molecules_for_breakpoint(
    model: NovoMolGen,
    tokenizer,
    prefix_safe: str,
    conditions: Dict,
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
    add_dot_to_prefix: bool = True,
):
    """Generate molecules for a single breakpoint with given conditions.
    
    Returns:
        tuple: (generated_smiles_list, conversion_stats)
    """
    model_dtype = next(model.parameters()).dtype
    
    # Extract condition features
    pocket_vec = torch.tensor([conditions["pocket_vec"]], dtype=torch.float32).to(device).to(model_dtype)
    evo_vec = torch.tensor([conditions["evo_vec"]], dtype=torch.float32).to(device).to(model_dtype)
    ifp = torch.tensor([conditions["ifp"]], dtype=torch.float32).to(device).to(model_dtype)
    ligand_vec = torch.tensor([conditions["ligand_vec"]], dtype=torch.float32).to(device).to(model_dtype)
    
    # Add trailing dot to signal fragment completion (like SAFE does)
    # This tells the model to start a new fragment
    if add_dot_to_prefix:
        # Check if parentheses are balanced (safe to add dot)
        if prefix_safe.count("(") == prefix_safe.count(")"):
            prefix_safe = prefix_safe.rstrip(".") # + "."
    
    # Tokenize prefix SAFE string
    prefix_encoding = tokenizer(
        prefix_safe,
        return_tensors="pt",
        add_special_tokens=True,
    )
    prefix_ids = prefix_encoding["input_ids"].to(device)
    
    # Remove EOS token if present (critical for continuation!)
    # The model will think sequence is finished if EOS is in the prefix
    if prefix_ids[0, -1] == tokenizer.eos_token_id:
        prefix_ids = prefix_ids[:, :-1]
    
    conversion_stats = {
        "total_safe_generated": 0,
        "conversion_failed": 0,
        "conversion_empty": 0,
        "conversion_fragments": 0,
        "conversion_success": 0,
        "duplicates_skipped": 0,
    }
    
    generated_molecules = []
    unique_smiles_set = set()  # Track unique SMILES for deduplication
    attempt = 0
    # Increase max attempts to ensure we get enough unique molecules
    max_total_attempts = max_retries * (num_samples * 3 // batch_size + 1)
    
    while len(generated_molecules) < num_samples and attempt < max_total_attempts:
        needed = num_samples - len(generated_molecules)
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
        
        # Generate using condition features
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
            if len(generated_molecules) >= num_samples:
                break
            try:
                # Try to convert as-is first
                smiles = safe_to_smiles(safe_str)
                
                # # If failed and has trailing dot, try removing it
                # if not smiles and safe_str.endswith('.'):
                #     smiles = safe_to_smiles(safe_str.rstrip('.'))
                
                if smiles and '.' not in smiles:
                    # Deduplication: only add if not already seen
                    if smiles not in unique_smiles_set:
                        unique_smiles_set.add(smiles)
                        generated_molecules.append(smiles)
                        conversion_stats["conversion_success"] += 1
                    else:
                        conversion_stats["duplicates_skipped"] += 1
                elif not smiles:
                    conversion_stats["conversion_empty"] += 1
                else:
                    conversion_stats["conversion_fragments"] += 1
            except Exception as e:
                conversion_stats["conversion_failed"] += 1
        
        attempt += 1
    
    return generated_molecules, conversion_stats


def main():
    parser = argparse.ArgumentParser(
        description="Lead optimization from CSV: Generate molecules for each breakpoint SAFE string"
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to full_model directory")
    parser.add_argument("--validation_set", type=str, required=True, 
                       help="Path to validation set (contains condition vectors)")
    parser.add_argument("--csv_file", type=str, required=True,
                       help="Path to CSV file with breakpoint SAFE strings")
    parser.add_argument("--num_samples_per_breakpoint", type=int, default=50, 
                       help="Number of molecules to generate per breakpoint")
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
                       help="Condition token merging: True=append new cond tokens to existing, False=replace. Does NOT affect prefix/generation (default: True)")
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
    parser.add_argument(
        "--add_dot_to_prefix",
        action="store_true",
        default=True,
        help="Add trailing dot to prefix to signal fragment completion (like SAFE) (default: True)",
    )
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup
    logger = setup_logging(args.log_level)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("Lead Optimization from CSV: Breakpoint SAFE Generation")
    logger.info("=" * 80)
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Validation set: {args.validation_set}")
    logger.info(f"CSV file: {args.csv_file}")
    logger.info(f"Samples per breakpoint: {args.num_samples_per_breakpoint}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Append mode: {args.append_mode}")
    logger.info(f"Add dot to prefix: {args.add_dot_to_prefix}")
    logger.info(f"Sample posterior: {args.sample_posterior}")
    logger.info(f"Temperature: {args.temperature}, top_k: {args.top_k}, top_p: {args.top_p}")
    
    # Load model and tokenizer
    logger.info("\n" + "=" * 80)
    logger.info("Loading model and tokenizer...")
    logger.info("=" * 80)
    model, tokenizer, device = load_model_and_tokenizer(args.model_path, logger)
    
    # Load validation set
    logger.info("\n" + "=" * 80)
    logger.info("Loading validation set...")
    logger.info("=" * 80)
    validation_set = load_validation_set(args.validation_set, logger)
    
    # Create UniProt_ID to conditions mapping
    logger.info("\n" + "=" * 80)
    logger.info("Creating UniProt_ID to conditions mapping...")
    logger.info("=" * 80)
    uniprot_to_conditions = create_uniprot_to_conditions_mapping(validation_set, logger)
    
    # Load breakpoint CSV
    logger.info("\n" + "=" * 80)
    logger.info("Loading breakpoint CSV...")
    logger.info("=" * 80)
    breakpoint_df = load_breakpoint_csv(args.csv_file, logger)
    
    # Filter CSV to only include UniProt_IDs that have conditions
    original_len = len(breakpoint_df)
    breakpoint_df = breakpoint_df[breakpoint_df['UniProt_ID'].isin(uniprot_to_conditions.keys())]
    filtered_len = len(breakpoint_df)
    
    if filtered_len < original_len:
        logger.warning(
            f"Filtered out {original_len - filtered_len} breakpoints without matching conditions. "
            f"Remaining: {filtered_len}"
        )
    
    if filtered_len == 0:
        logger.error("No breakpoints with matching conditions found. Exiting.")
        return
    
    # Generate molecules for each breakpoint
    logger.info("\n" + "=" * 80)
    logger.info("Generating molecules for breakpoints...")
    logger.info("=" * 80)
    logger.info(f"Total breakpoints to process: {len(breakpoint_df)}")
    
    all_results = []
    overall_stats = {
        "total_safe_generated": 0,
        "conversion_failed": 0,
        "conversion_empty": 0,
        "conversion_fragments": 0,
        "conversion_success": 0,
        "duplicates_skipped": 0,
    }
    
    for idx, row in tqdm(breakpoint_df.iterrows(), total=len(breakpoint_df), desc="Processing breakpoints"):
        uniprot_id = row['UniProt_ID']
        serise_id = row['SeriseID']
        breakpoint_safe = row['Breakpoint_SAFE']
        
        # Note: Trailing dot handling is now done in generate_molecules_for_breakpoint
        # based on the add_dot_to_prefix parameter
        
        conditions = uniprot_to_conditions[uniprot_id]
        
        logger.debug(f"Processing {serise_id} ({uniprot_id}): {breakpoint_safe[:50]}...")
        
        generated_mols, conv_stats = generate_molecules_for_breakpoint(
            model=model,
            tokenizer=tokenizer,
            prefix_safe=breakpoint_safe,
            conditions=conditions,
            num_samples=args.num_samples_per_breakpoint,
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
            add_dot_to_prefix=args.add_dot_to_prefix,
        )
        
        # Update overall stats
        for key in overall_stats:
            overall_stats[key] += conv_stats[key]
        
        # Store results
        result = {
            "UniProt_ID": uniprot_id,
            "SeriseID": serise_id,
            "Breakpoint_SAFE": breakpoint_safe,
            "num_generated": len(generated_mols),
            "generated_molecules": generated_mols,
            "conversion_stats": conv_stats,
        }
        all_results.append(result)
        
        if len(generated_mols) < args.num_samples_per_breakpoint:
            logger.warning(
                f"  {serise_id}: Only generated {len(generated_mols)}/{args.num_samples_per_breakpoint} molecules"
            )
    
    # Log overall statistics
    logger.info("\n" + "=" * 80)
    logger.info("Overall Generation Statistics")
    logger.info("=" * 80)
    logger.info(f"Total breakpoints processed: {len(all_results)}")
    logger.info(f"Total SAFE sequences generated: {overall_stats['total_safe_generated']}")
    logger.info(f"Successful conversions: {overall_stats['conversion_success']}")
    logger.info(f"Failed conversions (exceptions): {overall_stats['conversion_failed']}")
    logger.info(f"Empty conversions: {overall_stats['conversion_empty']}")
    logger.info(f"Fragment conversions: {overall_stats['conversion_fragments']}")
    if overall_stats['duplicates_skipped'] > 0:
        logger.info(f"Duplicates skipped: {overall_stats['duplicates_skipped']}")
    
    total_failed = (overall_stats['conversion_failed'] + 
                   overall_stats['conversion_empty'] + 
                   overall_stats['conversion_fragments'])
    if overall_stats['total_safe_generated'] > 0:
        success_rate = (overall_stats['conversion_success'] / overall_stats['total_safe_generated']) * 100
        failure_rate = (total_failed / overall_stats['total_safe_generated']) * 100
        logger.info(f"Success rate: {success_rate:.2f}%")
        logger.info(f"Failure rate: {failure_rate:.2f}%")
    
    # Save results as JSON
    results_json = {
        "model_path": str(args.model_path),
        "validation_set": str(args.validation_set),
        "csv_file": str(args.csv_file),
        "num_samples_per_breakpoint": args.num_samples_per_breakpoint,
        "generation_params": {
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "temperature": args.temperature,
            "sample_posterior": args.sample_posterior,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "ifp_ablation": args.ifp_ablation,
            "cond_ablation": args.cond_ablation,
            "append_mode": args.append_mode,
            "add_dot_to_prefix": args.add_dot_to_prefix,
            "seed": args.seed,
        },
        "overall_stats": overall_stats,
        "results": all_results,
    }
    
    output_json = output_dir / "lead_optimization_results.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    logger.info(f"\nSaved results to: {output_json}")
    
    # Save as CSV for easy inspection
    results_csv_data = []
    for result in all_results:
        for mol in result['generated_molecules']:
            results_csv_data.append({
                'UniProt_ID': result['UniProt_ID'],
                'SeriseID': result['SeriseID'],
                'Breakpoint_SAFE': result['Breakpoint_SAFE'],
                'Generated_SMILES': mol,
            })
    
    if results_csv_data:
        results_df = pd.DataFrame(results_csv_data)
        output_csv = output_dir / "lead_optimization_results.csv"
        results_df.to_csv(output_csv, index=False)
        logger.info(f"Saved CSV results to: {output_csv}")
    
    # Save as text file (one SMILES per line with metadata)
    output_txt = output_dir / "lead_optimization_results.txt"
    with open(output_txt, "w", encoding="utf-8") as f:
        for result in all_results:
            f.write(f"# {result['SeriseID']} ({result['UniProt_ID']})\n")
            for mol in result['generated_molecules']:
                f.write(f"{mol}\n")
    logger.info(f"Saved text results to: {output_txt}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Lead optimization completed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

