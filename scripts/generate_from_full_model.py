#!/usr/bin/env python3
"""
Generate molecules from a trained full model checkpoint for each validation set.

Usage:
    python scripts/generate_from_full_model.py \
        --model_path outputs/11_10_25_SAFEGen/checkpoint-33400/full_model \
        --validation_sets finetune_data/processed_data/hf_bdnv2_dataset/validation \
        --num_samples_per_val 50 \
        --output_dir outputs/generated_molecules
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

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.modeling_novomolgen_infonce_120225_dropout import NovoMolGen, NovoMolGenConfig
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
    logger = logging.getLogger("generate")
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
    # 在生成階段不需要「只訓練新模塊 + 凍結 backbone」，這個選項主要是爲了 finetune。
    # 如果保持 True，會在 __init__ 裏調用 _maybe_setup_finetune_freeze_only()，
    # 而我們目前的 cond_proj 是 nn.Parameter 而不是 nn.Module，可能觸發
    # `_unfreeze_new_modules` 對 Parameter 調用 .parameters() 的錯誤。
    # 因此讀取 checkpoint 做生成時，強制關閉這個行爲。
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
    
    # CRITICAL: Explicitly convert model to bfloat16 if on CUDA
    # from_pretrained may not always convert all parameters correctly
    if torch.cuda.is_available() and model_dtype == torch.bfloat16:
        model = model.to(torch.bfloat16)
        # Double-check: verify all parameters are bfloat16
        param_dtypes = set(p.dtype for p in model.parameters())
        if torch.float32 in param_dtypes:
            logger.warning(f"Some parameters are still float32: {param_dtypes}. Converting all to bfloat16...")
            model = model.to(torch.bfloat16)
        logger.info(f"Model converted to bfloat16. Parameter dtypes: {param_dtypes}")
    
    model.eval()
    
    # Ensure model is in the correct dtype for all operations
    if torch.cuda.is_available():
        # Set autocast dtype for the model
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Verify final model dtype
    final_model_dtype = next(model.parameters()).dtype
    logger.info(f"Model loaded on {device} with dtype {final_model_dtype}")
    
    # Load tokenizer
    tokenizer_path = model_path / "tokenizer.json"
    if tokenizer_path.exists():
        # Try loading as MoleculeTokenizer first
        try:
            mol_tokenizer = MoleculeTokenizer.load(str(tokenizer_path))
            tokenizer = mol_tokenizer.get_pretrained()
            logger.info(f"Loaded tokenizer as MoleculeTokenizer from {tokenizer_path}")
        except Exception:
            # Fallback to AutoTokenizer
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
            dataset = load_from_disk(str(val_path))
            name = val_path.name
            validation_sets[name] = dataset
            logger.info(f"Loaded validation set '{name}': {len(dataset)} samples")
            
            # Check if condition features exist
            if len(dataset) > 0:
                sample = dataset[0]
                has_conditions = all(key in sample for key in ["pocket_vec", "evo_vec", "ifp", "ligand_vec"])
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
    sample_posterior: bool = False,
    max_length: int = 64,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    max_retries: int = 10,
    ifp_ablation: str = "none",
    cond_ablation: str = "none",
):
    """Generate molecules for a validation set using condition features.
    
    Args:
        max_retries: Maximum number of generation attempts per batch if conversion fails
    
    Returns:
        tuple: (all_generated, all_condition_info, conversion_stats)
            - all_generated: List of generated SMILES
            - all_condition_info: List of condition info for each validation sample
            - conversion_stats: Dict with conversion statistics
    """
    all_generated = []
    all_condition_info = []
    sample_posterior = sample_posterior
    
    # Statistics for SAFE to SMILES conversion
    conversion_stats = {
        "total_safe_generated": 0,  # Total SAFE strings generated
        "conversion_failed": 0,     # Failed conversions (exceptions)
        "conversion_empty": 0,      # Conversions that returned empty/None
        "conversion_fragments": 0,  # Conversions that resulted in fragments (containing '.')
        "conversion_success": 0,    # Successful conversions
    }
    
    # Check if condition features are available
    if len(validation_set) == 0:
        logger.warning("Validation set is empty")
        return all_generated, all_condition_info, conversion_stats
    
    sample = validation_set[0]
    has_conditions = all(key in sample for key in ["pocket_vec", "evo_vec", "ifp", "ligand_vec"])
    
    if not has_conditions:
        logger.warning("Condition features not found in validation set, generating without conditions")
        # Generate without conditions - keep generating until we have num_samples valid SMILES
        logger.info(f"Generating {num_samples} molecules (will retry if conversion fails)")
        attempt = 0
        while len(all_generated) < num_samples and attempt < max_retries * (num_samples // batch_size + 1):
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
                # Model generates in SAFE format, convert to SMILES
                safe_strings = outputs[model.mol_type]
                conversion_stats["total_safe_generated"] += len(safe_strings)
                
                for safe_str in safe_strings:
                    if len(all_generated) >= num_samples:
                        break
                    try:
                        smiles = safe_to_smiles(safe_str)
                        if smiles and '.' not in smiles:  # Only add valid SMILES without fragments
                            all_generated.append(smiles)
                            conversion_stats["conversion_success"] += 1
                        elif not smiles:
                            # Conversion returned None or empty string
                            conversion_stats["conversion_empty"] += 1
                            logger.debug(f"Failed to convert SAFE to SMILES (empty result): {safe_str[:50]}...")
                        else:
                            # Contains fragments ('.')
                            conversion_stats["conversion_fragments"] += 1
                            logger.debug(f"Failed to convert SAFE to SMILES (fragments): {safe_str[:50]}...")
                    except Exception as e:
                        conversion_stats["conversion_failed"] += 1
                        logger.debug(f"Error converting SAFE to SMILES: {safe_str[:50]}... Error: {e}")
            
            attempt += 1
            if attempt % 10 == 0:
                logger.info(f"  Generated {len(all_generated)}/{num_samples} valid molecules (attempt {attempt})")
        
        if len(all_generated) < num_samples:
            logger.warning(f"Only generated {len(all_generated)}/{num_samples} valid molecules after {attempt} attempts")
        
        # Log conversion statistics
        logger.info(f"  Conversion statistics:")
        logger.info(f"    Total SAFE generated: {conversion_stats['total_safe_generated']}")
        logger.info(f"    Successful conversions: {conversion_stats['conversion_success']}")
        logger.info(f"    Failed conversions (exceptions): {conversion_stats['conversion_failed']}")
        logger.info(f"    Empty conversions: {conversion_stats['conversion_empty']}")
        logger.info(f"    Fragment conversions: {conversion_stats['conversion_fragments']}")
        total_failed = (conversion_stats['conversion_failed'] + 
                       conversion_stats['conversion_empty'] + 
                       conversion_stats['conversion_fragments'])
        if conversion_stats['total_safe_generated'] > 0:
            failure_rate = (total_failed / conversion_stats['total_safe_generated']) * 100
            logger.info(f"    Failure rate: {failure_rate:.2f}% ({total_failed}/{conversion_stats['total_safe_generated']})")
    else:
        # Generate with condition features
        # For each sample in validation set, generate num_samples molecules
        logger.info(f"Generating {num_samples} molecules for each of {len(validation_set)} samples in validation set")
        logger.info(f"  Total molecules to generate: {len(validation_set) * num_samples}")
        if ifp_ablation != "none":
            logger.info(f"  IFP ablation mode is ON: ifp_ablation = '{ifp_ablation}'")
            logger.info("    'zero'   : set all IFP vectors in the batch to 0 (removes IFP information)")
            logger.info("    'shuffle': shuffle IFP vectors within each batch (keeps distribution, breaks alignment)")
        if cond_ablation != "none":
            logger.info(f"  GLOBAL condition ablation mode is ON: cond_ablation = '{cond_ablation}'")
            logger.info("    This will zero/shuffle pocket_vec, evo_vec, ifp and ligand_vec together.")
        
        # Get model dtype to ensure type compatibility with cross-attention
        model_dtype = next(model.parameters()).dtype
        
        # Iterate over each sample in validation set
        for val_idx in tqdm(range(len(validation_set)), desc="Processing validation samples"):
            val_sample = validation_set[val_idx]
            
            # Extract condition features for this sample
            pocket_vec = torch.tensor([val_sample["pocket_vec"]], dtype=torch.float32).to(device).to(model_dtype)
            evo_vec = torch.tensor([val_sample["evo_vec"]], dtype=torch.float32).to(device).to(model_dtype)
            ifp = torch.tensor([val_sample["ifp"]], dtype=torch.float32).to(device).to(model_dtype)
            ligand_vec = torch.tensor([val_sample["ligand_vec"]], dtype=torch.float32).to(device).to(model_dtype)
            
            # Generate num_samples molecules for this validation sample
            # Keep generating until we have exactly num_samples valid SMILES
            samples_generated_for_this_val = []
            attempt = 0
            max_total_attempts = max_retries * (num_samples // batch_size + 1)
            
            while len(samples_generated_for_this_val) < num_samples and attempt < max_total_attempts:
                needed = num_samples - len(samples_generated_for_this_val)
                current_batch_size = min(batch_size, needed)
                
                # Repeat condition features for batch
                pocket_vec_batch = pocket_vec.repeat(current_batch_size, 1)
                evo_vec_batch = evo_vec.repeat(current_batch_size, 1)
                ifp_batch = ifp.repeat(current_batch_size, 1)
                ligand_vec_batch = ligand_vec.repeat(current_batch_size, 1)

                # Global condition ablation (applies to all condition vectors together)
                if cond_ablation == "zero":
                    # Set ALL condition vectors to zero: effectively unconditional
                    pocket_vec_batch = torch.zeros_like(pocket_vec_batch)
                    evo_vec_batch = torch.zeros_like(evo_vec_batch)
                    ifp_batch = torch.zeros_like(ifp_batch)
                    ligand_vec_batch = torch.zeros_like(ligand_vec_batch)
                elif cond_ablation == "shuffle":
                    # Shuffle ALL condition vectors within the batch using the SAME permutation,
                    # so their mutual alignment is preserved but broken w.r.t. current sample.
                    perm = torch.randperm(current_batch_size, device=device)
                    pocket_vec_batch = pocket_vec_batch[perm]
                    evo_vec_batch = evo_vec_batch[perm]
                    ifp_batch = ifp_batch[perm]
                    ligand_vec_batch = ligand_vec_batch[perm]

                # If no global ablation, still allow IFP-only ablation for backwards compatibility
                if cond_ablation == "none":
                    if ifp_ablation == "zero":
                        # Extreme ablation: completely remove IFP information
                        ifp_batch = torch.zeros_like(ifp_batch)
                    elif ifp_ablation == "shuffle":
                        # Distribution-preserving ablation:
                        # shuffle IFP vectors within the batch so that they no longer
                        # correspond to the current ligand, but keep global statistics
                        perm_ifp = torch.randperm(current_batch_size, device=device)
                        ifp_batch = ifp_batch[perm_ifp]
                
                # Generate using condition features
                with torch.no_grad():
                    # Use generate_with_condition method
                    input_ids = torch.tensor([[tokenizer.bos_token_id]], device=device).repeat(current_batch_size, 1)
                    
                    # Use autocast to ensure all operations (including encoder outputs, layer_norm, and cross-attention)
                    # use correct dtype. This is critical for flash_attn's cross-attention which requires bfloat16/float16.
                    # Note: autocast must wrap the entire generation call to ensure all intermediate tensors have correct dtype
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
                                max_length=max_length,
                                temperature=temperature,
                                top_k=top_k,
                                top_p=top_p,
                                eos_token_id=tokenizer.eos_token_id,
                                # pad_token_id is not supported by flash_attn's generate(), removed
                            )
                    else:
                        # For CPU or float32, no autocast needed
                        generated_sequences = model.generate_with_condition(
                            input_ids=input_ids,
                            pocket_vec=pocket_vec_batch,
                            evo_vec=evo_vec_batch,
                            ifp=ifp_batch,
                            ligand_vec=ligand_vec_batch,
                            sample_posterior=sample_posterior,
                            max_length=max_length,
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                            eos_token_id=tokenizer.eos_token_id,
                            # pad_token_id is not supported by flash_attn's generate(), removed
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
                
                # Convert SAFE to SMILES - only add valid ones
                for safe_str in decoded_strings:
                    if len(samples_generated_for_this_val) >= num_samples:
                        break
                    try:
                        smiles = safe_to_smiles(safe_str)
                        if smiles and '.' not in smiles:  # Only add if conversion succeeded and no fragments
                            samples_generated_for_this_val.append(smiles)
                            conversion_stats["conversion_success"] += 1
                        elif not smiles:
                            # Conversion returned None or empty string
                            conversion_stats["conversion_empty"] += 1
                            logger.debug(f"Failed to convert SAFE to SMILES (empty result): {safe_str[:50]}...")
                        else:
                            # Contains fragments ('.')
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
            
            # Store all generated molecules for this validation sample
            all_generated.extend(samples_generated_for_this_val)
            
            # Store condition info for reference
            all_condition_info.append({
                "validation_index": int(val_idx),
                "generated_molecules": samples_generated_for_this_val,  # Already converted to SMILES
            })
        
        # Log conversion statistics for conditional generation
        logger.info(f"\n  Overall conversion statistics:")
        logger.info(f"    Total SAFE generated: {conversion_stats['total_safe_generated']}")
        logger.info(f"    Successful conversions: {conversion_stats['conversion_success']}")
        logger.info(f"    Failed conversions (exceptions): {conversion_stats['conversion_failed']}")
        logger.info(f"    Empty conversions: {conversion_stats['conversion_empty']}")
        logger.info(f"    Fragment conversions: {conversion_stats['conversion_fragments']}")
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
    parser = argparse.ArgumentParser(description="Generate molecules from trained model for validation sets")
    parser.add_argument("--model_path", type=str, required=True, help="Path to full_model directory")
    parser.add_argument("--validation_sets", type=str, nargs="+", required=True, 
                       help="Path(s) to validation set(s)")
    parser.add_argument("--num_samples_per_val", type=int, default=50, 
                       help="Number of molecules to generate per validation set")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for generated molecules")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for generation")
    parser.add_argument("--max_length", type=int, default=64, help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p (nucleus) sampling")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--sample_posterior", action="store_true", help="Sample from posterior if ligand and ifp feature are not available")
    parser.add_argument("--max_retries", type=int, default=10, help="Maximum number of generation attempts per batch if conversion fails")
    parser.add_argument(
        "--ifp_ablation",
        type=str,
        default="none",
        choices=["none", "zero", "shuffle"],
        help=(
            "Ablation mode for ligand IFP condition:\n"
            "  - 'none'   : use full IFP as in training (default)\n"
            "  - 'zero'   : set IFP vectors to zero (removes IFP information)\n"
            "  - 'shuffle': shuffle IFP vectors within each batch (keeps distribution, breaks alignment)"
        ),
    )
    parser.add_argument(
        "--cond_ablation",
        type=str,
        default="none",
        choices=["none", "zero", "shuffle"],
        help=(
            "Global ablation mode for ALL condition features (pocket_vec, evo_vec, ifp, ligand_vec):\n"
            "  - 'none'   : no global ablation (default)\n"
            "  - 'zero'   : set ALL condition vectors to zero (effectively unconditional)\n"
            "  - 'shuffle': shuffle ALL condition vectors within each batch together"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Random seed for generation (default: 2025)",
    )
    
    args = parser.parse_args()
    # Set random seed for reproducibility
    set_seed(args.seed)

    # Setup
    logger = setup_logging(args.log_level)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Molecule Generation from Trained Model")
    logger.info("=" * 60)
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Validation sets: {args.validation_sets}")
    logger.info(f"Number of samples per validation set: {args.num_samples_per_val}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"IFP ablation mode: {args.ifp_ablation}")
    logger.info(f"Global condition ablation mode: {args.cond_ablation}")
    
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
        
        generated_molecules, condition_info, conversion_stats = generate_molecules_for_validation_set(
            model=model,
            tokenizer=tokenizer,
            validation_set=val_dataset,
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
        output_file = output_dir / f"generated_{val_name}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"  Saved results to: {output_file}")
        
        # Also save as simple text file (one molecule per line)
        txt_file = output_dir / f"generated_{val_name}.txt"
        with open(txt_file, "w", encoding="utf-8") as f:
            for mol in generated_molecules:
                f.write(f"{mol}\n")
        logger.info(f"  Saved molecules to: {txt_file}")
        
        all_results[val_name] = results
    
    # Save summary
    summary = {
        "model_path": str(args.model_path),
        "validation_sets": args.validation_sets,
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
    
    summary_file = output_dir / "generation_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"\nSaved summary to: {summary_file}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Generation completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

