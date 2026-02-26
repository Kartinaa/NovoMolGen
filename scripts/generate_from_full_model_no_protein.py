#!/usr/bin/env python3
"""
Generate molecules from a trained *no-protein-encoder ablation* model checkpoint.

相比 `generate_from_full_model_auto.py`，本脚本固定使用
`models.modeling_novomolgen_infonce_120225_no_protein` 中的 NovoMolGen / NovoMolGenConfig，
用于專門評估「去掉 protein_encoder、以零向量作為 protein_condition」的模型。

典型用法：

    python scripts/generate_from_full_model_no_protein.py \
        --model_path outputs/.../checkpoint-XXXX/full_model \
        --validation_sets finetune_data/processed_data/hf_bdnv2_dataset/validation \
        --num_samples_per_val 50 \
        --output_dir outputs/generated_molecules_no_protein
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import random
import torch
from datasets import load_from_disk
from tqdm import tqdm

import yaml  # 保持與原腳本一致，即便暫時未用到

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

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
    logger = logging.getLogger("generate_no_protein")
    logger.setLevel(getattr(logging, log_level.upper()))

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def load_model_and_tokenizer_no_protein(model_path: str, logger: logging.Logger):
    """
    Load *no-protein-encoder* NovoMolGen model and tokenizer from full_model directory.

    - 固定使用 `models.modeling_novomolgen_infonce_120225_no_protein.NovoMolGen`
    - 要求該 checkpoint 的 config.json 與 state_dict 與該類兼容
    """
    from models.modeling_novomolgen_infonce_120225_no_protein import (  # type: ignore
        NovoMolGen,
        NovoMolGenConfig,
    )

    model_path = Path(model_path)
    logger.info(f"[no_protein] Loading ablation model from: {model_path}")

    # Load config
    config = NovoMolGenConfig.from_pretrained(str(model_path))
    logger.info(
        "Model config: "
        f"enable_cross_attn={config.enable_cross_attn}, "
        f"cross_layers={config.cross_layers}, "
        f"cond_tokens_len={getattr(config, 'cond_tokens_len', 'N/A')}"
    )
    if hasattr(config, "infonce_weight"):
        logger.info(f"  infonce_weight={config.infonce_weight}")

    # 在生成階段不需要「只訓練新模塊 + 凍結 backbone」
    if getattr(config, "train_new_modules_only", False):
        logger.info(
            "[no_protein] Overriding config.train_new_modules_only=True -> False "
            "for generation (no freezing logic needed)."
        )
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

    # 顯式轉為 bfloat16（如果在 CUDA 上）
    if torch.cuda.is_available() and model_dtype == torch.bfloat16:
        model = model.to(torch.bfloat16)
        param_dtypes = set(p.dtype for p in model.parameters())
        if torch.float32 in param_dtypes:
            logger.warning(
                f"[no_protein] Some parameters are still float32: {param_dtypes}. "
                "Converting all to bfloat16 again..."
            )
            model = model.to(torch.bfloat16)
        logger.info(f"[no_protein] Model converted to bfloat16. Parameter dtypes: {param_dtypes}")

    model.eval()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    final_model_dtype = next(model.parameters()).dtype
    logger.info(f"[no_protein] Model loaded on {device} with dtype {final_model_dtype}")

    # Load tokenizer
    tokenizer_json = model_path / "tokenizer.json"
    if tokenizer_json.exists():
        # 優先嘗試 MoleculeTokenizer
        try:
            mol_tokenizer = MoleculeTokenizer.load(str(tokenizer_json))
            tokenizer = mol_tokenizer.get_pretrained()
            logger.info(f"Loaded tokenizer as MoleculeTokenizer from {tokenizer_json}")
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
            dataset = load_from_disk(str(val_path))
            name = val_path.name
            validation_sets[name] = dataset
            logger.info(f"Loaded validation set '{name}': {len(dataset)} samples")

            # Check if condition features exist
            if len(dataset) > 0:
                sample = dataset[0]
                has_conditions = all(
                    key in sample for key in ["pocket_vec", "evo_vec", "ifp", "ligand_vec"]
                )
                logger.info(f"  Condition features available: {has_conditions}")
        else:
            logger.warning(f"Validation set path does not exist: {val_path}")

    return validation_sets


def generate_molecules_for_validation_set_no_protein(
    model,
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
):
    """
    針對 *no-protein-encoder* 模型，對驗證集逐樣本生成分子。

    與原版 `generate_molecules_for_validation_set` 幾乎一致，唯一区别是：
    - 使用的模型是「內部已經將 protein_condition 設為零向量」的 ablation 版
    - 調用 `model.generate_with_condition(...)` 時仍會傳入 pocket_vec / evo_vec / ifp / ligand_vec，
      但 protein 信息在模型內部被忽略。
    """
    all_generated = []
    all_condition_info = []
    sample_posterior = sample_posterior

    # SAFE → SMILES 統計
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
    has_conditions = all(key in sample for key in ["pocket_vec", "evo_vec", "ifp", "ligand_vec"])

    if not has_conditions:
        logger.warning(
            "[no_protein] Condition features not found in validation set, generating without conditions"
        )
        # 無條件生成，與原腳本相同邏輯
        logger.info(f"Generating {num_samples} molecules (will retry if conversion fails)")
        attempt = 0
        while len(all_generated) < num_samples and attempt < max_retries * (
            num_samples // batch_size + 1
        ):
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
                        if smiles and "." not in smiles:
                            all_generated.append(smiles)
                            conversion_stats["conversion_success"] += 1
                        elif not smiles:
                            conversion_stats["conversion_empty"] += 1
                            logger.debug(
                                f"Failed to convert SAFE to SMILES (empty result): {safe_str[:50]}..."
                            )
                        else:
                            conversion_stats["conversion_fragments"] += 1
                            logger.debug(
                                f"Failed to convert SAFE to SMILES (fragments): {safe_str[:50]}..."
                            )
                    except Exception as e:
                        conversion_stats["conversion_failed"] += 1
                        logger.debug(
                            f"Error converting SAFE to SMILES: {safe_str[:50]}... Error: {e}"
                        )

            attempt += 1
            if attempt % 10 == 0:
                logger.info(
                    f"  Generated {len(all_generated)}/{num_samples} valid molecules (attempt {attempt})"
                )

        if len(all_generated) < num_samples:
            logger.warning(
                f"Only generated {len(all_generated)}/{num_samples} valid molecules after {attempt} attempts"
            )

        # 統計輸出
        logger.info("  Conversion statistics:")
        logger.info(f"    Total SAFE generated: {conversion_stats['total_safe_generated']}")
        logger.info(f"    Successful conversions: {conversion_stats['conversion_success']}")
        logger.info(f"    Failed conversions (exceptions): {conversion_stats['conversion_failed']}")
        logger.info(f"    Empty conversions: {conversion_stats['conversion_empty']}")
        logger.info(f"    Fragment conversions: {conversion_stats['conversion_fragments']}")
        total_failed = (
            conversion_stats["conversion_failed"]
            + conversion_stats["conversion_empty"]
            + conversion_stats["conversion_fragments"]
        )
        if conversion_stats["total_safe_generated"] > 0:
            failure_rate = (
                total_failed / conversion_stats["total_safe_generated"]
            ) * 100.0
            logger.info(
                f"    Failure rate: {failure_rate:.2f}% "
                f"({total_failed}/{conversion_stats['total_safe_generated']})"
            )
    else:
        # 有條件特徵：逐 validation sample 生成
        logger.info(
            f"[no_protein] Generating {num_samples} molecules for each of "
            f"{len(validation_set)} samples in validation set"
        )
        logger.info(
            f"  Total molecules to generate: {len(validation_set) * num_samples} "
            "(注意：protein_condition 在模型內部被設為零向量)"
        )

        model_dtype = next(model.parameters()).dtype

        for val_idx in tqdm(range(len(validation_set)), desc="Processing validation samples"):
            val_sample = validation_set[val_idx]

            pocket_vec = torch.tensor([val_sample["pocket_vec"]], dtype=torch.float32).to(device)
            evo_vec = torch.tensor([val_sample["evo_vec"]], dtype=torch.float32).to(device)
            ifp = torch.tensor([val_sample["ifp"]], dtype=torch.float32).to(device)
            ligand_vec = torch.tensor([val_sample["ligand_vec"]], dtype=torch.float32).to(device)

            pocket_vec = pocket_vec.to(model_dtype)
            evo_vec = evo_vec.to(model_dtype)
            ifp = ifp.to(model_dtype)
            ligand_vec = ligand_vec.to(model_dtype)

            samples_generated_for_this_val = []
            attempt = 0
            max_total_attempts = max_retries * (num_samples // batch_size + 1)

            while (
                len(samples_generated_for_this_val) < num_samples
                and attempt < max_total_attempts
            ):
                needed = num_samples - len(samples_generated_for_this_val)
                current_batch_size = min(batch_size, needed)

                pocket_vec_batch = pocket_vec.repeat(current_batch_size, 1)
                evo_vec_batch = evo_vec.repeat(current_batch_size, 1)
                ifp_batch = ifp.repeat(current_batch_size, 1)
                ligand_vec_batch = ligand_vec.repeat(current_batch_size, 1)

                with torch.no_grad():
                    input_ids = torch.tensor(
                        [[tokenizer.bos_token_id]], device=device
                    ).repeat(current_batch_size, 1)

                    autocast_enabled = device.type == "cuda" and model_dtype == torch.bfloat16
                    if autocast_enabled:
                        with torch.autocast(
                            device_type="cuda", dtype=model_dtype, enabled=True
                        ):
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
                            )
                    else:
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
                        )

                if isinstance(generated_sequences, dict):
                    sequences = generated_sequences.get(
                        "sequences", generated_sequences.get("full_safe")
                    )
                else:
                    sequences = generated_sequences

                sequences = model._filter_tokens_after_eos(
                    sequences, eos_id=tokenizer.eos_token_id
                )

                decoded_strings = tokenizer.batch_decode(
                    sequences, skip_special_tokens=True
                )
                decoded_strings = [s.replace(" ", "") for s in decoded_strings]
                conversion_stats["total_safe_generated"] += len(decoded_strings)

                for safe_str in decoded_strings:
                    if len(samples_generated_for_this_val) >= num_samples:
                        break
                    try:
                        smiles = safe_to_smiles(safe_str)
                        if smiles and "." not in smiles:
                            samples_generated_for_this_val.append(smiles)
                            conversion_stats["conversion_success"] += 1
                        elif not smiles:
                            conversion_stats["conversion_empty"] += 1
                            logger.debug(
                                f"Failed to convert SAFE to SMILES (empty result): {safe_str[:50]}..."
                            )
                        else:
                            conversion_stats["conversion_fragments"] += 1
                            logger.debug(
                                f"Failed to convert SAFE to SMILES (fragments): {safe_str[:50]}..."
                            )
                    except Exception as e:
                        conversion_stats["conversion_failed"] += 1
                        logger.debug(
                            f"Error converting SAFE to SMILES: {safe_str[:50]}... Error: {e}"
                        )

                attempt += 1
                if attempt % 10 == 0:
                    logger.debug(
                        f"  Sample {val_idx}: Generated "
                        f"{len(samples_generated_for_this_val)}/{num_samples} valid molecules "
                        f"(attempt {attempt})"
                    )

            if len(samples_generated_for_this_val) < num_samples:
                logger.warning(
                    f"  Sample {val_idx}: Only generated "
                    f"{len(samples_generated_for_this_val)}/{num_samples} valid molecules "
                    f"after {attempt} attempts"
                )

            all_generated.extend(samples_generated_for_this_val)
            all_condition_info.append(
                {
                    "validation_index": int(val_idx),
                    "generated_molecules": samples_generated_for_this_val,
                }
            )

        logger.info("\n  Overall conversion statistics:")
        logger.info(f"    Total SAFE generated: {conversion_stats['total_safe_generated']}")
        logger.info(f"    Successful conversions: {conversion_stats['conversion_success']}")
        logger.info(f"    Failed conversions (exceptions): {conversion_stats['conversion_failed']}")
        logger.info(f"    Empty conversions: {conversion_stats['conversion_empty']}")
        logger.info(f"    Fragment conversions: {conversion_stats['conversion_fragments']}")
        total_failed = (
            conversion_stats["conversion_failed"]
            + conversion_stats["conversion_empty"]
            + conversion_stats["conversion_fragments"]
        )
        if conversion_stats["total_safe_generated"] > 0:
            success_rate = (
                conversion_stats["conversion_success"]
                / conversion_stats["total_safe_generated"]
            ) * 100.0
            failure_rate = (total_failed / conversion_stats["total_safe_generated"]) * 100.0
            logger.info(
                f"    Success rate: {success_rate:.2f}% "
                f"({conversion_stats['conversion_success']}/{conversion_stats['total_safe_generated']})"
            )
            logger.info(
                f"    Failure rate: {failure_rate:.2f}% "
                f"({total_failed}/{conversion_stats['total_safe_generated']})"
            )

    return all_generated, all_condition_info, conversion_stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate molecules from *no-protein-encoder* ablation model for validation sets"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to full_model directory"
    )
    parser.add_argument(
        "--validation_sets",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to validation set(s)",
    )
    parser.add_argument(
        "--num_samples_per_val",
        type=int,
        default=50,
        help="Number of molecules to generate per validation set sample",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory for generated molecules"
    )
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for generation")
    parser.add_argument("--max_length", type=int, default=64, help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p (nucleus) sampling")
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--sample_posterior",
        action="store_true",
        help="Sample from posterior if ligand and ifp feature are available",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=10,
        help="Maximum number of generation attempts per batch if conversion fails",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Random seed for generation (default: 2025)",
    )

    args = parser.parse_args()
    set_seed(args.seed)

    logger = setup_logging(args.log_level)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Molecule Generation from No-Protein-Encoder Ablation Model")
    logger.info("=" * 60)
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Validation sets: {args.validation_sets}")
    logger.info(f"Number of samples per validation sample: {args.num_samples_per_val}")
    logger.info(f"Output directory: {output_dir}")

    # Load model and tokenizer
    logger.info("\n" + "=" * 60)
    logger.info("Loading ablation model and tokenizer...")
    logger.info("=" * 60)
    model, tokenizer, device = load_model_and_tokenizer_no_protein(args.model_path, logger)

    # Load validation sets
    logger.info("\n" + "=" * 60)
    logger.info("Loading validation sets...")
    logger.info("=" * 60)
    validation_sets = load_validation_sets(args.validation_sets, logger)

    if not validation_sets:
        logger.error("No validation sets loaded. Exiting.")
        return

    # Generate molecules
    logger.info("\n" + "=" * 60)
    logger.info("Generating molecules (no-protein ablation)...")
    logger.info("=" * 60)

    all_results = {}

    for val_name, val_dataset in validation_sets.items():
        logger.info(f"\nProcessing validation set: {val_name}")
        logger.info(f"  Dataset size: {len(val_dataset)}")

        generated_molecules, condition_info, conversion_stats = (
            generate_molecules_for_validation_set_no_protein(
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
            )
        )

        logger.info(f"  Generated {len(generated_molecules)} valid SMILES molecules")

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
            },
        }

        # Save as JSON
        output_file = output_dir / f"generated_{val_name}_no_protein.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"  Saved results to: {output_file}")

        # Also save as simple text file (one molecule per line)
        txt_file = output_dir / f"generated_{val_name}_no_protein.txt"
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
                "conversion_stats": r.get("conversion_stats", {}),
            }
            for name, r in all_results.items()
        },
        "ablation": {
            "type": "no_protein_encoder",
            "description": "Protein encoder removed; protein_condition is zero vector; "
            "ConditionFusion module kept.",
        },
    }

    summary_file = output_dir / "generation_summary_no_protein.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"\nSaved summary to: {summary_file}")

    logger.info("\n" + "=" * 60)
    logger.info("No-protein ablation generation completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()


