#!/usr/bin/env python3
"""
De novo molecule generation from the pretrained base LLM (no fine-tuning, no conditioning).

Mirrors the generation logic of generate_from_full_model.py so that QED/SA
can be directly compared across:
  - base pretrained model (this script)  ← upper bound: how good is the raw LM?
  - v1 fine-tuned model                 ← conditioning collapsed, backbone also shifted
  - v4 fine-tuned model                 ← active conditioning, backbone shifted more

Usage:
    python scripts/generate_from_base_model.py \
        --model_path models/novomolgen_32M_YB/checkpoint-309520 \
        --num_samples 1000 \
        --output_dir outputs/generated_molecules/base_model
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_loader.molecule_tokenizer import MoleculeTokenizer
from data_loader.utils import safe_to_smiles
# Base NovoMolGen (no cross-attn adapters, no condition encoders)
from models.modeling_novomolgen import NovoMolGen, NovoMolGenConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("generate_base")
    logger.setLevel(getattr(logging, log_level.upper()))
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    return logger


def load_model_and_tokenizer(model_path: str, logger: logging.Logger):
    model_path = Path(model_path)
    logger.info(f"Loading base model from: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    config = NovoMolGenConfig.from_pretrained(str(model_path))
    model = NovoMolGen.from_pretrained(
        str(model_path),
        config=config,
        torch_dtype=model_dtype,
    )
    model = model.to(device)
    model.eval()

    tokenizer_path = model_path / "tokenizer.json"
    if tokenizer_path.exists():
        try:
            mol_tokenizer = MoleculeTokenizer.load(str(tokenizer_path))
            tokenizer = mol_tokenizer.get_pretrained()
            logger.info("Loaded MoleculeTokenizer")
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Model on {device} | dtype {next(model.parameters()).dtype}")
    return model, tokenizer, device


def generate_molecules(
    model,
    tokenizer,
    num_samples: int,
    device: torch.device,
    logger: logging.Logger,
    batch_size: int = 50,
    max_length: int = 64,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    max_retries: int = 10,
) -> tuple:
    stats = {k: 0 for k in ("total_safe_generated", "conversion_failed",
                             "conversion_empty", "conversion_fragments", "conversion_success")}
    generated = []
    attempt = 0
    max_total_attempts = max_retries * (num_samples // batch_size + 1)

    pbar = tqdm(total=num_samples, desc="Generating")
    while len(generated) < num_samples and attempt < max_total_attempts:
        current_batch_size = min(batch_size, num_samples - len(generated) + batch_size)

        autocast_enabled = device.type == "cuda"
        with torch.no_grad():
            ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if autocast_enabled else torch.no_grad()
            with ctx:
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
        stats["total_safe_generated"] += len(safe_strings)

        prev_count = len(generated)
        for safe_str in safe_strings:
            if len(generated) >= num_samples:
                break
            try:
                smi = safe_to_smiles(safe_str)
                if smi and "." not in smi:
                    generated.append(smi)
                    stats["conversion_success"] += 1
                elif not smi:
                    stats["conversion_empty"] += 1
                else:
                    stats["conversion_fragments"] += 1
            except Exception:
                stats["conversion_failed"] += 1

        pbar.update(len(generated) - prev_count)
        attempt += 1

    pbar.close()

    if len(generated) < num_samples:
        logger.warning(f"Only {len(generated)}/{num_samples} valid molecules after {attempt} attempts.")
    logger.info(f"Conversion stats: {stats}")
    return generated, stats


def main():
    parser = argparse.ArgumentParser(description="De novo generation from base pretrained LLM")
    parser.add_argument("--model_path",    type=str, required=True,
                        help="Path to base model checkpoint (e.g. models/novomolgen_32M_YB/checkpoint-309520)")
    parser.add_argument("--num_samples",   type=int, default=1000,
                        help="Total valid SMILES to collect")
    parser.add_argument("--output_dir",    type=str, required=True)
    parser.add_argument("--batch_size",    type=int, default=50)
    parser.add_argument("--max_length",    type=int, default=64)
    parser.add_argument("--temperature",   type=float, default=1.0)
    parser.add_argument("--top_k",         type=int, default=50)
    parser.add_argument("--top_p",         type=float, default=0.95)
    parser.add_argument("--max_retries",   type=int, default=10)
    parser.add_argument("--seed",          type=int, default=2025)
    parser.add_argument("--log_level",     type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    set_seed(args.seed)
    logger = setup_logging(args.log_level)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("De Novo Generation — Base Pretrained Model")
    logger.info("=" * 60)
    logger.info(f"Model      : {args.model_path}")
    logger.info(f"Num samples: {args.num_samples}")

    model, tokenizer, device = load_model_and_tokenizer(args.model_path, logger)

    generated, stats = generate_molecules(
        model=model, tokenizer=tokenizer,
        num_samples=args.num_samples, device=device, logger=logger,
        batch_size=args.batch_size, max_length=args.max_length,
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
        max_retries=args.max_retries,
    )

    output_path = output_dir / "generated_validation.json"
    with open(output_path, "w") as f:
        json.dump({
            "model_path":    str(args.model_path),
            "model_type":    "base_pretrained",
            "num_samples":   args.num_samples,
            "temperature":   args.temperature,
            "top_k":         args.top_k,
            "top_p":         args.top_p,
            "conversion_stats": stats,
            "generated_molecules": generated,
        }, f, indent=2)

    txt_path = output_dir / "generated_validation.txt"
    txt_path.write_text("\n".join(generated))

    logger.info(f"Saved {len(generated)} molecules → {output_path}")


if __name__ == "__main__":
    main()
