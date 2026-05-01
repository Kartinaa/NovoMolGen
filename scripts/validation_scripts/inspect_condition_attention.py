#!/usr/bin/env python3
"""
Inspect how much protein vs ligand condition is used in cross-attention layers (v2 model).

Metrics per layer:
  - gate_value     : sigmoid(gate) — scale applied to cross-attn output
  - protein_attn   : mean attention weight on protein token (cond_tokens[:, 0, :])
  - ligand_attn    : mean attention weight on ligand  token (cond_tokens[:, 1, :])
                     averaged over batch × heads × sequence positions
  - attn_out_norm  : mean L2 norm of cross-attn output before gating
                     (how much signal the adapter is injecting)
  - effective_norm : gate_value × attn_out_norm (what actually reaches the residual stream)

Usage:
    python scripts/validation_scripts/inspect_condition_attention.py \
        --model_path outputs/models/040425_.../checkpoint-33400/full_model \
        --validation_set finetune_data/processed_data/hf_bdnv2_dataset/validation \
        --num_samples 64 \
        --convert_smiles_to_safe
"""

import argparse
import importlib
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import load_from_disk, load_dataset, Features, Value, Sequence
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))


def _resolve_model_module(module_name: str):
    """Dynamically import the requested model module and return (NovoMolGen, NovoMolGenConfig)."""
    full_name = f"models.{module_name}" if not module_name.startswith("models.") else module_name
    mod = importlib.import_module(full_name)
    return mod.NovoMolGen, mod.NovoMolGenConfig


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("inspect_cond_attn")
    logger.setLevel(getattr(logging, log_level.upper()))
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(h)
    return logger


# ---------------------------------------------------------------------------
# Model loading (same pattern as test_condition_effect_v2.py)
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_path: str, logger: logging.Logger,
                              NovoMolGen, NovoMolGenConfig):
    mp = Path(model_path)
    logger.info(f"Loading model from: {mp}")
    config = NovoMolGenConfig.from_pretrained(str(mp))
    logger.info(f"enable_cross_attn={config.enable_cross_attn}  cross_layers={config.cross_layers}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = NovoMolGen.from_pretrained(
        str(mp),
        config=config,
        torch_dtype=model_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = model.to(device)
    if torch.cuda.is_available():
        model = model.to(model_dtype)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(str(mp))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Model on {device}, dtype={next(model.parameters()).dtype}")
    return model, tokenizer, device


# ---------------------------------------------------------------------------
# Dataset loading + batch preparation
# ---------------------------------------------------------------------------

def load_validation_set(path: str, logger: logging.Logger):
    logger.info(f"Loading validation set from: {path}")
    try:
        ds = load_from_disk(path)
    except (TypeError, AttributeError, ValueError) as e:
        logger.warning(f"load_from_disk failed ({e}), trying arrow fallback...")
        arrow_files = sorted(Path(path).glob("*.arrow"))
        if not arrow_files:
            raise ValueError(f"No arrow files in {path}")
        features = Features({
            "SMILES":      Value("string"),
            "pocket_vec":  Sequence(Value("float64")),
            "evo_vec":     Sequence(Value("float32")),
            "ifp":         Sequence(Value("float32")),
            "ligand_vec":  Sequence(Value("float32")),
        })
        ds = load_dataset("arrow", data_files=[str(f) for f in arrow_files],
                          features=features, split="train")
    logger.info(f"  {len(ds)} samples")
    return ds


def make_batch(ds, indices, device, tokenizer, convert_to_safe, max_length, model_dtype):
    import safe as _safe

    batch = ds.select(indices)
    texts = batch["SMILES"]

    if convert_to_safe:
        safe_texts = []
        for s in texts:
            try:
                safe_texts.append(_safe.encode(s, ignore_stereo=True))
            except Exception:
                safe_texts.append("")
        texts = safe_texts

    enc = tokenizer(texts, return_tensors="pt", padding=True,
                    truncation=True, max_length=max_length)
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)
    labels         = input_ids.clone()
    labels[attention_mask == 0] = -100

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
        "pocket_vec":     torch.tensor(batch["pocket_vec"], dtype=model_dtype, device=device),
        "evo_vec":        torch.tensor(batch["evo_vec"],    dtype=model_dtype, device=device),
        "ifp":            torch.tensor(batch["ifp"],        dtype=model_dtype, device=device),
        "ligand_vec":     torch.tensor(batch["ligand_vec"], dtype=model_dtype, device=device),
    }


# ---------------------------------------------------------------------------
# Attention capture via temporary forward patch
# ---------------------------------------------------------------------------

def patch_adapters_for_capture(model) -> Dict[int, dict]:
    """
    Temporarily patch each CrossAttentionAdapter to also capture:
      - attn_weights  [B, num_heads, T, Lc]
      - attn_out_norm  mean L2 norm of adapter output (before gate)

    Returns a capture_store dict keyed by layer_idx.
    The patches are reversible: call restore_adapters() when done.
    """
    capture_store: Dict[int, dict] = {}
    original_forwards = {}

    if not (hasattr(model, "transformer") and
            hasattr(model.transformer, "cross_adapters") and
            model.transformer.cross_adapters is not None):
        return capture_store

    for layer_idx in model.transformer.cross_layers:
        key = str(layer_idx)
        if key not in model.transformer.cross_adapters:
            continue

        adapter = model.transformer.cross_adapters[key]
        original_forwards[layer_idx] = adapter.forward

        def make_patched(orig_forward, idx):
            def patched_forward(hidden_states, cond_tokens,
                                cond_attention_mask=None, return_attention_weights=False):
                # Always compute weights
                attn_out, weights = orig_forward(
                    hidden_states, cond_tokens, cond_attention_mask,
                    return_attention_weights=True,
                )
                # weights: [B, num_heads, T, Lc]  (Lc=2 for v2)
                if idx not in capture_store:
                    capture_store[idx] = {
                        "weights_sum":      None,
                        "weights_count":    0,
                        "out_norm_sum":     0.0,
                        "out_norm_count":   0,
                        "query_norm_sum":   0.0,   # hidden_states (normed) norm
                        "query_norm_count": 0,
                        "sample_count":     0,     # total number of (sample, position) pairs
                    }
                store = capture_store[idx]

                # Attention weight split: average over batch, heads, sequence positions
                if weights is not None:
                    w = weights.detach().float().cpu()   # [B, H, T, Lc]
                    # Accumulate per-batch mean so we can divide at the end
                    # Each call contributes one batch of B samples
                    w_mean = w.mean(dim=(0, 1, 2))       # [Lc]
                    if store["weights_sum"] is None:
                        store["weights_sum"] = w_mean
                    else:
                        store["weights_sum"] = store["weights_sum"] + w_mean
                    store["weights_count"] += 1

                B, T, _ = hidden_states.shape

                # L2 norm of cross-attn output (before gate), averaged over B×T
                out_norms = attn_out.detach().float().norm(dim=-1)   # [B, T]
                store["out_norm_sum"]   += float(out_norms.mean().item())
                store["out_norm_count"] += 1

                # L2 norm of query (hidden_states passed into adapter = normed main stream)
                # This is the signal the cross-attn is modifying
                q_norms = hidden_states.detach().float().norm(dim=-1)  # [B, T]
                store["query_norm_sum"]   += float(q_norms.mean().item())
                store["query_norm_count"] += 1

                store["sample_count"] += B

                if return_attention_weights:
                    return attn_out, weights
                return attn_out

            return patched_forward

        adapter.forward = make_patched(original_forwards[layer_idx], layer_idx)

    # Store originals so we can restore
    model._original_adapter_forwards = original_forwards
    return capture_store


def restore_adapters(model):
    if not hasattr(model, "_original_adapter_forwards"):
        return
    for layer_idx, orig in model._original_adapter_forwards.items():
        key = str(layer_idx)
        if key in model.transformer.cross_adapters:
            model.transformer.cross_adapters[key].forward = orig
    del model._original_adapter_forwards


# ---------------------------------------------------------------------------
# Main inspection
# ---------------------------------------------------------------------------

@torch.no_grad()
def inspect_attention(
    model,
    ds,
    device: torch.device,
    tokenizer,
    num_samples: int,
    batch_size: int,
    convert_to_safe: bool,
    max_length: int,
    logger: logging.Logger,
):
    model_dtype = next(model.parameters()).dtype
    n = len(ds)
    num_samples = min(num_samples, n)

    # 1. Gate values (static, no forward needed)
    logger.info("\n" + "=" * 60)
    logger.info("Gate values  [sigmoid(gate) = cross-attn mixing coefficient]")
    logger.info("=" * 60)
    gate_values: Dict[int, float] = {}
    for layer_idx in sorted(model.transformer.cross_layers):
        key = str(layer_idx)
        if key not in model.transformer.gates:
            continue
        gate_raw = model.transformer.gates[key].detach().cpu().float()
        gate_sig = float(torch.sigmoid(gate_raw).item())
        gate_values[layer_idx] = gate_sig
        logger.info(f"  Layer {layer_idx:2d}:  raw={float(gate_raw.item()):.4f}  sigmoid={gate_sig:.4f}")

    # 2. Patch adapters to capture attention weights
    capture_store = patch_adapters_for_capture(model)

    # 3. Forward passes
    logger.info(f"\nRunning {num_samples} samples in batches of {batch_size} ...")
    indices = list(range(num_samples))
    for start in range(0, num_samples, batch_size):
        batch_idx = indices[start: start + batch_size]
        batch = make_batch(ds, batch_idx, device, tokenizer,
                           convert_to_safe, max_length, model_dtype)
        model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            pocket_vec=batch["pocket_vec"],
            evo_vec=batch["evo_vec"],
            ifp=batch["ifp"],
            ligand_vec=batch["ligand_vec"],
        )

    restore_adapters(model)

    # 4. Report per-layer attention metrics
    logger.info("\n" + "=" * 60)
    logger.info("Cross-attention condition usage  (v2: token 0=protein, token 1=ligand)")
    logger.info(f"Averaged over all {num_samples} samples × attention heads × sequence positions")
    logger.info("=" * 60)
    logger.info(
        f"{'Layer':>6} {'gate':>7} {'protein%':>10} {'ligand%':>9} "
        f"{'query_norm':>11} {'attn_norm':>10} {'eff_norm':>10} {'ratio%':>8}"
    )
    logger.info("-" * 75)

    for layer_idx in sorted(model.transformer.cross_layers):
        gate = gate_values.get(layer_idx, float("nan"))
        store = capture_store.get(layer_idx, None)

        if store is None or store["weights_count"] == 0:
            logger.info(f"  Layer {layer_idx:2d}: no data captured")
            continue

        # Average attention weights: mean of per-batch means
        # Each batch already averaged over [B, H, T], so this is the global average
        w_avg = store["weights_sum"] / store["weights_count"]   # [Lc]
        if w_avg.shape[0] >= 2:
            prot_w = float(w_avg[0].item())
            lig_w  = float(w_avg[1].item())
        else:
            prot_w = float(w_avg[0].item())
            lig_w  = float("nan")

        # Mean norms (averaged the same way: mean of per-batch means)
        attn_norm  = store["out_norm_sum"]   / store["out_norm_count"]    # cross-attn output
        query_norm = store["query_norm_sum"] / store["query_norm_count"]  # main stream (normed)
        eff_norm   = gate * attn_norm

        # ratio: eff_norm / query_norm — how large is the injection relative to main stream
        ratio_pct = (eff_norm / query_norm * 100) if query_norm > 0 else float("nan")

        logger.info(
            f"  Layer {layer_idx:2d}: "
            f"gate={gate:.4f}  "
            f"prot={prot_w*100:5.1f}%  "
            f"lig={lig_w*100:5.1f}%  "
            f"query={query_norm:6.3f}  "
            f"attn={attn_norm:6.3f}  "
            f"eff={eff_norm:6.3f}  "
            f"ratio={ratio_pct:5.1f}%"
        )

    logger.info("=" * 75)
    logger.info("說明：")
    logger.info("  gate       : sigmoid(gate)，cross-attn輸出加入主幹的係數")
    logger.info("  protein%   : 注意力分配給protein token的比例（所有樣本×head×位置的平均）")
    logger.info("  ligand%    : 注意力分配給ligand  token的比例")
    logger.info("  query_norm : hidden_states（經過layer norm後）的平均L2範數 = main stream強度")
    logger.info("  attn_norm  : cross-attn輸出的平均L2範數（gate施加前）")
    logger.info("  eff_norm   : gate × attn_norm，實際注入residual stream的信號大小")
    logger.info("  ratio%     : eff_norm / query_norm × 100，conditioning佔主幹的百分比")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Inspect protein vs ligand attention weights in v2 cross-attention layers"
    )
    parser.add_argument("--model_path", required=True,
                        help="Path to full_model directory")
    parser.add_argument("--validation_set", required=True,
                        help="Path to HF validation dataset")
    parser.add_argument("--num_samples", type=int, default=64,
                        help="Number of validation samples to use (default: 64)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for forward pass (default: 32)")
    parser.add_argument("--convert_smiles_to_safe", action="store_true",
                        help="Convert SMILES column to SAFE before tokenization")
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--model_module", type=str,
                        default="modeling_novomolgen_infonce_120225_v4",
                        help="Model module name under src/models/ (e.g. modeling_novomolgen_infonce_120225_dropout for the 04/13/25 model)")
    parser.add_argument("--log_level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logger = setup_logging(args.log_level)
    logger.info("=" * 60)
    logger.info("Cross-Attention Condition Inspector")
    logger.info(f"  Using model module: {args.model_module}")
    logger.info("=" * 60)

    NovoMolGen, NovoMolGenConfig = _resolve_model_module(args.model_module)
    model, tokenizer, device = load_model_and_tokenizer(
        args.model_path, logger, NovoMolGen, NovoMolGenConfig
    )
    ds = load_validation_set(args.validation_set, logger)

    inspect_attention(
        model=model,
        ds=ds,
        device=device,
        tokenizer=tokenizer,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        convert_to_safe=args.convert_smiles_to_safe,
        max_length=args.max_length,
        logger=logger,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
