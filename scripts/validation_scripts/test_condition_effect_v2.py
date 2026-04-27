#!/usr/bin/env python3
"""
测试条件是否真正被模型使用：

思路：
  - 在 validation set 上，固定 input_ids（也就是同一批 SMILES 序列）
  - 计算两种 loss：
      1) 正确的条件：pocket_vec/evo_vec/ifp/ligand_vec 与 input_ids 一一对应
      2) 打乱的条件：将同一 batch 内的条件随机打乱后再喂给模型
  - 如果条件有用，(1) 的 NLL loss 应明显小于 (2)；如果几乎一样，说明条件可能没有被利用。

用法示例：
  python scripts/validation_scripts/test_condition_effect.py \
    --model_path outputs/11_19_25_SAFEGen_lr_KLnorm0.01_unfreezeall_noloar_newmodel/full_model \
    --validation_set finetune_data/processed_data/hf_bdnv2_dataset/validation \
    --num_batches 100 \
    --batch_size 32
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from datasets import load_from_disk, load_dataset, Features, Value, Sequence
import safe  # 用于 SMILES -> SAFE 转换

# 把 src 加到路径，方便导入模型
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from models.modeling_novomolgen_infonce_120225_v3 import NovoMolGen, NovoMolGenConfig  # type: ignore
from transformers import AutoTokenizer  # type: ignore


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("test_condition_effect")
    logger.setLevel(getattr(logging, log_level.upper()))
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def load_model_and_tokenizer(model_path: str, logger: logging.Logger):
    """与 generate_from_full_model.py 一致的加载方式。"""
    model_path = Path(model_path)

    logger.info(f"Loading model from: {model_path}")

    # Load config
    config = NovoMolGenConfig.from_pretrained(str(model_path))
    logger.info(
        f"Model config: enable_cross_attn={config.enable_cross_attn}, "
        f"cross_layers={config.cross_layers}"
    )

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

    # 确保 dtype 一致
    if torch.cuda.is_available() and model_dtype == torch.bfloat16:
        model = model.to(torch.bfloat16)
        param_dtypes = {p.dtype for p in model.parameters()}
        if torch.float32 in param_dtypes:
            logger.warning(
                f"Some parameters are still float32: {param_dtypes}. "
                "Converting all to bfloat16 again..."
            )
            model = model.to(torch.bfloat16)
        logger.info(f"Model converted to bfloat16. Parameter dtypes: {param_dtypes}")

    model.eval()

    # Load tokenizer（这里只是为了完整性，实际上我们直接用数据集里的 input_ids）
    tokenizer_path = model_path / "tokenizer.json"
    if tokenizer_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    final_model_dtype = next(model.parameters()).dtype
    logger.info(f"Model loaded on {device} with dtype {final_model_dtype}")

    return model, tokenizer, device


def load_validation_set(path: str, logger: logging.Logger, smiles_key: str = None):
    logger.info(f"Loading validation set from: {path}")
    try:
        ds = load_from_disk(path)
    except (TypeError, AttributeError, ValueError) as e:
        # Handle dataset compatibility issues (e.g., "must be called with a dataclass type or instance")
        # Fallback: load from arrow files with explicit features
        logger.warning(
            f"Failed to load validation set from {path} using load_from_disk: {e}\n"
            f"Attempting fallback: loading from arrow files with explicit features..."
        )
        try:
            path_obj = Path(path)
            arrow_files = sorted(path_obj.glob("*.arrow"))
            if arrow_files:
                # Define correct features with Sequence (not List)
                features = Features({
                    'SMILES': Value('string'),
                    'pocket_vec': Sequence(Value('float64')),
                    'evo_vec': Sequence(Value('float32')),
                    'ifp': Sequence(Value('float32')),
                    'ligand_vec': Sequence(Value('float32')),
                })
                ds = load_dataset(
                    "arrow",
                    data_files=[str(f) for f in arrow_files],
                    features=features,
                    split="train"
                )
                logger.info(f"Successfully loaded validation set using arrow fallback")
            else:
                raise ValueError(f"No arrow files found in {path}")
        except Exception as e2:
            logger.error(
                f"Fallback also failed: {e2}\n"
                f"Cannot load validation set: {path}"
            )
            raise
    
    logger.info(f"Validation set size: {len(ds)}")
    if len(ds) == 0:
        raise ValueError("Validation set is empty.")

    sample = ds[0]
    required_keys = ["pocket_vec", "evo_vec", "ifp", "ligand_vec"]
    if not all(k in sample for k in required_keys):
        raise ValueError(
            f"Validation sample does not contain all required condition keys: {required_keys}"
        )

    # 确定用于构造 input_ids 的文本字段（通常是 SMILES 或 SAFE）
    if smiles_key is not None:
        if smiles_key not in sample:
            raise ValueError(
                f"Specified smiles_key='{smiles_key}' not found in validation sample keys: "
                f"{list(sample.keys())}"
            )
        used_smiles_key = smiles_key
    else:
        candidate_keys = ["SMILES", "standardize_smi", "smiles", "safe", "SAFE"]
        used_smiles_key = None
        for k in candidate_keys:
            if k in sample:
                used_smiles_key = k
                break
        if used_smiles_key is None:
            raise ValueError(
                f"Could not find a SMILES/text field in validation sample. "
                f"Tried keys: {candidate_keys}. Available keys: {list(sample.keys())}"
            )

    logger.info(
        "Validation set contains required condition features. "
        f"Using '{used_smiles_key}' as text field for tokenization."
    )
    return ds, used_smiles_key


def make_batch(
    ds,
    indices: List[int],
    device: torch.device,
    tokenizer,
    smiles_key: str,
    convert_to_safe: bool = False,
    cond_dtype: torch.dtype = torch.float32,
    max_length: int = 64,
) -> Dict[str, torch.Tensor]:
    """从 datasets 中根据索引构造一个 batch（tensor 形式），用 tokenizer 从 SMILES/text 生成 input_ids。"""
    batch = ds.select(indices)

    texts = batch[smiles_key]

    # 如果需要，将 SMILES 转换为 SAFE，再喂给 SAFE tokenizer
    if convert_to_safe:
        safe_texts: List[str] = []
        for s in texts:
            try:
                safe_str = safe.encode(s, ignore_stereo=True)
            except Exception:
                # 转换失败时返回空字符串，后续 tokenizer 会只看到 padding，不影响相对比较
                safe_str = ""
            safe_texts.append(safe_str)
        texts = safe_texts

    # 使用与模型一致的 tokenizer 编码
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)

    # labels: 用 input_ids，并把 padding 位置置为 -100，避免影响 loss
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    # 条件特征使用与模型相同的 dtype，避免 Float/bfloat16 混用
    pocket_vec = torch.tensor(batch["pocket_vec"], dtype=cond_dtype, device=device)
    evo_vec = torch.tensor(batch["evo_vec"], dtype=cond_dtype, device=device)
    ifp = torch.tensor(batch["ifp"], dtype=cond_dtype, device=device)
    ligand_vec = torch.tensor(batch["ligand_vec"], dtype=cond_dtype, device=device)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "pocket_vec": pocket_vec,
        "evo_vec": evo_vec,
        "ifp": ifp,
        "ligand_vec": ligand_vec,
    }


@torch.inference_mode()
def measure_loss_difference(
    model: NovoMolGen,
    ds,
    device: torch.device,
    num_batches: int = 100,
    batch_size: int = 32,
    logger: logging.Logger = None,
    tokenizer=None,
    smiles_key: str = "SMILES",
    convert_smiles_to_safe: bool = False,
    shuffle_mode: str = "all",
    corrupt_type: str = "shuffle",
    max_length: int = 64,
) -> Tuple[float, float]:
    """在若干个 batch 上比较：
        - 正确条件 loss
        - 被“破坏”的条件 loss（打乱或置零）
    """
    import random

    n = len(ds)
    if n < batch_size:
        raise ValueError(f"Dataset too small ({n}) for batch_size={batch_size}")

    correct_losses: List[float] = []
    shuffled_losses: List[float] = []

    # 使用模型参数的 dtype 作为条件向量的 dtype
    model_dtype = next(model.parameters()).dtype

    for b in range(num_batches):
        # 随机抽一个 batch（不放回）
        indices = random.sample(range(n), batch_size)
        batch = make_batch(
            ds,
            indices,
            device,
            tokenizer,
            smiles_key,
            convert_to_safe=convert_smiles_to_safe,
            cond_dtype=model_dtype,
            max_length=max_length,
        )

        # 正确条件
        out_correct = model(
            input_ids=batch["input_ids"],
            labels=batch["labels"],
            attention_mask=batch["attention_mask"],
            pocket_vec=batch["pocket_vec"],
            evo_vec=batch["evo_vec"],
            ifp=batch["ifp"],
            ligand_vec=batch["ligand_vec"],
        )
        loss_correct = float(out_correct.loss.item())

        # 构造“破坏后的条件”（只在 batch 内打乱或置零）
        if corrupt_type == "shuffle":
            perm = torch.randperm(batch_size, device=device)
        else:
            perm = None

        # 根据 shuffle_mode 选择要打乱哪些条件：
        #   - "all"     : protein + ligand 都打乱（原始行为）
        #   - "protein" : 只打乱 pocket_vec / evo_vec
        #   - "ligand"  : 只打乱 ifp / ligand_vec
        if shuffle_mode in ("all", "protein"):
            if corrupt_type == "shuffle":
                pocket_vec_shuf = batch["pocket_vec"][perm]
                evo_vec_shuf = batch["evo_vec"][perm]
            elif corrupt_type == "zero":
                pocket_vec_shuf = torch.zeros_like(batch["pocket_vec"])
                evo_vec_shuf = torch.zeros_like(batch["evo_vec"])
        else:
            pocket_vec_shuf = batch["pocket_vec"]
            evo_vec_shuf = batch["evo_vec"]

        if shuffle_mode in ("all", "ligand"):
            if corrupt_type == "shuffle":
                ifp_shuf = batch["ifp"][perm]
                ligand_vec_shuf = batch["ligand_vec"][perm]
            elif corrupt_type == "zero":
                ifp_shuf = torch.zeros_like(batch["ifp"])
                ligand_vec_shuf = torch.zeros_like(batch["ligand_vec"])
        else:
            ifp_shuf = batch["ifp"]
            ligand_vec_shuf = batch["ligand_vec"]

        out_shuf = model(
            input_ids=batch["input_ids"],
            labels=batch["labels"],
            attention_mask=batch["attention_mask"],
            pocket_vec=pocket_vec_shuf,
            evo_vec=evo_vec_shuf,
            ifp=ifp_shuf,
            ligand_vec=ligand_vec_shuf,
        )
        loss_shuf = float(out_shuf.loss.item())

        correct_losses.append(loss_correct)
        shuffled_losses.append(loss_shuf)

        if logger is not None and (b + 1) % 10 == 0:
            logger.info(
                f"[Batch {b+1}/{num_batches}] "
                f"correct_loss={loss_correct:.4f}, shuffled_loss={loss_shuf:.4f}"
            )

    mean_correct = sum(correct_losses) / len(correct_losses)
    mean_shuffled = sum(shuffled_losses) / len(shuffled_losses)
    return mean_correct, mean_shuffled


def main():
    parser = argparse.ArgumentParser(
        description="Test whether conditional features affect NLL loss "
        "by comparing correct vs shuffled conditions."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to full_model directory",
    )
    parser.add_argument(
        "--validation_set",
        type=str,
        required=True,
        help="Path to validation set (HF dataset, same as training/ generation)",
    )
    parser.add_argument(
        "--smiles_key",
        type=str,
        default=None,
        help="Column name in validation set that contains SMILES/SAFE strings "
             "(default: auto-detect, prefer 'SMILES', 'standardize_smi', etc.)",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=100,
        help="Number of random batches to test (default: 100)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for testing (default: 32)",
    )
    parser.add_argument(
        "--convert_smiles_to_safe",
        action="store_true",
        help="If set, will convert the chosen text column (usually SMILES) to SAFE "
             "using safe.encode(..., ignore_stereo=True) before tokenization.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=64,
        help="Maximum tokenized sequence length for SAFE strings (default: 64)",
    )
    parser.add_argument(
        "--shuffle_mode",
        type=str,
        default="all",
        choices=["all", "protein", "ligand"],
        help=(
            "Which condition features to corrupt when computing the 'corrupted' loss:\n"
            "  - 'all'     : corrupt both protein (pocket_vec, evo_vec) and ligand (ifp, ligand_vec)\n"
            "  - 'protein' : only corrupt protein features (pocket_vec, evo_vec)\n"
            "  - 'ligand'  : only corrupt ligand features (ifp, ligand_vec)"
        ),
    )
    parser.add_argument(
        "--corrupt_type",
        type=str,
        default="shuffle",
        choices=["shuffle", "zero"],
        help=(
            "How to corrupt the selected condition features when computing the second loss:\n"
            "  - 'shuffle' : shuffle features within the batch (打乱对齐关系)\n"
            "  - 'zero'    : set selected condition features to zero (完全去掉条件信息)"
        ),
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()
    logger = setup_logging(args.log_level)

    logger.info("=" * 60)
    logger.info("Test Condition Effect via NLL Loss")
    logger.info("=" * 60)
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Validation set: {args.validation_set}")
    logger.info(f"num_batches: {args.num_batches}")
    logger.info(f"batch_size: {args.batch_size}")
    logger.info(f"shuffle_mode: {args.shuffle_mode}")
    logger.info(f"corrupt_type: {args.corrupt_type}")
    logger.info(f"max_length (tokens): {args.max_length}")

    # 加载模型和数据
    model, tokenizer, device = load_model_and_tokenizer(args.model_path, logger)
    ds, used_smiles_key = load_validation_set(
        args.validation_set, logger, smiles_key=args.smiles_key
    )

    # 计算 loss 差异
    mean_correct, mean_shuffled = measure_loss_difference(
        model=model,
        ds=ds,
        device=device,
        num_batches=args.num_batches,
        batch_size=args.batch_size,
        logger=logger,
        tokenizer=tokenizer,
        smiles_key=used_smiles_key,
        convert_smiles_to_safe=args.convert_smiles_to_safe,
        shuffle_mode=args.shuffle_mode,
        corrupt_type=args.corrupt_type,
        max_length=args.max_length,
    )

    logger.info("\n" + "=" * 60)
    logger.info("NLL Loss Comparison")
    logger.info("=" * 60)
    logger.info(f"Mean loss (correct conditions) : {mean_correct:.6f}")
    logger.info(f"Mean loss (shuffled conditions): {mean_shuffled:.6f}")
    diff = mean_shuffled - mean_correct
    rel = diff / abs(mean_correct) if mean_correct != 0 else 0.0
    logger.info(f"Absolute difference           : {diff:.6f}")
    logger.info(f"Relative difference           : {rel * 100:.2f}%")
    logger.info("=" * 60)

    logger.info(
        "解释：如果打乱条件后的 loss 明显更大，说明模型确实在利用条件信息；"
        "如果两者非常接近，则条件可能没有被有效使用。"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())


