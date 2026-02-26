#!/usr/bin/env python3
"""
从已编码的向量生成配体分子。

这个脚本加载之前步骤计算得到的向量（ligand_vec, pocket_vec, evo_vec, IFP），
使用训练好的StructureSAFE模型生成新的配体分子。

Usage:
    python scripts/generate_ligands_from_vectors.py \
        --output_dir outputs/encode_pac1r \
        --model_path outputs/models/12_19_25_300M_lr5e-5_KLnormclamp5_ufa_gate1_info0.10.2_xattn_new/checkpoint-33400/full_model \
        --num_samples 10000
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

# 添加项目路径
project_root = Path(__file__).parent.parent
src_dir = project_root / "src"
scripts_dir = project_root / "scripts"
utils_dir = project_root / "utils"

sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(scripts_dir))
if utils_dir.exists():
    sys.path.insert(0, str(utils_dir))

from data_loader.molecule_tokenizer import MoleculeTokenizer
from data_loader.utils import safe_to_smiles
from transformers import AutoTokenizer
from typing import List, Dict, Tuple


def detect_model_type(model_path: Path, logger: logging.Logger):
    """
    自动检测模型类型（Tanimoto 或 InfoNCE）通过读取 config.json。
    
    Returns:
        tuple: (NovoMolGen class, NovoMolGenConfig class, model_type_name)
    """
    config_json_path = model_path / "config.json"
    
    if not config_json_path.exists():
        logger.warning(f"Config file not found at {config_json_path}, using InfoNCE model as default")
        from models.modeling_novomolgen_infonce_depot import NovoMolGen, NovoMolGenConfig
        return NovoMolGen, NovoMolGenConfig, "InfoNCE"
    
    try:
        import json
        with open(config_json_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        tanimoto_weight = config_dict.get("tanimoto_weight", 0.0)
        
        if tanimoto_weight > 0:
            from models.modeling_novomolgen_tanimoto import NovoMolGen, NovoMolGenConfig
            logger.info(f"Detected Tanimoto model (tanimoto_weight={tanimoto_weight})")
            return NovoMolGen, NovoMolGenConfig, "Tanimoto"
        else:
            from models.modeling_novomolgen_infonce_depot import NovoMolGen, NovoMolGenConfig
            logger.info("Detected InfoNCE model (tanimoto_weight=0 or not set)")
            return NovoMolGen, NovoMolGenConfig, "InfoNCE"
    except Exception as e:
        logger.warning(f"Error reading config.json: {e}. Using InfoNCE model as default")
        from models.modeling_novomolgen_infonce_depot import NovoMolGen, NovoMolGenConfig
        return NovoMolGen, NovoMolGenConfig, "InfoNCE"


def load_model_and_tokenizer(model_path: str, logger: logging.Logger):
    """Load model and tokenizer with automatic model type detection."""
    model_path = Path(model_path)
    
    logger.info(f"Loading model from: {model_path}")
    
    # Auto-detect model type
    NovoMolGen, NovoMolGenConfig, model_type = detect_model_type(model_path, logger)
    logger.info(f"Using {model_type} model class")
    
    # Load config
    config = NovoMolGenConfig.from_pretrained(str(model_path))
    logger.info(f"Model config: enable_cross_attn={config.enable_cross_attn}, cross_layers={config.cross_layers}")
    
    if getattr(config, "train_new_modules_only", False):
        logger.info("Overriding config.train_new_modules_only=True -> False for generation")
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
    
    final_model_dtype = next(model.parameters()).dtype
    logger.info(f"Model loaded on {device} with dtype {final_model_dtype}")
    
    return model, tokenizer, device


def generate_molecules_from_conditions(
    model,
    tokenizer,
    pocket_vec: np.ndarray,
    evo_vec: np.ndarray,
    ifp: np.ndarray,
    ligand_vec: np.ndarray,
    num_samples: int,
    device: torch.device,
    logger: logging.Logger,
    batch_size: int = 50,
    max_length: int = 64,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    max_retries: int = 10,
    sample_posterior: bool = False,
) -> Tuple[List[str], Dict]:
    """
    使用条件特征生成分子。
    
    Args:
        model: 训练好的模型
        tokenizer: Tokenizer
        pocket_vec: 口袋向量 [512]
        evo_vec: 进化向量 [1280]
        ifp: 相互作用指纹 [16384]
        ligand_vec: 配体向量 [1536]
        num_samples: 要生成的分子数量
        device: 设备
        logger: Logger
        batch_size: 批次大小
        max_length: 最大生成长度
        temperature: 采样温度
        top_k: Top-k 采样
        top_p: Top-p 采样
        max_retries: 最大重试次数
        sample_posterior: 是否从后验采样
    
    Returns:
        tuple: (generated_smiles_list, conversion_stats)
    """
    logger.info(f"Generating {num_samples} molecules using condition features")
    
    # Convert condition features to tensors
    model_dtype = next(model.parameters()).dtype
    
    pocket_vec_tensor = torch.tensor([pocket_vec], dtype=torch.float32).to(device).to(model_dtype)
    evo_vec_tensor = torch.tensor([evo_vec], dtype=torch.float32).to(device).to(model_dtype)
    ifp_tensor = torch.tensor([ifp], dtype=torch.float32).to(device).to(model_dtype)
    ligand_vec_tensor = torch.tensor([ligand_vec], dtype=torch.float32).to(device).to(model_dtype)
    
    conversion_stats = {
        "total_safe_generated": 0,
        "conversion_failed": 0,
        "conversion_empty": 0,
        "conversion_fragments": 0,
        "conversion_success": 0,
    }
    
    generated_molecules = []
    attempt = 0
    max_total_attempts = max_retries * (num_samples // batch_size + 1)
    
    while len(generated_molecules) < num_samples and attempt < max_total_attempts:
        needed = num_samples - len(generated_molecules)
        current_batch_size = min(batch_size, needed)
        
        # Repeat condition features for batch
        pocket_vec_batch = pocket_vec_tensor.repeat(current_batch_size, 1)
        evo_vec_batch = evo_vec_tensor.repeat(current_batch_size, 1)
        ifp_batch = ifp_tensor.repeat(current_batch_size, 1)
        ligand_vec_batch = ligand_vec_tensor.repeat(current_batch_size, 1)
        
        # Generate using condition features
        with torch.no_grad():
            input_ids = torch.tensor([[tokenizer.bos_token_id]], device=device).repeat(current_batch_size, 1)
            
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
            if len(generated_molecules) >= num_samples:
                break
            try:
                smiles = safe_to_smiles(safe_str)
                if smiles and '.' not in smiles:  # Only add if conversion succeeded and no fragments
                    generated_molecules.append(smiles)
                    conversion_stats["conversion_success"] += 1
                elif not smiles:
                    conversion_stats["conversion_empty"] += 1
                elif '.' in smiles:
                    conversion_stats["conversion_fragments"] += 1
            except Exception as e:
                conversion_stats["conversion_failed"] += 1
                if logger:
                    logger.debug(f"Conversion failed for SAFE string: {safe_str[:50]}... Error: {e}")
        
        attempt += 1
        if logger and attempt % 10 == 0:
            logger.info(f"Generated {len(generated_molecules)}/{num_samples} molecules (attempt {attempt})")
    
    if len(generated_molecules) < num_samples:
        logger.warning(f"Only generated {len(generated_molecules)}/{num_samples} molecules after {attempt} attempts")
    
    logger.info(f"Generation complete: {conversion_stats}")
    
    return generated_molecules, conversion_stats


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """设置日志记录器"""
    logger = logging.getLogger("generate_ligands")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def main():
    parser = argparse.ArgumentParser(
        description="从已编码的向量生成配体分子",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出目录，包含ligand_vec.npy, pocket_vec.npy, evo_vec.npy, ifp.npy"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="训练好的模型路径"
    )
    
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="要生成的配体数量"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="生成时的批次大小"
    )
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=64,
        help="生成序列的最大长度"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="生成温度"
    )
    
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k采样参数"
    )
    
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p采样参数"
    )
    
    parser.add_argument(
        "--sample_posterior",
        action="store_true",
        default=False,
        help="是否从后验采样（sample_posterior=True）"
    )
    
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别"
    )
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging(args.log_level)
    
    # 转换为Path对象
    output_dir = Path(args.output_dir)
    model_path = Path(args.model_path)
    
    # 检查输入文件
    ligand_vec_path = output_dir / "ligand_vec.npy"
    pocket_vec_path = output_dir / "pocket_vec.npy"
    evo_vec_path = output_dir / "evo_vec.npy"
    ifp_path = output_dir / "ifp.npy"
    
    print("=" * 60)
    print("从向量生成配体分子")
    print("=" * 60)
    
    # 检查文件是否存在
    for vec_path in [ligand_vec_path, pocket_vec_path, evo_vec_path, ifp_path]:
        if not vec_path.exists():
            raise FileNotFoundError(f"向量文件不存在: {vec_path}")
    
    if not model_path.exists():
        raise FileNotFoundError(f"模型路径不存在: {model_path}")
    
    # 6.1 加载所有向量
    print(f"\n6.1 加载所有向量...")
    ligand_vec = np.load(str(ligand_vec_path))
    pocket_vec = np.load(str(pocket_vec_path))
    evo_vec = np.load(str(evo_vec_path))
    ifp = np.load(str(ifp_path))
    
    print(f"  ✓ 所有向量已加载")
    print(f"    ligand_vec: {ligand_vec.shape}")
    print(f"    pocket_vec: {pocket_vec.shape}")
    print(f"    evo_vec: {evo_vec.shape}")
    print(f"    ifp: {ifp.shape}")
    
    # 6.2 加载模型和tokenizer
    print(f"\n6.2 加载模型和tokenizer: {model_path}")
    model, tokenizer, device = load_model_and_tokenizer(str(model_path), logger)
    print(f"  ✓ 模型已加载 (device: {device})")
    
    # 6.3 生成配体
    print(f"\n6.3 生成 {args.num_samples} 个配体分子...")
    generated_molecules, conversion_stats = generate_molecules_from_conditions(
        model=model,
        tokenizer=tokenizer,
        pocket_vec=pocket_vec,
        evo_vec=evo_vec,
        ifp=ifp,
        ligand_vec=ligand_vec,
        num_samples=args.num_samples,
        device=device,
        logger=logger,
        batch_size=args.batch_size,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_retries=10,
        sample_posterior=args.sample_posterior,
    )
    
    print(f"\n  ✓ 生成了 {len(generated_molecules)} 个有效配体分子")
    
    # 6.4 保存结果
    print(f"\n6.4 保存结果...")
    
    # 保存为文本文件
    txt_file = output_dir / "generated_molecules.txt"
    with open(txt_file, "w", encoding="utf-8") as f:
        for mol in generated_molecules:
            f.write(f"{mol}\n")
    print(f"  ✓ 保存到: {txt_file}")
    
    # 保存为JSON文件
    json_file = output_dir / "generated_molecules.json"
    results = {
        "model_path": str(model_path),
        "num_samples": len(generated_molecules),
        "generated_molecules": generated_molecules,
        "conversion_stats": conversion_stats,
        "condition_features": {
            "ligand_vec_shape": list(ligand_vec.shape),
            "pocket_vec_shape": list(pocket_vec.shape),
            "evo_vec_shape": list(evo_vec.shape),
            "ifp_shape": list(ifp.shape),
        },
        "generation_params": {
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "sample_posterior": args.sample_posterior,
        }
    }
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  ✓ 保存到: {json_file}")
    
    print("\n" + "=" * 60)
    print("生成完成！")
    print("=" * 60)
    print(f"\n总共生成了 {len(generated_molecules)} 个有效配体分子")
    print(f"结果保存在: {output_dir}")


if __name__ == "__main__":
    main()

