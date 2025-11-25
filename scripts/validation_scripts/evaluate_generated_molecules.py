#!/usr/bin/env python3
"""
评估生成的分子，计算各种指标。

用法:
    python scripts/evaluate_generated_molecules.py \
        --generated_json outputs/generated_molecules/with_ligand_info/generated_validation.json \
        --reference_smiles outputs/validation_smiles/all_ligand_smiles.smi \
        --output_file outputs/evaluation_results.json \
        --train_smiles optional_train_smiles.txt
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional

import rootutils
import sys

# Setup root directory
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.eval.molecule_evaluation import MoleculeEvaluator
from src.eval.components.moses import compute_intermediate_statistics


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging."""
    logger = logging.getLogger("evaluate")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def load_generated_molecules(generated_json_path: str, logger: logging.Logger) -> List[str]:
    """从 JSON 文件中加载生成的分子 SMILES."""
    logger.info(f"Loading generated molecules from: {generated_json_path}")
    
    with open(generated_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    generated_molecules = data.get("generated_molecules", [])
    logger.info(f"Loaded {len(generated_molecules)} generated molecules")
    
    return generated_molecules


def load_reference_smiles(reference_smiles_path: str, logger: logging.Logger) -> List[str]:
    """从 .smi 文件中加载参考 SMILES（忽略名称列）。"""
    logger.info(f"Loading reference SMILES from: {reference_smiles_path}")
    
    reference_smiles = []
    with open(reference_smiles_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                # .smi 文件格式可能是 "SMILES name" 或只有 "SMILES"
                parts = line.split()
                if parts:
                    smiles = parts[0]  # 取第一部分作为 SMILES
                    reference_smiles.append(smiles)
    
    logger.info(f"Loaded {len(reference_smiles)} reference SMILES")
    return reference_smiles


def load_train_smiles(train_smiles_path: Optional[str], logger: logging.Logger) -> Optional[List[str]]:
    """从文件中加载训练集 SMILES（用于 novelty 计算）。"""
    if train_smiles_path is None:
        return None
    
    logger.info(f"Loading training SMILES from: {train_smiles_path}")
    
    train_smiles = []
    train_path = Path(train_smiles_path)
    
    if train_path.suffix == '.smi':
        # .smi 文件格式
        with open(train_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if parts:
                        smiles = parts[0]
                        train_smiles.append(smiles)
    elif train_path.suffix == '.txt':
        # 纯文本文件，每行一个 SMILES
        with open(train_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    train_smiles.append(line)
    else:
        logger.warning(f"Unknown file format for training SMILES: {train_path.suffix}")
        return None
    
    logger.info(f"Loaded {len(train_smiles)} training SMILES")
    return train_smiles


def evaluate_molecules(
    generated_json_path: str,
    reference_smiles_path: str,
    output_file: str,
    logger: logging.Logger,
    train_smiles_path: Optional[str] = None,
    batch_size: int = 512,
    n_jobs: int = 30,
    device: str = "cuda",
):
    """
    评估生成的分子。
    
    Args:
        generated_json_path: 生成的 JSON 文件路径
        reference_smiles_path: 参考 SMILES 文件路径（用于 FCD, SNN, Scaf, Frag）
        output_file: 输出结果 JSON 文件路径
        logger: Logger 实例
        train_smiles_path: 训练集 SMILES 文件路径（用于 novelty，可选）
        batch_size: 批处理大小
        n_jobs: 并行任务数
        device: PyTorch 设备
    """
    # 1. 加载生成的分子
    generated_molecules = load_generated_molecules(generated_json_path, logger)
    
    if not generated_molecules:
        logger.error("No generated molecules found. Exiting.")
        return
    
    # 2. 加载参考 SMILES（用于片段指标）
    reference_smiles = load_reference_smiles(reference_smiles_path, logger)
    
    if not reference_smiles:
        logger.error("No reference SMILES found. Exiting.")
        return
    
    # 3. 加载训练集 SMILES（用于 novelty，可选）
    train_smiles = load_train_smiles(train_smiles_path, logger)
    
    # 4. 计算参考集的统计信息（用于 FCD, SNN, Scaf, Frag）
    # Fragment metrics (FCD, SNN, Scaf, Frag) 需要预先计算参考集的统计信息
    # 这些统计信息包括：
    #   - FCD: ChemNet 特征的预计算统计
    #   - SNN: 分子指纹的预计算
    #   - Scaf: 骨架分布的预计算
    #   - Frag: 片段分布的预计算
    # 然后生成分子会与这些预计算的参考统计进行比较
    logger.info("\n" + "=" * 60)
    logger.info("Computing reference statistics for fragment metrics...")
    logger.info("=" * 60)
    logger.info("Fragment metrics (FCD, SNN, Scaf, Frag) compare generated molecules")
    logger.info("with precomputed statistics from the reference set.")
    logger.info("This may take a while...")
    
    # 使用相同的参考集作为 valid_stats 和 test_stats
    # 在 molecule_evaluation.py 中，fragment metrics 会同时计算与 valid_stats 和 test_stats 的比较
    # 这里我们使用相同的参考集，所以两个结果应该是一样的
    reference_stats = compute_intermediate_statistics(
        reference_smiles,
        n_jobs=n_jobs,
        device=device,
        batch_size=batch_size,
    )
    
    logger.info("✅ Reference statistics computed")
    
    # 5. 设置评估任务
    task_names = [
        "SA",           # Synthetic Accessibility Score
        "QED",          # Quantitative Estimate of Drug-likeness
        "unique@10k",   # Uniqueness at 10k
        "IntDiv",       # Internal Diversity
        "FCD",          # Fréchet ChemNet Distance
        "SNN",          # Average Maximum Similarity
        "Scaf",         # Scaffold Similarity
        "Frag",         # Fragment Distribution
    ]
    
    # 注意：novelty 任务当前不支持自定义训练集
    # novelty 函数从 HuggingFace 数据集加载训练集，不支持直接传递 train_smiles
    # 如果需要计算 novelty，需要修改 molecule_evaluation.py 中的 novelty 调用
    # 或者使用 HuggingFace 数据集名称
    if train_smiles is not None:
        logger.warning("⚠️  Novelty calculation is skipped because the current implementation")
        logger.warning("    only supports loading training set from HuggingFace datasets.")
        logger.warning("    Custom training SMILES files are not supported.")
    
    # 6. 初始化评估器
    logger.info("\n" + "=" * 60)
    logger.info("Initializing MoleculeEvaluator...")
    logger.info("=" * 60)
    
    # 注意：不传递 train_smiles，因为 novelty 函数不支持自定义训练集
    # 它只能从 HuggingFace 数据集加载
    evaluator = MoleculeEvaluator(
        task_names=task_names,
        train_smiles=None,  # novelty 不支持自定义训练集，所以设为 None
        valid_stats=reference_stats,
        test_stats=reference_stats,  # 使用相同的参考集
        batch_size=batch_size,
        n_jobs=n_jobs,
        device=device,
    )
    
    # 7. 执行评估
    logger.info("\n" + "=" * 60)
    logger.info("Evaluating generated molecules...")
    logger.info("=" * 60)
    logger.info(f"Evaluating {len(generated_molecules)} molecules")
    logger.info(f"Tasks: {', '.join(task_names)}")
    
    results = evaluator(
        gen_smiles=generated_molecules,
        filter=True,  # 过滤无效分子
        return_valid_index=False,
    )
    
    # 8. 保存结果
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 准备输出数据
    output_data = {
        "evaluation_summary": {
            "num_generated_molecules": len(generated_molecules),
            "num_reference_molecules": len(reference_smiles),
            "tasks_evaluated": task_names,
        },
        "results": results,
    }
    
    # 添加一些统计信息
    if "validity" in results:
        output_data["evaluation_summary"]["validity"] = results["validity"]
        output_data["evaluation_summary"]["num_valid_molecules"] = int(
            results["validity"] * len(generated_molecules)
        )
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n✅ Results saved to: {output_path}")
    
    # 9. 打印摘要
    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Summary")
    logger.info("=" * 60)
    
    if "validity" in results:
        logger.info(f"Validity: {results['validity']:.4f}")
    
    for task_name in task_names:
        if task_name in results:
            score = results[task_name]
            if isinstance(score, dict):
                logger.info(f"{task_name}:")
                for key, value in score.items():
                    if isinstance(value, (int, float)):
                        logger.info(f"  {key}: {value:.4f}")
                    else:
                        logger.info(f"  {key}: {value}")
            elif isinstance(score, (int, float)):
                logger.info(f"{task_name}: {score:.4f}")
            elif isinstance(score, list):
                logger.info(f"{task_name}: {len(score)} values")
            else:
                logger.info(f"{task_name}: {score}")
    
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated molecules using MoleculeEvaluator"
    )
    parser.add_argument("--generated_json", type=str, required=True,
                       help="Path to generated JSON file (from generate_from_full_model.py)")
    parser.add_argument("--reference_smiles", type=str, required=True,
                       help="Path to reference SMILES file (e.g., all_ligand_smiles.smi)")
    parser.add_argument("--output_file", type=str, required=True,
                       help="Output JSON file path for evaluation results")
    parser.add_argument("--train_smiles", type=str, default=None,
                       help="Path to training SMILES file (optional, for novelty calculation)")
    parser.add_argument("--batch_size", type=int, default=512,
                       help="Batch size for evaluation (default: 512)")
    parser.add_argument("--n_jobs", type=int, default=30,
                       help="Number of parallel jobs (default: 30)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="PyTorch device (default: cuda)")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging(args.log_level)
    
    logger.info("=" * 60)
    logger.info("Evaluate Generated Molecules")
    logger.info("=" * 60)
    logger.info(f"Generated JSON: {args.generated_json}")
    logger.info(f"Reference SMILES: {args.reference_smiles}")
    logger.info(f"Output file: {args.output_file}")
    if args.train_smiles:
        logger.info(f"Training SMILES: {args.train_smiles}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Number of jobs: {args.n_jobs}")
    logger.info(f"Device: {args.device}")
    
    # Check files exist
    if not Path(args.generated_json).exists():
        logger.error(f"Generated JSON file not found: {args.generated_json}")
        return
    
    if not Path(args.reference_smiles).exists():
        logger.error(f"Reference SMILES file not found: {args.reference_smiles}")
        return
    
    if args.train_smiles and not Path(args.train_smiles).exists():
        logger.warning(f"Training SMILES file not found: {args.train_smiles}")
        logger.warning("Novelty calculation will be skipped")
        args.train_smiles = None
    
    # Evaluate
    evaluate_molecules(
        generated_json_path=args.generated_json,
        reference_smiles_path=args.reference_smiles,
        output_file=args.output_file,
        logger=logger,
        train_smiles_path=args.train_smiles,
        batch_size=args.batch_size,
        n_jobs=args.n_jobs,
        device=args.device,
    )
    
    logger.info("\n✅ Evaluation completed!")


if __name__ == "__main__":
    main()

