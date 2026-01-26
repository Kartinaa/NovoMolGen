#!/usr/bin/env python3
"""
计算每个 validation group 的平均 SA, QED, unique compounds，然后计算所有组的平均值。

用法:
    python scripts/evaluate_per_group_metrics.py \
        --input_json outputs/generated_molecules/with_ligand_info/generated_validation.json \
        --output_json outputs/per_group_metrics.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import random

import rootutils
from tqdm import tqdm

# Setup root directory
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from src.eval.components.moses import SA, QED
from src.eval.utils import get_mol
from multiprocess.pool import Pool


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging."""
    logger = logging.getLogger("evaluate_per_group")
    logger.setLevel(getattr(logging, log_level.upper()))

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def calculate_group_metrics(
    generated_molecules: List[str],
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    计算一组分子的平均 SA, QED, 和 unique compounds。
    
    Args:
        generated_molecules: SMILES 字符串列表
        logger: Logger 实例
    
    Returns:
        包含 SA, QED, unique_count, unique_ratio 的字典
    """
    if not generated_molecules:
        return {
            "sa_mean": 0.0,
            "qed_mean": 0.0,
            "unique_count": 0,
            "unique_ratio": 0.0,
            "valid_count": 0,
            "total_count": 0,
        }
    
    # 这个函数现在不再在主流程中使用，仅保留作参考或小规模调试。
    # 主评估逻辑已经改为一次性并行处理所有分子，以加速计算。
    sa_scores = []
    qed_scores = []
    valid_smiles = []
    unique_smiles_set = set()
    
    for smiles in generated_molecules:
        mol = get_mol(smiles)
        if mol is None:
            continue
        
        valid_smiles.append(smiles)
        
        try:
            sa_score = SA(mol)
            qed_score = QED(mol)
            sa_scores.append(sa_score)
            qed_scores.append(qed_score)
        except Exception as e:
            if logger is not None:
                logger.debug(f"Error calculating SA/QED for {smiles}: {e}")
            continue
        
        try:
            canonical_smiles = Chem.MolToSmiles(mol)
            unique_smiles_set.add(canonical_smiles)
        except Exception as e:
            if logger is not None:
                logger.debug(f"Error canonicalizing {smiles}: {e}")
            continue
    
    sa_mean = sum(sa_scores) / len(sa_scores) if sa_scores else 0.0
    qed_mean = sum(qed_scores) / len(qed_scores) if qed_scores else 0.0
    unique_count = len(unique_smiles_set)
    unique_ratio = unique_count / len(generated_molecules) if generated_molecules else 0.0
    
    return {
        "sa_mean": sa_mean,
        "qed_mean": qed_mean,
        "unique_count": unique_count,
        "unique_ratio": unique_ratio,
        "valid_count": len(valid_smiles),
        "total_count": len(generated_molecules),
    }


def _process_single_smiles(smiles: str) -> Optional[Dict[str, Any]]:
    """
    单个 SMILES 的处理函数，用于并行调用：
      - 解析为 RDKit Mol
      - 计算 SA、QED
      - 生成 canonical SMILES
    返回:
      dict(sa=..., qed=..., canonical=...) 或 None（无效分子）
    """
    mol = get_mol(smiles)
    if mol is None:
        return None
    try:
        sa_score = SA(mol)
        qed_score = QED(mol)
        canonical_smiles = Chem.MolToSmiles(mol)
        # 使用 Morgan 指纹用于后续 Tanimoto 相似度计算
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        # 计算 Bemis–Murcko scaffold 的 SMILES
        scaf_mol = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_smiles = Chem.MolToSmiles(scaf_mol) if scaf_mol is not None else None
    except Exception:
        return None
    return {
        "sa": sa_score,
        "qed": qed_score,
        "canonical": canonical_smiles,
        "fp": fp,
        "scaffold": scaffold_smiles,
    }


def evaluate_per_group_metrics(
    input_json_path: str,
    output_json_path: str,
    logger: logging.Logger,
    n_jobs: int = 30,
    validation_folder_list: Optional[str] = None,
    ligand_filename: str = "ligand.sdf",
    validation_set_path: Optional[str] = None,  # Path to validation dataset (for crossdock)
) -> Dict[str, Any]:
    """
    评估每个组的指标，然后计算所有组的平均值。
    
    Args:
        input_json_path: 输入的 JSON 文件路径
        output_json_path: 输出的 JSON 文件路径
        logger: Logger 实例
    
    Returns:
        包含所有组指标和平均值的字典
    """
    # 1. 加载 JSON 文件
    logger.info(f"Loading JSON file: {input_json_path}")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if "condition_info" not in data:
        logger.error("JSON file does not contain 'condition_info' key")
        raise ValueError("JSON file must contain 'condition_info' key")
    
    condition_info = data["condition_info"]
    num_groups = len(condition_info)
    logger.info(f"Found {num_groups} validation groups")
    
    # 2. 展开所有组的分子列表，便于一次性并行计算
    all_smiles: List[str] = []
    group_indices: List[int] = []          # 与 all_smiles 同长度，记录每个分子属于哪个 group
    validation_indices: List[int] = []     # 记录每个 group 对应的 validation_index
    
    for group_idx, group_info in enumerate(condition_info):
        validation_index = group_info.get("validation_index", group_idx)
        validation_indices.append(validation_index)
        generated_molecules = group_info.get("generated_molecules", [])
        
        if not generated_molecules:
            logger.warning(f"Group {validation_index} has no generated molecules")
            continue
        
        for smi in generated_molecules:
            all_smiles.append(smi)
            group_indices.append(group_idx)
    
    if not all_smiles:
        logger.error("No generated molecules found in any group.")
        return {
            "num_groups": 0,
            "overall_metrics": {},
            "per_group_metrics": [],
        }
    
    total_mols = len(all_smiles)
    logger.info(f"Total molecules across all groups: {total_mols}")
    logger.info(f"Using {n_jobs} parallel workers for RDKit / SA / QED evaluation")
    
    # 3. 并行计算所有分子的 SA、QED、canonical SMILES 和指纹，并显示进度条
    logger.info("Computing SA, QED, canonical SMILES and fingerprints for all molecules...")
    processed_list: List[Optional[Dict[str, Any]]] = []
    if n_jobs is None or n_jobs <= 1:
        # 单进程，直接用 tqdm 包一层
        for smi in tqdm(all_smiles, desc="Processing molecules", unit="mol"):
            processed_list.append(_process_single_smiles(smi))
    else:
        # 多进程，使用 Pool.imap_unordered 并结合 tqdm 显示进度
        with Pool(n_jobs) as pool:
            for res in tqdm(
                pool.imap_unordered(_process_single_smiles, all_smiles, chunksize=64),
                total=total_mols,
                desc="Processing molecules",
                unit="mol",
            ):
                processed_list.append(res)
    
    # 4. 按 group 聚合统计量
    group_sa_sum = [0.0 for _ in range(num_groups)]
    group_qed_sum = [0.0 for _ in range(num_groups)]
    group_valid_count = [0 for _ in range(num_groups)]
    group_total_count = [0 for _ in range(num_groups)]
    group_unique_sets = [set() for _ in range(num_groups)]
    group_fps: List[List[DataStructs.ExplicitBitVect]] = [[] for _ in range(num_groups)]
    group_scaffolds: List[List[Optional[str]]] = [[] for _ in range(num_groups)]
    
    for idx, result in enumerate(processed_list):
        g = group_indices[idx]
        group_total_count[g] += 1
        if result is None:
            continue
        sa = result["sa"]
        qed = result["qed"]
        canonical = result["canonical"]
        fp = result["fp"]
        scaffold = result.get("scaffold")
        group_sa_sum[g] += sa
        group_qed_sum[g] += qed
        group_valid_count[g] += 1
        group_unique_sets[g].add(canonical)
        group_fps[g].append(fp)
        group_scaffolds[g].append(scaffold)
    
    # 5. 计算每个组的指标
    group_metrics = []
    all_sa_means = []
    all_qed_means = []
    all_unique_counts = []
    all_unique_ratios = []
    # 用于计算"全局 unique rate"的集合（跨所有组去重）
    global_unique_set = set()
    # 用于 Bemis–Murcko scaffold 匹配统计
    all_scaffold_match_counts = []
    
    logger.info("Aggregating metrics per group...")

    # 如果提供了 validation_set_path，从验证集加载参考 ligand SMILES（用于 crossdock）
    validation_smiles_map: Dict[int, str] = {}
    if validation_set_path is not None:
        logger.info(f"Loading reference ligands from validation set: {validation_set_path}")
        try:
            from datasets import load_from_disk, load_dataset, Features, Value, Sequence
            val_path = Path(validation_set_path)
            if val_path.exists():
                try:
                    val_dataset = load_from_disk(str(val_path))
                except (TypeError, AttributeError, ValueError):
                    # Fallback: load from arrow files
                    arrow_files = sorted(val_path.glob("*.arrow"))
                    if arrow_files:
                        features = Features({
                            'SMILES': Value('string'),
                            'pocket_vec': Sequence(Value('float64')),
                            'evo_vec': Sequence(Value('float32')),
                            'ifp': Sequence(Value('float32')),
                            'ligand_vec': Sequence(Value('float32')),
                        })
                        val_dataset = load_dataset(
                            "arrow",
                            data_files=[str(f) for f in arrow_files],
                            features=features,
                            split="train"
                        )
                    else:
                        raise ValueError(f"No arrow files found in {val_path}")
                
                for idx in range(len(val_dataset)):
                    if 'SMILES' in val_dataset[idx]:
                        validation_smiles_map[idx] = val_dataset[idx]['SMILES']
                
                logger.info(f"Loaded {len(validation_smiles_map)} reference ligand SMILES from validation set")
        except Exception as e:
            logger.warning(f"Failed to load validation set: {e}. Will skip Tanimoto similarity calculation.")

    # 如果提供了 validation_folder_list，则预先加载对应的文件夹路径
    folder_paths: List[Path] = []
    idx_to_folder: Dict[int, Path] = {}
    # 记录每个 validation_index 对应 ligand 的指纹（用于"随机 reference"比较）
    ligand_fps: Dict[int, DataStructs.ExplicitBitVect] = {}
    if validation_folder_list is not None:
        folder_list_path = Path(validation_folder_list)
        if not folder_list_path.exists():
            logger.warning(f"Validation folder list file not found: {validation_folder_list}")
            validation_folder_list = None
        else:
            with open(folder_list_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    folder = Path(parts[0])
                    # 新格式：`<folder_path> <validation_index>`
                    if len(parts) >= 2:
                        try:
                            idx = int(parts[1])
                            idx_to_folder[idx] = folder
                        except ValueError:
                            folder_paths.append(folder)
                    else:
                        # 旧格式：仅有路径，按行号作为索引
                        folder_paths.append(folder)
            if idx_to_folder:
                logger.info(f"Loaded {len(idx_to_folder)} validation folders from index-aware map")
            else:
                logger.info(f"Loaded {len(folder_paths)} validation folders (legacy list, using line index as validation_index)")
    for g in range(num_groups):
        total_cnt = group_total_count[g]
        valid_cnt = group_valid_count[g]
        if total_cnt == 0:
            # 该组没有分子（或全部被跳过）
            continue
        
        sa_mean = group_sa_sum[g] / valid_cnt if valid_cnt > 0 else 0.0
        qed_mean = group_qed_sum[g] / valid_cnt if valid_cnt > 0 else 0.0
        unique_count = len(group_unique_sets[g])
        # 更新全局 unique 集合（跨所有组去重）
        if unique_count > 0:
            global_unique_set.update(group_unique_sets[g])
        unique_ratio = unique_count / total_cnt if total_cnt > 0 else 0.0
        tanimoto_mean = None
        tanimoto_max = None
        scaffold_match_count = None
        scaffold_match_ratio = None

        # 计算该组生成分子与参考配体的 Tanimoto 相似度
        # 优先使用 validation_set_path（crossdock），否则使用 validation_folder_list（molgenbench）
        if valid_cnt > 0:
            lig_fp = None
            lig_scaf_smiles = None
            val_idx = validation_indices[g]
            
            # 方法1: 从验证集直接获取参考 SMILES（crossdock）
            if validation_smiles_map and val_idx in validation_smiles_map:
                ref_smiles = validation_smiles_map[val_idx]
                try:
                    ref_mol = Chem.MolFromSmiles(ref_smiles)
                    if ref_mol is not None:
                        lig_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                            ref_mol, radius=2, nBits=2048
                        )
                        lig_scaf_mol = MurckoScaffold.GetScaffoldForMol(ref_mol)
                        lig_scaf_smiles = (
                            Chem.MolToSmiles(lig_scaf_mol)
                            if lig_scaf_mol is not None
                            else None
                        )
                        ligand_fps[val_idx] = lig_fp
                except Exception as e:
                    logger.debug(f"Failed to process reference SMILES for validation_index {val_idx}: {e}")
            
            # 方法2: 从文件夹读取 ligand SDF（molgenbench）
            elif validation_folder_list is not None and (folder_paths or idx_to_folder):
                lig_folder: Optional[Path] = None
                if idx_to_folder:
                    lig_folder = idx_to_folder.get(val_idx)
                elif folder_paths and 0 <= val_idx < len(folder_paths):
                    lig_folder = folder_paths[val_idx]

                if lig_folder is not None:
                    lig_path = lig_folder / ligand_filename
                    if lig_path.exists():
                        lig_mol = Chem.MolFromMolFile(str(lig_path))
                        if lig_mol is not None:
                            lig_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                                lig_mol, radius=2, nBits=2048
                            )
                            # 计算 ligand 的 Bemis–Murcko scaffold
                            lig_scaf_mol = MurckoScaffold.GetScaffoldForMol(lig_mol)
                            lig_scaf_smiles = (
                                Chem.MolToSmiles(lig_scaf_mol)
                                if lig_scaf_mol is not None
                                else None
                            )

                            # 缓存该 validation_index 的 ligand 指纹，供后续"随机 reference"比较使用
                            ligand_fps[val_idx] = lig_fp
            
            # 如果成功获取了 ligand 指纹，计算相似度
            if lig_fp is not None:

                sims = []
                for fp in group_fps[g]:
                    try:
                        sims.append(DataStructs.TanimotoSimilarity(fp, lig_fp))
                    except Exception:
                        continue
                if sims:
                    tanimoto_mean = float(sum(sims) / len(sims))
                    tanimoto_max = float(max(sims))

                # 统计与参考 ligand 具有相同 BM scaffold 的分子个数
                if lig_scaf_smiles is not None:
                    match_cnt = sum(
                        1
                        for scaf in group_scaffolds[g]
                        if scaf is not None and scaf == lig_scaf_smiles
                    )
                    scaffold_match_count = int(match_cnt)
                    scaffold_match_ratio = (
                        float(match_cnt) / float(valid_cnt)
                        if valid_cnt > 0
                        else 0.0
                    )
        
        metrics = {
            "validation_index": validation_indices[g],
            "num_molecules": total_cnt,
            "sa_mean": sa_mean,
            "qed_mean": qed_mean,
            "unique_count": unique_count,
            "unique_ratio": unique_ratio,
            "valid_count": valid_cnt,
            "total_count": total_cnt,
            "tanimoto_mean": tanimoto_mean,
            "tanimoto_max": tanimoto_max,
            "scaffold_match_count": scaffold_match_count,
            "scaffold_match_ratio": scaffold_match_ratio,
            # 暂存内部组索引，方便后续做“随机 reference”比较
            "group_id": g,
        }
        group_metrics.append(metrics)
        
        if valid_cnt > 0:
            all_sa_means.append(sa_mean)
            all_qed_means.append(qed_mean)
            all_unique_counts.append(unique_count)
            all_unique_ratios.append(unique_ratio)
            # 只有在成功计算了 Tanimoto / scaffold 时才计入整体平均
            if scaffold_match_count is not None:
                all_scaffold_match_counts.append(scaffold_match_count)
    
    # 3. 计算所有组的平均值
    # 计算整体指标（基于“每组统计后再平均”）
    overall_metrics = {
        "sa_mean": sum(all_sa_means) / len(all_sa_means) if all_sa_means else 0.0,
        "qed_mean": sum(all_qed_means) / len(all_qed_means) if all_qed_means else 0.0,
        "unique_count_mean": sum(all_unique_counts) / len(all_unique_counts) if all_unique_counts else 0.0,
        "unique_ratio_mean": sum(all_unique_ratios) / len(all_unique_ratios) if all_unique_ratios else 0.0,
    }

    # 4. 计算“全局 unique rate”：所有组的 canonical SMILES 合并后再去重
    global_unique_count = len(global_unique_set)
    total_valid_mols = sum(group_valid_count)
    overall_metrics["global_unique_count"] = int(global_unique_count)
    overall_metrics["global_unique_rate"] = (
        float(global_unique_count) / float(total_valid_mols)
        if total_valid_mols > 0
        else 0.0
    )

    # 计算所有组的平均 Tanimoto，相似度只在成功计算的组上取平均
    tanimoto_values = [
        m["tanimoto_mean"]
        for m in group_metrics
        if m.get("tanimoto_mean") is not None
    ]
    if tanimoto_values:
        overall_metrics["tanimoto_mean"] = float(
            sum(tanimoto_values) / len(tanimoto_values)
        )

    # 计算所有组“最高相似度”的平均值：先对每组取最大，再在组之间取平均
    tanimoto_max_values = [
        m["tanimoto_max"]
        for m in group_metrics
        if m.get("tanimoto_max") is not None
    ]
    if tanimoto_max_values:
        overall_metrics["tanimoto_max_mean"] = float(
            sum(tanimoto_max_values) / len(tanimoto_max_values)
        )

    # 计算所有组的平均 scaffold 匹配个数（与 ligand 具有相同 BM scaffold）
    if all_scaffold_match_counts:
        overall_metrics["scaffold_match_count_mean"] = float(
            sum(all_scaffold_match_counts) / len(all_scaffold_match_counts)
        )

    # 5. 对每个组，再计算与“随机其它 reference ligand”的 Tanimoto 平均值
    #    用于与自身 reference 的 similarity 做对比，评估条件是否让生成更靠近“正确” ligand。
    random_tanimoto_means: List[float] = []
    available_val_indices = list(ligand_fps.keys())
    if available_val_indices:
        logger.info("Computing Tanimoto similarity to random reference ligands...")
        for metrics in group_metrics:
            g = metrics.get("group_id", None)
            val_idx = metrics.get("validation_index", None)
            if g is None or val_idx is None:
                continue

            fps_list = group_fps[g]
            if not fps_list:
                continue

            # 候选的“随机 reference” validation_index：必须有 ligand_fp，且 != 自己
            candidates = [idx for idx in available_val_indices if idx != val_idx]
            if not candidates:
                continue

            neg_idx = random.choice(candidates)
            neg_fp = ligand_fps.get(neg_idx, None)
            if neg_fp is None:
                continue

            sims = []
            for fp in fps_list:
                try:
                    sims.append(DataStructs.TanimotoSimilarity(fp, neg_fp))
                except Exception:
                    continue
            if sims:
                rand_mean = float(sum(sims) / len(sims))
                metrics["tanimoto_mean_random"] = rand_mean
                random_tanimoto_means.append(rand_mean)

    if random_tanimoto_means:
        overall_metrics["tanimoto_mean_random"] = float(
            sum(random_tanimoto_means) / len(random_tanimoto_means)
        )
    
    # 4. 准备输出结果
    results = {
        "num_groups": len(group_metrics),
        "overall_metrics": overall_metrics,
        "per_group_metrics": group_metrics,
    }
    
    # 5. 保存结果
    output_path = Path(output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to: {output_json_path}")
    
    # 6. 打印摘要
    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Summary")
    logger.info("=" * 60)
    logger.info(f"Total groups processed: {len(group_metrics)}")
    logger.info(f"\nOverall Metrics (averaged across all groups):")
    logger.info(f"  Average SA: {overall_metrics['sa_mean']:.4f}")
    logger.info(f"  Average QED: {overall_metrics['qed_mean']:.4f}")
    logger.info(f"  Average Unique Count: {overall_metrics['unique_count_mean']:.2f}")
    logger.info(f"  Average Unique Ratio: {overall_metrics['unique_ratio_mean']:.4f}")
    logger.info("=" * 60)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Calculate per-group metrics (SA, QED, unique) and overall averages"
    )
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="Path to input JSON file with condition_info (e.g., generated_validation.json)"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="outputs/per_group_metrics.json",
        help="Path to output JSON file for results"
    )
    parser.add_argument(
        "--validation_folder_list",
        type=str,
        default="test_structureSAFE/validation_set/validation_folder.txt",
        help="Path to validation_folder.txt mapping validation_index to CHEMBL folders",
    )
    parser.add_argument(
        "--ligand_filename",
        type=str,
        default="ligand.sdf",
        help="Ligand filename inside each CHEMBL folder (default: ligand.sdf)",
    )
    parser.add_argument(
        "--validation_set_path",
        type=str,
        default=None,
        help="Path to validation dataset (for crossdock, e.g., finetune_data/crossdock_dataset/hf_dataset_crossdock/validation)",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=30,
        help="Number of parallel workers for RDKit / SA / QED (default: 30)",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    logger.info("=" * 60)
    logger.info("Per-Group Metrics Evaluation")
    logger.info("=" * 60)
    logger.info(f"Input JSON: {args.input_json}")
    logger.info(f"Output JSON: {args.output_json}")
    logger.info(f"n_jobs: {args.n_jobs}")
    
    # Check input file exists
    input_path = Path(args.input_json)
    if not input_path.exists():
        logger.error(f"Input file not found: {args.input_json}")
        return 1
    
    # Evaluate metrics
    try:
        evaluate_per_group_metrics(
            input_json_path=args.input_json,
            output_json_path=args.output_json,
            logger=logger,
            n_jobs=args.n_jobs,
            validation_folder_list=args.validation_folder_list,
            ligand_filename=args.ligand_filename,
            validation_set_path=args.validation_set_path,
        )
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        return 1
    
    logger.info("\n✅ Evaluation completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

