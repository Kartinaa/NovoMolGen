#!/usr/bin/env python3
"""
将生成的分子文件分发到对应的 receptor 文件夹中。

用法:
    python scripts/distribute_generated_molecules.py \
        --generated_json outputs/generated_molecules/generated_validation.json \
        --validation_set finetune_data/processed_data/hf_bdnv2_dataset/validation \
        --csv_file finetune_data/bdnv2_standardize.csv \
        --output_base_dir test_structureSAFE/validation_set
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from datasets import load_from_disk
import re


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging."""
    logger = logging.getLogger("distribute")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def load_csv_mapping(csv_path: str, logger: logging.Logger) -> Dict[str, str]:
    """
    从 CSV 文件加载 SMILES 到 CHEMBL ID 的映射。
    
    Returns:
        Dict mapping standardize_smi -> CHEMBLxxx (从 ligand_path 提取)
    """
    logger.info(f"Loading CSV mapping from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    if 'standardize_smi' not in df.columns or 'ligand_path' not in df.columns:
        raise ValueError(f"CSV must contain 'standardize_smi' and 'ligand_path' columns")
    
    mapping = {}
    for idx, row in df.iterrows():
        standardize_smi = str(row['standardize_smi']).strip()
        ligand_path = str(row['ligand_path']).strip()
        
        # 从 ligand_path 中提取 CHEMBLxxx
        # 格式: .../CHEMBLxxx/ligand.sdf
        match = re.search(r'/(CHEMBL\d+)/', ligand_path)
        if match:
            chembl_id = match.group(1)
            mapping[standardize_smi] = chembl_id
        else:
            logger.warning(f"Could not extract CHEMBL ID from path: {ligand_path}")
    
    logger.info(f"Loaded {len(mapping)} SMILES -> CHEMBL ID mappings")
    return mapping


def load_validation_set(validation_set_path: str, logger: logging.Logger):
    """加载 validation set 以获取原始 SMILES."""
    logger.info(f"Loading validation set from: {validation_set_path}")
    dataset = load_from_disk(validation_set_path)
    logger.info(f"Loaded validation set with {len(dataset)} samples")
    return dataset


def extract_chembl_id_from_path(ligand_path: str) -> Optional[str]:
    """从 ligand_path 中提取 CHEMBL ID."""
    match = re.search(r'/(CHEMBL\d+)/', ligand_path)
    if match:
        return match.group(1)
    return None


def distribute_molecules(
    generated_json_path: str,
    validation_set_path: str,
    csv_path: str,
    output_base_dir: str,
    logger: logging.Logger,
    output_filename: str = "generated_ligands.smi",
    index_folder_map: Optional[str] = None,
):
    """
    将生成的分子分发到对应的文件夹。
    
    Args:
        generated_json_path: 生成的 JSON 文件路径
        validation_set_path: Validation set 路径
        csv_path: CSV 文件路径（包含 SMILES 到路径的映射）
        output_base_dir: 输出基础目录（如 test_structureSAFE/validation_set）
        logger: Logger 实例
        output_filename: 输出文件名（默认: generated_ligands.smi）
    """
    # 1. 加载 CSV 映射
    csv_mapping = load_csv_mapping(csv_path, logger)
    
    # 2. 加载 validation set
    validation_set = load_validation_set(validation_set_path, logger)
    
    # 3. 加载生成的 JSON
    logger.info(f"Loading generated molecules from: {generated_json_path}")
    with open(generated_json_path, 'r', encoding='utf-8') as f:
        generated_data = json.load(f)
    
    # 4. 获取生成的分子和 condition_info
    generated_molecules = generated_data.get("generated_molecules", [])
    condition_info = generated_data.get("condition_info", [])
    
    logger.info(f"Found {len(generated_molecules)} generated molecules")
    logger.info(f"Found {len(condition_info)} condition info entries")
    
    # 5. 创建 SMILES -> validation_index 的映射
    smiles_to_index = {}
    for idx, sample in enumerate(validation_set):
        if 'SMILES' in sample:
            smiles = str(sample['SMILES']).strip()
            smiles_to_index[smiles] = idx
    
    logger.info(f"Created SMILES -> index mapping for {len(smiles_to_index)} samples")
    
    # 6. 为每个 validation sample 分发生成的分子
    output_base = Path(output_base_dir)
    
    distributed_count = 0
    not_found_count = 0
    # 用于记录 validation_index -> 输出文件夹路径 的映射
    index_to_folder: Dict[int, Path] = {}
    
    # 按 validation_index 分组生成的分子
    molecules_by_index = {}
    for info in condition_info:
        val_idx = info.get("validation_index")
        if val_idx is not None:
            molecules = info.get("generated_molecules", [])
            if val_idx not in molecules_by_index:
                molecules_by_index[val_idx] = []
            molecules_by_index[val_idx].extend(molecules)
    
    logger.info(f"Grouped molecules for {len(molecules_by_index)} validation samples")
    
    # 7. 为每个 validation sample 找到对应的 CHEMBL ID 并保存
    for val_idx in range(len(validation_set)):
        sample = validation_set[val_idx]
        
        # 获取原始 SMILES
        if 'SMILES' not in sample:
            logger.warning(f"Sample {val_idx} does not have 'SMILES' key, skipping")
            continue
        
        original_smiles = str(sample['SMILES']).strip()
        
        # 从 CSV 映射中查找对应的 CHEMBL ID
        chembl_id = csv_mapping.get(original_smiles)
        
        if not chembl_id:
            logger.warning(f"Sample {val_idx}: Could not find CHEMBL ID for SMILES: {original_smiles[:50]}...")
            not_found_count += 1
            continue
        
        # 获取该 sample 生成的分子
        generated_for_this_sample = molecules_by_index.get(val_idx, [])
        
        if not generated_for_this_sample:
            logger.warning(f"Sample {val_idx} (CHEMBL {chembl_id}): No generated molecules found")
            continue
        
        # 创建输出目录
        output_dir = output_base / chembl_id
        output_dir.mkdir(parents=True, exist_ok=True)
        # 记录 index -> folder 路径
        index_to_folder[val_idx] = output_dir
        
        # 保存生成的分子到 SMILES 文件（每行一个 SMILES）
        output_file = output_dir / output_filename
        with open(output_file, 'w', encoding='utf-8') as f:
            for smiles in generated_for_this_sample:
                f.write(f"{smiles}\n")
        
        logger.info(f"✅ Sample {val_idx} (CHEMBL {chembl_id}): Saved {len(generated_for_this_sample)} molecules to {output_file}")
        distributed_count += 1
    
    # 8. 如果需要，保存 validation_index -> 文件夹 路径映射
    if index_folder_map is not None:
        map_path = Path(index_folder_map)
        map_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving validation_index -> folder mapping to: {map_path}")
        with open(map_path, "w", encoding="utf-8") as f:
            for idx in sorted(index_to_folder.keys()):
                folder = index_to_folder[idx]
                # 与原 validation_folder.txt 兼容：第一列是路径，第二列是 validation_index
                f.write(f"{folder} {idx}\n")
    
    # 9. 总结
    logger.info("\n" + "=" * 60)
    logger.info("Distribution Summary")
    logger.info("=" * 60)
    logger.info(f"✅ Successfully distributed: {distributed_count} samples")
    logger.info(f"⚠️  Not found in CSV mapping: {not_found_count} samples")
    logger.info(f"📁 Output base directory: {output_base_dir}")
    if index_folder_map is not None:
        logger.info(f"🗺  Index-folder map saved to: {index_folder_map}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Distribute generated molecules to corresponding receptor folders")
    parser.add_argument("--generated_json", type=str, required=True,
                       help="Path to generated JSON file (from generate_from_full_model.py)")
    parser.add_argument("--validation_set", type=str, required=True,
                       help="Path to validation set (HF dataset)")
    parser.add_argument("--csv_file", type=str, required=True,
                       help="Path to CSV file with SMILES to ligand_path mapping")
    parser.add_argument("--output_base_dir", type=str, required=True,
                       help="Base directory for output (e.g., test_structureSAFE/validation_set)")
    parser.add_argument("--output_filename", type=str, required=True,
                       help="Output SMILES filename in each receptor folder (e.g., generated_ligands.smi)")
    parser.add_argument("--index_folder_map", type=str, default=None,
                       help="Optional path to save validation_index -> receptor folder mapping "
                            "(format similar to validation_folder.txt, with an extra validation_index column)")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging(args.log_level)
    
    logger.info("=" * 60)
    logger.info("Distribute Generated Molecules to Receptor Folders")
    logger.info("=" * 60)
    logger.info(f"Generated JSON: {args.generated_json}")
    logger.info(f"Validation set: {args.validation_set}")
    logger.info(f"CSV file: {args.csv_file}")
    logger.info(f"Output base directory: {args.output_base_dir}")
    logger.info(f"Output filename: {args.output_filename}")
    if args.index_folder_map:
        logger.info(f"Index-folder map file: {args.index_folder_map}")
    
    # Check files exist
    if not Path(args.generated_json).exists():
        logger.error(f"Generated JSON file not found: {args.generated_json}")
        return
    
    if not Path(args.validation_set).exists():
        logger.error(f"Validation set not found: {args.validation_set}")
        return
    
    if not Path(args.csv_file).exists():
        logger.error(f"CSV file not found: {args.csv_file}")
        return
    
    # Distribute molecules
    distribute_molecules(
        generated_json_path=args.generated_json,
        validation_set_path=args.validation_set,
        csv_path=args.csv_file,
        output_base_dir=args.output_base_dir,
        logger=logger,
        output_filename=args.output_filename,
        index_folder_map=args.index_folder_map,
    )
    
    logger.info("\n✅ Distribution completed!")


if __name__ == "__main__":
    main()

