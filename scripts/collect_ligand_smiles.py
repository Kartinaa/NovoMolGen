#!/usr/bin/env python3
"""
从 validation_folder.txt 中列出的所有文件夹中提取 ligand.sdf，转换为 SMILES，并收集到一个 .smi 文件中。

用法:
    python scripts/collect_ligand_smiles.py \
        --folder_list test_structureSAFE/validation_set/validation_folder.txt \
        --output_file outputs/validation_smiles/all_ligand_smiles.smi
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available. Please install RDKit to use this script.")


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging."""
    logger = logging.getLogger("collect_smiles")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def sdf_to_smiles(sdf_path: Path, logger: logging.Logger) -> Optional[str]:
    """
    将 SDF 文件转换为 SMILES 字符串。
    
    Args:
        sdf_path: SDF 文件路径
        logger: Logger 实例
    
    Returns:
        SMILES 字符串，如果转换失败则返回 None
    """
    if not RDKIT_AVAILABLE:
        logger.error("RDKit is not available. Cannot convert SDF to SMILES.")
        return None
    
    try:
        # 使用 SDMolSupplier 读取 SDF 文件
        supplier = Chem.SDMolSupplier(str(sdf_path))
        
        # 获取第一个有效的分子
        for mol in supplier:
            if mol is not None:
                try:
                    # 移除氢原子
                    mol = Chem.RemoveHs(mol)
                    # 转换为 SMILES
                    smiles = Chem.MolToSmiles(mol)
                    if smiles:
                        return smiles
                except Exception as e:
                    logger.debug(f"Failed to convert molecule to SMILES in {sdf_path}: {e}")
                    continue
        
        logger.warning(f"No valid molecules found in {sdf_path}")
        return None
        
    except Exception as e:
        logger.error(f"Error reading SDF file {sdf_path}: {e}")
        return None


def read_folder_list(folder_list_path: Path, logger: logging.Logger) -> List[Path]:
    """
    从文件中读取文件夹路径列表。
    
    Args:
        folder_list_path: 包含文件夹路径的文件路径
        logger: Logger 实例
    
    Returns:
        文件夹路径列表
    """
    folders = []
    try:
        with open(folder_list_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):  # 跳过空行和注释
                    folder_path = Path(line)
                    folders.append(folder_path)
        
        logger.info(f"Read {len(folders)} folder paths from {folder_list_path}")
        return folders
        
    except Exception as e:
        logger.error(f"Error reading folder list from {folder_list_path}: {e}")
        return []


def collect_ligand_smiles(
    folder_list_path: str,
    output_file: str,
    logger: logging.Logger,
    ligand_filename: str = "ligand.sdf",
):
    """
    从所有文件夹中收集 ligand.sdf 文件，转换为 SMILES，并保存到一个 .smi 文件中。
    
    Args:
        folder_list_path: 包含文件夹路径列表的文件路径
        output_file: 输出 .smi 文件路径
        logger: Logger 实例
        ligand_filename: ligand 文件名（默认: ligand.sdf）
    """
    if not RDKIT_AVAILABLE:
        logger.error("RDKit is not available. Cannot convert SDF to SMILES.")
        return
    
    # 1. 读取文件夹列表
    folder_list = read_folder_list(Path(folder_list_path), logger)
    
    if not folder_list:
        logger.error("No folders found in folder list file.")
        return
    
    # 2. 收集所有 SMILES 和对应的文件夹名称
    all_smiles_with_names = []  # List of (smiles, folder_name) tuples
    processed_count = 0
    not_found_count = 0
    conversion_failed_count = 0
    
    logger.info(f"\nProcessing {len(folder_list)} folders...")
    
    for folder_path in folder_list:
        # 获取文件夹名称（CHEMBL ID）
        folder_name = folder_path.name
        
        # 查找 ligand.sdf 文件
        ligand_file = folder_path / ligand_filename
        
        if not ligand_file.exists():
            logger.warning(f"  ⚠️  {folder_name}: {ligand_filename} not found")
            not_found_count += 1
            continue
        
        # 转换为 SMILES
        smiles = sdf_to_smiles(ligand_file, logger)
        
        if smiles:
            all_smiles_with_names.append((smiles, folder_name))
            processed_count += 1
            logger.debug(f"  ✅ {folder_name}: Converted to SMILES")
        else:
            logger.warning(f"  ⚠️  {folder_name}: Failed to convert {ligand_filename} to SMILES")
            conversion_failed_count += 1
    
    # 3. 保存所有 SMILES 到一个文件（格式: SMILES name）
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\nSaving {len(all_smiles_with_names)} SMILES to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for smiles, folder_name in all_smiles_with_names:
            f.write(f"{smiles} {folder_name}\n")
    
    # 4. 总结
    logger.info("\n" + "=" * 60)
    logger.info("Collection Summary")
    logger.info("=" * 60)
    logger.info(f"✅ Successfully processed: {processed_count} folders")
    logger.info(f"✅ Total SMILES saved: {len(all_smiles_with_names)}")
    logger.info(f"⚠️  Ligand file not found: {not_found_count} folders")
    logger.info(f"⚠️  Conversion failed: {conversion_failed_count} folders")
    logger.info(f"📁 Output file: {output_file}")
    logger.info(f"📊 Total folders processed: {len(folder_list)}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Collect ligand.sdf files from folders listed in validation_folder.txt and convert to SMILES"
    )
    parser.add_argument("--folder_list", type=str, required=True,
                       help="Path to file containing list of folder paths (e.g., validation_folder.txt)")
    parser.add_argument("--output_file", type=str, required=True,
                       help="Output .smi file path (e.g., outputs/validation_smiles/all_ligand_smiles.smi)")
    parser.add_argument("--ligand_filename", type=str, default="ligand.sdf",
                       help="Name of ligand file to look for in each folder (default: ligand.sdf)")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging(args.log_level)
    
    logger.info("=" * 60)
    logger.info("Collect Ligand SMILES from Validation Folders")
    logger.info("=" * 60)
    logger.info(f"Folder list: {args.folder_list}")
    logger.info(f"Output file: {args.output_file}")
    logger.info(f"Ligand filename: {args.ligand_filename}")
    
    # Check files exist
    if not Path(args.folder_list).exists():
        logger.error(f"Folder list file not found: {args.folder_list}")
        return
    
    if not RDKIT_AVAILABLE:
        logger.error("RDKit is not available. Please install RDKit to use this script.")
        logger.error("Install with: conda install -c conda-forge rdkit")
        return
    
    # Collect SMILES
    collect_ligand_smiles(
        folder_list_path=args.folder_list,
        output_file=args.output_file,
        logger=logger,
        ligand_filename=args.ligand_filename,
    )
    
    logger.info("\n✅ Collection completed!")


if __name__ == "__main__":
    main()

