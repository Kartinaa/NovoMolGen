#!/usr/bin/env python3
"""
评估 SMILES 文件中分子的环系统频率。

用法:
    python scripts/validation_scripts/evaluate_chembl_ring.py \
        --input_smi path/to/molecules.smi \
        --output_csv path/to/output.csv (可选)
"""

import argparse
import logging
import sys
from pathlib import Path
from collections import Counter
import itertools

import pandas as pd
from rdkit import Chem
from tqdm import tqdm

try:
    import useful_rdkit_utils as uru
except ImportError:
    print("Error: useful_rdkit_utils not found. Please install it.")
    sys.exit(1)


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging."""
    logger = logging.getLogger("evaluate_ring")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def load_smiles_from_file(smi_path: Path, logger: logging.Logger) -> list:
    """
    从 .smi 文件中加载 SMILES。
    
    Args:
        smi_path: SMILES 文件路径
        logger: Logger 实例
    
    Returns:
        SMILES 字符串列表
    """
    logger.info(f"Loading SMILES from: {smi_path}")
    smi_list = []
    
    try:
        with open(smi_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    # .smi 文件格式可能是 "SMILES name" 或只有 "SMILES"
                    parts = line.split()
                    if parts:
                        smiles = parts[0]
                        smi_list.append(smiles)
        
        logger.info(f"Loaded {len(smi_list)} SMILES from file")
        return smi_list
        
    except Exception as e:
        logger.error(f"Error reading SMILES file: {e}")
        raise


def analyze_ring_systems(
    smi_list: list,
    logger: logging.Logger,
    min_frequency_threshold: int = 100,
) -> dict:
    """
    分析分子的环系统频率。
    
    Args:
        smi_list: SMILES 字符串列表
        logger: Logger 实例
        min_frequency_threshold: 最小频率阈值（默认: 100）
    
    Returns:
        包含分析结果的字典
    """
    logger.info(f"Analyzing {len(smi_list)} molecules...")
    
    # 创建 DataFrame
    df = pd.DataFrame(smi_list, columns=['SMILES'])
    
    # 转换为 RDKit 分子对象
    logger.info("Converting SMILES to RDKit molecules...")
    df['mol'] = df.SMILES.apply(Chem.MolFromSmiles)
    
    # 检查无效分子
    invalid_count = df['mol'].isna().sum()
    if invalid_count > 0:
        logger.warning(f"Found {invalid_count} invalid molecules (will be excluded)")
        df = df.dropna(subset=['mol'])
    
    valid_count = len(df)
    logger.info(f"Valid molecules: {valid_count}")
    
    if valid_count == 0:
        logger.error("No valid molecules found. Exiting.")
        return {}
    
    # 查找环系统
    logger.info("Finding ring systems...")
    ring_system_finder = uru.RingSystemFinder()
    df['ring_systems'] = df.mol.apply(ring_system_finder.find_ring_systems)
    
    # 统计环系统频率
    logger.info("Counting ring system frequencies...")
    ring_system_list = list(itertools.chain.from_iterable(df.ring_systems.values))
    ring_count_df = pd.DataFrame(
        Counter(ring_system_list).items(),
        columns=["SMILES", "Count"]
    )
    ring_count_df.sort_values("Count", ascending=False, inplace=True)
    
    logger.info(f"Found {len(ring_count_df)} unique ring systems")
    
    # 查找最小环频率
    # RingSystemLookup 是一个 dataclass，需要使用 default() 类方法创建实例
    # default() 会从内置的数据库文件加载环系统频率信息
    logger.info("Looking up ring system frequencies in database...")
    try:
        ring_system_lookup = uru.RingSystemLookup.default()
        logger.info("Successfully loaded ring system database")
    except Exception as e:
        logger.error(f"Failed to load ring system database: {e}")
        logger.error("RingSystemLookup.default() requires the ring system database file.")
        logger.error("Please ensure useful_rdkit_utils is properly installed with data files.")
        raise
    
    res = df.mol.apply(ring_system_lookup.process_mol)
    df[['min_ring', 'min_freq']] = res.apply(uru.get_min_ring_frequency).tolist()
    
    # 去重（基于 InChI）
    logger.info("Removing duplicates based on InChI...")
    original_count = len(df)
    df['inchi'] = df.mol.apply(Chem.MolToInchi)
    df = df.drop_duplicates("inchi", keep="first", ignore_index=True)
    unique_count = len(df)
    duplicates_removed = original_count - unique_count
    
    if duplicates_removed > 0:
        logger.info(f"Removed {duplicates_removed} duplicate molecules")
    
    # 筛选有环系统的分子
    df_ring = df[df.min_freq != -1]
    filtered_df_ring_freq = df[(df.min_freq >= min_frequency_threshold) & (df.min_freq != -1)]
    
    # 计算统计信息
    total_molecules = len(df)
    molecules_with_rings = len(df_ring)
    molecules_with_common_rings = len(filtered_df_ring_freq)
    
    percentage_with_rings = (molecules_with_rings / total_molecules * 100) if total_molecules > 0 else 0
    percentage_with_common_rings = (molecules_with_common_rings / total_molecules * 100) if total_molecules > 0 else 0
    
    results = {
        'total_molecules': total_molecules,
        'valid_molecules': valid_count,
        'unique_molecules': unique_count,
        'molecules_with_rings': molecules_with_rings,
        'molecules_with_common_rings': molecules_with_common_rings,
        'percentage_with_rings': percentage_with_rings,
        'percentage_with_common_rings': percentage_with_common_rings,
        'ring_count_df': ring_count_df,
        'df': df,
        'df_ring': df_ring,
        'filtered_df_ring_freq': filtered_df_ring_freq,
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ring system frequencies in SMILES molecules"
    )
    parser.add_argument(
        "--input_smi",
        type=str,
        required=True,
        help="Path to input .smi file"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path to output CSV file (optional, saves detailed results)"
    )
    parser.add_argument(
        "--min_frequency",
        type=int,
        default=100,
        help="Minimum frequency threshold for common ring systems (default: 100)"
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
    logger.info("Ring System Frequency Analysis")
    logger.info("=" * 60)
    logger.info(f"Input file: {args.input_smi}")
    if args.output_csv:
        logger.info(f"Output CSV: {args.output_csv}")
    logger.info(f"Min frequency threshold: {args.min_frequency}")
    
    # Check input file exists
    input_path = Path(args.input_smi)
    if not input_path.exists():
        logger.error(f"Input file not found: {args.input_smi}")
        return 1
    
    # Load SMILES
    try:
        smi_list = load_smiles_from_file(input_path, logger)
    except Exception as e:
        logger.error(f"Failed to load SMILES: {e}")
        return 1
    
    if not smi_list:
        logger.error("No SMILES found in input file")
        return 1
    
    # Analyze ring systems
    try:
        results = analyze_ring_systems(
            smi_list,
            logger,
            min_frequency_threshold=args.min_frequency,
        )
    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        return 1
    
    if not results:
        logger.error("Analysis failed")
        return 1
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Analysis Summary")
    logger.info("=" * 60)
    logger.info(f"Total molecules: {results['total_molecules']}")
    logger.info(f"Valid molecules: {results['valid_molecules']}")
    logger.info(f"Unique molecules: {results['unique_molecules']}")
    logger.info(f"Molecules with ring systems: {results['molecules_with_rings']}")
    logger.info(f"Molecules with common ring systems (freq >= {args.min_frequency}): {results['molecules_with_common_rings']}")
    logger.info(f"Percentage molecules with ring systems: {results['percentage_with_rings']:.2f}%")
    logger.info(f"Percentage molecules with common ring systems (freq >= {args.min_frequency}): {results['percentage_with_common_rings']:.2f}%")
    logger.info("=" * 60)
    
    # Save detailed results if requested
    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save main results with ring frequency information
            results['df'].to_csv(output_path, index=False)
            logger.info(f"\n✅ Detailed results saved to: {output_path}")
            
            # Also save ring count statistics
            ring_stats_path = output_path.parent / f"{output_path.stem}_ring_stats.csv"
            results['ring_count_df'].to_csv(ring_stats_path, index=False)
            logger.info(f"✅ Ring system statistics saved to: {ring_stats_path}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return 1
    
    logger.info("\n✅ Analysis completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

