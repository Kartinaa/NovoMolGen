#!/usr/bin/env python3
"""
计算 lead_optimization_results.txt 文件中分子的常见环系统百分比。

Usage:
    python scripts/data_processing/molgenbench/calculate_ring_system_from_txt.py \
        --input outputs/.../lead_optimization_results.txt \
        --output outputs/.../ring_system_stats.yaml
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import yaml

# 添加 MolGenBench 的路径（如果不在同一个 repo）
# 这里假设 ring_system_calculator 的逻辑可以直接使用
try:
    from rdkit import Chem
    from rdkit.Chem import inchi
    import useful_rdkit_utils as uru
except ImportError:
    print("Error: 需要安装 rdkit 和 useful_rdkit_utils")
    print("请确保在正确的环境中运行，或安装依赖：")
    print("  pip install rdkit")
    print("  pip install useful-rdkit-utils")
    sys.exit(1)


ring_system_lookup = uru.RingSystemLookup()


def extract_smiles_from_txt(txt_path: str) -> list:
    """
    从 lead_optimization_results.txt 文件中提取所有 SMILES。
    跳过以 # 开头的注释行。
    """
    smiles_list = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # 第一个非空白部分应该是 SMILES
            parts = line.split()
            if parts:
                smiles_list.append(parts[0])
    return smiles_list


def compute_ring_stats_from_smiles(smiles_list: list) -> dict:
    """
    计算环系统统计信息。
    
    Returns:
        dict with statistics
    """
    # 转换为 DataFrame
    df = pd.DataFrame({"SMILES": smiles_list})
    
    # 转换为 RDKit 分子对象
    df["mol"] = df["SMILES"].apply(lambda s: Chem.MolFromSmiles(s) if s else None)
    
    # 丢弃无效分子
    df = df[df["mol"].notnull()].reset_index(drop=True)
    
    if len(df) == 0:
        return {
            "n_total": 0,
            "n_unique": 0,
            "n_has_ring": 0,
            "n_common_ring": 0,
            "frac_unique": None,
            "frac_has_ring": None,
            "frac_common_ring_among_has_ring": None,
            "frac_common_ring_among_all": None,
        }
    
    n_total = len(df)
    
    # 计算 InChIKey 去重
    df["inchi_key"] = df["mol"].apply(inchi.MolToInchiKey)
    df_unique = df.drop_duplicates("inchi_key", keep="first").reset_index(drop=True)
    n_unique = len(df_unique)
    
    # 计算 ring system 频率
    res = df_unique["mol"].apply(ring_system_lookup.process_mol)
    df_unique[["min_ring", "min_freq"]] = res.apply(uru.get_min_ring_frequency).tolist()
    
    # 统计
    mask_has_ring = df_unique["min_freq"] != -1
    mask_common_ring = (df_unique["min_freq"] > 100) & mask_has_ring
    
    n_has_ring = int(mask_has_ring.sum())
    n_common_ring = int(mask_common_ring.sum())
    
    stats = {
        "n_total": int(n_total),
        "n_unique": int(n_unique),
        "frac_unique": float(n_unique / n_total) if n_total > 0 else None,
        "n_has_ring": n_has_ring,
        "frac_has_ring": float(n_has_ring / n_unique) if n_unique > 0 else None,
        "n_common_ring": n_common_ring,
        "frac_common_ring_among_has_ring": float(n_common_ring / n_has_ring) if n_has_ring > 0 else None,
        "frac_common_ring_among_all": float(n_common_ring / n_unique) if n_unique > 0 else None,
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="计算 lead_optimization_results.txt 文件中分子的常见环系统百分比"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入的 lead_optimization_results.txt 文件路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出的 YAML 文件路径（可选，默认保存在输入文件同目录）",
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 输入文件不存在: {input_path}")
        sys.exit(1)
    
    print(f"读取文件: {input_path}")
    print("提取 SMILES...")
    smiles_list = extract_smiles_from_txt(str(input_path))
    print(f"  提取到 {len(smiles_list)} 个 SMILES")
    
    print("计算环系统统计...")
    stats = compute_ring_stats_from_smiles(smiles_list)
    
    # 打印结果
    print("\n" + "=" * 80)
    print("环系统统计结果")
    print("=" * 80)
    print(f"总分子数（含重复，去除无效分子后）: {stats['n_total']}")
    print(f"按 InChIKey 去重后的分子数: {stats['n_unique']}  ({stats['frac_unique']*100:.2f}% of total)")
    print(f"具有已知环系频率 (min_freq != -1) 的分子数: {stats['n_has_ring']}  ({stats['frac_has_ring']*100:.2f}% of unique)")
    print(f"具有常见环系 ((min_freq > 100) & (min_freq != -1)) 的分子数: {stats['n_common_ring']}")
    print(f"  占所有去重分子的百分比: {stats['frac_common_ring_among_all']*100:.2f}%")
    print(f"  占具有环系分子的百分比: {stats['frac_common_ring_among_has_ring']*100:.2f}%")
    print("=" * 80)
    
    # 保存结果
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / "ring_system_stats.yaml"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(stats, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    
    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()

