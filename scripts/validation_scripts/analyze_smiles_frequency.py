#!/usr/bin/env python3
"""
从parquet文件中提取standardize_smi列，并计算top-N分子的频率统计。

用法示例：
  python scripts/validation_scripts/analyze_smiles_frequency.py \
    --input_file finetune_data/crossdock_dataset/base_dataset.parquet \
    --top_n 100
"""

import argparse
from pathlib import Path
from collections import Counter

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    try:
        import pyarrow.parquet as pq
        HAS_PYARROW = True
    except ImportError:
        HAS_PYARROW = False


def analyze_smiles_frequency(
    input_file: str,
    top_n: int = 100,
    smiles_column: str = "standardize_smi",
):
    """分析SMILES的频率分布。
    
    Args:
        input_file: 输入的parquet文件路径
        top_n: 要统计的top-N分子数量
        smiles_column: SMILES列的名称
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"文件不存在: {input_path}")
    
    print("=" * 60)
    print("SMILES频率分析")
    print("=" * 60)
    print(f"输入文件: {input_path}")
    print(f"SMILES列名: {smiles_column}")
    print(f"Top-N: {top_n}")
    print()
    
    # 读取parquet文件
    print("正在读取parquet文件...")
    if HAS_PANDAS:
        df = pd.read_parquet(input_path)
        print(f"✓ 成功读取，总行数: {len(df):,}")
        print(f"  列名: {list(df.columns)}")
        print()
        
        # 检查列是否存在
        if smiles_column not in df.columns:
            raise ValueError(
                f"列 '{smiles_column}' 不存在。可用列: {list(df.columns)}"
            )
        
        # 提取SMILES列
        print(f"正在提取 '{smiles_column}' 列...")
        smiles_series = df[smiles_column]
    elif HAS_PYARROW:
        table = pq.read_table(input_path)
        print(f"✓ 成功读取，总行数: {len(table):,}")
        print(f"  列名: {list(table.column_names)}")
        print()
        
        # 检查列是否存在
        if smiles_column not in table.column_names:
            raise ValueError(
                f"列 '{smiles_column}' 不存在。可用列: {list(table.column_names)}"
            )
        
        # 提取SMILES列
        print(f"正在提取 '{smiles_column}' 列...")
        smiles_series = table[smiles_column].to_pylist()
    else:
        raise ImportError(
            "需要安装 pandas 或 pyarrow 来读取parquet文件。\n"
            "请运行: pip install pandas 或 pip install pyarrow"
        )
    
    # 统计非空值
    if HAS_PANDAS:
        non_null_count = smiles_series.notna().sum()
        null_count = smiles_series.isna().sum()
        print(f"✓ 非空SMILES数量: {non_null_count:,}")
        if null_count > 0:
            print(f"  空值数量: {null_count:,}")
        print()
        
        # 过滤空值
        valid_smiles = smiles_series.dropna().tolist()
    else:
        # pyarrow方式
        valid_smiles = [s for s in smiles_series if s is not None]
        non_null_count = len(valid_smiles)
        null_count = len(smiles_series) - non_null_count
        print(f"✓ 非空SMILES数量: {non_null_count:,}")
        if null_count > 0:
            print(f"  空值数量: {null_count:,}")
        print()
    
    # 统计频率
    print("正在统计SMILES频率...")
    smiles_counter = Counter(valid_smiles)
    total_samples = len(valid_smiles)
    unique_smiles = len(smiles_counter)
    
    print(f"✓ 统计完成")
    print(f"  总样本数: {total_samples:,}")
    print(f"  唯一SMILES数: {unique_smiles:,}")
    print(f"  平均每个SMILES出现次数: {total_samples / unique_smiles:.2f}")
    print()
    
    # 获取top-N
    print(f"正在计算Top-{top_n}最常见的分子...")
    top_smiles = smiles_counter.most_common(top_n)
    
    # 计算top-N的累计样本数
    top_n_count = sum(count for _, count in top_smiles)
    top_n_percentage = (top_n_count / total_samples) * 100
    
    print("=" * 60)
    print(f"Top-{top_n} 分子统计结果")
    print("=" * 60)
    print(f"Top-{top_n} 分子的总出现次数: {top_n_count:,}")
    print(f"Top-{top_n} 分子占全部样本的百分比: {top_n_percentage:.2f}%")
    print()
    
    # 显示前10个最常见的分子
    print("前10个最常见的分子:")
    print("-" * 60)
    print(f"{'排名':<6} {'出现次数':<12} {'占比':<10} {'SMILES'}")
    print("-" * 60)
    for rank, (smiles, count) in enumerate(top_smiles[:10], 1):
        percentage = (count / total_samples) * 100
        # 如果SMILES太长，截断显示
        display_smiles = smiles if len(smiles) <= 50 else smiles[:47] + "..."
        print(f"{rank:<6} {count:<12,} {percentage:>6.2f}%  {display_smiles}")
    print()
    
    # 显示top-N的分布信息
    if top_n > 10:
        print(f"Top-{top_n} 的分布统计:")
        print("-" * 60)
        top_counts = [count for _, count in top_smiles]
        print(f"  最大出现次数: {max(top_counts):,}")
        print(f"  最小出现次数: {min(top_counts):,}")
        print(f"  平均出现次数: {sum(top_counts) / len(top_counts):.2f}")
        print(f"  中位数出现次数: {sorted(top_counts)[len(top_counts) // 2]:,}")
        print()
    
    # 保存结果到文件
    output_file = input_path.parent / f"top_{top_n}_smiles_frequency.txt"
    print(f"正在保存结果到: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"Top-{top_n} SMILES频率统计\n")
        f.write("=" * 60 + "\n")
        f.write(f"输入文件: {input_path}\n")
        f.write(f"总样本数: {total_samples:,}\n")
        f.write(f"唯一SMILES数: {unique_smiles:,}\n")
        f.write(f"Top-{top_n} 分子总出现次数: {top_n_count:,}\n")
        f.write(f"Top-{top_n} 分子占比: {top_n_percentage:.2f}%\n")
        f.write("\n" + "-" * 60 + "\n")
        f.write(f"{'排名':<6} {'出现次数':<12} {'占比':<10} {'SMILES'}\n")
        f.write("-" * 60 + "\n")
        for rank, (smiles, count) in enumerate(top_smiles, 1):
            percentage = (count / total_samples) * 100
            f.write(f"{rank:<6} {count:<12,} {percentage:>6.2f}%  {smiles}\n")
    
    print(f"✓ 结果已保存到: {output_file}")
    print()
    
    return {
        "total_samples": total_samples,
        "unique_smiles": unique_smiles,
        "top_n_count": top_n_count,
        "top_n_percentage": top_n_percentage,
        "top_smiles": top_smiles,
    }


def main():
    parser = argparse.ArgumentParser(
        description="分析parquet文件中SMILES的频率分布"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="输入的parquet文件路径",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=100,
        help="要统计的top-N分子数量 (默认: 100)",
    )
    parser.add_argument(
        "--smiles_column",
        type=str,
        default="standardize_smi",
        help="SMILES列的名称 (默认: standardize_smi)",
    )
    
    args = parser.parse_args()
    
    try:
        analyze_smiles_frequency(
            input_file=args.input_file,
            top_n=args.top_n,
            smiles_column=args.smiles_column,
        )
        return 0
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

