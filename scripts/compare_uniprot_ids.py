#!/usr/bin/env python3
"""
比较两个文件中的 UniProt ID，计算交集。

用法:
    python scripts/compare_uniprot_ids.py
"""

import csv
from pathlib import Path
from collections import Counter

# 文件路径
# 尝试多个可能的 CSV 文件
possible_csv_files = [
    Path("finetune_data/Index_fo_BindingNetv2.csv"),
]
txt_file = Path("finetune_data/molgenbench_dataset/MolGenBench_UniprotIDS.txt")

print("=" * 60)
print("UniProt ID 比较分析")
print("=" * 60)

# 1. 读取 CSV 文件
csv_uniprot_ids = set()
csv_file_used = None

for csv_file in possible_csv_files:
    if not csv_file.exists():
        continue
    
    print(f"\n1. 检查 CSV 文件: {csv_file}")
    
    # 检查 CSV 文件的所有列
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        print(f"   文件的列: {columns}")
        
        # 检查是否有 UniProt 列（不区分大小写）
        uniprot_col = None
        for col in columns:
            if 'uniprot' in col.lower():
                uniprot_col = col
                print(f"   ✓ 找到 UniProt 列: '{uniprot_col}'")
                break
        
        if uniprot_col:
            # 如果有 UniProt 列，读取所有 UniProt ID
            print(f"   正在读取 UniProt ID...")
            row_count = 0
            for row in reader:
                row_count += 1
                uniprot_id = row.get(uniprot_col, '').strip()
                if uniprot_id:
                    csv_uniprot_ids.add(uniprot_id)
            
            print(f"   总行数: {row_count:,}")
            print(f"   找到的唯一 UniProt ID 数量: {len(csv_uniprot_ids):,}")
            csv_file_used = csv_file
            break
        else:
            print(f"   ✗ 未找到 UniProt 列，跳过此文件")

if not csv_uniprot_ids:
    print(f"\n   警告: 在所有检查的 CSV 文件中都没有找到 UniProt 列")
    print(f"   检查的文件: {[str(f) for f in possible_csv_files]}")
else:
    print(f"\n   使用的 CSV 文件: {csv_file_used}")

# 2. 读取文本文件
print(f"\n2. 读取文本文件: {txt_file}")
txt_uniprot_ids = set()

if txt_file.exists():
    with open(txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            uniprot_id = line.strip()
            if uniprot_id:
                txt_uniprot_ids.add(uniprot_id)
    
    print(f"   文本文件中的 UniProt ID 数量: {len(txt_uniprot_ids)}")
    print(f"   前 10 个 ID: {list(txt_uniprot_ids)[:10]}")
else:
    print(f"   错误: 文件不存在: {txt_file}")

# 3. 计算交集
print(f"\n3. 计算结果:")
if csv_uniprot_ids and txt_uniprot_ids:
    intersection = csv_uniprot_ids & txt_uniprot_ids
    csv_only = csv_uniprot_ids - txt_uniprot_ids
    txt_only = txt_uniprot_ids - csv_uniprot_ids
    
    print(f"   CSV 文件中的 UniProt ID 总数: {len(csv_uniprot_ids)}")
    print(f"   文本文件中的 UniProt ID 总数: {len(txt_uniprot_ids)}")
    print(f"   交集（两个文件都有的）: {len(intersection)}")
    print(f"   仅在 CSV 文件中: {len(csv_only)}")
    print(f"   仅在文本文件中: {len(txt_only)}")
    
    if intersection:
        print(f"\n   交集中的前 20 个 ID:")
        for i, uid in enumerate(sorted(intersection)[:20], 1):
            print(f"     {i}. {uid}")
    
    if txt_only:
        print(f"\n   仅在文本文件中的前 20 个 ID:")
        for i, uid in enumerate(sorted(txt_only)[:20], 1):
            print(f"     {i}. {uid}")
    
    # 计算覆盖率
    if len(txt_uniprot_ids) > 0:
        coverage = (len(intersection) / len(txt_uniprot_ids)) * 100
        print(f"\n   覆盖率: {coverage:.2f}% ({len(intersection)}/{len(txt_uniprot_ids)})")
        print(f"   即：文本文件中有 {coverage:.2f}% 的 UniProt ID 也出现在 CSV 文件中")
else:
    if not csv_uniprot_ids:
        print("   警告: CSV 文件中没有找到 UniProt ID")
    if not txt_uniprot_ids:
        print("   警告: 文本文件中没有找到 UniProt ID")

print("\n" + "=" * 60)

