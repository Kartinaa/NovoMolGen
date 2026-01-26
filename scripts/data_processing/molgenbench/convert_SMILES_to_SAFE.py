#!/usr/bin/env python3
"""
简化版本：将 Breakpoint_SMILES 转换为 SAFE
模仿 test.py 的导入方式
"""

# 使用与 test.py 完全相同的导入顺序和方式
from safe import SAFEConverter
import safe as sf
import pandas as pd

# 初始化encoder（与test.py相同）
encoder = SAFEConverter(ignore_stereo=True)

# 文件路径
input_file = "finetune_data/molgenbench_dataset/h2l_scaffold/h2l_breakpoint_scaffolds_Version1.csv"
output_file = "finetune_data/molgenbench_dataset/h2l_scaffold/h2l_breakpoint_scaffolds_Version1_with_safe.csv"

print("="*60)
print("转换 Breakpoint SMILES 到 SAFE")
print("="*60)

# 读取CSV
print(f"\n读取: {input_file}")
df = pd.read_csv(input_file)
print(f"✓ 加载 {len(df)} 行")
print(f"列名: {list(df.columns)}")

# 检查列
if 'Breakpoint_SMILES' not in df.columns:
    print(f"\n❌ 错误: 'Breakpoint_SMILES' 列不存在!")
    exit(1)

# 转换函数（与test.py相同）
def convert_to_safe(smiles, verbose=False):
    # 确保输入是字符串
    if pd.isna(smiles):
        return ""
    
    # 转换为字符串以防万一
    smiles = str(smiles).strip()
    if not smiles or smiles == "" or smiles.lower() == "nan":
        return ""
    
    try:
        # 使用与 test.py 相同的方法
        with sf.utils.attr_as(encoder, 'slicer', None):
            encoded = encoder.encoder(smiles, allow_empty=True)
        if not encoded or encoded == "":
            return ""
        return encoded + '.'  # 添加 '.'
    except Exception as e:
        # 只在verbose模式下显示详细错误
        if verbose:
            print(f"  警告: 转换失败 '{smiles[:50]}...'")
        return ""

# 转换所有SMILES
print(f"\n转换 {len(df)} 个SMILES...")
safe_strings = []
failed_indices = []  # 记录失败的行索引
failed_smiles = []  # 记录失败的SMILES

for i, smiles in enumerate(df['Breakpoint_SMILES']):
    safe_str = convert_to_safe(smiles, verbose=False)  # 静默模式
    # 确保safe_str是字符串
    if not isinstance(safe_str, str):
        safe_str = ""
    safe_strings.append(safe_str)
    
    # 记录失败的行
    if safe_str == "":
        failed_indices.append(i)
        if not pd.isna(smiles):
            failed_smiles.append(str(smiles)[:50])
    
    # 只显示成功的转换（每10个或前3个）
    if (i + 1) % 10 == 0 or i < 3:
        if safe_str:  # 只显示成功的
            # 安全地显示SMILES和SAFE
            smiles_display = "N/A"
            if not pd.isna(smiles):
                smiles_str = str(smiles)
                smiles_display = smiles_str[:50] if len(smiles_str) > 50 else smiles_str
            
            safe_display = safe_str[:50] if len(safe_str) > 50 else safe_str
            print(f"  ✓ {i+1}/{len(df)}: {smiles_display} -> {safe_display}")

# 添加新列
df['Breakpoint_SAFE'] = safe_strings

# 统计（删除前）
original_count = len(df)
successful = sum(1 for s in safe_strings if s != "")
failed = len(failed_indices)

print(f"\n转换统计:")
print(f"  总数:   {original_count}")
print(f"  成功:   {successful} ({successful/original_count*100:.1f}%)")
print(f"  失败:   {failed} ({failed/original_count*100:.1f}%)")

# 显示前几个失败的示例
if failed_smiles:
    print(f"\n失败的SMILES示例 (前{min(5, len(failed_smiles))}个):")
    for idx, smiles in enumerate(failed_smiles[:5], 1):
        print(f"  {idx}. {smiles}...")

# 删除转换失败的行
if failed_indices:
    print(f"\n🗑️  正在删除 {failed} 个转换失败的行...")
    df_filtered = df[df['Breakpoint_SAFE'] != ""].copy()
    print(f"✓ 保留有效数据: {len(df_filtered)} 行 (删除了 {failed} 行)")
else:
    df_filtered = df.copy()
    print(f"\n✓ 所有行转换成功，无需删除")

# 保存
print(f"\n保存到: {output_file}")
df_filtered.to_csv(output_file, index=False)
print("✓ 完成!")

print("\n" + "="*60)
print("转换完成！")
print("="*60)

