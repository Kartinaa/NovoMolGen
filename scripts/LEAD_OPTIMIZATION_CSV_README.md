# Lead Optimization from CSV - 使用说明

## 概述

`lead_optimization_from_csv.py` 脚本用于从CSV文件中读取breakpoint SAFE字符串，并为每个breakpoint生成分子。生成时使用validation数据集中对应的条件向量（pocket_vec, evo_vec, ifp, ligand_vec）。

## 主要功能

1. **从CSV读取breakpoint数据**
   - CSV文件必须包含以下列：`UniProt_ID`, `SeriseID`, `Breakpoint_SAFE`
   
2. **匹配条件向量**
   - 通过从validation数据集的`ligand_path`中提取UniProt_ID
   - 将CSV中的UniProt_ID与validation数据集中的条件向量匹配
   
3. **生成分子**
   - 对每个breakpoint SAFE字符串生成指定数量的SMILES分子
   - 使用对应的蛋白质条件向量进行条件生成
   
4. **保存结果**
   - JSON格式：完整的生成结果和统计信息
   - CSV格式：UniProt_ID, SeriseID, Breakpoint_SAFE, Generated_SMILES
   - TXT格式：按SeriseID分组的SMILES列表

## 使用方法

### 方法1：使用便捷脚本（推荐）

```bash
cd /home/yang2531/Documents/Project/Structure_safe

# 使用默认参数
./scripts/run_lead_opt_csv.sh

# 或自定义参数
./scripts/run_lead_opt_csv.sh \
    --model_path outputs/your_model/checkpoint-xxx/full_model \
    --num_samples_per_breakpoint 100 \
    --output_dir outputs/my_lead_opt
```

### 方法2：直接运行Python脚本

```bash
cd /home/yang2531/Documents/Project/Structure_safe

# 需要设置环境变量
export LD_LIBRARY_PATH=$HOME/anaconda3/envs/NovoMol/lib:$LD_LIBRARY_PATH

conda run -n NovoMol python scripts/lead_optimization_from_csv.py \
    --model_path outputs/11_21_25_SAFEGen_lr_KLnormclaped0.1_unfreezeall_noloar_xattn67891011_newmodel/checkpoint-33400/full_model \
    --validation_set finetune_data/molgenbench_dataset/validation \
    --csv_file finetune_data/molgenbench_dataset/h2l_scaffold/h2l_breakpoint_scaffolds_with_safe.csv \
    --num_samples_per_breakpoint 50 \
    --output_dir outputs/lead_optimization_h2l_scaffold \
    --batch_size 10 \
    --max_length 64 \
    --log_level INFO
```

## 参数说明

### 必需参数

- `--model_path`: 模型路径（full_model目录）
- `--validation_set`: validation数据集路径（包含条件向量）
- `--csv_file`: breakpoint CSV文件路径
- `--output_dir`: 输出目录

### 可选参数

- `--num_samples_per_breakpoint`: 每个breakpoint生成的分子数量（默认：50）
- `--batch_size`: 批次大小（默认：10）
- `--max_length`: 最大生成长度（默认：64）
- `--temperature`: 采样温度（默认：1.0）
- `--top_k`: Top-k采样（默认：50）
- `--top_p`: Top-p采样（默认：0.95）
- `--sample_posterior`: 是否从posterior采样ligand latent
- `--append_mode`: 是否追加模式（默认：True）
- `--ifp_ablation`: IFP消融模式 [none/zero/shuffle]
- `--cond_ablation`: 全局条件消融模式 [none/zero/shuffle]
- `--log_level`: 日志级别 [DEBUG/INFO/WARNING/ERROR]
- `--seed`: 随机种子（默认：2025）

## CSV文件格式

输入CSV文件必须包含以下列：

```csv
UniProt_ID,SeriseID,Breakpoint_SAFE
O14757,Sries14139,C14CC2OCCCCCOC3NC(CNC3C#N)NC(=O)NC2CC1Cl.
O14757,Sries27479,C14CCC2C5CC3NNC(=O)N3C2C1.
```

**注意**：
- `Breakpoint_SAFE`末尾的"."会自动被移除
- 只有在validation数据集中存在对应UniProt_ID的breakpoint才会被处理

## 输出文件

脚本会在输出目录中生成以下文件：

1. **lead_optimization_results.json**
   - 完整的生成结果
   - 每个breakpoint的统计信息
   - 生成参数

2. **lead_optimization_results.csv**
   - 表格格式的结果
   - 列：UniProt_ID, SeriseID, Breakpoint_SAFE, Generated_SMILES
   - 每行一个生成的分子

3. **lead_optimization_results.txt**
   - 文本格式的SMILES列表
   - 按SeriseID分组

## 示例输出

```csv
UniProt_ID,SeriseID,Breakpoint_SAFE,Generated_SMILES
O14757,Sries14139,C14CC2OCCCCCOC3NC(CNC3C#N)NC(=O)NC2CC1Cl,C1CC2OCCCCCOC3NC(CNC3C#N)NC(=O)NC2CC1Cl
O14757,Sries14139,C14CC2OCCCCCOC3NC(CNC3C#N)NC(=O)NC2CC1Cl,C1CC2OCCCCCOC3NC(CNC3C#N)NC(=O)NC2CC1F
```

## 技术细节

### UniProt_ID提取

从validation数据集的`ligand_path`中提取UniProt_ID：

```
路径格式：/path/to/O14757/O14757_lig.sdf
提取结果：O14757
```

支持的UniProt_ID格式：
- 6个字符
- 以大写字母开头
- 第二个字符可以是数字或大写字母
- 剩余字符为字母数字

### Breakpoint_SAFE预处理

- 自动移除末尾的"."字符
- 使用SAFE库的decode功能验证格式

### 条件向量匹配

- 通过UniProt_ID将CSV中的breakpoint与validation数据集匹配
- 每个breakpoint使用对应蛋白质的条件向量进行生成

## 常见问题

### Q1: 为什么有些breakpoint被跳过？

A: 如果CSV中的UniProt_ID在validation数据集中不存在，该breakpoint会被跳过。检查日志中的警告信息。

### Q2: 生成失败率很高怎么办？

A: 
- 检查Breakpoint_SAFE格式是否正确
- 增加`--max_retries`参数
- 调整温度参数`--temperature`

### Q3: 如何加速生成？

A:
- 增加`--batch_size`（需要足够的GPU内存）
- 减少`--max_length`
- 使用更小的`--num_samples_per_breakpoint`

### Q4: 如何只生成部分breakpoint？

A: 创建一个包含所需breakpoint子集的新CSV文件。

## 测试

运行小规模测试：

```bash
# 创建测试CSV（前3个breakpoint）
head -4 finetune_data/molgenbench_dataset/h2l_scaffold/h2l_breakpoint_scaffolds_with_safe.csv > /tmp/test_breakpoints.csv

# 运行测试
./scripts/run_lead_opt_csv.sh \
    --csv_file /tmp/test_breakpoints.csv \
    --num_samples_per_breakpoint 3 \
    --output_dir /tmp/test_lead_opt \
    --batch_size 3
```

## 相关脚本

- `lead_optimization_from_full_model_xdock.py`: 原始的lead optimization脚本
- `run_lead_opt_csv.sh`: 便捷运行脚本

## 更新日志

- 2026-01-13: 初始版本
  - 支持从CSV文件读取breakpoint
  - 自动匹配UniProt_ID到条件向量
  - 生成JSON/CSV/TXT三种格式的输出

