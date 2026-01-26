# Recovery Rate Analysis

分析生成分子与FilteredMols（参考分子）的重合率。

## 功能

对比生成的分子（从JSON）和参考分子集合FilteredMols（从CSV），计算：
- 每组的recovery rate（恢复了多少百分比的参考分子）
- 总体recovery rate
- 平均recovery rate
- 统计分布

## 快速使用

### 方法1：使用便捷脚本

编辑 `scripts/data_processing/molgenbench/run_analyze_recovery.sh`，设置路径：

```bash
JSON_FILE="path/to/lead_optimization_results.json"
CSV_FILE="path/to/csv_with_FilteredMols.csv"
```

运行：

```bash
bash scripts/data_processing/molgenbench/run_analyze_recovery.sh
```

### 方法2：直接运行Python脚本

```bash
python scripts/data_processing/molgenbench/analyze_recovery_rate.py \
    --json_file outputs/.../lead_optimization_results.json \
    --csv_file finetune_data/.../h2l_single_breakpoint_scaffolds_Version1_min20_with_safe.csv \
    --output_file recovery_analysis.csv
```

## 输出

### 1. 终端输出摘要

```
================================================================================
RECOVERY RATE ANALYSIS SUMMARY
================================================================================
Total entries analyzed: 69
Entries with at least 1 match: 1 (1.4%)

Total FilteredMols (reference): 3374
Total recovered: 2
Overall recovery rate: 0.06%

Average recovery rate per entry: 0.06%
================================================================================

Recovery Rate Distribution:
  Min:    0.00%
  25th:   0.00%
  Median: 0.00%
  75th:   0.00%
  Max:    4.44%

Recovery Rate Distribution by Bins:
          0%:   68 entries ( 98.6%)
       1-10%:    1 entries (  1.4%)
      11-25%:    0 entries (  0.0%)
      26-50%:    0 entries (  0.0%)
      51-75%:    0 entries (  0.0%)
      76-99%:    0 entries (  0.0%)
        100%:    0 entries (  0.0%)
```

### 2. 详细CSV文件

保存在JSON文件所在目录，名为 `recovery_rate_analysis.csv`

包含每个entry的详细信息：
- `UniProt_ID`: UniProt ID
- `SeriseID`: Series ID
- `Num_FilteredMols`: 参考分子数量
- `Num_Generated`: 生成分子数量（去重后）
- `Num_Recovered`: 恢复的分子数量
- `Recovery_Rate_%`: 恢复率百分比
- `Recovered_Molecules`: 恢复的具体分子（逗号分隔）

## 参数说明

- `--json_file`: 生成结果的JSON文件（必需）
- `--csv_file`: 包含FilteredMols列的CSV文件（必需）
- `--output_file`: 输出的详细结果CSV（可选，默认保存在JSON文件目录）
- `--quiet`: 不显示进度信息

## 工作原理

1. **读取参考分子**：从CSV的`FilteredMols`列读取参考SMILES（逗号分隔）
2. **标准化**：使用RDKit将所有SMILES标准化为canonical SMILES
3. **对比**：对每个UniProt_ID/SeriseID组合：
   - 找到对应的FilteredMols
   - 找到生成的分子
   - 计算交集（重合的分子）
4. **统计**：计算recovery rate = (重合数量 / 参考分子数量) × 100%

## 解读结果

### Recovery Rate的含义

- **高recovery rate (>50%)**：模型成功恢复了大部分已知活性分子
  - 优点：验证了模型的可靠性
  - 缺点：可能缺乏新颖性

- **中等recovery rate (10-50%)**：部分恢复，同时有新分子
  - 理想的平衡点

- **低recovery rate (<10%)**：生成的分子大多是新颖的
  - 优点：高度新颖，可能发现新的化学空间
  - 缺点：需要验证这些新分子的活性

### 示例解读

从上面的结果：
- Overall recovery rate: 0.06%
- 68/69 entries: 0% recovery

**解读**：
1. ✅ **高度新颖性**：生成的分子几乎都是新的，不是简单复制已知分子
2. ⚠️ **需要验证**：这些新分子的活性需要通过对接或实验验证
3. 🎯 **单个成功案例**：P34913/Sries30256 恢复了2个分子（4.4%），说明模型在某些情况下能够重现已知结构

## 提高Recovery Rate的方法

如果需要更高的recovery rate（例如用于验证模型）：

1. **调整生成参数**：
   - 降低temperature（更确定性的生成）
   - 增加num_samples（生成更多样本）
   - 使用beam search而不是sampling

2. **使用条件生成**：
   - 确保条件向量（pocket_vec, evo_vec等）正确匹配

3. **检查scaffold**：
   - 确认Breakpoint_SAFE正确对应FilteredMols

## 常见问题

### Q: Recovery rate很低是否意味着模型不好？

A: 不一定。低recovery rate可能说明：
- 模型在探索新的化学空间（好事）
- 或者生成的分子偏离了活性区域（需要进一步验证）

关键是检查生成分子的质量（通过对接分数、药物性质等）。

### Q: 为什么某些entry有0个生成分子？

A: 可能原因：
- 生成失败（全部转换失败）
- SMILES去重后为空
- 检查对应的JSON entry的`num_generated`字段

### Q: 如何找出恢复了哪些具体分子？

A: 查看输出CSV的`Recovered_Molecules`列，包含所有恢复的SMILES。

## 相关脚本

- `analyze_recovery_rate.py` - 主分析脚本
- `run_analyze_recovery.sh` - 便捷运行脚本
- `organize_generated_molecules.py` - 组织生成的分子到文件夹
- `lead_optimization_from_csv.py` - 生成分子的脚本

