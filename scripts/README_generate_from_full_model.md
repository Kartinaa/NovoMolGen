# 使用训练好的模型生成分子

这个脚本可以从训练好的完整模型（full_model）checkpoint 中加载模型，并为每个 validation set 生成指定数量的分子。

## 使用方法

### 基本用法

```bash
python scripts/generate_from_full_model.py \
    --model_path outputs/11_10_25_SAFEGen/checkpoint-33400/full_model \
    --validation_sets finetune_data/processed_data/hf_bdnv2_dataset/validation \
    --num_samples_per_val 50 \
    --output_dir outputs/generated_molecules
```

### 参数说明

- `--model_path`: 训练好的完整模型目录路径（包含 `pytorch_model.bin`, `config.json`, `tokenizer.json` 等）
- `--validation_sets`: validation set 的路径（可以指定多个，用空格分隔）
- `--num_samples_per_val`: 每个 validation set 生成的分子数量（默认：50）
- `--output_dir`: 输出目录，生成的分子将保存在这里
- `--batch_size`: 生成时的批次大小（默认：10）
- `--max_length`: 最大生成长度（默认：64）
- `--temperature`: 采样温度（默认：1.0）
- `--top_k`: Top-k 采样参数（默认：50）
- `--top_p`: Top-p (nucleus) 采样参数（默认：0.95）
- `--log_level`: 日志级别（默认：INFO）

### 输出文件

对于每个 validation set，脚本会生成：

1. **`generated_{validation_set_name}.json`**: 包含生成分子的详细信息，包括：
   - 生成的分子列表
   - 条件特征信息（如果使用）
   - 生成参数

2. **`generated_{validation_set_name}.txt`**: 简单的文本文件，每行一个分子

3. **`generation_summary.json`**: 所有 validation sets 的生成摘要

### 示例

```bash
# 为单个 validation set 生成 50 个分子
python scripts/generate_from_full_model.py \
    --model_path outputs/11_10_25_SAFEGen/checkpoint-33400/full_model \
    --validation_sets finetune_data/processed_data/hf_bdnv2_dataset/validation \
    --num_samples_per_val 50 \
    --output_dir outputs/generated_molecules

# 为多个 validation sets 生成分子
python scripts/generate_from_full_model.py \
    --model_path outputs/11_10_25_SAFEGen/checkpoint-33400/full_model \
    --validation_sets \
        finetune_data/processed_data/hf_bdnv2_dataset/validation \
        finetune_data/processed_data/hf_bdnv2_dataset/test \
    --num_samples_per_val 100 \
    --output_dir outputs/generated_molecules \
    --temperature 0.8 \
    --top_p 0.9
```

### 条件特征支持

如果 validation set 包含条件特征（`pocket_vec`, `evo_vec`, `ifp`, `ligand_vec`），脚本会自动使用这些特征进行条件生成。如果没有条件特征，脚本会进行无条件生成。

### 注意事项

1. 确保模型路径包含完整的模型文件（`pytorch_model.bin`, `config.json`, `tokenizer.json` 等）
2. Validation set 应该是 HuggingFace datasets 格式（使用 `load_from_disk` 可以加载）
3. 如果使用条件特征，确保 validation set 包含所有必需的特征列
4. 生成过程会在 GPU 上运行（如果可用），否则使用 CPU

