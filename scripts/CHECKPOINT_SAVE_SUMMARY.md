# Checkpoint 保存内容总结

## 保存策略

脚本现在会保存**所有内容**，包括：

### 1. 训练过程中的 Checkpoint（每个 checkpoint-XXXXX 目录）

保存内容：
- ✅ **LoRA 参数**：所有 LoRA 适配器的权重（A 和 B 矩阵）
- ✅ **Cross-attention 适配器**：完整的 cross-attention 模块（通过 `modules_to_save`）
- ✅ **Gate 参数**：Cross-attention 的 gate 权重（通过 `modules_to_save`）
- ✅ **Encoder 模块**：ligand_encoder, protein_encoder, condition_fusion, cond_proj（通过 `modules_to_save`）
- ✅ **训练状态**：optimizer, scheduler, trainer_state, rng_state
- ✅ **配置文件**：config.yaml（训练配置）
- ✅ **Tokenizer**：tokenizer.json 和相关文件

**注意**：训练过程中的 checkpoint **不包括** pretrained model 的基础参数（因为它们是冻结的，不需要保存）。

### 2. 训练结束后的完整模型（`full_model/` 目录）

保存内容：
- ✅ **所有模型参数**：
  - Pretrained model 的基础参数（冻结的参数）
  - LoRA 权重（已合并到基础模型中）
  - Cross-attention 适配器
  - Gate 参数
  - Encoder 模块（ligand_encoder, protein_encoder, condition_fusion, cond_proj）
- ✅ **模型配置**：config.json
- ✅ **Tokenizer**：tokenizer.json 和相关文件
- ✅ **训练配置文件**：training_config.yaml（输入的配置文件）

**优势**：可以直接使用 `NovoMolGen.from_pretrained('full_model/')` 加载，无需先加载基础模型。

## 保存位置

```
output_dir/
├── checkpoint-700/          # 训练过程中的checkpoint
│   ├── adapter_model.bin   # PEFT格式：LoRA + cross-attention + encoders
│   ├── adapter_config.json
│   ├── config.yaml          # 训练配置
│   ├── tokenizer.json
│   ├── full_model/          # ✅ 完整模型（所有参数，用于推理）
│   │   ├── pytorch_model.bin    # 包含所有参数（pretrained + LoRA + cross-attention + encoders）
│   │   ├── config.json
│   │   ├── tokenizer.json
│   │   └── training_config.yaml
│   └── ...
├── checkpoint-1400/
│   ├── adapter_model.bin   # PEFT格式
│   ├── full_model/          # ✅ 完整模型
│   └── ...
├── final_model/             # 最终checkpoint（PEFT格式，用于继续训练）
│   └── ...
└── full_model/              # 训练结束后的完整模型
    ├── pytorch_model.bin    # 包含所有参数
    ├── config.json
    ├── tokenizer.json
    └── training_config.yaml # 输入的配置文件
```

**重要**：现在每个 checkpoint 都包含两种格式：
1. **PEFT格式**（checkpoint根目录）：用于继续训练
2. **完整模型格式**（checkpoint/full_model/）：用于推理，包含所有参数

## 确认所有参数都被保存

### 在训练过程中（checkpoint-XXXXX）

**PEFT格式**（checkpoint根目录）：
- 通过 `modules_to_save` 参数确保以下模块被完整保存：
  - `transformer.cross_adapters` - Cross-attention 适配器
  - `transformer.gates` - Gate 参数
  - `ligand_encoder` - Ligand encoder
  - `protein_encoder` - Protein encoder
  - `condition_fusion` - Condition fusion 模块
  - `cond_proj` - Condition projection 模块
- LoRA 参数会自动保存（PEFT 默认行为）

**完整模型格式**（checkpoint/full_model/）：
- ✅ 所有 pretrained model 参数（冻结的参数）
- ✅ 合并后的 LoRA 权重
- ✅ Cross-attention 适配器
- ✅ Gate 参数
- ✅ Encoder 模块
- ✅ 模型配置、tokenizer、训练配置文件

### 在训练结束后（full_model）

通过 `merge_and_unload()` 合并 LoRA 权重，然后保存完整模型，包括：
- 所有 pretrained model 的基础参数
- 合并后的 LoRA 权重
- Cross-attention 适配器
- Gate 参数
- Encoder 模块

## 使用方式

### 加载训练过程中的 checkpoint（继续训练）

使用 PEFT 格式（checkpoint根目录）：
```python
from peft import PeftModel
from src.models.modeling_novomolgen import NovoMolGen

# 加载基础模型
base_model = NovoMolGen.from_pretrained("pretrained_path")
# 加载 LoRA 适配器
model = PeftModel.from_pretrained(base_model, "checkpoint_path")
```

### 加载完整模型（推理）

**方式1**：使用 checkpoint 中的完整模型（推荐）
```python
from src.models.modeling_novomolgen import NovoMolGen

# 直接加载完整模型（从任意checkpoint）
model = NovoMolGen.from_pretrained("output_dir/checkpoint-700/full_model")
```

**方式2**：使用训练结束后的完整模型
```python
from src.models.modeling_novomolgen import NovoMolGen

# 直接加载完整模型
model = NovoMolGen.from_pretrained("output_dir/full_model")
```

**优势**：现在每个 checkpoint 都包含完整模型，可以随时用于推理，无需等待训练结束！

## 验证保存内容

可以使用 `scripts/check_checkpoint_contents.py` 来验证 checkpoint 中保存的内容：

```bash
python scripts/check_checkpoint_contents.py
```

