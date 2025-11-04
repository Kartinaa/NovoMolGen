# NovoMolGen 训练与评估系统详解

## 概述

NovoMolGen 是一个基于 Transformer 的分子生成模型，使用 LLaMA 架构和 Flash Attention 进行高效训练。本文档详细介绍了训练和评估的完整流程。

## 系统架构

### 核心组件
- **模型**: NovoMolGen (基于 LLaMA + Flash Attention)
- **训练器**: HFTrainer (HuggingFace Trainer 扩展)
- **评估器**: MoleculeEvaluator (分子属性评估)
- **回调系统**: Evaluator, WandbCallback

## 训练流程详解

### 1. 预训练 (Pre-training)

#### 1.1 数据准备
```bash
# 步骤1: 数据预处理和分词
python src/main.py tokenize_dataset \
    --config_name=ZINC_1B_safe_bpe \
    --config_dir=configs/dataset
```

**数据流程**:
1. **加载原始数据集**: ZINC-1B 分子数据集
2. **分子表示转换**: SMILES/SELFIES/SAFE/DeepSMILES
3. **分词处理**: BPE 或 Atomwise 分词
4. **缓存保存**: 预处理后的数据保存到磁盘

#### 1.2 模型初始化
```python
# 模型配置
model = NovoMolGen(
    config=NovoMolGenConfig(
        model_type="llama",
        hidden_size=768,
        num_hidden_layers=32,
        num_attention_heads=12,
        use_flash_attn=True,
        # ... 其他配置
    ),
    mol_type="SAFE"  # 分子表示类型
)
```

**模型特点**:
- **位置编码**: RoPE (Rotary Position Embedding)
- **注意力机制**: Flash Attention (内存高效)
- **架构**: 因果语言模型 (Causal Language Model)

#### 1.3 训练循环
```python
# 训练配置
trainer = HFTrainer(
    model=model,
    args=HFTrainingArguments(
        per_device_train_batch_size=300,      # 单GPU批次大小
        gradient_accumulation_steps=16,       # 梯度累积
        learning_rate=6e-4,                   # 学习率
        max_steps=77380,                      # 最大训练步数
        evaluation_strategy="steps",          # 评估策略
        eval_steps=1000,                      # 每1000步评估
        save_steps=1000,                      # 每1000步保存
        # ... 其他配置
    ),
    callbacks=[Evaluator(), WandbCallback()], # 回调函数
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
```

**训练步骤**:
1. **前向传播**: 计算模型输出和损失
2. **反向传播**: 计算梯度
3. **梯度累积**: 累积多个批次的梯度
4. **参数更新**: 使用 AdamW 优化器更新参数
5. **学习率调度**: 余弦退火学习率调度

### 2. 微调 (Fine-tuning)

#### 2.1 监督微调 (SFT)
```python
class SFTTrainer(PolicyTrainer):
    def _compute_loss(self, logits, labels):
        # 负对数似然损失
        sequence_log_probs = self.logprobs_from_logits(logits, labels)
        loss = -(sequence_log_probs.mean())
        return loss, {}
```

#### 2.2 强化学习微调 (REINVENT)
```python
class REINVENTTrainer(PolicyTrainer):
    def _compute_loss(self, model, agent_likelihood, prior_likelihood, reward, generated_smiles):
        # REINVENT 损失函数
        augmented_likelihood = prior_likelihood + self.config.sigma * reward
        loss = ((prior_likelihood - agent_likelihood + augmented_likelihood) ** 2).mean()
        return loss, metrics
```

#### 2.3 增强爬山算法 (AugmentedHC)
```python
class AugmentedHCTrainer(REINVENTTrainer):
    def _compute_loss(self, model, agent_likelihood, prior_likelihood, reward, generated_smiles):
        # 选择 top-k 样本进行训练
        top_k_indices = torch.topk(reward, k=int(len(reward) * self.config.fraction_selected)).indices
        # 计算增强损失
        loss = self._compute_augmented_loss(agent_likelihood[top_k_indices], 
                                          prior_likelihood[top_k_indices], 
                                          reward[top_k_indices])
        return loss, metrics
```

## 评估系统详解

### 1. 验证损失 (Validation Loss)

**计算方式**:
```python
def compute_loss(self, model, inputs):
    # 使用标签平滑的负对数似然损失
    loss = self.label_smoother(
        logits=outputs['logits'], 
        labels=labels,
        vocab_size=self.model.base_config.vocab_size
    )
    return loss
```

**作用**:
- 监控模型在验证集上的语言建模性能
- 检测过拟合
- 用于早停机制

### 2. 分子生成评估

#### 2.1 基础分子属性
```python
PROPERTY_DISTRIBUTION_TASKS = {
    "logP": logP,                    # 脂水分配系数
    "SA": SA,                        # 合成可达性 (Synthetic Accessibility)
    "QED": QED,                      # 药物相似性 (Quantitative Estimate of Drug-likeness)
    "weight": weight,                # 分子量
    "NP": NP,                        # 天然产物相似性 (Natural Product-likeness)
    "NumRings": get_n_rings,         # 环数量
    "Bertz": Bertz,                  # 分子复杂度
    "TPSA": TPSA,                    # 拓扑极性表面积
    "AliphaticRings": NumAliphaticRings,    # 脂肪环数量
    "AromaticRings": NumAromaticRings,      # 芳香环数量
    "RotatableBonds": NumRotatableBonds,    # 可旋转键数量
}
```

#### 2.2 生成质量指标
```python
# 默认评估任务
task_names = [
    "unique@1k",      # 唯一性: 生成分子的唯一比例
    "IntDiv",         # 内部多样性: 生成分子之间的多样性
    "filters",        # 过滤器通过率: 通过分子过滤器的比例
    "SA_mean",        # 平均合成可达性
    "logP_mean",      # 平均脂水分配系数
    "QED_mean",       # 平均药物相似性
]
```

#### 2.3 高级评估任务
```python
# GuacaMol 基准测试
GUACAMOL_TASKS = [
    "Albuterol_Similarity", "DRD2", "GSK3B", "JNK3",  # 分子相似性和生物活性
    "Celecoxib_Rediscovery", "Sitagliptin_MPO",        # 分子重发现
    "Scaffold Hop", "Deco Hop",                        # 骨架跳跃
    # ... 更多任务
]

# 片段和骨架评估
FRAGMENT_TASKS = {
    "FCD": FCDMetric,      # Fréchet ChemNet Distance
    "SNN": SNNMetric,      # 最近邻相似性
    "Frag": FragMetric,    # 片段相似性
    "Scaf": ScafMetric,    # 骨架相似性
}
```

### 3. 评估集成机制

#### 3.1 回调系统
```python
class Evaluator(TrainerCallback):
    def on_evaluate(self, args, state, control, model=None, tokenizer=None, **kwargs):
        # 生成分子样本
        generated_smiles = model.generate_smiles(
            tokenizer,
            n_samples=3000,              # 生成3000个样本
            num_return_sequences=1000,   # 返回1000个序列
            temperature=1.0,
            top_k=50,
            top_p=0.95,
            max_length=64,
        )
        
        # 评估生成的分子
        result = self.evaluator(gen_smiles=generated_smiles, filter=True)
        
        # 记录评估结果
        state.evaluation_task_results = {f"eval/{k}": v for k, v in result.items()}
```

#### 3.2 评估流程
1. **每1000步**触发评估
2. **生成3000个分子样本**
3. **计算各种分子属性指标**
4. **记录到训练日志和WandB**

## 配置系统

### 1. 训练配置
```yaml
# configs/train_ZINC_1B_bpe_safe_llama-300M.yaml
defaults:
  - dataset: ZINC_1B_safe_bpe_streaming
  - model: llama-300M
  - trainer: hf_trainer_one_epoch
  - _self_

trainer:
  per_device_train_batch_size: 300
  per_device_eval_batch_size: 300
  gradient_accumulation_steps: 16
  evaluation_strategy: "steps"
  eval_steps: 1000
  save_steps: 1000
```

### 2. 数据集配置
```yaml
# configs/dataset/ZINC_1B_safe_bpe_streaming.yaml
dataset_name: "chandar-lab/ZINC_22"
tokenizer_path: "./data/tokenizers/tokenizer_bpe_None_SAFE_500_0_0.1.json"
mol_type: "SAFE"
max_seq_length: 64
num_proc: 8
streaming: False  # 注意: 分词时需要设为False
validation_set_names: ["MolGen/zinc_scaffold", "MolGen/zinc_random"]
```

### 3. 模型配置
```yaml
# configs/model/llama-300M.yaml
model_type: "llama"
hidden_size: 768
num_hidden_layers: 32
num_attention_heads: 12
intermediate_size: 3072
use_flash_attn: true
fused_bias_fc: false
fused_mlp: false
fused_dropout_add_ln: false
residual_in_fp32: true
```

## 使用指南

### 1. 环境设置
```bash
# 安装依赖
pip install -r requirements.txt

# 设置环境变量
export HF_TOKEN="your_huggingface_token"
export WANDB_ENTITY="your_wandb_entity"
export WANDB_PROJECT="your_project_name"
export WANDB_TAGS="molecular_generation"
```

### 2. 完整训练流程
```bash
# 步骤1: 数据预处理
python src/main.py tokenize_dataset \
    --config_name=ZINC_1B_safe_bpe \
    --config_dir=configs/dataset

# 步骤2: 预训练
python src/main.py train \
    --config_name=train_ZINC_1B_bpe_safe_llama-300M \
    --config_dir=configs

# 步骤3: 微调 (可选)
python src/main.py finetune \
    --config_name=finetune_PMO_ZINC_1B_atomwise_smiles_llama-32M \
    --config_dir=configs
```

### 3. 单GPU训练配置
对于单A100 (80GB) 训练，推荐配置:
```yaml
trainer:
  per_device_train_batch_size: 300
  per_device_eval_batch_size: 300
  gradient_accumulation_steps: 16
  gradient_checkpointing: true  # 启用梯度检查点节省内存
```

## 监控和日志

### 1. WandB集成
- **训练指标**: 损失、学习率、梯度范数
- **评估指标**: 分子属性、生成质量
- **模型检查点**: 自动上传到WandB

### 2. 评估指标解释
- **unique@1k**: 生成分子的唯一性比例
- **IntDiv**: 生成分子之间的内部多样性
- **filters**: 通过分子过滤器的比例
- **SA_mean**: 平均合成可达性 (越低越好)
- **logP_mean**: 平均脂水分配系数
- **QED_mean**: 平均药物相似性 (0-1, 越高越好)

## 性能优化

### 1. 内存优化
- **梯度检查点**: 减少激活值内存占用
- **混合精度训练**: 使用bf16减少内存
- **梯度累积**: 模拟大批次训练

### 2. 计算优化
- **Flash Attention**: 高效注意力计算
- **TF32**: 在A100上启用TF32加速
- **数据并行**: 多GPU训练支持

## 故障排除

### 1. 常见问题
- **OOM错误**: 减少批次大小或启用梯度检查点
- **分词错误**: 确保streaming=False用于分词
- **评估失败**: 检查依赖包安装

### 2. 调试技巧
- 使用较小的模型进行测试
- 检查数据加载是否正确
- 监控GPU内存使用情况

## 总结

NovoMolGen 提供了一个完整的分子生成模型训练和评估系统，包括:

1. **灵活的配置系统**: 支持多种分子表示和模型架构
2. **全面的评估体系**: 从基础属性到高级生物活性评估
3. **高效的训练流程**: 支持预训练和多种微调策略
4. **完善的监控系统**: 集成WandB进行实验跟踪

该系统特别适合药物发现和分子设计任务，提供了从数据预处理到模型部署的完整解决方案。
