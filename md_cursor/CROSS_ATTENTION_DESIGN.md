# Cross-Attention Design Documentation

## 1. 总览 (Overview)

### 模块间关系图

```
NovoMolGen (继承 GPTLMHeadModel)
    ↓ 持有
transformer (GPTModel from flash-attn)
    ↓ 被包装为
CrossAttentionTransformer
    ↓ 在指定层注入
CrossAttentionAdapter (纯 Cross-Attn 模块)
```

### 训练/推理数据流

**训练流程**:
```
input_ids[B,T] + cond_tokens[B,Lc,d] 
    ↓
NovoMolGen.forward()
    ↓
CrossAttentionTransformer.forward()
    ↓ (在指定层)
CrossAttentionAdapter.forward()
    ↓
lm_logits[B,T,vocab_size] + loss
```

**推理流程 (generate())**:
```
input_ids[B,T] + cond_tokens[B,Lc,d]
    ↓
prepare_inputs_for_generation() (透传条件)
    ↓
model.generate() → 逐token生成
    ↓ (每步都应用跨注意力)
CrossAttentionAdapter.forward()
    ↓
generated_sequences[B,T_out]
```

## 2. 数据形状 (Shapes)

### 核心维度定义
- **B**: 批大小 (batch size)
- **T**: 主序列长度 (main sequence length) 
- **Lc**: 条件长度 (condition length)
- **d**: 隐藏维度 (hidden dimension)
- **num_heads**: 注意力头数
- **vocab_size**: 词汇表大小

### 输入张量形状

| 张量 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `input_ids` | `[B, T]` | `torch.LongTensor` | 输入token IDs |
| `hidden_states` | `[B, T, d]` | `torch.FloatTensor` | 隐藏状态表示 |
| `cond_tokens` | `[B, Lc, d]` | `torch.FloatTensor` | 条件token嵌入 |
| `cond_attention_mask` | `[B, Lc]` | `torch.BoolTensor` | 条件mask (True=padding) |

### 输出张量形状

| 张量 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `attn_output` | `[B, T, d]` | `torch.FloatTensor` | 跨注意力输出 |
| `attn_weights` | `[B, num_heads, T, Lc]` | `torch.FloatTensor` | 注意力权重 (可选) |
| `lm_logits` | `[B, T, vocab_size]` | `torch.FloatTensor` | 语言模型logits |
| `loss` | `[]` | `torch.FloatTensor` | 标量损失值 |

### 特殊情况
- **空条件序列**: `Lc=0` 时返回零输出 `[B, T, d]`
- **注意力权重**: 仅在 `return_attention_weights=True` 时返回

## 3. 模块逻辑

### 3.1 CrossAttentionAdapter (纯 Cross-Attn 版)

#### 设计原则
- **纯跨注意力**: 仅执行 MultiheadAttention，不含残差连接与LayerNorm
- **统一处理**: 便于上层统一管理残差和归一化
- **灵活输出**: 支持返回注意力权重用于分析

#### 核心逻辑
```python
def forward(hidden_states[B,T,d], cond_tokens[B,Lc,d], cond_attention_mask[B,Lc], return_attention_weights=False):
    # 1. 形状验证
    validate_shapes(hidden_states, cond_tokens, cond_attention_mask)
    
    # 2. 空条件处理
    if Lc == 0:
        return zeros_like(hidden_states)
    
    # 3. 跨注意力计算
    attn_output, attn_weights = self.cross_attn(
        query=hidden_states,           # [B, T, d]
        key=cond_tokens,               # [B, Lc, d] 
        value=cond_tokens,             # [B, Lc, d]
        key_padding_mask=cond_attention_mask,  # [B, Lc] (True=padding)
        batch_first=True
    )
    
    # 4. Dropout
    attn_output = self.dropout(attn_output)
    
    # 5. 返回
    return attn_output if not return_attention_weights else (attn_output, attn_weights)
```

#### 关键特性
- **Mask语义**: `cond_attention_mask=True` 表示padding token，会被忽略
- **MHA配置**: `batch_first=True` 确保张量格式一致
- **边界处理**: 自动处理空条件序列，避免MHA报错

### 3.2 CrossAttentionTransformer (注入器)

#### 配置控制
```python
config = NovoMolGenConfig(
    enable_cross_attn=True,        # 启用跨注意力
    cross_layers=[3, 6, 9]         # 在层3,6,9注入
)
```

#### 猴子补丁机制
```python
def _patch_transformer_blocks():
    for layer_idx in self.cross_layers:
        # 1. 保存原始forward方法
        original_forward = block.forward
        
        # 2. 创建补丁forward (修复闭包晚绑定)
        def make_patched_forward(original_forward, adapter, gate, 
                                block_bound=block, self_bound=self):
            def patched_forward(hidden_states, *args, **kwargs):
                # 执行原始层
                original_output = original_forward(hidden_states, *args, **kwargs)
                
                # 处理tuple返回
                if isinstance(original_output, tuple):
                    output, rest = original_output[0], original_output[1:]
                else:
                    output = original_output
                    rest = ()
                
                # 跨注意力注入
                if cond_tokens_available:
                    # LayerNorm预处理
                    normed_output = block_bound.norm1(output)
                    
                    # 纯跨注意力
                    attn_out = adapter(normed_output, cond_tokens, cond_mask)
                    
                    # 统一门控残差
                    gate_value = torch.sigmoid(gate)
                    output = output + gate_value * self_bound.cross_dropout(attn_out)
                
                # 返回原始格式
                return (output,) + rest if rest else output
            return patched_forward
        
        # 3. 应用补丁
        block.forward = make_patched_forward(...)
```

#### 闭包晚绑定修复
- **问题**: Python闭包会捕获循环变量的最终值
- **解决**: 使用默认参数 `block_bound=block, self_bound=self` 提前绑定
- **效果**: 确保每个层使用正确的block和self引用

#### 门控机制
```python
# 每层独立门控参数
self.gates = nn.ParameterDict({
    str(layer_idx): nn.Parameter(torch.zeros(1))  # 初始化为0
    for layer_idx in self.cross_layers
})

# 门控残差公式
output = output + sigmoid(gate[layer_idx]) * dropout(attn_out)
```

### 3.3 NovoMolGen

#### 配置映射
```python
# LlamaConfig → GPT2Config 转换
config = llama_config_to_gpt2_config(config)
config.use_flash_attn = self.base_config.use_flash_attn
config.enable_cross_attn = self.base_config.enable_cross_attn
config.cross_layers = self.base_config.cross_layers
```

#### Forward流程
```python
def forward(input_ids[B,T], cond_tokens[B,Lc,d]=None, cond_attention_mask[B,Lc]=None, ...):
    # 1. 输入验证
    assert input_ids.ndim == 2
    
    # 2. 通过transformer (可能包含跨注意力)
    hidden_states = self.transformer(
        input_ids, 
        cond_tokens=cond_tokens, 
        cond_attention_mask=cond_attention_mask
    )
    
    # 3. 输出投影
    lm_logits = self.lm_head(hidden_states)  # [B, T, vocab_size]
    
    # 4. 损失计算 (如果提供labels)
    loss = self.loss_function(lm_logits, labels) if labels is not None else None
    
    return CausalLMOutput(loss=loss, logits=lm_logits, hidden_states=hidden_states)
```

#### 生成透传
```python
def prepare_inputs_for_generation(input_ids, **kwargs):
    model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    
    # 透传跨注意力条件
    if "cond_tokens" in kwargs:
        model_inputs["cond_tokens"] = kwargs["cond_tokens"]
    if "cond_attention_mask" in kwargs:
        model_inputs["cond_attention_mask"] = kwargs["cond_attention_mask"]
    
    return model_inputs
```

#### 条件采样示例
```python
# 条件生成示例
def conditional_generation(model, tokenizer, cond_tokens, max_length=64):
    # 1. 准备输入
    input_ids = tokenizer.encode("", return_tensors="pt")  # [1, 1] (BOS)
    
    # 2. 条件生成
    generated = model.generate(
        input_ids=input_ids,
        cond_tokens=cond_tokens,           # [1, Lc, d]
        cond_attention_mask=cond_mask,     # [1, Lc] (可选)
        max_length=max_length,
        temperature=1.0,
        do_sample=True
    )
    
    # 3. 解码
    molecules = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return molecules
```

## 4. 训练与推理注意事项

### 4.1 计算复杂度
- **时间复杂度**: O(T·Lc) 每层跨注意力
- **空间复杂度**: O(T·Lc·num_heads) 注意力权重存储
- **多层注入**: 显存和耗时随注入层数线性增长

### 4.2 门控初始化策略
```python
# 默认策略: 初始化为0 (sigmoid(0) = 0.5)
self.gates[str(layer_idx)] = nn.Parameter(torch.zeros(1))

# 自定义策略示例
nn.init.normal_(self.gates[str(layer_idx)], mean=0.0, std=0.1)
```

### 4.3 注意力Mask使用
- **训练时**: 使用`cond_attention_mask`处理变长条件序列
- **推理时**: 通常为`None`，所有条件token都有效
- **Flash-Attn兼容**: 当前实现忽略主序列的attention_mask

### 4.4 最佳实践

#### 层选择建议
```python
# 12层模型推荐配置
cross_layers = [3, 6, 9]    # 早期、中期、后期注入

# 24层模型推荐配置  
cross_layers = [4, 8, 12, 16]  # 更密集的注入点

# 32层模型推荐配置
cross_layers = [5, 10, 15, 20, 25]  # 均匀分布
```

#### 条件序列处理
```python
# 条件序列预处理
def prepare_conditions(conditions_list, max_length=32):
    # 1. 截断或填充到固定长度
    padded_conditions = pad_sequences(conditions_list, max_length)
    
    # 2. 创建attention mask
    attention_mask = create_padding_mask(conditions_list, max_length)
    
    return padded_conditions, attention_mask
```

#### 内存优化
```python
# 梯度检查点 (如果显存不足)
model.gradient_checkpointing_enable()

# 混合精度训练
from torch.cuda.amp import autocast
with autocast():
    outputs = model(input_ids, cond_tokens=cond_tokens)
```

### 4.5 调试与监控

#### 门控值监控
```python
# 训练过程中监控门控值
for layer_idx in model.cross_layers:
    gate_value = torch.sigmoid(model.transformer.gates[str(layer_idx)]).item()
    print(f"Layer {layer_idx} gate: {gate_value:.4f}")
```

#### 注意力权重可视化
```python
# 获取注意力权重
outputs, attn_weights = model.transformer.cross_adapters["3"](
    hidden_states, cond_tokens, return_attention_weights=True
)

# 可视化 (需要额外工具)
visualize_attention(attn_weights[0, 0])  # 第一个样本，第一个头
```

---

## 总结

本设计文档详细说明了跨注意力注入机制的实现原理、数据流和最佳实践。通过模块化的设计，实现了灵活的跨注意力注入，支持条件生成和训练，同时保持了与现有flash-attn架构的兼容性。

关键优势：
- ✅ **模块化设计**: 纯跨注意力模块，便于测试和调试
- ✅ **灵活注入**: 可配置的层选择和门控机制  
- ✅ **生成支持**: 完整的条件生成流程
- ✅ **性能优化**: 避免双重残差和LayerNorm
- ✅ **健壮性**: 完善的边界情况处理

NovoMolGen.forward(input_ids, cond_tokens, cond_attention_mask)
    ↓
self.transformer(...)  # 这里调用的是 CrossAttentionTransformer.forward()
    ↓
CrossAttentionTransformer.forward():
    # 1. 存储条件到实例变量
    self._current_cond_tokens = cond_tokens
    self._current_cond_attention_mask = cond_attention_mask
    
    # 2. 调用原始transformer
    self.base_transformer(input_ids, ...)  # 原始的flash-attn transformer
        ↓
    # 3. 在指定层，猴子补丁的forward被调用
    patched_block.forward(hidden_states, ...):
        # 执行原始block逻辑
        output = original_forward(hidden_states, ...)
        
        # 检查是否有条件
        if self._current_cond_tokens is not None:
            # 应用跨注意力
            normed_output = block.norm1(output)
            attn_out = adapter(normed_output, self._current_cond_tokens, ...)
            output = output + sigmoid(gate) * dropout(attn_out)
        
        return output
    ↓
# 4. 清除条件
self._current_cond_tokens = None
self._current_cond_attention_mask = None