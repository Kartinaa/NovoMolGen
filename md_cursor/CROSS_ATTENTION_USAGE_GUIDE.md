# Cross-Attention Usage Guide

## 🎯 **Overview**

This guide shows how to use the cross-attention functionality in NovoMolGen for conditional molecular generation.

## 🧩 **Components**

### 1. **CrossAttentionAdapter**
Pure cross-attention module that performs MultiheadAttention between hidden states and condition tokens.

### 2. **CrossAttentionTransformer** 
Wrapper that injects cross-attention into specified transformer layers using monkey patching.

### 3. **NovoMolGen**
Full model with cross-attention support for conditional generation.

## 🚀 **Quick Start**

### **Basic Usage**

```python
import torch
from src.models.modeling_novomolgen import NovoMolGen, NovoMolGenConfig

# 1. Create configuration
config = NovoMolGenConfig(
    hidden_size=256,
    num_attention_heads=8,
    num_hidden_layers=6,
    vocab_size=1000,
    enable_cross_attn=True,
    cross_layers=[3, 6, 9],  # Inject at layers 3, 6, 9
    use_flash_attn=False,    # Set to True for GPU
    fused_bias_fc=False,
    fused_mlp=False,
    fused_dropout_add_ln=False
)

# 2. Create model
model = NovoMolGen(config, mol_type="SMILES")

# 3. Create test data
batch_size = 2
seq_len = 10
cond_len = 5

input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
cond_tokens = torch.randn(batch_size, cond_len, config.hidden_size)
cond_mask = torch.zeros(batch_size, cond_len, dtype=torch.bool)

# 4. Forward pass with conditions
outputs = model(
    input_ids=input_ids,
    cond_tokens=cond_tokens,
    cond_attention_mask=cond_mask,
    return_dict=True
)

print(f"Output logits: {outputs.logits.shape}")
print(f"Loss: {outputs.loss.item()}")
```

### **Generation with Conditions**

```python
# Generate molecules with conditions
generated = model.generate(
    input_ids=input_ids[:, :3],  # Use first 3 tokens as prompt
    cond_tokens=cond_tokens,
    cond_attention_mask=cond_mask,
    max_new_tokens=10,
    do_sample=True,
    temperature=0.8,
    pad_token_id=0
)

print(f"Generated shape: {generated.shape}")
print(f"Generated tokens: {generated[0].tolist()}")
```

## 🔧 **Component Testing**

### **Test CrossAttentionAdapter Only**

```python
from src.models.modeling_novomolgen import CrossAttentionAdapter, NovoMolGenConfig

# Create adapter
config = NovoMolGenConfig(hidden_size=128, num_attention_heads=8)
adapter = CrossAttentionAdapter(config)

# Test data
hidden_states = torch.randn(2, 10, 128)  # [batch, seq_len, hidden_size]
cond_tokens = torch.randn(2, 5, 128)     # [batch, cond_len, hidden_size]
cond_mask = torch.zeros(2, 5, dtype=torch.bool)  # [batch, cond_len]

# Forward pass
output = adapter(hidden_states, cond_tokens, cond_mask)
print(f"Output shape: {output.shape}")

# With attention weights
output, weights = adapter(hidden_states, cond_tokens, cond_mask, 
                         return_attention_weights=True)
print(f"Attention weights shape: {weights.shape}")
```

## 📊 **Data Shapes**

| Component | Input Shape | Output Shape | Description |
|-----------|-------------|--------------|-------------|
| `hidden_states` | `[B, T, d]` | - | Main sequence hidden states |
| `cond_tokens` | `[B, Lc, d]` | - | Condition token embeddings |
| `cond_mask` | `[B, Lc]` | - | Condition padding mask (True=padding) |
| `CrossAttentionAdapter` | `[B, T, d]` | `[B, T, d]` | Cross-attention output |
| `attn_weights` | - | `[B, H, T, Lc]` | Attention weights (optional) |

**Legend:**
- `B`: Batch size
- `T`: Main sequence length  
- `Lc`: Condition length
- `d`: Hidden dimension
- `H`: Number of attention heads

## ⚙️ **Configuration Options**

### **NovoMolGenConfig Parameters**

```python
config = NovoMolGenConfig(
    # Model architecture
    hidden_size=256,                    # Hidden dimension
    num_attention_heads=8,              # Number of attention heads
    num_hidden_layers=6,                # Number of transformer layers
    vocab_size=1000,                    # Vocabulary size
    max_position_embeddings=512,        # Maximum sequence length
    
    # Cross-attention settings
    enable_cross_attn=True,             # Enable cross-attention
    cross_layers=[3, 6, 9],             # Layers to inject cross-attention
    attn_dropout=0.1,                   # Attention dropout rate
    
    # Performance settings (for CPU testing)
    use_flash_attn=False,               # Use FlashAttention (requires GPU)
    fused_bias_fc=False,                # Fused bias operations
    fused_mlp=False,                    # Fused MLP operations
    fused_dropout_add_ln=False,         # Fused dropout + layer norm
    residual_in_fp32=False              # Residual in float32
)
```

## 🎛️ **Gate Mechanism**

The cross-attention uses learnable gates to control the influence of cross-attention:

```python
# Gate values (initialized to -1.0)
gate = -1.0
gate_weight = torch.sigmoid(gate)  # ≈ 0.27 (27% influence)

# Gated residual connection
output = hidden_states + gate_weight * cross_attention_output
```

### **Gate Initialization Values**

| Initial Value | Sigmoid Value | Influence | Use Case |
|---------------|---------------|-----------|----------|
| `0.0` | `0.50` | 50% | Balanced start |
| `-1.0` | `0.27` | 27% | **Recommended** - Conservative |
| `-2.0` | `0.12` | 12% | Very conservative |
| `-3.0` | `0.05` | 5% | Extremely conservative |

## 🧪 **Testing**

### **Run Simple Test**

```bash
cd /home/yang2531/Documents/Project/NovoMolGen
source /home/yang2531/anaconda3/bin/activate NovoMol
python test_cross_attn_simple.py
```

### **Run Full Test (requires GPU)**

```bash
python test_cross_attention_usage.py
```

## 🚨 **Common Issues**

### **1. Triton/GPU Errors**
```
ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)
```
**Solution**: Set `use_flash_attn=False` for CPU testing.

### **2. Shape Mismatch**
```
RuntimeError: The size of tensor a (128) must match the size of tensor b (256)
```
**Solution**: Ensure `hidden_size` is consistent between model and condition tokens.

### **3. Empty Conditions**
```python
# Handle empty conditions gracefully
if cond_tokens.shape[1] == 0:
    # CrossAttentionAdapter returns zeros automatically
    pass
```

## 💡 **Best Practices**

### **1. Conservative Initialization**
```python
# Use -1.0 for gate initialization (recommended)
config.gate_init_value = -1.0
```

### **2. Layer Selection**
```python
# Choose middle layers for cross-attention injection
cross_layers = [num_layers // 3, 2 * num_layers // 3, num_layers - 1]
```

### **3. Condition Preparation**
```python
# Ensure condition tokens have same hidden_size as model
cond_tokens = torch.randn(batch_size, cond_len, model.config.hidden_size)
```

### **4. Mask Handling**
```python
# Use boolean masks (True = padding)
cond_mask = torch.zeros(batch_size, cond_len, dtype=torch.bool)
cond_mask[0, -2:] = True  # Mark last 2 tokens as padding
```

## 🔄 **Training Example**

```python
import torch.optim as optim

# Create model and optimizer
model = NovoMolGen(config, mol_type="SMILES")
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for batch in dataloader:
    input_ids = batch['input_ids']
    cond_tokens = batch['cond_tokens']
    cond_mask = batch['cond_mask']
    labels = batch['labels']
    
    # Forward pass
    outputs = model(
        input_ids=input_ids,
        cond_tokens=cond_tokens,
        cond_attention_mask=cond_mask,
        labels=labels
    )
    
    # Backward pass
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    # Monitor gate values
    for name, gate in model.transformer.gates.items():
        gate_value = torch.sigmoid(gate).item()
        print(f"Gate {name}: {gate_value:.4f}")
```

## 📈 **Monitoring**

### **Check Gate Values**
```python
# Monitor how gates evolve during training
for name, gate in model.transformer.gates.items():
    gate_value = torch.sigmoid(gate).item()
    print(f"Layer {name} gate: {gate_value:.4f} ({gate_value*100:.1f}% influence)")
```

### **Check Attention Weights**
```python
# Get attention weights for analysis
output, weights = model.transformer.cross_adapters['3'](
    hidden_states, cond_tokens, cond_mask, return_attention_weights=True
)
print(f"Attention weights shape: {weights.shape}")
print(f"Attention pattern: {weights[0, 0, :, :].detach().numpy()}")
```

This guide provides everything you need to use the cross-attention functionality effectively!
