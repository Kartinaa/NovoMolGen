# NovoMolGen Batch Structure Analysis

## 🎯 **Overview**

This document analyzes the batch structure in NovoMolGen's training pipeline, including the data collator, dataset, and model forward method.

## 📦 **Data Collator: DataCollatorForLanguageModeling**

### **Location**
- **File**: `src/data_loader/molecule_data_module.py` (line 80-82)
- **Class**: `DataCollatorForLanguageModeling` from Hugging Face Transformers

### **Configuration**
```python
self.data_collator = DataCollatorForLanguageModeling(
    tokenizer=self.tokenizer, 
    mlm=False  # Causal language modeling (not masked LM)
)
```

### **Input to Collator**
```python
# List of dictionaries from dataset
batch_data = [
    {"input_ids": [4093, 46, 50256, ...]},      # Tokenized SMILES 1
    {"input_ids": [4093, 7, 28, 46, ...]},      # Tokenized SMILES 2  
    {"input_ids": [66, 16, 535, 535, ...]},     # Tokenized SMILES 3
]
```

### **Output from Collator**
```python
{
    "input_ids": torch.Size([batch_size, max_seq_len]),      # torch.int64
    "attention_mask": torch.Size([batch_size, max_seq_len]), # torch.int64  
    "labels": torch.Size([batch_size, max_seq_len])          # torch.int64
}
```

### **Key Features**
- **Padding**: Pads sequences to the longest in the batch
- **Labels**: Creates shifted labels for causal language modeling
- **Attention Mask**: Marks padding tokens (0) vs real tokens (1)
- **Label Masking**: Uses -100 for padding positions in labels

## 🧬 **Dataset: MolDataModule**

### **Location**
- **File**: `src/data_loader/molecule_data_module.py`
- **Class**: `MolDataModule`

### **Tokenization Process**
```python
def tokenize_function(element: dict, max_length: int, mol_type: str, tokenizer):
    outputs = tokenizer(
        element[mol_type],  # SMILES string
        truncation=True,
        max_length=max_length,
        padding="max_length",
        add_special_tokens=True,
    )
    return {"input_ids": outputs["input_ids"]}
```

### **Dataset Features**
- **Input**: `{"SMILES": "CCO"}` (molecular string)
- **Output**: `{"input_ids": [4093, 46, 50256, ...]}` (tokenized)
- **Processing**: Applied via `dataset.map(tokenize_function, batched=True)`

### **Batch Creation**
```python
# Remove unused columns, keep only input_ids
dataset_ = dataset.remove_columns(column_names)
dataloader_params = {
    "batch_size": per_device_train_batch_size,
    "collate_fn": DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
    "num_workers": 1,
    "pin_memory": True,
}
```

## 🤖 **Model Forward Method**

### **Location**
- **File**: `src/models/modeling_novomolgen.py`
- **Class**: `NovoMolGen.forward()`

### **Input Parameters**
```python
def forward(
    self, 
    input_ids,                    # [batch_size, seq_len] torch.LongTensor
    attention_mask=None,          # [batch_size, seq_len] torch.BoolTensor (optional)
    labels=None,                  # [batch_size, seq_len] torch.LongTensor (optional)
    return_dict=None,             # bool (optional)
    position_ids=None,            # [batch_size, seq_len] torch.LongTensor (optional)
    inference_params=None,        # dict (optional)
    num_last_tokens=0,           # int (optional)
    cond_tokens=None,            # [batch_size, cond_len, hidden_size] torch.FloatTensor (optional)
    cond_attention_mask=None,    # [batch_size, cond_len] torch.BoolTensor (optional)
    **loss_kwargs
):
```

### **Output**
```python
# Returns CausalLMOutput or tuple
{
    "logits": torch.Size([batch_size, seq_len, vocab_size]),  # torch.FloatTensor
    "loss": torch.Size([]),                                   # torch.FloatTensor (if labels provided)
    "hidden_states": torch.Size([batch_size, seq_len, hidden_size])  # torch.FloatTensor (optional)
}
```

## 📊 **Complete Batch Flow**

### **1. Raw Dataset**
```python
{"SMILES": "CCO"}  # Molecular string
```

### **2. Tokenized Dataset**
```python
{
    "SMILES": "CCO",
    "input_ids": [4093, 46, 50256, 50256, ...],      # Tokenized sequence
    "attention_mask": [1, 1, 0, 0, ...]              # Padding mask
}
```

### **3. DataLoader Batch (before collation)**
```python
[
    {"input_ids": [4093, 46, 50256, ...], "attention_mask": [1, 1, 0, ...]},
    {"input_ids": [4093, 7, 28, 46, ...], "attention_mask": [1, 1, 1, 1, ...]},
    {"input_ids": [66, 16, 535, 535, ...], "attention_mask": [1, 1, 1, 1, ...]}
]
```

### **4. Collated Batch (after DataCollatorForLanguageModeling)**
```python
{
    "input_ids": torch.Size([3, 10]),      # [batch_size, max_seq_len]
    "attention_mask": torch.Size([3, 10]), # [batch_size, max_seq_len]  
    "labels": torch.Size([3, 10])          # [batch_size, max_seq_len]
}
```

### **5. Model Input (with cross-attention)**
```python
{
    "input_ids": torch.Size([3, 10]),                    # From collator
    "attention_mask": torch.Size([3, 10]),               # From collator
    "labels": torch.Size([3, 10]),                       # From collator
    "cond_tokens": torch.Size([3, 5, 256]),              # Additional (optional)
    "cond_attention_mask": torch.Size([3, 5])            # Additional (optional)
}
```

## 🔍 **Key Insights**

### **Data Collator Behavior**
1. **Padding**: Automatically pads to longest sequence in batch
2. **Labels**: Creates shifted labels for next-token prediction
3. **Masking**: Uses -100 for padding positions in labels (ignored in loss)
4. **Attention Mask**: 1 for real tokens, 0 for padding

### **Label Creation Logic**
```python
# Labels are shifted input_ids for causal language modeling
labels = input_ids.clone()
labels[:, :-1] = input_ids[:, 1:]  # Shift left
labels[:, -1] = -100               # Last position ignored
# Padding positions also set to -100
```

### **Cross-Attention Integration**
- **Standard Training**: Only uses `input_ids`, `attention_mask`, `labels`
- **Cross-Attention Training**: Adds `cond_tokens` and `cond_attention_mask`
- **Generation**: Uses `prepare_inputs_for_generation()` to pass conditions

### **Memory Layout**
```python
# Typical batch shapes for training
batch_size = 32
max_seq_len = 64
hidden_size = 256
cond_len = 10

# Memory usage (approximate)
input_ids: 32 * 64 * 8 bytes = 16 KB
attention_mask: 32 * 64 * 8 bytes = 16 KB  
labels: 32 * 64 * 8 bytes = 16 KB
cond_tokens: 32 * 10 * 256 * 4 bytes = 320 KB
cond_attention_mask: 32 * 10 * 1 byte = 320 bytes
```

## 🎯 **Summary**

### **Batch Keys and Shapes**
| Key | Shape | Dtype | Source | Purpose |
|-----|-------|-------|--------|---------|
| `input_ids` | `[B, T]` | `torch.int64` | DataCollator | Input token IDs |
| `attention_mask` | `[B, T]` | `torch.int64` | DataCollator | Padding mask |
| `labels` | `[B, T]` | `torch.int64` | DataCollator | Shifted targets |
| `cond_tokens` | `[B, Lc, d]` | `torch.float32` | Manual | Cross-attention keys/values |
| `cond_attention_mask` | `[B, Lc]` | `torch.bool` | Manual | Condition padding mask |

**Legend**: B=batch_size, T=seq_len, Lc=cond_len, d=hidden_size

### **Model Consumption**
The `NovoMolGen.forward()` method consumes:
- **Required**: `input_ids`
- **Optional**: `attention_mask`, `labels`, `position_ids`, `inference_params`
- **Cross-Attention**: `cond_tokens`, `cond_attention_mask`

### **Training Pipeline**
1. **Dataset** → Raw molecular strings
2. **Tokenization** → Token IDs with padding
3. **DataCollator** → Batched tensors with labels
4. **Model** → Forward pass with optional cross-attention
5. **Loss** → Cross-entropy on non-padded positions

This structure enables both standard language modeling and conditional generation with cross-attention mechanisms.
