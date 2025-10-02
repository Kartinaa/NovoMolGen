# Using Your Own Dataset with NovoMolGen

This guide shows you how to use your own molecular dataset with NovoMolGen, including both SMILES and SAFE representations.

## Table of Contents
1. [Dataset Format Requirements](#dataset-format-requirements)
2. [Creating Your Dataset](#creating-your-dataset)
3. [Training Custom Tokenizers](#training-custom-tokenizers)
4. [Configuration Files](#configuration-files)
5. [Complete Workflow Examples](#complete-workflow-examples)
6. [Troubleshooting](#troubleshooting)

## Dataset Format Requirements

### Supported Formats
- **JSON**: `{"SMILES": [...], "SAFE": [...]}`
- **CSV**: Columns named `SMILES`, `SAFE`, etc.
- **Parquet**: Same column structure
- **Arrow**: Hugging Face datasets format

### Required Structure
Your dataset must have columns matching the molecular representation you want to use:

```json
{
  "SMILES": ["CCO", "CCN", "CC(C)O", ...],
  "SAFE": ["[C][C][O]", "[C][C][N]", "[C][C][Branch1][C][O]", ...]
}
```

Or for CSV:
```csv
SMILES,SAFE
CCO,[C][C][O]
CCN,[C][C][N]
CC(C)O,[C][C][Branch1][C][O]
```

## Creating Your Dataset

### Option 1: Local JSON File
```python
import json

# Your molecular data
molecules = {
    "SMILES": [
        "CCO",           # Ethanol
        "CCN",           # Ethylamine  
        "CC(C)O",        # Isopropanol
        "c1ccccc1",      # Benzene
        # ... add more molecules
    ],
    "SAFE": [
        "[C][C][O]",
        "[C][C][N]", 
        "[C][C][Branch1][C][O]",
        "[Ring1][C][C][C][C][C][C]",
        # ... corresponding SAFE representations
    ]
}

# Save to file
with open("my_molecules.json", "w") as f:
    json.dump(molecules, f, indent=2)
```

### Option 2: Local CSV File
```python
import pandas as pd

# Create DataFrame
df = pd.DataFrame({
    "SMILES": ["CCO", "CCN", "CC(C)O", "c1ccccc1"],
    "SAFE": ["[C][C][O]", "[C][C][N]", "[C][C][Branch1][C][O]", "[Ring1][C][C][C][C][C][C]"]
})

# Save to CSV
df.to_csv("my_molecules.csv", index=False)
```

### Option 3: Hugging Face Dataset (for sharing)
```python
from datasets import Dataset
from huggingface_hub import login

# Create dataset
dataset = Dataset.from_dict({
    "SMILES": ["CCO", "CCN", "CC(C)O"],
    "SAFE": ["[C][C][O]", "[C][C][N]", "[C][C][Branch1][C][O]"]
})

# Upload to Hugging Face Hub
login()  # Login to HF
dataset.push_to_hub("your-username/my-molecule-dataset")
```

## Training Custom Tokenizers

### For SMILES Data
```bash
python src/data_loader/molecule_tokenizer.py \
    --dataset "./my_molecules.json" \
    --mol_type "SMILES" \
    --tokenizer_type "bpe" \
    --splitter "atomwise" \
    --vocab_size 10000 \
    --min_frequency 2 \
    --batch_size 1000
```

### For SAFE Data
```bash
python src/data_loader/molecule_tokenizer.py \
    --dataset "./my_molecules.json" \
    --mol_type "SAFE" \
    --tokenizer_type "bpe" \
    --splitter "atomwise" \
    --vocab_size 10000 \
    --min_frequency 2 \
    --batch_size 1000
```

### Tokenizer Types Explained
- **`bpe`**: Best for most cases, learns subword patterns
- **`wordlevel`**: Simpler, each token is atomic
- **`wordpiece`**: Google's approach, similar to BPE
- **`unigram`**: Learns optimal tokenization

## Configuration Files

### Dataset Config (YAML)
Create `configs/dataset/my_dataset.yaml`:

```yaml
# For SMILES
dataset_name: "./my_molecules.json"  # Path to your data
tokenizer_path: "./data/tokenizers/tokenizer_bpe_atomwise_SMILES_10000_2_0.json"
mol_type: "SMILES"
max_seq_length: 128
num_proc: 4
streaming: false
validation_set_names: "./my_validation.json"  # Optional
filter_validation_set: true
```

Or for SAFE:
```yaml
# For SAFE
dataset_name: "./my_molecules.json"
tokenizer_path: "./data/tokenizers/tokenizer_bpe_atomwise_SAFE_10000_2_0.json"
mol_type: "SAFE"
max_seq_length: 128
num_proc: 4
streaming: false
validation_set_names: "./my_validation.json"
filter_validation_set: true
```

### Training Config
Create `configs/train_my_dataset.yaml`:

```yaml
# Dataset configuration
dataset:
  dataset_name: "./my_molecules.json"
  tokenizer_path: "./data/tokenizers/tokenizer_bpe_atomwise_SMILES_10000_2_0.json"
  mol_type: "SMILES"
  max_seq_length: 128
  num_proc: 4
  streaming: false

# Model configuration  
model:
  model_type: "llama"
  hidden_size: 512
  intermediate_size: 2048
  num_hidden_layers: 8
  num_attention_heads: 8
  vocab_size: 10000  # Must match your tokenizer
  max_position_embeddings: 128

# Training configuration
trainer:
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 4
  num_train_epochs: 3
  learning_rate: 5e-4
  warmup_steps: 1000
  logging_steps: 100
  save_steps: 1000
  eval_steps: 1000
  save_total_limit: 3
  dataloader_num_workers: 4
  fp16: true

# Other settings
seed: 42
save_path: "./checkpoints"
```

## Complete Workflow Examples

### Example 1: SMILES Dataset
```bash
# 1. Train tokenizer
python src/data_loader/molecule_tokenizer.py \
    --dataset "./my_smiles.json" \
    --mol_type "SMILES" \
    --tokenizer_type "bpe" \
    --vocab_size 10000

# 2. Tokenize dataset (optional, for faster training)
python src/main.py tokenize_dataset \
    --config_name=my_smiles_dataset \
    --config_dir=./configs/dataset

# 3. Train model
python src/main.py train \
    --config_name=train_my_smiles \
    --config_dir=./configs
```

### Example 2: SAFE Dataset
```bash
# 1. Train tokenizer
python src/data_loader/molecule_tokenizer.py \
    --dataset "./my_safe.json" \
    --mol_type "SAFE" \
    --tokenizer_type "bpe" \
    --vocab_size 10000

# 2. Train model
python src/main.py train \
    --config_name=train_my_safe \
    --config_dir=./configs
```

### Example 3: Mixed SMILES/SAFE Dataset
If your dataset has both representations:

```yaml
# configs/dataset/mixed_dataset.yaml
dataset_name: "./my_mixed_molecules.json"
tokenizer_path: "./data/tokenizers/tokenizer_bpe_atomwise_SMILES_10000_2_0.json"
mol_type: "SMILES"  # Use SMILES as primary, SAFE will be ignored
max_seq_length: 128
num_proc: 4
streaming: false
```

## Advanced Tips

### 1. Data Preprocessing
```python
from rdkit import Chem
from molvs import standardize_smiles

def clean_smiles(smiles_list):
    """Clean and standardize SMILES strings"""
    cleaned = []
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                cleaned_smiles = standardize_smiles(smiles)
                cleaned.append(cleaned_smiles)
        except:
            continue  # Skip invalid molecules
    return cleaned

# Apply to your data
cleaned_smiles = clean_smiles(your_smiles_list)
```

### 2. Validation Set Creation
```python
import json
from sklearn.model_selection import train_test_split

# Load your data
with open("my_molecules.json", "r") as f:
    data = json.load(f)

# Split into train/validation
train_smiles, val_smiles = train_test_split(
    data["SMILES"], test_size=0.1, random_state=42
)

# Save validation set
validation_data = {"SMILES": val_smiles}
with open("my_validation.json", "w") as f:
    json.dump(validation_data, f)
```

### 3. Memory-Efficient Large Datasets
For very large datasets, use streaming:

```yaml
# configs/dataset/large_dataset.yaml
dataset_name: "./huge_molecules.json"
tokenizer_path: "./data/tokenizers/tokenizer_bpe_atomwise_SMILES_30000_0_0.json"
mol_type: "SMILES"
max_seq_length: 128
num_proc: 8
streaming: true  # Enable streaming for large datasets
```

## Troubleshooting

### Common Issues

1. **"Column not found" error**
   - Ensure your dataset has the correct column name (e.g., "SMILES", "SAFE")
   - Check case sensitivity

2. **"Invalid tokenizer" error**
   - Make sure `vocab_size` in config matches your tokenizer
   - Verify tokenizer file path is correct

3. **"Out of memory" error**
   - Reduce `per_device_train_batch_size`
   - Increase `gradient_accumulation_steps`
   - Use `streaming: true` for large datasets

4. **"No valid molecules" error**
   - Check your SMILES/SAFE strings are valid
   - Use the cleaning function above

### Performance Tips

1. **Faster tokenization**: Use `num_proc` > 1
2. **Memory efficiency**: Use streaming for datasets > 1GB
3. **Better convergence**: Use validation set with `filter_validation_set: true`
4. **Optimal batch size**: Start with 8-16, adjust based on GPU memory

### File Structure
your_project/
├── my_molecules.json # Your dataset
├── my_validation.json # Validation set
├── configs/
│ ├── dataset/
│ │ └── my_dataset.yaml # Dataset config
│ └── train_my_dataset.yaml # Training config
├── data/
│ └── tokenizers/
│ └── tokenizer_.json # Your trained tokenizers
└── checkpoints/ # Model checkpoints