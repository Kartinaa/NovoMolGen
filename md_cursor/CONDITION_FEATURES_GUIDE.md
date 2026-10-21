# Loading Real Condition Features in NovoMolGen

## Overview

The NovoMolGen model now supports loading real condition features (`pocket_vec`, `evo_vec`, `ifp`, `ligand_vec`) from your dataset instead of using random initialization. This guide shows you how to set up your dataset and use real condition data.

## Dataset Format

Your dataset should include the following columns in addition to the molecular representation (SMILES, SAFE, etc.):

```python
{
    "SMILES": "CCO",  # or SAFE, SELFIES, etc.
    "pocket_vec": [0.1, 0.2, ...],  # List of 512 floats (Uni-Mol pocket embedding)
    "evo_vec": [0.3, 0.4, ...],     # List of 1280 floats (ESM-2 evolutionary embedding)
    "ifp": [0.5, 0.6, ...],         # List of 16384 floats (Interaction fingerprint)
    "ligand_vec": [0.7, 0.8, ...]   # List of 1536 floats (Ligand molecular representation)
}
```

## Method 1: Using MolDataModule with Real Data

### Step 1: Prepare Your Dataset

Create a dataset with condition features. **Each sample must include ALL required fields**:

```python
from datasets import Dataset
import numpy as np

# Example: Create a dataset with real condition features
data = {
    "SMILES": ["CCO", "CC(=O)O", "c1ccccc1"],
    "pocket_vec": [
        np.random.randn(512).tolist(),  # Replace with real pocket embeddings
        np.random.randn(512).tolist(),
        np.random.randn(512).tolist(),
    ],
    "evo_vec": [
        np.random.randn(1280).tolist(),  # Replace with real ESM-2 embeddings
        np.random.randn(1280).tolist(),
        np.random.randn(1280).tolist(),
    ],
    "ifp": [
        np.random.randn(16384).tolist(),  # Replace with real interaction fingerprints
        np.random.randn(16384).tolist(),
        np.random.randn(16384).tolist(),
    ],
    "ligand_vec": [
        np.random.randn(1536).tolist(),  # Replace with real ligand representations
        np.random.randn(1536).tolist(),
        np.random.randn(1536).tolist(),
    ]
}

dataset = Dataset.from_dict(data)
```

### Step 1.5: Understanding the `features` Parameter

When `MolDataModule` processes your dataset, each sample becomes a dictionary in the `features` list. **With condition features enabled, each feature dictionary should look like this:**

```python
# Example of what each item in features looks like:
features = [
    {
        "input_ids": [1, 2, 3, 4, 5, 6, 7, 8],  # Tokenized SMILES (added by tokenizer)
        "pocket_vec": [0.1, 0.2, 0.3, ...],     # 512 floats
        "evo_vec": [0.4, 0.5, 0.6, ...],        # 1280 floats
        "ifp": [0.7, 0.8, 0.9, ...],            # 16384 floats
        "ligand_vec": [1.0, 1.1, 1.2, ...]      # 1536 floats
    },
    {
        "input_ids": [9, 10, 11, 12, 13, 14, 15, 16],
        "pocket_vec": [1.3, 1.4, 1.5, ...],
        "evo_vec": [1.6, 1.7, 1.8, ...],
        "ifp": [1.9, 2.0, 2.1, ...],
        "ligand_vec": [2.2, 2.3, 2.4, ...]
    },
    # ... more samples
]
```

**Key Points:**
- `input_ids` is added automatically by the tokenizer during dataset processing
- `pocket_vec`, `evo_vec`, `ifp`, `ligand_vec` must be in your original dataset
- Each condition feature must be a **list of floats** (not numpy arrays)
- All samples must have the same structure

### Data Flow Explanation

Here's what happens when you use `MolDataModule` with condition features:

1. **Original Dataset**: Your dataset has columns `["SMILES", "pocket_vec", "evo_vec", "ifp", "ligand_vec"]`

2. **Tokenization**: The tokenizer processes `SMILES` and adds `input_ids` column

3. **Collate Function**: When creating batches, the `_collate` function receives a list of feature dictionaries:
   ```python
   features = [
       {
           "input_ids": [1, 2, 3, 4, 5, 6, 7, 8],  # From tokenizer
           "pocket_vec": [0.1, 0.2, ...],          # From your dataset
           "evo_vec": [0.3, 0.4, ...],             # From your dataset
           "ifp": [0.5, 0.6, ...],                 # From your dataset
           "ligand_vec": [0.7, 0.8, ...]           # From your dataset
       },
       # ... more samples
   ]
   ```

4. **Batch Creation**: The collate function stacks these into tensors:
   ```python
   batch = {
       "input_ids": torch.tensor([[1, 2, 3, ...], [9, 10, 11, ...]]),  # [B, seq_len]
       "pocket_vec": torch.tensor([[0.1, 0.2, ...], [1.3, 1.4, ...]]), # [B, 512]
       "evo_vec": torch.tensor([[0.3, 0.4, ...], [1.6, 1.7, ...]]),    # [B, 1280]
       "ifp": torch.tensor([[0.5, 0.6, ...], [1.9, 2.0, ...]]),        # [B, 16384]
       "ligand_vec": torch.tensor([[0.7, 0.8, ...], [2.2, 2.3, ...]])  # [B, 1536]
   }
   ```

### Step 2: Initialize MolDataModule

```python
from src.data_loader.molecule_data_module import MolDataModule

# Initialize with condition features enabled
dm = MolDataModule(
    dataset_name="your_dataset_name",  # or use local dataset
    tokenizer_path="path/to/your/tokenizer.json",
    mol_type="SMILES",
    max_seq_length=64,
    include_condition_features=True,  # Enable condition features
    device="cuda"
)

# Load the dataset
dm.load_tokenized_dataset()
```

### Step 3: Use in Training

```python
from torch.utils.data import DataLoader

# Create data loader
dataloader = DataLoader(
    dm.train_dataset,
    batch_size=32,
    collate_fn=dm.collate_fn,  # This will automatically load real condition features
    shuffle=True
)

# Training loop
for batch in dataloader:
    # batch now contains real condition features:
    # - batch["pocket_vec"]: [B, 512]
    # - batch["evo_vec"]: [B, 1280] 
    # - batch["ifp"]: [B, 16384]
    # - batch["ligand_vec"]: [B, 1536]
    
    outputs = model(**batch)
    loss = outputs.loss
    # ... training code
```

## Method 2: Using make_dummy_batch with Real Data

For testing or when you have condition data in memory:

```python
import torch
import numpy as np

# Prepare your condition data
condition_data = {
    "pocket_vec": [
        np.random.randn(512).tolist(),  # Real pocket embeddings
        np.random.randn(512).tolist(),
    ],
    "evo_vec": [
        np.random.randn(1280).tolist(),  # Real ESM-2 embeddings
        np.random.randn(1280).tolist(),
    ],
    "ifp": [
        np.random.randn(16384).tolist(),  # Real interaction fingerprints
        np.random.randn(16384).tolist(),
    ],
    "ligand_vec": [
        np.random.randn(1536).tolist(),  # Real ligand representations
        np.random.randn(1536).tolist(),
    ]
}

# Create batch with real condition data
batch = MolDataModule.make_dummy_batch(
    batch_size=2,
    seq_len=8,
    vocab_size=100,
    include_condition_features=True,
    device="cuda",
    condition_data=condition_data  # Pass real data
)

# Use in model
outputs = model(**batch)
```

## Method 3: Loading from Files

If your condition features are stored in separate files:

```python
import numpy as np
import json

def load_condition_features_from_files(sample_ids, base_path):
    """Load condition features from separate files"""
    condition_data = {
        "pocket_vec": [],
        "evo_vec": [],
        "ifp": [],
        "ligand_vec": []
    }
    
    for sample_id in sample_ids:
        # Load pocket embedding
        pocket_path = f"{base_path}/pocket_embeddings/{sample_id}.npy"
        condition_data["pocket_vec"].append(np.load(pocket_path).tolist())
        
        # Load ESM-2 embedding
        evo_path = f"{base_path}/esm2_embeddings/{sample_id}.npy"
        condition_data["evo_vec"].append(np.load(evo_path).tolist())
        
        # Load interaction fingerprint
        ifp_path = f"{base_path}/interaction_fingerprints/{sample_id}.npy"
        condition_data["ifp"].append(np.load(ifp_path).tolist())
        
        # Load ligand representation
        ligand_path = f"{base_path}/ligand_representations/{sample_id}.npy"
        condition_data["ligand_vec"].append(np.load(ligand_path).tolist())
    
    return condition_data

# Usage
sample_ids = ["sample_001", "sample_002", "sample_003"]
condition_data = load_condition_features_from_files(sample_ids, "/path/to/features")

batch = MolDataModule.make_dummy_batch(
    batch_size=len(sample_ids),
    include_condition_features=True,
    condition_data=condition_data
)
```

## Expected Data Formats

### Pocket Vector (pocket_vec)
- **Shape**: `[512]`
- **Type**: List of floats
- **Source**: Uni-Mol pocket embedding
- **Description**: 3D protein pocket representation

### Evolutionary Vector (evo_vec)
- **Shape**: `[1280]`
- **Type**: List of floats
- **Source**: ESM-2 evolutionary embedding
- **Description**: Protein sequence evolutionary information

### Interaction Fingerprint (ifp)
- **Shape**: `[16384]`
- **Type**: List of floats
- **Source**: Computed interaction fingerprint
- **Description**: Protein-ligand interaction features

### Ligand Vector (ligand_vec)
- **Shape**: `[1536]`
- **Type**: List of floats
- **Source**: Molecular representation (e.g., from RDKit, Uni-Mol)
- **Description**: Ligand molecular features

## Fallback Behavior

If condition features are not found in the dataset, the system will:

1. Print a warning: `"Warning: Condition features not found in dataset, using random initialization"`
2. Fall back to random initialization for testing purposes
3. Continue training with random features

## Integration with Existing Code

The changes are backward compatible:

- **Without condition features**: Works exactly as before
- **With condition features**: Automatically loads real data if available, falls back to random if not
- **Mixed datasets**: Some samples can have condition features, others can use random initialization

## Example: Complete Training Setup

```python
from src.data_loader.molecule_data_module import MolDataModule
from src.models.modeling_novomolgen import NovoMolGen, NovoMolGenConfig
from torch.utils.data import DataLoader

# 1. Setup data module with condition features
dm = MolDataModule(
    dataset_name="your_dataset_with_conditions",
    tokenizer_path="path/to/tokenizer.json",
    mol_type="SMILES",
    include_condition_features=True,
    device="cuda"
)

# 2. Load dataset
dm.load_tokenized_dataset()

# 3. Setup model
config = NovoMolGenConfig(
    hidden_size=512,
    num_attention_heads=8,
    num_hidden_layers=12,
    vocab_size=dm.tokenizer.vocab_size,
    enable_cross_attn=True,
    cross_layers=[3, 6, 9],
    cond_tokens_len=256,
    beta_kl=0.1
)

model = NovoMolGen(config, mol_type="SMILES")

# 4. Create data loader
dataloader = DataLoader(
    dm.train_dataset,
    batch_size=32,
    collate_fn=dm.collate_fn,
    shuffle=True
)

# 5. Training loop
for batch in dataloader:
    # batch contains real condition features if available
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    # ... optimizer step
```

This setup will automatically use real condition features from your dataset when available, making your model training much more effective!

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: "Condition features not found in dataset, using random initialization"

**Cause**: Your dataset doesn't include the required condition feature columns.

**Solution**: Make sure your dataset has these exact column names:
- `pocket_vec` (list of 512 floats)
- `evo_vec` (list of 1280 floats)  
- `ifp` (list of 16384 floats)
- `ligand_vec` (list of 1536 floats)

#### Issue 2: "KeyError: 'pocket_vec'" or similar

**Cause**: The condition features are missing from some samples in your dataset.

**Solution**: Ensure ALL samples have ALL condition features:
```python
# Check your dataset
for i, sample in enumerate(dataset):
    required_keys = ["pocket_vec", "evo_vec", "ifp", "ligand_vec"]
    missing_keys = [key for key in required_keys if key not in sample]
    if missing_keys:
        print(f"Sample {i} missing: {missing_keys}")
```

#### Issue 3: Shape mismatch errors

**Cause**: Condition features have wrong dimensions.

**Solution**: Verify the shapes:
```python
# Check shapes in your dataset
sample = dataset[0]
print(f"pocket_vec length: {len(sample['pocket_vec'])} (expected: 512)")
print(f"evo_vec length: {len(sample['evo_vec'])} (expected: 1280)")
print(f"ifp length: {len(sample['ifp'])} (expected: 16384)")
print(f"ligand_vec length: {len(sample['ligand_vec'])} (expected: 1536)")
```

#### Issue 4: Dtype errors (Float vs Half)

**Cause**: Condition features are in float32 but model expects float16.

**Solution**: The system handles this automatically, but if you see errors, ensure your data is in float32:
```python
# Convert numpy arrays to lists of floats
pocket_vec = your_numpy_array.astype(np.float32).tolist()
```

### Quick Verification Script

Use this script to verify your dataset format:

```python
def verify_condition_features(dataset):
    """Verify that dataset has correct condition features format"""
    print("🔍 Verifying condition features...")
    
    required_keys = ["pocket_vec", "evo_vec", "ifp", "ligand_vec"]
    expected_shapes = [512, 1280, 16384, 1536]
    
    # Check first sample
    sample = dataset[0]
    
    for key, expected_shape in zip(required_keys, expected_shapes):
        if key not in sample:
            print(f"❌ Missing key: {key}")
            return False
        
        if len(sample[key]) != expected_shape:
            print(f"❌ Wrong shape for {key}: {len(sample[key])} (expected: {expected_shape})")
            return False
        
        if not isinstance(sample[key], list):
            print(f"❌ {key} should be a list, got: {type(sample[key])}")
            return False
        
        if not all(isinstance(x, (int, float)) for x in sample[key]):
            print(f"❌ {key} should contain only numbers")
            return False
    
    print("✅ All condition features verified!")
    return True

# Usage
verify_condition_features(your_dataset)
```
