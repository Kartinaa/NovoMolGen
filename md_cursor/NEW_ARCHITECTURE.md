# NovoMolGen New Architecture Documentation

## Architecture Overview

The NovoMolGen model has been updated to use a more sophisticated conditioning architecture that processes protein and ligand information through dedicated encoders and fuses them using cross-attention mechanisms.

## Key Components

The new architecture consists of three main components:

1. **ProteinConditionEncoder**: Processes protein pocket and evolutionary embeddings
2. **LigandConditionEncoder**: Processes ligand interaction fingerprints and molecular representations  
3. **ConditionFusion**: Fuses protein and ligand conditions using cross-attention
4. **NovoMolGen**: Main model that integrates all components with cross-attention injection

## Overview

The architecture enables structure-based molecular generation by conditioning the language model on both protein structure (pocket + evolutionary information) and ligand properties (interaction fingerprints + molecular representations).

## Data Flow Architecture

### 1. Input Batch Structure

The model now receives the following batch structure from `MolDataModule`:

```python
batch = {
    "input_ids": torch.LongTensor,      # [B, seq_len] - Tokenized molecular sequences
    "attention_mask": torch.LongTensor, # [B, seq_len] - Attention mask
    "labels": torch.LongTensor,         # [B, seq_len] - Shifted labels for training
    "pocket_vec": torch.FloatTensor,    # [B, 512] - Uni-Mol pocket embedding
    "evo_vec": torch.FloatTensor,       # [B, 1280] - ESM-2 evolutionary embedding  
    "ifp": torch.FloatTensor,           # [B, 16384] - Interaction fingerprint
    "ligand_vec": torch.FloatTensor,    # [B, 1536] - Ligand molecular representation
}
```

### 2. Encoder Architecture

#### Protein Condition Encoder (`ProteinConditionEncoder`)

**Input**: `pocket_vec` [B, 512] + `evo_vec` [B, 1280]  
**Output**: `protein_condition` [B, z_dim]

```
pocket_vec [B, 512] ──┐
                      ├── ProteinConditionEncoder ──→ protein_condition [B, z_dim]
evo_vec [B, 1280] ────┘
```

**Architecture**:
- Projects both inputs to shared `common_dim` (default: 512)
- Applies LayerNorm + Dropout
- Early fusion: `concat([a, b, a*b, |a-b|])`
- MLP to `out_dim`
- Optional gated residual connection

#### Ligand Condition Encoder (`LigandConditionEncoder`)

**Input**: `ifp` [B, 16384] + `ligand_vec` [B, 1536]  
**Output**: `mu` [B, z_dim], `sigma` [B, z_dim], `logvar` [B, z_dim]

```
ifp [B, 16384] ──┐
                 ├── LigandConditionEncoder ──→ mu, sigma, logvar [B, z_dim]
ligand_vec [B, 1536] ──┘
```

**Architecture**:
- **IFP Tower**: 16384 → 4096 → 1024 → d_embed (aggressive reduction)
- **MOL Tower**: 1536 → 1024 → d_embed (light reduction)
- Early fusion: `concat([a, b, a*b, |a-b|])`
- MLP to `d_embed`
- Gated residual connection
- Output heads: `mu_head`, `rho_head` (sigma = softplus(rho) + eps)

### 3. Condition Fusion (`ConditionFusion`)

**Input**: `protein_condition` [B, z_dim] + `z` [B, z_dim] (sampled ligand latent)  
**Output**: `fused_condition` [B, z_dim]

```
protein_condition [B, z_dim] ──┐
                               ├── ConditionFusion ──→ fused_condition [B, z_dim]
z [B, z_dim] (sampled) ────────┘
```

**Note**: The `z` vector is sampled from the posterior q_φ(z|x) when ligand information is available, or from the prior N(0,I) when no ligand information is provided.

**Architecture**:
- Projects all inputs to shared `common_dim`
- Cross-attention fusion: protein attends to ligand
- Final MLP to `out_dim`

### 4. Complete Data Flow

```mermaid
graph TD
    A[Input Batch] --> B[MolDataModule]
    B --> C[NovoMolGen.forward]
    
    C --> D[ProteinConditionEncoder]
    C --> E[LigandConditionEncoder]
    
    D --> F[protein_condition]
    E --> G[mu, sigma, logvar]
    
    F --> H[ConditionFusion]
    G --> H
    
    H --> I[fused_condition]
    G --> J[Sample z from posterior]
    
    I --> K[Concatenate z + fused_condition] # Interesting
    J --> K
    
    K --> L[Project to cond_tokens]
    L --> M[CrossAttentionTransformer]
    
    C --> N[input_ids]
    N --> M
    
    M --> O[Language Model Output]
    O --> P[Loss: NLL + β*KL]
```

### 5. Training Loss

The model computes a combined loss:

```
loss = nll_loss + β * kl_loss
```

Where:
- `nll_loss`: Standard language modeling loss (cross-entropy)
- `kl_loss`: KL divergence between posterior q_φ(z|x) and prior N(0,I)
- `β`: Fixed weight (default: 0.1)

### 6. Key Components

#### MolDataModule Updates

- Added `include_condition_features` flag
- Generates dummy condition vectors when enabled
- Supports both legacy and new batch structures

#### NovoMolGen Updates

- **New forward signature**: Accepts `pocket_vec`, `evo_vec`, `ifp`, `ligand_vec`
- **Encoder initialization**: Creates `protein_encoder`, `ligand_encoder`, `condition_fusion`
- **Condition building**: Processes features through encoders, samples z from posterior, and fuses conditions
- **Backward compatibility**: Maintains support for legacy inputs

#### Cross-Attention Integration

- Injects cross-attention at specified transformer layers
- Uses `cond_tokens` and `cond_attention_mask` for conditioning
- Gated residual connections for stable training

### 7. Usage Examples

#### Training with New Architecture

```python
# Initialize data module with condition features
dm = MolDataModule(
    dataset_name="your_dataset",
    tokenizer_path="path/to/tokenizer",
    include_condition_features=True,  # Enable new features
    latent_dim=64,
)

# Initialize model with cross-attention
config = NovoMolGenConfig(
    enable_cross_attn=True,
    cross_layers=[3, 6, 9],
    cond_tokens_len=128,
    beta_kl=0.1,
)

model = NovoMolGen(config)

# Training batch will include:
# - input_ids, attention_mask, labels
# - pocket_vec, evo_vec, ifp, ligand_vec
```

#### Generation with Conditions

```python
# Generate with protein and ligand conditions
generated = model.generate_with_condition(
    input_ids=prompt_ids,
    pocket_vec=pocket_embedding,    # [B, 512]
    evo_vec=evo_embedding,          # [B, 1280] 
    ifp=interaction_fingerprint,    # [B, 16384]
    ligand_vec=ligand_representation, # [B, 1536]
    max_length=100,
    temperature=1.0,
)
```

### 8. File Structure

```
src/
├── models/
│   ├── condition/
│   │   ├── __init__.py
│   │   ├── protein_encoder.py      # ProteinConditionEncoder
│   │   ├── ligand_encoder.py       # LigandConditionEncoder  
│   │   └── condition_fusion.py     # ConditionFusion
│   └── modeling_novomolgen.py      # Updated NovoMolGen
├── data_loader/
│   └── molecule_data_module.py     # Updated MolDataModule
└── utils/
    ├── esm_embedding_extract.py    # ESM-2 embeddings
    └── unimol_pocket_embedding_extract.py  # Uni-Mol embeddings
```

### 9. Configuration Parameters

#### NovoMolGenConfig
- `enable_cross_attn`: Enable cross-attention conditioning
- `cross_layers`: List of layer indices for cross-attention injection
- `cond_tokens_len`: Total length of condition tokens (z_dim * 2)
- `beta_kl`: Weight for KL regularization term

#### MolDataModule
- `include_condition_features`: Include new condition vectors in batches
- `latent_dim`: Dimension of latent space (z_dim)

### 10. Backward Compatibility

The architecture maintains backward compatibility:
- Legacy `protein_inputs` and `posterior_params` still supported
- Old `generate_with_condition` signatures work
- Gradual migration path from old to new architecture

This new architecture provides a more principled approach to protein-ligand conditioning while maintaining the flexibility and performance of the original NovoMolGen model.
