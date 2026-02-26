# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a modified version of NovoMolGen focused on **conditional molecule generation** with protein-ligand conditioning. The key modifications add:
- Cross-attention adapters for conditioning on protein/ligand features
- VAE-style latent encoding with KL divergence loss and annealing
- InfoNCE contrastive loss for aligning molecule embeddings with ligand representations
- Progressive unfreezing and parameter-efficient fine-tuning (LoRA)

## Common Commands

### Conditional Fine-tuning (Main Workflow)
```bash
# Fine-tune with cross-attention + KL loss (BDNV2 dataset)
python scripts/finetune_full.py \
  --config configs/finetune/bdnv2_config_finetune.yaml \
  --output_dir outputs/my_experiment \
  --run_name "my_finetune_run"

# Fine-tune with InfoNCE loss (CrossDocked dataset)
python scripts/finetune_full.py \
  --config configs/finetune/bdnv2_config_finetune_infonce_crossdock.yaml \
  --output_dir outputs/infonce_experiment

# Resume from checkpoint
python scripts/finetune_full.py \
  --config configs/finetune/bdnv2_config_finetune.yaml \
  --output_dir outputs/my_experiment \
  --resume_from_checkpoint outputs/my_experiment/checkpoint-2100
```

### Generate Molecules from Trained Model
```bash
python scripts/generate_from_full_model.py \
  --model_path outputs/my_experiment/checkpoint-XXX/full_model \
  --validation_sets finetune_data/processed_data/hf_bdnv2_dataset/validation \
  --num_samples_per_val 50 \
  --output_dir outputs/generated_molecules
```

### Run Tests
```bash
pytest tests/
```

## Architecture Overview

### Condition Feature Pipeline

**Input Batch Structure**:
```python
batch = {
    "input_ids": [B, seq_len],       # Tokenized SAFE/SMILES molecules
    "attention_mask": [B, seq_len],
    "labels": [B, seq_len],          # Shifted for causal LM
    "pocket_vec": [B, 512],          # Uni-Mol pocket embedding
    "evo_vec": [B, 1280],            # ESM-2 evolutionary embedding
    "ifp": [B, 16384],               # Interaction fingerprint
    "ligand_vec": [B, 1536],         # Ligand molecular representation
}
```

**Data Flow**:
```
pocket_vec + evo_vec  →  ProteinConditionEncoder  →  protein_condition [B, z_dim]
                                                            ↓
ifp + ligand_vec  →  LigandConditionEncoder  →  mu, sigma, logvar [B, z_dim]
                                                            ↓
                                                    Sample z (reparameterization)
                                                            ↓
protein_condition + z  →  ConditionFusion  →  fused_condition [B, z_dim]
                                                            ↓
                                                    cond_proj → cond_tokens [B, Lc, d]
                                                            ↓
                                            Cross-attention at specified layers
```

### Loss Functions

**Total Loss**: `loss = NLL + β(t) × KL + λ × InfoNCE`

- **NLL**: Standard language modeling cross-entropy loss
- **KL**: KL divergence between posterior q(z|x) and prior N(0,I), normalized per-token
- **InfoNCE**: Contrastive alignment between hidden embeddings and ligand vectors (optional)

### Key Model Files

| File | Description |
|------|-------------|
| `src/models/modeling_novomolgen.py` | Main model with cross-attention, KL loss |
| `src/models/modeling_novomolgen_infonce_112925.py` | Model variant with InfoNCE loss |
| `src/models/condition/protein_encoder.py` | ProteinConditionEncoder (pocket + evo fusion) |
| `src/models/condition/ligand_encoder.py` | LigandConditionEncoder (IFP + mol → VAE posterior) |
| `src/models/condition/condition_fusion.py` | ConditionFusion (protein + ligand z) |
| `scripts/finetune_full.py` | Main fine-tuning script |
| `scripts/generate_from_full_model.py` | Generation from trained checkpoints |

## Configuration Reference

### Fine-tuning Config (`configs/finetune/bdnv2_config_finetune*.yaml`)

**Model Settings**:
```yaml
pretrained_path: "models/novomolgen_32M_YB/checkpoint-309520"
enable_cross_attn: true
cross_layers: [3, 6, 9]        # Layers for cross-attention injection (0-indexed)
train_new_modules_only: false  # true = freeze base, train only adapters/encoders
enable_lora: false
```

**InfoNCE Settings** (in `*_infonce*.yaml`):
```yaml
infonce_weight: 0.1            # λ weight for InfoNCE loss
infonce_temperature: 0.2       # Temperature for softmax
```

**VAE/KL Annealing**:
```yaml
vae:
  beta_kl_init: 0              # Initial β (start of training)
  beta_kl_final: 0.1           # Final β (after annealing)
  beta_kl_anneal_type: "linear"  # "linear", "cosine", "cyclical"
  beta_kl_anneal_steps: 30000  # Steps to anneal from init to final
  beta_kl_eval: 1              # Fixed β for evaluation (if set)
```

**Progressive Unfreezing** (optional):
```yaml
progressive_unfreezing:
  enabled: false
  layers_to_unfreeze: [11, 10, 9, 8, 7, 6, 5, 4]  # Top layers first
  unfreeze_steps: [5000, 10000, 15000, ...]       # Steps to unfreeze each
```

**Data Settings**:
```yaml
dataset_name: "finetune_data/processed_data/hf_bdnv2_dataset/train"
validation_set_names: ["finetune_data/processed_data/hf_bdnv2_dataset/validation"]
include_condition_features: true
latent_dim: 128
mol_type: "SAFE"
max_seq_length: 64
```

### Cross-Attention Layer Selection

For 12-layer model (32M): `cross_layers: [3, 6, 9]` or `[3, 6, 7, 8, 9, 10, 11]`
For 32-layer model (300M): `cross_layers: [3, 7, 11, 15, 19, 23, 27, 31]`

## Dataset Preparation

Datasets must include condition features as columns:
- `pocket_vec`: List of 512 floats (Uni-Mol pocket embedding)
- `evo_vec`: List of 1280 floats (ESM-2 evolutionary embedding)
- `ifp`: List of 16384 floats (Interaction fingerprint)
- `ligand_vec`: List of 1536 floats (Ligand molecular representation)

See `md_cursor/CONDITION_FEATURES_GUIDE.md` for detailed dataset format.

## Key Implementation Details

### Cross-Attention Injection (Monkey Patching)

Cross-attention is injected via monkey-patching transformer block forward methods:
```python
# In CrossAttentionTransformer._patch_transformer_blocks()
output = original_forward(hidden_states, ...)
if cond_tokens is not None:
    normed_output = block.norm1(output)
    attn_out = adapter(normed_output, cond_tokens, cond_mask)
    gate_value = torch.sigmoid(gate)
    output = output + gate_value * dropout(attn_out)
```

Gate parameters are initialized to 0 (sigmoid(0) = 0.5).

### Trainable Parameters

When `train_new_modules_only: true`, only these are trained:
- `transformer.cross_adapters.*`
- `transformer.gates.*`
- `ligand_encoder.*`, `protein_encoder.*`, `condition_fusion.*`, `cond_proj.*`
- LoRA parameters (if enabled)

### KL Normalization

KL loss is normalized to per-token scale to match NLL loss:
```python
kl_loss = kl_per_dim.sum(dim=-1).mean()  # Per sample
avg_seq_len = (labels != -100).sum(dim=1).float().mean()
kl_loss = kl_loss / avg_seq_len  # Per token
```

## Debugging

### Monitor Gate Values
```python
for layer_idx in model.transformer.cross_layers:
    gate = torch.sigmoid(model.transformer.gates[str(layer_idx)]).item()
    print(f"Layer {layer_idx} gate: {gate:.4f}")
```

### Check Trainable Parameters
```python
model.print_trainable_summary(max_lines=30)
```

### Verify Condition Features in Dataset
```python
from tests.verify_condition_features import verify_condition_features
verify_condition_features(your_dataset)
```

## Related Documentation

- `md_cursor/NEW_ARCHITECTURE.md`: Overall architecture documentation
- `md_cursor/CROSS_ATTENTION_DESIGN.md`: Cross-attention implementation details
- `md_cursor/CONDITION_FEATURES_GUIDE.md`: Dataset preparation guide
- `scripts/README_finetune.md`: Fine-tuning script documentation
