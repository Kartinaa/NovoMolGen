# Structure-Safe: Conditional Molecule Generation with Protein-Ligand Conditioning

Hi Tak! This is a guide to understanding how this repo works. It is built on top of **NovoMolGen**, a GPT-style molecule language model, and extends it with a conditioning mechanism that steers molecule generation towards a target protein pocket and reference ligand.

---

## Big Picture

The goal is: given a protein pocket and a known active ligand, generate new drug-like molecules that are likely to bind to the same pocket.

The model is a **pretrained causal language model** (NovoMolGen, ~300M params) that generates molecules token-by-token in [SAFE](https://github.com/datamol-io/safe) string format. We fine-tune it by injecting condition signals (protein + ligand embeddings) through **cross-attention adapters** at selected transformer layers.

```
Protein Pocket ──────────────────────────────────---------------
  pocket_vec [B, 512]   (Uni-Mol)                               │
  evo_vec    [B, 1280]  (ESM-2)                                 
                                                                ▼
                                                          ProteinConditionEncoder
                                                                │
                                                          protein_condition [B, z_dim]
                                                                │
Reference Ligand ────────────────────┐                          │
  ifp        [B, 16384] (PLEC-IFP)  │                           │
  ligand_vec [B, 1536]  (MolBERT)   │                           │
                                     ▼                          │
                           LigandConditionEncoder (VAE)         │
                             mu, sigma, logvar [B, z_dim]       │
                                 │ reparameterize               │
                                 z [B, z_dim]                   │
                                 │                              │
                                 └────────────────--------------
                                          │
                                   ConditionFusion
                                          │
                                  fused_condition [B, z_dim]
                                          │
                                      cond_proj
                                          │
                                  cond_tokens [B, Lc, d_model]
                                          │
                        ┌─────────────────┼─────────────────┐
                        ▼                 ▼                 ▼
                   Layer 1-2      Cross-Attn @ L3    ...  Cross-Attn @ L11
                  (unchanged)     Q=hidden, K/V=cond        (gated residual)
                                          │
                                   lm_head logits
                                          │
                                  Generated SAFE string
```

---

## Repository Structure (Key Files)

```
src/
  models/
    modeling_novomolgen_infonce_120225.py   # Main model (InfoNCE variant)
    modeling_novomolgen_tanimoto.py         # Model variant with Tanimoto loss
    condition/
      protein_encoder.py                   # ProteinConditionEncoder
      ligand_encoder.py                    # LigandConditionEncoder (VAE)
      condition_fusion.py                  # ConditionFusion
  data_loader/
    molecule_data_module.py                # Dataset loading, tokenization, collation
    molecule_data_module_augmented.py      # Variant for pre-computed SAFE strings

scripts/
  finetune_full.py                         # Main training entry point
  generate_from_full_model.py             # Generate from a trained checkpoint
  generate_from_pdb.py                    # Generate given a raw PDB file

configs/finetune/
  bdnv2_config_finetune_infonce.yaml      # Config used for InfoNCE training run
  bdnv2_config_finetune_tanimoto.yaml     # Config used for Tanimoto loss training run

utils/
  build_pocket_lmdb_and_infer.py          # Extract pocket from PDB + run Uni-Mol inference
  condition_extract.py                    # Compute IFP (PLEC) and ligand_vec embeddings
```

---

## Step 1 — Data Pipeline (`MolDataModule`)

**Source:** `src/data_loader/molecule_data_module.py`

The dataset is a HuggingFace `Dataset` (stored on disk) where each row contains:

| Column | Shape | Description |
|---|---|---|
| `SMILES` | string | Raw molecule SMILES |
| `pocket_vec` | [512] | Uni-Mol pocket embedding |
| `evo_vec` | [1280] | ESM-2 protein sequence embedding |
| `ifp` | [16384] | PLEC interaction fingerprint |
| `ligand_vec` | [1536] | Ligand molecular representation (MolBERT/similar) |

**Processing pipeline:**

1. **Mol-type conversion** — SMILES → SAFE encoding (`safe.encode(smiles, ignore_stereo=True)`)
2. **Tokenization** — BPE tokenizer with `max_seq_length=64`, padding + truncation
3. **Morgan fingerprint pre-computation** — 2048-bit Morgan FP (radius=2) stored as `morgan_fp`; invalid molecules (empty FP) are filtered out
4. **Condition features pass-through** — `pocket_vec`, `evo_vec`, `ifp`, `ligand_vec` are preserved through all `.map()` calls
5. **Custom collator** (`_build_collate_fn`) — at batch time, extracts and stacks condition tensors separately from token tensors, then merges them into one batch dict

The final batch fed to the model looks like:
```python
{
    "input_ids":      [B, 64],     # Tokenized SAFE
    "attention_mask": [B, 64],
    "labels":         [B, 64],     # Shifted input_ids (causal LM target)
    "pocket_vec":     [B, 512],
    "evo_vec":        [B, 1280],
    "ifp":            [B, 16384],
    "ligand_vec":     [B, 1536],
    "morgan_fp":      [B, 2048],   # For Tanimoto loss (optional)
}
```

---

## Step 2 — Model Architecture (`NovoMolGen`)

**Source:** `src/models/modeling_novomolgen_infonce_120225.py`

### Base Transformer

The backbone is a **flash-attention GPT-LLaMA hybrid** (`GPTLMHeadModel` from flash-attn library), initialized from a pretrained NovoMolGen checkpoint. The architecture is equivalent to a 12-layer LLaMA with flash attention.

### Condition Encoders

**ProteinConditionEncoder**
```
pocket_vec [B, 512]  ──┐
                        ├─► MLP fusion ──► protein_condition [B, z_dim]
evo_vec    [B, 1280] ──┘
```

**LigandConditionEncoder** (VAE-style)
```
ifp        [B, 16384] ──┐
                         ├─► Two-tower MLP ──► mu, sigma, logvar [B, z_dim]
ligand_vec [B, 1536]  ──┘

z = mu + exp(0.5 * logvar) * ε    # Reparameterization trick (ε ~ N(0,I))
```
- During **training**: sample z from the posterior q(z | ifp, ligand_vec)
- During **generation**: use mu directly (deterministic) or sample from prior N(0,I)

**ConditionFusion**
```
protein_condition [B, z_dim]  ──┐
                                 ├─► cross-attn fusion ──► fused_condition [B, z_dim]
z                [B, z_dim]  ──┘
```

**Condition Projection**
```
fused_condition [B, z_dim] ──► outer-product with cond_proj ──► cond_tokens [B, z_dim, d_model]
```
`cond_proj` is a learnable matrix `[z_dim, d_model]`. Each of the `z_dim` scalars in `fused_condition` is broadcast across the hidden dimension, yielding `z_dim` condition tokens.

### Cross-Attention Injection (Monkey Patching)

Cross-attention adapters are injected at specified layers (e.g. layers 3,7,11... for 300M model) by **monkey-patching** each transformer block's `forward()` method:

```python
# Pseudocode of the patched block forward:
output = original_self_attn_mlp_forward(hidden_states)   # Normal transformer block
if cond_tokens is not None:
    normed = LayerNorm(output)
    attn_out = CrossAttentionAdapter(Q=normed, K=cond_tokens, V=cond_tokens)
    attn_out = dropout(attn_out)
    gate_value = sigmoid(gate)                            # gate is a learned scalar
    output = output + gate_value * attn_out              # Gated residual
```

Key design choices:
- **Pure cross-attention**: no extra self-attention; the adapter is a single `nn.MultiheadAttention` module
- **Gated residual**: each layer has an independent learnable scalar `gate` (init=1.0), so `sigmoid(gate) ≈ 0.73` initially. The gate is looked up **dynamically** at every forward call (not captured at patch time) to keep it in the computation graph
- **Layer norm before cross-attn**: pre-norm on the query to stabilize training

---

## Step 3 — Loss Functions

**Total loss:**
```
L = NLL + β(t) × KL + λ × InfoNCE
```

### NLL (Language Modeling Loss)
Standard cross-entropy over the token sequence (next-token prediction).

### KL Divergence
```
KL = E[ KL(q(z | x) || N(0, I)) ]
   = E[ 0.5 * Σ(exp(logvar) + mu² - 1 - logvar) ]
```
- **Free bits**: KL is only penalized above a threshold (hinge: `max(KL_per_sample, 5.0)`) to prevent posterior collapse
- **Per-token normalization**: divided by average sequence length so it is on the same scale as NLL
- **KL annealing**: β starts at 0 and linearly ramps to 0.1 over 30k steps (configurable)

### InfoNCE (Contrastive Alignment)
Aligns the molecule's last-layer hidden state (mean-pooled) with its corresponding `ligand_vec`:
```python
q = infonce_hidden_proj(mean_pool(hidden_states))   # [B, D]  — from LM
k = infonce_ligand_proj(ligand_vec)                 # [B, D]  — from condition encoder
# Both L2-normalized
logits = (q @ k.T) / temperature                   # [B, B] similarity matrix
InfoNCE = CrossEntropy(logits, arange(B))           # Diagonal = positives
```
The purpose is to make the model's internal representations actually encode ligand identity, not just protein context.

---

## Step 4 — Training

**Entry point:** `scripts/finetune_full.py`
** example Config:** `configs/finetune/bdnv2_config_finetune_infonce.yaml`

```bash
python scripts/finetune_full.py \
  --config configs/finetune/bdnv2_config_finetune_infonce.yaml \
  --output_dir outputs/my_run \
  --run_name "my_run"
```

### Example Config Parameters

```yaml
# Base model
pretrained_path: "models/novomolgen_300M_YB/checkpoint-309520"

# Cross-attention
enable_cross_attn: true
cross_layers: [3,7,11,15,19,23,27,31]
train_new_modules_only: false   # false = fine-tune entire model
                                # true  = freeze base, train adapters+encoders only

# Losses
infonce_weight: 0.1
infonce_temperature: 0.2

# KL annealing
vae:
  beta_kl_init: 0
  beta_kl_final: 0.1
  beta_kl_anneal_type: "linear"
  beta_kl_anneal_steps: 30000

# Data
dataset_name: "finetune_data/processed_data/hf_bdnv2_dataset/train"
mol_type: "SAFE"
max_seq_length: 64
per_device_train_batch_size: 250
gradient_accumulation_steps: 4   # effective batch = 1000
learning_rate: 5e-5
```

### What Gets Trained

When `train_new_modules_only: false` (full fine-tuning), all parameters are updated. When `true`, only:
- `transformer.cross_adapters.*` — cross-attention weights
- `transformer.gates.*` — per-layer gate scalars
- `ligand_encoder.*`, `protein_encoder.*`, `condition_fusion.*`, `cond_proj.*`

### Callbacks
- `LossMonitoringCallback` — logs NLL, KL, InfoNCE separately to W&B
- `GateFixCallback` — verifies gates are in the optimizer and logs gate values + gradients every 500 steps
- `ProgressiveUnfreezingCallback` — optionally unfreezes base transformer layers one by one during training

---

## Step 5 — Generation

```python
# Load trained model
from models.modeling_novomolgen_infonce_120225 import NovoMolGen
model = NovoMolGen.from_pretrained("outputs/my_run/checkpoint-XXX/full_model")
model.eval().cuda()

# Generate conditioned on a protein-ligand pair
outputs = model.generate_with_condition(
    input_ids=bos_tokens,          # [B, 1] start token
    pocket_vec=pocket_vec,         # [B, 512]
    evo_vec=evo_vec,               # [B, 1280]
    ifp=ifp,                       # [B, 16384]
    ligand_vec=ligand_vec,         # [B, 1536]
    max_length=64,
    temperature=1.0,
    top_p=0.95,
)
# outputs: token sequences → decode with tokenizer → SAFE strings → SMILES
```

During generation, `generate_with_condition` sets `transformer._current_cond_tokens` before calling the base `generate()` loop, so every autoregressive forward call sees the condition.

---

## Preparing Condition Features for a New Target

To run generation on a new protein:

1. **Extract pocket & run Uni-Mol** → `pocket_vec [512]`
   ```bash
   python utils/build_pocket_lmdb_and_infer.py \
     --pdb-file protein.pdb --ligand-sdf ligand.sdf \
     --job-name my_target --dict-file ... --weights ...
   ```

2. **Compute ESM-2 embedding** → `evo_vec [1280]`
   (run ESM-2 on the protein sequence)

3. **Compute PLEC fingerprint** → `ifp [16384]`
   ```python
   from utils.condition_extract import calc_ifp_plec
   ifp = calc_ifp_plec("protein.pdb", "ligand.sdf")
   ```

4. **Compute ligand molecular representation** → `ligand_vec [1536]`
   ```python
   from utils.condition_extract import calc_smi_representation
   ligand_vec = calc_smi_representation([ligand_smiles])
   ```

Or use `scripts/generate_from_pdb.py` which wraps the whole pipeline.

---

## Quick Reference: Tensor Dimensions

| Name | Shape | Source |
|---|---|---|
| `pocket_vec` | [B, 512] | Uni-Mol pocket encoder |
| `evo_vec` | [B, 1280] | ESM-2 (last layer, mean pooled) |
| `ifp` | [B, 16384] | PLEC interaction fingerprint |
| `ligand_vec` | [B, 1536] | Ligand molecular encoder |
| `protein_condition` | [B, z_dim] | ProteinConditionEncoder output |
| `mu / logvar` | [B, z_dim] | LigandConditionEncoder VAE posterior |
| `z` | [B, z_dim] | Sampled latent (reparameterization) |
| `fused_condition` | [B, z_dim] | ConditionFusion output |
| `cond_tokens` | [B, z_dim, d_model] | Cross-attention key/value |
| `hidden_states` | [B, seq_len, d_model] | LM hidden states |

`z_dim` = `cond_tokens_len` = 512 (configurable via `latent_dim` in config).
`d_model` = 1024 (32M model) or 2048 (300M model).
