# Model Version History

This document records the key architectural and training changes across model versions.
All versions share the same base: `novomolgen_300M_YB/checkpoint-309520` (Mamba-2 300M LM).

---

## Version Overview

| Version | File | Key Change | Config |
|---------|------|------------|--------|
| v1/v2 | `modeling_novomolgen_infonce_120225.py` | Baseline cross-attention + VAE + InfoNCE | `bdnv2_config_finetune_augmented.yaml` |
| v3 | `modeling_novomolgen_infonce_120225_v3.py` | mu instead of sampled z for ligand token | — |
| v4 | `modeling_novomolgen_infonce_120225_v4.py` | CFG training + null ligand token + Tanimoto loss | `bdnv2_config_finetune_augmented_v4.yaml` |
| v5 | `modeling_novomolgen_infonce_120225_v5.py` | Multi-token conditioning (protein/ligand → 4 tokens each) | none (checkpoint `training_config.yaml` only) |

---

## v1 / v2 — Baseline

**File:** `src/models/modeling_novomolgen_infonce_120225.py`

### Architecture

```
pocket_vec [B, 512] + evo_vec [B, 1280]  →  ProteinEncoder  →  protein_condition [B, z_dim]
ifp [B, 16384] + ligand_vec [B, 1536]    →  LigandEncoder   →  mu, logvar  →  z (sampled)

protein_to_hidden(protein_condition)  →  protein_token [B, 1, h]
ligand_to_hidden(z)                   →  ligand_token  [B, 1, h]
cat([protein_token, ligand_token])    →  cond_tokens   [B, 2, h]
                                              ↓
                               Cross-attention at layers [3,7,11,15,19,23,27,31]
```

### Loss

```
total_loss = NLL + β(t) × KL + λ_infonce × InfoNCE
```

- **NLL**: causal LM cross-entropy
- **KL**: KL divergence between posterior q(z|x) and N(0,I), per-token normalised
- **InfoNCE**: contrastive alignment between transformer hidden states and ligand_vec

### Problems Identified

1. **Posterior collapse**: `cond_tokens_len` defaulted to `hidden_size=768` (yaml had no `cond_tokens_len` key), giving a 768-dim VAE. With `free_bits=5`, each of 768 dimensions gets only 0.0065 nats protection — effectively zero KL pressure, pushing `mu ≈ 0`.
2. **Noisy ligand signal**: sampled `z = mu + eps * sigma` adds stochastic noise. The model learns to average it out and effectively ignores the ligand.
3. **Gate saturation**: all 8 cross-attention gates converge to raw=1.0 (sigmoid=0.7305), uniform across layers — no layer-wise specialisation.

---

## v3 — Deterministic Ligand Conditioning

**File:** `src/models/modeling_novomolgen_infonce_120225_v3.py`

### Key Change vs v2

```python
# v2 (before)
ligand_token = self.ligand_to_hidden(z).unsqueeze(1)   # z = mu + noise

# v3 (after)
ligand_token = self.ligand_to_hidden(mu).unsqueeze(1)  # mu = deterministic encoder output
```

`z` (sampled from the posterior) is still used for computing the KL loss, but the actual
conditioning signal fed into cross-attention is the deterministic posterior mean `mu`.

### Why This Matters

The model can no longer average out stochastic noise to ignore the ligand. `mu` is a
deterministic function of `(ifp, ligand_vec)`, so the cross-attention adapter must learn
to use it or actively suppress it — there is no "noise" to exploit.

### Architecture Note: Independent Tokens (Introduced in v3, Retained in v4)

Both protein and ligand are passed as **separate, independent tokens** to cross-attention:

```python
protein_token = self.protein_to_hidden(protein_condition).unsqueeze(1)  # [B, 1, h]
ligand_token  = self.ligand_to_hidden(mu).unsqueeze(1)                  # [B, 1, h]
cond_tokens   = torch.cat([protein_token, ligand_token], dim=1)         # [B, 2, h]
```

There is **no** `ConditionFusion` MLP. Each modality has its own linear projection
(`protein_to_hidden`, `ligand_to_hidden`) and the two tokens are concatenated.

This is a significant architectural difference from designs where protein and ligand
are fused into a single vector before projection.

---

## v4 — Classifier-Free Guidance (CFG)

**File:** `src/models/modeling_novomolgen_infonce_120225_v4.py`
**Config:** `configs/finetune/bdnv2_config_finetune_augmented_v4.yaml`

### Motivation

Even with `mu` (v3), the model could still fail to learn strong ligand conditioning
because there was no explicit mechanism to measure the gap between conditioned and
unconditioned generation. CFG addresses this by training both modes simultaneously.

### New Module: `null_ligand_emb`

```python
self.null_ligand_emb = nn.Parameter(torch.zeros(1, cond_latent_dim))
```

A learnable embedding representing "no ligand information". Initialised to zero.
This is the unconditional ligand token used during training dropout and inference.

### Training: CFG Dropout

During `forward()`, the ligand token is randomly replaced with `null_ligand_emb`:

```python
ligand_drop_prob = getattr(self.base_config, "ligand_drop_prob", 0.15)
if self.training and torch.rand(1).item() < ligand_drop_prob:
    lig_emb = self.null_ligand_emb.to(dtype=mu.dtype).expand(bsz, -1)  # dropped
else:
    lig_emb = mu  # conditional (real ligand)

ligand_token = self.ligand_to_hidden(lig_emb).unsqueeze(1)
```

The model therefore sees:
- `(1 - drop_prob)` fraction → `[protein_token, real_ligand_token]` (conditional)
- `drop_prob` fraction → `[protein_token, null_ligand_token]` (unconditional)

**Config: `ligand_drop_prob: 0.3`** — higher than CFG image literature (0.1–0.2) because
prior versions showed historically weak ligand conditioning, requiring more unconditional
training signal.

### Independent Tokens Enable Clean CFG

Because protein and ligand are independent tokens (not fused), the CFG amplification
at inference is clean:

```
# conditional branch:   K/V = [protein_token, real_ligand_token]
# unconditional branch: K/V = [protein_token, null_ligand_token]

logits_cfg = logits_uncond + cfg_scale × (logits_cond - logits_uncond)
```

The `protein_token` is **identical** in both branches, so it cancels out. The difference
captures the pure ligand effect. This would not be clean if protein+ligand were fused
into a single vector through an MLP.

### Inference: `CFGLogitsProcessor`

```python
class CFGLogitsProcessor:
    def __init__(self, cfg_scale, batch_size): ...
    def __call__(self, input_ids, scores):
        cond_scores   = scores[:B]
        uncond_scores = scores[B:]
        cfg_scores    = uncond_scores + self.cfg_scale * (cond_scores - uncond_scores)
        return torch.cat([cfg_scores, uncond_scores], dim=0)
```

`generate_with_condition()` doubles the batch `[cond; uncond]`, runs generation jointly,
and uses `CFGLogitsProcessor` to interpolate logits at each token step.
`cfg_scale=1.0` is equivalent to no CFG (standard conditional generation).

### KL Loss Removed

```python
# v4: no KL loss
# total_loss = NLL only
loss = nll_loss
self._last_kl_loss = 0.0
self._last_beta = 0.0
```

The ligand encoder is used purely as a deterministic encoder (mu only). The VAE
reparameterisation trick (`z = mu + eps * sigma`) and the KL divergence term are gone.
This avoids posterior collapse entirely and simplifies the loss landscape.

### Loss: Tanimoto Instead of InfoNCE

```yaml
tanimoto_weight: 0.1      # Morgan FP similarity between mu and target
infonce_weight: 0.0       # disabled
```

**Why Tanimoto over InfoNCE for CFG:**

InfoNCE is computed on transformer hidden states. For the 30% of samples where the
ligand token is dropped, the hidden states carry no ligand signal — InfoNCE would
send contradictory gradients (penalising the model for not aligning hidden states to
a ligand that was deliberately withheld).

Tanimoto loss is computed directly on `mu` from the ligand encoder, which is always
run regardless of dropout. It is independent of whether the ligand token was dropped,
so it provides a clean auxiliary loss that regularises the encoder without conflicting
with CFG training.

### Config Changes (v4 vs augmented)

| Parameter | Before (augmented) | v4 |
|-----------|-------------------|-----|
| `cond_tokens_len` | missing (defaulted to 768) | `256` |
| `ligand_drop_prob` | missing | `0.3` |
| `tanimoto_weight` | `0` | `0.1` |
| `infonce_weight` | `0.1` | `0.0` |
| `beta_kl_final` | `0.1` | `0` |
| `beta_kl_eval` | `1` | `0` |

**`cond_tokens_len: 256` fix**: Previously, the yaml had only `latent_dim: 128` which
fed only into the DataModule (dead parameter). The actual VAE latent dimension was
read from `base_config.cond_tokens_len`, which defaulted to `hidden_size=768` — the
model was training a 768-dim encoder, not 128-dim. Setting `cond_tokens_len: 256`
correctly controls the encoder bottleneck. 256 is chosen because:
- `pocket_vec` is 512-dim; a 256-dim bottleneck gives 2× compression
- 512-dim would have no compression on `pocket_vec → protein_encoder`
- 128-dim may lose too much structural information

---

## v5 — Multi-Token Conditioning

**File:** `src/models/modeling_novomolgen_infonce_120225_v5.py`
**Config:** no yaml in `configs/finetune/` — see "Config gotcha" below.

### Key Change vs v4

v4 compressed each condition into a *single* cross-attention token. v5 projects each
condition into `N` independent tokens, giving cross-attention a wider K-V space:

```python
# v4: one token per condition → cond_tokens [B, 2, h]
self.protein_to_hidden = nn.Linear(cond_latent_dim, hidden_size)
protein_token = self.protein_to_hidden(protein_condition).unsqueeze(1)

# v5: N tokens per condition → cond_tokens [B, n_p + n_l, h]
self.protein_to_hidden = nn.Linear(cond_latent_dim, n_p * hidden_size)
protein_token = self.protein_to_hidden(protein_condition).view(bsz, n_p, hidden_size)
```

Two new config fields control this:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `n_protein_tokens` | `4` | protein tokens fed to cross-attention |
| `n_ligand_tokens` | `4` | ligand tokens fed to cross-attention |

**Setting both to 1 reproduces v4 exactly** — v5 is a strict superset of v4. Everything
else carries over from v4 unchanged: CFG training, `null_ligand_emb`, `ligand_drop_prob`,
no KL loss, Tanimoto instead of InfoNCE.

A single `nn.Linear(d, N*h)` + `view` is mathematically equivalent to `N` independent
linear heads but costs one GEMM instead of `N`.

### CFG Stays Clean

Widening the condition sequence does not break the CFG subtraction. Both branches still
carry byte-identical protein tokens, so they cancel and the difference isolates the
ligand effect:

```
# conditional branch:   K/V = [protein_token × n_p, real_ligand_token × n_l]
# unconditional branch: K/V = [protein_token × n_p, null_ligand_token × n_l]
```

This holds only because protein and ligand remain *separate* tokens (no `ConditionFusion`
MLP) — the same property v3 introduced and v4 relies on.

### Config Gotcha

There is **no v5 yaml** in `configs/finetune/` (the newest are `*_v4*.yaml`), and
`n_protein_tokens` / `n_ligand_tokens` appear in **no** config file — every v5 run used
the class defaults `4 + 4`. The authoritative record of what a v5 run actually used is
the `training_config.yaml` + `config.json` saved inside each checkpoint's `full_model/`
directory. Check those, not `configs/`, when reproducing a v5 result.

### Benchmark Checkpoints (v5)

Two v5 checkpoints back the main results, one per dataset:

| Benchmark | Checkpoint | Training |
|-----------|-----------|----------|
| BDNV2 scaffold-split — de novo, lead optimization, p53 case study | `outputs/models/050826_SAFEGen_lr5e-5_cfg_tani0.1_attnall_new_aug_drop0.5_scaffold_split_v5/checkpoint-33110/full_model` | lr 5e-5, 10 epochs, `processed_data_scaffold_splitting/hf_dataset_augmented` |
| MolGenBench (CrossDocked) | `outputs/models/051826_SAFEGen_lr2e-5_cfg05_tani01_attnall_v5_xdock_augmented_epoch70/checkpoint-30730/full_model` | lr 2e-5, 70 epochs, bs 250 × ga 4, `crossdock_dataset/hf_dataset_crossdock_augmented` |

Both share: base `models/novomolgen_300M_YB/checkpoint-309520`, `cross_layers` 0–31
(all 32 layers), `cond_tokens_len: 256`, `ligand_drop_prob: 0.5`, `tanimoto_weight: 0.1`,
`infonce_weight: 0.0`, `n_protein_tokens/n_ligand_tokens = 4/4`.

**Naming trap**: the MolGenBench output directories are named `..._epoch70_ckpt34350_...`,
but their `generation_summary.json` records `checkpoint-30730` as the real `model_path`.
`checkpoint-34350` belongs to a *different* run (`051826_..._v5_xdock_augmented_b128a1`).
Trust `generation_summary.json`, not the directory name.

### Generation Scripts

| Script | Model resolution |
|--------|-----------------|
| `scripts/generate_from_encoded_v5.py` | hard-imports v5; used for encoded single-target runs (p53 notebook) |
| `scripts/generate_from_full_model_xdock.py`, `data_processing/molgenbench/generate_from_full_model_xdock.py` | auto-detects `modeling_novomolgen*.py` inside the checkpoint, falls back to v5 |
| `scripts/generate_from_full_model_cfg.py` | auto-detects from checkpoint, falls back to **v4** |

Because v5 checkpoints ship their own `modeling_novomolgen_infonce_120225_v5.py` inside
`full_model/`, the auto-detecting scripts load the correct class regardless of fallback.

### Open Inconsistency: Ablations Are Still v4

The protein/ligand ablations (`051025_..._proteinabla_v4`, `051225_..._ligandabla_v4`)
were trained on the v4 architecture, while the main results are v5. A v5 ablation model
file exists (`modeling_novomolgen_infonce_120225_v5_proteinabla.py`) but has no trained
run under `outputs/models/`. Ablation tables built from these runs are therefore not
architecture-matched to the main results.

---

## Data: SAFE Augmentation

All v3/v4/v5 BDNV2 runs use the 5× SAFE-augmented scaffold-split dataset:

```
finetune_data/processed_data_scaffold_splitting/hf_dataset_augmented/train
```

- ~681,700 base training pairs × 5 augmentations = ~3.4M effective samples
- Validation uses the non-augmented set for consistent evaluation
- Dataset split: double-cold (MMseqs2 ≤30% protein identity + Tanimoto ≥0.4 ligand filter)

---

## File Map

```
src/models/
├── modeling_novomolgen_infonce_120225.py      # v1/v2: sampled z, KL+InfoNCE
├── modeling_novomolgen_infonce_120225_v3.py   # v3: mu for ligand, separate tokens
├── modeling_novomolgen_infonce_120225_v4.py   # v4: CFG + null_ligand_emb + Tanimoto
└── modeling_novomolgen_infonce_120225_v5.py   # v5: multi-token conditioning (4+4)

configs/finetune/
├── bdnv2_config_finetune_augmented.yaml       # v2 config
├── bdnv2_config_finetune_augmented_v4.yaml    # v4 config (cond256, cfg0.3, tani0.1)
├── bdnv2_config_finetune_v4_xdock_aug.yaml    # v4 CrossDocked (augmented)
└── (no v5 yaml — v5 settings live only in each checkpoint's training_config.yaml)
```
