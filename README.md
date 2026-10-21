# Structure-SAFE

Conditional small-molecule generation steered by a **protein pocket** and a **reference ligand**.

Structure-SAFE extends [NovoMolGen](https://github.com/chandar-lab/NovoMolGen) (a GPT-style
molecular language model generating [SAFE](https://github.com/datamol-io/safe) strings) with
cross-attention conditioning on structural features of a target. Given a pocket and a known
active, it generates new molecules for that pocket — de novo, or seeded with a SAFE prefix for
lead-optimization subtasks (scaffold hopping, fragment growing, linker design).

> **Branches.** This is the development branch, which keeps every model variant (v1–v5,
> ablations, CrossDocked/xdock experiments) and the upstream reinforcement-learning stack.
> The **`release-v5`** branch is a curated subset containing only what is needed to train
> and run the v5 CFG/Tanimoto model — point collaborators there.

## How conditioning works (one paragraph)

Four feature vectors describe the target. `pocket_vec` (512-d, Uni-Mol) and `evo_vec` (1280-d,
ESM-2) go through a `ProteinConditionEncoder`; `ifp` (16384-d, PLEC interaction fingerprint) and
`ligand_vec` (1536-d, Uni-Mol) go through a `LigandConditionEncoder`. Both are projected into a
small number of **condition tokens** (4 + 4 in v5) that every transformer block attends to via
gated cross-attention adapters. The current model (v5) is trained with **classifier-free guidance**:
the ligand token is replaced by a learnable null embedding with probability `ligand_drop_prob`,
so at inference `cfg_scale > 1` amplifies the ligand signal. Full details in
[README_to_Tak.md](README_to_Tak.md).

---

## Installation

### The important constraint

The encoding pipeline depends on **four mutually incompatible toolchains** (Uni-Mol pins old
`unicore`/torch, ODDT pins old numpy, ESM-2 and the generation stack want current torch). There
is no single environment that runs all of it. Use **separate conda envs** and pass data between
steps as `.npy` files on disk — that is exactly why
[`scripts/encode_and_generate.ipynb`](scripts/encode_and_generate.ipynb) saves an intermediate
file after every step and documents a required environment per cell.

### Env 1 — `safegen` (training + generation + base utilities)

```bash
conda create -n safegen -c conda-forge -c rdkit python=3.10 rdkit openbabel
conda activate safegen
pip install -r requirements.txt
# Flash-Attention (required: config sets use_flash_attn=true)
export CUDA_HOME=/usr/local/cuda-12.2 MAX_JOBS=8
pip install flash-attn --no-build-isolation
pip install safe-mol
```

Prerequisites: Python 3.10+, CUDA 11.8+ (12.2 tested), one 40 GB+ GPU for training.
Key pins come from [requirements.txt](requirements.txt): `transformers==4.46.2`,
`accelerate==0.34.0`, `datasets==2.20.0`.

### Env 2 — `unimol` (pocket_vec, ligand_vec)

```bash
conda create -n unimol python=3.9
conda activate unimol
pip install unimol_tools rdkit lmdb pandas scipy tqdm
```

Plus, for `pocket_vec` only, a **clone of the Uni-Mol repo** and two downloaded assets:

```bash
git clone https://github.com/deepmodeling/Uni-Mol.git
# install Uni-Core per that repo's instructions, then fetch:
#   unimol/example_data/pocket/dict_coarse.txt    → DICT_FILE
#   pocket_pre_220816.pt (pocket pretraining ckpt) → WEIGHTS
```

> **Known rough edge:** the Uni-Mol clone location is hardcoded at
> [utils/build_pocket_lmdb_and_infer.py:298](utils/build_pocket_lmdb_and_infer.py#L298)
> (`/home/yang2531/Documents/Softwares/Uni-Mol/unimol`). Edit that line to your path.

### Env 3 — `esm` (evo_vec)

```bash
conda create -n esm python=3.10
conda activate esm
pip install fair-esm torch biopython
```

The ESM-2 weights (`esm2_t33_650M_UR50D`) download automatically on first use into
`~/.cache/torch/hub/checkpoints`.

### Env 4 — `oddt` (ifp)

```bash
conda create -n oddt -c conda-forge python=3.9 openbabel
conda activate oddt
pip install oddt rdkit numpy
```

---

## Running the encoding → generation pipeline

[`scripts/encode_and_generate.ipynb`](scripts/encode_and_generate.ipynb) takes a **protein PDB +
ligand SDF** and produces generated SMILES. Set paths once in **cell 1**, then run each step
**with the matching conda kernel** — switch kernels between steps; intermediate `.npy` files are
the handoff.

| Step | Cell | Env | Needs | Writes |
|---|---|---|---|---|
| 1. Extract pocket | 3 | `safegen` | scipy, pandas, rdkit | `pocket.pdb`, `receptor.pdb` |
| 2. `ligand_vec` | 5 | `unimol` | `unimol_tools`, rdkit | `ligand_vec.npy` (1536) |
| 3. `pocket_vec` | 7 | `unimol` | Uni-Mol clone, `DICT_FILE`, `WEIGHTS`, lmdb | `pocket_vec.npy` (512) |
| 4. `evo_vec` | 9 | `esm` | `fair-esm`, biopython, torch | `evo_vec.npy` (1280) |
| 5a. Sanitize SDF | 14 | `safegen` | rdkit | `ligand_sanitized.sdf` |
| 5b. `ifp` | 15 | `oddt` | `oddt` (PLEC) | `ifp.npy` (16384) |
| 5.5. SAFE prefix | 17 | `safegen` | `safe-mol`, rdkit | sets `SAFE_PREFIX` |
| 6. Generate | 19 | `safegen` | torch, transformers, flash-attn | `generated_molecules_nopoint.smi` |

Cell 1 variables to set: `LIGAND_SDF_PATH`, `PROTEIN_PDB_PATH`, `OUTPUT_DIR`, `DICT_FILE`,
`WEIGHTS`, `MODEL_PATH`, `POCKET_RADIUS` (default 10 Å), `NUM_SAMPLES`.

### Generation modes

Step 5.5 offers five `PREFIX_MODE` values (`de_novo`, `full_prefix`, …) controlling whether
generation starts from scratch or grows from a chosen fragment of the reference ligand.
Step 6 then picks a conditioning mode:

| Mode | `PROTEIN_ONLY` | `CFG_SCALE` |
|---|---|---|
| Ligand-aware + CFG (paper default) | `False` | `1.5` |
| Ligand-aware, no guidance | `False` | `1.0` |
| Pocket-only | `True` | forced to `1.0` |

### CLI equivalent (skip the notebook for step 6)

```bash
conda activate safegen
python scripts/generate_from_encoded_v5.py \
  --model_path outputs/models/<RUN>/checkpoint-33110/full_model \
  --encoded_dir p53/encode_p53 \
  --num_samples 10000 \
  --cfg_scale 1.5 \
  --output_dir p53/generated
```

Add `--safe_prefix 'c1ccc(-c2nc3ccccc3s2)cc1.'` for lead optimization, or
`--protein_only --cfg_scale 1.0` for pocket-only.

---

## Training

### 1. Build a conditioned dataset

Each dataset row needs the molecule plus all four condition vectors. Run
[scripts/data_processing/](scripts/data_processing/) in order:

```bash
conda activate safegen && python scripts/data_processing/01_create_base_dataset.py
conda activate unimol  && python scripts/data_processing/02_compute_pocket_vec_auto_optimized.py
conda activate esm     && python scripts/data_processing/03_compute_evo_vec.py
conda activate unimol  && python scripts/data_processing/04_compute_ligand_vec.py
conda activate oddt    && python scripts/data_processing/04_1_compute_ifp.py
conda activate safegen && python scripts/data_processing/05_merge_ifp_with_dataset.py

# 5x SAFE augmentation (v5 configs expect the augmented train split)
python scripts/data_processing/10_augment_safe_representations.py \
  --input_path  finetune_data/processed_data_scaffold_splitting/hf_dataset/train \
  --output_path finetune_data/processed_data_scaffold_splitting/hf_dataset_augmented/train \
  --num_versions 5 --seed 42
```

Validation splits are deliberately **not** augmented. Format reference:
[md_cursor/CONDITION_FEATURES_GUIDE.md](md_cursor/CONDITION_FEATURES_GUIDE.md). Sanity-check with:

```python
from tests.verify_condition_features import verify_condition_features
verify_condition_features(your_dataset)
```

### 2. Fine-tune

```bash
conda activate safegen
python scripts/finetune_full.py \
  --config configs/finetune/bdnv2_config_finetune_augmented_v5.yaml \
  --output_dir outputs/models/my_v5_run \
  --run_name my_v5_run
```

Resume with `--resume_from_checkpoint outputs/models/my_v5_run/checkpoint-3000`.

**Which model class gets used is keyed off the config, not a flag.**
[scripts/finetune_full.py:67](scripts/finetune_full.py#L67) selects the v5 model if and only if
`n_protein_tokens` is present in the YAML — omit it and you silently train v4 instead. The
explicit `n_protein_tokens: 4` / `n_ligand_tokens: 4` in the v5 config exist for this reason.

Settings worth knowing in [configs/finetune/bdnv2_config_finetune_augmented_v5.yaml](configs/finetune/bdnv2_config_finetune_augmented_v5.yaml):

| Key | v5 value | Meaning |
|---|---|---|
| `pretrained_path` | `models/novomolgen_300M_YB/checkpoint-309520` | Frozen-start NovoMolGen backbone (32 layers) |
| `cross_layers` | `[0 … 31]` | Cross-attention on every block |
| `n_protein_tokens` / `n_ligand_tokens` | `4` / `4` | Condition tokens per modality |
| `cond_tokens_len` | `256` | Condition/latent width |
| `ligand_drop_prob` | `0.5` | CFG dropout rate |
| `tanimoto_weight` | `0.1` | Tanimoto loss on `mu` (replaces InfoNCE under CFG) |
| `beta_kl_final` | `0` | **KL/VAE disabled in v4/v5** |
| `learning_rate` | `5e-5` | cosine, `warmup_ratio: 0.1` |
| `per_device_train_batch_size` | `250` | ×4 grad accum = 1000 total |
| `num_train_epochs` | `10` | 10 on 5× augmented ≈ 50 on raw |

Total loss: `NLL + tanimoto_weight × Tanimoto` (v5). Earlier variants used
`NLL + β(t)·KL + λ·InfoNCE` — the `vae:` and `infonce_*` blocks are kept only for those.

### 3. Monitor

Training reports to W&B (`report_to: ["wandb"]`, disable with `wandb_logs: false`). Gate values
show whether cross-attention is actually being used:

```python
for layer_idx in model.transformer.cross_layers:
    print(layer_idx, torch.sigmoid(model.transformer.gates[str(layer_idx)]).item())
model.print_trainable_summary(max_lines=30)
```

## Checkpoints — what to release

### Release `full_model/`, not the checkpoint directory

A training checkpoint is ~3.0 GB, but most of that is resume-only state
(`optimizer.pt`, `scheduler.pt`, `rng_state.pth`, a duplicate `pytorch_model.bin`). The nested
`full_model/` subdirectory is the **self-contained 753 MB inference bundle**:

```
checkpoint-33110/full_model/
├── pytorch_model.bin                        753M   weights
├── config.json                              1.8K   architecture + CFG/Tanimoto settings
├── modeling_novomolgen_infonce_120225_v5.py  72K   model code, bundled
├── tokenizer.json, tokenizer_config.json, special_tokens_map.json
└── training_config.yaml                     5.9K   provenance of the run
```

### Download

The v5 checkpoint is distributed via **Google Drive**:

> **Weights:** <!-- TODO: paste the Google Drive share link here -->
> `structure-safe-v5-checkpoint-33110.tar.gz` (753 MB)

```bash
# after downloading the archive
mkdir -p outputs/models/structure_safe_v5
tar -xzf structure-safe-v5-checkpoint-33110.tar.gz -C outputs/models/structure_safe_v5
# → outputs/models/structure_safe_v5/full_model/
```

Then point `--model_path` (or the notebook's `MODEL_PATH`) at the extracted `full_model/`
directory.

### Packaging it for upload

```bash
cd "outputs/models/050826_SAFEGen_lr5e-5_cfg_tani0.1_attnall_new_aug_drop0.5_scaffold_split_v5/checkpoint-33110"
tar -czf ~/structure-safe-v5-checkpoint-33110.tar.gz full_model/
```

Upload that single archive to Drive and set the share link to "anyone with the link". One
archive beats a folder of loose files: Drive does not compress folder downloads reliably, and a
753 MB `.bin` downloaded on its own is easy to truncate silently.

**Do not commit weights to git** — `outputs/` is already in [.gitignore](.gitignore), and the
repo is only ~82 MB packed precisely because weights live outside it.

Alongside the weights, record: which run produced them, the config used, the pretrained backbone
(`models/novomolgen_300M_YB/checkpoint-309520`), and the training/validation split. `training_config.yaml`
inside the bundle already covers most of this.

> **Loading caveat:** `config.json` has a stale `auto_map` pointing at
> `modeling_novomolgen.NovoMolGen`, so plain
> `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` will **not** work. The
> checkpoint must be loaded through this repo — `scripts/generate_from_encoded_v5.py` imports
> `src/models/modeling_novomolgen_infonce_120225_v5.py` directly. This is expected, not a
> packaging error.

### The backbone, separately

`models/novomolgen_300M_YB/` (3.4 GB) is the pretrained NovoMolGen backbone needed to *train* from
scratch, not to run inference. Point users at the
[upstream HF collection](https://huggingface.co/collections/chandar-lab/novomolgen-681bce8b0e73b5dc7a3b0ff1)
rather than re-hosting it, unless yours diverges from theirs.

---

## Repository layout

```
configs/finetune/        Fine-tuning configs (bdnv2_config_finetune_augmented_v5.yaml = current)
src/models/
  modeling_novomolgen_infonce_120225_v5.py   v5 model: multi-token CFG + Tanimoto
  condition/protein_encoder.py               pocket_vec + evo_vec → protein condition
  condition/ligand_encoder.py                ifp + ligand_vec → mu/sigma
  condition/condition_fusion.py              protein + ligand → fused condition
scripts/
  finetune_full.py                 Training entry point
  generate_from_encoded_v5.py      Generation from encoded vectors (v5 + CFG)
  encode_and_generate.ipynb        End-to-end notebook: PDB+SDF → molecules
  data_processing/                 Numbered dataset-build pipeline
utils/                             Encoders (Uni-Mol pocket, ESM-2, PLEC IFP)
outputs/models/<run>/              Checkpoints (gitignored)
```

## Testing

```bash
conda activate safegen
pytest tests/
```

## Known rough edges

- `utils/build_pocket_lmdb_and_infer.py:298` hardcodes a local Uni-Mol path.
- `scripts/encode_and_generate.ipynb` cell 1 hardcodes absolute `/home/yang2531/...` paths; change them after cloning.
- `configs/finetune/` contains editor backup files (`*.yaml~`) and a stale `eval_stieps` typo in the v5 config (the misspelled key is ignored, so evaluation falls back to the default interval).
- The v5 `full_model/config.json` `auto_map` is stale (see loading caveat above).

