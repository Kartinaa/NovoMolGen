# Structure-SAFE

Conditional small-molecule generation steered by a **protein pocket** and optionally a **reference ligand**.

Structure-SAFE extends [NovoMolGen](https://github.com/chandar-lab/NovoMolGen) (a GPT-style
molecular language model generating [SAFE](https://github.com/datamol-io/safe) strings) with
cross-attention conditioning on structural features of a target. Given a pocket and a known
active, it generates new molecules for that pocket — de novo, or seeded with a SAFE prefix for
lead-optimization subtasks (scaffold hopping, fragment growing, linker design).

---

## 1. Installation

The encoding pipeline depends on **four mutually incompatible toolchains** (Uni-Mol pins old
`unicore`/torch, ODDT pins old numpy, ESM-2 and the generation stack want current torch). There
is no single environment that runs all of it, so create four conda envs and run each step in the
matching one. Intermediate `.npy` files on disk are the handoff between steps.

Prerequisites: Python 3.10+, CUDA 11.8+ (12.2 tested), one 40 GB+ GPU for training.

### Env 1 — `safegen` (generation + training)

```bash
conda create -n safegen -c conda-forge -c rdkit python=3.10 rdkit openbabel
conda activate safegen
pip install -r requirements.txt
pip install safe-mol
# Flash-Attention (required: the model config sets use_flash_attn=true)
export CUDA_HOME=/usr/local/cuda-12.2 MAX_JOBS=8
pip install flash-attn --no-build-isolation
```

### Env 2 — `unimol` (`pocket_vec`, `ligand_vec`)

```bash
conda create -n unimol python=3.9
conda activate unimol
pip install unimol_tools rdkit lmdb pandas scipy tqdm
```

`pocket_vec` additionally needs a clone of the Uni-Mol repo and two of its assets:

```bash
git clone https://github.com/deepmodeling/Uni-Mol.git
# install Uni-Core per that repo's instructions, then locate:
#   unimol/example_data/pocket/dict_coarse.txt     → DICT_FILE
#   pocket_pre_220816.pt (pocket pretraining ckpt) → WEIGHTS
```

Then **edit [utils/build_pocket_lmdb_and_infer.py:298](utils/build_pocket_lmdb_and_infer.py#L298)**
to point at your Uni-Mol clone — the path is currently hardcoded.

### Env 3 — `esm` (`evo_vec`)

```bash
conda create -n esm python=3.10
conda activate esm
pip install fair-esm torch biopython
```

ESM-2 weights (`esm2_t33_650M_UR50D`) download automatically on first use.

### Env 4 — `oddt` (`ifp`)

```bash
conda create -n oddt -c conda-forge python=3.9 openbabel
conda activate oddt
pip install oddt rdkit numpy
```

---

## 2. Download the checkpoints

Both are on [Google Drive](https://drive.google.com/drive/folders/1fQCoXZx7EAqkPMqC_NevEaLD14R_iN5Q?usp=drive_link).
Which one you need depends on what you are doing:

| File | What it is | Needed for |
|---|---|---|
| `full_model.zip` | Fine-tuned model for protein- (optionally ligand-) conditioned generation | Generating molecules — §3 |
| `checkpoint-309520.zip` | NovoMolGen backbone, pretrained on 1.5B SAFE strings from ZINC22 | Training your own model — §4 |

**To generate**, unzip the fine-tuned model anywhere you like:

```bash
mkdir -p outputs/models/structure_safe_v5
unzip full_model.zip -d outputs/models/structure_safe_v5
# → outputs/models/structure_safe_v5/full_model/
```

Pass that `full_model/` directory wherever a model path is required.

**To train**, unzip the backbone to the location the config already points at:

```bash
mkdir -p models/novomolgen_300M_YB
unzip checkpoint-309520.zip -d models/novomolgen_300M_YB
# → models/novomolgen_300M_YB/checkpoint-309520/  (= pretrained_path in the v5 config)
```

> Load either checkpoint **through this repo**, not with `AutoModelForCausalLM`. Their
> `config.json` carries a stale `auto_map`, so `from_pretrained(..., trust_remote_code=True)`
> fails; `scripts/generate_from_encoded_v5.py` imports the model class directly and handles it.

---

## 3. Generate molecules for your target

[`scripts/encode_and_generate.ipynb`](scripts/encode_and_generate.ipynb) takes a **protein PDB +
ligand SDF** and produces SMILES. Set your paths in **cell 1**
(`LIGAND_SDF_PATH`, `PROTEIN_PDB_PATH`, `OUTPUT_DIR`, `DICT_FILE`, `WEIGHTS`, `MODEL_PATH`,
`POCKET_RADIUS`, `NUM_SAMPLES`) — they currently hold absolute paths from the author's machine
and must be changed. Then run each step with the matching conda kernel, switching kernels as you go:

| Step | Cell | Env | Writes |
|---|---|---|---|
| 1. Extract pocket | 3 | `safegen` | `pocket.pdb`, `receptor.pdb` |
| 2. `ligand_vec` | 5 | `unimol` | `ligand_vec.npy` (1536-d) |
| 3. `pocket_vec` | 7 | `unimol` | `pocket_vec.npy` (512-d) |
| 4. `evo_vec` | 9 | `esm` | `evo_vec.npy` (1280-d) |
| 5a. Sanitize SDF | 14 | `safegen` | `ligand_sanitized.sdf` |
| 5b. `ifp` | 15 | `oddt` | `ifp.npy` (16384-d) |
| 5.5. SAFE prefix | 17 | `safegen` | sets `SAFE_PREFIX` |
| 6. Generate | 19 | `safegen` | `generated_molecules_nopoint.smi` |

Step 5.5 offers five `PREFIX_MODE` values (`de_novo`, `full_prefix`, …) choosing whether
generation starts from scratch or grows from a fragment of the reference ligand. Step 6 picks how
strongly the reference ligand steers generation:

| Mode | `PROTEIN_ONLY` | `CFG_SCALE` |
|---|---|---|
| Ligand-aware + classifier-free guidance (paper default) | `False` | `1.5` |
| Ligand-aware, no guidance | `False` | `1.0` |
| Pocket-only | `True` | forced to `1.0` |

### Command line

Steps 1–5 must still be run to produce the condition vectors, but step 6 has a CLI equivalent:

```bash
conda activate safegen
python scripts/generate_from_encoded_v5.py \
  --model_path outputs/models/structure_safe_v5/full_model \
  --encoded_dir <your OUTPUT_DIR from cell 1> \
  --num_samples 10000 \
  --cfg_scale 1.5 \
  --output_dir generated
```

Add `--safe_prefix 'c1ccc(-c2nc3ccccc3s2)cc1.'` for lead optimization, or
`--protein_only --cfg_scale 1.0` for pocket-only generation.

---

## 4. Train on your own data

### Build a conditioned dataset

Every row needs the molecule plus all four condition vectors. Run
[scripts/data_processing/](scripts/data_processing/) in order, each in its env:

```bash
conda activate safegen && python scripts/data_processing/01_create_base_dataset.py
conda activate unimol  && python scripts/data_processing/02_compute_pocket_vec_auto_optimized.py
conda activate esm     && python scripts/data_processing/03_compute_evo_vec.py
conda activate unimol  && python scripts/data_processing/04_compute_ligand_vec.py
conda activate oddt    && python scripts/data_processing/04_1_compute_ifp.py
conda activate safegen && python scripts/data_processing/05_merge_ifp_with_dataset.py

# 5x SAFE augmentation — the v5 config expects the augmented train split
python scripts/data_processing/10_augment_safe_representations.py \
  --input_path  <dataset>/train \
  --output_path <dataset>_augmented/train \
  --num_versions 5 --seed 42
```

Augment the training split only; keep validation unaugmented so evaluation stays comparable.
Field-by-field format reference:
[md_cursor/CONDITION_FEATURES_GUIDE.md](md_cursor/CONDITION_FEATURES_GUIDE.md). Check your
dataset before training:

```python
from tests.verify_condition_features import verify_condition_features
verify_condition_features(your_dataset)
```

### Fine-tune

```bash
conda activate safegen
python scripts/finetune_full.py \
  --config configs/finetune/bdnv2_config_finetune_augmented_v5.yaml \
  --output_dir outputs/models/my_v5_run \
  --run_name my_v5_run
```

Resume with `--resume_from_checkpoint outputs/models/my_v5_run/checkpoint-3000`.
Point `dataset_name` and `validation_set_names` in the config at your own splits.

**Keep `n_protein_tokens` in your config.** The model class is selected from the config, not a
flag: [finetune_full.py:67](scripts/finetune_full.py#L67) uses the v5 model only when
`n_protein_tokens` is present. Remove it and you train a different architecture.

Settings worth knowing in
[configs/finetune/bdnv2_config_finetune_augmented_v5.yaml](configs/finetune/bdnv2_config_finetune_augmented_v5.yaml):

| Key | Value | Meaning |
|---|---|---|
| `pretrained_path` | `your_path/checkpoint-309520` | Pretrained NovoMolGen backbone, 32 layers |
| `cross_layers` | `[0 … 31]` | Cross-attention on every block |
| `n_protein_tokens` / `n_ligand_tokens` | `4` / `4` | Condition tokens per modality |
| `cond_tokens_len` | `256` | Condition width |
| `ligand_drop_prob` | `0.5` | Classifier-free-guidance dropout rate |
| `tanimoto_weight` | `0.1` | Weight of the Tanimoto alignment loss |
| `learning_rate` | `5e-5` | cosine schedule, `warmup_ratio: 0.1` |
| `per_device_train_batch_size` | `250` | ×4 gradient accumulation = 1000 total |
| `num_train_epochs` | `10` | 10 epochs on the 5× augmented split |

Total loss is `NLL + tanimoto_weight × Tanimoto`. Training reports to W&B; set
`wandb_logs: false` to disable.

---

