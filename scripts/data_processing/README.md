# Data Processing Pipeline for BDNV2 Dataset

This directory contains scripts to process your BDNV2 dataset and prepare it for NovoMolGen finetuning with condition features.

## Overview

The pipeline processes your data in 6 steps:

1. **Create base dataset** - Load pickle files and CSV, create initial Parquet dataset
2. **Compute pocket vectors** - Extract Uni-Mol pocket embeddings (512D)
3. **Compute evolutionary vectors** - Extract ESM-2 evolutionary embeddings (1280D)
4. **Compute interaction fingerprints** - Calculate protein-ligand interaction fingerprints (16384D)
5. **Compute ligand vectors** - Extract ligand molecular representations (1536D)
6. **Create final dataset** - Combine everything into HuggingFace dataset format

## Data Format

### Input Data Structure
- `training_set_random_splitting.pkl`: List of tuples (protein_path, ligand_path)
- `validation_set_random_splitting.pkl`: List of tuples (protein_path, ligand_path)
- `bdnv2_standardize.csv`: Mapping from (protein_path, ligand_path) to standardize_smi

### Output Data Format
Final dataset will be in HuggingFace datasets format with:
- **Training set**: ~688K samples
- **Validation set**: ~1K samples
- **Features**:
  - `standardize_smi`: Standardized SMILES string
  - `pocket_vec`: Uni-Mol pocket embedding (512 dimensions)
  - `evo_vec`: ESM-2 evolutionary embedding (1280 dimensions)
  - `ifp`: Interaction fingerprint (16384 dimensions)
  - `ligand_vec`: Ligand molecular representation (1536 dimensions)

## Usage

### Option 1: Run All Steps at Once
```bash
cd /home/yang2531/Documents/Project/Structure_safe
python scripts/data_processing/run_all_steps.py \
  --data_dir finetune_data \
  --output_dir processed_data
```

### Option 2: Run Steps Individually
```bash
# Step 1: Create base dataset
python scripts/data_processing/01_create_base_dataset.py \
  --data_dir finetune_data \
  --output_dir processed_data

# Step 2: Compute pocket vectors
python scripts/data_processing/02_compute_pocket_vec.py \
  --input_file processed_data/base_dataset.parquet \
  --output_dir processed_data

# Step 3: Compute evolutionary vectors
python scripts/data_processing/03_compute_evo_vec.py \
  --input_file processed_data/dataset_with_pocket_vec.parquet \
  --output_dir processed_data

# Step 4: Compute interaction fingerprints
python scripts/data_processing/04_compute_ifp.py \
  --input_file processed_data/dataset_with_evo_vec.parquet \
  --output_dir processed_data

# Step 5: Compute ligand vectors
python scripts/data_processing/05_compute_ligand_vec.py \
  --input_file processed_data/dataset_with_ifp.parquet \
  --output_dir processed_data

# Step 6: Create final dataset
python scripts/data_processing/06_create_final_dataset.py \
  --input_file processed_data/dataset_with_ligand_vec.parquet \
  --output_dir processed_data
```

### Option 3: Resume from Specific Step
```bash
# Resume from step 3 (if steps 1-2 already completed)
python scripts/data_processing/run_all_steps.py \
  --data_dir finetune_data \
  --output_dir processed_data \
  --start_from 3
```

### Option 4: Skip Specific Steps
```bash
# Skip steps 2 and 4 (if you want to use placeholder vectors)
python scripts/data_processing/run_all_steps.py \
  --data_dir finetune_data \
  --output_dir processed_data \
  --skip_steps 2 4
```

## Environment Requirements

Each step may require different dependencies:

- **Step 1**: Basic Python (pandas, pyarrow)
- **Step 2**: Uni-Mol tools for pocket embedding
- **Step 3**: ESM-2 tools for evolutionary embedding
- **Step 4**: IFP computation tools
- **Step 5**: Ligand embedding tools
- **Step 6**: HuggingFace datasets

## Output Files

After running all steps, you'll have:

```
processed_data/
├── base_dataset.parquet                    # Step 1 output
├── dataset_with_pocket_vec.parquet         # Step 2 output
├── dataset_with_evo_vec.parquet            # Step 3 output
├── dataset_with_ifp.parquet                # Step 4 output
├── dataset_with_ligand_vec.parquet         # Step 5 output
├── final_dataset/                          # Step 6 output (HuggingFace format)
│   ├── train/
│   ├── validation/
│   └── dataset_info.json
├── dataset_metadata.json                   # Dataset metadata
└── sample_data.json                        # Sample data for inspection
```

## Uploading to HuggingFace

After processing, you can upload to HuggingFace:

```python
from datasets import load_from_disk
from huggingface_hub import HfApi

# Load the dataset
dataset = load_from_disk("processed_data/final_dataset")

# Upload to HuggingFace
api = HfApi()
api.upload_folder(
    folder_path="processed_data/final_dataset",
    repo_id="your-username/bdnv2-finetune-dataset",
    repo_type="dataset"
)
```

## Integration with Finetuning

Once uploaded, you can use the dataset in your finetuning config:

```yaml
# configs/finetune/bdnv2_config.yaml
pretrained_path: "/path/to/your/checkpoint"
dataset_name: "your-username/bdnv2-finetune-dataset"
include_condition_features: true
mol_type: "SMILES"
max_length: 128
# ... other config options
```

## Troubleshooting

### Memory Issues
- Reduce batch size in individual steps
- Process data in chunks if needed
- Use streaming for very large datasets

### Missing Dependencies
- Each step will use placeholder vectors if tools are not available
- Check the import statements in each script
- Install required packages for real feature extraction

### File Path Issues
- Ensure all file paths are correct
- Check that pickle files and CSV exist
- Verify output directory permissions

## Next Steps

After processing your data:

1. **Test the dataset**: Load and inspect the final dataset
2. **Upload to HuggingFace**: Make it available for finetuning
3. **Update finetune config**: Point to your new dataset
4. **Run finetuning**: Use the processed dataset for training

