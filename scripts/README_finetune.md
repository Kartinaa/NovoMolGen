# NovoMolGen Finetuning Script

This document explains how to use the `finetune_full.py` script for fine-tuning NovoMolGen models with cross-attention and LoRA support.

## Overview

The `finetune_full.py` script provides a comprehensive solution for fine-tuning NovoMolGen models with:
- **Cross-attention adapters** for conditional generation
- **LoRA (Low-Rank Adaptation)** for parameter-efficient fine-tuning
- **Parameter freezing** to train only new modules while keeping the base model frozen
- **Comprehensive logging** with Weights & Biases support
- **Flexible data loading** with support for both real datasets and dummy data

## Quick Start

### 1. Basic Usage

```bash
python scripts/finetune_full.py \
  --config configs/finetune/test_config.yaml \
  --output_dir outputs/my_experiment \
  --run_name "my_finetune_run"
```

### 2. With Resume from Checkpoint

```bash
python scripts/finetune_full.py \
  --config configs/finetune/test_config.yaml \
  --output_dir outputs/my_experiment \
  --run_name "my_finetune_run" \
  --resume_from_checkpoint outputs/my_experiment/checkpoint-100
```

## Configuration

The script uses YAML configuration files. Here's the structure:

```yaml
# Model configuration
pretrained_path: "bisectgroup/NovoMolGen_32M_SAFE_BPE"  # HuggingFace model ID or local path
enable_cross_attn: true                                  # Enable cross-attention adapters
cross_layers: [4, 8]                                     # Which transformer layers to add cross-attention
train_new_modules_only: true                             # Freeze base model, train only new modules
enable_lora: true                                        # Enable LoRA adapters
lora_r: 8                                                # LoRA rank
lora_alpha: 16                                           # LoRA alpha
lora_dropout: 0.05                                       # LoRA dropout

# Data configuration
dataset_name: "dummy"                                    # "dummy" for testing, or your dataset name
tokenizer_path: "data/tokenizers/tokenizer_bpe_None_SAFE_500_0_0.1.json"
mol_type: "SAFE"                                         # Molecule type: SAFE, SMILES, SELFIES
max_length: 128                                          # Maximum sequence length
include_condition_features: true                         # Include protein/ligand condition features

# Training configuration
seed: 2025                                               # Random seed
save_path: './save_finetune'                            # Save directory
learning_rate: 5e-5                                      # Learning rate
per_device_train_batch_size: 2                          # Batch size per device
gradient_accumulation_steps: 2                          # Gradient accumulation steps
weight_decay: 0.01                                       # Weight decay
max_grad_norm: 1.0                                       # Gradient clipping
warmup_ratio: 0.1                                        # Warmup ratio
bf16: true                                               # Use bfloat16 precision
num_train_epochs: 1                                      # Number of training epochs
logging_steps: 5                                         # Log every N steps
save_steps: 10                                           # Save checkpoint every N steps
save_spec_steps: 10                                      # Save special checkpoints every N steps
push_spec_checkpoints: false                             # Push checkpoints to HuggingFace Hub
repo_id: "MolGen"                                        # HuggingFace Hub repository ID
total_batch_size: 4                                      # Total batch size (per_device * accumulation)
wandb_logs: false                                        # Enable Weights & Biases logging
```

## Command Line Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--config` | str | Yes | Path to YAML configuration file |
| `--output_dir` | str | Yes | Output directory for checkpoints and logs |
| `--run_name` | str | No | Run name for logging and identification |
| `--resume_from_checkpoint` | str | No | Path to checkpoint to resume training from |
| `--log_level` | str | No | Logging level (DEBUG, INFO, WARNING, ERROR) |

## Features

### 1. Cross-Attention Adapters

Cross-attention adapters enable conditional generation by allowing the model to attend to condition features (protein/ligand information):

```yaml
enable_cross_attn: true
cross_layers: [4, 8]  # Add cross-attention at layers 4 and 8
```

### 2. LoRA (Low-Rank Adaptation)

LoRA adds small, trainable matrices to existing layers for parameter-efficient fine-tuning:

```yaml
enable_lora: true
lora_r: 8              # Rank of LoRA matrices
lora_alpha: 16         # LoRA scaling factor
lora_dropout: 0.05     # LoRA dropout rate
```

### 3. Parameter Freezing

When `train_new_modules_only: true`, the script:
- Freezes all base model parameters
- Only trains new modules (cross-attention adapters, LoRA, encoders)
- Significantly reduces memory usage and training time

### 4. Data Loading

The script supports two data modes:

#### Dummy Data (for testing)
```yaml
dataset_name: "dummy"
include_condition_features: true
```

#### Real Dataset
```yaml
dataset_name: "your_dataset_name"
tokenizer_path: "path/to/tokenizer.json"
mol_type: "SAFE"
```

### 5. Logging

#### Local Logging
- Console output with real-time progress
- Log files in `output_dir/logs/finetune.log`
- Detailed parameter summaries

#### Weights & Biases (Optional)
```yaml
wandb_logs: true
```
Enables cloud-based experiment tracking with:
- Interactive loss curves
- Hyperparameter tracking
- Model versioning
- Team collaboration

## Output Structure

The script creates the following output structure:

```
outputs/my_experiment/
├── logs/
│   └── finetune.log              # Training logs
├── checkpoint-10/                 # Regular checkpoints
├── checkpoint-20/
├── ...
├── checkpoint-225/
├── tmp-spec-checkpoint-10/        # Special checkpoints
├── tmp-spec-checkpoint-20/
├── ...
├── final_model/                   # Final trained model
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── tokenizer.json
│   └── ...
├── training_args.bin              # Training configuration
└── README.md                      # Model card
```

## Examples

### Example 1: Test Run with Dummy Data

```bash
# Create a test config
cat > configs/finetune/test.yaml << EOF
pretrained_path: "bisectgroup/NovoMolGen_32M_SAFE_BPE"
enable_cross_attn: true
cross_layers: [4, 8]
train_new_modules_only: true
enable_lora: true
lora_r: 8
dataset_name: "dummy"
include_condition_features: true
learning_rate: 5e-5
per_device_train_batch_size: 2
num_train_epochs: 1
wandb_logs: false
EOF

# Run training
python scripts/finetune_full.py \
  --config configs/finetune/test.yaml \
  --output_dir outputs/test_run \
  --run_name "test_run"
```

### Example 2: Production Run with Real Data

```bash
# Create a production config
cat > configs/finetune/production.yaml << EOF
pretrained_path: "bisectgroup/NovoMolGen_32M_SAFE_BPE"
enable_cross_attn: true
cross_layers: [4, 8, 12]
train_new_modules_only: true
enable_lora: true
lora_r: 16
lora_alpha: 32
dataset_name: "ZINC_1B_safe_atomwise"
tokenizer_path: "data/tokenizers/tokenizer_bpe_None_SAFE_500_0_0.1.json"
mol_type: "SAFE"
max_length: 256
include_condition_features: true
learning_rate: 2e-5
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
num_train_epochs: 3
save_steps: 500
wandb_logs: true
EOF

# Run training
python scripts/finetune_full.py \
  --config configs/finetune/production.yaml \
  --output_dir outputs/production_run \
  --run_name "production_finetune"
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `per_device_train_batch_size`
   - Increase `gradient_accumulation_steps`
   - Use `bf16: true` for mixed precision

2. **Training Loss is NaN**
   - Reduce `learning_rate`
   - Check `max_grad_norm` (try 0.5)
   - Ensure `bf16: true` for numerical stability

3. **Import Errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check that `peft` is installed for LoRA support

4. **Tokenizer Issues**
   - Verify `tokenizer_path` points to a valid tokenizer file
   - Ensure tokenizer matches the model's expected format

### Debug Mode

Run with debug logging to see detailed information:

```bash
python scripts/finetune_full.py \
  --config configs/finetune/test_config.yaml \
  --output_dir outputs/debug_run \
  --log_level DEBUG
```

## Performance Tips

1. **Memory Optimization**
   - Use `train_new_modules_only: true` to freeze base model
   - Enable LoRA with small `lora_r` values
   - Use `bf16: true` for mixed precision

2. **Speed Optimization**
   - Increase `per_device_train_batch_size` if memory allows
   - Use `gradient_accumulation_steps` to simulate larger batches
   - Enable `dataloader_drop_last: true`

3. **Convergence**
   - Start with small `learning_rate` (1e-5 to 5e-5)
   - Use `warmup_ratio: 0.1` for stable training
   - Monitor gradient norms in logs

## Integration with Existing Codebase

The script integrates seamlessly with your existing NovoMolGen codebase:

- Uses `src/models/modeling_novomolgen.py` for the model
- Uses `src/data_loader/molecule_data_module.py` for data loading
- Uses `src/trainer/hf_trainer.py` for custom training logic
- Follows the same configuration patterns as other training scripts

## Next Steps

After successful fine-tuning:

1. **Load the fine-tuned model**:
   ```python
   from models.modeling_novomolgen import NovoMolGen
   model = NovoMolGen.from_pretrained("outputs/my_experiment/final_model")
   ```

2. **Generate molecules with conditions**:
   ```python
   # Generate with protein/ligand conditions
   molecules = model.generate_with_condition(
       input_ids=input_ids,
       pocket_vec=pocket_features,
       evo_vec=evolutionary_features,
       ifp=interaction_fingerprints,
       ligand_vec=ligand_features,
       max_length=128
   )
   ```

3. **Evaluate the model** using your existing evaluation scripts

## Support

For issues or questions:
1. Check the logs in `output_dir/logs/finetune.log`
2. Review the troubleshooting section above
3. Ensure all dependencies are correctly installed
4. Verify your configuration file format
