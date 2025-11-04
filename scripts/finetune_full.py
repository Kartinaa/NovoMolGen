#!/usr/bin/env python3
"""
Comprehensive finetuning script for NovoMolGen with cross-attention and LoRA support.

Features:
- Configuration file support (YAML)
- Comprehensive logging (W&B, local logs)
- Data loading and preprocessing
- Model checkpointing and evaluation
- Cross-attention and LoRA support
- Gradient accumulation and mixed precision
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from transformers import (
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup
)
# Import will be done after sys.path.append
import wandb
from omegaconf import OmegaConf

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.modeling_novomolgen import NovoMolGen, NovoMolGenConfig
from data_loader.molecule_data_module import MolDataModule
from trainer.hf_trainer import HFTrainer, HFTrainingArguments


def setup_logging(log_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """Setup comprehensive logging."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("finetune")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_dir / "finetune.log")
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_wandb(config: Dict[str, Any], run_name: Optional[str] = None) -> None:
    """Setup Weights & Biases logging."""
    if config.get("wandb_logs", False):  # Only if enabled in config
        wandb.init(
            project="novomolgen-finetune",  # Project name
            name=run_name or "finetune",    # Run name
            config=config,                  # Log all config parameters
        )


def create_model(config: Dict[str, Any], logger: logging.Logger) -> NovoMolGen:
    """
    First load the pretrained model config, then apply the finetune modifications.
    Then load the pretrained model using the modified config.
    """
    # Load pretrained config first
    pretrained_path = config.get("pretrained_path", "chandar-lab/NovoMolGen_300M_SMILES_AtomWise")
    base_config = NovoMolGenConfig.from_pretrained(pretrained_path)
    
    # Apply finetune modifications
    base_config.enable_cross_attn = config.get("enable_cross_attn", False)
    base_config.cross_layers = config.get("cross_layers", [])
    base_config.train_new_modules_only = config.get("train_new_modules_only", True)
    base_config.enable_lora = False  # LoRA applied externally
    
    logger.info(f"Loading model from {pretrained_path}")
    model = NovoMolGen.from_pretrained(pretrained_path, config=base_config) # This could load both local and remote model
    
    # Move to device and set dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    if config.get("bf16", True) and torch.cuda.is_available():
        model = model.to(dtype=torch.bfloat16)
        logger.info("Using bfloat16 for numerical stability")
    
    model.train()
    
    # Apply LoRA if requested (BEFORE freezing to ensure proper parameter structure)
    if config.get("enable_lora", False):
        try:
            from peft import LoraConfig, get_peft_model
            
            target_modules = config.get("lora_target_modules", [])
            if not target_modules:
                target_modules = model.autodetect_lora_targets()
                logger.info(f"Auto-detected LoRA targets: {target_modules}")
            
            peft_config = LoraConfig(
                r=config.get("lora_r", 16),
                lora_alpha=config.get("lora_alpha", 32),
                lora_dropout=config.get("lora_dropout", 0.05),
                target_modules=target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            
            model = get_peft_model(model, peft_config)
            model.to(device)
            logger.info("Applied LoRA adapters")
            
        except ImportError:
            logger.error("LoRA requested but 'peft' not installed")
            raise
    
    # Apply freezing logic AFTER LoRA (to ensure proper parameter structure)
    if base_config.train_new_modules_only:
        # For PEFT-wrapped models, we need to access the base model
        if hasattr(model, 'base_model'):
            base_model = model.base_model.model
            base_model._maybe_setup_finetune_freeze_only()
        else:
            model._maybe_setup_finetune_freeze_only()
        
        # Ensure LoRA parameters are always trainable
        if config.get("enable_lora", False):
            for name, p in model.named_parameters():
                if 'lora_' in name:
                    p.requires_grad = True
        
        logger.info("Applied freezing: only new modules are trainable")
    
    # Debug: Print model structure for verification
    logger.info("Model structure debug:")
    if hasattr(model, 'base_model'):
        logger.info("  Model is PEFT-wrapped")
        logger.info(f"  Base model type: {type(model.base_model.model)}")
        if hasattr(model.base_model.model, 'ligand_encoder'):
            logger.info("  Base model has ligand_encoder")
        if hasattr(model.base_model.model, 'transformer'):
            logger.info("  Base model has transformer")
    else:
        logger.info("  Model is not PEFT-wrapped")
        if hasattr(model, 'ligand_encoder'):
            logger.info("  Model has ligand_encoder")
        if hasattr(model, 'transformer'):
            logger.info("  Model has transformer")
    
    return model


def create_tokenizer(config: Dict[str, Any], logger: logging.Logger) -> AutoTokenizer:
    """Create tokenizer."""
    pretrained_path = config.get("pretrained_path", "bisectgroup/NovoMolGen_32M_SAFE_BPE")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")
    
    return tokenizer


def create_data_module(config: Dict[str, Any], tokenizer: AutoTokenizer, logger: logging.Logger):
    """Create data module or dummy dataset."""
    # Handle dummy dataset case
    dataset_name = config.get("dataset_name")
    include_condition_features = config.get("include_condition_features", False)
    
    # Handle HuggingFace dataset case
    if dataset_name and not dataset_name == "dummy":
        logger.info(f"Loading HuggingFace dataset: {dataset_name}")
        try:
            from datasets import load_dataset, load_from_disk
            
            # Try to load from local disk first, then from HuggingFace Hub
            if Path(dataset_name).exists():
                dataset = load_from_disk(dataset_name)
                logger.info(f"Loaded dataset from local path: {dataset_name}")
            else:
                dataset = load_dataset(dataset_name)
                logger.info(f"Loaded dataset from HuggingFace Hub: {dataset_name}")
            
            # Create a simple data module-like object for HuggingFace datasets
            class HuggingFaceDataModule:
                def __init__(self, dataset, tokenizer, include_condition_features):
                    self.train_dataset = dataset['train']
                    self.eval_dataset = dataset.get('validation', None)
                    self.tokenizer = tokenizer
                    self.include_condition_features = include_condition_features
                    
                    # Create a simple collator
                    from transformers import DataCollatorForLanguageModeling
                    self.data_collator = DataCollatorForLanguageModeling(
                        tokenizer=tokenizer, mlm=False
                    )
                    
                    # Wrap with custom collate_fn
                    self.collate_fn = self._build_collate_fn()
                
                def _build_collate_fn(self):
                    """Create collator that handles condition features."""
                    base_collator = self.data_collator
                    include_cond_features = self.include_condition_features
                    
                    def _collate(features):
                        # Tokenize SMILES if not already tokenized
                        if 'input_ids' not in features[0]:
                            # Tokenize the SMILES
                            smiles_list = [f['standardize_smi'] for f in features]
                            tokenized = self.tokenizer(
                                smiles_list,
                                truncation=True,
                                max_length=128,  # Use config max_length
                                padding="max_length",
                                add_special_tokens=True,
                            )
                            
                            # Update features with tokenized data
                            for i, f in enumerate(features):
                                f['input_ids'] = tokenized['input_ids'][i]
                                f['attention_mask'] = tokenized['attention_mask'][i]
                                f['labels'] = tokenized['input_ids'][i].copy()
                        
                        batch = base_collator(features)
                        
                        if include_cond_features:
                            # Check if condition features are already in the dataset
                            if all(key in features[0] for key in ["pocket_vec", "evo_vec", "ifp", "ligand_vec"]):
                                # Load real condition features from dataset
                                batch["pocket_vec"] = torch.stack([torch.tensor(f["pocket_vec"], dtype=torch.float32) for f in features])
                                batch["evo_vec"] = torch.stack([torch.tensor(f["evo_vec"], dtype=torch.float32) for f in features])
                                batch["ifp"] = torch.stack([torch.tensor(f["ifp"], dtype=torch.float32) for f in features])
                                batch["ligand_vec"] = torch.stack([torch.tensor(f["ligand_vec"], dtype=torch.float32) for f in features])
                            else:
                                # Fallback: use random initialization if features not in dataset
                                print("Warning: Condition features not found in dataset, using random initialization")
                                B = batch["input_ids"].size(0)
                                batch["pocket_vec"] = torch.randn(B, 512, dtype=torch.float32)
                                batch["evo_vec"] = torch.randn(B, 1280, dtype=torch.float32)
                                batch["ifp"] = torch.randn(B, 16384, dtype=torch.float32)
                                batch["ligand_vec"] = torch.randn(B, 1536, dtype=torch.float32)
                        
                        return batch
                    
                    return _collate
            
            data_module = HuggingFaceDataModule(dataset, tokenizer, include_condition_features)
            logger.info(f"Created HuggingFace data module with {len(dataset['train'])} training samples")
            if 'validation' in dataset:
                logger.info(f"Validation samples: {len(dataset['validation'])}")
            logger.info(f"Condition features enabled: {include_condition_features}")
            return data_module
            
        except Exception as e:
            logger.error(f"Failed to load HuggingFace dataset {dataset_name}: {e}")
            logger.info("Falling back to dummy dataset...")
            dataset_name = "dummy"
    
    if dataset_name == "dummy":
        # For dummy data, create a simple dataset directly
        logger.info("Creating dummy dataset for testing...")
        from datasets import Dataset
        
        # Create dummy molecular sequences
        dummy_molecules = [
            "CCO", "CCN", "CC(C)O", "CC(C)N", "C1CCC1", 
            "C1CCCC1", "C1=CC=CC=C1", "CC(=O)O", "CCN(CC)CC"
        ] * 100  # Repeat to have enough data
        
        # Tokenize the molecules
        max_length = config.get("max_length", 128)
        tokenized_data = []
        
        for mol in dummy_molecules:
            # Tokenize the molecule
            tokens = tokenizer(
                mol,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                add_special_tokens=True,
            )
            
            # Create sample with condition features
            sample = {
                "input_ids": tokens["input_ids"],
                "attention_mask": tokens["attention_mask"],
                "labels": tokens["input_ids"].copy(),  # For language modeling
            }
            
            if include_condition_features:
                sample.update({
                    "pocket_vec": [0.1] * 512,
                    "evo_vec": [0.1] * 1280,
                    "ifp": [0.1] * 16384,
                    "ligand_vec": [0.1] * 1536,
                })
            
            tokenized_data.append(sample)
        
        # Create dataset
        train_dataset = Dataset.from_list(tokenized_data)
        
        # Create a simple data module-like object
        class DummyDataModule:
            def __init__(self, train_dataset, tokenizer, include_condition_features):
                self.train_dataset = train_dataset
                self.eval_dataset = None
                self.tokenizer = tokenizer
                self.include_condition_features = include_condition_features
                
                # Create a simple collator
                from transformers import DataCollatorForLanguageModeling
                self.data_collator = DataCollatorForLanguageModeling(
                    tokenizer=tokenizer, mlm=False
                )
                
                # Wrap with custom collate_fn
                self.collate_fn = self._build_collate_fn()
            
            def _build_collate_fn(self):
                """Create collator that handles condition features."""
                base_collator = self.data_collator
                include_cond_features = self.include_condition_features
                
                def _collate(features):
                    batch = base_collator(features)
                    
                    if include_cond_features:
                        B = batch["input_ids"].size(0)
                        batch["pocket_vec"] = torch.randn(B, 512, dtype=torch.float32)
                        batch["evo_vec"] = torch.randn(B, 1280, dtype=torch.float32)
                        batch["ifp"] = torch.randn(B, 16384, dtype=torch.float32)
                        batch["ligand_vec"] = torch.randn(B, 1536, dtype=torch.float32)
                    
                    return batch
                
                return _collate
        
        data_module = DummyDataModule(train_dataset, tokenizer, include_condition_features)
        logger.info(f"Created dummy dataset with {len(train_dataset)} samples")
        logger.info(f"Condition features enabled: {include_condition_features}")
        return data_module
    
    else:
        # Use real MolDataModule for actual datasets
        data_module = MolDataModule(
            tokenizer_path=config.get("tokenizer_path"),
            tokenizer_name=config.get("tokenizer_name"),
            dataset_name=dataset_name,
            mol_type=config.get("mol_type", "SMILES"),
            max_seq_length=config.get("max_length", 512),
            num_proc=config.get("num_workers", 4),
            streaming=config.get("streaming", False),
            validation_set_names=None,  # No validation set for testing
            filter_validation_set=False,
            include_condition_features=include_condition_features,
            latent_dim=config.get("latent_dim", 128),
            device=config.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
        )
        
        # Load the datasets
        data_module.load_tokenized_dataset()
        
        logger.info(f"Created data module for dataset: {dataset_name}")
        logger.info(f"Condition features enabled: {include_condition_features}")
        return data_module


def log_parameter_summary(model, logger: logging.Logger) -> None:
    """Log comprehensive parameter summary with proper LoRA handling."""
    # Handle both regular models and PEFT-wrapped models
    if hasattr(model, 'base_model'):
        # PEFT-wrapped model
        base_model = model.base_model.model
        is_peft = True
    else:
        base_model = model
        is_peft = False
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    logger.info(f"Model parameter summary:")
    logger.info(f"  Total: {total_params:,} ({total_params/1e6:.1f}M)")
    logger.info(f"  Trainable: {trainable_params:,} ({trainable_params/1e6:.1f}M) - {100*trainable_params/total_params:.1f}%")
    logger.info(f"  Frozen: {frozen_params:,} ({frozen_params/1e6:.1f}M) - {100*frozen_params/total_params:.1f}%")
    
    # Breakdown by component - handle both regular and PEFT-wrapped models
    cross_attn_params = 0
    lora_params = 0
    encoder_params = 0
    
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
            
        # Cross-attention adapters
        if 'cross_adapters' in name:
            cross_attn_params += p.numel()
        
        # LoRA parameters
        elif 'lora_' in name:
            lora_params += p.numel()
        
        # Encoder parameters (handle both regular and PEFT-wrapped names)
        elif any(x in name for x in ['ligand_encoder', 'protein_encoder', 'condition_fusion', 'cond_proj']):
            encoder_params += p.numel()
    
    other_params = trainable_params - cross_attn_params - lora_params - encoder_params
    
    logger.info(f"Parameter breakdown:")
    logger.info(f"  Cross-attention adapters: {cross_attn_params:,} ({100*cross_attn_params/trainable_params:.1f}%)")
    logger.info(f"  LoRA adapters: {lora_params:,} ({100*lora_params/trainable_params:.1f}%)")
    logger.info(f"  Encoders: {encoder_params:,} ({100*encoder_params/trainable_params:.1f}%)")
    if other_params > 0:
        logger.info(f"  Other: {other_params:,} ({100*other_params/trainable_params:.1f}%)")
    
    # # Debug: Print some example parameter names
    # logger.info("Sample trainable parameter names:")
    # trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    # for i, name in enumerate(trainable_names[:10]):  # Show first 10
    #     param_count = next(p.numel() for n, p in model.named_parameters() if n == name)
    #     logger.info(f"  {i+1}. {name}: {param_count:,} params")
    # if len(trainable_names) > 10:
    #     logger.info(f"  ... and {len(trainable_names) - 10} more")
    
    # Additional debug: Check if encoders are properly unfrozen
    # logger.info("Encoder parameter check:")
    # encoder_names = [name for name, p in model.named_parameters() 
    #                 if any(x in name for x in ['ligand_encoder', 'protein_encoder', 'condition_fusion', 'cond_proj'])]
    # for name in encoder_names:
    #     param = next(p for n, p in model.named_parameters() if n == name)
    #     status = "TRAINABLE" if param.requires_grad else "FROZEN"
    #     logger.info(f"  {name}: {param.numel():,} params - {status}")
    
    # if not encoder_names:
    #     logger.warning("No encoder parameters found! This might indicate a problem with parameter naming after LoRA wrapping.")


def create_training_arguments(config: Dict[str, Any], output_dir: Path, logger: logging.Logger) -> HFTrainingArguments:
    """Create training arguments."""
    # Extract custom attributes for HFTrainingArguments
    save_spec_steps = config.get("save_spec_steps", config.get("save_steps", 500))
    push_spec_checkpoints = config.get("push_spec_checkpoints", False)
    repo_id = config.get("repo_id", "MolGen")
    hub_token = config.get("hub_token", None)
    run_name = config.get("run_name", "finetune")
    
    # Calculate total_batch_size for HFTrainingArguments
    per_device_train_batch_size = config.get("per_device_train_batch_size", 2)
    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 2)
    total_batch_size = config.get("total_batch_size", per_device_train_batch_size * gradient_accumulation_steps)
    
    args = HFTrainingArguments(
        output_dir=str(output_dir),
        save_spec_steps=save_spec_steps,
        push_spec_checkpoints=push_spec_checkpoints,
        repo_id=repo_id,
        hub_token=hub_token,
        run_name=run_name,
        total_batch_size=total_batch_size,
        # Training arguments
        learning_rate=float(config.get("learning_rate", 5e-5)),
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        weight_decay=config.get("weight_decay", 0.01),
        max_grad_norm=config.get("max_grad_norm", 1.0),
        warmup_ratio=config.get("warmup_ratio", 0.1),
        bf16=config.get("bf16", True),
        num_train_epochs=config.get("num_train_epochs", 1),
        logging_steps=config.get("logging_steps", 5),
        save_steps=config.get("save_steps", 10),
        evaluation_strategy="no",  # No evaluation for testing
        save_strategy="steps",
        save_total_limit=2,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        report_to=[],
    )
    
    logger.info(f"Training arguments created successfully")
    return args


def create_trainer(
    model: NovoMolGen,
    tokenizer: AutoTokenizer,
    data_module: MolDataModule,
    training_args: TrainingArguments,
    config: Dict[str, Any],
    logger: logging.Logger
) -> Trainer:
    """Create trainer using the data module's built-in collator."""
    
    # Compute metrics function
    def compute_metrics(eval_preds):
        predictions, labels = eval_preds
        # Add your custom metrics here
        return {}
    
    # Handle case where eval_dataset might be None
    eval_dataset = data_module.eval_dataset if hasattr(data_module, 'eval_dataset') and data_module.eval_dataset is not None else None
    
    trainer = HFTrainer(
        model=model,
        args=training_args,
        train_dataset=data_module.train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_module.collate_fn,  # Use the data module's collator
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    
    return trainer


def main():
    parser = argparse.ArgumentParser(description="Finetune NovoMolGen with cross-attention and LoRA")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for checkpoints and logs")
    parser.add_argument("--run_name", type=str, help="Run name for logging")
    parser.add_argument("--resume_from_checkpoint", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(output_dir / "logs", args.log_level)
    logger.info(f"Starting finetuning with config: {args.config}")
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded config: {json.dumps(config, indent=2)}")
    
    # Setup W&B
    setup_wandb(config, args.run_name)
    
    try:
        # Create components
        logger.info("Creating model...")
        model = create_model(config, logger) ### Model is initiated, no Tokenizer or DataModule needed here.
        
        logger.info("Creating tokenizer...")
        tokenizer = create_tokenizer(config, logger)
        
        logger.info("Creating data module...")
        data_module = create_data_module(config, tokenizer, logger)
        
        # Log parameter summary
        log_parameter_summary(model, logger)
        
        # Debug: Check new modules (only for non-PEFT models)
        if not hasattr(model, 'base_model'):
            logger.info("Running debug check on new modules...")
            model.debug_new_modules()
        
        # Create training arguments
        logger.info("Creating training arguments...")
        training_args = create_training_arguments(config, output_dir, logger)
        
        # Create trainer
        logger.info("Creating trainer...")
        trainer = create_trainer(model, tokenizer, data_module, training_args, config, logger)
        
        # Start training
        logger.info("Starting training...")
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        
        # Save final model
        logger.info("Saving final model...")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir / "final_model")
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise
    finally:
        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
