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
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from transformers import (
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
    TrainerCallback
)
# Import will be done after sys.path.append
from omegaconf import OmegaConf

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Model import - will be done dynamically based on config
# Default to infonce model, but will switch to tanimoto model if tanimoto_weight > 0
NovoMolGen = None
NovoMolGenConfig = None


def _load_model_class(config: dict):
    """Dynamically load the appropriate model class based on config."""
    global NovoMolGen, NovoMolGenConfig

    tanimoto_weight = config.get("tanimoto_weight", 0.0)

    if tanimoto_weight > 0:
        # Use Tanimoto model (supports both Tanimoto and InfoNCE losses)
        from models.modeling_novomolgen_tanimoto import NovoMolGen as TanimotoModel
        from models.modeling_novomolgen_tanimoto import NovoMolGenConfig as TanimotoConfig
        NovoMolGen = TanimotoModel
        NovoMolGenConfig = TanimotoConfig
        print("Using Tanimoto model (tanimoto_weight > 0)")
    else:
        # Use InfoNCE model (original model)
        from models.modeling_novomolgen_infonce_120225_v2 import NovoMolGen as InfoNCEModel
        from models.modeling_novomolgen_infonce_120225_v2 import NovoMolGenConfig as InfoNCEConfig
        NovoMolGen = InfoNCEModel
        NovoMolGenConfig = InfoNCEConfig
        print("Using InfoNCE model (tanimoto_weight = 0)")
from data_loader.molecule_data_module import MolDataModule
from trainer.hf_trainer import HFTrainer, HFTrainingArguments


def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


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
    """
    Setup Weights & Biases logging.
    
    Note: If wandb_logs is True, WandbCallback will handle wandb.init() in its setup() method.
    This function is kept for backward compatibility and to set environment variables if needed.
    """
    if config.get("wandb_logs", False):
        # Set WANDB_WATCH environment variable if specified in config
        watch_mode = config.get("wandb_watch", "all")  # "all", "parameters", "gradients", or "false"
        os.environ["WANDB_WATCH"] = str(watch_mode)
        
        # Set other W&B environment variables if needed
        if config.get("wandb_log_model"):
            os.environ["WANDB_LOG_MODEL"] = config.get("wandb_log_model", "false")
        
        print("✅ W&B logging enabled (will be initialized by WandbCallback)")
        print(f"   WANDB_WATCH={watch_mode} (set to 'all', 'parameters', 'gradients', or 'false' to control model monitoring)")
    else:
        print("⚠️  W&B logging disabled (wandb_logs=False in config)")
        print("   To enable W&B, set in config file: wandb_logs: true")


def create_model(config: Dict[str, Any], logger: logging.Logger) -> NovoMolGen:
    """
    First load the pretrained model config, then apply the finetune modifications.
    Then load the pretrained model using the modified config.
    """
    # Load pretrained config first
    pretrained_path = config.get("pretrained_path", False)
    base_config = NovoMolGenConfig.from_pretrained(pretrained_path) ### Just 4 following kwargs with model config options.
    
    # Apply finetune modifications
    base_config.enable_cross_attn = config.get("enable_cross_attn", False)
    base_config.cross_layers = config.get("cross_layers", [])
    base_config.train_new_modules_only = config.get("train_new_modules_only", True)
    base_config.cond_tokens_len = config.get("cond_tokens_len", base_config.hidden_size)
    base_config.enable_lora = False  # LoRA applied externally

    # Optional ablation controls (e.g., remove IFP information during training)
    base_config.ablate_ifp = config.get("ablate_ifp", False)

    # InfoNCE alignment configuration (optional)
    if "infonce_weight" in config:
        base_config.infonce_weight = float(config.get("infonce_weight", 0.0))
    if "infonce_temperature" in config:
        base_config.infonce_temperature = float(config.get("infonce_temperature", 0.2))

    # Tanimoto loss configuration (optional)
    if "tanimoto_weight" in config:
        base_config.tanimoto_weight = float(config.get("tanimoto_weight", 0.0))
        logger.info(f"Tanimoto loss weight: {base_config.tanimoto_weight}")
    if "tanimoto_fp_radius" in config:
        base_config.tanimoto_fp_radius = int(config.get("tanimoto_fp_radius", 2))
    if "tanimoto_fp_bits" in config:
        base_config.tanimoto_fp_bits = int(config.get("tanimoto_fp_bits", 2048))
    
    # Ligand vector statistics for normalization (optional)
    # if "ligand_vec_stats_path" in config:
    #     base_config.ligand_vec_stats_path = config.get("ligand_vec_stats_path")
    #     logger.info(f"Using ligand_vec_stats_path: {base_config.ligand_vec_stats_path}")
    
    # Apply KL annealing configuration if provided
    vae_config = config.get("vae", {})
    if vae_config:
        base_config.beta_kl_init = vae_config.get("beta_kl_init", 0.0)
        base_config.beta_kl_final = vae_config.get("beta_kl_final", 1.0)
        base_config.beta_kl_anneal_type = vae_config.get("beta_kl_anneal_type", "linear")
        base_config.beta_kl_anneal_steps = vae_config.get("beta_kl_anneal_steps", 30000)
        base_config.beta_kl_eval = vae_config.get("beta_kl_eval", None)
        logger.info("KL annealing configured:")
        logger.info(f"  beta_kl_init: {base_config.beta_kl_init}")
        logger.info(f"  beta_kl_final: {base_config.beta_kl_final}")
        logger.info(f"  beta_kl_anneal_type: {base_config.beta_kl_anneal_type}")
        logger.info(f"  beta_kl_anneal_steps: {base_config.beta_kl_anneal_steps}")
        if base_config.beta_kl_eval is not None:
            logger.info(f"  beta_kl_eval: {base_config.beta_kl_eval}")
    else:
        # Use legacy fixed beta_kl if no VAE config
        base_config.beta_kl = config.get("beta_kl", 0.1)
        logger.info(f"Using fixed beta_kl: {base_config.beta_kl} (no annealing)")
    
    logger.info(f"Loading model from {pretrained_path}")
    model = NovoMolGen.from_pretrained(pretrained_path, config=base_config) # This is without
    
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
            
            # Load all LoRA config from YAML file
            target_modules = config.get("lora_target_modules", [])
            if not target_modules:
                target_modules = model.autodetect_lora_targets()
                logger.info(f"Auto-detected LoRA targets: {target_modules}")
            else:
                logger.info(f"Using LoRA target modules from config: {target_modules}")
            
            # Determine modules_to_save based on what's enabled
            # Note: PEFT's modules_to_save does NOT support ModuleDict/ParameterDict
            # So we cannot add "transformer.cross_adapters" or "transformer.gates" directly
            # These will be saved through the normal training flow since they are unfrozen
            modules_to_save = []
            
            # Save encoder modules if they exist and are trainable
            # These are regular nn.Module instances, not containers
            if config.get("include_condition_features", False):
                modules_to_save.extend([
                    "ligand_encoder",
                    "protein_encoder", 
                    "condition_fusion",
                    "cond_proj",
                ])
            
            # Note: cross_adapters (ModuleDict) and gates (ParameterDict) are NOT added here
            # because PEFT does not support container types in modules_to_save.
            # However, they will still be:
            # 1. Trained (unfrozen by _maybe_setup_finetune_freeze_only())
            # 2. Saved in full_model checkpoints (via SaveFullModelCallback)
            # 3. Saved in PEFT checkpoints (as part of base model state_dict if properly configured)
            
            # Remove duplicates and None values
            modules_to_save = list(set([m for m in modules_to_save if m]))
            
            peft_config = LoraConfig(
                r=config.get("lora_r", 16),
                lora_alpha=config.get("lora_alpha", 32),
                lora_dropout=config.get("lora_dropout", 0.05),
                target_modules=target_modules,
                bias=config.get("lora_bias", "none"),  # Load from config, default "none"
                task_type=config.get("lora_task_type", "CAUSAL_LM"),  # Load from config, default "CAUSAL_LM"
                modules_to_save=modules_to_save if modules_to_save else None,  # Save additional trainable modules
            )
            
            if modules_to_save:
                logger.info(f"modules_to_save: {modules_to_save}")
                logger.info("These modules will be fully saved in PEFT checkpoints (not just LoRA parameters):")
                logger.info("  - Encoder modules (ligand_encoder, protein_encoder, condition_fusion, cond_proj)")
                logger.info("")
                logger.info("Note: cross_adapters and gates are NOT in modules_to_save")
                logger.info("  Reason: PEFT does not support ModuleDict/ParameterDict container types")
                logger.info("  However, they will still be:")
                logger.info("    1. Trained (unfrozen by _maybe_setup_finetune_freeze_only())")
                logger.info("    2. Saved in full_model checkpoints (via SaveFullModelCallback)")
                logger.info("    3. Saved in PEFT checkpoints (as part of model state_dict)")
            else:
                logger.info("modules_to_save not set, only LoRA parameters will be saved")
            
            logger.info("")
            logger.info("⚠️  Important: modules_to_save only affects PEFT checkpoint saving, not training!")
            logger.info("   Which modules are trained is controlled by train_new_modules_only and _maybe_setup_finetune_freeze_only()")
            
            logger.info(f"LoRA configuration: r={peft_config.r}, alpha={peft_config.lora_alpha}, "
                       f"dropout={peft_config.lora_dropout}, bias={peft_config.bias}, "
                       f"task_type={peft_config.task_type}")
            
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
    """Create tokenizer from tokenizer_path or pretrained_path."""
    # Priority: tokenizer_path > pretrained_path
    tokenizer_path = config.get("tokenizer_path")
    pretrained_path = config.get("pretrained_path", "bisectgroup/NovoMolGen_32M_SAFE_BPE")
    
    if tokenizer_path:
        # Load from local tokenizer path using MoleculeTokenizer
        logger.info(f"Loading tokenizer from local path: {tokenizer_path}")
        try:
            # Import after sys.path is set up (src is already in path)
            from data_loader.molecule_tokenizer import MoleculeTokenizer
            mol_tokenizer = MoleculeTokenizer.load(tokenizer_path)
            tokenizer = mol_tokenizer.get_pretrained()
            logger.info(f"Successfully loaded tokenizer from {tokenizer_path}")
        except Exception as e:
            logger.warning(f"Failed to load tokenizer from {tokenizer_path}: {e}")
            logger.info(f"Falling back to pretrained tokenizer from {pretrained_path}")
            tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
    else:
        logger.info(f"Loading tokenizer from pretrained path: {pretrained_path}")
        tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")
    
    return tokenizer


def create_data_module(config: Dict[str, Any], tokenizer: AutoTokenizer, logger: logging.Logger):
    """Create data module using MolDataModule for tokenized datasets or HuggingFace datasets."""
    # Handle dummy dataset case
    dataset_name = config.get("dataset_name")
    include_condition_features = config.get("include_condition_features", False)

    # Check if we should use MolDataModule (for tokenized datasets)
    # This is indicated by having tokenizer_path or tokenizer_name in config
    use_mol_data_module = config.get("tokenizer_path") is not None or config.get("tokenizer_name") is not None

    # Check if we should use the augmented data module (for pre-computed SAFE representations)
    use_augmented_data_module = config.get("use_augmented_data_module", False)

    if use_mol_data_module and dataset_name and dataset_name != "dummy":
        # Use MolDataModule for tokenized datasets
        # Select appropriate data module class based on config
        if use_augmented_data_module:
            from data_loader.molecule_data_module_augmented import MolDataModuleAugmented as DataModuleClass
            logger.info("Using MolDataModuleAugmented for pre-computed SAFE representations")
        else:
            DataModuleClass = MolDataModule
            logger.info("Using MolDataModule for tokenized dataset")
        try:
            data_module = DataModuleClass(
                tokenizer_path=config.get("tokenizer_path"),
                tokenizer_name=config.get("tokenizer_name", None),
                dataset_name=dataset_name,
                mol_type=config.get("mol_type", "SMILES"),
                max_seq_length=config.get("max_length", config.get("max_seq_length", 64)),
                num_proc=config.get("num_proc", config.get("num_workers", 4)),
                streaming=config.get("streaming", False),
                validation_set_names=config.get("validation_set_names", None),
                filter_validation_set=config.get("filter_validation_set", False),
                include_condition_features=include_condition_features,
                latent_dim=config.get("latent_dim", 128),
                device=config.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
            )
            
            # Load the tokenized datasets
            data_module.load_tokenized_dataset()
            
            logger.info(f"Created {DataModuleClass.__name__} for dataset: {dataset_name}")
            logger.info(f"Training samples: {len(data_module.train_dataset) if data_module.train_dataset else 0}")
            if data_module.eval_dataset:
                if isinstance(data_module.eval_dataset, dict):
                    for name, val_ds in data_module.eval_dataset.items():
                        logger.info(f"Validation samples ({name}): {len(val_ds)}")
                else:
                    logger.info(f"Validation samples: {len(data_module.eval_dataset)}")
            logger.info(f"Condition features enabled: {include_condition_features}")
            logger.info(f"Tokenizer vocab size: {data_module.tokenizer.vocab_size}")
            logger.info(f"Max sequence length: {data_module.max_seq_length}")
            
            return data_module
            
        except Exception as e:
            logger.error(f"Failed to create MolDataModule: {e}", exc_info=True)
            logger.info("Falling back to HuggingFace dataset loader...")
            # Fall through to HuggingFace dataset case
    
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
                def __init__(self, dataset, tokenizer, include_condition_features, max_length=128):
                    self.train_dataset = dataset['train']
                    self.eval_dataset = dataset.get('validation', None)
                    self.tokenizer = tokenizer
                    self.include_condition_features = include_condition_features
                    self.max_length = max_length
                    
                    # Check if dataset is already tokenized
                    self.is_tokenized = self._check_if_tokenized()
                    
                    # Create a simple collator
                    from transformers import DataCollatorForLanguageModeling
                    self.data_collator = DataCollatorForLanguageModeling(
                        tokenizer=tokenizer, mlm=False
                    )
                    
                    # Wrap with custom collate_fn
                    self.collate_fn = self._build_collate_fn()
                
                def _check_if_tokenized(self):
                    """Check if dataset is already tokenized."""
                    sample = self.train_dataset[0] if len(self.train_dataset) > 0 else {}
                    has_input_ids = 'input_ids' in sample
                    has_attention_mask = 'attention_mask' in sample
                    has_labels = 'labels' in sample
                    
                    is_tokenized = has_input_ids and has_attention_mask
                    
                    if is_tokenized:
                        logger.info("✓ Dataset is already tokenized (found input_ids and attention_mask)")
                    else:
                        logger.info("⚠️  Dataset is not tokenized, will tokenize on-the-fly")
                        if has_input_ids:
                            logger.warning("  Found input_ids but missing attention_mask - may cause issues")
                    
                    return is_tokenized
                
                def _build_collate_fn(self):
                    """Create collator that handles condition features."""
                    base_collator = self.data_collator
                    include_cond_features = self.include_condition_features
                    is_tokenized = self.is_tokenized
                    tokenizer = self.tokenizer
                    max_length = self.max_length
                    
                    def _collate(features):
                        # Tokenize SMILES if not already tokenized
                        if not is_tokenized:
                            # Find SMILES column (could be 'standardize_smi' or other names)
                            smiles_column = None
                            for col in ['standardize_smi', 'smiles', 'SMILES']:
                                if col in features[0]:
                                    smiles_column = col
                                    break
                            
                            if smiles_column is None:
                                available_cols = list(features[0].keys())
                                raise ValueError(
                                    f"Dataset is not tokenized and no SMILES column found. "
                                    f"Available columns: {available_cols}. "
                                    f"Please either: "
                                    f"1) Pre-tokenize using scripts/data_processing/08_tokenize_hf_dataset.py, or "
                                    f"2) Ensure dataset has 'standardize_smi' column"
                                )
                            
                            # Tokenize the SMILES
                            smiles_list = [f[smiles_column] for f in features]
                            tokenized = tokenizer(
                                smiles_list,
                                truncation=True,
                                max_length=max_length,
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
            
            max_length = config.get("max_length", 128)
            data_module = HuggingFaceDataModule(
                dataset, tokenizer, include_condition_features, max_length=max_length
            )
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
    
    # If we reach here, something went wrong
    logger.error("Failed to create data module. Please check your configuration.")
    raise ValueError(
        "Failed to create data module. "
        "Please ensure either: "
        "1) tokenizer_path is set (for MolDataModule), or "
        "2) dataset_name points to a valid HuggingFace dataset, or "
        "3) dataset_name is 'dummy' for testing"
    )


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
    
    # Debug: Print some example parameter names
    logger.info("Sample trainable parameter names:")
    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    for i, name in enumerate(trainable_names[:10]):  # Show first 10
        param_count = next(p.numel() for n, p in model.named_parameters() if n == name)
        logger.info(f"  {i+1}. {name}: {param_count:,} params")
    if len(trainable_names) > 10:
        logger.info(f"  ... and {len(trainable_names) - 10} more")
    
    # Additional debug: Check if encoders are properly unfrozen
    logger.info("Encoder parameter check:")
    encoder_names = [name for name, p in model.named_parameters() 
                    if any(x in name for x in ['ligand_encoder', 'protein_encoder', 'condition_fusion', 'cond_proj'])]
    for name in encoder_names:
        param = next(p for n, p in model.named_parameters() if n == name)
        status = "TRAINABLE" if param.requires_grad else "FROZEN"
        logger.info(f"  {name}: {param.numel():,} params - {status}")
    
    if not encoder_names:
        logger.warning("No encoder parameters found! This might indicate a problem with parameter naming after LoRA wrapping.")


# Prefixes to strip when mapping modules_to_save keys
PREFIXES_TO_STRIP = [
    r'^base_model\.model\.',
    r'^base_model\.',
    r'^model\.',
]


def map_modules_to_save_key(orig_key: str) -> Optional[str]:
    """
    Map a PEFT modules_to_save key to the target key in merged model.
    
    PEFT stores modules_to_save as: base_model.model.{module}.modules_to_save.default.{param}
    After merge, target should be: {module}.{param}
    
    Args:
        orig_key: Original key from PEFT model state_dict
        
    Returns:
        Target key in merged model, or None if not a modules_to_save key
    """
    # Only process keys containing modules_to_save
    if '.modules_to_save.' not in orig_key:
        return None
    
    # Replace '.modules_to_save.default.' with '.'
    target = orig_key.replace('.modules_to_save.default.', '.')
    
    # Try stripping various prefixes
    for prefix_pattern in PREFIXES_TO_STRIP:
        target = re.sub(prefix_pattern, '', target)
    
    return target


def copy_modules_to_save_parameters(
    full_state_dict: Dict[str, torch.Tensor],
    merged_model: torch.nn.Module,
    logger: logging.Logger
) -> int:
    """
    Copy modules_to_save parameters from PEFT model to merged model.
    
    Args:
        full_state_dict: State dict from PEFT model (before merge)
        merged_model: Merged model (after merge_and_unload())
        logger: Logger instance
        
    Returns:
        Number of parameters successfully copied
    """
    merged_state_dict = merged_model.state_dict()
    
    # Find all modules_to_save keys
    modules_to_save_keys = [k for k in full_state_dict.keys() if '.modules_to_save.default.' in k]
    expected_count = len(modules_to_save_keys)
    
    if expected_count == 0:
        logger.info("No modules_to_save parameters found in state dict")
        return 0
    
    logger.info(f"Found {expected_count} modules_to_save parameters to copy")
    
    copied_count = 0
    failed_keys = []
    
    for orig_key in modules_to_save_keys:
        target_key = map_modules_to_save_key(orig_key)
        
        if target_key is None:
            failed_keys.append((orig_key, "Failed to map key"))
            continue
        
        if target_key not in merged_state_dict:
            failed_keys.append((orig_key, f"Target key not found: {target_key}"))
            continue
        
        # Check shape compatibility
        orig_shape = full_state_dict[orig_key].shape
        target_shape = merged_state_dict[target_key].shape
        
        if orig_shape != target_shape:
            failed_keys.append((orig_key, f"Shape mismatch: {orig_shape} vs {target_shape}"))
            continue
        
        # Copy the parameter
        merged_state_dict[target_key].copy_(full_state_dict[orig_key])
        copied_count += 1
    
    # Log results
    logger.info(f"Successfully copied {copied_count}/{expected_count} modules_to_save parameters")
    
    if copied_count < expected_count:
        logger.warning(f"⚠️  {expected_count - copied_count} modules_to_save parameters were NOT copied!")
        logger.warning("Sample failed keys (showing first 5):")
        for orig_key, reason in failed_keys[:5]:
            logger.warning(f"  - {orig_key[:80]}... -> {reason}")
        if len(failed_keys) > 5:
            logger.warning(f"  ... and {len(failed_keys) - 5} more")
    else:
        logger.info("✅ All modules_to_save parameters copied successfully!")
    
    # Load updated state dict
    merged_model.load_state_dict(merged_state_dict, strict=False)
    
    return copied_count


def create_training_arguments(config: Dict[str, Any], output_dir: Path, logger: logging.Logger) -> HFTrainingArguments:
    """Create training arguments."""
    # Extract custom attributes for HFTrainingArguments
    save_spec_steps = config.get("save_spec_steps", config.get("save_steps", 500))
    push_spec_checkpoints = config.get("push_spec_checkpoints", False)
    repo_id = config.get("repo_id", None)
    hub_token = config.get("hub_token", None)
    run_name = config.get("run_name", "finetune")
    
    # Calculate total_batch_size for HFTrainingArguments
    per_device_train_batch_size = config.get("per_device_train_batch_size", 2)
    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 2)
    total_batch_size = config.get("total_batch_size", per_device_train_batch_size * gradient_accumulation_steps)
    
    # Handle report_to: if using WandbCallback, don't add "wandb" to report_to
    # (to avoid duplicate callbacks from get_reporting_integration_callbacks)
    report_to = config.get("report_to", [])
    if config.get("wandb_logs", False):
        # Ensure report_to is a list
        if not isinstance(report_to, list):
            report_to = [report_to] if report_to else []
        # Remove "wandb" if present (we'll use WandbCallback instead)
        if "wandb" in report_to:
            report_to.remove("wandb")
            logger.info("✅ W&B logging enabled: using WandbCallback (removed 'wandb' from report_to to avoid duplicates)")
        else:
            logger.info("✅ W&B logging enabled: using WandbCallback")
        logger.info("   W&B will automatically log: train/loss, eval/loss, learning_rate, model/grad_norm, etc.")
    else:
        logger.info("⚠️  W&B logging disabled (wandb_logs=False)")
        logger.info("   To view training curves, set in config file: wandb_logs: true")
    
    # Build kwargs dict with all training arguments
    training_kwargs = {
        # Core training arguments
        "learning_rate": float(config.get("learning_rate", 5e-5)),
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": config.get("per_device_eval_batch_size", per_device_train_batch_size),
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "weight_decay": config.get("weight_decay", 0.01),
        "max_grad_norm": config.get("max_grad_norm", 1.0),
        "warmup_ratio": config.get("warmup_ratio", 0.1),
        "bf16": config.get("bf16", True),
        "bf16_full_eval": config.get("bf16_full_eval", False),
        "num_train_epochs": config.get("num_train_epochs", 1),
        "logging_steps": config.get("logging_steps", 5),
        "save_steps": config.get("save_steps", 10),
        "save_strategy": config.get("save_strategy", "steps"),
        "save_total_limit": config.get("save_total_limit", 3),
        "dataloader_drop_last": config.get("dataloader_drop_last", True),
        "remove_unused_columns": config.get("remove_unused_columns", False),
        "report_to": report_to,
    }
    
    # Optional training arguments (only add if specified in config)
    optional_args = {
        "max_steps": config.get("max_steps"),
        "gradient_checkpointing": config.get("gradient_checkpointing"),
        "lr_scheduler_type": config.get("lr_scheduler_type"),
        "optim": config.get("optim"),
        "adam_beta1": config.get("adam_beta1"),
        "adam_beta2": config.get("adam_beta2"),
        "evaluation_strategy": config.get("evaluation_strategy"),
        "eval_steps": config.get("eval_steps"),
        "do_train": config.get("do_train"),
        "do_eval": config.get("do_eval"),
        "dataloader_num_workers": config.get("dataloader_num_workers"),
        "dataloader_pin_memory": config.get("dataloader_pin_memory"),
        "log_on_each_node": config.get("log_on_each_node"),
        "torch_compile": config.get("torch_compile"),
        "save_safetensors": config.get("save_safetensors"),
        "hub_private_repo": config.get("hub_private_repo"),
        "include_num_input_tokens_seen": config.get("include_num_input_tokens_seen"),
        "deepspeed": config.get("deepspeed"),  # DeepSpeed config file path
    }
    
    # Add optional args only if they are not None
    for key, value in optional_args.items():
        if value is not None:
            training_kwargs[key] = value
    
    args = HFTrainingArguments(
        output_dir=str(output_dir),
        save_spec_steps=save_spec_steps,
        push_spec_checkpoints=push_spec_checkpoints,
        repo_id=repo_id,
        hub_token=hub_token,
        run_name=run_name,
        total_batch_size=total_batch_size,
        **training_kwargs,
    )
    
    logger.info(f"Training arguments created successfully")
    return args


class KLAnnealingCallback(TrainerCallback):
    """Callback to update model's training step for KL annealing."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Update model's training step after each step."""
        if model is None:
            return
        
        # Handle PEFT-wrapped models: access base model
        actual_model = model
        if hasattr(model, 'base_model'):
            actual_model = model.base_model.model
        
        # Update training step for KL annealing
        if hasattr(actual_model, 'set_training_step'):
            actual_model.set_training_step(state.global_step)


class LossMonitoringCallback(TrainerCallback):
    """Callback to monitor NLL loss and KL loss separately."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """Called when logs are written. Extract and log NLL and KL losses."""
        if logs is None or model is None:
            return
        
        # Handle PEFT-wrapped models: access base model
        actual_model = model
        if hasattr(model, 'base_model'):
            actual_model = model.base_model.model
        
        # Try to extract nll_loss and kl_loss from model's last forward pass
        if hasattr(actual_model, '_last_nll_loss'):
            nll_loss = actual_model._last_nll_loss
            kl_loss = actual_model._last_kl_loss
            kl_loss_unscaled = actual_model._last_kl_loss_unscaled
            beta = getattr(actual_model, '_last_beta', 0.1)
            
            # Add to logs (will be logged to W&B and console)
            logs['train/nll_loss'] = nll_loss
            logs['train/kl_loss'] = kl_loss  # Scaled KL loss (beta * kl_loss)
            logs['train/kl_loss_unscaled'] = kl_loss_unscaled  # Unscaled KL loss
            logs['train/beta_kl'] = beta
            
            # Calculate ratio for monitoring
            if nll_loss > 0:
                kl_nll_ratio = kl_loss_unscaled / nll_loss
                logs['train/kl_nll_ratio'] = kl_nll_ratio
                
                # Log warning if ratio is extreme (only log occasionally to avoid spam)
                if state.global_step % args.logging_steps == 0:
                    if kl_nll_ratio > 10:
                        self.logger.warning(
                            f"Step {state.global_step}: ⚠️  KL loss ({kl_loss_unscaled:.4f}) is {kl_nll_ratio:.2f}x larger than NLL loss ({nll_loss:.4f}). "
                            f"Consider reducing beta_kl (current: {beta:.4f})."
                        )
                    elif kl_nll_ratio < 0.01:
                        self.logger.info(
                            f"Step {state.global_step}: ℹ️  KL loss ({kl_loss_unscaled:.4f}) is very small ({kl_nll_ratio:.4f}x NLL). "
                            f"Consider increasing beta_kl (current: {beta:.4f}) if regularization is needed."
                        )

        # Log InfoNCE loss and cosine statistics if available (for conditional alignment diagnostics)
        if hasattr(actual_model, '_last_infonce_loss'):
            logs['train/infonce_loss'] = actual_model._last_infonce_loss
        if hasattr(actual_model, '_last_infonce_cos_mean'):
            logs['train/infonce_cos_mean'] = actual_model._last_infonce_cos_mean
        if hasattr(actual_model, '_last_infonce_cos_std'):
            logs['train/infonce_cos_std'] = actual_model._last_infonce_cos_std

        # Log Tanimoto loss if available (for ligand conditioning enforcement)
        if hasattr(actual_model, '_last_tanimoto_loss'):
            logs['train/tanimoto_loss'] = actual_model._last_tanimoto_loss

        # Histogram of cosine similarities (q·k) for InfoNCE, if stored by the model
        if hasattr(actual_model, '_last_infonce_cos_hist'):
            try:
                import wandb
                cos_vals = actual_model._last_infonce_cos_hist
                if isinstance(cos_vals, torch.Tensor):
                    cos_vals = cos_vals.cpu().numpy()
                logs['train/infonce_cos_hist'] = wandb.Histogram(cos_vals)
            except Exception:
                # 如果 wandb 不可用或出錯，就安靜跳過 histogram
                pass
    
    def on_evaluate(self, args, state, control, model=None, logs=None, **kwargs):
        """Called after evaluation. Extract and log eval NLL and KL losses."""
        if logs is None or model is None:
            return
        
        # Handle PEFT-wrapped models: access base model
        actual_model = model
        if hasattr(model, 'base_model'):
            actual_model = model.base_model.model
        
        # Update step for evaluation (so beta_kl_eval can be used if configured)
        if hasattr(actual_model, 'set_training_step'):
            actual_model.set_training_step(state.global_step)
        
        # For evaluation, losses are computed but model might not have _last_* attributes
        # We can try to extract from the evaluation outputs if available
        if hasattr(actual_model, '_last_nll_loss'):
            logs['eval/nll_loss'] = actual_model._last_nll_loss
            logs['eval/kl_loss'] = actual_model._last_kl_loss
            logs['eval/kl_loss_unscaled'] = actual_model._last_kl_loss_unscaled
            logs['eval/beta_kl'] = getattr(actual_model, '_last_beta', 0.1)


class GateFixCallback(TrainerCallback):
    """Callback to ensure cross-attention gates are properly trainable and monitor gradient flow."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._checked = False
        self._fixed = False
        self._initial_gate_values = {}  # Track initial values to detect updates
        self._last_grad_norms = {}  # Store gradient norms captured by hooks
        self._hooks = []  # Store hook handles for cleanup

    def on_train_begin(self, args, state, control, model=None, optimizer=None, **kwargs):
        """Check and fix gates at the start of training, register backward hooks."""
        if model is None or self._checked:
            return

        self._checked = True

        # Get the actual model (handle PEFT wrapping)
        actual_model = model
        if hasattr(model, 'base_model'):
            actual_model = model.base_model.model

        # Check if model has gates
        if not hasattr(actual_model, 'transformer'):
            self.logger.warning("Model has no transformer attribute, skipping gate check")
            return

        transformer = actual_model.transformer
        if not hasattr(transformer, 'gates') or transformer.gates is None:
            self.logger.warning("Transformer has no gates, skipping gate check")
            return

        # Check and fix requires_grad for all gates
        gates = transformer.gates
        gate_params = []
        gate_ptrs = set()
        for layer_idx, gate in gates.items():
            if not gate.requires_grad:
                gate.requires_grad = True
                self.logger.info(f"Fixed: Gate {layer_idx} requires_grad was False, now True")
            gate_params.append(gate)
            gate_ptrs.add(gate.data_ptr())
            self._initial_gate_values[layer_idx] = gate.item()
            self.logger.info(f"Gate {layer_idx}: value={gate.item():.4f}, requires_grad={gate.requires_grad}, data_ptr={gate.data_ptr()}")

            # Register backward hook to capture gradient during backward pass
            def make_hook(lid):
                def hook(grad):
                    if grad is not None:
                        self._last_grad_norms[lid] = grad.item()
                    return grad
                return hook
            handle = gate.register_hook(make_hook(layer_idx))
            self._hooks.append(handle)

        self.logger.info(f"Total gate parameters: {len(gate_params)}")
        self.logger.info(f"Registered backward hooks on {len(self._hooks)} gates")

        # Check if gates are in model.parameters()
        model_param_ptrs = {p.data_ptr() for p in model.parameters()}
        gates_in_model = gate_ptrs.issubset(model_param_ptrs)
        self.logger.info(f"Gates in model.parameters(): {gates_in_model}")

        if not gates_in_model:
            self.logger.warning("⚠️ Gates NOT in model.parameters()! They won't be optimized!")
            # Find which gates are missing
            missing = gate_ptrs - model_param_ptrs
            self.logger.warning(f"Missing gate pointers: {len(missing)}/{len(gate_ptrs)}")

        # Check optimizer param groups
        if optimizer is not None:
            opt_param_ptrs = set()
            for group in optimizer.param_groups:
                for p in group['params']:
                    opt_param_ptrs.add(p.data_ptr())
            gates_in_optimizer = gate_ptrs.issubset(opt_param_ptrs)
            self.logger.info(f"Gates in optimizer: {gates_in_optimizer}")
            if not gates_in_optimizer:
                self.logger.warning("⚠️ Gates NOT in optimizer param groups!")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Log gate values and gradient info periodically."""
        if model is None:
            return

        # Log every 500 steps
        if state.global_step % 500 != 0:
            return

        # Get the actual model
        actual_model = model
        if hasattr(model, 'base_model'):
            actual_model = model.base_model.model

        if not hasattr(actual_model, 'transformer') or not hasattr(actual_model.transformer, 'gates'):
            return

        gates = actual_model.transformer.gates
        if gates is None:
            return

        # Log gate values, changes from initial, and captured gradients from hooks
        gate_info = []
        for layer_idx, gate in sorted(gates.items(), key=lambda x: int(x[0])):
            current_val = gate.item()
            initial_val = self._initial_gate_values.get(layer_idx, current_val)
            delta = current_val - initial_val

            # Get gradient from hook (captured during backward, before zero_grad)
            grad_val = self._last_grad_norms.get(layer_idx, None)
            grad_str = f"grad={grad_val:.6f}" if grad_val is not None else "no_grad_hook"

            gate_info.append(f"L{layer_idx}:{current_val:.4f}(Δ={delta:+.4f},{grad_str})")

        self.logger.info(f"Step {state.global_step} Gates: {', '.join(gate_info)}")

        # Clear captured gradients for next step
        self._last_grad_norms.clear()

    def on_train_end(self, args, state, control, **kwargs):
        """Clean up hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()


class ProgressiveUnfreezingCallback(TrainerCallback):
    """Callback to progressively unfreeze transformer layers during training."""

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.progressive_config = config.get("progressive_unfreezing", {})
        self.enabled = self.progressive_config.get("enabled", False)
        
        if self.enabled:
            self.layers_to_unfreeze = self.progressive_config.get("layers_to_unfreeze", [])
            self.unfreeze_steps = self.progressive_config.get("unfreeze_steps", [])
            
            if len(self.layers_to_unfreeze) != len(self.unfreeze_steps):
                self.logger.error(
                    f"progressive_unfreezing: layers_to_unfreeze ({len(self.layers_to_unfreeze)}) "
                    f"and unfreeze_steps ({len(self.unfreeze_steps)}) must have the same length"
                )
                self.enabled = False
            else:
                # Sort by step to ensure correct order
                sorted_pairs = sorted(zip(self.unfreeze_steps, self.layers_to_unfreeze))
                self.unfreeze_steps, self.layers_to_unfreeze = zip(*sorted_pairs)
                self.unfreeze_steps = list(self.unfreeze_steps)
                self.layers_to_unfreeze = list(self.layers_to_unfreeze)
                
                self.logger.info("=" * 60)
                self.logger.info("Progressive unfreezing enabled:")
                self.logger.info(f"  Layers to unfreeze: {self.layers_to_unfreeze}")
                self.logger.info(f"  Unfreeze steps: {self.unfreeze_steps}")
                self.logger.info("=" * 60)
                self._unfrozen_layers = set()
        else:
            self._unfrozen_layers = set()
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Check if we should unfreeze a layer at this step."""
        if not self.enabled or model is None:
            return
        
        # Check if we need to unfreeze any layers at this step
        for step, layer_idx in zip(self.unfreeze_steps, self.layers_to_unfreeze):
            if state.global_step >= step and layer_idx not in self._unfrozen_layers:
                # Unfreeze this layer
                actual_model = model
                if hasattr(model, 'base_model'):
                    actual_model = model.base_model.model
                
                if hasattr(actual_model, 'unfreeze_transformer_layer'):
                    actual_model.unfreeze_transformer_layer(layer_idx)
                    self._unfrozen_layers.add(layer_idx)
                    self.logger.info(
                        f"Step {state.global_step}: ✅ Unfroze transformer layer {layer_idx} "
                        f"(unfrozen layers: {sorted(self._unfrozen_layers)})"
                    )
                else:
                    self.logger.warning(
                        f"Model does not have unfreeze_transformer_layer method, "
                        f"cannot unfreeze layer {layer_idx}"
                    )


class SaveFullModelCallback(TrainerCallback):
    """Callback to save full model (merged) at each checkpoint."""
    
    def __init__(self, tokenizer, config: Dict[str, Any], config_file_path: str, logger: logging.Logger):
        self.tokenizer = tokenizer
        self.config = config
        self.config_file_path = config_file_path
        self.logger = logger
    
    def on_save(self, args, state, control, model=None, **kwargs):
        """Called when a checkpoint is saved."""
        if model is None:
            return
        
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        full_model_dir = checkpoint_dir / "full_model"
        full_model_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Saving full model to checkpoint-{state.global_step}: {full_model_dir}")
        
        # Check if this is a PEFT model
        if hasattr(model, 'base_model'):
            try:
                # CRITICAL: merge_and_unload() only merges LoRA, but doesn't properly handle modules_to_save
                # We need to manually copy modules_to_save parameters to original_module before merging
                # Get the full state dict BEFORE merging (this is critical!)
                was_training = model.training
                model.eval()
                
                # IMPORTANT: Get state dict with modules_to_save parameters BEFORE merge_and_unload()
                # After merge_and_unload(), modules_to_save parameters are removed from state_dict
                # We need to access them through the PEFT model's get_peft_model_state_dict() method
                # Note: get_peft_model_state_dict() returns adapter state_dict including modules_to_save
                # We need the FULL model state_dict to get modules_to_save parameters
                from peft import get_peft_model_state_dict
                
                # Get the full model state_dict (includes base model + adapters + modules_to_save)
                # This is the complete state_dict before merging
                full_state_dict = model.state_dict()
                
                # Also get PEFT adapter state_dict for verification (includes modules_to_save)
                peft_state_dict = get_peft_model_state_dict(model)
                
                # Verify that modules_to_save parameters are in the full state_dict
                modules_to_save_keys_in_full = [k for k in full_state_dict.keys() if '.modules_to_save.default.' in k]
                if modules_to_save_keys_in_full:
                    self.logger.info(f"Found {len(modules_to_save_keys_in_full)} modules_to_save parameters in full state_dict")
                else:
                    self.logger.warning("⚠️  No modules_to_save parameters found in full state_dict! This may indicate a problem.")
                
                # Merge LoRA weights into base model first
                merged_model = model.merge_and_unload()
                self.logger.info(f"LoRA weights merged into base model (checkpoint-{state.global_step})")
                
                # Copy modules_to_save parameters to the merged model
                modules_copied = copy_modules_to_save_parameters(
                    full_state_dict, merged_model, self.logger
                )
                
                # Save full model
                merged_model.eval()
                merged_model.save_pretrained(str(full_model_dir))
                
                # Note: After merge_and_unload, base model weights are updated, but adapter structure still exists
                # Training can continue, but subsequent updates will affect the base model
                # Since checkpoint has already saved adapters, they will be restored on next load, so this is safe
                if was_training:
                    model.train()
                    
            except Exception as e:
                self.logger.warning(f"Failed to merge LoRA weights: {e}, skipping full model save")
                return
        else:
            merged_model = model
            merged_model.eval()
            merged_model.save_pretrained(str(full_model_dir))
        
        # Save model config
        if hasattr(merged_model, 'base_config'):
            merged_model.base_config.save_pretrained(str(full_model_dir))
        elif hasattr(merged_model, 'config'):
            merged_model.config.save_pretrained(str(full_model_dir))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(str(full_model_dir))
        
        # Save training config file
        if self.config_file_path and Path(self.config_file_path).exists():
            import shutil
            config_dest = full_model_dir / "training_config.yaml"
            shutil.copy2(self.config_file_path, config_dest)
        
        self.logger.info(f"Full model saved to: {full_model_dir} (includes all parameters)")


def save_full_model(
    model: NovoMolGen,
    tokenizer: AutoTokenizer,
    output_dir: Path,
    config: Dict[str, Any],
    config_file_path: str,
    logger: logging.Logger
) -> None:
    """
    Save full model (including all parameters: pretrained model, LoRA, cross-attention, encoders, etc.).
    
    For PEFT models, merges LoRA weights into the base model, then saves the full model.
    This way, loading doesn't require loading base model first and then adapters.
    """
    full_model_dir = output_dir / "full_model"
    full_model_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Saving full model to: %s", full_model_dir)
    logger.info("Including all parameters: pretrained model, LoRA, cross-attention, encoders, gates")
    
    # Check if this is a PEFT model
    if hasattr(model, 'base_model'):
        logger.info("Detected PEFT model, merging LoRA weights and modules_to_save into base model...")
        try:
            # CRITICAL: merge_and_unload() only merges LoRA, but doesn't properly handle modules_to_save
            # We need to manually copy modules_to_save parameters after merging
            # IMPORTANT: Get state dict with modules_to_save parameters BEFORE merge_and_unload()
            # After merge_and_unload(), modules_to_save parameters are removed from state_dict
            # We need the FULL model state_dict to get modules_to_save parameters
            # Note: model.state_dict() should include all parameters including modules_to_save
            full_state_dict = model.state_dict()
            
            # Verify that modules_to_save parameters are present
            modules_to_save_keys = [k for k in full_state_dict.keys() if '.modules_to_save.default.' in k]
            if modules_to_save_keys:
                logger.info(f"Found {len(modules_to_save_keys)} modules_to_save parameters in state_dict")
            else:
                logger.warning("⚠️  No modules_to_save parameters found in state_dict! This may indicate a problem.")
            
            # Merge LoRA weights into base model first
            merged_model = model.merge_and_unload()
            logger.info("LoRA weights merged into base model")
            
            # Copy modules_to_save parameters to the merged model
            modules_copied = copy_modules_to_save_parameters(
                full_state_dict, merged_model, logger
            )
        except Exception as e:
            logger.warning(f"Failed to merge LoRA weights: {e}, will save current model state")
            import traceback
            traceback.print_exc()
            if hasattr(model, 'base_model'):
                merged_model = model.base_model.model
            else:
                merged_model = model
    else:
        logger.info("Non-PEFT model, saving full model directly")
        merged_model = model
    
    # Save full model (including all parameters: pretrained + LoRA + cross-attention + encoders)
    merged_model.eval()
    merged_model.save_pretrained(str(full_model_dir))
    
    # Save model config
    if hasattr(merged_model, 'base_config'):
        merged_model.base_config.save_pretrained(str(full_model_dir))
    elif hasattr(merged_model, 'config'):
        merged_model.config.save_pretrained(str(full_model_dir))
    
    # Save tokenizer
    tokenizer.save_pretrained(str(full_model_dir))
    
    # Save training config file (input configuration file)
    if config_file_path and Path(config_file_path).exists():
        import shutil
        config_dest = full_model_dir / "training_config.yaml"
        shutil.copy2(config_file_path, config_dest)
        logger.info(f"Training config file saved: {config_dest}")
    
    logger.info("Full model saved to: %s", full_model_dir)
    logger.info("Contains: all model parameters + tokenizer + config file")
    logger.info("Can be used directly: NovoMolGen.from_pretrained('%s')", full_model_dir)


def create_trainer(
    model: NovoMolGen,
    tokenizer: AutoTokenizer,
    data_module,
    training_args: TrainingArguments,
    config: Dict[str, Any],
    config_file_path: str,
    logger: logging.Logger,
    run_name: Optional[str] = None
) -> Trainer:
    """Create trainer using the data module's built-in collator."""
    
    # Compute metrics function
    def compute_metrics(eval_preds):
        """
        Compute evaluation metrics.
        
        Note: eval/loss is automatically logged by Trainer, so we don't need to return it here.
        This function is for additional custom metrics (e.g., perplexity, accuracy, etc.).
        """
        predictions, labels = eval_preds
        metrics = {}

        return metrics
    
    # Use tokenizer from data_module if available (for MolDataModule), otherwise use provided tokenizer
    if hasattr(data_module, 'tokenizer') and data_module.tokenizer is not None:
        trainer_tokenizer = data_module.tokenizer
        logger.info("Using tokenizer from data_module")
    else:
        trainer_tokenizer = tokenizer
        logger.info("Using provided tokenizer")
    
    # Handle case where eval_dataset might be None
    eval_dataset = data_module.eval_dataset if hasattr(data_module, 'eval_dataset') and data_module.eval_dataset is not None else None
    
    # Create callbacks list
    callbacks = []
    
    # Add KLAnnealingCallback to update training step for KL annealing
    kl_annealing_callback = KLAnnealingCallback(logger)
    callbacks.append(kl_annealing_callback)
    logger.info("✅ Added KLAnnealingCallback to update training step for KL annealing")
    
    # Add LossMonitoringCallback to monitor NLL and KL losses
    loss_monitoring_callback = LossMonitoringCallback(logger)
    callbacks.append(loss_monitoring_callback)
    logger.info("✅ Added LossMonitoringCallback to track NLL loss and KL loss separately")

    # Add GateFixCallback to ensure gates are trainable
    gate_fix_callback = GateFixCallback(logger)
    callbacks.append(gate_fix_callback)
    logger.info("✅ Added GateFixCallback to monitor and fix cross-attention gates")

    # Add ProgressiveUnfreezingCallback if enabled
    progressive_unfreezing_callback = ProgressiveUnfreezingCallback(config, logger)
    if progressive_unfreezing_callback.enabled:
        callbacks.append(progressive_unfreezing_callback)
        logger.info("✅ Added ProgressiveUnfreezingCallback for progressive layer unfreezing")
    
    # Add WandbCallback if wandb_logs is enabled
    if config.get("wandb_logs", False):
        try:
            from src.callbacks import WandbCallback
            # Use run_name parameter if provided, otherwise use from config (which may include timestamp)
            wb_run_name = run_name if run_name else config.get("run_name", training_args.run_name or "finetune")
            wandb_callback = WandbCallback(
                model=model,
                entity=config.get("wandb_entity", os.getenv("WANDB_ENTITY", None)),
                project=config.get("wandb_project", "novomolgen-finetune"),
                name=wb_run_name,
                config=config,
                tags=config.get("wandb_tags", []),
                mode=config.get("wandb_mode", "online"),
                resume=config.get("wandb_resume", None),
                log_weight_grad_norm=config.get("wandb_log_weight_grad_norm", False),
            )
            callbacks.append(wandb_callback)
            logger.info("✅ Added WandbCallback for W&B logging")
            logger.info(f"   Project: {wandb_callback.project}, Name: {wandb_callback.name}")
        except Exception as e:
            logger.warning(f"Failed to create WandbCallback: {e}, falling back to default W&B integration")
    
    # Add callback to save full model at each checkpoint
    save_full_model_callback = SaveFullModelCallback(
        tokenizer=trainer_tokenizer,
        config=config,
        config_file_path=config_file_path,
        logger=logger
    )
    callbacks.append(save_full_model_callback)
    
    trainer = HFTrainer(
        model=model,
        args=training_args,
        train_dataset=data_module.train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_module.collate_fn,  # Use the data module's collator
        compute_metrics=compute_metrics,
        tokenizer=trainer_tokenizer,
        callbacks=callbacks,
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

    # Load appropriate model class based on config (tanimoto vs infonce)
    _load_model_class(config)

    # Set random seed for reproducibility
    seed = config.get("seed", 42)
    set_seed(seed)
    logger.info(f"Random seed set to: {seed}")
    
    # Generate unique run name if not provided
    if args.run_name is None:
        # Use config run_name as base, or default to "finetune"
        base_name = config.get("run_name", "finetune")
        # Add timestamp to make it unique
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"{base_name}_{timestamp}"
        logger.info(f"Generated unique run name: {args.run_name}")
    else:
        logger.info(f"Using provided run name: {args.run_name}")
    
    # Update config with the final run_name (for W&B and other uses)
    config["run_name"] = args.run_name
    
    # Setup W&B (sets environment variables, actual init is done by WandbCallback)
    setup_wandb(config, args.run_name)
    
    # TF32 opt-in (needs to be set before model creation)
    if torch.cuda.is_available() and config.get("tf32", False):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("TF32 enabled for faster training on Ampere GPUs")
    
    # DeepSpeed configuration will be passed to TrainingArguments
    # Note: DeepSpeed should be launched via deepspeed command, not directly
    
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
        
        # Create training arguments (use the run_name from config, which may have been updated)
        logger.info("Creating training arguments...")
        training_args = create_training_arguments(config, output_dir, logger)
        
        # Create trainer (use run_name from config, which includes timestamp if auto-generated)
        logger.info("Creating trainer...")
        trainer = create_trainer(model, tokenizer, data_module, training_args, config, args.config, logger, config.get("run_name"))
        
        # Start training
        logger.info("Starting training...")
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        
        # Save final model (PEFT format - only adapters, for resuming training)
        logger.info("Saving final model (PEFT format for resuming training)...")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir / "final_model")
        
        # Save full model (merged, including ALL parameters: pretrained + LoRA + cross-attention + encoders)
        logger.info("Saving full model (merged, including ALL parameters)...")
        save_full_model(model, tokenizer, output_dir, config, args.config, logger)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise
    finally:
        # Finish W&B run if it was initialized
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish()
        except (ImportError, AttributeError):
            pass  # W&B not available or not initialized


if __name__ == "__main__":
    main()
