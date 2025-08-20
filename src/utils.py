import copy
import hashlib
import json
import os
import subprocess
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple, Type, Union

import datasets
import psutil
import rootutils
import torch
import transformers
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.callbacks import Evaluator, WandbCallback
from src.data_loader import MoleculeTokenizer
from src.logging_utils import get_logger
from src.models import NovoMolGen, NovoMolGenConfig
from src.trainer import (
    HFTrainingArguments,          # only for type hints
    SFTConfig, SFTTrainer,
    REINVENTConfig, REINVENTTrainer,
    AugmentedHCConfig, AugmentedHCTrainer,
)

logger = get_logger(__name__)


def unroll_configs(cfg: Dict[str, Any], parent_key='', sep='_') -> Dict[str, Any]:
    """
    Recursively unroll a nested dictionary of configurations and remove keys with None values.

    Args:
        cfg (Dict[str, Any]): The input dictionary containing configuration options.
        parent_key (str): The parent key for the current level of recursion.
        sep (str): The separator used to separate parent and child keys.

    Returns:
        Dict[str, Any]: The output unrolled dictionary.
    """
    items = {}
    for key, value in cfg.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.update(unroll_configs(value, new_key, sep=sep))
        elif value is not None:  # Exclude keys with None values
            items[new_key] = value
    return items


def creat_unique_experiment_name(config: DictConfig,
                                 ) -> str:
    """
    Generate a unique experiment name based on the provided configurations.

    Args:
        config (Dict[str, Any]): The input dictionary containing experiment configurations.

    Returns:
        str: A unique experiment name.
    """
    _config = OmegaConf.to_container(copy.deepcopy(config))
    if 'eval' in _config.keys():
        _config.pop('eval', None)
    model_arch = _config['model']['model_type']
    data_name = Path(_config['dataset']['dataset_name']).name
    if 'tokenizer_name' in _config['dataset'].keys():
        _tok_name = Path(_config['dataset']['tokenizer_name']).name
    else:
        _tok_name = Path(_config['dataset']['tokenizer_path']).name
    _tok_name = _tok_name.replace(f"_{data_name}", "").replace(".json", "").replace("tokenizer_wordlevel_", "").replace(
        "_30000_0_0", "").replace(" ", "_")
    if 'label_smoothing_factor' in _config['trainer'].keys():
        post_fix = '_smooth'
    else:
        post_fix = ''
    _config = unroll_configs(_config)
    # Convert the unrolled dictionary to a JSON string and hash it
    unrolled_json = json.dumps(_config, sort_keys=True)
    hash_name = hashlib.md5(unrolled_json.encode()).hexdigest()[:8]
    exp_name = f"{model_arch}_{data_name}_{_tok_name}_{hash_name}"
    exp_name += post_fix
    return exp_name


def creat_unique_experiment_name_for_finetune(
        config: DictConfig,
        include_finetune_hash_name: bool = True,
):
    """
    Generate a unique experiment name based on the provided configurations.

    Args:
        config (Dict[str, Any]): The input dictionary containing experiment configurations.
        include_finetune_hash_name (bool): whether to include finetune spec name or not 

    Returns:
        exp_name, base_exp_name, base_output_dir
    """
    _config = OmegaConf.to_container(copy.deepcopy(config))

    finetune_config = _config['finetune']
    finetune_target = _config['finetune']['task_name']
    finetune_target = finetune_target.replace(" ", "_")
    finetune_checkpoint = _config['finetune']['checkpoint']
    mol_type = _config['dataset']['mol_type']
    mol_type = mol_type.replace(" ", "_")

    if 'eval' in _config.keys():
        _config.pop('eval', None)
    if 'finetune' in _config.keys():
        _config.pop('finetune', None)

    model_arch = _config['model']['model_type']
    data_name = Path(_config['dataset']['dataset_name']).name
    if 'tokenizer_name' in _config['dataset'].keys():
        _tok_name = Path(_config['dataset']['tokenizer_name']).name
    else:
        _tok_name = Path(_config['dataset']['tokenizer_path']).name
    _tok_name = _tok_name.replace(f"_{data_name}", "").replace(".json", "").replace("tokenizer_wordlevel_", "").replace(
        "_30000_0_0", "").replace(" ", "_")
    _config = unroll_configs(_config)
    # Convert the unrolled dictionary to a JSON string and hash it
    unrolled_json = json.dumps(_config, sort_keys=True)
    hash_name = hashlib.md5(unrolled_json.encode()).hexdigest()[:8]
    base_exp_name = f"{model_arch}_{data_name}_{_tok_name}_{hash_name}"
    base_output_dir = os.path.join(_config['save_path'], base_exp_name)

    finetune_config = unroll_configs(finetune_config)
    finetune_unrolled_json = json.dumps(finetune_config, sort_keys=True)
    finetune_hash_name = hashlib.md5(finetune_unrolled_json.encode()).hexdigest()[:8]

    if include_finetune_hash_name:
        if isinstance(finetune_checkpoint, str):
            finetune_checkpoint = Path(finetune_checkpoint).name
            if hash_name in finetune_checkpoint:
                exp_name = f"{finetune_checkpoint}-{finetune_hash_name}"
            else:
                exp_name = f"{mol_type}-{hash_name}-{finetune_target}-{finetune_hash_name}"
        else:
            exp_name = f"{mol_type}-{hash_name}-{finetune_target}-{finetune_hash_name}"
    else:
        if isinstance(finetune_checkpoint, str):
            finetune_checkpoint = Path(finetune_checkpoint).name
            if hash_name in finetune_checkpoint:
                exp_name = f"{finetune_checkpoint}"
            else:
                exp_name = f"{mol_type}-{hash_name}-{finetune_target}-{finetune_checkpoint}"
        else:
            exp_name = f"{mol_type}-{hash_name}-{finetune_target}-{finetune_checkpoint}"

    return exp_name, base_exp_name, base_output_dir


# code from: https://github.com/huggingface/transformers/blob/bd50402b56980ff17e957342ef69bd9b0dd45a7b/src/transformers/trainer.py#L2758
def is_world_process_zero(train_args) -> bool:
    """
    Whether this process is the global main process (when training in a distributed fashion on several
    machines, this is only going to be `True` for one process).
    """
    # Special case for SageMaker ModelParallel since there process_index is dp_process_index, not the global
    # process index.
    from transformers.utils.import_utils import is_sagemaker_mp_enabled
    if is_sagemaker_mp_enabled():
        import smdistributed.modelparallel.torch as smp
        return smp.rank() == 0
    else:
        return train_args.process_index == 0


def get_real_cpu_cores() -> int:
    """Return the number of CPU *cores* (not HT threads)."""
    try:
        return int(subprocess.run(["nproc"], stdout=subprocess.PIPE, text=True).stdout.strip())
    except Exception as e:  # pragma: no cover
        logger.warning("Falling back to psutil.cpu_count(): %s", e)
        return psutil.cpu_count(logical=False)

def init_model(
    model_cfg: DictConfig,
    *,
    max_seq_length: int,
    vocab_size: int,
    eos_token_id: int,
    bos_token_id: int,
    mol_type: str,
) -> NovoMolGen:
    """Factory for `NovoMolGen` models."""
    cfg_dict  = OmegaConf.to_container(model_cfg)
    model_typ = cfg_dict.get("model_type", "llama").lower()

    if model_typ == "llama":
        conf = NovoMolGenConfig(
            **cfg_dict,
            max_seq_length=max_seq_length,
            vocab_size=vocab_size,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
        )
        return NovoMolGen(conf, mol_type=mol_type)

    raise ValueError(f"Unsupported model_type={model_typ!r}")

def _checkpoint_exists(output_dir: str) -> bool:
    """Detect an HF checkpoint directory."""
    return any(Path(output_dir, p).is_dir() and p.startswith("checkpoint")
               for p in os.listdir(output_dir))

def _load_cfg(
    config_name: str,
    config_dir: str,
    **overrides: Any,
) -> DictConfig:
    """Hydra config loader that supports CLI key=value overrides."""
    with initialize(version_base=None, config_path=config_dir):
        ov = [f"{k}={v}" for k, v in overrides.items()]
        return compose(config_name=config_name, overrides=ov)


def _build_callbacks(
    cfg: DictConfig,
    model,
    exp_name: str,
) -> list:
    """Assemble callbacks (evaluator, WandB, …) once."""
    cbs = []

    if "eval" in cfg:
        if cfg.eval.n_jobs > get_real_cpu_cores():
            warnings.warn("Reducing eval n_jobs to available CPU cores")
        cbs.append(Evaluator(**cfg.eval))

    if cfg.get("wandb_logs", False):
        import wandb  # local import to avoid dependency when disabled

        cbs.append(
            WandbCallback(
                model=model,
                entity=os.getenv("WANDB_ENTITY"),
                project=os.getenv("WANDB_PROJECT"),
                name=exp_name,
                config=OmegaConf.to_container(cfg),
                tags=os.getenv("WANDB_TAGS", "").split(","),
                resume=True,
                mode=os.getenv("WANDB_MODE", "online"),
            )
        )
    return cbs


_FINETUNE_REGISTRY: Dict[
    str, Tuple[Type[SFTConfig], Type[SFTTrainer]]
] = {
    "SFT": (SFTConfig, SFTTrainer),
    "REINVENT": (REINVENTConfig, REINVENTTrainer),
    "AugmentedHC": (AugmentedHCConfig, AugmentedHCTrainer),
}


def _prepare_base_model(config: DictConfig) -> Tuple[NovoMolGen, Any]:
    """Load checkpoint or fresh model + tokenizer."""
    tokenizer = MoleculeTokenizer.load(
        config.dataset.tokenizer_path
    ).get_pretrained()

    if isinstance(config.finetune.checkpoint, str):
        model = NovoMolGen.from_pretrained(config.finetune.checkpoint)
        logger.info("Loaded model from %s", config.finetune.checkpoint)
    elif config.finetune.checkpoint == 0:
        model = init_model(
            config.model,
            max_seq_length=config.dataset.max_seq_length,
            vocab_size=tokenizer.vocab_size,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            mol_type=config.dataset.mol_type,
        )
        logger.info("Initialized new model from scratch")
    else:
        base_name = creat_unique_experiment_name(config)
        ckpt = f"tmp-spec-checkpoint-{config.finetune.checkpoint}"
        model = NovoMolGen.from_pretrained(f"MolGen/{base_name}", checkpoint_path=ckpt)
        logger.info("Loaded model from MolGen/%s @ %s", base_name, ckpt)

    # Keep mol_type in sync with dataset
    if model.mol_type != config.dataset.mol_type:
        logger.info("Overriding model.mol_type ➜ %s", config.dataset.mol_type)
        model.mol_type = config.dataset.mol_type

    return model, tokenizer