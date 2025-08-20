import os
import warnings
from pathlib import Path

import fire
import rootutils
import torch

from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data_loader import MolDataModule
from src.eval import MoleculeEvaluator
from src.trainer import HFTrainer, HFTrainingArguments
from src.logging_utils import get_logger
from utils import (
    creat_unique_experiment_name,
    creat_unique_experiment_name_for_finetune,
    get_real_cpu_cores,
    init_model,
    _build_callbacks,
    _checkpoint_exists,
    _load_cfg,
    _prepare_base_model,
    _FINETUNE_REGISTRY,
)

logger = get_logger(__name__)

def run_tokenize(cfg: DictConfig) -> None:
    """
    Tokenise the raw dataset once and write the Arrow cache to
    `MolDataModule.save_directory`.  If the cache already exists, we’ll
    just notify the user and exit.

    The function works with either

      • a *pure* dataset YAML (keys match the MolDataModule signature), or
      • a full experiment YAML that has a top-level `dataset` block.
    """
    # extract the kwargs for MolDataModule
    dm_kwargs = dict(cfg) if "dataset_name" in cfg else dict(cfg.dataset)
    dm = MolDataModule(**dm_kwargs)

    cache_dir = Path(dm.save_directory)
    if cache_dir.exists():
        print(f"Tokenised cache already present → {cache_dir}")
        return

    # build and save
    print(f"Tokenising {dm.dataset_name} and writing to {cache_dir} …")
    dm.create_tokenized_datasets()
    print(f"Done.  Train={len(dm.train_dataset)}  |  "
          f"Eval={dm.eval_dataset and len(dm.eval_dataset) or 0}")

def run_train(cfg: DictConfig) -> None:
    exp_name = creat_unique_experiment_name(cfg)
    output_dir = Path(cfg.save_path, exp_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    # HF training args
    train_args = HFTrainingArguments(
        data_seed=cfg.seed,
        seed=cfg.seed,
        output_dir=str(output_dir),
        hub_token=os.getenv("HF_TOKEN", None),
        **cfg.trainer,
    )
    if train_args.dataloader_num_workers > get_real_cpu_cores():
        warnings.warn("dataloader_num_workers > CPU cores; reducing.")

    # TF32 opt-in
    if torch.cuda.is_available() and cfg.trainer.get("tf32", False):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Data
    if cfg.dataset.num_proc > get_real_cpu_cores():
        warnings.warn("dataset.num_proc > CPU cores; reducing.")
    dm = MolDataModule(**cfg.dataset)
    dm.load_tokenized_dataset()

    # Model
    model = init_model(
        cfg.model,
        max_seq_length=dm.max_seq_length,
        vocab_size=dm.tokenizer.vocab_size,
        eos_token_id=dm.tokenizer.eos_token_id,
        bos_token_id=dm.tokenizer.bos_token_id,
        mol_type=dm.mol_type,
    )

    # Callbacks & trainer
    trainer = HFTrainer(
        model=model,
        args=train_args,
        callbacks=_build_callbacks(cfg, model, exp_name),
        tokenizer=dm.tokenizer,
        data_collator=dm.data_collator,
        train_dataset=dm.train_dataset,
        eval_dataset=dm.eval_dataset,
    )

    resume = _checkpoint_exists(train_args.output_dir)
    train_result = trainer.train(resume_from_checkpoint=resume)
    logger.info("Training finished: %s", train_result.metrics)
    trainer.save_model()

def run_finetune(cfg: DictConfig) -> None:
    exp_name, base_exp_name, base_output_dir = creat_unique_experiment_name_for_finetune(
        cfg, include_finetune_hash_name=True
    )
    exp_name = f"{cfg.finetune.type}_{exp_name}"
    output_dir = Path(cfg.finetune.save_path, exp_name)

    logger.info("Loading base experiment: %s", base_exp_name)
    model, tokenizer = _prepare_base_model(cfg)

    # Reward function
    n_jobs = get_real_cpu_cores()
    task = MoleculeEvaluator(
        task_names=[cfg.finetune.task_name], batch_size=128, n_jobs=n_jobs
    )

    def reward_fn(smiles):
        res = task(smiles, filter=True, return_valid_index=True)
        reward = torch.zeros(len(smiles))
        reward[res["valid_index"]] = torch.tensor(
            res[cfg.finetune.task_name], dtype=torch.float32
        )
        # Convert docking scores (lower is better) to negative reward
        if cfg.finetune.task_name.startswith("docking"):
            reward[reward > 0] = 0
        return reward

    # Select trainer/config via registry
    try:
        CfgCls, TrainerCls = _FINETUNE_REGISTRY[cfg.finetune.type]
    except KeyError as e:
        raise NotImplementedError(f"Unsupported finetune type: {e}")

    finetune_cfg = CfgCls(output_dir=str(output_dir), **cfg.finetune)
    finetune_cfg.max_length = cfg.dataset.max_seq_length  # enforce consistency

    trainer = TrainerCls(
        config=finetune_cfg,
        model=model,
        reward_fn=reward_fn,
        tokenizer=tokenizer,
    )

    # Optional WandB
    if cfg.get("wandb_logs", False):
        import wandb

        wandb.init(
            entity=os.getenv("WANDB_ENTITY"),
            project=os.getenv("WANDB_PROJECT"),
            name=exp_name,
            config=OmegaConf.to_container(cfg),
            tags=os.getenv("WANDB_TAGS", "").split(","),
            mode=os.getenv("WANDB_MODE", "online"),
            resume=True,
        )

    trainer.train(resume_from_checkpoint=False)

class EntryPoint:
    def train(
        self,
        config_name: str = "train_ZINC_1B_atomwise_smiles_llama-32M",
        config_dir: str = "../configs",
        **overrides,
    ):
        cfg = _load_cfg(config_name, config_dir, **overrides)
        run_train(cfg)

    def finetune(
        self,
        config_name: str = "finetune_PMO_ZINC_1B_atomwise_smiles_llama-32M",
        config_dir: str = "../configs",
        **overrides,
    ):
        cfg = _load_cfg(config_name, config_dir, **overrides)
        run_finetune(cfg)

    def tokenize_dataset(
        self,
        config_name: str = "ZINC_1B_smiles_atomwise",   # a dataset-only YAML
        config_dir: str = "../configs/dataset",
        **overrides,
    ):
        cfg = _load_cfg(config_name, config_dir, **overrides)
        run_tokenize(cfg)

if __name__ == "__main__":
    fire.Fire(EntryPoint)
