import hashlib
import json
import os
from dataclasses import asdict
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import rootutils
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from datasets import load_dataset, Dataset, load_from_disk
from datasets.config import HF_CACHE_HOME
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from transformers.trainer_pt_utils import get_model_param_count
from transformers import DataCollatorForLanguageModeling
from dataclasses import dataclass

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.trainer.utils import PolicyTrainerConfig
from src.trainer.policy_trainer import PolicyTrainer
from src.models import NovoMolGen

from src.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class SFTConfig(PolicyTrainerConfig):
    type: str = "SFT"
    dataset_name: str = "MolGen/ZINC_250K_prop"
    top_reward_ratio: float = 0.25
    num_epochs: int = 10



class SFTTrainer(PolicyTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _prepare_sft_dataset(self):
        """
        Prepares a supervised fine-tuning dataset by selecting top-reward molecules.

        - Loads the dataset specified by `config.dataset_name`.
        - Selects the top-k molecules based on task-specific reward scores.
        - Tokenizes selected molecules using the tokenizer and returns a HuggingFace Dataset.

        Selection is based on top-reward ratio:
            top_k = int(total * config.top_reward_ratio)

        Returns:
            Dataset: Tokenized training dataset containing only high-reward molecules.
        """
        ds = load_dataset(self.config.dataset_name, split='train')
        rewards = torch.tensor(ds[f'{self.config.task_name}'])

        top_k = int(len(rewards) * self.config.top_reward_ratio)
        mask = torch.zeros_like(rewards, dtype=torch.bool)
        mask[torch.topk(rewards, top_k, largest=self.config.higher_is_better).indices] = True
        selected_mol = [ep for ep, m in zip(list(ds[self.model.mol_type]), mask) if m]
        train_set = Dataset.from_dict({self.model.mol_type: selected_mol})

        logger.info(f"select {len(train_set)} top molecules from dataset with top % score of {rewards.mean().item()}")

        def tokenize_function(
                element: dict,
                max_length: int,
                mol_type: str,
                tokenizer,
        ) -> dict:
            outputs = tokenizer(
                element[mol_type],
                truncation=True,
                max_length=max_length,
                padding="max_length",
                add_special_tokens=True,
            )
            return {"input_ids": outputs["input_ids"]}

        tokenized_train_dataset = train_set.map(
            tokenize_function,
            batched=True,
            num_proc=1,
            fn_kwargs={
                "max_length": self.config.max_length,
                "tokenizer": self.tokenizer,
                "mol_type": self.model.mol_type,
            },
        )

        return tokenized_train_dataset

    def _prepare_data_loader(self, dataset: Dataset, per_device_train_batch_size: int):
        """
        Prepares the DataLoader for SFT training.

        Removes unused columns, tokenizes inputs, and creates a DataLoader
        with random sampling (optional) and standard LM-style padding.

        Args:
            dataset (Dataset): The tokenized training set.
            per_device_train_batch_size (int): Batch size per device.

        Returns:
            DataLoader: The training data loader.
        """
        column_names = list(dataset.features)
        column_names = [c for c in column_names if c != "input_ids"]
        dataset_ = dataset.remove_columns(column_names)
        dataloader_params = {
            "batch_size": per_device_train_batch_size,
            "collate_fn": DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
            "num_workers": 1,
            "pin_memory": True,
        }
        if self.config.use_random_sampling:
            sampler = RandomSampler(dataset_)
            dataloader_params["sampler"] = sampler
        data_loader = DataLoader(dataset_, **dataloader_params)

        return data_loader


    def _compute_loss(self, logits, labels)-> Tuple[torch.FloatTensor, Dict[str, torch.Tensor]]:
        """
        Computes the negative log-likelihood (NLL) loss for language modeling.

        Uses the input as both context and target (causal LM):
            loss = -mean(log P_agent(x))

        KL is computed separately during training.

        Args:
            logits (Tensor): Model output logits.
            labels (Tensor): Token indices to predict.

        Returns:
            loss (Tensor): Scalar loss.
            metrics (dict): Empty, placeholder for extensibility.
        """
        sequence_log_probs = self.logprobs_from_logits(logits=logits, labels=labels)
        loss = -(sequence_log_probs.mean())
        metrics = {}

        return loss, metrics
    
    def _train_step(self,
                    inputs: Dict[str, torch.Tensor],
                    model: NovoMolGen,
                    ref_model: NovoMolGen,
                    optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler.LRScheduler,
                    accelerator: Accelerator) -> Dict[str, float]:
        """
        Executes one training step for supervised fine-tuning (SFT) with KL monitoring.

        Args:
            inputs (dict): Batch of input_ids.
            model (NovoMolGen): Trainable agent model.
            ref_model (NovoMolGen): Frozen reference model.
            optimizer: Optimizer for training.
            scheduler: LR scheduler.
            accelerator: Accelerator for gradient scaling.

        Returns:
            Dict[str, float]: Metrics for logging.
        """
        model.train()
        inputs = self._prepare_parallel_inputs(inputs)
        logits = model(inputs['input_ids']).logits
        with torch.no_grad():
            ref_logit = ref_model(inputs['input_ids']).logits

        loss, metrics = self._compute_loss(logits=logits, labels=inputs['input_ids'])
        kl = self._compute_kl(policy_logit=logits, ref_logit=ref_logit, input_ids=inputs['input_ids'])

        accelerator.backward(loss)
        if self.config.max_grad_norm is not None and self.config.max_grad_norm > 0:
            grad_norm = self.accelerator.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
        else:
            grad_norm = None
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        metrics.update({'train/loss': loss.item(), 'train/kl': kl.item(), 'train/learning_rate': scheduler.get_last_lr()[0]})
        if grad_norm:
            metrics['train/grad_norm'] = grad_norm

        return metrics

    def train(self, resume_from_checkpoint: bool = False):
        """
        Runs supervised fine-tuning on top-reward molecules.

        Selection of training data is biased toward high-reward molecules, 
        encouraging the agent to imitate top-performing examples in a supervised fashion.
        """
        train_set = self._prepare_sft_dataset()
        model, ref_model = self.model, self.ref_model

        per_device_train_batch_size = min(self.config.per_device_train_batch_size, len(train_set))
        train_dataloader = self._prepare_data_loader(train_set, self.config.per_device_train_batch_size)
        total_num_training_steps = (
                self.config.num_epochs
                * len(train_dataloader)
        )
        optimizer, scheduler = self._prepare_optimizer_and_scheduler(model=model,
                                                                     num_training_steps=total_num_training_steps)

        model, ref_model, optimizer, scheduler, train_dataloader = self.accelerator.prepare(model, ref_model, optimizer, scheduler, train_dataloader)

        logger.info(f"***** Running SFT on top molecules from {self.config.dataset_name} *****")
        logger.info(f"  Num Total Iterations = {total_num_training_steps:,}")
        logger.info(f"  Batch Size = {per_device_train_batch_size:,}")
        logger.info(f"  Number of Epochs = {self.config.num_epochs:,}")
        logger.info(f"  Number of Trainable Parameters = {get_model_param_count(model, trainable_only=True):,}")

        dataloader_iter = iter(train_dataloader)
        starting_epoch = 0
        should_stop = False

        progress_bar = tqdm(
            total=total_num_training_steps,
            disable=False,
            desc=f"Training Step",
            dynamic_ncols=True,
        )
        progress_bar.update(self.trainer_state.global_step)

        for epoch in range(starting_epoch, self.config.num_epochs):
            for step, inputs in enumerate(dataloader_iter):
                metrics = self._train_step(
                    inputs=inputs, 
                    model=model, 
                    ref_model=ref_model,
                    optimizer=optimizer, 
                    scheduler=scheduler,
                    accelerator=self.accelerator
                    )
                self._log_training_metrics(metrics, progress_bar=progress_bar)

                if self.trainer_state.global_step % self.config.eval_step == 0:

                    eval_metrics = self.compute_metrics(model=model)
                    self._log_training_metrics({f"eval/{k}": v for k, v in eval_metrics.items()})
                    if self.stopping_criteria.check(eval_metrics):
                        should_stop = True
                        break
                        
                self.trainer_state.global_step += 1
                progress_bar.update(1)

            if should_stop:
                logger.info("Stopping criteria met. Exiting training loop.")
                break
            # Recreate the dataloader iterator
            dataloader_iter = iter(train_dataloader)

        self._final_evaluation(model=model, save_path=self.config.output_dir)
        self._save_checkpoint(model=self.model, final_checkpoint=True)
        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")