from typing import Optional, Callable, Dict, List, Union, Tuple
import os
import pickle
import random
import re
import shutil
import warnings
from pathlib import Path
from typing import Optional, Callable, Dict, List, Union, Tuple

import numpy as np
import rootutils
import torch
import wandb
import json
import pandas as pd
from datasets import Dataset, load_dataset
from datasets.config import HF_CACHE_HOME
from datasets.naming import camelcase_to_snakecase
from rdkit.Chem import AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity, ExplicitBitVect, cDataStructs
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase
from transformers import set_seed
from transformers.trainer import OPTIMIZER_NAME, SCHEDULER_NAME, TRAINER_STATE_NAME
from transformers.trainer_pt_utils import reissue_pt_warnings
from transformers.trainer_utils import get_last_checkpoint, PREFIX_CHECKPOINT_DIR
from transformers.utils import WEIGHTS_NAME

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from accelerate import Accelerator
from src.eval.utils import get_mol, mapper
from src.models import NovoMolGen, generate_valid_smiles
from src.logging_utils import get_logger
from src.eval import MoleculeEvaluator

from src.trainer.utils import PolicyTrainerConfig, create_reference_model, disable_dropout_in_model, TrainerState, \
    StoppingCriteria

logger = get_logger(__name__)

MAXIMUM_TANIMATO_SIMILARITY = 0.4


class PolicyTrainer:
    """
    Base class for fine-tuning pretrained molecular language models using reinforcement learning (RL).

    This trainer coordinates model generation, reward computation, likelihood comparison against a fixed
    reference model, and policy optimization using various RL objectives (e.g., SFT, REINVENT, AugmentedHC).

    """
    def __init__(self,
                 config: Optional[PolicyTrainerConfig],
                 model: Optional[NovoMolGen],
                 tokenizer: Optional[PreTrainedTokenizerBase],
                 reward_fn: Optional[Callable] = None,
                 ref_model: Optional[NovoMolGen] = None,
                 n_jobs: int = None,
                 ):
        """
        Initializes the trainer with the model, tokenizer, reward function, and other utilities.

        Args:
            config: Training configuration with hyperparameters.
            model: The trainable molecular generation model.
            tokenizer: Tokenizer used for tokenizing SMILES strings.
            reward_fn: Callable that takes a list of SMILES and returns scores.
            ref_model: Optional fixed reference model (defaults to frozen copy of `model`).
            n_jobs: Number of parallel jobs for fingerprint computation and evaluation.
        """

        self.n_jobs = n_jobs
        self.config = config
        self.tokenizer = tokenizer
        assert reward_fn is not None, "reward function should be given"
        self.reward_fn = reward_fn

        for module in [model]:
            if module is not None:
                disable_dropout_in_model(module)
        self.model = model
        self.ref_model = ref_model if ref_model else create_reference_model(model)

        self.trainer_state = TrainerState()
        self.accelerator = Accelerator(mixed_precision=self.config.mixed_precision_dtype)
        self.accelerator.wait_for_everyone()
        from accelerate.utils import set_seed as acc_seed
        acc_seed(self.config.seed + self.accelerator.process_index)

        self.stopping_criteria = StoppingCriteria(self.config.stop_criteria_rules)
        self._dataset_fps: Optional[List[ExplicitBitVect]] = None
        os.makedirs(self.config.output_dir, exist_ok=True)

    def compute_metrics(self, model):
        """
        Computes novelty and uniqueness of generated molecules, and their rewards.

        Args:
            model: The model to use for generation.

        Returns:
            Dictionary with metrics like novelty, uniqueness, average reward, etc.
        """
        results = {'novelty': 0.0, 'uniqness': 0.0}
        model.eval()
        generated_smiles = generate_valid_smiles(
            model=model,
            tokenizer=self.tokenizer,
            batch_size=self.config.num_evaluation_samples,
            max_length=self.config.max_length,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            device=self.accelerator.device,
            return_canonical_unique=True,
        )['SMILES']
        if len(generated_smiles) < 1:
            return results

        novel_idx = self._get_novel_molecules(generated_smiles)
        results["uniqness"] = len(generated_smiles) / self.config.num_evaluation_samples
        results["novelty"] = len(novel_idx) / len(generated_smiles)

        if self.config.compute_metric_eval:
            true_rewards = self.reward_fn(generated_smiles)
            novel_true_rewards = true_rewards[novel_idx]

            results["reward"] = true_rewards.numpy()
            results["avg_reward"] = true_rewards.mean().item()
            results["top_1_reward"] = torch.topk(true_rewards, 1,
                                                 largest=self.config.higher_is_better).values.mean().item()
            results["top_10_reward"] = torch.topk(true_rewards, min(10, len(true_rewards)),
                                                  largest=self.config.higher_is_better).values.mean().item()
            results["top_100_reward"] = torch.topk(true_rewards, min(100, len(true_rewards)),
                                                   largest=self.config.higher_is_better).values.mean().item()
            results["novel_reward"] = novel_true_rewards.numpy()
            results["novel_top_10_reward"] = torch.topk(novel_true_rewards, min(10, len(novel_true_rewards)),
                                                        largest=self.config.higher_is_better).values.mean().item()

        return results

    def _get_novel_molecules(self, molecules: List[str]) -> List[int]:
        """Find novel molecules in a list of SMILES strings.

        Args:
            molecules: List of SMILES strings
            n_jobs: Number of parallel processing jobs

        Returns:
            List[int]: List of indices of novel molecules
        """

        def np_to_bitvect(arr: np.ndarray) -> ExplicitBitVect:
            # arr is your uint8 array of 0/1, length = nBits
            packed = np.packbits(arr)  # vectorized C routine
            bv = cDataStructs.CreateFromBinaryText(  # C++ routine :contentReference[oaicite:0]{index=0}
                packed.tobytes()
            )
            return bv

        def get_fingerprint(inputs):
            input_smile = inputs["SMILES"]
            mol = get_mol(input_smile)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)

            return {"fingerprint": np.asarray(fp, dtype="uint8")}

        def get_similarity(inputs, dataset_fps):
            try:
                bv = np_to_bitvect(inputs["fingerprint"])
                sims = BulkTanimotoSimilarity(bv, dataset_fps)
            except:
                sims = [1.0]
            return {'max_sims': float(max(sims))}

        if self._dataset_fps is None:
            _dat_path = self._get_cache_dir(self.config.dataset_name)
            dataset_path = os.path.join(HF_CACHE_HOME, "datasets", _dat_path, "fingerprints.pkl")
            if os.path.exists(dataset_path):
                with open(dataset_path, "rb") as f:
                    self._dataset_fps = pickle.load(f)
                logger.info(f"load fingerprints from: {dataset_path}")
            else:
                ds = load_dataset(self.config.dataset_name, split='train')
                # ds = Dataset.from_dict({"SMILES": ds[:10000]["SMILES"]})
                ds = ds.map(get_fingerprint, batched=False, num_proc=self.n_jobs, desc="compute fingerprints")
                logger.info(f"Convert numpy fingerprint arrays to RDKit bit vectors for training dataset")
                self._dataset_fps = mapper(self.n_jobs)(np_to_bitvect, ds['fingerprint'])
                with open(dataset_path, "wb") as f:
                    pickle.dump(self._dataset_fps, f, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(f"save fingerprints to: {dataset_path}")

        gen_ds = Dataset.from_dict({"SMILES": molecules})
        gen_ds = gen_ds.map(get_fingerprint, batched=False, num_proc=self.n_jobs, desc="compute fingerprints")
        gen_ds = gen_ds.map(get_similarity, batched=False, num_proc=self.n_jobs, desc="compute similarity",
                            fn_kwargs={"dataset_fps": self._dataset_fps})
        max_sims = gen_ds['max_sims']
        novel_idx = np.where(np.array(max_sims) < MAXIMUM_TANIMATO_SIMILARITY)[0]
        return novel_idx

    @staticmethod
    def _get_cache_dir(dataset_name):
        """
        Get cache directory.

        Args:
            dataset_name (str): Dataset name.

        Returns:
            str: Cache directory.
        """
        namespace_and_dataset_name = dataset_name.split("/")
        namespace_and_dataset_name[-1] = camelcase_to_snakecase(
            namespace_and_dataset_name[-1]
        )
        cached_relative_path = "___".join(namespace_and_dataset_name)
        return cached_relative_path

    @staticmethod
    def _prepare_parallel_inputs(inputs: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, torch.Tensor]]:
        device = torch.cuda.current_device()

        inputs_ = {}
        for key, value in inputs.items():
            if isinstance(value, Dict):
                inputs_[key] = {k: v.to(device) for k, v in value.items()}
            else:
                inputs_[key] = value.to(device)

        return inputs_

    def _prepare_optimizer_and_scheduler(self, model: torch.nn.Module, num_training_steps: int):
        """
        Prepares AdamW optimizer and learning rate scheduler with weight decay separation.

        Args:
            model: The model whose parameters should be optimized.
            num_training_steps: Total number of optimization steps.

        Returns:
            Tuple of (optimizer, scheduler).
        """

        if hasattr(self, "_optimizer") and hasattr(self, "_lr_scheduler"):
            optimzier = self._optimizer
            lr_scheduler = self._lr_scheduler

            return optimzier, lr_scheduler

        from transformers import get_scheduler
        from torch.optim import AdamW
        import math

        decay_parameters = self._get_decay_parameter_names(model)
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate)
        lr_scheduler = get_scheduler(
            self.config.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=math.ceil(num_training_steps * self.config.warmup_ratio),
            num_training_steps=num_training_steps,
            **self.config.lr_scheduler_kwargs,
        )

        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler

        return optimizer, lr_scheduler

    @staticmethod
    def _get_decay_parameter_names(model) -> List[str]:
        from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
        from transformers.trainer_pt_utils import get_parameter_names
        """
        Get all parameter names that weight decay will be applied to

        Note that some models implement their own layernorm instead of calling nn.LayerNorm, weight decay could still
        apply to those modules since this function only filter out instance of nn.LayerNorm
        """
        decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        return decay_parameters

    def logprobs_from_logits(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        Computes total (optionally length-normalized) log-probability of sequences from model logits.

        Args:
            logits: Model output logits of shape [B, T, V].
            labels: Target input_ids tensor of shape [B, T].

        Returns:
            Tensor of shape [B] with per-sequence log-probabilities.
        """
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Clone before modifying to avoid the in-place error
        shift_labels = shift_labels.clone()

        shift_label_mask = (shift_labels != self.tokenizer.pad_token_id).to(shift_logits.dtype)

        # Now safe to do in-place on the clone
        shift_labels[shift_labels == self.tokenizer.pad_token_id] = 0

        log_probs = shift_logits.log_softmax(-1)
        per_token_log_probs = torch.gather(log_probs, dim=2, index=shift_labels.unsqueeze(2))
        per_token_log_probs = per_token_log_probs.squeeze(2)

        per_token_log_probs = per_token_log_probs * shift_label_mask
        sequence_log_probs = per_token_log_probs.sum(dim=-1)
        if self.config.sequence_logp_reduction == "mean":
            sequence_log_probs = sequence_log_probs / shift_label_mask.sum(dim=-1)

        return sequence_log_probs

    def _compute_kl(self, policy_logit: torch.Tensor, ref_logit: torch.Tensor, input_ids: torch.Tensor):
        with torch.no_grad():
            # log‑softmax probabilities for policy & reference
            p_logp = policy_logit.log_softmax(-1)
            r_logp = ref_logit.log_softmax(-1)

            # KL per token
            kl_token = (p_logp.exp() * (p_logp - r_logp)).sum(-1)  # shape: [B, seq_len]

            # mask out padding, average over tokens, then over batch
            mask = (input_ids != self.tokenizer.pad_token_id).float()
            kl_seq = (kl_token * mask).sum(-1) / mask.sum(-1)
            true_kl = kl_seq.mean()

        return true_kl

    def _compute_loss(
            self,
    ) -> Tuple[torch.FloatTensor, Dict[str, torch.Tensor]]:

        raise NotImplementedError

    def _train_step(
            self,
    ) -> Dict[str, float]:

        raise NotImplementedError

    def train(self, resume_from_checkpoint: bool = False, skip_eval: bool = False):

        raise NotImplementedError

    def _log_training_metrics(
            self,
            metrics: Dict[str, Union[float, torch.Tensor]],
            progress_bar: tqdm = None,
    ):
        """
        Logs and records training metrics.

        Args:
            metrics (Dict[str, Union[float, torch.Tensor]]): Dictionary of metric names and values.
            progress_bar (tqdm, optional): Progress bar to update with latest metrics.
        """
        new_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                new_metrics[k] = v.item()
            else:
                new_metrics[k] = v
        logs = {**new_metrics, **{"train/global_step": self.trainer_state.global_step}}
        logs_ = {k: v for k, v in logs.items() if not isinstance(v, np.ndarray)}
        log_str = ', '.join(f'{k}: {v:.4f}' if isinstance(v, float) else f'{k}: {v}' for k, v in logs_.items())
        if progress_bar:
            progress_bar.set_postfix_str(log_str)
        else:
            logger.info(log_str)
        if wandb.run is not None:
            wandb.log(logs)

        logs_ = {}
        for k, v in logs.items():
            if isinstance(v, np.ndarray):
                logs_[k] = v.tolist()
            else:
                logs_[k] = v
        self.trainer_state.log_history.append(logs_)

    @staticmethod
    def _save_model(model, output_dir: str):
        """
        Saves the model weights to disk.

        Args:
            model: The model to save.
            output_dir (str): Directory to save the model weights to.
        """
        if not isinstance(model, NovoMolGen):
            state_dict = model.state_dict()
            torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            model.save_pretrained(output_dir)

    @staticmethod
    def _load_model(model, output_dir: str, device):
        """
        Loads model weights from a checkpoint directory.

        Args:
            model: The model to load weights into.
            output_dir (str): Directory containing saved weights.
            device: Device to map the loaded weights to.
        """
        weights_file = os.path.join(output_dir, WEIGHTS_NAME)
        state_dict = torch.load(weights_file, map_location=device)
        model.load_state_dict(state_dict)

    def _find_checkpoint_path(self, final_checkpoint: bool = False):
        """
        Determines the checkpoint directory path for saving the current model.

        Args:
            final_checkpoint (bool): If True, returns the final checkpoint directory path.

        Returns:
            str: Full path to the checkpoint directory.
        """
        if final_checkpoint:
            output_dir = self.config.output_dir
        else:
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.trainer_state.global_step}"
            output_dir = os.path.join(self.config.output_dir, checkpoint_folder)
        os.makedirs(output_dir, exist_ok=True)

        return output_dir

    @staticmethod
    def _save_rng_state(output_dir: str):
        """
        Saves the current state of all random number generators (Python, NumPy, PyTorch).

        Args:
            output_dir (str): Directory to save the RNG state file.
        """
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng_states["cuda"] = torch.cuda.random.get_rng_state()
        torch.save(rng_states, os.path.join(output_dir, "rng_state.pth"))

    def _save_checkpoint(self, model, final_checkpoint: bool = False, optimizer=None, lr_scheduler=None):
        """
        Saves a full training checkpoint, including model, optimizer, scheduler, RNG state, and trainer state.

        Args:
            model: The model to save.
            final_checkpoint (bool): Whether this is the final checkpoint.
            optimizer (optional): Optimizer to save.
            lr_scheduler (optional): Scheduler to save.
        """
        output_dir = self._find_checkpoint_path(final_checkpoint=final_checkpoint)

        # save models and tokenizer
        model.save_pretrained(os.path.join(output_dir, "model"))

        # Save optimizer and scheduler
        if optimizer is not None:
            torch.save(optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
        if lr_scheduler is not None:
            with warnings.catch_warnings(record=True) as caught_warnings:
                torch.save(lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
            reissue_pt_warnings(caught_warnings)

        # Save RNG state
        self._save_rng_state(output_dir=output_dir)

        # Save trainer state
        self.trainer_state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        logger.info(f"save checkpoints to: {output_dir}")

    def _rotate_checkpoints(self, use_mtime=False) -> None:
        """
        Removes older checkpoints if the total number exceeds `save_total_limit`.

        Args:
            use_mtime (bool): Whether to sort checkpoints by modified time (default: False).
        """
        if self.config.save_total_limit is None or self.config.save_total_limit <= 0:
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=use_mtime, output_dir=self.config.output_dir)
        if len(checkpoints_sorted) <= self.config.save_total_limit:
            return

        # If save_total_limit=1 with load_best_model_at_end=True, we could end up deleting the last checkpoint, which
        # we don't do to allow resuming.
        save_total_limit = self.config.save_total_limit

        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            logger.info(f"Deleting older checkpoint [{checkpoint}] due to config.save_total_limit")
            shutil.rmtree(checkpoint, ignore_errors=True)

    @staticmethod
    def _sorted_checkpoints(
            output_dir=None, checkpoint_prefix=PREFIX_CHECKPOINT_DIR, use_mtime=False
    ) -> List[str]:
        """
        Returns a sorted list of existing checkpoint directories.

        Args:
            output_dir (str): Path to the checkpoint root directory.
            checkpoint_prefix (str): Prefix of the checkpoint folders.
            use_mtime (bool): If True, sort by modified time instead of step index.

        Returns:
            List[str]: Sorted list of checkpoint directory paths.
        """
        ordering_and_checkpoint_path = []

        glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"{checkpoint_prefix}-*") if os.path.isdir(x)]

        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
            else:
                regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
                if regex_match is not None and regex_match.groups() is not None:
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]

        return checkpoints_sorted

    def _load_checkpoint(self, model, optimizer=None, scheduler=None):
        """
        Loads a full training checkpoint, including model, optimizer, scheduler, RNG state, and trainer state.

        Args:
            model: The model to load weights into.
            optimizer (optional): Optimizer to restore.
            scheduler (optional): Scheduler to restore.
        """
        resume_from_checkpoint = get_last_checkpoint(self.config.output_dir)
        if resume_from_checkpoint is None:
            raise ValueError(f"No valid checkpoint found in output directory ({self.config.output_dir})")

        logger.info(f" >>>>>>>>>> Resuming from checkpoint: {resume_from_checkpoint} <<<<<<<<<< ")
        # load model
        self._load_model(model, output_dir=os.path.join(resume_from_checkpoint, "model"),
                         device=self.accelerator.device)
        logger.info(f">>> load model from: {os.path.join(resume_from_checkpoint, 'model')}")

        # load trainer state
        self.trainer_state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
        logger.info(f">>> load trainer_state from: {os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)}")

        # load optimizer
        if optimizer is not None:
            optimizer.load_state_dict(
                torch.load(os.path.join(resume_from_checkpoint, OPTIMIZER_NAME), map_location=self.accelerator.device))
            logger.info(f">>> load optimizer from: {os.path.join(resume_from_checkpoint, OPTIMIZER_NAME)}")

        # load scheduler
        if scheduler is not None:
            scheduler.load_state_dict(torch.load(os.path.join(resume_from_checkpoint, SCHEDULER_NAME)))
            logger.info(f">>> load scheduler from: {os.path.join(resume_from_checkpoint, SCHEDULER_NAME)}")

        # load rng states
        rng_file = os.path.join(resume_from_checkpoint, "rng_state.pth")
        checkpoint_rng_state = torch.load(rng_file)
        random.setstate(checkpoint_rng_state["python"])
        np.random.set_state(checkpoint_rng_state["numpy"])
        torch.random.set_rng_state(checkpoint_rng_state["cpu"])
        if torch.cuda.is_available():
            torch.cuda.random.set_rng_state(checkpoint_rng_state["cuda"])
        logger.info(f">>> load rng states from: {os.path.join(resume_from_checkpoint, 'rng_state.pth')}")


    def _final_evaluation(self, model: NovoMolGen, save_path: str):
        """
        Generates 10,000 molecules using the trained model and evaluates them for reward, SA, and IntDiv.

        Saves the generated molecules and their scores to disk and logs results to Weights & Biases.

        Args:
            model (NovoMolGen): The trained model to evaluate.
            save_path (str): Directory to save evaluation results.
        """
        model.eval()
        generated_smiles = []
        while (len(generated_smiles)<10_000):
            outputs = generate_valid_smiles(
                model=model,
                tokenizer=self.tokenizer,
                batch_size=2000,
                max_length=self.config.max_length,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                device=self.accelerator.device,
                return_canonical_unique=True,
            )
            generated_smiles += outputs['SMILES']

        generated_smiles = generated_smiles[:10_000]
        scores = self.reward_fn(generated_smiles)
        logger.info(f"Generate and evalute {len(generated_smiles)} valid molecules.")

        final_results = {
            'SMILES': generated_smiles,
            'Score': scores.tolist(),
        }
        json_string = json.dumps(final_results, indent=2, sort_keys=True) + "\n"
        with open(os.path.join(save_path, "generated_molecules.json"), "w", encoding="utf-8") as f:
            f.write(json_string)

        top_scores_idx = torch.topk(scores, 100, largest=self.config.higher_is_better).indices
        top_generated_smiles = np.array(generated_smiles)[top_scores_idx].tolist()
        top_scores = scores[top_scores_idx]

        evaluator = MoleculeEvaluator(task_names=['SA', 'IntDiv'], batch_size=512, n_jobs=self.n_jobs, device='cpu')
        res = evaluator(top_generated_smiles, filter=True)
        sa_scores = torch.tensor(res['SA'])
        intdiv_scores = torch.tensor(res['IntDiv'])

        novel_idx = self._get_novel_molecules(top_generated_smiles)
        novel_top_scores = top_scores[novel_idx]

        final_results = {
            'SMILES': top_generated_smiles,
            'Score': top_scores.tolist(),
            'SA': sa_scores.tolist(),
            'IntDiv': intdiv_scores.tolist(),
            'Novel': [x in  novel_idx for x in np.arange(len(top_generated_smiles))],
        }
        
        df = pd.DataFrame.from_dict(final_results)
        df = (df.sort_values("Score", ascending=not self.config.higher_is_better).reset_index(drop=True))
        wandb_table = wandb.Table(dataframe=df)
        if wandb.run is not None:
            wandb.log({"eval/results": wandb_table})

        
        metrics = {
            'eval/final/IntDiv': intdiv_scores.mean().item(),
            'eval/final/SA_avg': sa_scores.mean().item(),
            'eval/final/top_10_reward':torch.topk(scores, 10,largest=self.config.higher_is_better).values.mean().item(),
            'eval/final/top_100_reward':torch.topk(scores, 100,largest=self.config.higher_is_better).values.mean().item(),
            'eval/final/novel_top_10_reward':torch.topk(novel_top_scores, min(10, len(novel_top_scores)),largest=self.config.higher_is_better).values.mean().item(),
            'eval/final/reward': scores.numpy()
        }
        self._log_training_metrics(metrics=metrics)
        

