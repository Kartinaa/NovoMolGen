import json
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import rootutils
import torch
import wandb
from accelerate import Accelerator
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase, DataCollatorForLanguageModeling
from transformers.trainer_pt_utils import get_model_param_count

from rdkit.Chem import AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity, ExplicitBitVect, cDataStructs

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.trainer.utils import PolicyTrainerConfig
from src.trainer.policy_trainer import PolicyTrainer
from src.models import NovoMolGen, generate_valid_smiles
from src.eval import MoleculeEvaluator
from src.eval.molecule_evaluation import top_auc
from src.eval.utils import get_mol, mapper
from src.trainer.policy_trainer import MAXIMUM_TANIMATO_SIMILARITY
from src.logging_utils import get_logger

logger = get_logger(__name__)


class Experience:
    """
    Prioritized replay buffer that remembers highest scored sequences
    (SMILES, score, prior_likelihood) and samples in proportion to score.

    - Uses torch for sampling (torch.multinomial).
    - Uses a Hugging Face PreTrainedTokenizerFast for tokenization.
    - Automatically creates a DataCollatorForLanguageModeling with mlm=False.
    """

    def __init__(self,
    tokenizer: PreTrainedTokenizerBase,
    max_size: int = 100,
    sampling: str = "weighted",
    seed: int = 42,
    ):
        """
        Args:
          tokenizer: A Hugging Face tokenizer (PreTrainedTokenizerFast),
                     already set up for your SMILES or plain text.
          max_size: Maximum number of (SMILES, score, prior_likelihood) entries to keep.
          sampling: Sampling method which by deafult is weighted by the importance of the scores. 
        """
        self.memory = []  # will hold [(smiles_str, score, prior_likelihood), ...]
        self.max_size = max_size
        self.tokenizer = tokenizer
        self.sampling = sampling
        self.g = torch.Generator(device='cpu').manual_seed(seed)

        # Create a data collator for LM tasks, but with mlm=False by default
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

    def add_experience(self, experience):
        """
        Adds a list of (smiles, score, prior_likelihood) tuples to memory, then:
          - Removes duplicates
          - Retains top max_size by descending score
        """
        self.memory.extend(experience)

        if len(self.memory) > self.max_size:
            # Deduplicate by SMILES string
            unique_smiles = set()
            unique_data = []
            for sm, sc, pl in self.memory:
                if sm not in unique_smiles:
                    unique_smiles.add(sm)
                    unique_data.append((sm, sc, pl))

            # Sort by descending score and truncate
            unique_data.sort(key=lambda x: x[1], reverse=True)
            self.memory = unique_data[:self.max_size]

    def sample(self, n: int):
        """
        Sample 'n' items from memory in proportion to their score.

        Returns:
          batch_dict: A dictionary (from the DataCollatorForLanguageModeling)
                      containing 'input_ids', 'attention_mask', etc.
          scores_t:    1D FloatTensor of shape [n]
          prior_t:     1D FloatTensor of shape [n]

        Typical usage in training:
          batch_dict, scores, priors = exp.sample(32)
          # Move them to GPU or pass to model:
          batch_dict = {k: v.cuda() for k, v in batch_dict.items()}
          scores = scores.cuda()
          priors = priors.cuda()
        """
        if len(self.memory) < n:
            raise IndexError(f"Not enough memory to sample {n} items (have {len(self.memory)}).")

        if self.sampling == "weighted":
            weights = torch.tensor([m[1] for m in self.memory], dtype=torch.float) + 1e-10
            sample_indices = torch.multinomial(weights, n, replacement=False, generator=self.g)

        elif self.sampling == "uniform":
            sample_indices = torch.randperm(len(self.memory), generator=self.g)[:n]

        else:
            raise ValueError(f"Unknown sampling mode: {self.sampling}")

        # Gather the sampled items
        sampled = [self.memory[i] for i in sample_indices.tolist()]
        smiles_list = [x[0] for x in sampled]
        scores_list = [x[1] for x in sampled]
        prior_list = [x[2] for x in sampled]

        # Tokenize SMILES in a batch
        encodings = self.tokenizer(
            smiles_list,
            return_tensors='pt',
            padding=True,  # pad to longest in batch
            truncation=True  # truncate if beyond model's max length
        )

        # Convert encodings dict -> list of dicts for data collator
        batch_for_collator = []
        for i in range(encodings['input_ids'].size(0)):
            example_dict = {}
            for key in encodings.keys():
                example_dict[key] = encodings[key][i]
            batch_for_collator.append(example_dict)

        # Collate via data collator (lm=False => 'labels' = None by default)
        batch_dict = self.data_collator(batch_for_collator)

        # Tensors for scores & prior
        scores_t = torch.tensor(scores_list, dtype=torch.float)
        prior_t = torch.tensor(prior_list, dtype=torch.float)

        return batch_dict, scores_t, prior_t

    def print_memory(self, path):
        """
        Prints the top 100 SMILES stored in memory (by descending score),
        shows up to 50 on-screen, and writes all 100 to file.
        """
        print("\n" + "*" * 80 + "\n")
        print("         Best recorded SMILES:\n")
        print("Score     Prior log P     SMILES\n")

        # The memory should already be sorted by descending score,
        # but if not, ensure it is here:
        sorted_memory = sorted(self.memory, key=lambda x: x[1], reverse=True)

        with open(path, 'w') as f:
            f.write("SMILES Score PriorLogP\n")
            for i, exp in enumerate(sorted_memory[:100]):
                # exp = (smiles, score, prior_likelihood)
                smiles_str = exp[0]
                score_val = exp[1]
                prior_val = exp[2]

                # Print up to 50 on-screen
                if i < 50:
                    print("{:4.2f}   {:6.2f}        {}".format(score_val, prior_val, smiles_str))

                # Write all 100 to file
                f.write("{} {:4.2f} {:6.2f}\n".format(smiles_str, score_val, prior_val))

        print("\n" + "*" * 80 + "\n")

    def get_top_smiles(self, topk: int = 100):

        sorted_memory = sorted(self.memory, key=lambda x: x[1], reverse=True)
        smiles_list = [x[0] for x in sorted_memory[:topk]]
        return smiles_list

    def __len__(self):
        return len(self.memory)


@dataclass
class REINVENTConfig(PolicyTrainerConfig):
    type: str = "REINVENT"
    oracle_call_budget: int = 10000
    """Oracle call budget"""
    penalty_coef: float = 5 * 1e3
    """ Penalty coefficient for regularizer loss"""
    sigma: float = 60
    """Sigma for REINVENT augmented loss"""
    experience_replay: int = 0
    """Experience replay"""
    experience_sampling: str = "weighted"
    """Experience sampling method which could be 'uniform' or 'weighted' with scores"""
    experience_replay_max_size: int = 100
    """Maximum size of experience replay buffer"""
    early_stopping: int = 0
    """Early stopping"""
    only_novel_samples: bool = False
    """Compute reward only on novel moelcules"""
    maximum_tanimato_similarity_threshold: float = None
    """Maximum tanimato similarity treshold for computing novel molecules"""
    prefill_experience_replay: bool = False
    """Prefill experience replay buffer with molecules from dataset"""
    # --- novelty-temperature scheduler --------------------------------
    temp_patience        : int   = 5      # steps in a row w/ no new SMILES
    temp_multiplier      : float = 1.10   # scale factor (e.g. 1.10 → +10 %)
    temp_max             : float = 2.0    # don’t let T explode

    def __post_init__(self):
        super().__post_init__()

        if self.only_novel_samples and self.maximum_tanimato_similarity_threshold is None:
            logger.warning(f"use novel samples but maximum_tanimato_similarity_threshold was not set. we will set to to 0.5 by default")
            self.maximum_tanimato_similarity_threshold = 0.5

        if self.experience_replay_max_size < self.experience_replay:
            raise ValueError(f"experience_replay_max_size:{self.experience_replay} should larger than experience_replay:{self.experience_replay}")


class REINVENTTrainer(PolicyTrainer):

    def __init__(self, **kwargs):
        """
        Initializes the REINVENT trainer.

        Sets up experience replay buffer and reward tracking for generated molecules.

        Args:
            **kwargs: Configuration and dependencies passed to the base PolicyTrainer.
        """
        super().__init__(**kwargs)
        self.generated_molecules_to_reward = dict()
        self.experience = Experience(
            tokenizer=self.tokenizer, 
            sampling=self.config.experience_sampling, 
            max_size=self.config.experience_replay_max_size, 
            seed=self.config.seed
            )
        self._no_new_counter = 0

    def generate(self, model: NovoMolGen):
        """
        Generates a batch of molecules using the current model and computes uniqueness and novelty metrics.

        Args:
            model (NovoMolGen): The generative model.

        Returns:
            dict: Dictionary containing SMILES strings and token sequences.
            dict: Dictionary of generation metrics (e.g., uniqueness, novelty).
        """
        model.eval()
        metrics = {}
        outputs = generate_valid_smiles(
            model=model,
            tokenizer=self.tokenizer,
            batch_size=self.config.per_device_train_batch_size,
            max_length=self.config.max_length,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            device=self.accelerator.device,
            return_canonical_unique=True,
        )

        generated_smiles = outputs['SMILES']
        sequences = outputs["sequences"]
        metrics["train/uniqness"] = len(generated_smiles) / self.config.per_device_train_batch_size
        if len(generated_smiles) == 0:
            return {'SMILES': [], 'sequences': sequences}, metrics
        
        if self.config.only_novel_samples:
            novel_idx = self._get_novel_molecules(generated_smiles)
            if len(novel_idx) == 0:
                metrics["train/novelty"] = 0
                return {'SMILES': [], 'sequences': sequences}, metrics

            metrics["train/novelty"] = len(novel_idx) / len(generated_smiles)
            generated_smiles = np.array(generated_smiles)[novel_idx].tolist()
            sequences = sequences[torch.tensor(novel_idx)]

            if self.model.mol_type != "SMILES":
                generated_mol = np.array(outputs[self.model.mol_type])[novel_idx].tolist()
                deocde_smiles = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)
                token_detoken_check = [x == y for x, y in zip(generated_mol, deocde_smiles)]
                assert sum(token_detoken_check) == len(generated_mol)
            else:
                deocde_smiles = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)
                token_detoken_check = [x == y for x, y in zip(generated_smiles, deocde_smiles)]
                assert sum(token_detoken_check) == len(generated_smiles)

        return {'SMILES': generated_smiles, 'sequences': sequences.clone()}, metrics

    
    def _get_novel_molecules(self, molecules: List[str]) -> List[int]:
        """Find novel molecules in a list of SMILES strings.

        Args:
            molecules: List of SMILES strings

        Returns:
            List[int]: List of indices of novel molecules
        """

        def get_fingerprint(input_smile):
            mol = get_mol(input_smile)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            return fp

        def get_similarity(inputs_fp, fps):
            try:
                sims = BulkTanimotoSimilarity(inputs_fp, fps)
            except:
                sims = [1.0]
            return float(max(sims))

        new_fps = mapper(self.n_jobs)(get_fingerprint, molecules)

        if self._dataset_fps is None:
            novel_idx = []
            for i, new_fp in enumerate(new_fps):
                new_list_of_fps = [x for x in new_fps if x != new_fp]
                max_sim = get_similarity(inputs_fp=new_fp, fps=new_list_of_fps)
                if max_sim < self.config.maximum_tanimato_similarity_threshold:
                    novel_idx.append(i)

            # only append novel one in _dataset_fps
            self._dataset_fps = [new_fp for i, new_fp in enumerate(new_fps) if i in novel_idx]
            return novel_idx
        else:
            novel_idx = []
            for i, new_fp in enumerate(new_fps):
                new_list_of_fps = [x for x in new_fps if x != new_fp]
                max_sim = get_similarity(inputs_fp=new_fp, fps=new_list_of_fps + self._dataset_fps)
                if max_sim < self.config.maximum_tanimato_similarity_threshold:
                    novel_idx.append(i)
            
            # only append novel one in _dataset_fps
            self._dataset_fps += [new_fp for i, new_fp in enumerate(new_fps) if i in novel_idx]
            return novel_idx

    def compute_reward(self, generated_smiles):
        """
        Computes task-specific reward for generated molecules, with caching to avoid recomputation.

        Args:
            generated_smiles (List[str]): List of SMILES strings.

        Returns:
            torch.Tensor: Tensor of rewards aligned with input.
            dict: Dictionary of reward metrics (e.g., top-1, top-10, average).
        """
        new_smiles = set(generated_smiles) - set(self.generated_molecules_to_reward.keys())
        if new_smiles:
            new_rewards = self.reward_fn(list(new_smiles))
            self.generated_molecules_to_reward.update(zip(new_smiles, new_rewards))

        score = torch.tensor([self.generated_molecules_to_reward[sm] for sm in generated_smiles])
        if 'docking' in self.config.task_name:
            score[score > 0] = 0
            score = score / self.config.normalize_docking

        metrics = {"train/average_reward": score.mean().item(), "train/new_smiles": len(new_smiles)}

        all_rewards = torch.tensor(list(self.generated_molecules_to_reward.values()))
        top_1_reward = torch.topk(all_rewards, 1, largest=self.config.higher_is_better).values.mean().item()
        try:
            top_10_reward = torch.topk(all_rewards, 10, largest=self.config.higher_is_better).values.mean().item()
        except:
            top_10_reward = 0
        five_percentile_reward = torch.topk(all_rewards, int(0.05 * len(all_rewards)),
                                            largest=self.config.higher_is_better).values.mean().item()
        num_oracle_calls = len(all_rewards)

        metrics.update({
            "train/top_1_reward": top_1_reward,
            "train/top_10_reward": top_10_reward,
            "train/five_percentile_reward": five_percentile_reward,
            "train/num_oracle_calls": num_oracle_calls})

        return score.to(self.accelerator.device), metrics

    def _compute_loss(self,
                      model: NovoMolGen,
                      agent_likelihood: torch.Tensor,
                      prior_likelihood: torch.Tensor,
                      reward: torch.Tensor,
                      generated_smiles: List[str],
                      ):
        """
        Computes the REINVENT loss to align the agent policy with high-reward molecular candidates.

        For each molecule x in the batch, the loss is:

            J(x) = [log P_prior(x) - log P_agent(x) + σ · s(x)]²

        where:
            - log P_prior(x): log-likelihood of x under the frozen prior,
            - log P_agent(x): log-likelihood of x under the trainable agent,
            - s(x): task-specific reward,
            - σ (sigma): scaling factor for reward influence.

        This encourages the agent to generate molecules that score well on the reward
        while staying close to the prior distribution.

        A penalty regularization term is added to avoid degenerate solutions:

            J_p = -1 / mean(log P_agent(x))

        which discourages extremely low-likelihood outputs. The final loss becomes:

            J_total = mean(J) + λ · J_p

        where λ is a hyperparameter controlling the regularization strength.

        If experience replay is enabled, additional loss terms from the replay buffer
        are appended to reinforce high-quality past samples.

        Args:
            model (NovoMolGen): The generative agent model.
            agent_likelihood (Tensor): Log-likelihoods under the agent model.
            prior_likelihood (Tensor): Log-likelihoods under the prior model.
            reward (Tensor): Scalar reward for each generated molecule.
            generated_smiles (List[str]): List of SMILES strings corresponding to generated molecules.

        Returns:
            loss (Tensor): Final scalar loss to be minimized.
            metrics (dict): Dictionary of loss diagnostics for logging.
        """
        augmented_likelihood = prior_likelihood + self.config.sigma * reward
        loss = torch.pow((augmented_likelihood - agent_likelihood), 2)

        if self.config.experience_replay and len(self.experience) > self.config.experience_replay:
            exp_seqs, exp_score, exp_prior_likelihood = self.experience.sample(self.config.experience_replay)
            exp_seqs = exp_seqs['input_ids'].to(self.accelerator.device)
            exp_score = exp_score.to(self.accelerator.device)
            exp_prior_likelihood = exp_prior_likelihood.to(self.accelerator.device)
            exp_agent_logits = model(exp_seqs).logits
            exp_agent_likelihood = self.logprobs_from_logits(exp_agent_logits, exp_seqs)
            exp_augmented_likelihood = exp_prior_likelihood + self.config.sigma * exp_score
            exp_loss = torch.pow((exp_augmented_likelihood - exp_agent_likelihood), 2)
            loss = torch.cat((loss, exp_loss), 0)
            agent_likelihood = torch.cat((agent_likelihood, exp_agent_likelihood), 0)

        new_experience = [
            (sm, rew.item(), pr.item()) for sm, rew, pr in zip(generated_smiles, reward, prior_likelihood)
        ]
        self.experience.add_experience(new_experience)

        loss = loss.mean()
        loss_p = - (1 / agent_likelihood).mean()
        loss += self.config.penalty_coef * loss_p

        metrics = {
            "train/loss": loss.item(),
            "train/augmented_likelihood": augmented_likelihood.mean().item(),
            "train/agent_likelihood": agent_likelihood.mean().item(),
            "train/prior_likelihood": prior_likelihood.mean().item(),
        }

        return loss, metrics

    def _train_step(self,
                    model: NovoMolGen,
                    ref_model: NovoMolGen,
                    optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler.LRScheduler,
                    accelerator: Accelerator) -> Dict[str, float]:
        """
        Performs a single training step including generation, reward computation, loss backpropagation, and optimization.

        Args:
            model (NovoMolGen): The trainable agent model.
            ref_model (NovoMolGen): The fixed reference model.
            optimizer (Optimizer): Optimizer for updating model parameters.
            scheduler (LRScheduler): Learning rate scheduler.
            accelerator (Accelerator): Accelerator for distributed training.

        Returns:
            dict: Dictionary of training metrics.
        """
        outputs, generation_metrics = self.generate(model=model)
        generated_smiles = outputs['SMILES']
        sequences = outputs['sequences']
        if len(generated_smiles) == 0 or sequences.shape[0] == 0:
            logger.info("No valid/unique SMILES generated this step; skipping update.")
            return generation_metrics

        reward, reward_metrics = self.compute_reward(generated_smiles)

        model.train()
        agent_logits = model(sequences).logits
        with torch.no_grad():
            prior_logits = ref_model(sequences).logits

        agent_likelihood = self.logprobs_from_logits(agent_logits, sequences)
        prior_likelihood = self.logprobs_from_logits(prior_logits, sequences)
        loss, loss_metrics = self._compute_loss(
            model=model,
            agent_likelihood=agent_likelihood,
            prior_likelihood=prior_likelihood,
            reward=reward,
            generated_smiles=generated_smiles,
        )

        metrics = {**generation_metrics, **reward_metrics, **loss_metrics}
        accelerator.backward(loss)
        if self.config.max_grad_norm is not None and self.config.max_grad_norm > 0:
            grad_norm = accelerator.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
            metrics.update({"train/grad_norm": grad_norm.item()})
            metrics.update({"train/weight_norm": (torch.tensor(
                [p.data.norm(2).item() ** 2 for p in model.parameters()]).sum() ** 0.5).item()})
        optimizer.step()
        optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()
            metrics.update({'train/learning_rate': scheduler.get_last_lr()[0]})

        metrics.update({'train/kl': self._compute_kl(policy_logit=agent_likelihood, ref_logit=prior_likelihood,
                                                     input_ids=sequences)})
        
        self._maybe_increase_temperature(
            new_smiles_this_step=int(metrics.get("train/new_smiles", 0))
        )
        return metrics

    def train(self, resume_from_checkpoint: bool = False, skip_eval: bool = False):
        """
        Full training loop for REINVENT, including experience replay preloading,
        iterative optimization, logging, and evaluation.

        Args:
            resume_from_checkpoint (bool): Whether to resume from an existing checkpoint.
        """
        total_num_optimization_steps = self.config.early_stopping
        assert self.config.early_stopping > 0, "number of early stopping steps should be higher than 0"

        model, ref_model = self.model, self.ref_model
        optimizer, scheduler = self._prepare_optimizer_and_scheduler(
            model=model,
            num_training_steps=total_num_optimization_steps
        )

        model, ref_model, optimizer, scheduler = self.accelerator.prepare(model, ref_model, optimizer, scheduler)

        logger.info("***** Running REINVENT trainer *****")
        logger.info(f"  Num Max Iterations = {total_num_optimization_steps:,}")
        logger.info(f"  Num Oracle Calls = {self.config.oracle_call_budget:,}")
        logger.info(f"  Batch Size = {self.config.per_device_train_batch_size:,}")
        logger.info(f"  Number of Trainable Parameters = {get_model_param_count(model, trainable_only=True):,}")

        if self.config.prefill_experience_replay:
            logger.info(f"Prefill experience replay buffer with {self.config.experience_replay_max_size} molecules from {self.config.dataset_name} dataset")
            if not resume_from_checkpoint:
                self._preload_experience_buffer(ref_model=ref_model)
            else:
                logger.warning("Skipping preloading experience replay buffer because we are resuming from checkpoint")

        completed_optim_steps = 0
        progress_bar = tqdm(
            total=total_num_optimization_steps,
            disable=False,
            desc=f"Training",
            dynamic_ncols=True,
        )
        progress_bar.update(completed_optim_steps)

        for iteration in range(total_num_optimization_steps):
            metrics = self._train_step(
                model=model,
                ref_model=ref_model,
                optimizer=optimizer,
                scheduler=scheduler,
                accelerator=self.accelerator)
            self._log_training_metrics(metrics, progress_bar=progress_bar)
            self.trainer_state.global_step += 1
            progress_bar.update(1)
            if len(self.generated_molecules_to_reward.keys()) > self.config.oracle_call_budget:
                break

        if not skip_eval:
            self._final_evaluation(model=model, save_path=self.config.output_dir)
            self._save_checkpoint(model=model, final_checkpoint=True)
        logger.info("\n\nTraining completed\n")
        logger.info(f"Training logs:\n{self.trainer_state.log_history[-1]}")

    def _final_evaluation(self, model: NovoMolGen, save_path: str):
        """
        Performs final evaluation by generating up to the oracle call budget, logging top molecules,
        and computing SA/IntDiv scores and top-N AUCs.

        Args:
            model (NovoMolGen): Trained model to evaluate.
            save_path (str): Directory to save evaluation results.
        """
        if len(self.generated_molecules_to_reward) < self.config.oracle_call_budget:
            logger.info(f"Number of generated molecules is less than oracle call budget. start generating and evaluating new molecules")
            model.eval() 
            pbar = tqdm(
                total=self.config.oracle_call_budget,
                initial=len(self.generated_molecules_to_reward),
                desc="Filling oracle-call buffer",
                dynamic_ncols=True,
            )
            max_empty_iters = 5
            empty_iters = 0
            max_total_iters = 50
            total_iters = 0
            while len(self.generated_molecules_to_reward) < self.config.oracle_call_budget and empty_iters < max_empty_iters:
                remaining = self.config.oracle_call_budget - len(self.generated_molecules_to_reward)
                outputs = generate_valid_smiles(
                    model=model,
                    tokenizer=self.tokenizer,
                    batch_size=min(2_000, remaining),
                    max_length=self.config.max_length,
                    temperature=self.config.temperature,
                    top_k=self.config.top_k,
                    top_p=self.config.top_p,
                    device=self.accelerator.device,
                    return_canonical_unique=True,
                )    
                new_smiles = list(set(outputs['SMILES']) - set(self.generated_molecules_to_reward.keys()))[:remaining]
                self._maybe_increase_temperature(new_smiles_this_step=len(new_smiles), minimum_new_smiles=100)
                if len(new_smiles) == 0:
                    empty_iters += 1
                    continue
                new_scores = self.reward_fn(new_smiles)
                self.generated_molecules_to_reward.update(zip(new_smiles, new_scores))
                empty_iters = 0
                total_iters += 1
                if total_iters >= max_total_iters:
                    logger.warning(f"Reached maximum total iterations ({max_total_iters}). Stopping generation.")
                    break
                pbar.update(len(new_smiles))
        
        generated_smiles = list(self.generated_molecules_to_reward.keys())
        scores = torch.tensor(list(self.generated_molecules_to_reward.values()))

        final_results = {
            'SMILES': generated_smiles,
            'Score': scores.tolist(),
        }
        self.experience.print_memory(os.path.join(save_path, "memory.txt"))
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

        final_results = {
            'SMILES': top_generated_smiles,
            'Score': top_scores.tolist(),
            'SA': sa_scores.tolist(),
            'IntDiv': intdiv_scores.tolist(),
        }
        
        df = pd.DataFrame.from_dict(final_results)
        df = (df.sort_values("Score", ascending=not self.config.higher_is_better).reset_index(drop=True))
        wandb_table = wandb.Table(dataframe=df)
        if wandb.run is not None:
            wandb.log({"eval/results": wandb_table})

        top10_indices = torch.topk(top_scores, 10, largest=self.config.higher_is_better).indices

        mol_buffer = {
            smiles: (score_tensor.item(), idx)
            for idx, (smiles, score_tensor) in enumerate(
                self.generated_molecules_to_reward.items(), start=1)
        }

        metrics = {
            'eval/IntDiv': intdiv_scores.mean().item(),
            'eval/top100/SA_avg': sa_scores.mean().item(),
            'eval/top100/reward_avg': top_scores.mean().item(),
            'eval/top10/SA_avg': sa_scores[top10_indices].mean().item(),
            'eval/top10/reward_avg': top_scores[top10_indices].mean().item(),
            'eval/top1/auc': top_auc(mol_buffer=mol_buffer, top_n=1, finish=True, freq_log=100, max_oracle_calls=self.config.oracle_call_budget),
            'eval/top10/auc': top_auc(mol_buffer=mol_buffer, top_n=10, finish=True, freq_log=1, max_oracle_calls=self.config.oracle_call_budget),
            'eval/top100/auc': top_auc(mol_buffer=mol_buffer, top_n=100, finish=True, freq_log=100, max_oracle_calls=self.config.oracle_call_budget),
        }
        self._log_training_metrics(metrics=metrics)

    @torch.no_grad()
    def _preload_experience_buffer(
        self,
        ref_model: NovoMolGen,
        dataset_name: str = "MolGen/ZINC_250K_prop",
        split: str = "train",
    ):
        """
        Pre-fills the experience replay buffer using the top-k molecules from a public dataset.

        Args:
            ref_model (NovoMolGen): Frozen reference model for computing prior log-likelihoods.
            dataset_name (str): Hugging Face dataset path.
            split (str): Dataset split (e.g., "train").
        """
        from datasets import load_dataset

        reward_column = self.config.task_name
        top_k         = self.config.experience_replay_max_size

        ds = load_dataset(
            path=dataset_name,
            split=split,
            num_proc=1,
        )

        if reward_column not in ds.column_names:
            raise ValueError(
                f"`{reward_column}` not found in {dataset_name}. "
                f"Available columns: {ds.column_names}"
            )

        # Select the k best rows
        scores      = torch.tensor(ds[reward_column])
        top_scores, top_idx = scores.topk(top_k, sorted=False)
        smiles_list = np.array(ds[self.model.mol_type])[top_idx.numpy()].tolist()

        # Tokenise once
        encodings = self.tokenizer(
            smiles_list,
            add_special_tokens=True,
            max_length=self.config.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        # Ensure the encodings are on the correct device
        decoded = self.tokenizer.batch_decode(
            encodings["input_ids"],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,   # important to avoid re-inserting spaces
        )
        if smiles_list != decoded:                       # mismatch ➜ warn & fix
            diffs = [
                f"{orig}  →  {dec}"
                for orig, dec in zip(smiles_list, decoded)
                if orig != dec
            ]
            logger.warning(
                "[Replay-Buffer preload] Tokenisation round-trip mismatch on "
                f"{len(diffs)} / {len(smiles_list)} molecules:\n"
                + "\n".join(diffs[:20])                  # show up to 20 lines
                + ("\n…" if len(diffs) > 20 else "")
            )
            smiles_list = decoded                        # use the recovered strings


        # Compute log-prob under the *prior* (frozen) model
        ref_model.eval()            # just to be safe
        prior_logp = []

        batch_ids  = encodings["input_ids"].to(self.accelerator.device)
        logits     = ref_model(batch_ids).logits
        logp       = self.logprobs_from_logits(logits, batch_ids)
        prior_logp = logp.cpu().tolist()

        # Add to replay buffer
        new_experience = list(zip(smiles_list, top_scores.tolist(), prior_logp))
        self.experience.add_experience(new_experience)

        # ---------------- extra bookkeeping & logging -----------------
        avg_reward      = float(torch.tensor(top_scores).mean())
        top1_reward     = float(torch.tensor(top_scores).max())
        top10_reward    = float(torch.topk(torch.tensor(top_scores), 
                                           k=min(10, len(top_scores))).values.mean())
        avg_prior_logp  = float(np.mean(prior_logp))

        log_dict = {
            "preload/buffer_size"     : len(self.experience),
            "preload/top1_reward"     : top1_reward,
            "preload/top10_reward"    : top10_reward,
            "preload/avg_reward"      : avg_reward,
            "preload/avg_prior_logp"  : avg_prior_logp,
        }

        logger.info(
            f"Pre-loaded {len(new_experience)} samples from '{dataset_name}' "
            f"(reward column = '{reward_column}'). "
            f"top-1 = {top1_reward:4.3f} | top-10 = {top10_reward:4.3f} | "
            f"avg = {avg_reward:4.3f} | avg_prior_logp = {avg_prior_logp:4.3f}"
        )

    def _maybe_increase_temperature(self, new_smiles_this_step: int, minimum_new_smiles: int = 0):
        """
        Increases sampling temperature if novelty has stalled for a number of steps.

        Args:
            new_smiles_this_step (int): Number of novel molecules generated in the current step.
            minimum_new_smiles (int): Minimum threshold to reset the no-novelty counter.
        """
        cfg = self.config

        # reset the streak whenever we see novelty
        if new_smiles_this_step > minimum_new_smiles:
            self._no_new_counter = 0
            return

        # otherwise keep counting
        self._no_new_counter += 1

        if self._no_new_counter < cfg.temp_patience:
            return

        # *** reached the patience => scale temperature
        old_T = cfg.temperature
        new_T = min(old_T * cfg.temp_multiplier, cfg.temp_max)

        if new_T > old_T:
            cfg.temperature = new_T
            logger.info(
                f"[TempScheduler] No-novelty streak = {cfg.temp_patience} → "
                f"increasing temperature  {old_T:.3f}  →  {new_T:.3f}"
            )
            self._log_training_metrics(metrics={'train/temperature': new_T})

        # restart the counter so we need another full streak before next bump
        self._no_new_counter = 0

    
