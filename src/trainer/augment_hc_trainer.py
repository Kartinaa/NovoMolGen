from dataclasses import dataclass
from typing import List

import rootutils
import torch
from transformers import PreTrainedTokenizerBase, DataCollatorForLanguageModeling


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.trainer.reinvent_trainer import REINVENTConfig, REINVENTTrainer
from src.models import NovoMolGen
from src.logging_utils import get_logger
from src.data_loader.ScaffoldFilter import ScaffoldSimilarity

logger = get_logger(__name__)


class Experience:
    """
    Prioritized replay buffer that remembers highest scored sequences
    (SMILES, score, prior_likelihood) and samples in proportion to score.

    - Uses torch for sampling (torch.multinomial).
    - Uses a Hugging Face PreTrainedTokenizerFast for tokenization.
    - Automatically creates a DataCollatorForLanguageModeling with mlm=False.
    """

    def __init__(
        self,
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
        self.g = torch.Generator(device="cpu").manual_seed(seed)

        # Create a data collator for LM tasks, but with mlm=False by default
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

    def add_experience(self, experience):
        """
        Adds a list of (smiles, score, prior_likelihood) tuples to memory, then:
          - Removes duplicates
          - Retains top max_size by descending score
        """
        self.memory.extend(experience)

        if len(self.memory) > self.max_size:
            # sort by descending score
            self.memory.sort(key=lambda x: x[1], reverse=True)
            seen, uniq = set(), []
            for sm, sc, pl in self.memory:
                if sm not in seen:
                    seen.add(sm)
                    uniq.append((sm, sc, pl))
                    if len(uniq) == self.max_size:
                        break
            self.memory = uniq

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
            raise IndexError(
                f"Not enough memory to sample {n} items (have {len(self.memory)})."
            )

        if self.sampling == "weighted":
            ws = torch.tensor([m[1] for m in self.memory], dtype=torch.float)
            ws = ws - ws.min() + 1e-6
            sample_indices = torch.multinomial(
                ws, n, replacement=False, generator=self.g
            )

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
            return_tensors="pt",
            padding=True,  # pad to longest in batch
            truncation=True,  # truncate if beyond model's max length
        )

        # Convert encodings dict -> list of dicts for data collator
        batch_for_collator = []
        for i in range(encodings["input_ids"].size(0)):
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

        with open(path, "w") as f:
            f.write("SMILES Score PriorLogP\n")
            for i, exp in enumerate(sorted_memory[:100]):
                # exp = (smiles, score, prior_likelihood)
                smiles_str = exp[0]
                score_val = exp[1]
                prior_val = exp[2]

                # Print up to 50 on-screen
                if i < 50:
                    print(
                        "{:4.2f}   {:6.2f}        {}".format(
                            score_val, prior_val, smiles_str
                        )
                    )

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
class AugmentedHCConfig(REINVENTConfig):
    type: str = "AugmentedHC"
    """Type of the trainer"""
    fraction_selected: float = 0.5
    """Fraction of selected samples for loss computation"""
    scaffold_filter: bool = True
    """Whether to apply scaffold filtering"""

    def __post_init__(self):
        super().__post_init__()

        if self.experience_replay_max_size < self.experience_replay:
            # Fix the error message
            raise ValueError(
                f"experience_replay_max_size:{self.experience_replay_max_size} should be larger than experience_replay:{self.experience_replay}"
            )


class AugmentedHCTrainer(REINVENTTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.generated_molecules_to_reward = dict()
        if self.config.scaffold_filter:
            self.scaffold_filter = ScaffoldSimilarity()
        self.experience = Experience(
            tokenizer=self.tokenizer,
            sampling=self.config.experience_sampling,
            max_size=self.config.experience_replay_max_size,
            seed=self.config.seed,
        )

    def compute_reward(self, generated_smiles):
        new_smiles = set(generated_smiles) - set(
            self.generated_molecules_to_reward.keys()
        )
        if new_smiles:
            new_rewards = self.reward_fn(list(new_smiles))
            self.generated_molecules_to_reward.update(zip(new_smiles, new_rewards))

        score = torch.tensor(
            [self.generated_molecules_to_reward[sm] for sm in generated_smiles]
        )
        if "docking" in self.config.task_name:
            score[score > 0] = 0
            score = score / self.config.normalize_docking

        metrics = {
            "train/average_reward": score.mean().item(),
            "train/new_smiles": len(new_smiles),
        }

        all_rewards = torch.tensor(list(self.generated_molecules_to_reward.values()))
        top_1_reward = (
            torch.topk(all_rewards, 1, largest=self.config.higher_is_better)
            .values.mean()
            .item()
        )
        k = min(10, all_rewards.numel()) 
        top_10_reward = (
            torch.topk(all_rewards, k, largest=self.config.higher_is_better)
            .values.mean()
            .item()
        ) if k > 0 else 0 

        k = max(1, int(0.05 * all_rewards.numel()))
        k = min(k, all_rewards.numel())        # guard when len < 20
        five_percentile_reward = (
            torch.topk(all_rewards, k, largest=self.config.higher_is_better)
            .values.mean()
            .item()
        ) if all_rewards.numel() else 0

        num_oracle_calls = len(all_rewards)

        metrics.update(
            {
                "train/top_1_reward": top_1_reward,
                "train/top_10_reward": top_10_reward,
                "train/five_percentile_reward": five_percentile_reward,
                "train/num_oracle_calls": num_oracle_calls,
            }
        )

        if self.config.scaffold_filter:
            score = self.scaffold_filter.score(generated_smiles, {"total_score": score})
            metrics["train/scaffold_filter"] = score.mean().item()

        return score.to(self.accelerator.device), metrics

    def _compute_loss(
        self,
        model: NovoMolGen,
        agent_likelihood: torch.Tensor,
        prior_likelihood: torch.Tensor,
        reward: torch.Tensor,
        generated_smiles: List[str],
    ):
        """
        Computes the fine-tuning loss using Augmented Hill-Climb (AHC) with top-k selection and regularization.

        From a batch of generated molecules, only the top-k (highest reward) samples are selected
        to compute the main objective:

            J(X_k) = (1/k) · Σ_{x ∈ X_k} [log P_prior(x) - log P_agent(x) + σ · s(x)]²

        where:
            - X_k: top-k molecules (defined by fraction_selected),
            - s(x): task-specific reward,
            - σ: reward scaling factor.

        A penalty term is added to discourage low-likelihood samples:

            J_p = -1 / mean(log P_agent(x))

        Final loss:

            J_total = mean(J) + λ · J_p

        Experience replay is optionally used to add more high-reward molecules from memory,
        boosting stability and sample efficiency.

        Args:
            model (NovoMolGen): The trainable agent model.
            agent_likelihood (Tensor): Log-likelihoods under the agent model.
            prior_likelihood (Tensor): Log-likelihoods under the frozen prior.
            reward (Tensor): Reward scores for generated molecules.
            generated_smiles (List[str]): Generated molecules (SMILES format).

        Returns:
            loss (Tensor): Scalar loss for optimization.
            metrics (dict): Dictionary of loss diagnostics for logging.
        """
        augmented_likelihood = prior_likelihood + self.config.sigma * reward
        loss = torch.pow((augmented_likelihood - agent_likelihood), 2)

        sorted_scores, score_idxs = reward.sort(descending=self.config.higher_is_better)
        num_selected = (
            int(len(score_idxs) * self.config.fraction_selected)
            if hasattr(self.config, "fraction_selected")
            else len(score_idxs)
        )
        num_selected = max(1, num_selected)
        selected_idxs = score_idxs[:num_selected]

        selected_loss = loss[selected_idxs]
        selected_agent_likelihood = agent_likelihood[selected_idxs]

        # Experience Replay
        if (
            self.config.experience_replay
            and len(self.experience) > self.config.experience_replay
        ):
            exp_seqs, exp_score, exp_prior_likelihood = self.experience.sample(
                self.config.experience_replay
            )
            exp_seqs = exp_seqs["input_ids"].to(self.accelerator.device)
            exp_score = exp_score.to(self.accelerator.device)
            exp_prior_likelihood = exp_prior_likelihood.to(self.accelerator.device)
            exp_agent_logits = model(exp_seqs).logits
            exp_agent_likelihood = self.logprobs_from_logits(exp_agent_logits, exp_seqs)
            exp_augmented_likelihood = (
                exp_prior_likelihood + self.config.sigma * exp_score
            )
            exp_loss = torch.pow((exp_augmented_likelihood - exp_agent_likelihood), 2)

            # Combine losses
            selected_loss = torch.cat((selected_loss, exp_loss), 0)
            selected_agent_likelihood = torch.cat(
                (selected_agent_likelihood, exp_agent_likelihood), 0
            )

        # Compute final loss
        loss = selected_loss.mean()
        loss_p = -(1 / selected_agent_likelihood + 1e-10).mean()
        loss += self.config.penalty_coef * loss_p

        new_experience = [
            (sm, rew.item(), pr.item())
            for sm, rew, pr in zip(generated_smiles, reward, prior_likelihood)
        ]
        self.experience.add_experience(new_experience)

        metrics = {
            "train/loss": loss.item(),
            "train/augmented_likelihood": augmented_likelihood.mean().item(),
            "train/agent_likelihood": agent_likelihood.mean().item(),
            "train/prior_likelihood": prior_likelihood.mean().item(),
        }

        return loss, metrics