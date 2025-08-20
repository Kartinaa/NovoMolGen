#!/usr/bin/env python

import json
import os
import time

import numpy as np
import rootutils
import torch
import wandb
from transformers import set_seed

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.REINVENT.model import RNN
from src.REINVENT.data_structs import Vocabulary, Experience
from src.REINVENT.scoring_functions import get_scoring_function
from src.REINVENT.utils import Variable, seq_to_smiles, unique


def fix_checkpoint(checkpoint_path):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # List of GRU bias keys to fix
    keys_to_fix = ['gru_1.bias_ih', 'gru_1.bias_hh',
                   'gru_2.bias_ih', 'gru_2.bias_hh',
                   'gru_3.bias_ih', 'gru_3.bias_hh']

    for key in keys_to_fix:
        if key in checkpoint:
            # Remove the singleton dimension
            checkpoint[key] = checkpoint[key].squeeze(0)

    # Save the fixed checkpoint
    fixed_checkpoint_path = checkpoint_path.replace('.ckpt', '_fixed.ckpt')
    torch.save(checkpoint, fixed_checkpoint_path)
    return fixed_checkpoint_path


def log_metrics(generated_molecules_to_reward, logs={}, higher_is_better: bool = True):
    rewards = list(generated_molecules_to_reward.values())
    if higher_is_better:
        top_1_reward = np.max(rewards)
        top_10_reward = np.mean(sorted(rewards, reverse=True)[:10])
        top_100_reward = np.mean(sorted(rewards, reverse=True)[:100])
    else:
        top_1_reward = np.min(rewards)
        top_10_reward = np.mean(sorted(rewards)[:10])
        top_100_reward = np.mean(sorted(rewards)[:100])
    num_oracle_calls = len(rewards)
    logs_new = {"top_1_reward": top_1_reward,
                "top_10_reward": top_10_reward,
                "top_100_reward": top_100_reward,
                "num_oracle_calls": num_oracle_calls}

    logs_new.update(logs)
    print([f"{k}: {v}" for k, v in logs_new.items() if k != "reward_histogram"])

    if wandb.run is not None:
        wandb.log(logs_new)

    return num_oracle_calls


def train_agent(restore_prior_from='./src/REINVENT/data/Prior.ckpt',
                restore_agent_from='./src/REINVENT/data/Prior.ckpt',
                vocab_from='./src/REINVENT/data/Voc',
                scoring_function='QED',
                scoring_function_kwargs=None,
                exp_name=None, learning_rate=0.0005,
                batch_size=64, num_processes=0, sigma=60,
                experience_replay=0, seed=42, oracle_call_budget=10000):
    set_seed(seed)
    voc = Vocabulary(init_from_file=vocab_from)

    Prior = RNN(voc)
    Agent = RNN(voc)

    # By default restore Agent to same model as Prior, but can restore from already trained Agent too.
    # Saved models are partially on the GPU, but if we dont have cuda enabled we can remap these
    # to the CPU.
    # Fix the checkpoint before loading it
    fixed_prior_ckpt = fix_checkpoint(restore_prior_from)
    fixed_agent_ckpt = fix_checkpoint(restore_agent_from)

    # Load the fixed checkpoints
    if torch.cuda.is_available():
        Prior.rnn.load_state_dict(torch.load(fixed_prior_ckpt))
        Agent.rnn.load_state_dict(torch.load(fixed_agent_ckpt))
    else:
        Prior.rnn.load_state_dict(torch.load(fixed_prior_ckpt, map_location=lambda storage, loc: storage))
        Agent.rnn.load_state_dict(torch.load(fixed_agent_ckpt, map_location=lambda storage, loc: storage))

    # We dont need gradients with respect to Prior
    for param in Prior.rnn.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(Agent.rnn.parameters(), lr=learning_rate)

    # Scoring_function
    if 'docking' in scoring_function:
        higher_is_better = False
    else:
        higher_is_better = True
    scoring_func = get_scoring_function(scoring_function=scoring_function, num_processes=num_processes,
                                        **scoring_function_kwargs)
    generated_molecules_to_reward = {}

    # For policy based RL, we normally train on-policy and correct for the fact that more likely actions
    # occur more often (which means the agent can get biased towards them). Using experience replay is
    # therefor not as theoretically sound as it is for value based RL, but it seems to work well.
    experience = Experience(voc)

    print("Model initialized, starting training...")

    while True:

        # Sample from Agent
        seqs, agent_likelihood, entropy = Agent.sample(batch_size)

        # Remove duplicates, ie only consider unique seqs
        unique_idxs = unique(seqs)
        seqs = seqs[unique_idxs]
        agent_likelihood = agent_likelihood[unique_idxs]
        entropy = entropy[unique_idxs]

        # Get prior likelihood and score
        prior_likelihood, _ = Prior.likelihood(Variable(seqs))
        smiles = seq_to_smiles(seqs, voc)
        # score = scoring_function(smiles)
        new_smiles = set(smiles) - set(generated_molecules_to_reward.keys())
        if new_smiles:
            new_rewards = scoring_func(list(new_smiles))
            generated_molecules_to_reward.update(zip(new_smiles, new_rewards))

        score = np.array([generated_molecules_to_reward[sm] for sm in smiles])
        if 'docking' in scoring_function:
            score[score>0] = 0
            score = score / -20
            print(f"dividing by -20 for docking score. new average score: {score.mean()}")

        # Calculate augmented likelihood
        augmented_likelihood = prior_likelihood + sigma * Variable(score)
        loss = torch.pow((augmented_likelihood - agent_likelihood), 2)

        # log metrics
        logs = {'loss': loss.mean().item(), 'agent_likelihood': agent_likelihood.mean().item(),
                'entropy': entropy.mean().item(), 'prior_likelihood': prior_likelihood.mean().item(),
                'new_smiles': len(new_smiles)}
        num_oracle_calls = log_metrics(generated_molecules_to_reward, higher_is_better=higher_is_better, logs=logs)

        # Experience Replay
        # First sample
        if experience_replay and len(experience) > 4:
            exp_seqs, exp_score, exp_prior_likelihood = experience.sample(experience_replay)
            exp_agent_likelihood, exp_entropy = Agent.likelihood(exp_seqs.long())
            exp_augmented_likelihood = exp_prior_likelihood + sigma * exp_score
            exp_loss = torch.pow((Variable(exp_augmented_likelihood) - exp_agent_likelihood), 2)
            loss = torch.cat((loss, exp_loss), 0)
            agent_likelihood = torch.cat((agent_likelihood, exp_agent_likelihood), 0)

        # Then add new experience
        prior_likelihood = prior_likelihood.data.cpu().numpy()
        new_experience = zip(smiles, score, prior_likelihood)
        experience.add_experience(new_experience)

        # Calculate loss
        loss = loss.mean()

        # Add regularizer that penalizes high likelihood for the entire sequence
        loss_p = - (1 / agent_likelihood).mean()
        loss += 5 * 1e3 * loss_p

        # Calculate gradients and make an update to the network weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if num_oracle_calls >= oracle_call_budget:
            break

    # If the entire training finishes, we create a new folder where we save this python file
    # as well as some sampled sequences and the contents of the experinence (which are the highest
    # scored sequences seen during training)
    if exp_name:
        save_dir = os.path.join('./save_finetune/REINVENT', exp_name)
    else:
        save_dir = './save_finetune/REINVENT/run_' + time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    os.makedirs(save_dir, exist_ok=True)

    experience.print_memory(os.path.join(save_dir, "memory"))
    torch.save(Agent.rnn.state_dict(), os.path.join(save_dir, 'Agent.ckpt'))

    json_string = json.dumps(generated_molecules_to_reward, indent=2, sort_keys=True) + "\n"
    with open(os.path.join(save_dir, "generated_molecules.json"), "w", encoding="utf-8") as f:
        f.write(json_string)


if __name__ == "__main__":
    train_agent()
