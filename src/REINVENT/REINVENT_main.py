#!/usr/bin/env python
import argparse
import json
import wandb
import hashlib
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.REINVENT.train_agent import train_agent


parser = argparse.ArgumentParser(description="Main script for running the model")
parser.add_argument('--scoring-function', action='store', dest='scoring_function',
                    default='QED',
                    help='What type of scoring function to use.')
parser.add_argument('--scoring-function-kwargs', action='store', dest='scoring_function_kwargs',
                    nargs="*", default=dict(),
                    help='Additional arguments for the scoring function. Should be supplied with a '\
                    'list of "keyword_name argument". For pharmacophoric and tanimoto '\
                    'the keyword is "query_structure" and requires a SMILES. ' \
                    'For activity_model it is "clf_path " '\
                    'pointing to a sklearn classifier. '\
                    'For example: "--scoring-function-kwargs query_structure COc1ccccc1".')
parser.add_argument('--learning-rate', action='store', dest='learning_rate',
                    type=float, default=0.0005)
parser.add_argument('--oracle-call-budget', action='store', dest='oracle_call_budget', type=int,
                    default=10000)
parser.add_argument('--batch-size', action='store', dest='batch_size', type=int,
                    default=64)
parser.add_argument('--sigma', action='store', dest='sigma', type=int,
                    default=500)
parser.add_argument('--experience', action='store', dest='experience_replay', type=int,
                    default=24, help='Number of experience sequences to sample each step. '\
                    '0 means no experience replay.')
parser.add_argument('--num-processes', action='store', dest='num_processes',
                    type=int, default=0,
                    help='Number of processes used to run the scoring function. "0" means ' \
                    'that the scoring function will be run in the main process.')
parser.add_argument('--prior', action='store', dest='restore_prior_from',
                    default='./src/REINVENT/data/Prior_PMO.ckpt',
                    help='Path to an RNN checkpoint file to use as a Prior')
parser.add_argument('--agent', action='store', dest='restore_agent_from',
                    default='./src/REINVENT/data/Prior_PMO.ckpt',
                    help='Path to an RNN checkpoint file to use as a Agent.')
parser.add_argument('--voc', action='store', dest='vocab_from',
                    default='./src/REINVENT/data/Voc_PMO',
                    help='Path to an RNN checkpoint file to use as a Agent.')
parser.add_argument('--seed', action='store', dest='seed', type=int, default=42)
parser.add_argument('--use_wandb', action='store_true', dest='use_wandb', default=False)

if __name__ == "__main__":

    arg_dict = vars(parser.parse_args())
    use_wandb = arg_dict.pop('use_wandb', False)

    unrolled_json = json.dumps(arg_dict, sort_keys=True)
    hash_name = hashlib.md5(unrolled_json.encode()).hexdigest()[:8]
    exp_name = f"REINVENT_{arg_dict['scoring_function']}_{hash_name}"
    if use_wandb:
        wandb.init(entity="drug-discovery", project=f"small-molecule-generation-finetune", name=exp_name,
                   config=arg_dict, tags=['REINVENT'], mode='online')

    train_agent(exp_name=exp_name, **arg_dict)
