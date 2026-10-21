import warnings
import rootutils

import time
from transformers.trainer_callback import TrainerCallback
from transformers.utils import logging

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.eval import MoleculeEvaluator

logger = logging.get_logger(__name__)

class Evaluator(TrainerCallback):
    """
    A bare [`TrainerCallback`] that generates samples and evaluates them after an evaluation phase.

    Args:
        task_names: List of tasks to evaluate on
        batch_size: Batch size
        n_jobs: Number of jobs to use for evaluation
        n_samples: Number of samples to generate
        num_return_sequences: Number of sequences to return
        prompt: Prompt to use
        temperature: Temperature
        top_k: Top k
        top_p: Top p
        max_length: Maximum length

    Returns:
        A dictionary of evaluation results

    """

    def __init__(self,
                 task_names=None,
                 batch_size=512,
                 n_jobs=8,
                 n_samples: int = 3000,
                 num_return_sequences: int = 1000,
                 prompt: str = "",
                 temperature: float = 1.0,
                 top_k: int = 50,
                 top_p: float = 0.95,
                 max_length: int = 64,
                 ):
        if task_names is None:
            task_names = ["unique@1k", "IntDiv", "filters", "SA_mean", "logP_mean", "QED_mean"]
        self.n_samples = n_samples
        self.num_return_sequences = num_return_sequences
        self.prompt = prompt
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_length = max_length
        self.task_names = task_names

        self.evaluator = MoleculeEvaluator(
            task_names=task_names,
            batch_size=batch_size,
            n_jobs=n_jobs,
        )

    def on_evaluate(self, args, state, control, model=None, tokenizer=None, **kwargs):
        """
        Event called after an evaluation phase.
        """
        generated_smiles = model.generate_smiles(
            tokenizer,
            n_samples=self.n_samples,
            num_return_sequences=self.num_return_sequences,
            prompt=self.prompt,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            max_length=self.max_length,
        )

        if state.is_world_process_zero:
            st = time.time()
            try:
                result = self.evaluator(gen_smiles=generated_smiles, filter=True)
            except Exception as e:
                warnings.warn(f"evaluator metrics failed {e}", UserWarning)
                result = {k: 0.0 for k in self.task_names}

            logger.info(f"evaluation finished in: {time.time() - st} seconds")

            result.update({"valid_gen_smiles": len(generated_smiles)/self.n_samples})
            state.evaluation_task_results = {f"eval/{k}": v for k, v in result.items()}
            state.log_history.append({**result, **{"step": state.global_step}})

        return control
