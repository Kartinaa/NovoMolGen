#!/usr/bin/env python

import numpy as np
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.eval import MoleculeEvaluator
from src.logging_utils import get_logger
from src.utils import get_real_cpu_cores

"""Scoring function should be a class where some tasks that are shared for every call
   can be reallocated to the __init__, and has a __call__ method which takes a single SMILES of
   argument and returns a float. A multiprocessing class will then spawn workers and divide the
   list of SMILES given between them.

   Passing *args and **kwargs through a subprocess call is slightly tricky because we need to know
   their types - everything will be a string once we have passed it. Therefor, we instead use class
   attributes which we can modify in place before any subprocess is created. Any **kwarg left over in
   the call to get_scoring_function will be checked against a list of (allowed) kwargs for the class
   and if a match is found the value of the item will be the new value for the class.

   If num_processes == 0, the scoring function will be run in the main process. Depending on how
   demanding the scoring function is and how well the OS handles the multiprocessing, this might
   be faster than multiprocessing in some cases."""


logger = get_logger(__name__)

def get_scoring_function(scoring_function, num_processes=None, **kwargs):
    """Function that initializes and returns a scoring function by name"""
    n_jobs = get_real_cpu_cores()
    logger.info(f"Use numer of jobs {n_jobs}")
    task = MoleculeEvaluator(task_names=[scoring_function], batch_size=128, n_jobs=n_jobs)

    def reward_function(smiles):
        res = task(smiles, filter=True, return_valid_index=True)
        valid_idx = res['valid_index']
        reward = np.zeros(len(smiles))
        reward[valid_idx] = np.array(res[scoring_function], dtype=np.float32)
        return reward

    return reward_function
