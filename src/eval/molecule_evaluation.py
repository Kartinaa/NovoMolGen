import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import rootutils
from loguru import logger
from rdkit.Chem import Mol
from rdkit.DataStructs import BulkTanimotoSimilarity, ExplicitBitVect
from syba.syba import SybaClassifier
from tdc import Oracle

# Setup root directory
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

try:
    from src.eval.components.reactivity import get_reactivity  # noqa
except BaseException as e:
    logger.warning(f"Failed to import reactivity: {e}")
    get_reactivity = None

from src.eval.components.pce import get_pce  # noqa

try:
    from src.eval.components.tadf import get_tadf  # noqa
except BaseException as e:
    logger.warning(f"Failed to import tadf: {e}")
    get_tadf = None
from fcd_torch import FCD as FCDMetric  # noqa

from src.eval.components.dbpp import DrugPredictor  # noqa
from src.eval.components.docking import Docking, DockingConfig  # noqa
from src.eval.components.moses import (  # noqa
    NP,
    QED,
    SA,
    TPSA,
    Bertz,
    FragMetric,
    NumAliphaticRings,
    NumAromaticRings,
    NumRotatableBonds,
    ScafMetric,
    SNNMetric,
    WassersteinMetric,
    compute_intermediate_statistics,
    fingerprints,
    fraction_passes_filters,
    fraction_unique,
    fraction_valid,
    get_n_rings,
    internal_diversity,
    logP,
    novelty,
    remove_invalid,
    weight,
)
from src.eval.components.ra_score import RAScorerXGB  # noqa
from src.eval.utils import get_mol, mapper  # noqa

GUACAMOL_TASKS = [
    "Albuterol_Similarity",
    "Amlodipine_MPO",
    "Celecoxib_Rediscovery",
    "Deco Hop",
    "DRD2",
    "Fexofenadine_MPO",
    "GSK3B",
    "Isomers_C7H8N2O2",
    "Isomers_C9H10N2O2PF2Cl",
    "JNK3",
    "Median 1",
    "Median 2",
    "Mestranol_Similarity",
    "Osimertinib_MPO",
    "Perindopril_MPO",
    "Ranolazine_MPO",
    "Scaffold Hop",
    "Sitagliptin_MPO",
    "Thiothixene_Rediscovery",
    "Troglitazone_Rediscovery",
    "Valsartan_SMARTS",
    "Zaleplon_MPO",
]

PROPERTY_DISTRIBUTION_TASKS = {
    "logP": logP,
    "SA": SA,
    "QED": QED,
    "weight": weight,
    "NP": NP,
    "NumRings": get_n_rings,
    "Bertz": Bertz,
    "PenalizedLogP": Oracle(name="LogP"),
    "TPSA": TPSA,
    "AliphaticRings": NumAliphaticRings,
    "AromaticRings": NumAromaticRings,
    "RotatableBonds": NumRotatableBonds,
}

FRAGMENT_TASKS = {
    "FCD": FCDMetric,
    "SNN": SNNMetric,
    "Frag": FragMetric,
    "Scaf": ScafMetric,
}

TARTARUS_TASKS = {"pce": get_pce, "tadf": get_tadf, "reactivity": get_reactivity}


class MoleculeEvaluator:
    """Evaluation suite for evaluating molecule generation models."""

    def __init__(
        self,
        task_names: str | List[str] = "novelty",
        train_smiles: Optional[List[str]] = None,
        valid_stats: Optional[dict] = None,
        test_stats: Optional[dict] = None,
        batch_size: int = 512,
        n_jobs: int = 30,
        device: str = "cuda",
        **kwargs,
    ):
        """Initialize the evaluator.

        Args:
            task_names (str | List[str], optional): Task names to evaluate, defaults to "novelty"
            train_smiles (Optional[List[str]], optional): List of training SMILES, defaults to None
            valid_stats (Optional[dict], optional): List of validation statistics, defaults to None
            test_stats (Optional[dict], optional): List of test statistics, defaults to None
            batch_size (int, optional): Batch size, defaults to 512
            n_jobs (int, optional): Number of parallel processing jobs, defaults to 30
            device (str, optional): Pytorch device, defaults to "cuda"
            **kwargs: Additional keyword arguments
        """
        self.task_names = task_names
        self.train_smiles = train_smiles
        self.valid_stats = valid_stats
        self.test_stats = test_stats
        self.kwargs = kwargs
        self.batch_size = batch_size
        self.device = device
        self.n_jobs = n_jobs

    def __call__(
        self,
        gen_smiles: List[str],
        gen_mols: Optional[List[Mol]] = None,
        filter: bool = True,
        return_valid_index: bool = False,
    ) -> dict:
        """Evaluate the generated molecules.

        Args:
            gen_smiles (List[str]): List of generated SMILES
            gen_mols (List[Mol], optional): List of generated RDKit molecules, defaults to None
            filter (bool, optional): Whether to filter valid molecules, defaults to True
            return_valid_index (bool, optional): Whether to return the valid index, defaults to False

        Returns:
            dict: Evaluation metric for the generated molecules
        """

        score_dict = {}

        # Filter valid molecules
        if filter:
            validity = fraction_valid(gen_smiles, self.n_jobs)
            logger.info(f"Fraction of valid molecules: {validity:.2f}")
            score_dict["validity"] = validity
            gen_smiles, valid_index = remove_invalid(gen=gen_smiles, n_jobs=self.n_jobs)
            if return_valid_index:
                score_dict["valid_index"] = valid_index
        else:
            score_dict["validity"] = 1.0

        # Get RDKit molecules
        if gen_mols is None:
            gen_mols = mapper(self.n_jobs)(get_mol, gen_smiles)

        if isinstance(self.task_names, str):
            self.task_names = [self.task_names]

        for task_name in self.task_names:
            logger.info(f"Evaluating task: {task_name}")
            # Assign evaluator function for distribution oracles
            if task_name == "unique@1k":
                score = fraction_unique(gen_smiles, k=1000, n_jobs=self.n_jobs)
            elif task_name == "unique@10k":
                score = fraction_unique(gen_smiles, k=10000, n_jobs=self.n_jobs)
            elif task_name == "IntDiv":
                score = internal_diversity(gen_smiles, device=self.device, n_jobs=self.n_jobs)
            elif task_name == "IntDiv2":
                score = internal_diversity(gen_smiles, device=self.device, n_jobs=self.n_jobs, p=2)
            elif task_name == "filters":
                score = fraction_passes_filters(gen_smiles, n_jobs=self.n_jobs)
            elif task_name == "novelty":
                assert (
                    self.train_smiles is not None
                ), "Training SMILES must be provided for novelty"
                score = novelty(gen_smiles, train=self.train_smiles, n_jobs=self.n_jobs)
            elif task_name == "kl_divergence":
                assert (
                    self.train_smiles is not None
                ), "Training SMILES must be provided for KL Divergence"
                oracle = Oracle(name="KL_Divergence")
                score = oracle(gen_smiles, training_smiles_lst=self.train_smiles)

            # Assign evaluator function for docking
            elif "docking" in task_name:
                target_name = task_name.split("_")[1]
                docking_cfg = DockingConfig(
                    target_name=target_name, num_sub_proc=self.n_jobs, **self.kwargs
                )
                evaluator_func = Docking(docking_cfg)
                score = evaluator_func(gen_smiles)

            # Assign evaluator function for fragment metrics
            elif task_name in FRAGMENT_TASKS.keys():
                evaluator_func = FRAGMENT_TASKS[task_name](
                    batch_size=self.batch_size,
                    device=self.device,
                    n_jobs=self.n_jobs,
                    **self.kwargs,
                )
                assert (
                    self.valid_stats is not None
                ), "Validation statistics must be provided for fragment metrics"
                assert (
                    self.test_stats is not None
                ), "Test statistics must be provided for fragment metrics"
                score = {}
                if task_name == "FCD":
                    score["FCD"] = evaluator_func(gen=gen_smiles, pref=self.valid_stats[task_name])  # type: ignore
                    score["FCD/test"] = evaluator_func(gen=gen_smiles, pref=self.test_stats[task_name])  # type: ignore
                else:
                    score[task_name] = evaluator_func(gen=gen_mols, pref=self.valid_stats[task_name])  # type: ignore
                    score[f"{task_name}/test"] = evaluator_func(gen=gen_mols, pref=self.test_stats[task_name])  # type: ignore

            # Assign evaluator function for property distribution metrics
            elif task_name.split("_")[0] in PROPERTY_DISTRIBUTION_TASKS.keys():
                evaluator_func = PROPERTY_DISTRIBUTION_TASKS[task_name.split("_")[0]]
                if "wasserstein" in task_name:
                    assert (
                        self.valid_stats is not None
                    ), "Validation statistics must be provided for Wasserstein metrics"
                    assert (
                        self.test_stats is not None
                    ), "Test statistics must be provided for Wasserstein metrics"
                    score = {}
                    wassertein_evaluator_func = WassersteinMetric(
                        evaluator_func,
                        batch_size=self.batch_size,
                        device=self.device,
                        n_jobs=self.n_jobs,
                    )
                    base_task_name = task_name.split("_")[0]
                    score[task_name] = wassertein_evaluator_func(gen=gen_mols, pref=self.valid_stats[base_task_name])  # type: ignore
                    score[f"{task_name}/test"] = wassertein_evaluator_func(gen=gen_mols, pref=self.test_stats[base_task_name])  # type: ignore
                elif "mean" in task_name:
                    score_list = mapper(self.n_jobs)(evaluator_func, gen_mols)
                    score = sum(score_list) / len(score_list)
                else:
                    score = mapper(self.n_jobs)(evaluator_func, gen_mols)

            # Assign evaluator function for custom qed and sa scores
            elif task_name == "DBPP":
                score = [
                    DrugPredictor().predict(smi, mol) for smi, mol in zip(gen_smiles, gen_mols)
                ]
            elif task_name == "RAScore":
                score = mapper(self.n_jobs)(RAScorerXGB().predict, gen_mols)
            elif task_name == "SYBA":
                syba = SybaClassifier()
                syba.fitDefaultScore()
                score = [syba.predict(smi, mol) for smi, mol in zip(gen_smiles, gen_mols)]

            # Assign evaluator function for guacamol oracles
            elif task_name in GUACAMOL_TASKS:
                score = mapper(self.n_jobs)(Oracle(name=task_name), gen_smiles)

            # Assign evaluator function for tartarus tasks
            elif task_name in TARTARUS_TASKS.keys():
                if task_name == "pce":
                    score = {"PCE_PCBM - SAS": [], "PCE_PCDTBT - SAS": []}
                    for smi in gen_smiles:
                        pce_pcbm_sas, pce_pcdtbt_sas = TARTARUS_TASKS[task_name](smi)
                        score["PCE_PCBM - SAS"].append(pce_pcbm_sas)
                        score["PCE_PCDTBT - SAS"].append(pce_pcdtbt_sas)
                elif task_name == "tadf":
                    score = {"Singlet-triplet": [], "Oscillator strength": [], "Combined": []}
                    for smi in gen_smiles:
                        st, osc, combined = TARTARUS_TASKS[task_name](smi, verbose=True)
                        score["Singlet-triplet"].append(st)
                        score["Oscillator strength"].append(osc)
                        score["Combined"].append(combined)
                elif task_name == "reactivity":
                    score = {"Ea": [], "Er": [], "sum_Ea_Er": [], "diff_Ea_Er": []}
                    for smi in gen_smiles:
                        Ea, Er, sum_Ea_Er, diff_Ea_Er = TARTARUS_TASKS[task_name](
                            smi, verbose=True, n_procs=self.n_jobs
                        )
                        score["Ea"].append(Ea)
                        score["Er"].append(Er)
                        score["sum_Ea_Er"].append(sum_Ea_Er)
                        score["diff_Ea_Er"].append(diff_Ea_Er)
            else:
                raise ValueError(f"Unknown task name: {self.task_names}")
            score_dict[task_name] = score
        return score_dict


def hit_ratio(scores: np.ndarray, task_name: str) -> float:
    """Calculate the hit ratio.

    Args:
        scores: List of scores
        task_name: Name of the task

    Raises:
        ValueError: If the target protein is not supported

    Returns:
        float: Hit ratio
    """
    if task_name == "docking_parp1":
        hit_thr = -10.0
    elif task_name == "docking_fa7":
        hit_thr = -8.5
    elif task_name == "docking_5ht1b":
        hit_thr = -8.7845
    elif task_name == "docking_jak2":
        hit_thr = -9.1
    elif task_name == "docking_braf":
        hit_thr = -10.3
    else:
        raise ValueError("Wrong target protein")
    hits = [1 if score < hit_thr else 0 for score in scores]
    return (sum(hits) / len(hits)) * 100


def top_auc(
    mol_buffer: Dict[str, Tuple[float, int]],
    top_n: int,
    finish: bool,
    freq_log: int,
    max_oracle_calls: int,
) -> float:
    """Calculate the top AUC score.

    Args:
        mol_buffer: Buffer containing the molecules, their scores and indices
        top_n: Number of top molecules to consider
        finish: Whether evaluation is finished
        freq_log: Frequency of logging
        max_oracle_calls: Maximum number of oracle calls

    Returns:
        float: Top AUC score
    """
    sum = 0
    prev = 0
    called = 0
    ordered_results = list(sorted(mol_buffer.items(), key=lambda kv: kv[1][1], reverse=False))
    for idx in range(freq_log, min(len(mol_buffer), max_oracle_calls), freq_log):
        temp_result = ordered_results[:idx]
        temp_result = list(sorted(temp_result, key=lambda kv: kv[1][0], reverse=True))[:top_n]
        top_n_now = np.mean([item[1][0] for item in temp_result])
        sum += freq_log * (top_n_now + prev) / 2
        prev = top_n_now
        called = idx
    temp_result = list(sorted(ordered_results, key=lambda kv: kv[1][0], reverse=True))[:top_n]
    top_n_now = np.mean([item[1][0] for item in temp_result])
    sum += (len(mol_buffer) - called) * (top_n_now + prev) / 2
    if finish and len(mol_buffer) < max_oracle_calls:
        sum += (max_oracle_calls - len(mol_buffer)) * top_n_now
    return sum / max_oracle_calls


if __name__ == "__main__":
    st = time.time()
    gen_smiles = ["CCO"] * 1000
    train_smiles = ["CCO"] * 1000
    valid_smiles = ["CCO"] * 1000
    test_smiles = ["CCO"] * 1000
    valid_stats = compute_intermediate_statistics(valid_smiles, 10, device="cuda")
    test_stats = compute_intermediate_statistics(test_smiles, 10, device="cuda")
    evaluator = MoleculeEvaluator(
        task_names=[
            "reactivity",
            "SA",
            "logP_wasserstein",
            "logP_mean",
            "FCD",
            "docking_fa7",
            "unique@1k",
            "GSK3B",
        ],
        train_smiles=train_smiles,
        valid_stats=valid_stats,
        test_stats=test_stats,
    )
    print(evaluator(gen_smiles))
    print(f"finished in {time.time() - st} seconds")
