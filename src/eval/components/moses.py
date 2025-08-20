"""Code for model evaluation.

Adapted from https://github.com/molecularsets/moses/
"""

import os
import warnings
from collections import Counter
from functools import partial
from multiprocessing import Pool
import json
import sys
import signal
from typing import List

import numpy as np
import pandas as pd
import rootutils
import scipy.sparse
import torch
from fcd_torch import FCD as FCDMetric
from tqdm import tqdm
from loguru import logger
from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP  # type: ignore
from rdkit.Chem.Descriptors import MolWt  # type: ignore
from rdkit.Chem.GraphDescriptors import BertzCT
from rdkit.Chem.MACCSkeys import GenMACCSKeys  # type: ignore
from rdkit.Chem.QED import qed
from rdkit.Chem.rdMolDescriptors import (
    CalcNumAliphaticRings,
    CalcNumAromaticRings,
    CalcNumRotatableBonds,
    CalcTPSA,
)
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect as Morgan
from rdkit.Chem.rdmolops import FragmentOnBRICSBonds
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol  # type: ignore
from scipy.stats import wasserstein_distance
from torchdata.stateful_dataloader import StatefulDataLoader
from datasets.config import HF_CACHE_HOME
from datasets import load_dataset

# Set up the project root
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.eval.components import npscorer, sascorer  # noqa
from src.eval.utils import get_mol, mapper  # noqa

# Read drug filters
_mcf = pd.read_csv(os.path.join("./data/filters", "mcf.csv"))
_pains = pd.read_csv(os.path.join("./data/filters", "wehi_pains.csv"), names=["smarts", "names"])
_filters = [
    Chem.MolFromSmarts(x)
    for x in pd.concat([_mcf, _pains], ignore_index=True, sort=True)["smarts"].values
]


def canonic_smiles(smiles_or_mol):
    mol = get_mol(smiles_or_mol)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def TPSA(mol):
    """Computes RDKit's TPSA score."""
    return CalcTPSA(mol)


def NumRotatableBonds(mol):
    """Computes RDKit's NumRotatableBonds score."""
    return CalcNumRotatableBonds(mol)


def NumAliphaticRings(mol):
    """Computes RDKit's NumAliphaticRings score."""
    return CalcNumAliphaticRings(mol)


def NumAromaticRings(mol):
    """Computes RDKit's NumAromaticRings score."""
    return CalcNumAromaticRings(mol)


def Bertz(mol):
    """Computes RDKit's BertzCT score."""
    return BertzCT(mol)


def logP(mol):
    """Computes RDKit's logP."""
    return MolLogP(mol)


def SA(mol):
    """Computes RDKit's Synthetic Accessibility score."""
    return sascorer.calculateScore(mol)


def NP(mol):
    """Computes RDKit's Natural Product-likeness score."""
    return npscorer.scoreMol(mol)


def QED(mol):
    """Computes RDKit's QED score."""
    return qed(mol)


def weight(mol):
    """Computes molecular weight for given molecule.

    Returns float,
    """
    return MolWt(mol)


def get_n_rings(mol):
    """Computes the number of rings in a molecule."""
    return mol.GetRingInfo().NumRings()


def fragmenter(mol):
    """Fragment mol using BRICS and return smiles list."""
    fgs = FragmentOnBRICSBonds(get_mol(mol))
    fgs_smi = Chem.MolToSmiles(fgs).split(".")
    return fgs_smi


def compute_fragments(mol_list, n_jobs=1):
    """Fragment list of mols using BRICS and return smiles list."""
    fragments = Counter()
    for mol_frag in mapper(n_jobs)(fragmenter, mol_list):
        fragments.update(mol_frag)
    return fragments


def compute_scaffolds(mol_list, n_jobs=1, min_rings=2):
    """Extracts a scaffold from a molecule in a form of a canonic SMILES."""
    scaffolds = Counter()
    map_ = mapper(n_jobs)
    scaffolds = Counter(map_(partial(compute_scaffold, min_rings=min_rings), mol_list))
    if None in scaffolds:
        scaffolds.pop(None)
    return scaffolds


def compute_scaffold(mol, min_rings=2):
    mol = get_mol(mol)
    try:
        scaffold = GetScaffoldForMol(mol)
    except (ValueError, RuntimeError):
        return None
    n_rings = get_n_rings(scaffold)
    scaffold_smiles = Chem.MolToSmiles(scaffold)
    if scaffold_smiles == "" or n_rings < min_rings:
        return None
    return scaffold_smiles


def average_agg_tanimoto(stock_vecs, gen_vecs, batch_size=5000, agg="max", device="cpu", p=1):
    """For each molecule in gen_vecs finds closest molecule in stock_vecs. Returns average tanimoto
    score for between these molecules.

    Parameters:
        stock_vecs: numpy array <n_vectors x dim>
        gen_vecs: numpy array <n_vectors' x dim>
        agg: max or mean
        p: power for averaging: (mean x^p)^(1/p)
    """
    assert agg in ["max", "mean"], "Can aggregate only max or mean"
    agg_tanimoto = np.zeros(len(gen_vecs))
    total = np.zeros(len(gen_vecs))
    for j in range(0, stock_vecs.shape[0], batch_size):
        x_stock = torch.tensor(stock_vecs[j : j + batch_size]).to(device).float()
        for i in range(0, gen_vecs.shape[0], batch_size):
            y_gen = torch.tensor(gen_vecs[i : i + batch_size]).to(device).float()
            y_gen = y_gen.transpose(0, 1)
            tp = torch.mm(x_stock, y_gen)
            jac = (
                (tp / (x_stock.sum(1, keepdim=True) + y_gen.sum(0, keepdim=True) - tp))
                .cpu()
                .numpy()
            )
            jac[np.isnan(jac)] = 1
            if p != 1:
                jac = jac**p
            if agg == "max":
                agg_tanimoto[i : i + y_gen.shape[1]] = np.maximum(
                    agg_tanimoto[i : i + y_gen.shape[1]], jac.max(0)
                )
            elif agg == "mean":
                agg_tanimoto[i : i + y_gen.shape[1]] += jac.sum(0)
                total[i : i + y_gen.shape[1]] += jac.shape[0]
    if agg == "mean":
        agg_tanimoto /= total
    if p != 1:
        agg_tanimoto = (agg_tanimoto) ** (1 / p)
    return np.mean(agg_tanimoto)


def fingerprint(
    smiles_or_mol,
    fp_type="maccs",
    dtype=None,
    morgan__r=2,
    morgan__n=1024,
    *args,
    **kwargs,
):
    """Generates fingerprint for SMILES If smiles is invalid, returns None Returns numpy array of
    fingerprint bits.

    Parameters:
        smiles: SMILES string
        type: type of fingerprint: [MACCS|morgan]
        dtype: if not None, specifies the dtype of returned array
    """
    fp_type = fp_type.lower()
    molecule = get_mol(smiles_or_mol, *args, **kwargs)
    if molecule is None:
        return None
    if fp_type == "maccs":
        keys = GenMACCSKeys(molecule)
        keys = np.array(keys.GetOnBits())
        fingerprint = np.zeros(166, dtype="uint8")
        if len(keys) != 0:
            fingerprint[keys - 1] = 1  # We drop 0-th key that is always zero
    elif fp_type == "morgan":
        fingerprint = np.asarray(Morgan(molecule, morgan__r, nBits=morgan__n), dtype="uint8")
    else:
        raise ValueError(f"Unknown fingerprint type {fp_type}")
    if dtype is not None:
        fingerprint = fingerprint.astype(dtype)
    return fingerprint


def fingerprints(smiles_mols_array, n_jobs=1, already_unique=False, *args, **kwargs):
    """Computes fingerprints of smiles np.array/list/pd.Series with n_jobs workers
    e.g.fingerprints(smiles_mols_array, type='morgan', n_jobs=10) Inserts np.NaN to rows
    corresponding to incorrect smiles.

    IMPORTANT: if there is at least one np.NaN, the dtype would be float
    Parameters:
        smiles_mols_array: list/array/pd.Series of smiles or already computed
            RDKit molecules
        n_jobs: number of parallel workers to execute
        already_unique: flag for performance reasons, if smiles array is big
            and already unique. Its value is set to True if smiles_mols_array
            contain RDKit molecules already.
    """
    if isinstance(smiles_mols_array, pd.Series):
        smiles_mols_array = smiles_mols_array.values
    else:
        smiles_mols_array = np.asarray(smiles_mols_array)
    if not isinstance(smiles_mols_array[0], str):
        already_unique = True

    if not already_unique:
        smiles_mols_array, inv_index = np.unique(smiles_mols_array, return_inverse=True)  # type: ignore

    fps = mapper(n_jobs)(partial(fingerprint, *args, **kwargs), smiles_mols_array)

    length = 1
    for fp in fps:
        if fp is not None:
            length = fp.shape[-1]
            first_fp = fp
            break
    fps = [fp if fp is not None else np.array([np.NaN]).repeat(length)[None, :] for fp in fps]
    if scipy.sparse.issparse(first_fp):
        fps = scipy.sparse.vstack(fps).tocsr()
    else:
        fps = np.vstack(fps)
    if not already_unique:
        return fps[inv_index]
    return fps


def mol_passes_filters(mol, allowed=None, isomericSmiles=False):
    """Checks if mol.

    * passes MCF and PAINS filters,
    * has only allowed atoms
    * is not charged
    """
    allowed = allowed or {"C", "N", "S", "O", "F", "Cl", "Br", "H"}
    mol = get_mol(mol)
    if mol is None:
        return False
    ring_info = mol.GetRingInfo()
    if ring_info.NumRings() != 0 and any(len(x) >= 8 for x in ring_info.AtomRings()):
        return False
    h_mol = Chem.AddHs(mol)
    if any(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms()):  # type: ignore
        return False
    if any(atom.GetSymbol() not in allowed for atom in mol.GetAtoms()):  # type: ignore
        return False
    if any(h_mol.HasSubstructMatch(smarts) for smarts in _filters):
        return False
    smiles = Chem.MolToSmiles(mol, isomericSmiles=isomericSmiles)
    if smiles is None or len(smiles) == 0:
        return False
    if Chem.MolFromSmiles(smiles) is None:
        return False
    return True


def compute_intermediate_statistics(smiles, n_jobs=1, device="cpu", batch_size=512, pool=None):
    """The function precomputes statistics such as mean and variance for FCD, etc.

    It is useful to compute the statistics for test and scaffold test sets to speedup metrics
    calculation.
    """
    function_mappings = {
        "logP": logP,
        "SA": SA,
        "QED": QED,
        "weight": weight,
        "NP": NP,
        "NumRings": get_n_rings,
        "Bertz": Bertz,
        "TPSA": TPSA,
        "AliphaticRings": NumAliphaticRings,
        "AromaticRings": NumAromaticRings,
        "RotatableBonds": NumRotatableBonds,
    }
    close_pool = False
    if pool is None:
        if n_jobs != 1:
            pool = Pool(n_jobs)
            close_pool = True
        else:
            pool = 1
    statistics = {}
    mols = mapper(pool)(get_mol, smiles)
    kwargs = {"n_jobs": pool, "device": device, "batch_size": batch_size}
    kwargs_fcd = {"n_jobs": n_jobs, "device": device, "batch_size": batch_size}
    statistics["FCD"] = FCDMetric(**kwargs_fcd).precalc(smiles)
    statistics["SNN"] = SNNMetric(**kwargs).precalc(mols)
    statistics["Frag"] = FragMetric(**kwargs).precalc(mols)
    statistics["Scaf"] = ScafMetric(**kwargs).precalc(mols)
    # Loop through the function mappings and calculate statistics
    for name, func in function_mappings.items():
        statistics[name] = WassersteinMetric(func, **kwargs).precalc(mols)
        logger.info(f"Computed {name} statistics")
    if close_pool:
        pool.terminate()  # type: ignore
    return statistics


def fraction_passes_filters(gen, n_jobs=1):
    """
    Computes the fraction of molecules that pass filters:
    * MCF
    * PAINS
    * Only allowed atoms ('C','N','S','O','F','Cl','Br','H')
    * No charges
    """
    passes = mapper(n_jobs)(mol_passes_filters, gen)
    return np.mean(passes)


def internal_diversity(gen, n_jobs=1, device="cpu", fp_type="morgan", gen_fps=None, p=1):
    """
    Computes internal diversity as:
    1/|A|^2 sum_{x, y in AxA} (1-tanimoto(x, y))
    """
    if gen_fps is None:
        gen_fps = fingerprints(gen, fp_type=fp_type, n_jobs=n_jobs)
    return 1 - (average_agg_tanimoto(gen_fps, gen_fps, agg="mean", device=device, p=p)).mean()


def fraction_unique(gen, k=None, n_jobs=1, check_validity=True):
    """
    Computes a number of unique molecules
    Parameters:
        gen: list of SMILES
        k: compute unique@k
        n_jobs: number of threads for calculation
        check_validity: raises ValueError if invalid molecules are present
    """
    if k is not None:
        if len(gen) < k:
            warnings.warn(f"Can't compute unique@{k}." + f"gen contains only {len(gen)} molecules")
        gen = gen[:k]
    canonic = set(mapper(n_jobs)(canonic_smiles, gen))
    if None in canonic and check_validity:
        raise ValueError("Invalid molecule passed to unique@k")
    return len(canonic) / len(gen)


def fraction_valid(gen, n_jobs=1):
    """
    Computes a number of valid molecules
    Parameters:
        gen: list of SMILES
        n_jobs: number of threads for calculation
    """
    gen = mapper(n_jobs)(get_mol, gen)
    return 1 - gen.count(None) / len(gen)


def novelty(
        gen_smiles: List[str],
        batch_size: int = 100_000,
        n_jobs: int = 8,
        train_dataset_name: str = "MolGen/ZINC_1B-raw",
        mol_type: str = "SMILES",
        output_dir: str = "tmp-novelty",
        load_from_saved: bool = True,
):
    len_gen_smiles = len(gen_smiles)
    gen_smiles = mapper(8)(canonic_smiles, gen_smiles)
    gen_smiles_set = set(gen_smiles) - {None}

    if load_from_saved:
        path_to_hf_dataset = os.path.join(HF_CACHE_HOME, "datasets", "MolGen___zinc_1_b-raw/default/0.0.0/5fb538cc0d89b1aec508ff84de45dedd1e1c5391")
        data_files = []
        for i in range(1212):
            data_files.append(os.path.join(path_to_hf_dataset, f'zinc_1_b-raw-train-{str(i).zfill(5)}-of-01212.arrow'))
        raw_data = load_dataset("arrow", data_files=data_files, streaming=True, split="train")

        dataset_info_path = os.path.join(path_to_hf_dataset, 'dataset_info.json')
        with open(dataset_info_path) as f:
            d = json.load(f)
        dataset_lenght = d['splits']['train']['num_examples']
    else:
        raw_data = load_dataset(train_dataset_name, split='train', streaming=True)
        dataset_lenght = raw_data._info.splits['train'].num_examples

    smiles_data = raw_data.map(lambda x: {mol_type: x[mol_type]},
                               remove_columns=['Deep SMILES', 'SELFIES', 'SAFE'])

    dataloader = StatefulDataLoader(smiles_data, batch_size=batch_size, num_workers=n_jobs)

    similar_smiles_count = 0

    def handler(signum, frame):
        print(f"Signal {signum} received, checkpointing...")
        dataloader_state = dataloader.state_dict()
        checkpoint_state = {
            'similar_smiles_count': similar_smiles_count,
            'dataloader_state': dataloader_state
        }
        torch.save(checkpoint_state, os.path.join(output_dir, "smiles_checkpoint_for_novelty.tar"))
        print(f"State saved with {similar_smiles_count} similar SMILES.")
        sys.exit(0)

    signal.signal(signal.SIGTERM, handler)

    if os.path.exists(os.path.join(output_dir, "smiles_checkpoint_for_novelty.tar")):
        checkpoint_state = torch.load(os.path.join(output_dir, "smiles_checkpoint_for_novelty.tar"))
        similar_smiles_count = checkpoint_state['similar_smiles_count']
        dataloader.load_state_dict(checkpoint_state['dataloader_state'])
        dataloader._update_state_dict()
        print(f"Loaded state with {similar_smiles_count} similar SMILES.")

    print(f"Starting with {similar_smiles_count} similar SMILES.")
    with tqdm(total=dataset_lenght // batch_size, desc=" ") as pbar:
        for idx, batch in enumerate(dataloader):
            train_set = set(batch[mol_type])
            similar_smiles_count += len(gen_smiles_set & train_set)
            desc = f" similar found: {similar_smiles_count}"
            pbar.set_description(desc)
            pbar.update(1)

    return {'novelty_total': 1 - (similar_smiles_count / len_gen_smiles)}


def remove_invalid(gen, canonize=True, n_jobs=1):
    """Removes invalid molecules from the dataset."""
    # TODO: check if it's compatible with rest of the code
    if not canonize:
        mols = mapper(n_jobs)(get_mol, gen)
        valid_index = [True if x else False for x in mols]
        valid_smiles = [x for x in mols if x is not None]
        return valid_smiles, valid_index

    canonic_gen_smiles = mapper(n_jobs)(canonic_smiles, gen)
    valid_index = [True if x else False for x in canonic_gen_smiles]
    valid_smiles = [x for x in canonic_gen_smiles if x is not None]
    return valid_smiles, valid_index


class Metric:
    def __init__(self, n_jobs=1, device="cpu", batch_size=512, **kwargs):
        self.n_jobs = n_jobs
        self.device = device
        self.batch_size = batch_size
        for k, v in kwargs.values():
            setattr(self, k, v)

    def __call__(self, ref=None, gen=None, pref=None, pgen=None):
        assert (ref is None) != (pref is None), "specify ref xor pref"
        assert (gen is None) != (pgen is None), "specify gen xor pgen"
        if pref is None:
            pref = self.precalc(ref)
        if pgen is None:
            pgen = self.precalc(gen)
        return self.metric(pref, pgen)

    def precalc(self, molecules):
        raise NotImplementedError

    def metric(self, pref, pgen):
        raise NotImplementedError


class SNNMetric(Metric):
    """Computes average max similarities of gen SMILES to ref SMILES."""

    def __init__(self, fp_type="morgan", **kwargs):
        self.fp_type = fp_type
        super().__init__(**kwargs)

    def precalc(self, mols):
        return {"fps": fingerprints(mols, n_jobs=self.n_jobs, fp_type=self.fp_type)}

    def metric(self, pref, pgen):
        return average_agg_tanimoto(pref["fps"], pgen["fps"], device=self.device)


def cos_similarity(ref_counts, gen_counts):
    """
    Computes cosine similarity between
     dictionaries of form {name: count}. Non-present
     elements are considered zero:

     sim = <r, g> / ||r|| / ||g||
    """
    if len(ref_counts) == 0 or len(gen_counts) == 0:
        return np.nan
    keys = np.unique(list(ref_counts.keys()) + list(gen_counts.keys()))
    ref_vec = np.array([ref_counts.get(k, 0) for k in keys])
    gen_vec = np.array([gen_counts.get(k, 0) for k in keys])
    cos_distance_calc = (
        np.dot(ref_vec, gen_vec) / np.linalg.norm(ref_vec) / np.linalg.norm(gen_vec)
    )
    return cos_distance_calc


class FragMetric(Metric):
    def precalc(self, mols):
        return {"frag": compute_fragments(mols, n_jobs=self.n_jobs)}

    def metric(self, pref, pgen):
        return cos_similarity(pref["frag"], pgen["frag"])


class ScafMetric(Metric):
    def precalc(self, mols):
        return {"scaf": compute_scaffolds(mols, n_jobs=self.n_jobs)}

    def metric(self, pref, pgen):
        return cos_similarity(pref["scaf"], pgen["scaf"])


class WassersteinMetric(Metric):
    def __init__(self, func=None, **kwargs):
        self.func = func
        super().__init__(**kwargs)

    def precalc(self, mols):
        if self.func is not None:
            values = mapper(self.n_jobs)(self.func, mols)
        else:
            values = mols
        return {"values": values}

    def metric(self, pref, pgen):
        return wasserstein_distance(pref["values"], pgen["values"])
