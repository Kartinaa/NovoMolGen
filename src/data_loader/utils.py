from typing import Any, Tuple
from pathlib import Path

import safe
import selfies as sf
from deepsmiles import Converter

import pandas as pd
from datasets.config import HF_CACHE_HOME
from datasets.naming import camelcase_to_snakecase
from huggingface_hub import hf_hub_download


def safe_to_smiles(safe_str: str) -> str:
    """Converts a SMILES string to a SAFE string.

    :param safe_str: SAFE string to convert
    :return: SMILES string
    """
    try:
        return safe.decode(safe_str)
    except Exception as e:
        print(f"Error decoding SAFE string {safe_str}: {e}")
        return ""


def selfies_to_smiles(selfies: str) -> str | Any:
    """Converts a SELFIES string to a SMILES string.

    :param selfies: SELFIES string to convert
    :return: SMILES string
    """
    try:
        return sf.decoder(selfies)
    except Exception as e:
        print(f"Error decoding SELFIES string {selfies}: {e}")
        return ""


def deepsmiles_to_smiles(deepsmiles: str) -> str:
    """Converts a DeepSMILES string to a SMILES string.

    :param deepsmiles: DeepSMILES string to convert
    :return: SMILES string
    """
    try:
        converter = Converter(rings=True, branches=True)
        return converter.decode(deepsmiles)
    except Exception as e:
        print(f"Error decoding DeepSMILES string {deepsmiles}: {e}")
        return ""

def get_cache_dir(dataset_name: str) -> Path:
    """
    Return the local cache directory used by 🤗 Datasets for a given repo.

    The transformation replicates what 🤗 does internally:
        `namespace/dataset-name`  -->  ~/.cache/huggingface/datasets/<hash>...
    but without the hash, so we can build an absolute path deterministically.

    Example
    -------
    >>> get_cache_dir("MolGen/ZINC_1B-raw")
    PosixPath('~/.cache/huggingface/datasets/MolGen___zinc_1b_raw')
    """
    ns, ds = dataset_name.split("/", 1)
    ds_snake = camelcase_to_snakecase(ds)
    rel_path = f"{ns}___{ds_snake}"
    return Path(HF_CACHE_HOME) / "datasets" / rel_path



def load_valid_and_test_data(
    dataset_name: str = "MolGen/ZINC_1B-raw",
    subset: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download (if needed) and return the *pickled* validity–novelty statistics
    released alongside NovoMolGen.

    Parameters
    ----------
    dataset_name : str
        Hugging-Face dataset repo ID, e.g. `"MolGen/ZINC_1B-raw"`.
    subset : str | None
        Size tag such as `"175k"`; if ``None`` the full files are used.

    Returns
    -------
    df_test, df_valid : (pd.DataFrame, pd.DataFrame)
        DataFrames loaded from `test_stats[_subset].pkl`
        and `valid_stats[_subset].pkl`.
    """
    cache_dir = get_cache_dir(dataset_name)

    tag = f"_{subset}" if subset else ""
    test_file = f"test_stats{tag}.pkl"
    valid_file = f"valid_stats{tag}.pkl"

    test_path = hf_hub_download(
        repo_id=dataset_name,
        filename=test_file,
        cache_dir=cache_dir,
        repo_type="dataset",
    )
    valid_path = hf_hub_download(
        repo_id=dataset_name,
        filename=valid_file,
        cache_dir=cache_dir,
        repo_type="dataset",
    )

    return pd.read_pickle(test_path), pd.read_pickle(valid_path)