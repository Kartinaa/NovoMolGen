import torch
import pandas as pd
from typing import Union


def load_mol_repr_tensor(pkl_path: str) -> torch.Tensor:
    """Load Uni-Mol inference pickle and return mol_repr_cls as a float32 tensor.

    The pickle produced by unimol/infer.py is a list of batch dicts. Each batch
    contains a key "mol_repr_cls" which is an array-like per item.
    """
    predict: Union[list, pd.DataFrame] = pd.read_pickle(pkl_path)
    mol_repr_list = []
    for batch in predict:
        size = batch.get("bsz", 0)
        for i in range(size):
            mol_repr_list.append(batch["mol_repr_cls"][i])
    return torch.tensor(mol_repr_list, dtype=torch.float32)


