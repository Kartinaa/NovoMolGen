import numpy as np
from unimol_tools import UniMolRepr
import torch
import oddt as od
from oddt import fingerprints

def calc_smi_representation(smi_list: list[str]) -> torch.Tensor:
    '''
    Calculate the molecule representation of a list of SMILES strings using Unimol-tools.
    Args:
        smi_list: list[str]
    Returns:
        list[torch.Tensor]
    '''
    clf = UniMolRepr(data_type='molecule', 
                 remove_hs=False,
                 model_name='unimolv2', # avaliable: unimolv1, unimolv2
                 model_size='1.1B', # work when model_name is unimolv2. avaliable: 84m, 164m, 310m, 570m, 1.1B.
                 )
    
    unimol_repr = clf.get_repr(smi_list, return_atomic_reprs=True)
    return torch.tensor(unimol_repr['cls_repr']).squeeze(0)


def calc_ifp_plec(prot_path: str, lig_path: str) -> torch.Tensor:
    try:
        protein=next(od.toolkit.readfile('pdb', prot_path))
        protein.protein=True
        
        ligand=next(od.toolkit.readfile('sdf', lig_path))
        
        ifp = od.fingerprints.PLEC(
            protein,ligand,
            depth_ligand = 1,
            depth_protein  = 5,
            distance_cutoff= 4.5,
            size           = 16384,
            count_bits     = True,
            sparse         = False,
            ignore_hoh     = True,
        )
        
        return torch.tensor(ifp, dtype=torch.float32)
    except Exception as e:
        return f"{prot_path} or {lig_path}: {e}"
    

if __name__ == "__main__":
    print('test')