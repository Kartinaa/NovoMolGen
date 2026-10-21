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
    """
    计算PLEC相互作用指纹。
    
    如果遇到ODDT解析错误，会尝试sanitize配体分子后重新计算。
    """
    import tempfile
    import os
    from rdkit import Chem
    
    try:
        protein = next(od.toolkit.readfile('pdb', prot_path))
        protein.protein = True
        ligand = next(od.toolkit.readfile('sdf', lig_path))
        
        # 尝试直接读取配体
        try:
            ligand = next(od.toolkit.readfile('sdf', lig_path))
        except Exception as e1:
            # 如果直接读取失败，尝试sanitize配体
            print(f"Warning: ODDT failed to read ligand directly: {e1}")
            print(f"  Attempting to sanitize ligand using RDKit...")
            
            # 使用RDKit读取并sanitize配体
            supplier = Chem.SDMolSupplier(lig_path)
            mol = None
            for m in supplier:
                if m is not None:
                    mol = m
                    break
            
            if mol is None:
                raise ValueError(f"Could not read molecule from {lig_path}")
            
            # Sanitize分子
            try:
                Chem.SanitizeMol(mol)
            except:
                # 如果sanitize失败，尝试移除问题原子或使用更宽松的方式
                print(f"  Warning: Sanitization failed, trying to fix...")
                # 移除氢原子后重新sanitize
                mol = Chem.RemoveHs(mol)
                try:
                    Chem.SanitizeMol(mol)
                except:
                    print(f"  Warning: Sanitization still failed, using molecule as-is")
            
            # 保存sanitized配体到临时文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.sdf', delete=False) as tmp_file:
                writer = Chem.SDWriter(tmp_file.name)
                writer.write(mol)
                writer.close()
                tmp_sdf_path = tmp_file.name
            
            try:
                # 使用sanitized的SDF文件
                ligand = next(od.toolkit.readfile('sdf', tmp_sdf_path))
                print(f"  ✓ Successfully read sanitized ligand")
            finally:
                # 清理临时文件
                if os.path.exists(tmp_sdf_path):
                    os.unlink(tmp_sdf_path)
        
        ifp = od.fingerprints.PLEC(
            protein, ligand,
            depth_ligand=1,
            depth_protein=5,
            distance_cutoff=4.5,
            size=16384,
            count_bits=True,
            sparse=False,
            ignore_hoh=True,
        )
        
        return torch.tensor(ifp, dtype=torch.float32)
    except Exception as e:
        return f"{prot_path} or {lig_path}: {e}"
    

if __name__ == "__main__":
    print('test')