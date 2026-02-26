"""
辅助函数：用于编码配体和蛋白质的实用工具函数
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import pandas as pd
from scipy.spatial import cKDTree


def extract_ligand_coords_from_sdf(sdf_path: str) -> np.ndarray:
    """
    从SDF文件提取配体原子坐标。
    
    Args:
        sdf_path: SDF文件路径
    
    Returns:
        ligand_coords: numpy array of shape (N, 3) containing atom coordinates
    """
    try:
        from rdkit import Chem
    except ImportError:
        raise ImportError("RDKit is required to extract ligand coordinates from SDF")
    
    supplier = Chem.SDMolSupplier(sdf_path)
    mol = None
    for m in supplier:
        if m is not None:
            mol = m
            break
    
    if mol is None:
        raise ValueError(f"No valid molecule found in {sdf_path}")
    
    # Get conformer (3D coordinates)
    if mol.GetNumConformers() == 0:
        raise ValueError(f"Molecule in {sdf_path} has no conformer (3D coordinates)")
    
    conformer = mol.GetConformer()
    coords = []
    for i in range(mol.GetNumAtoms()):
        pos = conformer.GetAtomPosition(i)
        coords.append([pos.x, pos.y, pos.z])
    
    return np.array(coords, dtype=np.float32)


def _parse_pdb_atoms(pdb_path: str) -> pd.DataFrame:
    """
    解析PDB文件中的ATOM记录，返回DataFrame。
    
    Args:
        pdb_path: PDB文件路径
    
    Returns:
        DataFrame包含ATOM记录
    """
    atoms = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM  ') or line.startswith('HETATM'):
                record_type = line[0:6].strip()
                atom_serial = int(line[6:11].strip())
                atom_name = line[12:16].strip()
                alt_loc = line[16:17].strip() if len(line) > 16 else ' '
                res_name = line[17:20].strip()
                chain_id = line[21:22].strip() if len(line) > 21 else ' '
                res_seq = int(line[22:26].strip())
                i_code = line[26:27].strip() if len(line) > 26 else ' '
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                occupancy = float(line[54:60].strip()) if len(line) > 54 and line[54:60].strip() else 1.0
                temp_factor = float(line[60:66].strip()) if len(line) > 60 and line[60:66].strip() else 0.0
                element = line[76:78].strip() if len(line) > 76 and line[76:78].strip() else (atom_name[0] if atom_name else '')
                
                atoms.append({
                    'record_type': record_type,
                    'atom_serial': atom_serial,
                    'atom_name': atom_name,
                    'alt_loc': alt_loc,
                    'residue_name': res_name,
                    'chain_id': chain_id,
                    'residue_number': res_seq,
                    'insertion': i_code,
                    'x_coord': x,
                    'y_coord': y,
                    'z_coord': z,
                    'occupancy': occupancy,
                    'temp_factor': temp_factor,
                    'element_symbol': element,
                })
    
    return pd.DataFrame(atoms)


def _write_pdb_atoms(df: pd.DataFrame, output_path: str) -> None:
    """
    将DataFrame写入PDB文件（仅ATOM记录）。
    
    Args:
        df: 包含ATOM记录的DataFrame
        output_path: 输出PDB文件路径
    """
    with open(output_path, 'w') as f:
        for _, row in df.iterrows():
            # Format ATOM record according to PDB format
            record = "ATOM  "
            serial = f"{int(row['atom_serial']):5d}"
            name = f"{row['atom_name']:>4s}"
            alt_loc = row.get('alt_loc', ' ') if 'alt_loc' in row else ' '
            res_name = f"{row['residue_name']:>3s}"
            chain_id = row.get('chain_id', ' ') if 'chain_id' in row else ' '
            res_seq = f"{int(row['residue_number']):4d}"
            i_code = row.get('insertion', ' ') if 'insertion' in row else ' '
            x = f"{row['x_coord']:8.3f}"
            y = f"{row['y_coord']:8.3f}"
            z = f"{row['z_coord']:8.3f}"
            occ = f"{row.get('occupancy', 1.0):6.2f}" if 'occupancy' in row else "  1.00"
            temp = f"{row.get('temp_factor', 0.0):6.2f}" if 'temp_factor' in row else "  0.00"
            element = f"{row.get('element_symbol', ''):>2s}" if 'element_symbol' in row else "  "
            
            line = f"{record}{serial} {name}{alt_loc}{res_name} {chain_id}{res_seq}{i_code}   {x}{y}{z}{occ}{temp}          {element}\n"
            f.write(line)
        f.write("END\n")


def extract_pocket_from_pdb_with_ligand(
    pdb_path: str,
    ligand_coords: np.ndarray,
    output_pdb_path: str,
    radius: float = 10.0,
    include_waters: bool = False,
    exclude_h: bool = False,
) -> str:
    """
    使用配体坐标从PDB文件提取口袋并保存为独立的PDB文件。
    
    Args:
        pdb_path: 输入PDB文件路径
        ligand_coords: 配体原子坐标，shape (N, 3)
        output_pdb_path: 输出口袋PDB文件路径
        radius: 口袋提取半径（Å）
        include_waters: 是否包含水分子
        exclude_h: 是否排除氢原子
    
    Returns:
        输出PDB文件路径
    """
    # Read PDB
    atom_df = _parse_pdb_atoms(pdb_path)
    # Only keep ATOM records (not HETATM)
    atom_df = atom_df[atom_df['record_type'] == 'ATOM'].copy()
    
    if atom_df.empty:
        raise ValueError(f"No ATOM records found in {pdb_path}")
    
    WAT_NAMES = {"HOH", "WAT", "DOD"}
    
    # Use KD-tree to find atoms within radius
    coords = atom_df[["x_coord", "y_coord", "z_coord"]].to_numpy(dtype=np.float32)
    kdtree = cKDTree(coords)
    hits = kdtree.query_ball_point(ligand_coords, r=radius)
    
    hit_indices = set()
    for h in hits:
        if isinstance(h, list):
            hit_indices.update(h)
    
    if len(hit_indices) == 0:
        raise ValueError(f"No protein atoms found within {radius}Å of ligand")
    
    # Get touched residues
    neighbor_atoms = atom_df.iloc[list(hit_indices)].copy()
    
    # Build residue keys
    def build_residue_keys(df):
        chain = df['chain_id'].astype(str)
        resnum = df['residue_number'].astype(str)
        if 'insertion' in df.columns:
            ins = df['insertion'].astype(str).fillna('')
        else:
            ins = pd.Series([''] * len(df))
        tuple_keys = list(zip(chain.tolist(), resnum.tolist(), ins.tolist()))
        return pd.Series(tuple_keys)
    
    touched_keys = set(build_residue_keys(neighbor_atoms).tolist())
    
    # Filter out waters if needed
    if not include_waters:
        all_keys = build_residue_keys(atom_df)
        is_water = atom_df['residue_name'].isin(WAT_NAMES)
        water_keys = set(all_keys[is_water].tolist())
        touched_keys = {k for k in touched_keys if k not in water_keys}
    
    # Select all atoms from touched residues
    all_keys = build_residue_keys(atom_df)
    keep_mask = all_keys.apply(lambda k: k in touched_keys)
    selected_atom_df = atom_df[keep_mask].copy()
    
    # Optionally exclude hydrogens
    if exclude_h and not selected_atom_df.empty:
        if 'element_symbol' in selected_atom_df.columns:
            selected_atom_df = selected_atom_df[selected_atom_df['element_symbol'] != 'H'].copy()
    
    # Save to PDB file
    output_path = Path(output_pdb_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    _write_pdb_atoms(selected_atom_df, str(output_path))
    
    return str(output_path)


def separate_receptor_pdb(
    pdb_path: str,
    output_pdb_path: str,
) -> str:
    """
    从PDB文件中分离receptor（移除所有HETATM记录）。
    
    Args:
        pdb_path: 输入PDB文件路径
        output_pdb_path: 输出receptor PDB文件路径
    
    Returns:
        输出PDB文件路径
    """
    output_path = Path(output_pdb_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Read PDB and filter only ATOM records
    atom_df = _parse_pdb_atoms(pdb_path)
    atom_df = atom_df[atom_df['record_type'] == 'ATOM'].copy()
    
    # Save only ATOM records
    _write_pdb_atoms(atom_df, str(output_path))
    
    return str(output_path)


def save_vector(vector: np.ndarray, output_path: str) -> None:
    """
    保存向量为.npy文件。
    
    Args:
        vector: numpy数组
        output_path: 输出文件路径
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), vector)
    print(f"Saved vector to {output_path} (shape: {vector.shape}, dtype: {vector.dtype})")


def load_vector(input_path: str) -> np.ndarray:
    """
    从.npy文件加载向量。
    
    Args:
        input_path: 输入文件路径
    
    Returns:
        numpy数组
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Vector file not found: {input_path}")
    
    vector = np.load(str(input_path))
    print(f"Loaded vector from {input_path} (shape: {vector.shape}, dtype: {vector.dtype})")
    return vector

