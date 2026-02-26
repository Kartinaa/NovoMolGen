#!/usr/bin/env python3
"""
从单个 PDB 文件（包含 receptor 和 ligand）生成配体分子。

完整流程：
1. 提取口袋信息（从 PDB 中识别 ligand 并提取周围口袋区域）
2. 分离 Receptor PDB（移除 HETATM 记录）
3. 计算 evolutionary embedding (evo_vec)
4. 计算 binding pocket embedding (pocket_vec)
5. 计算 SMILES embedding (ligand_vec)
6. 计算 interaction fingerprint (IFP)
7. 使用训练好的模型生成 10000 个配体分子

Usage:
    python scripts/generate_from_pdb.py \
        --pdb_path PAC1R/Bay_after_MD.pdb \
        --ligand_sdf_path PAC1R/ligand.sdf \
        --model_path outputs/xxx/checkpoint-xxx/full_model \
        --dict_file path/to/dict_coarse.txt \
        --weights path/to/pocket_pre_220816.pt \
        --output_dir outputs/generated_from_pdb \
        --num_samples 10000
"""

import argparse
import json
import logging
import pickle
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from biopandas.pdb import PandasPdb
from scipy.spatial import cKDTree
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
sys.path.append(str(Path(__file__).parent.parent / "utils"))

from data_loader.molecule_tokenizer import MoleculeTokenizer
from data_loader.utils import safe_to_smiles
from transformers import AutoTokenizer

# Import utility functions
try:
    from build_pocket_lmdb_and_infer import (
        extract_pocket_from_pdb,
        run_unimol_pocket_infer,
        write_lmdb_from_pdb,
    )
    from pocket_tensor import load_mol_repr_tensor
except ImportError as e:
    print(f"Warning: Could not import pocket utilities: {e}")

try:
    from esm_embedding_extract import ESM2PocketEmbedder
except ImportError as e:
    print(f"Warning: Could not import ESM utilities: {e}")
    ESM2PocketEmbedder = None

try:
    from condition_extract import calc_ifp_plec, calc_smi_representation
except ImportError as e:
    print(f"Warning: Could not import condition extract utilities: {e}")

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available. Cannot extract SMILES from SDF.")


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging."""
    logger = logging.getLogger("generate_from_pdb")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def detect_model_type(model_path: Path, logger: logging.Logger):
    """
    自动检测模型类型（Tanimoto 或 InfoNCE）通过读取 config.json。
    
    Returns:
        tuple: (NovoMolGen class, NovoMolGenConfig class, model_type_name)
    """
    config_json_path = model_path / "config.json"
    
    if not config_json_path.exists():
        logger.warning(f"Config file not found at {config_json_path}, using InfoNCE model as default")
        from models.modeling_novomolgen_infonce_depot import NovoMolGen, NovoMolGenConfig
        return NovoMolGen, NovoMolGenConfig, "InfoNCE"
    
    try:
        import json
        with open(config_json_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        tanimoto_weight = config_dict.get("tanimoto_weight", 0.0)
        
        if tanimoto_weight > 0:
            from models.modeling_novomolgen_tanimoto import NovoMolGen, NovoMolGenConfig
            logger.info(f"Detected Tanimoto model (tanimoto_weight={tanimoto_weight})")
            return NovoMolGen, NovoMolGenConfig, "Tanimoto"
        else:
            from models.modeling_novomolgen_infonce_depot import NovoMolGen, NovoMolGenConfig
            logger.info("Detected InfoNCE model (tanimoto_weight=0 or not set)")
            return NovoMolGen, NovoMolGenConfig, "InfoNCE"
    except Exception as e:
        logger.warning(f"Error reading config.json: {e}. Using InfoNCE model as default")
        from models.modeling_novomolgen_infonce_depot import NovoMolGen, NovoMolGenConfig
        return NovoMolGen, NovoMolGenConfig, "InfoNCE"


def extract_pocket_and_save_pdb(
    pdb_path: str,
    output_pdb_path: str,
    radius: float = 10.0,
    include_waters: bool = False,
    exclude_h: bool = False,
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    从 PDB 文件中提取口袋并保存为独立的 PDB 文件。
    
    Args:
        pdb_path: 输入 PDB 文件路径
        output_pdb_path: 输出口袋 PDB 文件路径
        radius: 口袋提取半径（Å）
        include_waters: 是否包含水分子
        exclude_h: 是否排除氢原子
        logger: Logger 实例
    
    Returns:
        输出 PDB 文件路径
    """
    if logger:
        logger.info(f"Extracting pocket from {pdb_path} with radius={radius}Å")
    
    # Read original PDB
    pmol = PandasPdb().read_pdb(pdb_path)
    atom_df = pmol.df["ATOM"].copy()
    het_df = pmol.df["HETATM"].copy() if not pmol.df["HETATM"].empty else pd.DataFrame()
    
    # Identify ligand centers from HETATM
    WAT_NAMES = {"HOH", "WAT", "DOD"}
    ligand_centers = None
    if not het_df.empty:
        non_water_mask = ~het_df["residue_name"].isin(WAT_NAMES)
        heavy_mask = het_df["element_symbol"] != "H"
        lig_df = het_df[non_water_mask & heavy_mask]
        if not lig_df.empty:
            ligand_centers = lig_df[["x_coord", "y_coord", "z_coord"]].to_numpy(dtype=np.float32)
    
    # Select pocket atoms
    selected_atom_df = None
    if ligand_centers is not None and not atom_df.empty:
        # Use KD-tree to find atoms within radius
        kdtree = cKDTree(atom_df[["x_coord", "y_coord", "z_coord"]].to_numpy(dtype=np.float32))
        hits = kdtree.query_ball_point(ligand_centers, r=radius)
        
        hit_indices = set()
        for h in hits:
            if isinstance(h, list):
                hit_indices.update(h)
        
        if len(hit_indices) > 0:
            # Get touched residues
            neighbor_atoms = atom_df.iloc[list(hit_indices)].copy()
            
            # Build residue keys
            def build_residue_keys(df):
                has_ins = "insertion" in df.columns
                chain = df["chain_id"].astype(str)
                resnum = df["residue_number"].astype(str)
                if has_ins:
                    ins = df["insertion"].astype(str).fillna("")
                else:
                    ins = ""
                tuple_keys = list(zip(chain.tolist(), resnum.tolist(), 
                                     (ins if has_ins else pd.Series([""]*len(df))).tolist()))
                return pd.Series(tuple_keys)
            
            touched_keys = set(build_residue_keys(neighbor_atoms).tolist())
            
            # Filter out waters if needed
            if not include_waters:
                all_keys = build_residue_keys(atom_df)
                is_water = atom_df["residue_name"].isin(WAT_NAMES)
                water_keys = set(all_keys[is_water].tolist())
                touched_keys = {k for k in touched_keys if k not in water_keys}
            
            # Select all atoms from touched residues
            all_keys = build_residue_keys(atom_df)
            keep_mask = all_keys.apply(lambda k: k in touched_keys)
            selected_atom_df = atom_df[keep_mask].copy()
    
    # Fallback: use all atoms if no ligand or no hits
    if selected_atom_df is None or selected_atom_df.empty:
        if logger:
            logger.warning("No pocket atoms found, using all protein atoms")
        selected_atom_df = atom_df.copy()
    
    # Optionally exclude hydrogens
    if exclude_h and not selected_atom_df.empty:
        selected_atom_df = selected_atom_df[selected_atom_df["element_symbol"] != "H"].copy()
    
    # Save to PDB file
    output_path = Path(output_pdb_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use PandasPdb to write PDB
    output_pmol = PandasPdb()
    output_pmol.df["ATOM"] = selected_atom_df
    output_pmol.to_pdb(path=str(output_path), records=['ATOM'], append_newline=True)
    
    if logger:
        logger.info(f"Saved pocket PDB to {output_path} ({len(selected_atom_df)} atoms)")
    
    return str(output_path)


def separate_receptor_pdb(
    pdb_path: str,
    output_pdb_path: str,
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    从 PDB 文件中分离 receptor（移除所有 HETATM 记录）。
    
    Args:
        pdb_path: 输入 PDB 文件路径
        output_pdb_path: 输出 receptor PDB 文件路径
        logger: Logger 实例
    
    Returns:
        输出 PDB 文件路径
    """
    if logger:
        logger.info(f"Separating receptor from {pdb_path}")
    
    output_path = Path(output_pdb_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Read PDB using PandasPdb
    pmol = PandasPdb().read_pdb(pdb_path)
    atom_df = pmol.df["ATOM"].copy()
    
    # Save only ATOM records
    output_pmol = PandasPdb()
    output_pmol.df["ATOM"] = atom_df
    # Also preserve header/remarks if available
    if hasattr(pmol, 'header') and pmol.header:
        output_pmol.header = pmol.header
    
    output_pmol.to_pdb(path=str(output_path), records=['ATOM'], append_newline=True)
    
    if logger:
        logger.info(f"Saved receptor PDB to {output_path} ({len(atom_df)} atoms)")
    
    return str(output_path)


def compute_evo_vec(
    receptor_pdb_path: str,
    pocket_pdb_path: str,
    chain_id: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """
    计算 evolutionary embedding (evo_vec)。
    
    Args:
        receptor_pdb_path: 完整 receptor PDB 文件路径
        pocket_pdb_path: 口袋 PDB 文件路径
        chain_id: 链 ID（可选）
        logger: Logger 实例
    
    Returns:
        evo_vec: numpy array of shape (1280,)
    """
    if logger:
        logger.info("Computing evolutionary embedding (evo_vec)...")
    
    if ESM2PocketEmbedder is None:
        logger.warning("ESM2PocketEmbedder not available, using zero vector")
        return np.zeros(1280, dtype=np.float32)
    
    try:
        embedder = ESM2PocketEmbedder(
            model_name="esm2_t33_650M_UR50D",
            repr_layer=33,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        pocket_res_list, pocket_emb = embedder.pocket_embeddings(
            full_pdb_path=receptor_pdb_path,
            pocket_pdb_path=pocket_pdb_path,
            chain_id=chain_id
        )
        
        # Compute mean embedding
        evo_vec = pocket_emb.mean(dim=0).numpy().astype(np.float32)
        
        if logger:
            logger.info(f"Computed evo_vec: shape {evo_vec.shape}, mean={evo_vec.mean():.4f}")
        
        return evo_vec
    except Exception as e:
        if logger:
            logger.warning(f"Failed to compute evo_vec: {e}. Using zero vector.")
        return np.zeros(1280, dtype=np.float32)


def compute_pocket_vec(
    pocket_pdb_path: str,
    dict_file: str,
    weights: str,
    output_dir: str,
    radius: float = 10.0,
    batch_size: int = 16,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """
    计算 binding pocket embedding (pocket_vec)。
    
    Args:
        pocket_pdb_path: 口袋 PDB 文件路径
        dict_file: Uni-Mol 字典文件路径
        weights: Uni-Mol 模型权重文件路径
        output_dir: 临时输出目录
        radius: 口袋半径（用于 LMDB 创建，通常与提取时相同）
        batch_size: 批次大小
        logger: Logger 实例
    
    Returns:
        pocket_vec: numpy array of shape (512,)
    """
    if logger:
        logger.info("Computing binding pocket embedding (pocket_vec)...")
    
    try:
        # Create temporary directory for this computation
        temp_dir = Path(output_dir) / "temp_pocket"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        job_name = "pocket_compute"
        
        # Step 1: Create LMDB from pocket PDB
        lmdb_path = write_lmdb_from_pdb(
            pdb_path=pocket_pdb_path,
            out_dir=str(temp_dir),
            job_name=job_name,
            radius=radius,
            include_waters=False,
            exclude_h=False
        )
        
        if logger:
            logger.info(f"Created LMDB: {lmdb_path}")
        
        # Step 2: Run Uni-Mol inference
        results_dir = str(temp_dir / "results")
        pkl_path = run_unimol_pocket_infer(
            data_dir=str(temp_dir),
            job_name=job_name,
            dict_file=dict_file,
            weights=weights,
            results_dir=results_dir,
            batch_size=batch_size,
            num_workers=4
        )
        
        if logger:
            logger.info(f"Uni-Mol inference completed: {pkl_path}")
        
        # Step 3: Load pocket vector
        pocket_tensor = load_mol_repr_tensor(pkl_path)
        pocket_vec = pocket_tensor.numpy().astype(np.float32)
        
        if logger:
            logger.info(f"Computed pocket_vec: shape {pocket_vec.shape}, mean={pocket_vec.mean():.4f}")
        
        # Cleanup temporary files (optional)
        # import shutil
        # shutil.rmtree(temp_dir, ignore_errors=True)
        
        return pocket_vec
    except Exception as e:
        if logger:
            logger.warning(f"Failed to compute pocket_vec: {e}. Using zero vector.")
        return np.zeros(512, dtype=np.float32)


def compute_ligand_vec(
    ligand_sdf_path: str,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """
    计算 SMILES embedding (ligand_vec)。
    
    Args:
        ligand_sdf_path: Ligand SDF 文件路径
        logger: Logger 实例
    
    Returns:
        ligand_vec: numpy array of shape (1536,)
    """
    if logger:
        logger.info("Computing ligand embedding (ligand_vec)...")
    
    try:
        # Step 1: Extract SMILES from SDF
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit not available")
        
        supplier = Chem.SDMolSupplier(ligand_sdf_path)
        mol = None
        for m in supplier:
            if m is not None:
                mol = m
                break
        
        if mol is None:
            raise ValueError(f"No valid molecule found in {ligand_sdf_path}")
        
        # Remove hydrogens and convert to SMILES
        mol = Chem.RemoveHs(mol)
        smiles = Chem.MolToSmiles(mol)
        
        if logger:
            logger.info(f"Extracted SMILES: {smiles[:50]}...")
        
        # Step 2: Compute embedding using UniMolRepr
        ligand_vec_tensor = calc_smi_representation([smiles])
        ligand_vec = ligand_vec_tensor.squeeze(0).numpy().astype(np.float32)
        
        if logger:
            logger.info(f"Computed ligand_vec: shape {ligand_vec.shape}, mean={ligand_vec.mean():.4f}")
        
        return ligand_vec
    except Exception as e:
        if logger:
            logger.warning(f"Failed to compute ligand_vec: {e}. Using zero vector.")
        return np.zeros(1536, dtype=np.float32)


def compute_ifp(
    receptor_pdb_path: str,
    ligand_sdf_path: str,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """
    计算 interaction fingerprint (IFP)。
    
    Args:
        receptor_pdb_path: Receptor PDB 文件路径
        ligand_sdf_path: Ligand SDF 文件路径
        logger: Logger 实例
    
    Returns:
        ifp: numpy array of shape (16384,)
    """
    if logger:
        logger.info("Computing interaction fingerprint (IFP)...")
    
    try:
        ifp_tensor = calc_ifp_plec(receptor_pdb_path, ligand_sdf_path)
        
        if isinstance(ifp_tensor, str):
            # Error message returned
            raise ValueError(ifp_tensor)
        
        ifp = ifp_tensor.numpy().astype(np.float32)
        
        if logger:
            logger.info(f"Computed IFP: shape {ifp.shape}, mean={ifp.mean():.4f}, sum={ifp.sum():.4f}")
        
        return ifp
    except Exception as e:
        if logger:
            logger.warning(f"Failed to compute IFP: {e}. Using zero vector.")
        return np.zeros(16384, dtype=np.float32)


def load_model_and_tokenizer(model_path: str, logger: logging.Logger):
    """Load model and tokenizer with automatic model type detection."""
    model_path = Path(model_path)
    
    logger.info(f"Loading model from: {model_path}")
    
    # Auto-detect model type
    NovoMolGen, NovoMolGenConfig, model_type = detect_model_type(model_path, logger)
    logger.info(f"Using {model_type} model class")
    
    # Load config
    config = NovoMolGenConfig.from_pretrained(str(model_path))
    logger.info(f"Model config: enable_cross_attn={config.enable_cross_attn}, cross_layers={config.cross_layers}")
    
    if getattr(config, "train_new_modules_only", False):
        logger.info("Overriding config.train_new_modules_only=True -> False for generation")
        config.train_new_modules_only = False
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = NovoMolGen.from_pretrained(
        str(model_path),
        config=config,
        torch_dtype=model_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = model.to(device)
    
    if torch.cuda.is_available() and model_dtype == torch.bfloat16:
        model = model.to(torch.bfloat16)
        param_dtypes = set(p.dtype for p in model.parameters())
        if torch.float32 in param_dtypes:
            logger.warning(f"Some parameters are still float32: {param_dtypes}. Converting all to bfloat16...")
            model = model.to(torch.bfloat16)
        logger.info(f"Model converted to bfloat16. Parameter dtypes: {param_dtypes}")
    
    model.eval()
    
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Load tokenizer
    tokenizer_path = model_path / "tokenizer.json"
    if tokenizer_path.exists():
        try:
            mol_tokenizer = MoleculeTokenizer.load(str(tokenizer_path))
            tokenizer = mol_tokenizer.get_pretrained()
            logger.info(f"Loaded tokenizer as MoleculeTokenizer from {tokenizer_path}")
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            logger.info(f"Loaded tokenizer as AutoTokenizer from {model_path}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        logger.info(f"Loaded tokenizer from {model_path}")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    final_model_dtype = next(model.parameters()).dtype
    logger.info(f"Model loaded on {device} with dtype {final_model_dtype}")
    
    return model, tokenizer, device


def generate_molecules_from_conditions(
    model,
    tokenizer,
    pocket_vec: np.ndarray,
    evo_vec: np.ndarray,
    ifp: np.ndarray,
    ligand_vec: np.ndarray,
    num_samples: int,
    device: torch.device,
    logger: logging.Logger,
    batch_size: int = 50,
    max_length: int = 64,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    max_retries: int = 10,
    sample_posterior: bool = False,
) -> Tuple[List[str], Dict]:
    """
    使用条件特征生成分子。
    
    Args:
        model: 训练好的模型
        tokenizer: Tokenizer
        pocket_vec: 口袋向量 [512]
        evo_vec: 进化向量 [1280]
        ifp: 相互作用指纹 [16384]
        ligand_vec: 配体向量 [1536]
        num_samples: 要生成的分子数量
        device: 设备
        logger: Logger
        batch_size: 批次大小
        max_length: 最大生成长度
        temperature: 采样温度
        top_k: Top-k 采样
        top_p: Top-p 采样
        max_retries: 最大重试次数
        sample_posterior: 是否从后验采样
    
    Returns:
        tuple: (generated_smiles_list, conversion_stats)
    """
    logger.info(f"Generating {num_samples} molecules using condition features")
    
    # Convert condition features to tensors
    model_dtype = next(model.parameters()).dtype
    
    pocket_vec_tensor = torch.tensor([pocket_vec], dtype=torch.float32).to(device).to(model_dtype)
    evo_vec_tensor = torch.tensor([evo_vec], dtype=torch.float32).to(device).to(model_dtype)
    ifp_tensor = torch.tensor([ifp], dtype=torch.float32).to(device).to(model_dtype)
    ligand_vec_tensor = torch.tensor([ligand_vec], dtype=torch.float32).to(device).to(model_dtype)
    
    conversion_stats = {
        "total_safe_generated": 0,
        "conversion_failed": 0,
        "conversion_empty": 0,
        "conversion_fragments": 0,
        "conversion_success": 0,
    }
    
    generated_molecules = []
    attempt = 0
    max_total_attempts = max_retries * (num_samples // batch_size + 1)
    
    while len(generated_molecules) < num_samples and attempt < max_total_attempts:
        needed = num_samples - len(generated_molecules)
        current_batch_size = min(batch_size, needed)
        
        # Repeat condition features for batch
        pocket_vec_batch = pocket_vec_tensor.repeat(current_batch_size, 1)
        evo_vec_batch = evo_vec_tensor.repeat(current_batch_size, 1)
        ifp_batch = ifp_tensor.repeat(current_batch_size, 1)
        ligand_vec_batch = ligand_vec_tensor.repeat(current_batch_size, 1)
        
        # Generate using condition features
        with torch.no_grad():
            input_ids = torch.tensor([[tokenizer.bos_token_id]], device=device).repeat(current_batch_size, 1)
            
            autocast_enabled = device.type == "cuda" and model_dtype == torch.bfloat16
            if autocast_enabled:
                with torch.autocast(device_type="cuda", dtype=model_dtype, enabled=True):
                    generated_sequences = model.generate_with_condition(
                        input_ids=input_ids,
                        pocket_vec=pocket_vec_batch,
                        evo_vec=evo_vec_batch,
                        ifp=ifp_batch,
                        ligand_vec=ligand_vec_batch,
                        sample_posterior=sample_posterior,
                        max_length=max_length,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        eos_token_id=tokenizer.eos_token_id,
                    )
            else:
                generated_sequences = model.generate_with_condition(
                    input_ids=input_ids,
                    pocket_vec=pocket_vec_batch,
                    evo_vec=evo_vec_batch,
                    ifp=ifp_batch,
                    ligand_vec=ligand_vec_batch,
                    sample_posterior=sample_posterior,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    eos_token_id=tokenizer.eos_token_id,
                )
        
        # Decode sequences
        if isinstance(generated_sequences, dict):
            sequences = generated_sequences.get('sequences', generated_sequences.get('full_safe'))
        else:
            sequences = generated_sequences
        
        # Filter EOS tokens
        sequences = model._filter_tokens_after_eos(sequences, eos_id=tokenizer.eos_token_id)
        
        # Decode to strings (SAFE format)
        decoded_strings = tokenizer.batch_decode(sequences, skip_special_tokens=True)
        decoded_strings = [s.replace(" ", "") for s in decoded_strings]
        conversion_stats["total_safe_generated"] += len(decoded_strings)
        
        # Convert SAFE to SMILES - only add valid ones
        for safe_str in decoded_strings:
            if len(generated_molecules) >= num_samples:
                break
            try:
                smiles = safe_to_smiles(safe_str)
                if smiles and '.' not in smiles:  # Only add if conversion succeeded and no fragments
                    generated_molecules.append(smiles)
                    conversion_stats["conversion_success"] += 1
                elif not smiles:
                    conversion_stats["conversion_empty"] += 1
                    logger.debug(f"Failed to convert SAFE to SMILES (empty result): {safe_str[:50]}...")
                else:
                    conversion_stats["conversion_fragments"] += 1
                    logger.debug(f"Failed to convert SAFE to SMILES (fragments): {safe_str[:50]}...")
            except Exception as e:
                conversion_stats["conversion_failed"] += 1
                logger.debug(f"Error converting SAFE to SMILES: {safe_str[:50]}... Error: {e}")
        
        attempt += 1
        if attempt % 10 == 0:
            logger.info(f"  Generated {len(generated_molecules)}/{num_samples} valid molecules (attempt {attempt})")
    
    if len(generated_molecules) < num_samples:
        logger.warning(f"Only generated {len(generated_molecules)}/{num_samples} valid molecules after {attempt} attempts")
    
    # Log conversion statistics
    logger.info(f"\n  Conversion statistics:")
    logger.info(f"    Total SAFE generated: {conversion_stats['total_safe_generated']}")
    logger.info(f"    Successful conversions: {conversion_stats['conversion_success']}")
    logger.info(f"    Failed conversions (exceptions): {conversion_stats['conversion_failed']}")
    logger.info(f"    Empty conversions: {conversion_stats['conversion_empty']}")
    logger.info(f"    Fragment conversions: {conversion_stats['conversion_fragments']}")
    total_failed = (conversion_stats['conversion_failed'] + 
                   conversion_stats['conversion_empty'] + 
                   conversion_stats['conversion_fragments'])
    if conversion_stats['total_safe_generated'] > 0:
        failure_rate = (total_failed / conversion_stats['total_safe_generated']) * 100
        success_rate = (conversion_stats['conversion_success'] / conversion_stats['total_safe_generated']) * 100
        logger.info(f"    Success rate: {success_rate:.2f}% ({conversion_stats['conversion_success']}/{conversion_stats['total_safe_generated']})")
        logger.info(f"    Failure rate: {failure_rate:.2f}% ({total_failed}/{conversion_stats['total_safe_generated']})")
    
    return generated_molecules, conversion_stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate ligand molecules from a PDB file containing receptor and ligand"
    )
    parser.add_argument(
        "--pdb_path",
        type=str,
        required=True,
        help="Path to input PDB file (contains both receptor and ligand)"
    )
    parser.add_argument(
        "--ligand_sdf_path",
        type=str,
        required=True,
        help="Path to ligand SDF file"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model directory (full_model)"
    )
    parser.add_argument(
        "--dict_file",
        type=str,
        required=True,
        help="Path to Uni-Mol dictionary file (e.g., dict_coarse.txt)"
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to Uni-Mol model weights file (e.g., pocket_pre_220816.pt)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for generated molecules and intermediate files"
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=10.0,
        help="Pocket extraction radius in Å (default: 10.0)"
    )
    parser.add_argument(
        "--chain_id",
        type=str,
        default=None,
        help="Chain ID for protein (default: first chain)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of molecules to generate (default: 10000)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="Batch size for generation (default: 50)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=64,
        help="Maximum generation length (default: 64)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling (default: 50)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling (default: 0.95)"
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=10,
        help="Maximum number of generation attempts per batch if conversion fails (default: 10)"
    )
    parser.add_argument(
        "--sample_posterior",
        action="store_true",
        help="Sample from posterior if ligand and ifp feature are not available"
    )
    parser.add_argument(
        "--save_intermediate",
        action="store_true",
        help="Save intermediate files (pocket.pdb, receptor.pdb, feature vectors)"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    logger.info("=" * 60)
    logger.info("Generate Ligands from PDB File")
    logger.info("=" * 60)
    logger.info(f"Input PDB: {args.pdb_path}")
    logger.info(f"Ligand SDF: {args.ligand_sdf_path}")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Number of samples: {args.num_samples}")
    logger.info(f"Pocket radius: {args.radius}Å")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Extract pocket
    logger.info("\n" + "=" * 60)
    logger.info("Step 1: Extracting pocket from PDB")
    logger.info("=" * 60)
    pocket_pdb_path = output_dir / "pocket.pdb"
    extract_pocket_and_save_pdb(
        args.pdb_path,
        str(pocket_pdb_path),
        radius=args.radius,
        logger=logger
    )
    
    # Step 2: Separate receptor
    logger.info("\n" + "=" * 60)
    logger.info("Step 2: Separating receptor from PDB")
    logger.info("=" * 60)
    receptor_pdb_path = output_dir / "receptor.pdb"
    separate_receptor_pdb(
        args.pdb_path,
        str(receptor_pdb_path),
        logger=logger
    )
    
    # Step 3: Compute evo_vec
    logger.info("\n" + "=" * 60)
    logger.info("Step 3: Computing evolutionary embedding (evo_vec)")
    logger.info("=" * 60)
    evo_vec = compute_evo_vec(
        str(receptor_pdb_path),
        str(pocket_pdb_path),
        chain_id=args.chain_id,
        logger=logger
    )
    
    # Step 4: Compute pocket_vec
    logger.info("\n" + "=" * 60)
    logger.info("Step 4: Computing binding pocket embedding (pocket_vec)")
    logger.info("=" * 60)
    pocket_vec = compute_pocket_vec(
        str(pocket_pdb_path),
        args.dict_file,
        args.weights,
        str(output_dir),
        radius=args.radius,
        logger=logger
    )
    
    # Step 5: Compute ligand_vec
    logger.info("\n" + "=" * 60)
    logger.info("Step 5: Computing ligand embedding (ligand_vec)")
    logger.info("=" * 60)
    ligand_vec = compute_ligand_vec(
        args.ligand_sdf_path,
        logger=logger
    )
    
    # Step 6: Compute IFP
    logger.info("\n" + "=" * 60)
    logger.info("Step 6: Computing interaction fingerprint (IFP)")
    logger.info("=" * 60)
    ifp = compute_ifp(
        str(receptor_pdb_path),
        args.ligand_sdf_path,
        logger=logger
    )
    
    # Save intermediate files if requested
    if args.save_intermediate:
        logger.info("\nSaving intermediate feature vectors...")
        np.save(output_dir / "pocket_vec.npy", pocket_vec)
        np.save(output_dir / "evo_vec.npy", evo_vec)
        np.save(output_dir / "ligand_vec.npy", ligand_vec)
        np.save(output_dir / "ifp.npy", ifp)
        logger.info("Saved intermediate files")
    
    # Step 7: Load model and generate molecules
    logger.info("\n" + "=" * 60)
    logger.info("Step 7: Loading model and generating molecules")
    logger.info("=" * 60)
    model, tokenizer, device = load_model_and_tokenizer(args.model_path, logger)
    
    generated_molecules, conversion_stats = generate_molecules_from_conditions(
        model=model,
        tokenizer=tokenizer,
        pocket_vec=pocket_vec,
        evo_vec=evo_vec,
        ifp=ifp,
        ligand_vec=ligand_vec,
        num_samples=args.num_samples,
        device=device,
        logger=logger,
        batch_size=args.batch_size,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_retries=args.max_retries,
        sample_posterior=args.sample_posterior,
    )
    
    logger.info(f"\nGenerated {len(generated_molecules)} valid SMILES molecules")
    
    # Save results
    logger.info("\n" + "=" * 60)
    logger.info("Saving results")
    logger.info("=" * 60)
    
    # Save as text file
    txt_file = output_dir / "generated_molecules.txt"
    with open(txt_file, "w", encoding="utf-8") as f:
        for mol in generated_molecules:
            f.write(f"{mol}\n")
    logger.info(f"Saved molecules to: {txt_file}")
    
    # Save as JSON
    results = {
        "input_files": {
            "pdb_path": args.pdb_path,
            "ligand_sdf_path": args.ligand_sdf_path,
        },
        "model_path": args.model_path,
        "num_samples": len(generated_molecules),
        "generated_molecules": generated_molecules,
        "conversion_stats": conversion_stats,
        "condition_features": {
            "pocket_vec_shape": list(pocket_vec.shape),
            "evo_vec_shape": list(evo_vec.shape),
            "ligand_vec_shape": list(ligand_vec.shape),
            "ifp_shape": list(ifp.shape),
        },
        "generation_params": {
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "sample_posterior": args.sample_posterior,
        }
    }
    
    json_file = output_dir / "generated_molecules.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved results to: {json_file}")
    
    # Save summary
    summary = {
        "model_path": args.model_path,
        "num_samples": len(generated_molecules),
        "conversion_stats": conversion_stats,
    }
    
    summary_file = output_dir / "generation_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved summary to: {summary_file}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Generation completed successfully!")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

