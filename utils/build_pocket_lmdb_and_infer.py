#!/usr/bin/env python3
"""
Build a pocket LMDB from a PDB file and run Uni-Mol pocket inference to export embeddings.

Usage examples:
  python unimol/unimol/scripts/build_pocket_lmdb_and_infer.py \
    --pdb /abs/path/to/protein.pdb \
    --out-dir /abs/path/to/output \
    --job-name my_pocket \
    --dict-file /abs/path/to/unimol/example_data/pocket/dict_coarse.txt \
    --weights /abs/path/to/unimol/notebooks/pocket_pre_220816.pt \
    --radius 6.0 \
    --batch-size 16

Notes:
  - Produces an LMDB at {out-dir}/{job-name}.lmdb
  - Runs inference via unimol/infer.py (task=unimol_pocket) and writes:
      {out-dir}/ckp_{job-name}.out.pkl and {out-dir}/mol_repr.csv
"""

import argparse
import os
import re
import pickle
import lmdb
import numpy as np
import pandas as pd
from tqdm import tqdm
from biopandas.pdb import PandasPdb
import subprocess
import glob
from scipy.spatial import cKDTree


ALLOWED_MAIN_ATOMS = ["N", "CA", "C", "O", "H"]
WAT_NAMES = {"HOH", "WAT", "DOD"}


def normalize_atom_name(atom_name: str) -> str:
    atom_name = atom_name.strip().upper()
    return re.sub("\d+", "", atom_name)


def extract_pocket_from_pdb(pdb_path: str, radius: float, include_waters: bool = False, exclude_h: bool = False) -> bytes:
    pmol = PandasPdb().read_pdb(pdb_path)

    # Basic frames
    atom_df = pmol.df["ATOM"].copy()
    het_df = pmol.df["HETATM"].copy()

    if (atom_df is None or atom_df.empty) and (het_df is None or het_df.empty):
        raise ValueError(f"No ATOM/HETATM records parsed from PDB: {pdb_path}")

    # Normalize key fields for robust filtering
    for df in (atom_df, het_df):
        if df is None or df.empty:
            continue
        if "element_symbol" in df.columns:
            df["element_symbol"] = df["element_symbol"].astype(str).str.upper().str.strip()
        else:
            df["element_symbol"] = ""
        if "residue_name" in df.columns:
            df["residue_name"] = df["residue_name"].astype(str).str.upper().str.strip()
        else:
            df["residue_name"] = ""
        # Normalize atom name once
        df["normalize_atom"] = df["atom_name"].astype(str).map(normalize_atom_name)

    # Identify ligand centers from HETATM: non-water heavy atoms (element != H)
    ligand_centers = pd.DataFrame(columns=["x_coord","y_coord","z_coord"])  # default empty
    if het_df is not None and not het_df.empty:
        non_water_mask = ~het_df["residue_name"].isin(WAT_NAMES)
        heavy_mask = het_df["element_symbol"] != "H"
        lig_df = het_df[non_water_mask & heavy_mask]
        if not lig_df.empty:
            ligand_centers = lig_df[["x_coord","y_coord","z_coord"]].to_numpy(dtype=np.float32)

    # Build residue key helper (supports optional insertion code)
    def build_residue_keys(df: pd.DataFrame) -> pd.Series:
        has_ins = "insertion" in df.columns
        chain = df["chain_id"].astype(str)
        resnum = df["residue_number"].astype(str)
        if has_ins:
            ins = df["insertion"].astype(str).fillna("")
        else:
            ins = ""
        # Residue display string: "{chain}:{resnum}{ins}" (omit ins if empty)
        if has_ins:
            res_str = chain + ":" + resnum + ins.replace("nan", "")
        else:
            res_str = chain + ":" + resnum
        # Also provide tuple keys to compare
        tuple_keys = list(zip(chain.tolist(), resnum.tolist(), (ins if has_ins else pd.Series([""]*len(df))).tolist()))
        return pd.Series(tuple_keys), res_str

    # Pocket selection
    selected_atom_df = None
    mode = "no_ligand"
    if isinstance(ligand_centers, np.ndarray) and ligand_centers.size > 0 and atom_df is not None and not atom_df.empty:
        mode = "ligand"
        # Optionally exclude water residues from being candidates
        protein_df = atom_df.copy()
        # KD-tree on protein atoms (include H by default, filtered later by exclude_h)
        kdtree = cKDTree(protein_df[["x_coord","y_coord","z_coord"]].to_numpy(dtype=np.float32))
        # Query union of neighbors within radius for all ligand centers
        hits = kdtree.query_ball_point(ligand_centers, r=radius)
        # Flatten and unique indices
        if isinstance(hits, list) and len(hits) > 0:
            hit_indices = set()
            for h in hits:
                if isinstance(h, list):
                    hit_indices.update(h)
        else:
            hit_indices = set()

        if len(hit_indices) > 0:
            neighbor_atoms = protein_df.iloc[list(hit_indices)].copy()
            # Build residue keys for touched residues
            tuple_keys, _ = build_residue_keys(neighbor_atoms)
            touched_tuple_keys = set(tuple_keys.tolist())

            # Optionally filter out water residues from the touched set
            if not include_waters and not protein_df.empty and "residue_name" in protein_df.columns:
                protein_df_keys, _ = build_residue_keys(protein_df)
                is_water = protein_df["residue_name"].isin(WAT_NAMES)
                water_tuple_keys = set(protein_df_keys[is_water].tolist())
                touched_tuple_keys = {k for k in touched_tuple_keys if k not in water_tuple_keys}

            # Pocket = all atoms from protein residues in touched_tuple_keys
            prot_tuple_keys, prot_res_str = build_residue_keys(protein_df)
            keep_mask = prot_tuple_keys.apply(lambda k: k in touched_tuple_keys)
            selected_atom_df = protein_df[keep_mask].copy()

            # If nothing selected after filtering, fall back to no-ligand mode
            if selected_atom_df.empty:
                print("[WARN] Ligand detected but no residues found within radius; falling back to full-structure mode.")
                mode = "no_ligand"
        else:
            print("[WARN] Ligand detected but no nearby protein atoms within radius; falling back to full-structure mode.")
            mode = "no_ligand"

    if mode == "no_ligand":
        # Pocket = whole structure (ATOM + HETATM)
        frames = []
        if atom_df is not None and not atom_df.empty:
            frames.append(atom_df)
        if het_df is not None and not het_df.empty:
            frames.append(het_df)
        selected_atom_df = pd.concat(frames, axis=0, ignore_index=True)

    # Optionally drop hydrogens in the final pocket
    if exclude_h and not selected_atom_df.empty:
        selected_atom_df = selected_atom_df[selected_atom_df["element_symbol"] != "H"].copy()

    # Final cleanup: remove empty normalized names
    selected_atom_df = selected_atom_df[selected_atom_df["normalize_atom"] != ""].copy()

    # Build outputs
    patoms = selected_atom_df["normalize_atom"].values.tolist()
    pcoords = [selected_atom_df[["x_coord", "y_coord", "z_coord"]].to_numpy(dtype=np.float32)]
    side = [0 if a in ALLOWED_MAIN_ATOMS else 1 for a in patoms]

    # Residue strings using insertion code if available
    _, residue_str = build_residue_keys(selected_atom_df)
    residues = residue_str.values.tolist()

    pdb_id = os.path.splitext(os.path.basename(pdb_path))[0]

    return pickle.dumps(
        {
            "atoms": patoms,
            "coordinates": pcoords,
            "side": side,
            "residue": residues,
            "pdbid": pdb_id,
        },
        protocol=-1,
    )


def write_lmdb_from_pdb(pdb_path: str, out_dir: str, job_name: str, radius: float, include_waters: bool = False, exclude_h: bool = False) -> str:
    os.makedirs(out_dir, exist_ok=True)
    outputfilename = os.path.join(out_dir, job_name + ".lmdb")
    try:
        os.remove(outputfilename)
    except Exception:
        pass

    env_new = lmdb.open(
        outputfilename,
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=int(10e9),
    )
    txn_write = env_new.begin(write=True)
    # Extract pocket bytes with configured options
    inner_output = extract_pocket_from_pdb(pdb_path, radius, include_waters=include_waters, exclude_h=exclude_h)
    txn_write.put(b"0", inner_output)
    txn_write.commit()
    env_new.close()
    return outputfilename


def run_unimol_pocket_infer(data_dir: str, job_name: str, dict_file: str, weights: str, results_dir: str, batch_size: int, num_workers: int = 8) -> str:
    os.makedirs(results_dir, exist_ok=True)

    # Copy dictionary expected name next to data_dir as in notebooks
    dict_basename = os.path.basename(dict_file)
    dst_dict_path = os.path.join(data_dir, dict_basename)
    if os.path.abspath(dict_file) != os.path.abspath(dst_dict_path):
        subprocess.run(["/usr/bin/bash", "-lc", f"cp '{dict_file}' '{dst_dict_path}'"], check=True)

    # Change to Uni-Mol directory and run from there (like in the notebook)
    unimol_dir = "/home/yang2531/Documents/Softwares/Uni-Mol/unimol"
    cmd = (
        f"cd {unimol_dir} && "
        f"python unimol/infer.py --user-dir unimol {data_dir} --valid-subset {job_name} "
        f"--results-path {results_dir} --num-workers {num_workers} --ddp-backend=c10d --batch-size {batch_size} "
        f"--task unimol_pocket --loss unimol_infer --arch unimol_base --path {weights} "
        f"--dict-name {dict_basename} --log-interval 50 --log-format simple --random-token-prob 0 --leave-unmasked-prob 1.0 --mode infer"
    )
    subprocess.run(["/usr/bin/bash", "-lc", cmd], check=True)

    # Return produced pkl path
    matches = glob.glob(os.path.join(results_dir, f"*_{job_name}.out.pkl"))
    if not matches:
        raise FileNotFoundError(f"Inference pickle not found in {results_dir}")
    return matches[0]


def export_embeddings_csv(pkl_path: str, out_csv: str) -> None:
    predict = pd.read_pickle(pkl_path)
    pdb_id_list, mol_repr_list, atom_repr_list, pair_repr_list = [], [], [], []
    for batch in predict:
        sz = batch.get("bsz", 0)
        for i in range(sz):
            pdb_id_list.append(batch["data_name"][i])
            mol_repr_list.append(batch["mol_repr_cls"][i])
            atom_repr_list.append(batch["atom_repr"][i])
            pair_repr_list.append(batch["pair_repr"][i])
    df = pd.DataFrame(
        {
            "pdb_id": pdb_id_list,
            "mol_repr": mol_repr_list,
            "atom_repr": atom_repr_list,
            "pair_repr": pair_repr_list,
        }
    )
    df.to_csv(out_csv, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pocket LMDB from PDB and run Uni-Mol pocket inference")
    parser.add_argument("--pdb", required=True, help="Absolute path to input PDB file")
    parser.add_argument("--out-dir", required=True, help="Directory to write LMDB and results")
    parser.add_argument("--job-name", required=True, help="LMDB dataset name (also valid-subset)")
    parser.add_argument("--dict-file", required=True, help="Path to pocket dict file (e.g., unimol/example_data/pocket/dict_coarse.txt)")
    parser.add_argument("--weights", required=True, help="Path to pocket checkpoint .pt (e.g., pocket_pre_220816.pt)")
    parser.add_argument("--radius", type=float, default=6.0, help="Pocket selection radius in Å")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    # Optional selection controls
    parser.add_argument("--include-waters", action='store_true', help="Include water residues (HOH/WAT/DOD) in pocket selection")
    parser.add_argument("--exclude-h", action='store_true', help="Exclude hydrogen atoms from final pocket")

    args = parser.parse_args()

    pdb_path = os.path.abspath(args.pdb)
    out_dir = os.path.abspath(args.out_dir)
    dict_file = os.path.abspath(args.dict_file)
    weights = os.path.abspath(args.weights)

    lmdb_path = write_lmdb_from_pdb(pdb_path, out_dir, args.job_name, args.radius, include_waters=args.include_waters, exclude_h=args.exclude_h)
    print(f"Wrote LMDB: {lmdb_path}")

    pkl_path = run_unimol_pocket_infer(
        data_dir=out_dir,
        job_name=args.job_name,
        dict_file=dict_file,
        weights=weights,
        results_dir=out_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Inference done: {pkl_path}")

    csv_path = os.path.join(out_dir, "mol_repr.csv")
    export_embeddings_csv(pkl_path, csv_path)
    print(f"Wrote embeddings CSV: {csv_path}")


if __name__ == "__main__":
    main()


