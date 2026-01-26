#!/usr/bin/env python3
"""
Post-process generated molecules and write per-ligand .smi files.

用法示例：

  python scripts/data_processing/molgenbench/export_generated_to_smi.py \
    --generated_json outputs/generated_molecules/molgenbench_1000_keep_invalid/generated_validation.json

功能：
- 讀取 `generated_validation.json`（或類似結構的 JSON）；
- 對於其中每一條記錄，讀取：
    - `ligand_path`
    - `generated_molecules`（該 ligand 對應的生成 SMILES 列表）
- 在 `ligand_path` 對應目錄下創建：
    `<folder>/Round1/De_novo_Results/StructureSAFE_generated_molecules/StructureSAFE_generated_molecules.smi`
- .smi 文件格式爲一行一個分子：
    SMILES<TAB>name
  例如：
    CCN(CC)CC    gen_0001
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm


def load_records(json_path: Path) -> List[Dict[str, Any]]:
    """Load generation records from JSON and return a list of dicts each with ligand_path."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Case 1: 我們的 generate_from_full_model 結構：
    # {
    #   "validation_set": "...",
    #   "num_samples": ...,
    #   "generated_molecules": [...],
    #   "condition_info": [
    #       {
    #           "validation_index": int,
    #           "ligand_path": str or null,
    #           "generated_molecules": [...]
    #       },
    #       ...
    #   ],
    #   ...
    # }
    if isinstance(data, dict) and "condition_info" in data:
        records = data["condition_info"]
        # 每條 record 形如：{"validation_index": ..., "ligand_path": ..., "generated_molecules": [...]}
        return records

    # Case 2: 頂層本身就是一個 list，每個元素有 ligand_path / generated_molecules
    if isinstance(data, list):
        return data

    raise ValueError(
        f"Unrecognized JSON structure in {json_path}. "
        f"Expected dict with 'condition_info' or list of records."
    )


def write_smi_for_record(
    record: Dict[str, Any],
    smi_subdir: Path,
    smi_filename: str = "StructureSAFE_generated_molecules.smi",
) -> int:
    """
    For a single record (one ligand), write a .smi file under the requested subfolder.

    Returns:
        Number of SMILES written.
    """
    ligand_path = record.get("ligand_path", None)
    if not ligand_path:
        return 0

    ligand_path = Path(ligand_path)
    # ligand_path 所在的資料夾，例如：.../P00749
    base_dir = ligand_path.parent

    target_dir = base_dir / smi_subdir
    target_dir.mkdir(parents=True, exist_ok=True)

    smi_path = target_dir / smi_filename

    smiles_list = record.get("generated_molecules", [])
    written = 0
    with open(smi_path, "w", encoding="utf-8") as f:
        for idx, smi in enumerate(smiles_list):
            # 跳過 None 或空字串（例如 keep_invalid 情況下的失敗記錄）
            if smi is None or smi == "":
                continue
            name = f"gen_{idx:05d}"
            f.write(f"{smi}\t{name}\n")
            written += 1

    return written


def main():
    parser = argparse.ArgumentParser(
        description="Export generated SMILES into per-ligand .smi files."
    )
    parser.add_argument(
        "--generated_json",
        type=str,
        required=True,
        help="Path to generated_validation.json (or similar output from generate_from_full_model.py)",
    )
    parser.add_argument(
        "--subdir",
        type=str,
        default="Round3/De_novo_Results/StructureSAFE_xdock_122325_generated_molecules",
        help=(
            "Subdirectory (relative to each ligand folder) where .smi will be written. "
            "Default: 'Round1/De_novo_Results/StructureSAFE_generated_molecules'"
        ),
    )
    parser.add_argument(
        "--smi_filename",
        type=str,
        default="StructureSAFE_xdock_122325_generated_molecules.smi",
        help="Name of the .smi file to create in each folder (default: StructureSAFE_generated_molecules.smi)",
    )

    args = parser.parse_args()

    json_path = Path(args.generated_json).resolve()
    if not json_path.exists():
        raise FileNotFoundError(f"Generated JSON not found: {json_path}")

    smi_subdir = Path(args.subdir)

    records = load_records(json_path)
    print(f"Loaded {len(records)} records from {json_path}")

    total_written = 0
    num_ligands = 0

    for rec in tqdm(records, desc="Writing .smi per ligand"):
        n = write_smi_for_record(rec, smi_subdir, smi_filename=args.smi_filename)
        if n > 0:
            num_ligands += 1
            total_written += n

    print("Done.")
    print(f"  Ligands with written .smi files: {num_ligands}")
    print(f"  Total SMILES written: {total_written}")


if __name__ == "__main__":
    main()


