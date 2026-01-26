#!/usr/bin/env python3
"""
Organize generated molecules from JSON file into MolGenBench folder structure.

Usage:
    python scripts/organize_generated_molecules.py \
        --json_file outputs/.../lead_optimization_results.json \
        --base_dir /path/to/MolGenBench_Version1 \
        --output_folder_name "MyCustomFolder" \
        --format smi
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict
import pandas as pd
from tqdm import tqdm


def save_molecules_to_file(
    molecules: List[str],
    output_file: Path,
    format: str = "smi",
    uniprot_id: str = "",
    serise_id: str = "",
    deduplicate: bool = True,
):
    """Save molecules to file in specified format.
    
    Args:
        molecules: List of SMILES strings
        output_file: Output file path
        format: Output format (smi, csv, txt)
        uniprot_id: UniProt ID (for metadata)
        serise_id: Series ID (for metadata)
        deduplicate: Remove duplicate SMILES (default: True)
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Deduplicate while preserving order
    if deduplicate:
        original_count = len(molecules)
        # Use dict to preserve order (Python 3.7+)
        molecules = list(dict.fromkeys(molecules))
        dedup_count = len(molecules)
        if original_count > dedup_count:
            print(f"    Deduplicated: {original_count} -> {dedup_count} ({original_count - dedup_count} duplicates removed)")
    else:
        dedup_count = len(molecules)
    
    if format == "smi":
        # SMILES format: one SMILES per line
        with open(output_file, 'w') as f:
            for smiles in molecules:
                f.write(f"{smiles}\n")
    
    elif format == "csv":
        # CSV format with metadata
        df = pd.DataFrame({
            'SMILES': molecules,
            'UniProt_ID': [uniprot_id] * len(molecules),
            'SeriseID': [serise_id] * len(molecules),
            'Molecule_Index': range(1, len(molecules) + 1),
        })
        df.to_csv(output_file, index=False)
    
    elif format == "txt":
        # Text format with header
        with open(output_file, 'w') as f:
            f.write(f"# UniProt_ID: {uniprot_id}\n")
            f.write(f"# SeriseID: {serise_id}\n")
            f.write(f"# Total molecules: {len(molecules)}\n")
            f.write("#" + "="*60 + "\n")
            for i, smiles in enumerate(molecules, 1):
                f.write(f"{i}\t{smiles}\n")
    
    else:
        raise ValueError(f"Unsupported format: {format}")


def organize_molecules(
    json_file: str,
    base_dir: str,
    output_folder_name: str,
    format: str = "smi",
    verbose: bool = True,
    deduplicate: bool = True,
):
    """Organize generated molecules into MolGenBench folder structure.
    
    Args:
        json_file: Path to lead_optimization_results.json
        base_dir: Base directory (e.g., MolGenBench_Version1)
        output_folder_name: Custom name for the final output folder
        format: Output file format (smi, csv, txt)
        verbose: Print progress information
        deduplicate: Remove duplicate SMILES (default: True)
    """
    # Load JSON file
    if verbose:
        print(f"Loading JSON file: {json_file}")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    results = data.get('results', [])
    
    if verbose:
        print(f"Found {len(results)} entries in JSON file")
        print(f"Base directory: {base_dir}")
        print(f"Output folder name: {output_folder_name}")
        print(f"Output format: {format}")
        print(f"Deduplication: {'ON' if deduplicate else 'OFF'}")
        print()
    
    base_path = Path(base_dir)
    stats = {
        'total_entries': len(results),
        'successful': 0,
        'failed': 0,
        'empty': 0,
    }
    
    # Process each entry
    for entry in tqdm(results, desc="Organizing molecules", disable=not verbose):
        uniprot_id = entry.get('UniProt_ID', '')
        serise_id = entry.get('SeriseID', '')
        generated_mols = entry.get('generated_molecules', [])
        
        if not uniprot_id or not serise_id:
            if verbose:
                print(f"  Warning: Missing UniProt_ID or SeriseID in entry: {entry}")
            stats['failed'] += 1
            continue
        
        if not generated_mols:
            if verbose:
                print(f"  Warning: No molecules generated for {uniprot_id}/{serise_id}")
            stats['empty'] += 1
            continue
        
        # Construct output path:
        # {base_dir}/{UniProt_ID}/Round1/Hit_to_Lead_Results/{SeriseID}/{output_folder_name}
        output_dir = (
            base_path / 
            uniprot_id / 
            "Round1" / 
            "Hit_to_Lead_Results" / 
            serise_id / 
            output_folder_name
        )
        
        # Determine output file name based on format and naming convention:
        # {UniProt_ID}_{SeriseID}_{output_folder_name}.{ext}
        ext = format
        if ext == "smi":
            ext = "smi"
        elif ext == "csv":
            ext = "csv"
        elif ext == "txt":
            ext = "txt"
        # Remove potentially problematic characters from folder name for file naming
        file_name = f"{uniprot_id}_{serise_id}_{output_folder_name}.{ext}"
        output_file = output_dir / file_name
        
        try:
            save_molecules_to_file(
                molecules=generated_mols,
                output_file=output_file,
                format=format,
                uniprot_id=uniprot_id,
                serise_id=serise_id,
                deduplicate=deduplicate,
            )
            stats['successful'] += 1
            
            if verbose:
                print(f"  ✓ Saved {len(generated_mols)} molecules to: {output_file}")
        
        except Exception as e:
            if verbose:
                print(f"  ✗ Failed to save {uniprot_id}/{serise_id}: {e}")
            stats['failed'] += 1
    
    # Print summary
    if verbose:
        print()
        print("=" * 60)
        print("Summary:")
        print(f"  Total entries: {stats['total_entries']}")
        print(f"  Successfully saved: {stats['successful']}")
        print(f"  Empty (no molecules): {stats['empty']}")
        print(f"  Failed: {stats['failed']}")
        print("=" * 60)
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Organize generated molecules into MolGenBench folder structure"
    )
    parser.add_argument(
        "--json_file",
        type=str,
        required=True,
        help="Path to lead_optimization_results.json",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Base directory (e.g., /path/to/MolGenBench_Version1)",
    )
    parser.add_argument(
        "--output_folder_name",
        type=str,
        required=True,
        help="Custom name for the final output folder (replaces 'DeleteHit2Lead(CrossDock)_Hit_to_Lead')",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="smi",
        choices=["smi", "csv", "txt"],
        help="Output file format (default: smi)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--no-deduplicate",
        action="store_true",
        help="Do not remove duplicate SMILES (default: deduplicate is ON)",
    )
    
    args = parser.parse_args()
    
    organize_molecules(
        json_file=args.json_file,
        base_dir=args.base_dir,
        output_folder_name=args.output_folder_name,
        format=args.format,
        verbose=not args.quiet,
        deduplicate=not args.no_deduplicate,
    )


if __name__ == "__main__":
    main()

