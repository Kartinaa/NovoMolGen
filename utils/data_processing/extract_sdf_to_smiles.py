#!/usr/bin/env python3
"""
Script to extract all ligand.sdf files from finetune_data subfolders,
convert them to SMILES, remove duplicates, and save as .smi files.
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
import logging

# Try to import RDKit, fall back to basic parsing if not available
try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    RDKIT_AVAILABLE = True
    print("RDKit is available - using RDKit for SDF to SMILES conversion")
except ImportError:
    RDKIT_AVAILABLE = False
    print("RDKit not available - using basic SDF parsing (less reliable)")

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('sdf_extraction.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def parse_sdf_basic(sdf_content):
    """
    Basic SDF parser that extracts SMILES from SDF content without RDKit.
    This is a fallback method and may not be as reliable as RDKit.
    """
    smiles_list = []
    lines = sdf_content.split('\n')
    
    for i, line in enumerate(lines):
        # Look for SMILES in various common formats
        if 'SMILES' in line.upper() or 'smiles' in line:
            # Try to extract SMILES from the next line or same line
            if i + 1 < len(lines):
                potential_smiles = lines[i + 1].strip()
                if potential_smiles and not potential_smiles.startswith('>'):
                    smiles_list.append(potential_smiles)
    
    return smiles_list

def extract_smiles_from_sdf(sdf_file_path):
    """
    Extract SMILES from an SDF file using RDKit or basic parsing.
    """
    try:
        with open(sdf_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            sdf_content = f.read()
        
        if RDKIT_AVAILABLE:
            # Use RDKit for reliable SDF to SMILES conversion
            supplier = Chem.SDMolSupplier(sdf_file_path)
            smiles_list = []
            
            for mol in supplier:
                if mol is not None:
                    try:
                        smiles = Chem.MolToSmiles(mol)
                        if smiles:
                            smiles_list.append(smiles)
                    except Exception as e:
                        logging.warning(f"Failed to convert molecule to SMILES: {e}")
                        continue
            
            return smiles_list
        else:
            # Fallback to basic parsing
            return parse_sdf_basic(sdf_content)
            
    except Exception as e:
        logging.error(f"Error reading SDF file {sdf_file_path}: {e}")
        return []

def find_all_ligand_sdf_files(base_dir):
    """
    Recursively find all ligand.sdf files in the directory structure.
    """
    ligand_files = []
    base_path = Path(base_dir)
    
    if not base_path.exists():
        logging.error(f"Base directory {base_dir} does not exist")
        return ligand_files
    
    # Search for ligand.sdf files recursively
    for sdf_file in base_path.rglob("ligand.sdf"):
        ligand_files.append(sdf_file)
    
    logging.info(f"Found {len(ligand_files)} ligand.sdf files")
    return ligand_files

def extract_and_deduplicate_smiles(ligand_files):
    """
    Extract SMILES from all SDF files and remove duplicates.
    """
    all_smiles = set()  # Use set to automatically handle duplicates
    file_stats = defaultdict(int)
    
    total_files = len(ligand_files)
    processed = 0
    
    for sdf_file in ligand_files:
        processed += 1
        if processed % 100 == 0:
            logging.info(f"Processed {processed}/{total_files} files")
        
        try:
            smiles_list = extract_smiles_from_sdf(sdf_file)
            
            if smiles_list:
                # Add to set (automatically removes duplicates)
                for smiles in smiles_list:
                    if smiles.strip():  # Only add non-empty SMILES
                        all_smiles.add(smiles.strip())
                
                file_stats['successful'] += 1
                file_stats['total_smiles'] += len(smiles_list)
            else:
                file_stats['no_smiles'] += 1
                logging.warning(f"No SMILES found in {sdf_file}")
                
        except Exception as e:
            file_stats['failed'] += 1
            logging.error(f"Failed to process {sdf_file}: {e}")
    
    logging.info(f"Processing complete:")
    logging.info(f"  - Successful files: {file_stats['successful']}")
    logging.info(f"  - Files with no SMILES: {file_stats['no_smiles']}")
    logging.info(f"  - Failed files: {file_stats['failed']}")
    logging.info(f"  - Total SMILES extracted: {file_stats['total_smiles']}")
    logging.info(f"  - Unique SMILES: {len(all_smiles)}")
    
    return list(all_smiles)

def save_smiles_to_file(smiles_list, output_file):
    """
    Save SMILES list to a file.
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for smiles in sorted(smiles_list):  # Sort for consistency
                f.write(f"{smiles}\n")
        
        logging.info(f"Saved {len(smiles_list)} unique SMILES to {output_file}")
        return True
        
    except Exception as e:
        logging.error(f"Failed to save SMILES to {output_file}: {e}")
        return False

def main():
    """Main function to orchestrate the SDF to SMILES extraction process."""
    setup_logging()
    
    # Configuration
    base_dir = "/home/yang2531/Documents/Project/NovoMolGen/finetune_data"
    output_file = "/home/yang2531/Documents/Project/NovoMolGen/extracted_ligands.smi"
    
    logging.info("Starting SDF to SMILES extraction process")
    logging.info(f"Base directory: {base_dir}")
    logging.info(f"Output file: {output_file}")
    
    # Step 1: Find all ligand.sdf files
    logging.info("Step 1: Finding all ligand.sdf files...")
    ligand_files = find_all_ligand_sdf_files(base_dir)
    
    if not ligand_files:
        logging.error("No ligand.sdf files found!")
        return False
    
    # Step 2: Extract SMILES and remove duplicates
    logging.info("Step 2: Extracting SMILES and removing duplicates...")
    unique_smiles = extract_and_deduplicate_smiles(ligand_files)
    
    if not unique_smiles:
        logging.error("No SMILES extracted!")
        return False
    
    # Step 3: Save to file
    logging.info("Step 3: Saving unique SMILES to file...")
    success = save_smiles_to_file(unique_smiles, output_file)
    
    if success:
        logging.info("SDF to SMILES extraction completed successfully!")
        return True
    else:
        logging.error("Failed to save SMILES to file!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
