#!/usr/bin/env python3
"""
Analyze recovery rate: how many generated molecules match FilteredMols.

Usage:
    python scripts/data_processing/molgenbench/analyze_recovery_rate.py \
        --json_file outputs/.../lead_optimization_results.json \
        --csv_file finetune_data/.../h2l_single_breakpoint_scaffolds_Version1_min20_with_safe.csv \
        --output_file recovery_rate_analysis.csv
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Set, Tuple
import pandas as pd
from tqdm import tqdm

try:
    from rdkit import Chem
    from rdkit import RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog('rdApp.*')
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available. Will use string comparison only.")


def canonicalize_smiles(smiles: str) -> str:
    """Canonicalize SMILES string using RDKit.
    
    Args:
        smiles: Input SMILES string
        
    Returns:
        Canonical SMILES or original string if canonicalization fails
    """
    if not RDKIT_AVAILABLE:
        return smiles.strip()
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles.strip()
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return smiles.strip()


def get_bemis_murcko_scaffold(smiles: str) -> str:
    """Extract Bemis-Murcko scaffold from SMILES.
    
    Args:
        smiles: Input SMILES string
        
    Returns:
        Canonical SMILES of the Bemis-Murcko scaffold, or empty string if extraction fails
    """
    if not RDKIT_AVAILABLE:
        return ""
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        
        # Get Bemis-Murcko scaffold
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None:
            return ""
        
        scaffold_smiles = Chem.MolToSmiles(scaffold, canonical=True)
        return scaffold_smiles
    except Exception:
        return ""


def load_filtered_mols_from_csv(csv_file: str) -> Tuple[Dict[Tuple[str, str], Set[str]], Dict[Tuple[str, str], Set[str]]]:
    """Load FilteredMols from CSV file and extract scaffolds.
    
    Args:
        csv_file: Path to CSV file
        
    Returns:
        Tuple of:
        - Dict mapping (UniProt_ID, SeriseID) -> set of canonical SMILES
        - Dict mapping (UniProt_ID, SeriseID) -> set of canonical scaffold SMILES
    """
    print(f"Loading FilteredMols from: {csv_file}")
    df = pd.read_csv(csv_file)
    
    filtered_mols_dict = {}
    filtered_scaffolds_dict = {}
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing CSV"):
        uniprot_id = row['UniProt_ID']
        serise_id = row['SeriseID']
        filtered_mols_str = row.get('FilteredMols', '')
        
        if pd.isna(filtered_mols_str) or not filtered_mols_str:
            continue
        
        # Split by comma and canonicalize
        filtered_mols = []
        filtered_scaffolds = []
        
        for smiles in filtered_mols_str.split(','):
            smiles = smiles.strip()
            if smiles:
                canonical = canonicalize_smiles(smiles)
                if canonical:
                    filtered_mols.append(canonical)
                    
                    # Extract scaffold
                    scaffold = get_bemis_murcko_scaffold(canonical)
                    if scaffold:
                        filtered_scaffolds.append(scaffold)
        
        key = (uniprot_id, serise_id)
        filtered_mols_dict[key] = set(filtered_mols)
        filtered_scaffolds_dict[key] = set(filtered_scaffolds)
    
    print(f"Loaded {len(filtered_mols_dict)} entries with FilteredMols")
    return filtered_mols_dict, filtered_scaffolds_dict


def analyze_recovery_rate(
    json_file: str,
    csv_file: str,
    output_file: str = None,
    verbose: bool = True,
):
    """Analyze recovery rate between generated molecules and FilteredMols.
    Calculates both SMILES recovery rate and scaffold (Bemis-Murcko) recovery rate.
    
    Args:
        json_file: Path to lead_optimization_results.json
        csv_file: Path to CSV with FilteredMols
        output_file: Optional output CSV file for detailed results
        verbose: Print progress information
    """
    # Load FilteredMols and their scaffolds
    filtered_mols_dict, filtered_scaffolds_dict = load_filtered_mols_from_csv(csv_file)
    
    # Load generated molecules
    if verbose:
        print(f"\nLoading generated molecules from: {json_file}")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    results = data.get('results', [])
    if verbose:
        print(f"Found {len(results)} entries in JSON file\n")
    
    # Analyze each entry
    analysis_results = []
    total_recovered = 0
    total_filtered = 0
    total_scaffold_recovered = 0
    total_scaffold_filtered = 0
    entries_with_match = 0
    entries_with_scaffold_match = 0
    entries_processed = 0
    
    for entry in tqdm(results, desc="Analyzing recovery", disable=not verbose):
        uniprot_id = entry.get('UniProt_ID', '')
        serise_id = entry.get('SeriseID', '')
        generated_mols = entry.get('generated_molecules', [])
        
        key = (uniprot_id, serise_id)
        
        # Get FilteredMols for this entry
        filtered_mols = filtered_mols_dict.get(key, set())
        filtered_scaffolds = filtered_scaffolds_dict.get(key, set())
        
        if not filtered_mols:
            if verbose:
                print(f"  Warning: No FilteredMols found for {uniprot_id}/{serise_id}")
            continue
        
        # Canonicalize generated molecules and extract scaffolds
        generated_canonical = set()
        generated_scaffolds = set()
        
        for smiles in generated_mols:
            canonical = canonicalize_smiles(smiles)
            if canonical:
                generated_canonical.add(canonical)
        
                # Extract scaffold
                scaffold = get_bemis_murcko_scaffold(canonical)
                if scaffold:
                    generated_scaffolds.add(scaffold)
        
        # Find overlap for SMILES
        recovered = generated_canonical.intersection(filtered_mols)
        num_recovered = len(recovered)
        num_filtered = len(filtered_mols)
        num_generated = len(generated_canonical)
        
        recovery_rate = (num_recovered / num_filtered * 100) if num_filtered > 0 else 0.0
        
        # Find overlap for scaffolds
        scaffold_recovered = generated_scaffolds.intersection(filtered_scaffolds)
        num_scaffold_recovered = len(scaffold_recovered)
        num_scaffold_filtered = len(filtered_scaffolds)
        num_scaffold_generated = len(generated_scaffolds)
        
        scaffold_recovery_rate = (num_scaffold_recovered / num_scaffold_filtered * 100) if num_scaffold_filtered > 0 else 0.0
        
        # Update totals
        total_recovered += num_recovered
        total_filtered += num_filtered
        total_scaffold_recovered += num_scaffold_recovered
        total_scaffold_filtered += num_scaffold_filtered
        entries_processed += 1
        
        if num_recovered > 0:
            entries_with_match += 1
        if num_scaffold_recovered > 0:
            entries_with_scaffold_match += 1
        
        # Store results
        result = {
            'UniProt_ID': uniprot_id,
            'SeriseID': serise_id,
            'Num_FilteredMols': num_filtered,
            'Num_Generated': num_generated,
            'Num_Recovered': num_recovered,
            'Recovery_Rate_%': recovery_rate,
            'Num_FilteredScaffolds': num_scaffold_filtered,
            'Num_GeneratedScaffolds': num_scaffold_generated,
            'Num_ScaffoldRecovered': num_scaffold_recovered,
            'Scaffold_Recovery_Rate_%': scaffold_recovery_rate,
            'Recovered_Molecules': ','.join(sorted(recovered)) if recovered else '',
            'Recovered_Scaffolds': ','.join(sorted(scaffold_recovered)) if scaffold_recovered else '',
        }
        analysis_results.append(result)
        
        if verbose and (num_recovered > 0 or num_scaffold_recovered > 0):
            print(f"  ✓ {uniprot_id}/{serise_id}: SMILES {num_recovered}/{num_filtered} ({recovery_rate:.1f}%), Scaffold {num_scaffold_recovered}/{num_scaffold_filtered} ({scaffold_recovery_rate:.1f}%)")
    
    # Calculate overall statistics
    overall_recovery_rate = (total_recovered / total_filtered * 100) if total_filtered > 0 else 0.0
    avg_recovery_rate = sum(r['Recovery_Rate_%'] for r in analysis_results) / len(analysis_results) if analysis_results else 0.0
    
    overall_scaffold_recovery_rate = (total_scaffold_recovered / total_scaffold_filtered * 100) if total_scaffold_filtered > 0 else 0.0
    avg_scaffold_recovery_rate = sum(r['Scaffold_Recovery_Rate_%'] for r in analysis_results) / len(analysis_results) if analysis_results else 0.0
    
    # Print summary
    print("\n" + "=" * 80)
    print("RECOVERY RATE ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Total entries analyzed: {entries_processed}")
    print(f"Entries with at least 1 SMILES match: {entries_with_match} ({entries_with_match/entries_processed*100:.1f}%)")
    print(f"Entries with at least 1 scaffold match: {entries_with_scaffold_match} ({entries_with_scaffold_match/entries_processed*100:.1f}%)")
    print()
    print("SMILES Recovery:")
    print(f"  Total FilteredMols (reference): {total_filtered}")
    print(f"  Total recovered: {total_recovered}")
    print(f"  Overall recovery rate: {overall_recovery_rate:.2f}%")
    print(f"  Average recovery rate per entry: {avg_recovery_rate:.2f}%")
    print()
    print("Scaffold (Bemis-Murcko) Recovery:")
    print(f"  Total filtered scaffolds (reference): {total_scaffold_filtered}")
    print(f"  Total recovered scaffolds: {total_scaffold_recovered}")
    print(f"  Overall scaffold recovery rate: {overall_scaffold_recovery_rate:.2f}%")
    print(f"  Average scaffold recovery rate per entry: {avg_scaffold_recovery_rate:.2f}%")
    print("=" * 80)
    
    # Distribution statistics
    recovery_rates = []
    recovery_rates_sorted = []
    scaffold_recovery_rates = []
    scaffold_recovery_rates_sorted = []
    smiles_bins = {}
    scaffold_bins = {}
    
    if analysis_results:
        recovery_rates = [r['Recovery_Rate_%'] for r in analysis_results]
        recovery_rates_sorted = sorted(recovery_rates)
        
        scaffold_recovery_rates = [r['Scaffold_Recovery_Rate_%'] for r in analysis_results]
        scaffold_recovery_rates_sorted = sorted(scaffold_recovery_rates)
        
        print("\nSMILES Recovery Rate Distribution:")
        print(f"  Min:    {min(recovery_rates):.2f}%")
        print(f"  25th:   {recovery_rates_sorted[len(recovery_rates_sorted)//4]:.2f}%")
        print(f"  Median: {recovery_rates_sorted[len(recovery_rates_sorted)//2]:.2f}%")
        print(f"  75th:   {recovery_rates_sorted[len(recovery_rates_sorted)*3//4]:.2f}%")
        print(f"  Max:    {max(recovery_rates):.2f}%")
        print()
        
        print("Scaffold Recovery Rate Distribution:")
        print(f"  Min:    {min(scaffold_recovery_rates):.2f}%")
        print(f"  25th:   {scaffold_recovery_rates_sorted[len(scaffold_recovery_rates_sorted)//4]:.2f}%")
        print(f"  Median: {scaffold_recovery_rates_sorted[len(scaffold_recovery_rates_sorted)//2]:.2f}%")
        print(f"  75th:   {scaffold_recovery_rates_sorted[len(scaffold_recovery_rates_sorted)*3//4]:.2f}%")
        print(f"  Max:    {max(scaffold_recovery_rates):.2f}%")
        print()
        
        # Count by recovery rate bins
        def count_bins(rates):
            bins = {
                '0%': 0,
                '1-10%': 0,
                '11-25%': 0,
                '26-50%': 0,
                '51-75%': 0,
                '76-99%': 0,
                '100%': 0,
            }
            
            for rate in rates:
                if rate == 0:
                    bins['0%'] += 1
                elif rate <= 10:
                    bins['1-10%'] += 1
                elif rate <= 25:
                    bins['11-25%'] += 1
                elif rate <= 50:
                    bins['26-50%'] += 1
                elif rate <= 75:
                    bins['51-75%'] += 1
                elif rate < 100:
                    bins['76-99%'] += 1
                else:
                    bins['100%'] += 1
            return bins
        
        smiles_bins = count_bins(recovery_rates)
        scaffold_bins = count_bins(scaffold_recovery_rates)
        
        print("SMILES Recovery Rate Distribution by Bins:")
        for bin_name, count in smiles_bins.items():
            pct = count / len(recovery_rates) * 100
            print(f"  {bin_name:>10}: {count:4d} entries ({pct:5.1f}%)")
    
        print()
        print("Scaffold Recovery Rate Distribution by Bins:")
        for bin_name, count in scaffold_bins.items():
            pct = count / len(scaffold_recovery_rates) * 100
            print(f"  {bin_name:>10}: {count:4d} entries ({pct:5.1f}%)")
    
    # Prepare return data
    summary_data = {
        'total_entries': entries_processed,
        'entries_with_match': entries_with_match,
        'entries_with_scaffold_match': entries_with_scaffold_match,
        'total_filtered': total_filtered,
        'total_recovered': total_recovered,
        'overall_recovery_rate': overall_recovery_rate,
        'average_recovery_rate': avg_recovery_rate,
        'total_scaffold_filtered': total_scaffold_filtered,
        'total_scaffold_recovered': total_scaffold_recovered,
        'overall_scaffold_recovery_rate': overall_scaffold_recovery_rate,
        'average_scaffold_recovery_rate': avg_scaffold_recovery_rate,
        'smiles_distribution': {},
        'scaffold_distribution': {},
        'results': analysis_results,
    }
    
    # Add distribution data
    if analysis_results:
        summary_data['smiles_distribution'] = {
            'min': min(recovery_rates),
            'percentile_25': recovery_rates_sorted[len(recovery_rates_sorted)//4],
            'median': recovery_rates_sorted[len(recovery_rates_sorted)//2],
            'percentile_75': recovery_rates_sorted[len(recovery_rates_sorted)*3//4],
            'max': max(recovery_rates),
            'bins': smiles_bins,
        }
        
        summary_data['scaffold_distribution'] = {
            'min': min(scaffold_recovery_rates),
            'percentile_25': scaffold_recovery_rates_sorted[len(scaffold_recovery_rates_sorted)//4],
            'median': scaffold_recovery_rates_sorted[len(scaffold_recovery_rates_sorted)//2],
            'percentile_75': scaffold_recovery_rates_sorted[len(scaffold_recovery_rates_sorted)*3//4],
            'max': max(scaffold_recovery_rates),
            'bins': scaffold_bins,
        }
    
    # Save detailed results to CSV
    if output_file:
        output_path = Path(output_file)
        output_dir = output_path.parent
        output_stem = output_path.stem
        
        # Ensure the output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Save detailed CSV
        results_df = pd.DataFrame(analysis_results)
        # Sort by SMILES recovery rate first, then by scaffold recovery rate
        results_df = results_df.sort_values(['Recovery_Rate_%', 'Scaffold_Recovery_Rate_%'], ascending=[False, False])
        results_df.to_csv(output_file, index=False)
        print(f"\nDetailed results saved to: {output_file}")
        
        # 2. Save summary JSON
        summary_json_file = output_dir / f"{output_stem}_summary.json"
        with open(summary_json_file, 'w', encoding='utf-8') as f:
            # Create a copy without the full results for cleaner summary
            summary_for_json = summary_data.copy()
            summary_for_json.pop('results', None)  # Remove detailed results
            json.dump(summary_for_json, f, indent=2, ensure_ascii=False)
        print(f"Summary statistics saved to: {summary_json_file}")
        
        # 3. Save summary text report
        summary_txt_file = output_dir / f"{output_stem}_summary.txt"
        with open(summary_txt_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RECOVERY RATE ANALYSIS SUMMARY\n")
            f.write("=" * 80 + "\n")
            f.write(f"Total entries analyzed: {entries_processed}\n")
            f.write(f"Entries with at least 1 SMILES match: {entries_with_match} ({entries_with_match/entries_processed*100:.1f}%)\n")
            f.write(f"Entries with at least 1 scaffold match: {entries_with_scaffold_match} ({entries_with_scaffold_match/entries_processed*100:.1f}%)\n")
            f.write("\n")
            
            f.write("SMILES Recovery:\n")
            f.write(f"  Total FilteredMols (reference): {total_filtered}\n")
            f.write(f"  Total recovered: {total_recovered}\n")
            f.write(f"  Overall recovery rate: {overall_recovery_rate:.2f}%\n")
            f.write(f"  Average recovery rate per entry: {avg_recovery_rate:.2f}%\n")
            f.write("\n")
            
            f.write("Scaffold (Bemis-Murcko) Recovery:\n")
            f.write(f"  Total filtered scaffolds (reference): {total_scaffold_filtered}\n")
            f.write(f"  Total recovered scaffolds: {total_scaffold_recovered}\n")
            f.write(f"  Overall scaffold recovery rate: {overall_scaffold_recovery_rate:.2f}%\n")
            f.write(f"  Average scaffold recovery rate per entry: {avg_scaffold_recovery_rate:.2f}%\n")
            f.write("=" * 80 + "\n")
            
            if analysis_results:
                f.write("\nSMILES Recovery Rate Distribution:\n")
                f.write(f"  Min:    {min(recovery_rates):.2f}%\n")
                f.write(f"  25th:   {recovery_rates_sorted[len(recovery_rates_sorted)//4]:.2f}%\n")
                f.write(f"  Median: {recovery_rates_sorted[len(recovery_rates_sorted)//2]:.2f}%\n")
                f.write(f"  75th:   {recovery_rates_sorted[len(recovery_rates_sorted)*3//4]:.2f}%\n")
                f.write(f"  Max:    {max(recovery_rates):.2f}%\n")
                f.write("\n")
                
                f.write("Scaffold Recovery Rate Distribution:\n")
                f.write(f"  Min:    {min(scaffold_recovery_rates):.2f}%\n")
                f.write(f"  25th:   {scaffold_recovery_rates_sorted[len(scaffold_recovery_rates_sorted)//4]:.2f}%\n")
                f.write(f"  Median: {scaffold_recovery_rates_sorted[len(scaffold_recovery_rates_sorted)//2]:.2f}%\n")
                f.write(f"  75th:   {scaffold_recovery_rates_sorted[len(scaffold_recovery_rates_sorted)*3//4]:.2f}%\n")
                f.write(f"  Max:    {max(scaffold_recovery_rates):.2f}%\n")
                f.write("\n")
                
                f.write("SMILES Recovery Rate Distribution by Bins:\n")
                for bin_name, count in smiles_bins.items():
                    pct = count / len(recovery_rates) * 100
                    f.write(f"  {bin_name:>10}: {count:4d} entries ({pct:5.1f}%)\n")
                
                f.write("\n")
                f.write("Scaffold Recovery Rate Distribution by Bins:\n")
                for bin_name, count in scaffold_bins.items():
                    pct = count / len(scaffold_recovery_rates) * 100
                    f.write(f"  {bin_name:>10}: {count:4d} entries ({pct:5.1f}%)\n")
        
        print(f"Summary report saved to: {summary_txt_file}")
    
    return summary_data


def main():
    parser = argparse.ArgumentParser(
        description="Analyze recovery rate between generated molecules and FilteredMols"
    )
    parser.add_argument(
        "--json_file",
        type=str,
        required=True,
        help="Path to lead_optimization_results.json",
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        required=True,
        help="Path to CSV file with FilteredMols column",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output CSV file for detailed results (default: recovery_rate_analysis.csv)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    
    args = parser.parse_args()
    
    # Set default output file if not specified
    if args.output_file is None:
        json_path = Path(args.json_file)
        args.output_file = str(json_path.parent / "recovery_rate_analysis.csv")
    
    analyze_recovery_rate(
        json_file=args.json_file,
        csv_file=args.csv_file,
        output_file=args.output_file,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()

