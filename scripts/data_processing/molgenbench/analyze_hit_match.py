#!/usr/bin/env python3
"""
Analyze how many entries in the second CSV have smiles/scaffold hits
by matching with the first CSV.

Usage:
    python scripts/data_processing/molgenbench/analyze_hit_match.py
"""

import pandas as pd
import json
from pathlib import Path

# File paths
hit_discovery_csv = "finetune_data/molgenbench_dataset/h2l_scaffold/molgenbench_paper_result/diffSBDD_moad/hitrediscovery_detail.csv"
breakpoint_csv = "finetune_data/molgenbench_dataset/h2l_scaffold/h2l_single_breakpoint_scaffolds_Version1_min20.csv"

print("=" * 80)
print("Analyzing Hit Match Between Two CSV Files")
print("=" * 80)

# Load CSV files
print(f"\nLoading hit discovery CSV: {hit_discovery_csv}")
df_hits = pd.read_csv(hit_discovery_csv)
print(f"  Columns: {df_hits.columns.tolist()}")
print(f"  Total rows: {len(df_hits)}")
print(f"\nFirst few rows:")
print(df_hits.head())

print(f"\n{'='*80}")
print(f"Loading breakpoint CSV: {breakpoint_csv}")
df_breakpoint = pd.read_csv(breakpoint_csv)
print(f"  Columns: {df_breakpoint.columns.tolist()}")
print(f"  Total rows: {len(df_breakpoint)}")
print(f"\nFirst few rows:")
print(df_breakpoint.head())

# Create mapping key for matching
# Assuming: 'uniprot' matches 'UniProt_ID' and 'series' matches 'SeriseID'
print(f"\n{'='*80}")
print("Creating matching keys...")

# Create a mapping dictionary from hit discovery CSV
# Key: (uniprot, series) -> (has_smiles_hit, has_scaffold_hit)
hit_mapping = {}
for _, row in df_hits.iterrows():
    key = (str(row['uniprot']).strip(), str(row['series']).strip())
    hit_mapping[key] = {
        'has_smiles_hit': row['has_smiles_hit'],
        'has_scaffold_hit': row['has_scaffold_hit']
    }

print(f"Created mapping for {len(hit_mapping)} entries from hit discovery CSV")

# Match with breakpoint CSV
print(f"\n{'='*80}")
print("Matching with breakpoint CSV...")

matched_count = 0
smiles_hit_count = 0
scaffold_hit_count = 0
both_hit_count = 0
either_hit_count = 0

matched_details = []

for _, row in df_breakpoint.iterrows():
    uniprot_id = str(row['UniProt_ID']).strip()
    series_id = str(row['SeriseID']).strip()
    key = (uniprot_id, series_id)
    
    if key in hit_mapping:
        matched_count += 1
        hit_info = hit_mapping[key]
        
        has_smiles = hit_info['has_smiles_hit']
        has_scaffold = hit_info['has_scaffold_hit']
        
        if has_smiles:
            smiles_hit_count += 1
        if has_scaffold:
            scaffold_hit_count += 1
        if has_smiles and has_scaffold:
            both_hit_count += 1
        if has_smiles or has_scaffold:
            either_hit_count += 1
        
        matched_details.append({
            'UniProt_ID': uniprot_id,
            'SeriseID': series_id,
            'has_smiles_hit': has_smiles,
            'has_scaffold_hit': has_scaffold
        })

# Calculate percentages
total_breakpoint = len(df_breakpoint)
match_pct = matched_count/total_breakpoint*100 if total_breakpoint > 0 else 0
unmatch_pct = (total_breakpoint - matched_count)/total_breakpoint*100 if total_breakpoint > 0 else 0

smiles_hit_pct_matched = smiles_hit_count/matched_count*100 if matched_count > 0 else 0
scaffold_hit_pct_matched = scaffold_hit_count/matched_count*100 if matched_count > 0 else 0
both_hit_pct_matched = both_hit_count/matched_count*100 if matched_count > 0 else 0
either_hit_pct_matched = either_hit_count/matched_count*100 if matched_count > 0 else 0

smiles_hit_pct_all = smiles_hit_count/total_breakpoint*100 if total_breakpoint > 0 else 0
scaffold_hit_pct_all = scaffold_hit_count/total_breakpoint*100 if total_breakpoint > 0 else 0
both_hit_pct_all = both_hit_count/total_breakpoint*100 if total_breakpoint > 0 else 0
either_hit_pct_all = either_hit_count/total_breakpoint*100 if total_breakpoint > 0 else 0

# Create statistics dictionary
statistics = {
    "total_rows_breakpoint_csv": total_breakpoint,
    "matched_rows": matched_count,
    "matched_percentage": round(match_pct, 2),
    "unmatched_rows": total_breakpoint - matched_count,
    "unmatched_percentage": round(unmatch_pct, 2),
    "among_matched_rows": {
        "total": matched_count,
        "smiles_hit_count": smiles_hit_count,
        "smiles_hit_percentage": round(smiles_hit_pct_matched, 2),
        "scaffold_hit_count": scaffold_hit_count,
        "scaffold_hit_percentage": round(scaffold_hit_pct_matched, 2),
        "both_hits_count": both_hit_count,
        "both_hits_percentage": round(both_hit_pct_matched, 2),
        "at_least_one_hit_count": either_hit_count,
        "at_least_one_hit_percentage": round(either_hit_pct_matched, 2)
    },
    "among_all_breakpoint_rows": {
        "total": total_breakpoint,
        "smiles_hit_count": smiles_hit_count,
        "smiles_hit_percentage": round(smiles_hit_pct_all, 2),
        "scaffold_hit_count": scaffold_hit_count,
        "scaffold_hit_percentage": round(scaffold_hit_pct_all, 2),
        "both_hits_count": both_hit_count,
        "both_hits_percentage": round(both_hit_pct_all, 2),
        "at_least_one_hit_count": either_hit_count,
        "at_least_one_hit_percentage": round(either_hit_pct_all, 2)
    }
}

# Print statistics
print(f"\n{'='*80}")
print("MATCHING STATISTICS")
print(f"{'='*80}")
print(f"Total rows in breakpoint CSV: {total_breakpoint}")
print(f"Matched rows (found in hit discovery CSV): {matched_count} ({match_pct:.1f}%)")
print(f"Unmatched rows: {total_breakpoint - matched_count} ({unmatch_pct:.1f}%)")
print()
print(f"Among matched rows ({matched_count} total):")
print(f"  Rows with SMILES hit: {smiles_hit_count} ({smiles_hit_pct_matched:.1f}%)")
print(f"  Rows with scaffold hit: {scaffold_hit_count} ({scaffold_hit_pct_matched:.1f}%)")
print(f"  Rows with both hits: {both_hit_count} ({both_hit_pct_matched:.1f}%)")
print(f"  Rows with at least one hit: {either_hit_count} ({either_hit_pct_matched:.1f}%)")
print()
print(f"Among all rows in breakpoint CSV ({total_breakpoint} total):")
print(f"  Rows with SMILES hit: {smiles_hit_count} ({smiles_hit_pct_all:.1f}%)")
print(f"  Rows with scaffold hit: {scaffold_hit_count} ({scaffold_hit_pct_all:.1f}%)")
print(f"  Rows with both hits: {both_hit_count} ({both_hit_pct_all:.1f}%)")
print(f"  Rows with at least one hit: {either_hit_count} ({either_hit_pct_all:.1f}%)")
print(f"{'='*80}")

# Save matched details to CSV
output_dir = Path("finetune_data/molgenbench_dataset/h2l_scaffold/molgenbench_paper_result/diffSBDD_moad")
output_file = output_dir / "hit_match_analysis.csv"
output_json = output_dir / "hit_match_statistics.json"
output_txt = output_dir / "hit_match_statistics.txt"

if matched_details:
    df_output = pd.DataFrame(matched_details)
    df_output.to_csv(output_file, index=False)
    print(f"\nMatched details saved to: {output_file}")
else:
    print("\nNo matches found, no detail CSV file created.")

# Save statistics as JSON
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(statistics, f, indent=2, ensure_ascii=False)
print(f"Statistics saved to: {output_json}")

# Save statistics as readable text
with open(output_txt, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("MATCHING STATISTICS\n")
    f.write("=" * 80 + "\n")
    f.write(f"Total rows in breakpoint CSV: {total_breakpoint}\n")
    f.write(f"Matched rows (found in hit discovery CSV): {matched_count} ({match_pct:.1f}%)\n")
    f.write(f"Unmatched rows: {total_breakpoint - matched_count} ({unmatch_pct:.1f}%)\n")
    f.write("\n")
    f.write(f"Among matched rows ({matched_count} total):\n")
    f.write(f"  Rows with SMILES hit: {smiles_hit_count} ({smiles_hit_pct_matched:.1f}%)\n")
    f.write(f"  Rows with scaffold hit: {scaffold_hit_count} ({scaffold_hit_pct_matched:.1f}%)\n")
    f.write(f"  Rows with both hits: {both_hit_count} ({both_hit_pct_matched:.1f}%)\n")
    f.write(f"  Rows with at least one hit: {either_hit_count} ({either_hit_pct_matched:.1f}%)\n")
    f.write("\n")
    f.write(f"Among all rows in breakpoint CSV ({total_breakpoint} total):\n")
    f.write(f"  Rows with SMILES hit: {smiles_hit_count} ({smiles_hit_pct_all:.1f}%)\n")
    f.write(f"  Rows with scaffold hit: {scaffold_hit_count} ({scaffold_hit_pct_all:.1f}%)\n")
    f.write(f"  Rows with both hits: {both_hit_count} ({both_hit_pct_all:.1f}%)\n")
    f.write(f"  Rows with at least one hit: {either_hit_count} ({either_hit_pct_all:.1f}%)\n")
    f.write("=" * 80 + "\n")
print(f"Statistics report saved to: {output_txt}")

# Show some examples
if matched_details:
    print(f"\nExample matched entries (first 10):")
    df_output_display = pd.DataFrame(matched_details).head(10)
    print(df_output_display.to_string(index=False))

