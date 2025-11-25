#!/usr/bin/env python3
"""
Check if max_seq_length is sufficient for a dataset.

This script loads a dataset, tokenizes samples, and checks the distribution
of sequence lengths to determine if the configured max_seq_length is adequate.
"""

import argparse
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from datasets import load_from_disk
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings

# Import conversion utilities
try:
    import safe
    SAFE_AVAILABLE = True
except ImportError:
    SAFE_AVAILABLE = False
    print("Warning: 'safe' package not available. SAFE conversion will not work.")


def check_sequence_length(
    dataset_path: str,
    tokenizer_path: str,
    mol_column: str = "SMILES",
    mol_type: str = "SMILES",
    max_seq_length: int = 64,
    sample_size: int = 10000,
    plot: bool = False,
):
    """
    Check sequence length distribution in a dataset.
    
    Args:
        dataset_path: Path to the dataset
        tokenizer_path: Path to the tokenizer
        mol_column: Column name containing molecule strings (e.g., "SMILES")
        mol_type: Target molecule type for tokenization (e.g., "SMILES", "SAFE", "SELFIES")
        max_seq_length: Configured max sequence length
        sample_size: Number of samples to check (0 = all)
        plot: Whether to create a plot
    """
    print("=" * 60)
    print("Checking Sequence Length Distribution")
    print("=" * 60)
    
    # Load dataset
    print(f"\nLoading dataset from: {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    print(f"✓ Loaded dataset with {len(dataset)} samples")
    
    # Check if column exists
    if mol_column not in dataset.features:
        available_cols = list(dataset.features.keys())
        raise ValueError(
            f"Column '{mol_column}' not found. Available columns: {available_cols}"
        )
    
    # Load tokenizer
    print(f"\nLoading tokenizer from: {tokenizer_path}...")
    tokenizer_path_obj = Path(tokenizer_path)
    if tokenizer_path_obj.is_file() and tokenizer_path_obj.suffix == '.json':
        # If it's a tokenizer.json file, load it directly
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path_obj))
    elif tokenizer_path_obj.is_dir():
        # If it's a directory, try AutoTokenizer first, then fallback
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path_obj))
        except:
            # Try loading tokenizer.json from directory
            tokenizer_json = tokenizer_path_obj / "tokenizer.json"
            if tokenizer_json.exists():
                tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_json))
            else:
                raise ValueError(f"Could not find tokenizer in {tokenizer_path_obj}")
    else:
        # Try as HuggingFace Hub name or local path
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        except:
            # Try as a tokenizer.json file path
            if Path(tokenizer_path).exists():
                tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
            else:
                raise ValueError(f"Could not load tokenizer from {tokenizer_path}")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"✓ Tokenizer loaded (vocab_size={tokenizer.vocab_size})")
    
    # Check if conversion is needed
    if mol_type != mol_column and mol_type != "SMILES":
        print(f"\n⚠️  Note: Dataset has '{mol_column}' column, but tokenizer expects '{mol_type}'")
        print(f"   Will convert {mol_column} → {mol_type} before tokenization")
        if mol_type == "SAFE" and not SAFE_AVAILABLE:
            raise ImportError("SAFE conversion requires 'safe' package. Install with: pip install safe")
    
    # Sample dataset if needed
    if sample_size > 0 and len(dataset) > sample_size:
        print(f"\nSampling {sample_size} samples for analysis...")
        indices = np.random.choice(len(dataset), sample_size, replace=False)
        dataset = dataset.select(indices)
    else:
        print(f"\nUsing all {len(dataset)} samples for analysis...")
    
    # Tokenize and collect lengths
    print("\nTokenizing samples and collecting lengths...")
    sequence_lengths = []
    truncated_count = 0
    max_length_found = 0
    
    for i in tqdm(range(len(dataset)), desc="Processing"):
        sample = dataset[i]
        mol_string = sample[mol_column]
        
        # Convert to target mol_type if needed
        if mol_type != mol_column and mol_type != "SMILES":
            if mol_type == "SAFE":
                try:
                    mol_string = safe.encode(mol_string, ignore_stereo=True)
                except Exception as e:
                    warnings.warn(f"Cannot convert to SAFE: {e}")
                    mol_string = ""  # Skip this sample
            elif mol_type == "SELFIES":
                try:
                    import selfies as sf
                    mol_string = sf.encoder(mol_string)
                except Exception as e:
                    warnings.warn(f"Cannot convert to SELFIES: {e}")
                    mol_string = ""  # Skip this sample
            elif mol_type == "Deep SMILES":
                try:
                    from deepsmiles import Converter
                    converter = Converter(rings=True, branches=True)
                    mol_string = converter.encode(mol_string)
                except Exception as e:
                    warnings.warn(f"Cannot convert to Deep SMILES: {e}")
                    mol_string = ""  # Skip this sample
        
        if not mol_string:  # Skip empty strings
            continue
        
        # Tokenize
        tokenized = tokenizer(
            mol_string,
            truncation=False,  # Don't truncate to see actual length
            add_special_tokens=True,
        )
        
        length = len(tokenized["input_ids"])
        sequence_lengths.append(length)
        
        if length > max_length_found:
            max_length_found = length
        
        if length > max_seq_length:
            truncated_count += 1
    
    sequence_lengths = np.array(sequence_lengths)
    
    # Statistics
    print("\n" + "=" * 60)
    print("Sequence Length Statistics")
    print("=" * 60)
    print(f"Total samples analyzed: {len(sequence_lengths):,}")
    print(f"Configured max_seq_length: {max_seq_length}")
    print(f"\nActual length statistics:")
    print(f"  Min: {sequence_lengths.min()}")
    print(f"  Max: {sequence_lengths.max()}")
    print(f"  Mean: {sequence_lengths.mean():.2f}")
    print(f"  Median: {np.median(sequence_lengths):.2f}")
    print(f"  Std: {sequence_lengths.std():.2f}")
    
    # Percentiles
    percentiles = [50, 75, 90, 95, 99, 99.9]
    print(f"\nPercentiles:")
    for p in percentiles:
        value = np.percentile(sequence_lengths, p)
        print(f"  {p}th percentile: {value:.2f}")
    
    # Truncation analysis
    print(f"\nTruncation Analysis:")
    print(f"  Samples that would be truncated: {truncated_count:,} ({100*truncated_count/len(sequence_lengths):.2f}%)")
    print(f"  Samples that fit: {len(sequence_lengths) - truncated_count:,} ({100*(len(sequence_lengths) - truncated_count)/len(sequence_lengths):.2f}%)")
    
    # Recommendations
    print("\n" + "=" * 60)
    print("Recommendations")
    print("=" * 60)
    
    p95 = np.percentile(sequence_lengths, 95)
    p99 = np.percentile(sequence_lengths, 99)
    
    if max_seq_length >= max_length_found:
        print(f"✅ Current max_seq_length ({max_seq_length}) is sufficient!")
        print(f"   All samples fit within the limit.")
    elif max_seq_length >= p99:
        print(f"⚠️  Current max_seq_length ({max_seq_length}) covers 99% of samples")
        print(f"   {truncated_count} samples ({100*truncated_count/len(sequence_lengths):.2f}%) will be truncated")
        print(f"   Consider increasing to {int(np.ceil(p99))} to cover 99% of samples")
    elif max_seq_length >= p95:
        print(f"⚠️  Current max_seq_length ({max_seq_length}) covers 95% of samples")
        print(f"   {truncated_count} samples ({100*truncated_count/len(sequence_lengths):.2f}%) will be truncated")
        print(f"   Recommended: {int(np.ceil(p99))} (covers 99%) or {int(np.ceil(max_length_found))} (covers all)")
    else:
        print(f"❌ Current max_seq_length ({max_seq_length}) may be too small")
        print(f"   {truncated_count} samples ({100*truncated_count/len(sequence_lengths):.2f}%) will be truncated")
        print(f"   Recommended: {int(np.ceil(p95))} (covers 95%), {int(np.ceil(p99))} (covers 99%), or {int(np.ceil(max_length_found))} (covers all)")
    
    # Plot if requested
    if plot:
        print("\nGenerating plot...")
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(sequence_lengths, bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(max_seq_length, color='r', linestyle='--', linewidth=2, label=f'Current max ({max_seq_length})')
        plt.axvline(p95, color='orange', linestyle='--', linewidth=2, label=f'95th percentile ({p95:.1f})')
        plt.axvline(p99, color='green', linestyle='--', linewidth=2, label=f'99th percentile ({p99:.1f})')
        plt.xlabel('Sequence Length (tokens)')
        plt.ylabel('Frequency')
        plt.title('Sequence Length Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        # Cumulative distribution
        sorted_lengths = np.sort(sequence_lengths)
        cumulative = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths) * 100
        plt.plot(sorted_lengths, cumulative, linewidth=2)
        plt.axvline(max_seq_length, color='r', linestyle='--', linewidth=2, label=f'Current max ({max_seq_length})')
        plt.axvline(p95, color='orange', linestyle='--', linewidth=2, label=f'95th percentile ({p95:.1f})')
        plt.axvline(p99, color='green', linestyle='--', linewidth=2, label=f'99th percentile ({p99:.1f})')
        plt.xlabel('Sequence Length (tokens)')
        plt.ylabel('Cumulative Percentage')
        plt.title('Cumulative Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        output_file = Path(dataset_path).parent / "sequence_length_analysis.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved to: {output_file}")
    
    return {
        'max_length': max_length_found,
        'mean': sequence_lengths.mean(),
        'median': np.median(sequence_lengths),
        'p95': p95,
        'p99': p99,
        'truncated_count': truncated_count,
        'truncated_percentage': 100 * truncated_count / len(sequence_lengths),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Check if max_seq_length is sufficient for a dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check training dataset with SAFE tokenizer
  python 09_check_sequence_length.py \\
    --dataset_path finetune_data/processed_data/hf_bdnv2_dataset/train \\
    --tokenizer_path ./data/tokenizers/tokenizer_bpe_None_SAFE_500_0_0.1.json \\
    --mol_column SMILES \\
    --mol_type SAFE \\
    --max_seq_length 64
  
  # Check with plot
  python 09_check_sequence_length.py \\
    --dataset_path finetune_data/processed_data/hf_bdnv2_dataset/train \\
    --tokenizer_path ./data/tokenizers/tokenizer_bpe_None_SAFE_500_0_0.1.json \\
    --max_seq_length 64 \\
    --plot
        """
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset directory"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        help="Path to the tokenizer"
    )
    parser.add_argument(
        "--mol_column",
        type=str,
        default="SMILES",
        help="Column name containing molecule strings (default: SMILES)"
    )
    parser.add_argument(
        "--mol_type",
        type=str,
        default="SMILES",
        help="Target molecule type for tokenization: SMILES, SAFE, SELFIES, or Deep SMILES (default: SMILES)"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=64,
        help="Configured max_seq_length to check (default: 64)"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=10000,
        help="Number of samples to analyze (0 = all, default: 10000)"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate a plot of the distribution"
    )
    
    args = parser.parse_args()
    
    check_sequence_length(
        args.dataset_path,
        args.tokenizer_path,
        args.mol_column,
        args.mol_type,
        args.max_seq_length,
        args.sample_size,
        args.plot,
    )


if __name__ == "__main__":
    main()

