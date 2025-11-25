#!/usr/bin/env python3
"""
Filter out samples with None values in condition features from training and validation datasets.
"""

import sys
from pathlib import Path
import yaml
import numpy as np
from datasets import load_from_disk, load_dataset

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def load_config(config_path: str):
    """Load dataset configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def has_invalid_condition_features(sample):
    """Check if a sample has invalid condition features (None, NaN, Inf)."""
    condition_keys = ["ifp"]
    
    for key in condition_keys:
        if key not in sample:
            continue
        
        value = sample[key]
        
        # Check for None
        if value is None:
            return True, key, "None"
        
        # Check for empty
        if isinstance(value, (list, np.ndarray)) and len(value) == 0:
            return True, key, "empty"
        
        # Check for None in list/array
        if isinstance(value, (list, np.ndarray)):
            try:
                arr = np.array(value, dtype=np.float32)
                if np.any(np.isnan(arr)):
                    return True, key, "NaN"
                if np.any(np.isinf(arr)):
                    return True, key, "Inf"
                # Check for None in list
                if None in value or any(v is None for v in value):
                    return True, key, "None in array"
            except (ValueError, TypeError):
                # Contains non-numeric values
                return True, key, "non-numeric"
    
    return False, None, None

def filter_dataset(dataset, name: str):
    """Filter out samples with invalid condition features."""
    print(f"\n{'='*80}")
    print(f"Filtering {name}")
    print(f"{'='*80}")
    print(f"Original size: {len(dataset):,}")
    
    # Find invalid samples
    invalid_indices = []
    invalid_reasons = {}
    
    for i in range(len(dataset)):
        is_invalid, key, reason = has_invalid_condition_features(dataset[i])
        if is_invalid:
            invalid_indices.append(i)
            reason_str = f"{key}: {reason}"
            if reason_str not in invalid_reasons:
                invalid_reasons[reason_str] = 0
            invalid_reasons[reason_str] += 1
    
    print(f"\nFound {len(invalid_indices):,} invalid samples ({100*len(invalid_indices)/len(dataset):.2f}%)")
    print(f"Invalid reasons breakdown:")
    for reason, count in sorted(invalid_reasons.items(), key=lambda x: x[1], reverse=True):
        print(f"  {reason}: {count:,} samples")
    
    if len(invalid_indices) == 0:
        print("✅ No invalid samples found, dataset is clean!")
        return dataset, []
    
    # Create valid indices
    valid_indices = [i for i in range(len(dataset)) if i not in invalid_indices]
    
    # Filter dataset
    filtered_dataset = dataset.select(valid_indices)
    
    print(f"\nFiltered size: {len(filtered_dataset):,} samples")
    print(f"Removed: {len(invalid_indices):,} samples")
    print(f"Kept: {len(valid_indices):,} samples ({100*len(valid_indices)/len(dataset):.2f}%)")
    
    return filtered_dataset, invalid_indices

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Filter None values from datasets")
    parser.add_argument("--config", type=str, default="configs/dataset/bdnv2.yaml",
                       help="Path to dataset config YAML file")
    parser.add_argument("--output_suffix", type=str, default="_filtered",
                       help="Suffix to add to output dataset names")
    parser.add_argument("--dry_run", action="store_true",
                       help="Only check, don't save filtered datasets")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    print("🔍 Filtering datasets to remove None values")
    print("="*80)
    
    # Process training dataset
    train_dataset_name = config.get("dataset_name")
    if Path(train_dataset_name).exists():
        print(f"\n📂 Loading training dataset from: {train_dataset_name}")
        train_dataset = load_from_disk(train_dataset_name)
    else:
        print(f"\n📂 Loading training dataset from HuggingFace: {train_dataset_name}")
        train_dataset = load_dataset(train_dataset_name, split="train")
    
    filtered_train, train_invalid = filter_dataset(train_dataset, "Training Dataset")
    
    # Process validation dataset
    filtered_val = None
    val_invalid = []
    validation_set_names = config.get("validation_set_names")
    if validation_set_names:
        if isinstance(validation_set_names, str):
            validation_set_names = [validation_set_names]
        
        for val_name in validation_set_names:
            if Path(val_name).exists():
                print(f"\n📂 Loading validation dataset from: {val_name}")
                val_dataset = load_from_disk(val_name)
            else:
                print(f"\n📂 Loading validation dataset from HuggingFace: {val_name}")
                val_dataset = load_dataset(val_name, split="train")
            
            filtered_val, val_invalid = filter_dataset(val_dataset, f"Validation Dataset ({Path(val_name).name})")
            break
    
    if args.dry_run:
        print(f"\n{'='*80}")
        print("🔍 DRY RUN - No datasets saved")
        print(f"{'='*80}")
        print(f"\nWould remove:")
        print(f"  Training: {len(train_invalid):,} samples")
        if val_invalid:
            print(f"  Validation: {len(val_invalid):,} samples")
        return
    
    # Save filtered datasets
    print(f"\n{'='*80}")
    print("💾 Saving filtered datasets")
    print(f"{'='*80}")
    
    # Save training dataset
    train_output_path = Path(train_dataset_name).parent / f"{Path(train_dataset_name).name}{args.output_suffix}"
    print(f"\n💾 Saving filtered training dataset to: {train_output_path}")
    filtered_train.save_to_disk(str(train_output_path))
    print(f"✅ Saved {len(filtered_train):,} samples")
    
    # Save validation dataset
    if filtered_val is not None:
        val_output_path = Path(validation_set_names[0]).parent / f"{Path(validation_set_names[0]).name}{args.output_suffix}"
        print(f"\n💾 Saving filtered validation dataset to: {val_output_path}")
        filtered_val.save_to_disk(str(val_output_path))
        print(f"✅ Saved {len(filtered_val):,} samples")
    
    print(f"\n{'='*80}")
    print("✅ Filtering complete!")
    print(f"{'='*80}")
    print(f"\n📝 Next steps:")
    print(f"  1. Update your dataset config to use the filtered datasets:")
    print(f"     dataset_name: \"{train_output_path}\"")
    if filtered_val is not None:
        print(f"     validation_set_names: [\"{val_output_path}\"]")
    print(f"  2. Re-run tokenization if needed:")
    print(f"     python src/main.py tokenize_dataset --config_name bdnv2 --config_dir configs/dataset")

if __name__ == "__main__":
    main()

