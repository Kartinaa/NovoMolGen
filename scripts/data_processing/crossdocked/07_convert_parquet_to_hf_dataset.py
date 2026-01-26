#!/usr/bin/env python3
"""
Convert parquet files to HuggingFace datasets format using native HuggingFace methods.

This script uses `datasets.load_dataset("parquet", ...)` to efficiently load parquet files
and convert them to HuggingFace datasets format. It automatically handles numpy array
conversion and supports parallel processing.

Input structure:
  train_data/
    train_part_0000.parquet
    train_part_0001.parquet
    ...
    validation.parquet

Output:
  A HuggingFace dataset directory with train and validation splits
"""

import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Any, Optional
import glob

from datasets import load_dataset, DatasetDict, Features, Value, Sequence, Array2D


def convert_numpy_to_list_batch(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    """
    Convert numpy arrays in a batch to Python lists.
    
    This function is designed to be used with Dataset.map(batched=True).
    It processes entire columns at once for efficiency.
    
    Args:
        batch: Dictionary with column names as keys and lists of values as values
        
    Returns:
        Dictionary with numpy arrays converted to lists
    """
    converted_batch = {}
    
    for key, values in batch.items():
        # Check if first element is a numpy array
        if len(values) > 0 and isinstance(values[0], np.ndarray):
            # Convert entire column of numpy arrays to lists
            converted_batch[key] = [arr.tolist() if isinstance(arr, np.ndarray) else arr for arr in values]
        elif len(values) > 0 and isinstance(values[0], np.generic):
            # Handle numpy scalars
            converted_batch[key] = [val.item() if isinstance(val, np.generic) else val for val in values]
        else:
            # No conversion needed
            converted_batch[key] = values
    
    return converted_batch


def infer_features_schema(dataset, sample_size: int = 100) -> Optional[Features]:
    """
    Infer HuggingFace Features schema from a dataset sample.
    
    Args:
        dataset: HuggingFace Dataset to infer schema from
        sample_size: Number of samples to check for inference
        
    Returns:
        Features schema or None if inference fails
    """
    if len(dataset) == 0:
        return None
    
    try:
        # Sample a few rows to infer types
        sample_indices = min(sample_size, len(dataset))
        sample = dataset.select(range(sample_indices))
        
        features_dict = {}
        
        for feature_name in sample.features.keys():
            # Try to use existing feature type if it's already proper
            existing_type = sample.features[feature_name]
            
            # Check if it's already a proper Arrow-compatible type
            if hasattr(existing_type, 'dtype') or hasattr(existing_type, '_type'):
                # Keep existing type if it's already Arrow-compatible
                features_dict[feature_name] = existing_type
            else:
                # Infer from first sample value
                first_value = sample[0][feature_name]
                
                if isinstance(first_value, str):
                    features_dict[feature_name] = Value("string")
                elif isinstance(first_value, bool):
                    features_dict[feature_name] = Value("bool")
                elif isinstance(first_value, (int, np.integer)):
                    features_dict[feature_name] = Value("int64")
                elif isinstance(first_value, (float, np.floating)):
                    features_dict[feature_name] = Value("float32")
                elif isinstance(first_value, (list, np.ndarray)):
                    # Check if it's a 1D or 2D array
                    arr = np.asarray(first_value)
                    if arr.ndim == 1:
                        # Use Sequence without fixed length for flexibility
                        features_dict[feature_name] = Sequence(Value("float32"))
                    elif arr.ndim == 2:
                        features_dict[feature_name] = Array2D(
                            shape=(arr.shape[0], arr.shape[1]),
                            dtype="float32"
                        )
                    else:
                        # Fallback to Sequence for higher dimensions
                        features_dict[feature_name] = Sequence(Value("float32"))
                elif first_value is None:
                    # Handle None values - use string as fallback
                    features_dict[feature_name] = Value("string")
                else:
                    # Fallback to string for unknown types
                    features_dict[feature_name] = Value("string")
        
        return Features(features_dict)
    except Exception as e:
        print(f"⚠️  Schema inference failed: {e}")
        print("   Continuing without explicit schema...")
        return None


def convert_parquet_to_hf_dataset(
    train_data_dir: Path,
    output_dir: Path,
    validation_file: str = "validation.parquet",
    train_pattern: str = "train_part_*.parquet",
    num_proc: int = 4,
    batch_size: int = 1000,
    infer_schema: bool = True,
    remove_columns: Optional[List[str]] = None,
) -> DatasetDict:
    """
    Convert parquet files to HuggingFace datasets format using native HuggingFace methods.
    
    This function uses `load_dataset("parquet", ...)` to efficiently load parquet files
    and automatically handles numpy array conversion with parallel processing.
    
    Args:
        train_data_dir: Directory containing parquet files
        output_dir: Output directory for HuggingFace dataset
        validation_file: Name of validation parquet file
        train_pattern: Pattern to match training parquet files
        num_proc: Number of processes for parallel conversion
        batch_size: Batch size for numpy array conversion
        infer_schema: Whether to infer and apply Features schema
        remove_columns: List of column names to remove (e.g., ['id', 'split', 'protein_path', 'ligand_path'])
        
    Returns:
        DatasetDict with train and validation splits
    """
    print("=" * 60)
    print("Converting Parquet Files to HuggingFace Datasets")
    print("=" * 60)
    
    train_data_dir = Path(train_data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all training parquet files
    train_files = sorted(glob.glob(str(train_data_dir / train_pattern)))
    
    # If no files found with pattern, try to use train_data_dir as a single file
    if not train_files:
        if train_data_dir.is_file() and train_data_dir.suffix == '.parquet':
            # train_data_dir is actually a single parquet file
            train_files = [str(train_data_dir)]
            train_data_dir = train_data_dir.parent
            print(f"\n⚠️  Treating input as single parquet file: {train_files[0]}")
        else:
            raise FileNotFoundError(
                f"No training files found matching pattern: {train_pattern}\n"
                f"  Searched in: {train_data_dir}\n"
                f"  Tip: If you have a single parquet file, pass it directly as --train_data_dir"
            )
    
    validation_file_path = train_data_dir / validation_file
    
    print(f"\nFound {len(train_files)} training parquet file(s)")
    if len(train_files) <= 5:
        for f in train_files:
            print(f"  - {Path(f).name}")
    else:
        for f in train_files[:3]:
            print(f"  - {Path(f).name}")
        print(f"  ... and {len(train_files) - 3} more")
    print(f"Validation file: {validation_file_path}")
    print(f"Using {num_proc} processes for parallel conversion")
    
    # Load training data using HuggingFace's native parquet loader
    print("\n" + "=" * 60)
    print("Loading training data from parquet files...")
    print("=" * 60)
    
    # Prepare data_files dict for load_dataset
    # Use glob pattern or list of files
    train_data_files = [str(f) for f in train_files]
    
    print(f"Loading {len(train_data_files)} training files...")
    train_dataset = load_dataset(
        "parquet",
        data_files=train_data_files,
        split="train",
        num_proc=num_proc,
    )
    
    print(f"✓ Loaded {len(train_dataset)} training samples")
    print(f"  Features: {list(train_dataset.features.keys())}")
    
    # Rename 'standardize_smi' to 'SMILES' if needed (for compatibility with main.py)
    if 'standardize_smi' in train_dataset.features and 'SMILES' not in train_dataset.features:
        print("\nRenaming 'standardize_smi' to 'SMILES' for compatibility with main.py...")
        train_dataset = train_dataset.rename_column('standardize_smi', 'SMILES')
        print("✓ Renamed 'standardize_smi' to 'SMILES'")
    
    # Remove unwanted columns if specified
    if remove_columns:
        # Only remove columns that actually exist
        columns_to_remove = [col for col in remove_columns if col in train_dataset.features]
        if columns_to_remove:
            print(f"\nRemoving columns: {columns_to_remove}")
            train_dataset = train_dataset.remove_columns(columns_to_remove)
            print(f"✓ Removed {len(columns_to_remove)} columns")
            print(f"  Remaining features: {list(train_dataset.features.keys())}")
        else:
            print(f"\n⚠️  No columns to remove found (specified: {remove_columns})")
    
    # Convert numpy arrays to lists
    print("\n" + "=" * 60)
    print("Converting numpy arrays to lists...")
    print("=" * 60)
    
    train_dataset = train_dataset.map(
        convert_numpy_to_list_batch,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        desc="Converting numpy arrays",
    )
    
    print("✓ Conversion completed")
    
    # Load validation data if it exists
    print("\n" + "=" * 60)
    print("Loading validation data...")
    print("=" * 60)
    
    if validation_file_path.exists():
        print(f"Loading {validation_file_path.name}...")
        validation_dataset = load_dataset(
            "parquet",
            data_files=str(validation_file_path),
            split="train",
            num_proc=num_proc,
        )
        
        print(f"✓ Loaded {len(validation_dataset)} validation samples")
        
        # Rename 'standardize_smi' to 'SMILES' if needed (for compatibility with main.py)
        if 'standardize_smi' in validation_dataset.features and 'SMILES' not in validation_dataset.features:
            print("\nRenaming 'standardize_smi' to 'SMILES' for validation set...")
            validation_dataset = validation_dataset.rename_column('standardize_smi', 'SMILES')
            print("✓ Renamed 'standardize_smi' to 'SMILES'")
        
        # Remove unwanted columns if specified
        if remove_columns:
            columns_to_remove = [col for col in remove_columns if col in validation_dataset.features]
            if columns_to_remove:
                validation_dataset = validation_dataset.remove_columns(columns_to_remove)
                print(f"✓ Removed {len(columns_to_remove)} columns from validation set")
        
        # Convert numpy arrays to lists
        validation_dataset = validation_dataset.map(
            convert_numpy_to_list_batch,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            desc="Converting numpy arrays",
        )
        
        print("✓ Conversion completed")
    else:
        print(f"⚠️  Validation file not found: {validation_file_path}")
        validation_dataset = None
    
    # Infer and apply schema if requested
    if infer_schema:
        print("\n" + "=" * 60)
        print("Inferring and applying Features schema...")
        print("=" * 60)
        
        train_features = infer_features_schema(train_dataset)
        if train_features is not None:
            print("✓ Inferred training features schema")
            train_dataset = train_dataset.cast(train_features)
        
        if validation_dataset is not None:
            val_features = infer_features_schema(validation_dataset)
            if val_features is not None:
                print("✓ Inferred validation features schema")
                validation_dataset = validation_dataset.cast(val_features)
    
    # Save as separate Datasets (not DatasetDict) for compatibility with main.py
    print("\n" + "=" * 60)
    print("Saving HuggingFace datasets as separate Datasets...")
    print("=" * 60)
    print(f"Saving to: {output_dir}")
    
    # Save training dataset
    train_output_dir = output_dir / "train"
    train_output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset.save_to_disk(str(train_output_dir))
    print(f"✓ Training dataset saved to: {train_output_dir}")
    
    # Save validation dataset if it exists
    validation_output_dir = None
    if validation_dataset is not None:
        validation_output_dir = output_dir / "validation"
        validation_output_dir.mkdir(parents=True, exist_ok=True)
        validation_dataset.save_to_disk(str(validation_output_dir))
        print(f"✓ Validation dataset saved to: {validation_output_dir}")
    
    # Create a DatasetDict for summary (but don't save it)
    dataset_dict = {"train": train_dataset}
    if validation_dataset is not None:
        dataset_dict["validation"] = validation_dataset
    hf_dataset = DatasetDict(dataset_dict)
    
    print("\n✓ Datasets saved successfully!")
    print("\n💡 Usage:")
    print(f"   Training dataset: {train_output_dir}")
    if validation_output_dir:
        print(f"   Validation dataset: {validation_output_dir}")
    print("\n   In your config file, use:")
    print(f"     dataset_name: \"{train_output_dir}\"")
    if validation_output_dir:
        print(f"     validation_set_names: [\"{validation_output_dir}\"]")
    
    # Print summary
    print_dataset_summary(hf_dataset)
    
    return hf_dataset


def print_dataset_summary(dataset_dict: DatasetDict) -> None:
    """Print a comprehensive summary of the dataset."""
    print("\n" + "=" * 60)
    print("Dataset Summary")
    print("=" * 60)
    
    for split_name, dataset in dataset_dict.items():
        print(f"\n{split_name.upper()} Split:")
        print(f"  Samples: {len(dataset):,}")
        print(f"  Features: {list(dataset.features.keys())}")
        
        # Print feature types
        print("\n  Feature Types:")
        for feature_name, feature_type in dataset.features.items():
            print(f"    {feature_name}: {feature_type}")
        
        # Print sample data
        if len(dataset) > 0:
            print("\n  Sample Data (first row):")
            sample = dataset[0]
            for key, value in sample.items():
                if isinstance(value, list):
                    print(f"    {key}: list, length={len(value)}")
                    if len(value) > 0:
                        print(f"      First element type: {type(value[0]).__name__}")
                elif isinstance(value, np.ndarray):
                    print(f"    {key}: numpy.ndarray, shape={value.shape}, dtype={value.dtype}")
                else:
                    value_str = str(value)
                    if len(value_str) > 50:
                        value_str = value_str[:50] + "..."
                    print(f"    {key}: {type(value).__name__}, value={value_str}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert parquet files to HuggingFace datasets format using native methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python 07_convert_parquet_to_hf_dataset.py \\
    --train_data_dir finetune_data/processed_data/train_data \\
    --output_dir finetune_data/processed_data/hf_dataset
  
  # Remove unwanted columns (saves space)
  python 07_convert_parquet_to_hf_dataset.py \\
    --train_data_dir finetune_data/processed_data/train_data \\
    --output_dir finetune_data/processed_data/hf_dataset \\
    --remove_columns id split protein_path ligand_path
  
  # With parallel processing
  python 07_convert_parquet_to_hf_dataset.py \\
    --train_data_dir finetune_data/processed_data/train_data \\
    --output_dir finetune_data/processed_data/hf_dataset \\
    --num_proc 8 \\
    --batch_size 2000 \\
    --remove_columns id split protein_path ligand_path
        """
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        required=True,
        help="Directory containing parquet files, or path to a single parquet file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for HuggingFace dataset"
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        default="validation.parquet",
        help="Name of validation parquet file (default: validation.parquet)"
    )
    parser.add_argument(
        "--train_pattern",
        type=str,
        default="train_part_*.parquet",
        help="Pattern to match training files (default: train_part_*.parquet)"
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=4,
        help="Number of processes for parallel conversion (default: 4)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Batch size for numpy array conversion (default: 1000)"
    )
    parser.add_argument(
        "--no_infer_schema",
        action="store_true",
        help="Disable schema inference (faster but less type-safe)"
    )
    parser.add_argument(
        "--remove_columns",
        type=str,
        nargs="+",
        default=None,
        help="Columns to remove from dataset (e.g., --remove_columns id split protein_path ligand_path)"
    )
    
    args = parser.parse_args()
    
    train_data_dir = Path(args.train_data_dir)
    output_dir = Path(args.output_dir)
    
    if not train_data_dir.exists():
        raise FileNotFoundError(f"Training data path not found: {train_data_dir}")
    
    # If train_data_dir is a file, adjust the pattern to match it
    if train_data_dir.is_file() and train_data_dir.suffix == '.parquet':
        # Use the filename as pattern
        train_pattern = train_data_dir.name
        train_data_dir = train_data_dir.parent
    else:
        train_pattern = args.train_pattern
    
    # Convert to HuggingFace dataset
    hf_dataset = convert_parquet_to_hf_dataset(
        train_data_dir,
        output_dir,
        args.validation_file,
        train_pattern,
        args.num_proc,
        args.batch_size,
        infer_schema=not args.no_infer_schema,
        remove_columns=args.remove_columns,
    )
    
    print("\n" + "=" * 60)
    print("✅ Conversion completed successfully!")
    print("=" * 60)
    print(f"📊 Final Summary:")
    print(f"   Output directory: {output_dir}")
    print(f"   Training dataset: {output_dir / 'train'}")
    print(f"   Training samples: {len(hf_dataset['train']):,}")
    if 'validation' in hf_dataset:
        print(f"   Validation dataset: {output_dir / 'validation'}")
        print(f"   Validation samples: {len(hf_dataset['validation']):,}")
    print(f"\n💡 Usage with main.py tokenize_dataset:")
    print(f"   Create a config file (e.g., configs/dataset/bdnv2.yaml):")
    print(f"     dataset_name: \"{output_dir / 'train'}\"")
    if 'validation' in hf_dataset:
        print(f"     validation_set_names: [\"{output_dir / 'validation'}\"]")
    print(f"     tokenizer_path: \"your_tokenizer_path\"")
    print(f"     mol_type: \"SMILES\"")
    print(f"     max_seq_length: 128")
    print(f"     num_proc: 8")
    print(f"     streaming: False")


if __name__ == "__main__":
    main()
