#!/usr/bin/env python3
"""
Convert a single parquet file (validation only) to a HuggingFace dataset.

本腳本專門用於「只有 validation.parquet」的情況：
- 用 `datasets.load_dataset("parquet", ...)` 讀入一個 parquet；
- 把裡面的 numpy 向量轉成 Python list；
- 可選：把 `standardize_smi` 重命名為 `SMILES`，刪掉不需要的列；
- 保存為 HuggingFace dataset 到 `<output_dir>/validation`。
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
    data_dir: Path,
    output_dir: Path,
    validation_file: str = "validation.parquet",
    num_proc: int = 4,
    batch_size: int = 1000,
    infer_schema: bool = True,
    remove_columns: Optional[List[str]] = None,
) -> DatasetDict:
    """
    Convert a single validation parquet file to a HuggingFace dataset.
    
    Args:
        data_dir: Directory containing the validation parquet
        output_dir: Output directory for HuggingFace dataset
        validation_file: Name of validation parquet file (e.g. 'validation.parquet')
        num_proc: Number of processes for parallel conversion
        batch_size: Batch size for numpy array conversion
        infer_schema: Whether to infer and apply Features schema
        remove_columns: List of column names to remove (e.g., ['id', 'split', 'protein_path', 'ligand_path'])
        
    Returns:
        DatasetDict with a single 'validation' split
    """
    print("=" * 60)
    print("Converting Validation Parquet to HuggingFace Dataset")
    print("=" * 60)
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    validation_file_path = data_dir / validation_file
    print(f"Validation file: {validation_file_path}")
    
    if not validation_file_path.exists():
        raise FileNotFoundError(f"Validation file not found: {validation_file_path}")
    
    # Load validation data
    validation_dataset = load_dataset(
        "parquet",
        data_files=str(validation_file_path),
        split="train",
        num_proc=num_proc,
    )
    print(f"✓ Loaded {len(validation_dataset)} validation samples")
    print(f"  Features: {list(validation_dataset.features.keys())}")
    
    # Rename 'standardize_smi' to 'SMILES' if needed
    if 'standardize_smi' in validation_dataset.features and 'SMILES' not in validation_dataset.features:
        print("\nRenaming 'standardize_smi' to 'SMILES' for validation set...")
        validation_dataset = validation_dataset.rename_column('standardize_smi', 'SMILES')
        print("✓ Renamed 'standardize_smi' to 'SMILES'")
    
    # Remove unwanted columns if specified
    if remove_columns:
        columns_to_remove = [col for col in remove_columns if col in validation_dataset.features]
        if columns_to_remove:
            print(f"\nRemoving columns from validation set: {columns_to_remove}")
            validation_dataset = validation_dataset.remove_columns(columns_to_remove)
            print(f"✓ Removed {len(columns_to_remove)} columns from validation set")
    
    # Convert numpy arrays to lists
    print("\n" + "=" * 60)
    print("Converting numpy arrays to lists (validation)...")
    print("=" * 60)
    validation_dataset = validation_dataset.map(
        convert_numpy_to_list_batch,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        desc="Converting numpy arrays",
    )
    print("✓ Conversion completed")
    
    # Infer and apply schema if requested
    if infer_schema:
        print("\n" + "=" * 60)
        print("Inferring and applying Features schema (validation)...")
        print("=" * 60)
        val_features = infer_features_schema(validation_dataset)
        if val_features is not None:
            print("✓ Inferred validation features schema")
            validation_dataset = validation_dataset.cast(val_features)
    
    # Save validation dataset
    print("\n" + "=" * 60)
    print("Saving HuggingFace dataset (validation only)...")
    print("=" * 60)
    print(f"Saving to: {output_dir}")
    
    validation_output_dir = output_dir / "validation"
    validation_output_dir.mkdir(parents=True, exist_ok=True)
    validation_dataset.save_to_disk(str(validation_output_dir))
    print(f"✓ Validation dataset saved to: {validation_output_dir}")
    
    hf_dataset = DatasetDict({"validation": validation_dataset})
    
    print("\n✓ Dataset saved successfully!")
    print("\n💡 Usage:")
    print(f"   Validation dataset: {validation_output_dir}")
    print("\n   In your config or scripts, you can pass this path directly, e.g.:")
    print(f"     --validation_sets {validation_output_dir}")
    
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
        description="Convert a single validation parquet to HuggingFace dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (validation only)
  python 07_convert_parquet_to_hf_dataset.py \\
    --data_dir finetune_data/molgenbench_parquet \\
    --output_dir finetune_data/molgenbench_validation \\
    --validation_file validation.parquet \\
    --remove_columns id split protein_path ligand_path
        """
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing the validation parquet file"
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
        help="Name of validation parquet file in data_dir (default: validation.parquet)"
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
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Convert to HuggingFace dataset
    hf_dataset = convert_parquet_to_hf_dataset(
        data_dir,
        output_dir,
        args.validation_file,
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
    print(f"   Validation dataset: {output_dir / 'validation'}")
    print(f"   Validation samples: {len(hf_dataset['validation']):,}")
    print(f"\n💡 Usage with generate_from_full_model.py or main.py:")
    print(f"   --validation_sets {output_dir / 'validation'}")


if __name__ == "__main__":
    main()
