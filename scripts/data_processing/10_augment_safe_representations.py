#!/usr/bin/env python3
"""
Pre-compute multiple SAFE randomizations for data augmentation.
Generates N SAFE versions (1 canonical + N-1 randomized) per SMILES.

Usage:
    python scripts/data_processing/10_augment_safe_representations.py \
        --input_path finetune_data/crossdock_dataset/hf_dataset_crossdock/train \
        --output_path finetune_data/crossdock_dataset/hf_dataset_crossdock_augmented/train \
        --num_versions 5 \
        --seed 42
"""

import argparse
import safe
import warnings
import shutil
from pathlib import Path
from datasets import load_from_disk, load_dataset, Dataset, Features, Value, Sequence, concatenate_datasets
from tqdm import tqdm


def load_dataset_with_fallback(input_path: str, num_proc: int = 4):
    """Load dataset with fallback for compatibility issues.

    Handles datasets saved with older versions of the datasets library
    that use List instead of Sequence for features.
    """
    try:
        return load_from_disk(input_path)
    except (TypeError, AttributeError, ValueError) as e:
        warnings.warn(
            f"Failed to load dataset from {input_path}: {e}\n"
            f"Attempting fallback: loading from arrow files with explicit features..."
        )
        arrow_files = sorted(Path(input_path).glob("*.arrow"))
        if not arrow_files:
            raise ValueError(f"No arrow files found in {input_path}")

        # Define correct features with Sequence (not List)
        features = Features({
            'SMILES': Value('string'),
            'pocket_vec': Sequence(Value('float64')),
            'evo_vec': Sequence(Value('float32')),
            'ifp': Sequence(Value('float32')),
            'ligand_vec': Sequence(Value('float32')),
        })
        dataset = load_dataset(
            "arrow",
            data_files=[str(f) for f in arrow_files],
            features=features,
            split="train",
            num_proc=num_proc
        )
        print(f"Successfully loaded dataset using arrow fallback")
        return dataset


def generate_safe_versions(
    smiles: str,
    num_versions: int = 5,
    sample_seed: int = 0,
    ignore_stereo: bool = True,
) -> list:
    """Generate multiple SAFE versions for a single SMILES.

    Args:
        smiles: Input SMILES string
        num_versions: Number of SAFE versions to generate (1 canonical + N-1 randomized)
        sample_seed: Base seed for reproducible randomization
        ignore_stereo: Whether to ignore stereochemistry in SAFE encoding

    Returns:
        List of SAFE strings, or empty list if encoding fails
    """
    versions = []

    # Version 0: Canonical (deterministic)
    try:
        canonical = safe.encode(smiles, ignore_stereo=ignore_stereo)
        versions.append(canonical)
    except Exception as e:
        warnings.warn(f"Failed SAFE encoding for {smiles}: {e}")
        return []

    # Versions 1 to N-1: Randomized
    attempts = 0
    max_attempts = num_versions * 3  # Allow extra attempts for duplicates

    while len(versions) < num_versions and attempts < max_attempts:
        attempts += 1
        try:
            randomized = safe.encode(
                smiles,
                ignore_stereo=ignore_stereo,
                canonical=False,
                randomize=True,
                seed=sample_seed * 1000 + attempts,
            )
            if randomized not in versions:  # Avoid duplicates
                versions.append(randomized)
        except Exception:
            pass  # Skip failed randomizations

    # Pad with canonical if not enough unique versions
    while len(versions) < num_versions:
        versions.append(canonical)

    return versions[:num_versions]


def augment_dataset(
    input_path: str,
    output_path: str,
    num_versions: int = 5,
    seed: int = 42,
    num_proc: int = 4,
    batch_size: int = 10000,
):
    """Create augmented dataset with multiple SAFE versions per sample.

    Processes in batches to avoid memory issues with large datasets.

    Args:
        input_path: Path to input HuggingFace dataset
        output_path: Path to save augmented dataset
        num_versions: Number of SAFE versions per SMILES
        seed: Random seed for reproducible randomization
        num_proc: Number of processes for loading dataset
        batch_size: Number of samples to process before saving to disk
    """

    print(f"Loading dataset from {input_path}...")
    dataset = load_dataset_with_fallback(input_path, num_proc=num_proc)
    total_samples = len(dataset)
    print(f"Original dataset size: {total_samples}")

    # Create output directory and temp directory for shards
    output_dir = Path(output_path)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir.parent / f"{output_dir.name}_temp_shards"
    temp_dir.mkdir(parents=True, exist_ok=True)

    failed_count = 0
    shard_paths = []
    current_batch = []
    shard_idx = 0
    total_augmented = 0

    pbar = tqdm(enumerate(dataset), total=total_samples, desc="Generating SAFE versions")

    for idx, sample in pbar:
        smiles = sample.get("SMILES", sample.get("standardize_smi", ""))
        if not smiles:
            failed_count += 1
            continue

        safe_versions = generate_safe_versions(
            smiles,
            num_versions=num_versions,
            sample_seed=seed + idx,
        )

        if not safe_versions:
            failed_count += 1
            continue

        for version_idx, safe_str in enumerate(safe_versions):
            augmented_sample = dict(sample)  # Copy all fields
            augmented_sample["SAFE"] = safe_str
            augmented_sample["safe_version_idx"] = version_idx
            augmented_sample["is_canonical"] = (version_idx == 0)
            current_batch.append(augmented_sample)

        # Save shard when batch is full
        if len(current_batch) >= batch_size * num_versions:
            shard_path = temp_dir / f"shard_{shard_idx:05d}"
            shard_dataset = Dataset.from_list(current_batch)
            shard_dataset.save_to_disk(str(shard_path))
            shard_paths.append(str(shard_path))
            total_augmented += len(current_batch)
            pbar.set_postfix({"shards": shard_idx + 1, "augmented": total_augmented})
            current_batch = []
            shard_idx += 1

    # Save remaining samples
    if current_batch:
        shard_path = temp_dir / f"shard_{shard_idx:05d}"
        shard_dataset = Dataset.from_list(current_batch)
        shard_dataset.save_to_disk(str(shard_path))
        shard_paths.append(str(shard_path))
        total_augmented += len(current_batch)
        shard_idx += 1

    print(f"\nSaved {shard_idx} shards to {temp_dir}")
    print(f"Total augmented samples: {total_augmented}")
    print(f"Failed samples: {failed_count}")
    print(f"Augmentation factor: {total_augmented / total_samples:.2f}x")

    # Concatenate all shards into final dataset
    print(f"\nConcatenating {len(shard_paths)} shards...")
    all_shards = []
    for shard_path in tqdm(shard_paths, desc="Loading shards"):
        shard = load_from_disk(shard_path)
        all_shards.append(shard)

    print("Concatenating datasets...")
    final_dataset = concatenate_datasets(all_shards)

    print(f"Saving final dataset to {output_path}...")
    final_dataset.save_to_disk(output_path)

    # Clean up temp shards (with retry for race conditions)
    print("Cleaning up temporary shards...")
    import time
    for attempt in range(3):
        try:
            shutil.rmtree(temp_dir)
            break
        except OSError as e:
            if attempt < 2:
                print(f"Cleanup attempt {attempt+1} failed, retrying in 2s...")
                time.sleep(2)
            else:
                print(f"Warning: Could not fully clean up temp directory: {e}")
                print(f"You can manually delete: {temp_dir}")

    print(f"\nDone! Final dataset saved to {output_path}")
    print(f"  Total samples: {len(final_dataset)}")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute SAFE augmentations for data augmentation"
    )
    parser.add_argument("--input_path", required=True, help="Input HF dataset path")
    parser.add_argument("--output_path", required=True, help="Output HF dataset path")
    parser.add_argument("--num_versions", type=int, default=5,
                        help="Number of SAFE versions per SMILES (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_proc", type=int, default=4,
                        help="Number of processes for loading dataset (default: 4)")
    parser.add_argument("--batch_size", type=int, default=10000,
                        help="Samples per batch before saving to disk (default: 10000)")
    args = parser.parse_args()

    augment_dataset(
        args.input_path,
        args.output_path,
        args.num_versions,
        args.seed,
        args.num_proc,
        args.batch_size,
    )


if __name__ == "__main__":
    main()
