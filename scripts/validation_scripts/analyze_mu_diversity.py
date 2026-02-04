#!/usr/bin/env python3
"""
Analyze mu diversity across different ligands to diagnose ligand conditioning.

This script checks if the ligand encoder produces different mu vectors for different ligands.
If mu vectors are all very similar, it means the ligand encoder isn't differentiating ligands,
which would explain why generated molecules have similar Tanimoto to correct vs random ligands.

Usage:
    python scripts/validation_scripts/analyze_mu_diversity.py \
        --model_path outputs/your_model/checkpoint-XXXX/full_model \
        --validation_set finetune_data/crossdock_dataset/hf_dataset_crossdock/validation \
        --num_samples 100
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple, Optional

import torch
import numpy as np
import pyarrow as pa

# Add src to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("analyze_mu_diversity")
    logger.setLevel(getattr(logging, log_level.upper()))
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def load_validation_data(path: str, logger: logging.Logger) -> dict:
    """Load validation data from arrow file."""
    val_path = Path(path)

    if not val_path.exists():
        raise FileNotFoundError(f"Validation path does not exist: {path}")

    # Find arrow files
    arrow_files = list(val_path.glob("*.arrow"))
    if not arrow_files:
        raise FileNotFoundError(f"No arrow files found in: {path}")

    # Load using PyArrow streaming format (HF datasets format)
    logger.info(f"Loading data from: {arrow_files[0]}")
    with pa.memory_map(str(arrow_files[0]), 'r') as source:
        table = pa.ipc.open_stream(source).read_all()

    # Convert to dict
    data = {col: table.column(col).to_pylist() for col in table.column_names}
    logger.info(f"Loaded {table.num_rows} samples with columns: {table.column_names}")

    return data


def load_model(model_path: str, logger: logging.Logger):
    """Load model with automatic detection of model type."""
    model_path = Path(model_path)
    logger.info(f"Loading model from: {model_path}")

    # Try to detect model type from config
    config_path = model_path / "config.json"
    model_class = None
    config_class = None

    # Try different model imports
    try:
        from models.modeling_novomolgen_tanimoto import NovoMolGen, NovoMolGenConfig
        model_class = NovoMolGen
        config_class = NovoMolGenConfig
        logger.info("Using Tanimoto model")
    except ImportError:
        pass

    if model_class is None:
        try:
            from models.modeling_novomolgen_infonce_depot import NovoMolGen, NovoMolGenConfig
            model_class = NovoMolGen
            config_class = NovoMolGenConfig
            logger.info("Using InfoNCE model")
        except ImportError:
            pass

    if model_class is None:
        raise ImportError("Could not import any NovoMolGen model class")

    # Load config and model
    config = config_class.from_pretrained(str(model_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = model_class.from_pretrained(
        str(model_path),
        config=config,
        torch_dtype=model_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = model.to(device)

    if torch.cuda.is_available() and model_dtype == torch.bfloat16:
        model = model.to(torch.bfloat16)

    model.eval()
    logger.info(f"Model loaded on {device}")

    return model, device


def analyze_mu_diversity(
    model,
    device: torch.device,
    data: dict,
    num_samples: int,
    logger: logging.Logger,
) -> dict:
    """
    Analyze mu diversity across different ligands.

    Returns dict with analysis results.
    """
    if not hasattr(model, "ligand_encoder"):
        raise AttributeError("Model does not have ligand_encoder attribute")

    # Check required keys
    required_keys = ["ifp", "ligand_vec"]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Data missing required key: {key}")

    n_total = len(data["ifp"])
    num_samples = min(num_samples, n_total)
    logger.info(f"Analyzing mu for {num_samples} samples out of {n_total}")

    model_dtype = next(model.parameters()).dtype

    # Compute mu for all samples
    mu_vectors = []
    sigma_vectors = []

    model.eval()
    with torch.no_grad():
        # Process in batches
        batch_size = 64
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)

            ifp = torch.tensor(data["ifp"][start:end], dtype=torch.float32, device=device).to(model_dtype)
            ligand_vec = torch.tensor(data["ligand_vec"][start:end], dtype=torch.float32, device=device).to(model_dtype)

            mu, sigma, logvar = model.ligand_encoder(ifp, ligand_vec, return_logvar=True)

            mu_vectors.append(mu.detach().cpu())
            sigma_vectors.append(sigma.detach().cpu())

    mu_all = torch.cat(mu_vectors, dim=0)  # [num_samples, latent_dim]
    sigma_all = torch.cat(sigma_vectors, dim=0)  # [num_samples, latent_dim]

    latent_dim = mu_all.shape[1]
    logger.info(f"Collected mu vectors: shape = {mu_all.shape}")

    results = {"num_samples": num_samples, "latent_dim": latent_dim}

    # ========== 1. Per-dimension statistics ==========
    logger.info("\n" + "=" * 60)
    logger.info("1. Per-dimension statistics")
    logger.info("=" * 60)

    # Variance of mu across samples (for each dimension)
    mu_var_per_dim = mu_all.var(dim=0)  # [latent_dim]
    mu_mean_per_dim = mu_all.mean(dim=0)  # [latent_dim]

    results["mu_var_per_dim_mean"] = float(mu_var_per_dim.mean().item())
    results["mu_var_per_dim_min"] = float(mu_var_per_dim.min().item())
    results["mu_var_per_dim_max"] = float(mu_var_per_dim.max().item())
    results["mu_var_nonzero_dims"] = int((mu_var_per_dim > 1e-6).sum().item())

    logger.info(f"  mu variance across samples (per dim):")
    logger.info(f"    Mean: {results['mu_var_per_dim_mean']:.6f}")
    logger.info(f"    Min:  {results['mu_var_per_dim_min']:.6f}")
    logger.info(f"    Max:  {results['mu_var_per_dim_max']:.6f}")
    logger.info(f"    Dims with var > 1e-6: {results['mu_var_nonzero_dims']}/{latent_dim}")

    # Mean sigma (average uncertainty)
    sigma_mean = sigma_all.mean()
    results["sigma_mean"] = float(sigma_mean.item())
    logger.info(f"  Mean sigma (uncertainty): {results['sigma_mean']:.6f}")

    # ========== 2. Pairwise Euclidean distances ==========
    logger.info("\n" + "=" * 60)
    logger.info("2. Pairwise Euclidean distances between mu vectors")
    logger.info("=" * 60)

    # Compute pairwise distances
    diff = mu_all.unsqueeze(0) - mu_all.unsqueeze(1)  # [N, N, D]
    dist_matrix = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)  # [N, N]

    # Get upper triangular (exclude diagonal)
    triu_idx = torch.triu_indices(num_samples, num_samples, offset=1)
    pairwise_dists = dist_matrix[triu_idx[0], triu_idx[1]]

    results["dist_mean"] = float(pairwise_dists.mean().item())
    results["dist_std"] = float(pairwise_dists.std().item())
    results["dist_min"] = float(pairwise_dists.min().item())
    results["dist_max"] = float(pairwise_dists.max().item())
    results["dist_median"] = float(pairwise_dists.median().item())

    logger.info(f"  Mean distance: {results['dist_mean']:.4f}")
    logger.info(f"  Std distance:  {results['dist_std']:.4f}")
    logger.info(f"  Min distance:  {results['dist_min']:.4f}")
    logger.info(f"  Max distance:  {results['dist_max']:.4f}")
    logger.info(f"  Median:        {results['dist_median']:.4f}")

    # ========== 3. Pairwise cosine similarities ==========
    logger.info("\n" + "=" * 60)
    logger.info("3. Pairwise cosine similarities between mu vectors")
    logger.info("=" * 60)

    mu_normalized = mu_all / (mu_all.norm(dim=-1, keepdim=True) + 1e-8)
    cos_matrix = torch.mm(mu_normalized, mu_normalized.t())  # [N, N]
    pairwise_cos = cos_matrix[triu_idx[0], triu_idx[1]]

    results["cos_mean"] = float(pairwise_cos.mean().item())
    results["cos_std"] = float(pairwise_cos.std().item())
    results["cos_min"] = float(pairwise_cos.min().item())
    results["cos_max"] = float(pairwise_cos.max().item())
    results["cos_median"] = float(pairwise_cos.median().item())

    logger.info(f"  Mean cosine sim: {results['cos_mean']:.4f}")
    logger.info(f"  Std cosine sim:  {results['cos_std']:.4f}")
    logger.info(f"  Min cosine sim:  {results['cos_min']:.4f}")
    logger.info(f"  Max cosine sim:  {results['cos_max']:.4f}")
    logger.info(f"  Median:          {results['cos_median']:.4f}")

    # ========== 4. Compare with random baseline ==========
    logger.info("\n" + "=" * 60)
    logger.info("4. Comparison with random vectors")
    logger.info("=" * 60)

    # Generate random vectors with same mean and std as mu
    random_mu = torch.randn_like(mu_all) * mu_all.std() + mu_all.mean()
    random_normalized = random_mu / (random_mu.norm(dim=-1, keepdim=True) + 1e-8)
    random_cos_matrix = torch.mm(random_normalized, random_normalized.t())
    random_cos = random_cos_matrix[triu_idx[0], triu_idx[1]]

    results["random_cos_mean"] = float(random_cos.mean().item())
    logger.info(f"  Random vectors cosine sim mean: {results['random_cos_mean']:.4f}")
    logger.info(f"  Actual mu vectors cosine sim mean: {results['cos_mean']:.4f}")

    # ========== 5. Interpretation ==========
    logger.info("\n" + "=" * 60)
    logger.info("INTERPRETATION")
    logger.info("=" * 60)

    issues = []

    if results["mu_var_per_dim_mean"] < 0.01:
        issues.append("⚠️  Very low variance: mu vectors are nearly identical across ligands!")
        issues.append("   → The ligand encoder may not be learning meaningful representations.")

    if results["cos_mean"] > 0.95:
        issues.append("⚠️  Very high cosine similarity: mu vectors point in similar directions!")
        issues.append("   → Different ligands produce very similar conditions.")

    if results["dist_max"] < 1.0:
        issues.append("⚠️  Small distance range: mu vectors are all clustered together!")
        issues.append("   → The latent space isn't being utilized effectively.")

    if abs(results["cos_mean"] - results["random_cos_mean"]) < 0.1:
        issues.append("⚠️  mu similarity is close to random baseline!")
        issues.append("   → The encoder output may be essentially random/uninformative.")

    if issues:
        for issue in issues:
            logger.warning(issue)
        results["diagnosis"] = "POTENTIAL_ISSUE"
    else:
        logger.info("✓ mu vectors show reasonable diversity across different ligands.")
        logger.info(f"  - Variance per dim: {results['mu_var_per_dim_mean']:.4f} (should be > 0.01)")
        logger.info(f"  - Cosine sim range: [{results['cos_min']:.2f}, {results['cos_max']:.2f}]")
        logger.info(f"  - Distance range: [{results['dist_min']:.2f}, {results['dist_max']:.2f}]")
        results["diagnosis"] = "OK"

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze mu diversity across different ligands to diagnose conditioning."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to full_model directory",
    )
    parser.add_argument(
        "--validation_set",
        type=str,
        required=True,
        help="Path to validation set (HF dataset directory with arrow files)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to analyze (default: 100)",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional: Save results to JSON file",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()
    logger = setup_logging(args.log_level)

    logger.info("=" * 60)
    logger.info("Mu Diversity Analysis")
    logger.info("=" * 60)
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Validation set: {args.validation_set}")
    logger.info(f"Num samples: {args.num_samples}")

    try:
        # Load data
        data = load_validation_data(args.validation_set, logger)

        # Load model
        model, device = load_model(args.model_path, logger)

        # Analyze
        results = analyze_mu_diversity(
            model=model,
            device=device,
            data=data,
            num_samples=args.num_samples,
            logger=logger,
        )

        # Save results if requested
        if args.output_json:
            import json
            output_path = Path(args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"\nResults saved to: {args.output_json}")

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1

    logger.info("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
