#!/usr/bin/env python3
"""
检查 cross-attention gate 参数的大小，判断门控大概开到什么程度。

用法示例：
    python scripts/validation_scripts/inspect_gates.py \
        --model_path outputs/11_23_25_SAFEGen_lr_KLnormclaped0.1_unfreezeall_noloar_xattn67891011/checkpoint-33400/full_model
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple

import torch
import pyarrow as pa
from datasets import load_from_disk

# 把 src 加到路径，方便导入模型（与其它脚本保持一致）
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from models.modeling_novomolgen_tanimoto import NovoMolGen, NovoMolGenConfig  # type: ignore


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("inspect_gates")
    logger.setLevel(getattr(logging, log_level.upper()))
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def load_model(model_path: str, logger: logging.Logger) -> Tuple[NovoMolGen, torch.device]:
    """与 generate_from_full_model.py 相同风格的加载逻辑，只加载模型本身。"""
    model_path = Path(model_path)
    logger.info(f"Loading model from: {model_path}")

    # Load config
    config = NovoMolGenConfig.from_pretrained(str(model_path))
    logger.info(
        f"Model config: enable_cross_attn={config.enable_cross_attn}, "
        f"cross_layers={config.cross_layers}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = NovoMolGen.from_pretrained(
        str(model_path),
        config=config,
        torch_dtype=model_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = model.to(device)

    if torch.cuda.is_available() and model_dtype == torch.bfloat16:
        model = model.to(torch.bfloat16)
        param_dtypes = {p.dtype for p in model.parameters()}
        if torch.float32 in param_dtypes:
            logger.warning(
                f"Some parameters are still float32: {param_dtypes}. "
                "Converting all to bfloat16 again..."
            )
            model = model.to(torch.bfloat16)
        logger.info(f"Model converted to bfloat16. Parameter dtypes: {param_dtypes}")

    model.eval()
    final_dtype = next(model.parameters()).dtype
    logger.info(f"Model loaded on {device} with dtype {final_dtype}")
    return model, device


def inspect_gates(model: NovoMolGen, logger: logging.Logger) -> None:
    """
    打印 cross-attention gates 的统计信息：
      - 原始 gate 值的 mean/min/max
      - sigmoid(gate) 之後的 mean/min/max（真正用於加權的係數）
    """
    has_gates = False
    all_sigmoid_vals = []

    logger.info("\n" + "=" * 60)
    logger.info("Inspecting transformer.gates parameters")
    logger.info("=" * 60)

    for name, p in model.named_parameters():
        if "transformer.gates" not in name:
            continue
        has_gates = True

        tensor = p.detach().cpu()
        gate_mean = float(tensor.mean().item())
        gate_min = float(tensor.min().item())
        gate_max = float(tensor.max().item())

        sig = torch.sigmoid(tensor)
        sig_mean = float(sig.mean().item())
        sig_min = float(sig.min().item())
        sig_max = float(sig.max().item())

        all_sigmoid_vals.append(sig.view(-1))

        logger.info(f"Gate parameter: {name}")
        logger.info(f"  raw gate   : mean={gate_mean:.4f}, min={gate_min:.4f}, max={gate_max:.4f}")
        logger.info(f"  sigmoid(g) : mean={sig_mean:.4f}, min={sig_min:.4f}, max={sig_max:.4f}")

    if not has_gates:
        logger.warning("No parameters with 'transformer.gates' in their name were found.")
        logger.warning("This usually means enable_cross_attn=False 或模型没有注入 cross-attn。")
        return

    # 全局統計
    if all_sigmoid_vals:
        all_sigmoid_cat = torch.cat(all_sigmoid_vals, dim=0)
        g_mean = float(all_sigmoid_cat.mean().item())
        g_min = float(all_sigmoid_cat.min().item())
        g_max = float(all_sigmoid_cat.max().item())

        logger.info("\n" + "=" * 60)
        logger.info("Global sigmoid(gate) statistics over all layers")
        logger.info("=" * 60)
        logger.info(f"  mean={g_mean:.4f}, min={g_min:.4f}, max={g_max:.4f}")
        logger.info(
            "說明：sigmoid(gate) 越接近 0，cross-attn 貢獻越小；越接近 1，cross-attn 貢獻越大。"
        )


def inspect_mu(
    model: NovoMolGen,
    device: torch.device,
    validation_path: str,
    logger: logging.Logger,
    batch_size: int = 64,
    max_batches: int = 20,
) -> None:
    """
    在 validation set 上檢查 ligand_encoder 輸出的 mu 統計信息。
    
    只採樣前 max_batches * batch_size 個樣本，避免太慢。
    """
    if not hasattr(model, "ligand_encoder"):
        logger.warning("Model does not have attribute 'ligand_encoder', skipping mu inspection.")
        return

    val_path = Path(validation_path)
    if not val_path.exists():
        logger.error(f"Validation set path does not exist: {validation_path}")
        return

    logger.info("\n" + "=" * 60)
    logger.info("Inspecting ligand_encoder mu on validation set")
    logger.info("=" * 60)
    logger.info(f"Validation set path: {validation_path}")

    # Try to load dataset - first with PyArrow (for problematic datasets), then with load_from_disk
    data = None
    try:
        # Try PyArrow streaming format first (handles HF datasets metadata issues)
        arrow_files = list(val_path.glob("*.arrow"))
        if arrow_files:
            with pa.memory_map(str(arrow_files[0]), 'r') as source:
                table = pa.ipc.open_stream(source).read_all()
            data = {col: table.column(col).to_pylist() for col in table.column_names}
            n = table.num_rows
            logger.info(f"Loaded validation set via PyArrow: {n} samples")
    except Exception as e:
        logger.debug(f"PyArrow loading failed: {e}, trying load_from_disk...")

    if data is None:
        try:
            ds = load_from_disk(str(val_path))
            n = len(ds)
            # Convert to dict format for consistent handling
            data = {key: [ds[i][key] for i in range(n)] for key in ds[0].keys()}
            logger.info(f"Loaded validation set via load_from_disk: {n} samples")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return

    if n == 0:
        logger.warning("Validation set is empty, skipping mu inspection.")
        return

    required_keys = ["ifp", "ligand_vec"]
    if not all(k in data for k in required_keys):
        logger.error(
            f"Validation data does not contain required keys {required_keys}. "
            f"Available keys: {list(data.keys())}"
        )
        return

    # 準備統計量
    model_dtype = next(model.parameters()).dtype
    total_samples = min(n, max_batches * batch_size)
    logger.info(
        f"Will inspect mu on up to {total_samples} samples "
        f"(batch_size={batch_size}, max_batches={max_batches})"
    )

    sum_mu = 0.0
    sum_sq_mu = 0.0
    count = 0
    global_min = None
    global_max = None
    latent_dim = None

    model.eval()
    with torch.no_grad():
        for start in range(0, total_samples, batch_size):
            end = min(start + batch_size, total_samples)
            # Use data dict directly instead of ds.select()
            ifp = torch.tensor(data["ifp"][start:end], dtype=torch.float32, device=device).to(model_dtype)
            ligand_vec = torch.tensor(data["ligand_vec"][start:end], dtype=torch.float32, device=device).to(model_dtype)

            mu, sigma, logvar = model.ligand_encoder(ifp, ligand_vec, return_logvar=True)
            mu_cpu = mu.detach().cpu()

            if latent_dim is None:
                latent_dim = mu_cpu.shape[1]

            sum_mu += float(mu_cpu.sum().item())
            sum_sq_mu += float((mu_cpu ** 2).sum().item())
            numel = mu_cpu.numel()
            count += numel

            batch_min = float(mu_cpu.min().item())
            batch_max = float(mu_cpu.max().item())
            if global_min is None or batch_min < global_min:
                global_min = batch_min
            if global_max is None or batch_max > global_max:
                global_max = batch_max

    if count == 0 or latent_dim is None:
        logger.warning("No mu values collected, something went wrong.")
        return

    mean_mu = sum_mu / count
    mean_sq = sum_sq_mu / count
    var_mu = max(mean_sq - mean_mu ** 2, 0.0)
    std_mu = var_mu ** 0.5

    logger.info(f"Latent dimension (mu size): {latent_dim}")
    logger.info(f"Number of mu elements aggregated: {count}")
    logger.info(f"Global mu mean: {mean_mu:.6f}")
    logger.info(f"Global mu std : {std_mu:.6f}")
    logger.info(f"Global mu min : {global_min:.6f}")
    logger.info(f"Global mu max : {global_max:.6f}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect cross-attention gate parameters of a trained NovoMolGen model."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to full_model directory (e.g. outputs/.../checkpoint-XXXX/full_model)",
    )
    parser.add_argument(
        "--validation_set",
        type=str,
        default=None,
        help="Optional: Path to validation set (HF dataset) to inspect ligand_encoder mu statistics",
    )
    parser.add_argument(
        "--mu_batches",
        type=int,
        default=20,
        help="Maximum number of batches to use when computing mu statistics (default: 20)",
    )
    parser.add_argument(
        "--mu_batch_size",
        type=int,
        default=64,
        help="Batch size for computing mu statistics (default: 64)",
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
    logger.info("Cross-attention Gate Inspector")
    logger.info("=" * 60)
    logger.info(f"Model path: {args.model_path}")

    model, device = load_model(args.model_path, logger)
    inspect_gates(model, logger)

    # 如果提供了 validation_set，額外檢查 ligand_encoder 的 mu
    if args.validation_set is not None:
        inspect_mu(
            model=model,
            device=device,
            validation_path=args.validation_set,
            logger=logger,
            batch_size=args.mu_batch_size,
            max_batches=args.mu_batches,
        )

    logger.info("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())


