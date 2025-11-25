#!/usr/bin/env python3
"""
计算训练数据集中ligand_vector的每个维度的均值和标准差。

使用Welford在线算法进行数值稳定的增量计算，支持大数据集的分批处理。
支持的文件格式：.pt (PyTorch tensor), .npy (NumPy array), .parquet (Pandas DataFrame)
"""

import argparse
import sys
from pathlib import Path
from typing import Union, Tuple, Optional
import warnings

import numpy as np
import pandas as pd
import torch


def detect_file_format(file_path: Path) -> str:
    """检测文件格式。
    
    Args:
        file_path: 文件路径
        
    Returns:
        文件格式字符串: 'pt', 'npy', 'parquet', 或 'unknown'
    """
    suffix = file_path.suffix.lower()
    if suffix == '.pt':
        return 'pt'
    elif suffix == '.npy':
        return 'npy'
    elif suffix == '.parquet':
        return 'parquet'
    else:
        return 'unknown'


def load_ligand_vectors_pt(file_path: Path) -> torch.Tensor:
    """从.pt文件加载ligand向量。
    
    Args:
        file_path: .pt文件路径
        
    Returns:
        torch.Tensor: shape [N, D]的tensor
    """
    data = torch.load(file_path, map_location='cpu')
    
    # 如果保存的是字典，尝试提取ligand_vec或ligand_vector
    if isinstance(data, dict):
        if 'ligand_vec' in data:
            vectors = data['ligand_vec']
        elif 'ligand_vector' in data:
            vectors = data['ligand_vector']
        elif 'vectors' in data:
            vectors = data['vectors']
        else:
            # 尝试使用第一个值
            keys = list(data.keys())
            if len(keys) > 0:
                print(f"警告: 未找到'ligand_vec'或'ligand_vector'键，使用'{keys[0]}'")
                vectors = data[keys[0]]
            else:
                raise ValueError("字典中没有找到ligand向量数据")
    else:
        vectors = data
    
    # 确保是tensor
    if not isinstance(vectors, torch.Tensor):
        vectors = torch.tensor(vectors, dtype=torch.float32)
    
    # 确保是2D
    if vectors.dim() == 1:
        vectors = vectors.unsqueeze(0)
    elif vectors.dim() > 2:
        raise ValueError(f"期望2D tensor [N, D]，但得到{vectors.dim()}D tensor")
    
    return vectors.float()


def load_ligand_vectors_npy(file_path: Path) -> torch.Tensor:
    """从.npy文件加载ligand向量。
    
    Args:
        file_path: .npy文件路径
        
    Returns:
        torch.Tensor: shape [N, D]的tensor
    """
    array = np.load(file_path)
    
    # 确保是2D
    if array.ndim == 1:
        array = array.reshape(1, -1)
    elif array.ndim > 2:
        raise ValueError(f"期望2D array [N, D]，但得到{array.ndim}D array")
    
    return torch.from_numpy(array).float()


def load_ligand_vectors_parquet(file_path: Path, column_name: str = 'ligand_vec') -> torch.Tensor:
    """从.parquet文件加载ligand向量。
    
    Args:
        file_path: .parquet文件路径
        column_name: 包含ligand向量的列名，默认为'ligand_vec'
        
    Returns:
        torch.Tensor: shape [N, D]的tensor
    """
    df = pd.read_parquet(file_path)
    
    # 检查列是否存在
    if column_name not in df.columns:
        # 尝试其他可能的列名
        possible_names = ['ligand_vector', 'ligand_vec', 'mol_vec', 'molecule_vec']
        found = False
        for name in possible_names:
            if name in df.columns:
                column_name = name
                found = True
                print(f"使用列名: {column_name}")
                break
        
        if not found:
            raise ValueError(
                f"未找到ligand向量列。可用列: {list(df.columns)}\n"
                f"请使用--column_name指定正确的列名"
            )
    
    # 提取向量
    vectors_list = df[column_name].tolist()
    
    # 过滤None值
    valid_vectors = [v for v in vectors_list if v is not None]
    if len(valid_vectors) < len(vectors_list):
        print(f"警告: 发现{len(vectors_list) - len(valid_vectors)}个None值，已过滤")
    
    if len(valid_vectors) == 0:
        raise ValueError("没有有效的ligand向量")
    
    # 转换为tensor
    # 处理numpy数组
    if isinstance(valid_vectors[0], np.ndarray):
        vectors = np.stack(valid_vectors)
    # 处理torch tensor
    elif isinstance(valid_vectors[0], torch.Tensor):
        vectors = torch.stack(valid_vectors)
    # 处理列表
    elif isinstance(valid_vectors[0], (list, tuple)):
        vectors = np.array(valid_vectors)
    else:
        raise TypeError(f"不支持的数据类型: {type(valid_vectors[0])}")
    
    # 转换为torch tensor
    if isinstance(vectors, np.ndarray):
        vectors = torch.from_numpy(vectors)
    
    # 确保是2D
    if vectors.dim() == 1:
        vectors = vectors.unsqueeze(0)
    elif vectors.dim() > 2:
        raise ValueError(f"期望2D tensor [N, D]，但得到{vectors.dim()}D tensor")
    
    return vectors.float()


def load_ligand_vectors(
    input_path: Union[str, Path],
    column_name: Optional[str] = None
) -> torch.Tensor:
    """加载ligand向量，自动检测文件格式。
    
    Args:
        input_path: 输入文件路径
        column_name: 对于parquet文件，指定列名（可选）
        
    Returns:
        torch.Tensor: shape [N, D]的tensor
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"文件不存在: {input_path}")
    
    file_format = detect_file_format(input_path)
    
    print(f"检测到文件格式: {file_format}")
    print(f"加载文件: {input_path}")
    
    if file_format == 'pt':
        vectors = load_ligand_vectors_pt(input_path)
    elif file_format == 'npy':
        vectors = load_ligand_vectors_npy(input_path)
    elif file_format == 'parquet':
        col_name = column_name if column_name else 'ligand_vec'
        vectors = load_ligand_vectors_parquet(input_path, col_name)
    else:
        raise ValueError(
            f"不支持的文件格式: {file_format}\n"
            f"支持格式: .pt, .npy, .parquet"
        )
    
    print(f"加载完成: shape {vectors.shape}")
    return vectors


def compute_stats_welford(
    vectors: torch.Tensor,
    batch_size: int = 10000
) -> Tuple[torch.Tensor, torch.Tensor]:
    """使用Welford在线算法计算每个维度的均值和标准差。
    
    使用批处理版本的Welford算法，通过合并两个统计量来实现。
    算法参考: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    
    Args:
        vectors: shape [N, D]的tensor
        batch_size: 批处理大小
        
    Returns:
        mu: shape [D]的均值tensor
        sigma: shape [D]的标准差tensor
    """
    N, D = vectors.shape
    
    # 初始化Welford算法变量
    count = 0
    mean = None  # [D]
    M2 = None    # [D] - 用于计算方差的中间变量 (sum of squared differences from mean)
    
    print(f"使用Welford算法计算统计量 (N={N}, D={D}, batch_size={batch_size})...")
    
    # 分批处理
    num_batches = (N + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, N)
        batch = vectors[start_idx:end_idx]  # [batch_size, D]
        batch_size_actual = batch.shape[0]
        
        if i % max(1, num_batches // 10) == 0 or i == num_batches - 1:
            print(f"  处理批次 {i+1}/{num_batches} (样本 {start_idx+1}-{end_idx}/{N})")
        
        # 计算当前批次的统计量
        batch_mean = batch.mean(dim=0)  # [D]
        batch_M2 = ((batch - batch_mean.unsqueeze(0)) ** 2).sum(dim=0)  # [D] - sum of squared differences
        
        # 合并统计量
        if mean is None:
            # 第一个批次：直接使用
            mean = batch_mean
            M2 = batch_M2
            count = batch_size_actual
        else:
            # 合并两个统计量: 现有统计量 + 新批次统计量
            # 使用并行算法合并公式
            count_old = count
            count += batch_size_actual
            
            # 计算均值差
            delta = batch_mean - mean  # [D]
            
            # 更新均值
            mean = mean + delta * batch_size_actual / count  # [D]
            
            # 更新M2 (合并两个统计量的M2)
            # M2_combined = M2_old + M2_new + delta^2 * (n_old * n_new / n_total)
            M2 = M2 + batch_M2 + delta ** 2 * (count_old * batch_size_actual / count)  # [D]
    
    # 计算最终方差和标准差
    variance = M2 / count  # [D] - 总体方差 (population variance)
    sigma = torch.sqrt(variance)  # [D]
    
    return mean, sigma


def print_summary(vectors: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor):
    """打印统计摘要。
    
    Args:
        vectors: 原始向量 [N, D]
        mu: 均值 [D]
        sigma: 标准差 [D]
    """
    N, D = vectors.shape
    
    print("\n" + "="*60)
    print("统计摘要")
    print("="*60)
    print(f"数据集大小: ({N}, {D})")
    print(f"均值统计:")
    print(f"  - 均值向量的平均值: {mu.mean().item():.6f}")
    print(f"  - 均值向量的最小值: {mu.min().item():.6f}")
    print(f"  - 均值向量的最大值: {mu.max().item():.6f}")
    print(f"标准差统计:")
    print(f"  - 标准差向量的平均值: {sigma.mean().item():.6f}")
    print(f"  - 标准差向量的最小值: {sigma.min().item():.6f}")
    print(f"  - 标准差向量的最大值: {sigma.max().item():.6f}")
    print(f"\n前5个维度的统计量:")
    print(f"{'维度':<8} {'均值 (μ)':<15} {'标准差 (σ)':<15}")
    print("-" * 40)
    for j in range(min(5, D)):
        print(f"{j:<8} {mu[j].item():<15.6f} {sigma[j].item():<15.6f}")
    if D > 5:
        print(f"... (共 {D} 个维度)")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="计算训练数据集中ligand_vector的每个维度的均值和标准差",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 从.pt文件计算
  python compute_ligand_stats.py --input_path ./data/train_ligand_vectors.pt

  # 从.parquet文件计算，指定列名
  python compute_ligand_stats.py --input_path ./data/train.parquet --column_name ligand_vec

  # 自定义批大小和输出路径
  python compute_ligand_stats.py \\
      --input_path ./data/train_ligand_vectors.pt \\
      --batch_size 5000 \\
      --output_path ./ligand_stats.pt
        """
    )
    
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="训练ligand向量文件路径 (.pt, .npy, 或 .parquet)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10000,
        help="批处理大小（默认: 10000）"
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        default="ligand_stats.pt",
        help="输出文件路径（默认: ligand_stats.pt）"
    )
    
    parser.add_argument(
        "--column_name",
        type=str,
        default=None,
        help="对于parquet文件，指定包含ligand向量的列名（默认: 'ligand_vec'）"
    )
    
    args = parser.parse_args()
    
    # 加载数据
    try:
        vectors = load_ligand_vectors(args.input_path, args.column_name)
    except Exception as e:
        print(f"错误: 加载数据失败: {e}", file=sys.stderr)
        sys.exit(1)
    
    # 计算统计量
    try:
        mu, sigma = compute_stats_welford(vectors, args.batch_size)
    except Exception as e:
        print(f"错误: 计算统计量失败: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 打印摘要
    print_summary(vectors, mu, sigma)
    
    # 保存结果
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        torch.save({
            "mu": mu,
            "sigma": sigma,
            "shape": vectors.shape,
            "num_samples": vectors.shape[0],
            "num_dims": vectors.shape[1],
        }, output_path)
        print(f"\n✓ 统计量已保存到: {output_path}")
        print(f"  包含键: mu, sigma, shape, num_samples, num_dims")
    except Exception as e:
        print(f"错误: 保存文件失败: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

