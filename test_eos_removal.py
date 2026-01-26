#!/usr/bin/env python3
"""
测试EOS token移除的影响
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from data_loader.molecule_tokenizer import MoleculeTokenizer

# 加载tokenizer
tokenizer_path = Path("outputs_xdock/12_23_25_300M_xdock_lr6e-6_KLnormclamp5_ufa_gate1_info0.10.2_xattn_new_b128a1/checkpoint-34350/full_model/tokenizer.json")
mol_tokenizer = MoleculeTokenizer.load(str(tokenizer_path))
tokenizer = mol_tokenizer.get_pretrained()
print(f"✓ Loaded tokenizer from {tokenizer_path}")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"BOS token ID: {tokenizer.bos_token_id}")
print(f"EOS token ID: {tokenizer.eos_token_id}")
print(f"PAD token ID: {tokenizer.pad_token_id}")

# 测试SAFE字符串
test_safe = "C14CCC2C5CC3NNC(=O)N3C2C1"
print(f"\n{'='*60}")
print(f"测试SAFE字符串: {test_safe}")
print(f"{'='*60}")

# 不加dot
print("\n1. 不加trailing dot:")
encoding = tokenizer(test_safe, return_tensors="pt", add_special_tokens=True)
ids = encoding["input_ids"][0]
print(f"  Token IDs: {ids.tolist()}")
tokens = [tokenizer.decode([t]) for t in ids]
print(f"  Tokens: {tokens}")
print(f"  最后一个是EOS? {ids[-1] == tokenizer.eos_token_id}")

# 加dot
test_safe_with_dot = test_safe + "."
print(f"\n2. 加trailing dot: {test_safe_with_dot}")
encoding = tokenizer(test_safe_with_dot, return_tensors="pt", add_special_tokens=True)
ids = encoding["input_ids"][0]
print(f"  Token IDs: {ids.tolist()}")
tokens = [tokenizer.decode([t]) for t in ids]
print(f"  Tokens: {tokens}")
print(f"  最后一个是EOS? {ids[-1] == tokenizer.eos_token_id}")

# 移除EOS后
print(f"\n3. 移除最后的EOS token (像SAFE那样):")
if ids[-1] == tokenizer.eos_token_id:
    ids_no_eos = ids[:-1]
    print(f"  Token IDs: {ids_no_eos.tolist()}")
    tokens_no_eos = [tokenizer.decode([t]) for t in ids_no_eos]
    print(f"  Tokens: {tokens_no_eos}")
    print(f"  最后一个是EOS? {ids_no_eos[-1] == tokenizer.eos_token_id if len(ids_no_eos) > 0 else 'N/A'}")
    
print("\n" + "="*60)
print("结论:")
print("  - Tokenizer的add_special_tokens=True会自动添加BOS和EOS")
print("  - 如果prefix包含EOS，模型会认为序列已结束，不继续生成")
print("  - SAFE的做法：移除tokenization后的EOS token")
print("  - 添加trailing dot (.) 提示模型开始生成新的fragment")
print("="*60)
