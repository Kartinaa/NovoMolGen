#!/usr/bin/env python3
"""
检查脚本：验证所有新添加的模块是否都被正确保存

用法:
    python scripts/check_saved_modules.py [checkpoint_path]
    
示例:
    python scripts/check_saved_modules.py outputs/11_13_25_SAFEGen_lr_KLnorm_unfreeze/checkpoint-1400
"""
import torch
import sys
from pathlib import Path


def check_saved_modules(checkpoint_path: str):
    """全面检查所有新添加的模块是否被保存"""
    
    model_path = Path(checkpoint_path) / "full_model" / "pytorch_model.bin"
    
    if not model_path.exists():
        print(f"❌ 错误：找不到模型文件 {model_path}")
        print(f"   请确认检查点路径是否正确")
        return False
    
    print("=" * 80)
    print("全面检查所有新添加的模块是否被保存")
    print("=" * 80)
    print(f"检查点路径: {checkpoint_path}")
    print(f"模型文件: {model_path}")
    
    try:
        state_dict = torch.load(str(model_path), map_location='cpu', weights_only=False)
        all_keys = list(state_dict.keys())
        
        print(f"\n总参数数量: {len(all_keys)}")
        
        # 1. 检查 encoder 模块
        print("\n" + "=" * 80)
        print("1. Encoder 模块检查")
        print("=" * 80)
        
        ligand_encoder_keys = [k for k in all_keys if 'ligand_encoder' in k]
        protein_encoder_keys = [k for k in all_keys if 'protein_encoder' in k]
        condition_fusion_keys = [k for k in all_keys if 'condition_fusion' in k]
        cond_proj_keys = [k for k in all_keys if 'cond_proj' in k]
        
        print(f"\n✅ ligand_encoder: {len(ligand_encoder_keys)} 个参数")
        if ligand_encoder_keys:
            print("   示例:")
            for k in ligand_encoder_keys[:5]:
                print(f"     - {k}")
            if len(ligand_encoder_keys) > 5:
                print(f"     ... 还有 {len(ligand_encoder_keys) - 5} 个参数")
        else:
            print("   ❌ 未找到！")
        
        print(f"\n✅ protein_encoder: {len(protein_encoder_keys)} 个参数")
        if protein_encoder_keys:
            print("   示例:")
            for k in protein_encoder_keys[:5]:
                print(f"     - {k}")
            if len(protein_encoder_keys) > 5:
                print(f"     ... 还有 {len(protein_encoder_keys) - 5} 个参数")
        else:
            print("   ❌ 未找到！")
        
        print(f"\n✅ condition_fusion: {len(condition_fusion_keys)} 个参数")
        if condition_fusion_keys:
            print("   示例:")
            for k in condition_fusion_keys[:5]:
                print(f"     - {k}")
            if len(condition_fusion_keys) > 5:
                print(f"     ... 还有 {len(condition_fusion_keys) - 5} 个参数")
        else:
            print("   ❌ 未找到！")
        
        print(f"\n✅ cond_proj: {len(cond_proj_keys)} 个参数")
        if cond_proj_keys:
            for k in cond_proj_keys:
                print(f"     - {k}")
        else:
            print("   ❌ 未找到！")
        
        # 2. 检查 cross-attention 模块
        print("\n" + "=" * 80)
        print("2. Cross-Attention 模块检查")
        print("=" * 80)
        
        cross_adapters_keys = [k for k in all_keys if 'cross_adapters' in k]
        gates_keys = [k for k in all_keys if 'gates' in k and 'cross' not in k]
        
        print(f"\n✅ cross_adapters: {len(cross_adapters_keys)} 个参数")
        if cross_adapters_keys:
            print("   按层分组:")
            layers = {}
            for k in cross_adapters_keys:
                parts = k.split('.')
                if len(parts) >= 3 and parts[1] == 'cross_adapters':
                    layer_num = parts[2]
                    if layer_num not in layers:
                        layers[layer_num] = []
                    layers[layer_num].append(k)
            
            for layer_num in sorted(layers.keys(), key=int):
                print(f"   层 {layer_num}: {len(layers[layer_num])} 个参数")
                for k in layers[layer_num][:3]:
                    print(f"     - {k}")
                if len(layers[layer_num]) > 3:
                    print(f"     ... 还有 {len(layers[layer_num]) - 3} 个参数")
        else:
            print("   ❌ 未找到！")
        
        print(f"\n✅ gates: {len(gates_keys)} 个参数")
        if gates_keys:
            print("   按层分组:")
            layers = {}
            for k in gates_keys:
                parts = k.split('.')
                if len(parts) >= 3 and parts[1] == 'gates':
                    layer_num = parts[2]
                    if layer_num not in layers:
                        layers[layer_num] = []
                    layers[layer_num].append(k)
            
            for layer_num in sorted(layers.keys(), key=int):
                print(f"   层 {layer_num}: {len(layers[layer_num])} 个参数")
                for k in layers[layer_num]:
                    print(f"     - {k}")
        else:
            print("   ❌ 未找到！")
        
        # 3. 统计总结
        print("\n" + "=" * 80)
        print("3. 保存情况总结")
        print("=" * 80)
        
        expected_modules = {
            'ligand_encoder': len(ligand_encoder_keys),
            'protein_encoder': len(protein_encoder_keys),
            'condition_fusion': len(condition_fusion_keys),
            'cond_proj': len(cond_proj_keys),
            'cross_adapters': len(cross_adapters_keys),
            'gates': len(gates_keys),
        }
        
        all_saved = True
        missing_modules = []
        for module_name, param_count in expected_modules.items():
            status = "✅" if param_count > 0 else "❌"
            print(f"{status} {module_name}: {param_count} 个参数")
            if param_count == 0:
                all_saved = False
                missing_modules.append(module_name)
        
        # 4. 检查是否有 modules_to_save 残留（不应该有）
        print("\n" + "=" * 80)
        print("4. 检查是否有 PEFT modules_to_save 残留")
        print("=" * 80)
        
        modules_to_save_residue = [k for k in all_keys if 'modules_to_save' in k]
        if modules_to_save_residue:
            print(f"⚠️  发现 {len(modules_to_save_residue)} 个残留的 modules_to_save 键（这不应该存在）:")
            for k in modules_to_save_residue[:10]:
                print(f"     - {k}")
            if len(modules_to_save_residue) > 10:
                print(f"     ... 还有 {len(modules_to_save_residue) - 10} 个")
        else:
            print("✅ 没有残留的 modules_to_save 键（这是正常的，说明已经合并）")
        
        # 5. 最终结论
        print("\n" + "=" * 80)
        if all_saved:
            print("✅ 所有新添加的模块都已正确保存！")
            print("=" * 80)
            return True
        else:
            print(f"❌ 警告：以下模块未找到: {', '.join(missing_modules)}")
            print("=" * 80)
            return False
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        checkpoint_path = "outputs/11_13_25_SAFEGen_lr_KLnorm_unfreeze/checkpoint-1400"
        print(f"使用默认路径: {checkpoint_path}")
        print("提示: 可以通过命令行参数指定检查点路径")
        print("用法: python scripts/check_saved_modules.py <checkpoint_path>")
        print()
    else:
        checkpoint_path = sys.argv[1]
    
    success = check_saved_modules(checkpoint_path)
    sys.exit(0 if success else 1)

