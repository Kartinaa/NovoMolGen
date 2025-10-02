#!/usr/bin/env python3
"""
Quick test script for cross-attention functionality
"""

import torch
from src.models.modeling_novomolgen import NovoMolGen, NovoMolGenConfig

def quick_test():
    """Quick test of cross-attention functionality"""
    print("🚀 Quick Cross-Attention Test")
    print("=" * 50)
    
    # 1. Create configuration
    config = NovoMolGenConfig(
    hidden_size=256,
    num_attention_heads=8,
    num_hidden_layers=6,
    vocab_size=1000,
    enable_cross_attn=True,
    cross_layers=[3, 6, 9],  # Inject at layers 3, 6, 9
    use_flash_attn=False,    # Set to True for GPU
    fused_bias_fc=False,
    fused_mlp=False,
    fused_dropout_add_ln=False
    )
    
    print(f"✅ Configuration created")
    print(f"   - Cross-attention layers: {config.cross_layers}")
    print(f"   - Hidden size: {config.hidden_size}")
    
    # 2. Create model
    model = NovoMolGen(config, mol_type="SMILES")
    print(f"✅ Model created")
    
    # 3. Check cross-attention setup
    if hasattr(model.transformer, 'cross_adapters'):
        print(f"✅ Cross-attention enabled")
        print(f"   - Adapter layers: {list(model.transformer.cross_adapters.keys())}")
        gate_values = [torch.sigmoid(gate).item() for gate in model.transformer.gates.values()]
        print(f"   - Gate values: {gate_values}")
    else:
        print(f"❌ Cross-attention not enabled")
        return
    
    # 4. Create test data
    batch_size = 2
    seq_len = 8
    cond_len = 4
    
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    cond_tokens = torch.randn(batch_size, cond_len, config.hidden_size)
    cond_mask = torch.zeros(batch_size, cond_len, dtype=torch.bool)
    
    print(f"✅ Test data created")
    print(f"   - input_ids: {input_ids.shape}")
    print(f"   - cond_tokens: {cond_tokens.shape}")
    
    # 5. Test forward pass
    print(f"\n🧪 Testing forward pass...")
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            cond_tokens=cond_tokens,
            cond_attention_mask=cond_mask,
            return_dict=True
        )
        print(f"✅ Forward pass successful")
        print(f"   - Output logits: {outputs.logits.shape}")
    
    # 6. Test generation
    print(f"\n🧪 Testing generation...")
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids[:, :3],  # 使用前3个token
            cond_tokens=cond_tokens,
            cond_attention_mask=cond_mask,
            max_new_tokens=2,
            do_sample=False,
            pad_token_id=0
        )
        print(f"✅ Generation successful")
        print(f"   - Generated shape: {generated.shape}")
        print(f"   - Generated tokens: {generated[0].tolist()}")
    
    # 7. Test without conditions
    print(f"\n🧪 Testing without conditions...")
    with torch.no_grad():
        outputs_no_cond = model(input_ids=input_ids, return_dict=True)
        print(f"✅ Forward pass without conditions successful")
        print(f"   - Output logits: {outputs_no_cond.logits.shape}")
    
    print(f"\n🎉 All tests passed!")
    print(f"=" * 50)
    
    return model

if __name__ == "__main__":
    model = quick_test()
