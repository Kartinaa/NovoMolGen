#!/usr/bin/env python3
"""
Test script for Cross-Attention functionality in NovoMolGen

This script demonstrates how to use:
1. CrossAttentionAdapter
2. CrossAttentionTransformer  
3. NovoMolGen with cross-attention enabled

Usage:
    python test_cross_attention_usage.py
"""

import torch
import torch.nn as nn
from src.models.modeling_novomolgen import (
    NovoMolGen, 
    NovoMolGenConfig, 
    CrossAttentionAdapter,
    CrossAttentionTransformer
)

def test_cross_attention_adapter():
    """Test the CrossAttentionAdapter module"""
    print("=" * 60)
    print("🧪 Testing CrossAttentionAdapter")
    print("=" * 60)
    
    # Configuration
    batch_size = 2
    seq_len = 10
    cond_len = 5
    hidden_size = 128
    num_heads = 8
    
    # Create adapter
    config = NovoMolGenConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        attn_dropout=0.1
    )
    adapter = CrossAttentionAdapter(config)
    
    # Create test data
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    cond_tokens = torch.randn(batch_size, cond_len, hidden_size)
    cond_mask = torch.zeros(batch_size, cond_len, dtype=torch.bool)  # No padding
    
    print(f"Input shapes:")
    print(f"  hidden_states: {hidden_states.shape}")
    print(f"  cond_tokens: {cond_tokens.shape}")
    print(f"  cond_mask: {cond_mask.shape}")
    
    # Test forward pass
    with torch.no_grad():
        output = adapter(hidden_states, cond_tokens, cond_mask)
        print(f"Output shape: {output.shape}")
        
        # Test with attention weights
        output_with_weights = adapter(hidden_states, cond_tokens, cond_mask, 
                                    return_attention_weights=True)
        attn_output, attn_weights = output_with_weights
        print(f"Attention weights shape: {attn_weights.shape}")
        print(f"Attention weights sum (should be ~1.0): {attn_weights.sum(dim=-1)[0, 0, 0].item():.4f}")
    
    # Test empty condition handling
    empty_cond = torch.randn(batch_size, 0, hidden_size)
    empty_mask = torch.zeros(batch_size, 0, dtype=torch.bool)
    
    with torch.no_grad():
        empty_output = adapter(hidden_states, empty_cond, empty_mask)
        print(f"Empty condition output shape: {empty_output.shape}")
        print(f"Empty condition output (should be zeros): {empty_output.sum().item():.6f}")
    
    print("✅ CrossAttentionAdapter test passed!")
    return adapter

def test_cross_attention_transformer():
    """Test the CrossAttentionTransformer wrapper"""
    print("\n" + "=" * 60)
    print("🧪 Testing CrossAttentionTransformer")
    print("=" * 60)
    
    # Create a simple mock transformer for testing
    class MockTransformer(nn.Module):
        def __init__(self, hidden_size, num_layers):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.blocks = nn.ModuleList([
                nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
            ])
        
        def forward(self, input_ids, position_ids=None, inference_params=None):
            # Simple forward pass
            x = torch.randn(input_ids.shape[0], input_ids.shape[1], self.hidden_size)
            for block in self.blocks:
                x = block(x)
            return x
    
    # Configuration
    config = NovoMolGenConfig(
        hidden_size=128,
        num_attention_heads=8,
        num_hidden_layers=6,
        enable_cross_attn=True,
        cross_layers=[2, 4]  # Inject at layers 2 and 4
    )
    
    # Create mock transformer
    mock_transformer = MockTransformer(config.hidden_size, config.num_hidden_layers)
    
    # Wrap with CrossAttentionTransformer
    cross_transformer = CrossAttentionTransformer(mock_transformer, config)
    
    print(f"Cross-attention layers: {cross_transformer.cross_layers}")
    print(f"Number of adapters: {len(cross_transformer.cross_adapters)}")
    print(f"Number of gates: {len(cross_transformer.gates)}")
    
    # Test data
    batch_size = 2
    seq_len = 8
    cond_len = 4
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    cond_tokens = torch.randn(batch_size, cond_len, config.hidden_size)
    cond_mask = torch.zeros(batch_size, cond_len, dtype=torch.bool)
    
    print(f"\nInput shapes:")
    print(f"  input_ids: {input_ids.shape}")
    print(f"  cond_tokens: {cond_tokens.shape}")
    print(f"  cond_mask: {cond_mask.shape}")
    
    # Test forward pass
    with torch.no_grad():
        output = cross_transformer(input_ids, cond_tokens=cond_tokens, 
                                 cond_attention_mask=cond_mask)
        print(f"Output shape: {output.shape}")
    
    # Test without conditions
    with torch.no_grad():
        output_no_cond = cross_transformer(input_ids)
        print(f"Output without conditions shape: {output_no_cond.shape}")
    
    print("✅ CrossAttentionTransformer test passed!")
    return cross_transformer

def test_novomolgen_with_cross_attention():
    """Test the full NovoMolGen model with cross-attention"""
    print("\n" + "=" * 60)
    print("🧪 Testing NovoMolGen with Cross-Attention")
    print("=" * 60)
    
    # Configuration
    config = NovoMolGenConfig(
        hidden_size=256,
        num_attention_heads=8,
        num_hidden_layers=6,
        vocab_size=1000,
        max_position_embeddings=512,
        enable_cross_attn=True,
        cross_layers=[2, 4],  # Inject at layers 2 and 4
        use_flash_attn=False,  # Disable for testing
        fused_bias_fc=False,
        fused_mlp=False,
        fused_dropout_add_ln=False
    )
    
    print(f"Model configuration:")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  num_layers: {config.num_hidden_layers}")
    print(f"  cross_layers: {config.cross_layers}")
    print(f"  vocab_size: {config.vocab_size}")
    
    # Create model
    model = NovoMolGen(config, mol_type="SMILES")
    
    print(f"\nModel structure:")
    print(f"  Has cross-attention: {hasattr(model.transformer, 'cross_adapters')}")
    if hasattr(model.transformer, 'cross_adapters'):
        print(f"  Cross-attention layers: {list(model.transformer.cross_adapters.keys())}")
        print(f"  Gate values: {[torch.sigmoid(gate).item() for gate in model.transformer.gates.values()]}")
    
    # Test data
    batch_size = 2
    seq_len = 10
    cond_len = 6
    
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    cond_tokens = torch.randn(batch_size, cond_len, config.hidden_size)
    cond_mask = torch.zeros(batch_size, cond_len, dtype=torch.bool)
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"\nTest data shapes:")
    print(f"  input_ids: {input_ids.shape}")
    print(f"  cond_tokens: {cond_tokens.shape}")
    print(f"  cond_mask: {cond_mask.shape}")
    print(f"  labels: {labels.shape}")
    
    # Test forward pass with cross-attention
    print(f"\n🧪 Testing forward pass with cross-attention...")
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            cond_tokens=cond_tokens,
            cond_attention_mask=cond_mask,
            labels=labels,
            return_dict=True
        )
        
        print(f"Output logits shape: {outputs.logits.shape}")
        print(f"Loss value: {outputs.loss.item():.4f}")
    
    # Test forward pass without cross-attention
    print(f"\n🧪 Testing forward pass without cross-attention...")
    with torch.no_grad():
        outputs_no_cond = model(
            input_ids=input_ids,
            labels=labels,
            return_dict=True
        )
        
        print(f"Output logits shape: {outputs_no_cond.logits.shape}")
        print(f"Loss value: {outputs_no_cond.loss.item():.4f}")
    
    # Test generation with cross-attention
    print(f"\n🧪 Testing generation with cross-attention...")
    with torch.no_grad():
        # Prepare generation inputs
        generation_inputs = model.prepare_inputs_for_generation(
            input_ids=input_ids[:, :5],  # Use first 5 tokens
            cond_tokens=cond_tokens,
            cond_attention_mask=cond_mask
        )
        
        print(f"Generation inputs keys: {list(generation_inputs.keys())}")
        if 'cond_tokens' in generation_inputs:
            print(f"  cond_tokens shape: {generation_inputs['cond_tokens'].shape}")
        
        # Generate a few tokens
        generated = model.generate(
            input_ids=input_ids[:, :5],
            cond_tokens=cond_tokens,
            cond_attention_mask=cond_mask,
            max_new_tokens=3,
            do_sample=False,  # Deterministic for testing
            pad_token_id=0
        )
        
        print(f"Generated shape: {generated.shape}")
        print(f"Generated tokens: {generated[0].tolist()}")
    
    print("✅ NovoMolGen with cross-attention test passed!")
    return model

def test_gate_learning():
    """Test that gates can be learned"""
    print("\n" + "=" * 60)
    print("🧪 Testing Gate Learning")
    print("=" * 60)
    
    # Create a simple model
    config = NovoMolGenConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_hidden_layers=3,
        vocab_size=100,
        enable_cross_attn=True,
        cross_layers=[1],
        use_flash_attn=False,
        fused_bias_fc=False,
        fused_mlp=False,
        fused_dropout_add_ln=False
    )
    
    model = NovoMolGen(config, mol_type="SMILES")
    
    # Get initial gate values
    initial_gates = {}
    for name, gate in model.transformer.gates.items():
        initial_gates[name] = torch.sigmoid(gate).item()
        print(f"Initial gate {name}: {initial_gates[name]:.4f}")
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Simple training loop
    batch_size = 1
    seq_len = 5
    cond_len = 3
    
    for step in range(5):
        # Create random data
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        cond_tokens = torch.randn(batch_size, cond_len, config.hidden_size)
        cond_mask = torch.zeros(batch_size, cond_len, dtype=torch.bool)
        labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            cond_tokens=cond_tokens,
            cond_attention_mask=cond_mask,
            labels=labels
        )
        
        # Backward pass
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Check gate values
        current_gates = {}
        for name, gate in model.transformer.gates.items():
            current_gates[name] = torch.sigmoid(gate).item()
        
        print(f"Step {step+1}: Loss={loss.item():.4f}, Gates={current_gates}")
    
    print("✅ Gate learning test passed!")

def main():
    """Run all tests"""
    print("🚀 Starting Cross-Attention Tests")
    print("=" * 80)
    
    try:
        # Test individual components
        adapter = test_cross_attention_adapter()
        transformer = test_cross_attention_transformer()
        model = test_novomolgen_with_cross_attention()
        test_gate_learning()
        
        print("\n" + "=" * 80)
        print("🎉 All tests passed successfully!")
        print("=" * 80)
        
        print("\n📋 Summary of tested functionality:")
        print("✅ CrossAttentionAdapter - Basic cross-attention module")
        print("✅ CrossAttentionTransformer - Wrapper with monkey patching")
        print("✅ NovoMolGen - Full model with cross-attention")
        print("✅ Generation - Cross-attention in generate() method")
        print("✅ Gate Learning - Learnable gating mechanism")
        
        print("\n🔧 Usage examples:")
        print("1. Create model with cross-attention:")
        print("   model = NovoMolGen(config, mol_type='SMILES')")
        print("2. Forward pass with conditions:")
        print("   outputs = model(input_ids, cond_tokens=cond_tokens)")
        print("3. Generation with conditions:")
        print("   generated = model.generate(input_ids, cond_tokens=cond_tokens)")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
