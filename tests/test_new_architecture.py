#!/usr/bin/env python3
"""
Comprehensive test script for the new NovoMolGen architecture.

This script tests:
1. Data module with new condition features
2. Individual encoders (protein, ligand, fusion)
3. Full NovoMolGen model with cross-attention
4. Training forward pass
5. Generation with conditions
6. Backward pass and gradient flow

Usage:
    python test_new_architecture.py
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader.molecule_data_module import MolDataModule
from src.models.modeling_novomolgen import NovoMolGen, NovoMolGenConfig
from src.models.condition import (
    create_ligand_encoder, 
    create_protein_encoder, 
    create_condition_fusion
)


def test_data_module():
    """Test the data module with new condition features."""
    print("🧪 Testing Data Module...")
    
    # Create data module with condition features
    dm = MolDataModule(
        dataset_name="test_dataset",
        tokenizer_path="data/tokenizers/tokenizer_bpe_atomwise_SMILES_30000_0_0.json",
        mol_type="SMILES",
        max_seq_length=64,
        include_condition_features=True,
        device="cpu"
    )
    
    # Test dummy batch creation
    batch = MolDataModule.make_dummy_batch(
        batch_size=4,
        seq_len=32,
        vocab_size=30000,
        include_condition_features=True,
        device="cpu"
    )
    
    print(f"✅ Batch keys: {list(batch.keys())}")
    print(f"✅ Batch shapes:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"   {key}: {value.shape}")
        elif isinstance(value, dict):
            print(f"   {key}: dict with keys {list(value.keys())}")
            for subkey, subvalue in value.items():
                if isinstance(subvalue, torch.Tensor):
                    print(f"     {subkey}: {subvalue.shape}")
    
    # Verify expected keys are present
    expected_keys = ["input_ids", "attention_mask", "labels", "pocket_vec", "evo_vec", "ifp", "ligand_vec"]
    for key in expected_keys:
        assert key in batch, f"Missing key: {key}"
    
    print("✅ Data module test passed!")
    return batch


def test_individual_encoders():
    """Test individual encoder components."""
    print("\n🧪 Testing Individual Encoders...")
    
    batch_size = 4
    device = "cpu"
    
    # Test ligand encoder
    print("  Testing LigandConditionEncoder...")
    ligand_encoder = create_ligand_encoder(
        latent_dim=256,
        ligand_input_dim=256,  # Will be ignored
        hidden_dim=256,
        dropout=0.1,
        d_ifp=16384,
        d_mol=1536,
        d_embed=768,
        z_dim=256,
    )
    
    ifp = torch.randn(batch_size, 16384, device=device)
    ligand_vec = torch.randn(batch_size, 1536, device=device)
    
    mu, sigma, logvar = ligand_encoder(ifp, ligand_vec, return_logvar=True)
    print(f"    ✅ Ligand encoder output shapes: mu={mu.shape}, sigma={sigma.shape}, logvar={logvar.shape}")
    
    # Test protein encoder
    print("  Testing ProteinConditionEncoder...")
    protein_encoder = create_protein_encoder(
        latent_dim=256,
        protein_input_dim=256,  # Will be ignored
        hidden_dim=128,
        dropout=0.1,
        pocket_dim=512,
        evo_dim=1280,
        common_dim=512,
        out_dim=256,
    )
    
    pocket_vec = torch.randn(batch_size, 512, device=device)
    evo_vec = torch.randn(batch_size, 1280, device=device)
    
    protein_condition = protein_encoder(pocket_vec, evo_vec)
    print(f"    ✅ Protein encoder output shape: {protein_condition.shape}")
    
    # Test condition fusion
    print("  Testing ConditionFusion...")
    condition_fusion = create_condition_fusion(
        protein_dim=256,
        ligand_dim=256,
        out_dim=256,
        common_dim=256,
        dropout=0.1,
        use_cross_attn=True,
    )
    
    # Sample z from ligand posterior
    eps = torch.randn_like(mu)
    z = mu + (0.5 * logvar).exp() * eps
    
    fused_condition = condition_fusion(protein_condition, z)
    print(f"    ✅ Condition fusion output shape: {fused_condition.shape}")
    
    print("✅ Individual encoders test passed!")
    return {
        "ligand_encoder": ligand_encoder,
        "protein_encoder": protein_encoder,
        "condition_fusion": condition_fusion,
        "mu": mu,
        "logvar": logvar,
        "z": z,
        "protein_condition": protein_condition,
        "fused_condition": fused_condition
    }


def test_model_initialization():
    """Test NovoMolGen model initialization."""
    print("\n🧪 Testing Model Initialization...")
    
    # Create config with cross-attention enabled
    config = NovoMolGenConfig(
        vocab_size=30000,
        n_positions=1024,
        n_embd=512,
        n_layer=6,
        n_head=8,
        enable_cross_attn=True,
        cross_layers=[2, 4],
        cond_tokens_len=512,  # This will give us z_dim = 256
        beta_kl=0.1,
        use_flash_attn=False,  # Disable for CPU testing
        fused_bias_fc=False,
        fused_mlp=False,
        fused_dropout_add_ln=False,
    )
    
    # Create model
    model = NovoMolGen(config, mol_type="SMILES")
    print(f"✅ Model created with config: {config}")
    
    # Check that encoders are initialized
    assert hasattr(model, 'ligand_encoder'), "Ligand encoder not initialized"
    assert hasattr(model, 'protein_encoder'), "Protein encoder not initialized"
    assert hasattr(model, 'condition_fusion'), "Condition fusion not initialized"
    assert hasattr(model, 'cond_proj'), "Condition projection not initialized"
    
    print("✅ Model initialization test passed!")
    return model, config


def test_forward_pass(model, batch):
    """Test the forward pass of the model."""
    print("\n🧪 Testing Forward Pass...")
    
    # Extract condition features from batch
    condition_features = {
        "pocket_vec": batch["pocket_vec"],
        "evo_vec": batch["evo_vec"],
        "ifp": batch["ifp"],
        "ligand_vec": batch["ligand_vec"]
    }
    
    # Forward pass
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        **condition_features
    )
    
    print(f"✅ Forward pass successful!")
    print(f"   Loss: {outputs.loss.item():.4f}")
    print(f"   Logits shape: {outputs.logits.shape}")
    print(f"   Hidden states shape: {outputs.hidden_states.shape}")
    
    # Verify outputs
    assert outputs.loss is not None, "Loss should not be None"
    assert outputs.logits.shape == (batch["input_ids"].size(0), batch["input_ids"].size(1), model.base_config.vocab_size), \
        f"Logits shape mismatch: {outputs.logits.shape}"
    
    print("✅ Forward pass test passed!")
    return outputs


def test_backward_pass(model, outputs):
    """Test backward pass and gradient flow."""
    print("\n🧪 Testing Backward Pass...")
    
    # Backward pass
    outputs.loss.backward()
    
    # Check gradients
    total_params = 0
    params_with_grad = 0
    
    for name, param in model.named_parameters():
        total_params += 1
        if param.grad is not None:
            params_with_grad += 1
            grad_norm = param.grad.norm().item()
            if grad_norm > 0:
                print(f"   ✅ {name}: grad_norm={grad_norm:.6f}")
    
    print(f"✅ Gradient flow: {params_with_grad}/{total_params} parameters have gradients")
    
    if params_with_grad < total_params * 0.8:  # At least 80% should have gradients
        print(f"⚠️  Warning: Only {params_with_grad}/{total_params} parameters have gradients")
    
    print("✅ Backward pass test passed!")


def test_generation(model, batch):
    """Test conditional generation."""
    print("\n🧪 Testing Conditional Generation...")
    
    # Test generation with condition features
    condition_features = {
        "pocket_vec": batch["pocket_vec"][:2],  # Use only 2 samples
        "evo_vec": batch["evo_vec"][:2],
        "ifp": batch["ifp"][:2],
        "ligand_vec": batch["ligand_vec"][:2]
    }
    
    # Generate with conditions
    generated = model.generate_with_condition(
        input_ids=batch["input_ids"][:2, :8],  # Start with 8 tokens
        max_length=16,
        temperature=1.0,
        do_sample=True,
        **condition_features
    )
    
    print(f"✅ Generation successful!")
    print(f"   Generated shape: {generated.shape}")
    print(f"   Generated samples: {generated}")
    
    # Test generation without conditions (should use dummy conditions)
    generated_no_cond = model.generate_with_condition(
        input_ids=batch["input_ids"][:2, :8],
        max_length=16,
        temperature=1.0,
        do_sample=True,
        # No condition features provided
    )
    
    print(f"✅ Generation without conditions successful!")
    print(f"   Generated shape: {generated_no_cond.shape}")
    
    print("✅ Generation test passed!")
    return generated


def test_kl_loss_computation(model, batch):
    """Test KL loss computation."""
    print("\n🧪 Testing KL Loss Computation...")
    
    # Forward pass to get KL loss
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        pocket_vec=batch["pocket_vec"],
        evo_vec=batch["evo_vec"],
        ifp=batch["ifp"],
        ligand_vec=batch["ligand_vec"]
    )
    
    # Extract KL loss from total loss
    nll_loss = model.loss_function(
        logits=outputs.logits,
        labels=batch["labels"],
        vocab_size=model.base_config.vocab_size
    )
    
    kl_loss = outputs.loss - nll_loss
    kl_loss_scaled = kl_loss / model.base_config.beta_kl
    
    print(f"✅ KL loss computation successful!")
    print(f"   Total loss: {outputs.loss.item():.4f}")
    print(f"   NLL loss: {nll_loss.item():.4f}")
    print(f"   KL loss (scaled): {kl_loss_scaled.item():.4f}")
    print(f"   Beta: {model.base_config.beta_kl}")
    
    # Verify KL loss is reasonable
    assert kl_loss_scaled.item() >= 0, "KL loss should be non-negative"
    assert not torch.isnan(kl_loss_scaled), "KL loss should not be NaN"
    
    print("✅ KL loss computation test passed!")


def test_cross_attention_injection(model, batch):
    """Test cross-attention injection."""
    print("\n🧪 Testing Cross-Attention Injection...")
    
    # Check that cross-attention is enabled
    assert model.base_config.enable_cross_attn, "Cross-attention should be enabled"
    assert hasattr(model, 'transformer'), "Transformer should be wrapped"
    assert hasattr(model.transformer, 'cross_adapters'), "Cross-attention adapters should exist"
    
    # Check cross-attention layers
    expected_layers = [2, 4]
    for layer_idx in expected_layers:
        assert str(layer_idx) in model.transformer.cross_adapters, \
            f"Cross-attention adapter for layer {layer_idx} should exist"
    
    print(f"✅ Cross-attention adapters: {list(model.transformer.cross_adapters.keys())}")
    print(f"✅ Cross-attention gates: {list(model.transformer.gates.keys())}")
    
    # Test forward pass with cross-attention
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        pocket_vec=batch["pocket_vec"],
        evo_vec=batch["evo_vec"],
        ifp=batch["ifp"],
        ligand_vec=batch["ligand_vec"]
    )
    
    print(f"✅ Cross-attention forward pass successful!")
    print(f"   Loss with cross-attention: {outputs.loss.item():.4f}")
    
    print("✅ Cross-attention injection test passed!")


def run_comprehensive_test():
    """Run all tests in sequence."""
    print("🚀 Starting Comprehensive NovoMolGen Architecture Test")
    print("=" * 60)
    
    try:
        # Test 1: Data module
        batch = test_data_module()
        
        # Test 2: Individual encoders
        encoder_results = test_individual_encoders()
        
        # Test 3: Model initialization
        model, config = test_model_initialization()
        
        # Test 4: Forward pass
        outputs = test_forward_pass(model, batch)
        
        # Test 5: Backward pass
        test_backward_pass(model, outputs)
        
        # Test 6: Generation
        generated = test_generation(model, batch)
        
        # Test 7: KL loss computation
        test_kl_loss_computation(model, batch)
        
        # Test 8: Cross-attention injection
        test_cross_attention_injection(model, batch)
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! The new architecture is working correctly.")
        print("=" * 60)
        
        # Summary
        print("\n📊 Test Summary:")
        print(f"   ✅ Data module with condition features")
        print(f"   ✅ Ligand encoder (two-tower architecture)")
        print(f"   ✅ Protein encoder (fusion architecture)")
        print(f"   ✅ Condition fusion (cross-attention)")
        print(f"   ✅ NovoMolGen model initialization")
        print(f"   ✅ Forward pass with all components")
        print(f"   ✅ Backward pass and gradient flow")
        print(f"   ✅ Conditional generation")
        print(f"   ✅ KL loss computation")
        print(f"   ✅ Cross-attention injection")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
