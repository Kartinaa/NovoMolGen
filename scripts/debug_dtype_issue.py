#!/usr/bin/env python3
"""
Debug script to identify dtype issues in conditional generation.
"""
import torch
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.modeling_novomolgen import NovoMolGen, NovoMolGenConfig
from data_loader.molecule_tokenizer import MoleculeTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_tensor_dtype(tensor, name, expected_dtype=None):
    """Check and log tensor dtype."""
    actual_dtype = tensor.dtype
    status = "✅" if expected_dtype is None or actual_dtype == expected_dtype else "❌"
    logger.info(f"{status} {name}: dtype={actual_dtype}, shape={tensor.shape}, device={tensor.device}")
    if expected_dtype and actual_dtype != expected_dtype:
        logger.warning(f"  Expected {expected_dtype}, got {actual_dtype}")
    return actual_dtype

def test_model_dtype():
    """Test model dtype and intermediate tensors."""
    logger.info("=" * 80)
    logger.info("Testing Model Dtype Issues")
    logger.info("=" * 80)
    
    # Load model
    model_path = "outputs/11_10_25_SAFEGen/checkpoint-33400/full_model"
    logger.info(f"\n1. Loading model from {model_path}")
    
    try:
        device = "cuda"
        model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = NovoMolGen.from_pretrained(
            model_path,
            torch_dtype=model_dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        model = model.to(device)
        
        # CRITICAL: Explicitly convert model to bfloat16 if on CUDA
        if torch.cuda.is_available() and model_dtype == torch.bfloat16:
            model = model.to(torch.bfloat16)
        
        model.eval()
        
        # Check model dtype
        model_dtype = next(model.parameters()).dtype
        logger.info(f"   Model dtype: {model_dtype}")
        
        # Check transformer dtype
        transformer_dtype = next(model.transformer.parameters()).dtype
        logger.info(f"   Transformer dtype: {transformer_dtype}")
        
        # Check base_transformer dtype
        base_transformer_dtype = next(model.transformer.base_transformer.parameters()).dtype
        logger.info(f"   Base transformer dtype: {base_transformer_dtype}")
        
        # Check encoder dtypes
        if hasattr(model, 'ligand_encoder'):
            ligand_encoder_dtype = next(model.ligand_encoder.parameters()).dtype
            logger.info(f"   Ligand encoder dtype: {ligand_encoder_dtype}")
        
        if hasattr(model, 'protein_encoder'):
            protein_encoder_dtype = next(model.protein_encoder.parameters()).dtype
            logger.info(f"   Protein encoder dtype: {protein_encoder_dtype}")
        
        if hasattr(model, 'condition_fusion'):
            condition_fusion_dtype = next(model.condition_fusion.parameters()).dtype
            logger.info(f"   Condition fusion dtype: {condition_fusion_dtype}")
        
        if hasattr(model, 'cond_proj'):
            cond_proj_dtype = next(model.cond_proj.parameters()).dtype
            logger.info(f"   Cond proj dtype: {cond_proj_dtype}")
        
        # Check cross-attention adapters
        if hasattr(model.transformer, 'cross_adapters'):
            for layer_idx, adapter in model.transformer.cross_adapters.items():
                adapter_dtype = next(adapter.parameters()).dtype
                logger.info(f"   Cross-attention adapter layer {layer_idx} dtype: {adapter_dtype}")
        
    except Exception as e:
        logger.error(f"   Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Load tokenizer
    logger.info(f"\n2. Loading tokenizer")
    try:
        tokenizer_path = f"{model_path}/tokenizer.json"
        tokenizer = MoleculeTokenizer.load(tokenizer_path)
        tokenizer = tokenizer.get_pretrained()
        logger.info(f"   Tokenizer loaded: {tokenizer_path}")
    except Exception as e:
        logger.error(f"   Failed to load tokenizer: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create test inputs
    logger.info(f"\n3. Creating test inputs")
    device = "cuda"
    batch_size = 2
    model_dtype = next(model.parameters()).dtype
    
    # Input IDs
    input_ids = torch.tensor([[tokenizer.bos_token_id]], device=device).repeat(batch_size, 1)
    check_tensor_dtype(input_ids, "input_ids", torch.long)
    
    # Condition features
    pocket_vec = torch.randn(batch_size, 512, device=device, dtype=torch.float32)
    evo_vec = torch.randn(batch_size, 1280, device=device, dtype=torch.float32)
    ifp = torch.randn(batch_size, 16384, device=device, dtype=torch.float32)
    ligand_vec = torch.randn(batch_size, 1536, device=device, dtype=torch.float32)
    
    logger.info(f"\n4. Condition features (before conversion):")
    check_tensor_dtype(pocket_vec, "pocket_vec", torch.float32)
    check_tensor_dtype(evo_vec, "evo_vec", torch.float32)
    check_tensor_dtype(ifp, "ifp", torch.float32)
    check_tensor_dtype(ligand_vec, "ligand_vec", torch.float32)
    
    # Convert to model dtype
    pocket_vec = pocket_vec.to(model_dtype)
    evo_vec = evo_vec.to(model_dtype)
    ifp = ifp.to(model_dtype)
    ligand_vec = ligand_vec.to(model_dtype)
    
    logger.info(f"\n5. Condition features (after conversion to {model_dtype}):")
    check_tensor_dtype(pocket_vec, "pocket_vec", model_dtype)
    check_tensor_dtype(evo_vec, "evo_vec", model_dtype)
    check_tensor_dtype(ifp, "ifp", model_dtype)
    check_tensor_dtype(ligand_vec, "ligand_vec", model_dtype)
    
    # Test encoder forward passes
    logger.info(f"\n6. Testing encoder forward passes:")
    try:
        with torch.no_grad():
            protein_condition = model.protein_encoder(pocket_vec, evo_vec)
            check_tensor_dtype(protein_condition, "protein_condition", model_dtype)
            
            mu, sigma, logvar = model.ligand_encoder(ifp, ligand_vec, return_logvar=True)
            check_tensor_dtype(mu, "mu", model_dtype)
            check_tensor_dtype(sigma, "sigma", model_dtype)
            check_tensor_dtype(logvar, "logvar", model_dtype)
            
            z = mu  # Use mean, not sampling
            check_tensor_dtype(z, "z", model_dtype)
            
            fused_condition = model.condition_fusion(protein_condition, z)
            check_tensor_dtype(fused_condition, "fused_condition", model_dtype)
            
            cond_vector = torch.cat([z, fused_condition], dim=-1)
            check_tensor_dtype(cond_vector, "cond_vector", model_dtype)
            
            new_cond_tokens = model.cond_proj(cond_vector.unsqueeze(-1))
            check_tensor_dtype(new_cond_tokens, "new_cond_tokens", model_dtype)
            
    except Exception as e:
        logger.error(f"   Failed in encoder forward: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test transformer forward with hook
    logger.info(f"\n7. Testing transformer forward with dtype hooks:")
    
    dtype_issues = []
    
    def hook_fn(module, input, output):
        """Hook to check dtype of inputs and outputs."""
        if isinstance(input, tuple):
            for i, inp in enumerate(input):
                if isinstance(inp, torch.Tensor):
                    if inp.dtype not in [torch.float16, torch.bfloat16]:
                        dtype_issues.append(f"{module.__class__.__name__} input[{i}]: {inp.dtype}")
        if isinstance(output, tuple):
            for i, out in enumerate(output):
                if isinstance(out, torch.Tensor):
                    if out.dtype not in [torch.float16, torch.bfloat16]:
                        dtype_issues.append(f"{module.__class__.__name__} output[{i}]: {out.dtype}")
        elif isinstance(output, torch.Tensor):
            if output.dtype not in [torch.float16, torch.bfloat16]:
                dtype_issues.append(f"{module.__class__.__name__} output: {output.dtype}")
    
    # Register hooks on base_transformer layers
    hooks = []
    try:
        for i, layer in enumerate(model.transformer.base_transformer.layers):
            hook = layer.register_forward_hook(hook_fn)
            hooks.append(hook)
        
        # Test forward pass
        logger.info(f"   Testing forward pass...")
        with torch.no_grad():
            # Set condition tokens
            model.transformer._current_cond_tokens = new_cond_tokens
            model.transformer._current_cond_attention_mask = torch.zeros(
                batch_size, new_cond_tokens.size(1), dtype=torch.bool, device=device
            )
            
            # Forward pass
            hidden_states = model.transformer(input_ids)
            check_tensor_dtype(hidden_states, "transformer output (hidden_states)", model_dtype)
            
    except Exception as e:
        logger.error(f"   Failed in transformer forward: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()
    
    if dtype_issues:
        logger.warning(f"\n⚠️  Found {len(dtype_issues)} dtype issues:")
        for issue in dtype_issues[:10]:  # Show first 10
            logger.warning(f"   - {issue}")
    else:
        logger.info(f"\n✅ No dtype issues found in forward pass")
    
    # Test generation (this is where the error occurs)
    logger.info(f"\n8. Testing generation (this is where the error occurs):")
    try:
        with torch.no_grad():
            # Clear condition tokens (will be set by generate_with_condition)
            if hasattr(model.transformer, '_current_cond_tokens'):
                model.transformer._current_cond_tokens = None
                model.transformer._current_cond_attention_mask = None
            
            logger.info(f"   Calling generate_with_condition...")
            generated = model.generate_with_condition(
                input_ids=input_ids,
                pocket_vec=pocket_vec,
                evo_vec=evo_vec,
                ifp=ifp,
                ligand_vec=ligand_vec,
                sample_posterior=False,
                max_length=10,  # Short for testing
                temperature=1.0,
                top_k=50,
                top_p=0.95,
                eos_token_id=tokenizer.eos_token_id,
            )
            logger.info(f"   ✅ Generation successful! Generated shape: {generated.shape}")
            
    except AssertionError as e:
        logger.error(f"   ❌ AssertionError (dtype issue): {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        logger.error(f"   ❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_dtype()

