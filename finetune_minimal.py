import os
import torch
from transformers import AutoTokenizer

from src.models.modeling_novomolgen import NovoMolGen, NovoMolGenConfig


# Removed count_trainable_parameters function - now handled inline for better accuracy


def list_trainable_modules(model: torch.nn.Module):
    names = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            names.append(name)
    return names


def build_optimizer(model: torch.nn.Module, lr: float = 1e-4, optimizer_type: str = "adamw"):
    # Only include parameters that require grad (new modules + optional LoRA)
    params = [p for p in model.parameters() if p.requires_grad]
    
    if optimizer_type == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    elif optimizer_type == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=0.01)
    elif optimizer_type == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.01)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=str, required=True, help="Path or HF id of pretrained NovoMolGen")
    parser.add_argument("--checkpoint_path", type=str, default="", help="subdir for weights if needed")
    parser.add_argument("--enable_cross_attn", action="store_true")
    parser.add_argument("--debug_no_cross_attn", action="store_true", help="Debug: disable cross-attn to test base model")
    parser.add_argument("--cross_layers", type=str, default="", help="comma-separated layer indices e.g. 4,8,12")
    parser.add_argument("--train_new_modules_only", action="store_true")
    parser.add_argument("--enable_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default="", help="comma-separated module names; auto-detect if empty")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adam", "sgd"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cross_layers = [int(x) for x in args.cross_layers.split(',') if x.strip()] if args.cross_layers else []
    lora_targets = [x.strip() for x in args.lora_target_modules.split(',') if x.strip()] if args.lora_target_modules else []

    print("Loading model...")
    # Load the pretrained config first, then modify it BEFORE creating the model
    config = NovoMolGenConfig.from_pretrained(args.pretrained, checkpoint_path=args.checkpoint_path)
    
    # Apply our finetune modifications to the config
    config.enable_cross_attn = args.enable_cross_attn and not args.debug_no_cross_attn
    config.cross_layers = cross_layers
    config.train_new_modules_only = bool(args.train_new_modules_only)
    config.enable_lora = False  # LoRA applied externally below
    
    # If train_new_modules_only=True but cross_attn is disabled, we need to create new modules
    # or disable train_new_modules_only to avoid having 0 trainable parameters
    if config.train_new_modules_only and not args.enable_cross_attn:
        print("Warning: train_new_modules_only=True but cross_attn is disabled.")
        print("This would result in 0 trainable parameters. Setting train_new_modules_only=False.")
        config.train_new_modules_only = False
    
    # Now create the model with the modified config
    model = NovoMolGen.from_pretrained(
        args.pretrained, checkpoint_path=args.checkpoint_path, config=config
    )
    
    model.to(device)
    # Use bfloat16 for flash-attn compatibility with better numerical stability than float16
    if hasattr(model, 'to') and torch.cuda.is_available():
        model = model.to(dtype=torch.bfloat16)
    model.train()

    # Apply freeze logic in the model (freeze all, unfreeze only new modules)
    if config.train_new_modules_only:
        model._maybe_setup_finetune_freeze_only()
        # Quick summary for sanity check
        if hasattr(model, 'print_trainable_summary'):
            model.print_trainable_summary(max_lines=50)
        
        # Reinitialize cross-attention adapters with smaller weights to prevent explosion
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'cross_adapters'):
            print("Reinitializing cross-attention adapters...")
            for name, adapter in model.transformer.cross_adapters.items():
                print(f"  Reinitializing adapter {name}")
                # Reinitialize with smaller weights
                for param_name, param in adapter.named_parameters():
                    if param.dim() >= 2:  # Weight matrices
                        torch.nn.init.xavier_uniform_(param, gain=0.01)  # Even smaller gain
                        print(f"    {param_name}: {param.shape}, norm={param.norm().item():.4f}")
                    else:  # Bias vectors
                        torch.nn.init.zeros_(param)
                        print(f"    {param_name}: {param.shape}, norm={param.norm().item():.4f}")

    # Optionally attach LoRA externally via peft
    if args.enable_lora:
        try:
            from peft import LoraConfig, get_peft_model
        except Exception as e:
            raise ImportError(f"LoRA requested but 'peft' is not installed/available: {e}")

        # autodetect targets if not specified
        if not lora_targets:
            lora_targets = model.autodetect_lora_targets()
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=lora_targets,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        model.to(device)
        # ensure LoRA params are trainable even if base is frozen
        for name, p in model.named_parameters():
            if 'lora_' in name:
                p.requires_grad = True

    # Count parameters correctly, handling both regular model and PEFT-wrapped model
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"Total params: {total_params:,}; Trainable: {trainable_params:,}; Frozen: {frozen_params:,}")
    print(f"Trainable percentage: {100 * trainable_params / total_params:.1f}%")

    trainable_names = list_trainable_modules(model)
    print("Trainable parameter tensors (first 50):")
    for n in trainable_names[:50]:
        print(f"  - {n}")
    
    # Show breakdown by component type
    cross_attn_params = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and 'cross_adapters' in n)
    lora_params = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and 'lora_' in n)
    encoder_params = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and any(x in n for x in ['ligand_encoder', 'protein_encoder', 'condition_fusion', 'cond_proj']))
    other_params = trainable_params - cross_attn_params - lora_params - encoder_params
    
    print(f"\nParameter breakdown:")
    print(f"  Cross-attention adapters: {cross_attn_params:,} ({100 * cross_attn_params / trainable_params:.1f}%)")
    print(f"  LoRA adapters: {lora_params:,} ({100 * lora_params / trainable_params:.1f}%)")
    print(f"  Encoders (ligand/protein/fusion): {encoder_params:,} ({100 * encoder_params / trainable_params:.1f}%)")
    if other_params > 0:
        print(f"  Other: {other_params:,} ({100 * other_params / trainable_params:.1f}%)")

    # Assert that no base param (non-LoRA) is trainable when requested
    if config.train_new_modules_only:
        for name, p in model.named_parameters():
            if p.requires_grad and 'lora_' not in name:
                allowed_prefixes = (
                    'transformer.cross_adapters',
                    'transformer.gates',
                    'ligand_encoder',
                    'protein_encoder',
                    'condition_fusion',
                    'cond_proj',
                )
                assert name.startswith(allowed_prefixes), f"Base param unexpectedly trainable: {name}"

    # Tiny synthetic batch
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained)
    tokenizer.pad_token = tokenizer.eos_token
    batch_size = args.batch_size
    seq_len = args.seq_len
    input_ids = torch.randint(low=0, high=tokenizer.vocab_size, size=(batch_size, seq_len), device=device)
    labels = input_ids.clone()

    # Dummy condition features for a forward/backward pass
    cond_b = batch_size
    # Use bfloat16 for condition features to match model dtype
    model_dtype = next(model.parameters()).dtype
    pocket_vec = torch.randn(cond_b, 512, device=device, dtype=model_dtype)
    evo_vec = torch.randn(cond_b, 1280, device=device, dtype=model_dtype)
    ifp = torch.randn(cond_b, 16384, device=device, dtype=model_dtype)
    ligand_vec = torch.randn(cond_b, 1536, device=device, dtype=model_dtype)

    optim = build_optimizer(model, lr=args.lr, optimizer_type=args.optimizer)
    
    # Add gradient clipping to prevent explosion
    max_grad_norm = 1.0  # Standard clipping

    for step in range(args.steps):
        optim.zero_grad(set_to_none=True)
        out = model(
            input_ids=input_ids,
            labels=labels,
            pocket_vec=pocket_vec,
            evo_vec=evo_vec,
            ifp=ifp,
            ligand_vec=ligand_vec,
        )
        loss = out.loss
        
        # Debug: Check for NaN/Inf values
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"step={step} WARNING: loss is {'NaN' if torch.isnan(loss) else 'Inf'}")
            print("Checking model outputs...")
            print(f"  logits shape: {out.logits.shape}")
            print(f"  logits min/max: {out.logits.min().item():.4f}/{out.logits.max().item():.4f}")
            print(f"  logits has NaN: {torch.isnan(out.logits).any()}")
            print(f"  logits has Inf: {torch.isinf(out.logits).any()}")
            
            # Check gradients before backward
            print("Checking trainable parameters...")
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(f"  {name}: shape={param.shape}, has NaN={torch.isnan(param).any()}, has Inf={torch.isinf(param).any()}")
                    break  # Just check first few
            break  # Stop training on NaN
        else:
            print(f"step={step} loss={loss.item():.4f}")
            loss.backward()
            
            # Check gradients after backward
            grad_norms = []
            has_nan_grads = False
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    grad_norms.append(grad_norm)
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        print(f"  WARNING: {name} has NaN/Inf gradients")
                        has_nan_grads = True
            
            if grad_norms:
                print(f"  grad norm stats: min={min(grad_norms):.4f}, max={max(grad_norms):.4f}, mean={sum(grad_norms)/len(grad_norms):.4f}")
            
            # Skip optimizer step if gradients are NaN/Inf
            if has_nan_grads:
                print("  Skipping optimizer step due to NaN/Inf gradients")
                continue
            
            # Apply gradient clipping before optimizer step
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            
            optim.step()

    print("Finished tiny finetune loop.")


if __name__ == "__main__":
    main()


