import copy
import json
import os.path
import re
import shutil
import inspect
from typing import Optional, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig
from transformers.loss.loss_utils import LOSS_MAPPING
from transformers.modeling_outputs import CausalLMOutput
from transformers.utils.hub import cached_file, get_checkpoint_shard_files
from transformers.utils import (
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
)
from transformers.modeling_utils import unwrap_model, logger
from functools import partial
from safetensors.torch import load_file as safe_load_file
from typing import Iterable

try:
    from flash_attn.models.gpt import GPTLMHeadModel
except ImportError:
    GPTLMHeadModel = None

try:
    from flash_attn.models.llama import llama_config_to_gpt2_config, inv_remap_state_dict_hf_llama
except ImportError:
    llama_config_to_gpt2_config = None
    inv_remap_state_dict_hf_llama = None

class CrossAttentionAdapter(nn.Module):
    """
    Pure Cross-Attention adapter module for conditional generation.
    
    Takes hidden_states as Query and cond_tokens as Key/Value.
    Only performs cross-attention without residual connection or layer norm.
    
    Args:
        config: Model configuration containing hidden_size, num_attention_heads
        dropout: Dropout rate for attention output
    """
    
    def __init__(self, config, dropout: float = 0.1):
        super().__init__()
        
        # Validate config
        if not hasattr(config, 'hidden_size'):
            raise ValueError("Config must have 'hidden_size' attribute")
        if not hasattr(config, 'num_attention_heads'):
            raise ValueError("Config must have 'num_attention_heads' attribute")
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        # Validate head dimension
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads ({self.num_heads})")
        
        # Multi-head cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_heads,
            dropout=dropout,
            batch_first=True,
            bias=True
        )
        
        # Note: Dropout is handled in the patched forward, not here
    
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        cond_tokens: torch.Tensor,
        cond_attention_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Union[torch.Tensor, tuple]:
        """
        Forward pass of CrossAttention adapter.
        
        Args:
            hidden_states: Query tensor [B, T, d] where B=batch, T=sequence_length, d=hidden_size
            cond_tokens: Key/Value tensor [B, Lc, d] where Lc=condition_length (Lc need to be 
            padded and don't need to have the same length as hidden_states)
            cond_attention_mask: Optional attention mask for cond_tokens [B, Lc] (True=padding) (Optional, 
            highly likely `None`)
            return_attention_weights: Whether to return attention weights
            
        Returns:
            If return_attention_weights=False: attention output tensor [B, T, d]
            If return_attention_weights=True: (attention output tensor [B, T, d], attention_weights [B, num_heads, T, Lc])
        """
        
        # Shape validation
        if hidden_states.dim() != 3:
            raise ValueError(f"hidden_states must be 3D tensor [B, T, d], got shape {hidden_states.shape}")
        if cond_tokens.dim() != 3:
            raise ValueError(f"cond_tokens must be 3D tensor [B, Lc, d], got shape {cond_tokens.shape}")
        
        B, T, d = hidden_states.shape
        B_cond, Lc, d_cond = cond_tokens.shape
        
        # Batch size consistency check
        if B != B_cond:
            raise ValueError(f"Batch size mismatch: hidden_states has B={B}, cond_tokens has B={B_cond}")
        
        # Hidden size consistency check
        # TODO: yb: Do we really need same hidden size?
        if d != self.hidden_size:
            raise ValueError(f"hidden_states hidden_size {d} != expected {self.hidden_size}")
        if d_cond != self.hidden_size:
            raise ValueError(f"cond_tokens hidden_size {d_cond} != expected {self.hidden_size}")
        
        # Handle empty condition sequences
        if Lc == 0:
            # Return zero output for empty conditions
            zero_output = torch.zeros_like(hidden_states)
            if return_attention_weights:
                # Return empty attention weights
                attn_weights = torch.zeros(B, self.num_heads, T, 0, device=hidden_states.device, dtype=hidden_states.dtype)
                return zero_output, attn_weights
            else:
                return zero_output
        
        # Attention mask validation
        if cond_attention_mask is not None:
            if cond_attention_mask.dim() != 2:
                raise ValueError(f"cond_attention_mask must be 2D tensor [B, Lc], got shape {cond_attention_mask.shape}")
            if cond_attention_mask.shape[0] != B:
                raise ValueError(f"cond_attention_mask batch size {cond_attention_mask.shape[0]} != {B}")
            if cond_attention_mask.shape[1] != Lc:
                raise ValueError(f"cond_attention_mask length {cond_attention_mask.shape[1]} != {Lc}")
            # Ensure boolean mask
            if cond_attention_mask.dtype != torch.bool:
                cond_attention_mask = cond_attention_mask.bool()
        
        # Cross-attention: hidden_states (Q) attends to cond_tokens (K, V)
        attn_output, attn_weights = self.cross_attn(
            query=hidden_states,           # [B, T, d]
            key=cond_tokens,               # [B, Lc, d] 
            value=cond_tokens,             # [B, Lc, d]
            key_padding_mask=cond_attention_mask,  # [B, Lc] - True for padding tokens
            need_weights=return_attention_weights,
            average_attn_weights=False
        )
        
        # Note: Dropout is applied in the patched forward method, not here
        
        if return_attention_weights:
            return attn_output, attn_weights
        else:
            return attn_output
    
    def extra_repr(self) -> str:
        """Extra representation string for debugging."""
        return f"hidden_size={self.hidden_size}, num_heads={self.num_heads}"

def state_dict_from_pretrained(model_name, checkpoint_path: str = "", device=None, dtype=None, **kwargs):
    """
    code modified from: https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/utils/pretrained.py
    """

    # If not fp32, then we don't want to load directly to the GPU
    mapped_device = "cpu" if dtype not in [torch.float32, None] else device
    is_sharded = False
    load_safe = False

    # Try loading from HF hub instead of from local files
    resolved_archive_file = cached_file(model_name, os.path.join(checkpoint_path, WEIGHTS_NAME),
                                        _raise_exceptions_for_missing_entries=False, **kwargs)
    if resolved_archive_file is None:
        resolved_archive_file = cached_file(model_name, os.path.join(checkpoint_path, WEIGHTS_INDEX_NAME),
                                            _raise_exceptions_for_missing_entries=False, **kwargs)
        if resolved_archive_file is not None:
            is_sharded = True

    if resolved_archive_file is None:
        raise EnvironmentError(f"Model name {model_name} was not found.")

    if load_safe:
        loader = partial(safe_load_file, device=mapped_device)
    else:
        loader = partial(torch.load, map_location=mapped_device)

    if is_sharded:
        # resolved_archive_file becomes a list of files that point to the different
        # checkpoint shards in this case.
        resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(
            model_name, resolved_archive_file
        )
        state_dict = {}
        for sharded_file in resolved_archive_file:
            state_dict.update(loader(sharded_file))
    else:
        state_dict = loader(resolved_archive_file)
    # Convert dtype before moving to GPU to save memory
    if dtype is not None:
        state_dict = {k: v.to(dtype=dtype) for k, v in state_dict.items()}
    state_dict = {k: v.to(device=device) for k, v in state_dict.items()}

    return state_dict


class NovoMolGenConfig(LlamaConfig):

    def __init__(self,
                 use_flash_attn: bool = True,
                 fused_bias_fc: bool = True,
                 fused_mlp: bool = False,
                 fused_dropout_add_ln: bool = True,
                 residual_in_fp32: bool = True,
                 loss_type: str = 'ForCausalLM',
                 beta_kl: float = 0.1,
                 # KL annealing parameters
                 beta_kl_init: float = 0.0,
                 beta_kl_final: float = 1.0,
                 beta_kl_anneal_type: str = "linear",  # "linear", "cosine", "cyclical"
                 beta_kl_anneal_steps: int = 30000,
                 beta_kl_eval: Optional[float] = None,  # Fixed beta for evaluation
                 enable_cross_attn: bool = False,
                 cross_layers: Optional[List[int]] = None,
                 cond_tokens_len: int = 512,
                 # Finetune / LoRA controls
                 train_new_modules_only: bool = True,
                 enable_lora: bool = False,
                 lora_r: int = 16,
                 lora_alpha: int = 32,
                 lora_dropout: float = 0.05,
                 lora_target_modules: Optional[List[str]] = None,
                 # Ligand normalization
                 ligand_vec_stats_path: Optional[str] = None,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.use_flash_attn = use_flash_attn
        self.fused_bias_fc = fused_bias_fc
        self.fused_mlp = fused_mlp
        self.fused_dropout_add_ln = fused_dropout_add_ln
        self.residual_in_fp32 = residual_in_fp32
        self.loss_type = loss_type
        self.beta_kl = float(beta_kl)  # Legacy fixed beta (used if annealing not configured)
        # KL annealing parameters
        self.beta_kl_init = float(beta_kl_init)
        self.beta_kl_final = float(beta_kl_final)
        self.beta_kl_anneal_type = str(beta_kl_anneal_type)
        self.beta_kl_anneal_steps = int(beta_kl_anneal_steps)
        self.beta_kl_eval = float(beta_kl_eval) if beta_kl_eval is not None else None
        self.enable_cross_attn = enable_cross_attn
        self.cross_layers = cross_layers if cross_layers is not None else []
        self.cond_tokens_len = int(cond_tokens_len)
        # Finetune / LoRA controls (defaults are safe no-op unless toggled)
        self.train_new_modules_only = bool(train_new_modules_only)
        self.enable_lora = bool(enable_lora)
        self.lora_r = int(lora_r)
        self.lora_alpha = int(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        # If None, we will auto-detect at runtime
        self.lora_target_modules = lora_target_modules or []
        # Ligand normalization statistics path
        self.ligand_vec_stats_path = ligand_vec_stats_path
        self.auto_map = {"AutoModelForCausalLM": "modeling_novomolgen.NovoMolGen"}

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: Union[str, os.PathLike],
            checkpoint_path: str = "",
            cache_dir: Optional[Union[str, os.PathLike]] = None,
            force_download: bool = False,
            local_files_only: bool = False,
            token: Optional[Union[str, bool]] = None,
            revision: str = "main",
            **kwargs,
    ):

        resolved_archive_config_file = cached_file(pretrained_model_name_or_path,
                                                   os.path.join(checkpoint_path, "config.json"),
                                                   _raise_exceptions_for_missing_entries=False, force_download=force_download)

        if resolved_archive_config_file is not None:
            with open(resolved_archive_config_file, "r", encoding="utf-8") as reader:
                text = reader.read()
            config_dict = json.loads(text)

        else:
            raise EnvironmentError(f"config for {pretrained_model_name_or_path} was not found.")

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            print(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class CrossAttentionTransformer(nn.Module):
    """
    Custom transformer that supports cross-attention injection at specified layers.
    This wraps the flash-attn transformer and patches the decoder blocks.
    """
    
    def __init__(self, base_transformer, config):
        super().__init__()
        self.base_transformer = base_transformer
        self.config = config
        self.enable_cross_attn = getattr(config, 'enable_cross_attn', False)
        self.cross_layers = getattr(config, 'cross_layers', [])
        
        # Initialize cross-attention adapters if enabled
        if self.enable_cross_attn and self.cross_layers:
            self.cross_adapters = nn.ModuleDict({
                str(layer_idx): CrossAttentionAdapter(config) 
                for layer_idx in self.cross_layers
            })
            
            # Initialize learnable gates for each layer
            # Use -1.0 for conservative initialization: sigmoid(-1) ≈ 0.27
            self.gates = nn.ParameterDict({
                str(layer_idx): nn.Parameter(torch.tensor(0.5))
                for layer_idx in self.cross_layers
            })
            
            # Initialize cross-attention dropout
            self.cross_dropout = nn.Dropout(getattr(config, 'attn_dropout', 0.1))
            
            # Patch the transformer blocks to inject cross-attention
            self._patch_transformer_blocks()
        else:
            self.cross_adapters = None
            self.gates = None
            self.cross_dropout = None
    
    def _patch_transformer_blocks(self):
        """Patch the transformer blocks to inject cross-attention at specified layers."""
        # Get the transformer blocks from the base transformer
        if hasattr(self.base_transformer, 'blocks'):
            blocks = self.base_transformer.blocks
        elif hasattr(self.base_transformer, 'layers'):
            blocks = self.base_transformer.layers
        else:
            # Try to find blocks in nested structure
            blocks = None
            for attr_name in ['transformer', 'model', 'encoder', 'decoder']:
                if hasattr(self.base_transformer, attr_name):
                    nested = getattr(self.base_transformer, attr_name)
                    if hasattr(nested, 'blocks'):
                        blocks = nested.blocks
                        break
                    elif hasattr(nested, 'layers'):
                        blocks = nested.layers
                        break
        
        if blocks is None:
            raise ValueError("Could not find transformer blocks to patch")
        
        # Store original forward methods
        self._original_forwards = {}
        
        # Patch each specified layer
        for layer_idx in self.cross_layers:
            if layer_idx < len(blocks):
                block = blocks[layer_idx]
                self._original_forwards[layer_idx] = block.forward
                
                # Create patched forward method with proper closure binding
                def make_patched_forward(original_forward, adapter, gate, layer_idx, block_bound=block, self_bound=self):
                    def patched_forward(hidden_states, *args, **kwargs):
                        # Ensure input hidden_states have correct dtype for flash_attn
                        # This is critical because flash_attn's internal operations require float16/bfloat16
                        model_dtype = next(self_bound.base_transformer.parameters()).dtype
                        if hidden_states.dtype != model_dtype:
                            hidden_states = hidden_states.to(model_dtype)
                        
                        # Call original forward (self-attention + MLP)
                        original_output = original_forward(hidden_states, *args, **kwargs)
                        
                        # Handle tuple return (some blocks return (output, rest))
                        if isinstance(original_output, tuple):
                            output, rest = original_output[0], original_output[1:]
                        else:
                            output = original_output
                            rest = ()
                        
                        # Apply cross-attention if cond_tokens are available
                        if hasattr(self_bound, '_current_cond_tokens') and self_bound._current_cond_tokens is not None and str(layer_idx) in self_bound.cross_adapters:
                            # Apply layer norm before cross-attention
                            normed_output = F.layer_norm(output, output.shape[-1:])
                            
                            # Ensure normed_output and cond_tokens have correct dtype for flash_attn cross-attention
                            # flash_attn requires float16 or bfloat16, not float32
                            model_dtype = next(self_bound.base_transformer.parameters()).dtype
                            if normed_output.dtype != model_dtype:
                                normed_output = normed_output.to(model_dtype)
                            cond_tokens = self_bound._current_cond_tokens
                            if cond_tokens.dtype != model_dtype:
                                cond_tokens = cond_tokens.to(model_dtype)
                            
                            # Apply cross-attention adapter (pure cross-attention, no residual)
                            attn_out = adapter(normed_output, cond_tokens, 
                                             self_bound._current_cond_attention_mask)
                            
                            # Apply dropout to attention output (single dropout point)
                            attn_out = self_bound.cross_dropout(attn_out)
                            
                            # Unified gated residual connection: output = output + sigmoid(gate) * attn_out
                            gate_value = torch.sigmoid(gate)
                            output = output + gate_value * attn_out
                        
                        # Return in original format
                        if rest:
                            return (output,) + rest
                        else:
                            return output
                    return patched_forward
                
                # Apply the patch with proper binding
                block.forward = make_patched_forward(
                    block.forward, 
                    self.cross_adapters[str(layer_idx)], 
                    self.gates[str(layer_idx)], 
                    layer_idx
                )
    
    def forward(self, input_ids, position_ids=None, inference_params=None, 
                cond_tokens=None, cond_attention_mask=None):
        """
        Forward pass with cross-attention injection at specified layers.
        
        Args:
            input_ids: Input token IDs [B, T]
            position_ids: Position IDs [B, T] 
            inference_params: Inference parameters for generation
            cond_tokens: Condition tokens for cross-attention [B, Lc, d]
            cond_attention_mask: Attention mask for condition tokens [B, Lc] (True=padding)
        """
        # Store condition tokens for use in patched blocks
        # Only update if cond_tokens is explicitly provided (not None)
        # This allows generate_with_condition to set _current_cond_tokens before calling generate()
        # and prevents generate() from overwriting it when calling forward() with cond_tokens=None
        if self.enable_cross_attn and self.cross_adapters:
            if cond_tokens is not None:
                # Explicitly provided cond_tokens: update _current_cond_tokens
                self._current_cond_tokens = cond_tokens
                self._current_cond_attention_mask = cond_attention_mask
            # If cond_tokens is None, keep the existing _current_cond_tokens (set by generate_with_condition)
        
        # Call the base transformer (with patched blocks)
        hidden_states = self.base_transformer(input_ids, position_ids=position_ids, 
                                            inference_params=inference_params)
        
        # Ensure output hidden_states have correct dtype for flash_attn
        # This is critical because flash_attn's internal operations (including inner_cross_attn for KV cache)
        # require float16 or bfloat16, not float32
        model_dtype = next(self.base_transformer.parameters()).dtype
        if hidden_states.dtype != model_dtype:
            hidden_states = hidden_states.to(model_dtype)
        
        # Clear condition tokens
        if hasattr(self, '_current_cond_tokens'):
            self._current_cond_tokens = None
            self._current_cond_attention_mask = None
        
        return hidden_states


class NovoMolGen(GPTLMHeadModel):
    def __init__(
            self,
            config: NovoMolGenConfig,
            mol_type: str = "SMILES",
    ):
        self.base_config = config
        self.mol_type = mol_type
        config = llama_config_to_gpt2_config(config)
        config.use_flash_attn = self.base_config.use_flash_attn
        config.fused_bias_fc = self.base_config.fused_bias_fc
        config.fused_mlp = self.base_config.fused_mlp
        config.fused_dropout_add_ln = self.base_config.fused_dropout_add_ln
        config.residual_in_fp32 = self.base_config.residual_in_fp32
        config.enable_cross_attn = self.base_config.enable_cross_attn
        config.cross_layers = self.base_config.cross_layers
        GPTLMHeadModel.__init__(self, config)
        
        # Replace the transformer with our custom cross-attention transformer
        if self.base_config.enable_cross_attn:
            self.transformer = CrossAttentionTransformer(self.transformer, config)
            # Projection to convert a condition vector of length Lc to tokens [B, Lc, d]
            # We simply lift each scalar to the model hidden size with a linear layer.
            self.cond_proj = nn.Linear(1, config.n_embd if hasattr(config, 'n_embd') else config.hidden_size, bias=False)
            
            # Initialize encoders with proper dimensions for the new architecture
            from .condition import create_ligand_encoder, create_protein_encoder, create_condition_fusion
            cond_latent_dim = self.base_config.cond_tokens_len  # z/2 as specified
            print(f"cond_latent_dim: {cond_latent_dim}")
            
            # Ligand encoder: takes ifp [B, 16384] and ligand_vec [B, 1536], outputs mu/logvar [B, z_dim]
            self.ligand_encoder = create_ligand_encoder(
                latent_dim=cond_latent_dim,
                ligand_input_dim=cond_latent_dim,  # This will be ignored in the new two-tower architecture
                hidden_dim=512,
                dropout=0.1,
                d_ifp=16384,
                d_mol=1536,
                d_embed=512,
                z_dim=cond_latent_dim,
                ligand_vec_stats_path=getattr(config, 'ligand_vec_stats_path', None),
            )
            
            # Protein encoder: takes pocket_vec [B, 512] and evo_vec [B, 1280], outputs protein condition [B, z_dim]
            self.protein_encoder = create_protein_encoder(
                latent_dim=cond_latent_dim,
                protein_input_dim=cond_latent_dim,  # This will be ignored in the new fusion architecture
                dropout=0.1,
                pocket_dim=512,
                evo_dim=1280,
                common_dim=cond_latent_dim,
                out_dim=cond_latent_dim,
            )
            
            # Condition fusion: combines protein and ligand conditions
            self.condition_fusion = create_condition_fusion(
                protein_dim=cond_latent_dim,
                ligand_dim=cond_latent_dim,
                out_dim=cond_latent_dim,
                common_dim=cond_latent_dim,
                dropout=0.1,
                use_cross_attn=True,
            )

        # Finetune controls: optionally freeze base; LoRA applied externally
        self._maybe_setup_finetune_freeze_only()
        
        # Track current training step for KL annealing
        self._current_step = 0

    # ------------------------
    # Finetune & LoRA utilities
    # ------------------------
    def _iter_new_modules(self) -> Iterable[nn.Module]:
        """
        Return only the set of newly added modules to train:
        - If cross-attn is disabled: train only encoders/fusion/cond_proj
        - If cross-attn is enabled: train only cross_adapters/gates (keep the rest of the
          transformer frozen), plus encoders/fusion/cond_proj
        """
        # Encoders, fusion, and cond_proj are always considered "new modules"
        for m in [getattr(self, 'ligand_encoder', None),
                  getattr(self, 'protein_encoder', None),
                  getattr(self, 'condition_fusion', None)]:
            if m is not None:
                yield m
        if hasattr(self, 'cond_proj'):
            yield self.cond_proj

        # For cross-attn: return only adapters/gates, not the entire transformer
        if getattr(self.base_config, 'enable_cross_attn', False) \
           and hasattr(self, 'transformer') \
           and isinstance(self.transformer, CrossAttentionTransformer):

            # cross_adapters is a ModuleDict; yield it so its parameters are unfrozen
            if hasattr(self.transformer, 'cross_adapters') and self.transformer.cross_adapters is not None:
                yield self.transformer.cross_adapters

            # gates is a ParameterDict (no .modules()), but named_parameters() works.
            # Wrap it with a lightweight module so .parameters() can iterate them uniformly.
            class _GateWrapper(nn.Module):
                def __init__(self, pd: nn.ParameterDict):
                    super().__init__()
                    # Register parameters directly so .parameters() can see them
                    for k, v in pd.items():
                        self.register_parameter(k, v)
            if hasattr(self.transformer, 'gates') and self.transformer.gates is not None:
                yield _GateWrapper(self.transformer.gates)

    def _freeze_all_parameters(self):
        for p in self.parameters():
            p.requires_grad = False

    def _unfreeze_new_modules(self):
        for module in self._iter_new_modules():
            for p in module.parameters():
                p.requires_grad = True

    def unfreeze_transformer_layer(self, layer_idx: int):
        """
        Unfreeze a specific transformer layer.
        
        Args:
            layer_idx: Index of the layer to unfreeze (0-based, from bottom to top)
                      For a model with 12 layers, indices are 0-11, where 11 is the top layer.
        """
        # Get transformer blocks from the wrapped transformer
        if hasattr(self.transformer, 'base_transformer'):
            base_transformer = self.transformer.base_transformer
        else:
            base_transformer = self.transformer
        
        # Try to find blocks/layers
        blocks = None
        if hasattr(base_transformer, 'blocks'):
            blocks = base_transformer.blocks
        elif hasattr(base_transformer, 'layers'):
            blocks = base_transformer.layers
        else:
            # Try nested structures
            for attr_name in ['transformer', 'model', 'encoder', 'decoder']:
                if hasattr(base_transformer, attr_name):
                    nested = getattr(base_transformer, attr_name)
                    if hasattr(nested, 'blocks'):
                        blocks = nested.blocks
                        break
                    elif hasattr(nested, 'layers'):
                        blocks = nested.layers
                        break
        
        if blocks is None:
            logger.warning(f"Could not find transformer blocks/layers to unfreeze layer {layer_idx}")
            return
        
        # Check if layer index is valid
        if layer_idx < 0 or layer_idx >= len(blocks):
            logger.warning(f"Layer index {layer_idx} is out of range (0-{len(blocks)-1})")
            return
        
        # Unfreeze the layer
        layer = blocks[layer_idx]
        num_unfrozen = 0
        for param in layer.parameters():
            param.requires_grad = True
            num_unfrozen += 1
        
        logger.info(f"✅ Unfroze transformer layer {layer_idx} ({num_unfrozen} parameters, total layers: {len(blocks)})")
        
        # Also unfreeze the corresponding cross-attention adapter and gate if they exist
        if hasattr(self.transformer, 'cross_adapters') and self.transformer.cross_adapters is not None:
            if str(layer_idx) in self.transformer.cross_adapters:
                adapter = self.transformer.cross_adapters[str(layer_idx)]
                adapter_num_unfrozen = 0
                for param in adapter.parameters():
                    param.requires_grad = True
                    adapter_num_unfrozen += 1
                logger.info(f"✅ Unfroze cross-attention adapter for layer {layer_idx} ({adapter_num_unfrozen} parameters)")
        
        if hasattr(self.transformer, 'gates') and self.transformer.gates is not None:
            if str(layer_idx) in self.transformer.gates:
                gate = self.transformer.gates[str(layer_idx)]
                gate.requires_grad = True
                logger.info(f"✅ Unfroze gate for layer {layer_idx}")

    def _compute_beta_kl(self) -> float:
        """
        Compute current beta_kl value with annealing.
        
        Returns:
            Current beta_kl value (float)
        """
        config = self.base_config
        
        # Check if annealing is configured (anneal_steps > 0 means annealing is enabled)
        if hasattr(config, 'beta_kl_anneal_steps') and config.beta_kl_anneal_steps > 0:
            # Use annealing
            if not self.training and hasattr(config, 'beta_kl_eval') and config.beta_kl_eval is not None:
                # Use fixed beta for evaluation
                return float(config.beta_kl_eval)
            
            # Get current step
            current_step = getattr(self, '_current_step', 0)
            anneal_steps = config.beta_kl_anneal_steps
            beta_init = config.beta_kl_init
            beta_final = config.beta_kl_final
            anneal_type = getattr(config, 'beta_kl_anneal_type', 'linear')
            
            # Clamp current_step to anneal_steps
            progress = min(current_step / anneal_steps, 1.0)
            
            if anneal_type == 'linear':
                # Linear interpolation: beta = beta_init + (beta_final - beta_init) * progress
                beta = beta_init + (beta_final - beta_init) * progress
            elif anneal_type == 'cosine':
                # Cosine annealing: smooth transition
                import math
                beta = beta_init + (beta_final - beta_init) * (1.0 - math.cos(math.pi * progress)) / 2.0
            elif anneal_type == 'cyclical':
                # Cyclical annealing: cycle between init and final
                import math
                cycle_progress = (math.sin(2.0 * math.pi * progress) + 1.0) / 2.0  # 0 to 1
                beta = beta_init + (beta_final - beta_init) * cycle_progress
            else:
                # Unknown type, fall back to linear
                beta = beta_init + (beta_final - beta_init) * progress
            
            return float(beta)
        else:
            # Use fixed beta (legacy behavior)
            return float(getattr(config, 'beta_kl', 0.1))
    
    def set_training_step(self, step: int):
        """
        Set current training step for KL annealing.
        Should be called by Trainer callback or training loop.
        
        Args:
            step: Current global training step
        """
        self._current_step = int(step)
    
    def _maybe_setup_finetune_freeze_only(self):
        """
        If base_config.train_new_modules_only is True (default),
        freeze all parameters, then unfreeze only new modules
        (adapters/gates + encoders/fusion/cond_proj).
        """
        if getattr(self.base_config, 'train_new_modules_only', True):
            self._freeze_all_parameters()
            self._unfreeze_new_modules()

    def _param_belongs_to_new_module(self, param_name: str) -> bool:
        """
        Check by name whether a parameter belongs to new modules. Match only:
          - transformer.cross_adapters.*
          - transformer.gates.*
          - ligand_encoder.*, protein_encoder.*, condition_fusion.*, cond_proj.*
        """
        prefixes = [
            'transformer.cross_adapters',
            'transformer.gates',
            'ligand_encoder',
            'protein_encoder',
            'condition_fusion',
            'cond_proj',
        ]
        return any(param_name.startswith(pf) for pf in prefixes)

    def autodetect_lora_targets(self) -> List[str]:
        """
        Auto-detect common linear layer short-names for LoRA injection; keep original logic.
        """
        common_names = [
            'Wqkv', 'out_proj',
            'q_proj', 'k_proj', 'v_proj', 'o_proj',
            'Wq', 'Wk', 'Wv', 'Wo',
            'qkv_proj'
        ]
        present: List[str] = []
        for name, module in self.named_modules():
            # only consider leaf linear layers
            if isinstance(module, nn.Linear):
                short = name.split('.')[-1]
                if short in common_names and short not in present:
                    present.append(short)
        # Fallback to safe defaults if nothing found
        if not present:
            present = ['Wqkv', 'out_proj']
        return present

    def print_trainable_summary(self, max_lines: int = 30):
        """
        Print a concise summary of trainable parameters to verify that only
        adapters/gates + encoders/fusion/cond_proj are being trained.
        """
        total = sum(1 for _ in self.parameters())
        trainable = [(n, p.numel()) for n, p in self.named_parameters() if p.requires_grad]
        tot_train = sum(x[1] for x in trainable)
        print(f"[trainable] tensors: {len(trainable)} / {total}, params: {tot_train:,}")
        head = trainable[:max_lines]
        for n, k in head:
            print("  +", n, f"({k})")
        if len(trainable) > max_lines:
            print(f"  ... and {len(trainable) - max_lines} more")
    
    def debug_new_modules(self):
        """Debug method to check which modules are considered 'new'."""
        print("=== Debug: New Modules ===")
        print("Modules returned by _iter_new_modules():")
        for i, module in enumerate(self._iter_new_modules()):
            print(f"  {i+1}. {type(module).__name__}: {module}")
            if hasattr(module, 'parameters'):
                param_count = sum(p.numel() for p in module.parameters())
                trainable_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
                print(f"     Total params: {param_count:,}, Trainable: {trainable_count:,}")
        
        print("\nAll modules with 'encoder' in name:")
        for name, module in self.named_modules():
            if 'encoder' in name.lower():
                param_count = sum(p.numel() for p in module.parameters())
                trainable_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
                print(f"  {name}: {param_count:,} total, {trainable_count:,} trainable")
        
        print("\nAll modules with 'fusion' in name:")
        for name, module in self.named_modules():
            if 'fusion' in name.lower():
                param_count = sum(p.numel() for p in module.parameters())
                trainable_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
                print(f"  {name}: {param_count:,} total, {trainable_count:,} trainable")

    # TODO: here we ignore attention_mask to make it compatible with HF trainer. The MHA in flash-attention should
    #  be reimplement and integrate attention_mask like here:
    #  https://github.com/huggingface/transformers/blob/0864dd3beb238b7bec3528a3d1d6c17a28f51a51/src/transformers/models/llama/modeling_llama.py#L536
    def forward(self, input_ids, attention_mask: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None, return_dict: Optional[bool] = None,
                position_ids=None, inference_params=None, num_last_tokens=0,
                cond_tokens: Optional[torch.Tensor] = None,
                cond_attention_mask: Optional[torch.Tensor] = None,
                # Condition feature inputs
                pocket_vec: Optional[torch.Tensor] = None,
                evo_vec: Optional[torch.Tensor] = None,
                ifp: Optional[torch.Tensor] = None,
                ligand_vec: Optional[torch.Tensor] = None,
                **loss_kwargs):
        """
                input_ids: (batch, seqlen) int tensor
                inference_params: for generation. Adapted from Megatron-LM (and Apex)
                https://github.com/NVIDIA/apex/blob/3ff1a10f72ec07067c4e44759442329804ac5162/apex/transformer/testing/standalone_transformer_lm.py#L470
                num_last_tokens: if > 0, only return the logits for the last n tokens
                cond_tokens: (batch, cond_len, hidden_size) tensor for cross-attention
                cond_attention_mask: (batch, cond_len) tensor for condition masking
                """
        assert (
                input_ids.ndim == 2
        ), f"Expected `input_ids` to have shape [b, slen], but got shape {input_ids.shape}"
        b, slen = input_ids.shape # b = batch size, slen = sequence length
        ### position_ids: (batch, seqlen) int tensor, hidden_states: (batch, seqlen, hidden_size)
        # Build cond_tokens from provided condition features using the new encoder architecture
        if (
            self.base_config.enable_cross_attn
            and cond_tokens is None
            and (pocket_vec is not None or evo_vec is not None or ifp is not None or ligand_vec is not None)
        ):
            bsz = input_ids.size(0)
            device = input_ids.device
            dtype = torch.float32
            
            # Process protein condition: pocket_vec + evo_vec -> protein_condition
            if pocket_vec is not None and evo_vec is not None:
                protein_condition = self.protein_encoder(pocket_vec, evo_vec)
            else:
                # Fallback to dummy protein condition
                print(f"Warning: No protein condition provided, breaking the training loop")
                cond_latent_dim = self.base_config.cond_tokens_len
                protein_condition = torch.zeros(bsz, cond_latent_dim, dtype=dtype, device=device)
                raise ValueError("No protein condition provided, breaking the training loop")
            
            # Process ligand condition: ifp + ligand_vec -> mu, logvar, then sample z
            if ifp is not None and ligand_vec is not None:
                mu, sigma, logvar = self.ligand_encoder(ifp, ligand_vec, return_logvar=True)
                # Sample z from posterior using reparameterization trick
                eps = torch.randn_like(mu)
                z = mu + (0.5 * logvar).exp() * eps
            else:
                # Fallback: sample z from standard normal (no ligand condition)
                print(f"Warning: No ligand condition provided, sampling z from standard normal")
                cond_latent_dim = self.base_config.cond_tokens_len
                z = torch.randn(bsz, cond_latent_dim, dtype=dtype, device=device)
            
            # Fuse protein and ligand conditions using sampled z
            fused_condition = self.condition_fusion(protein_condition, z)
            
            # Project to cond_tokens: [B, Lc, d] where Lc = cond_tokens_len
            # cond_vector = torch.cat([protein_condition, fused_condition], dim=-1)  # [B, 2*dz]
            cond_vector = fused_condition
            cond_tokens = self.cond_proj(cond_vector.unsqueeze(-1))  # [B, Lc (2*dz), d]
            cond_attention_mask = torch.zeros(cond_tokens.size(0), cond_tokens.size(1), dtype=torch.bool, device=device)

        # Build args for calling the underlying transformer. Only pass cross-attention
        # kwargs when cross-attention is enabled and the wrapped transformer supports it.
        transformer_kwargs = {
            "position_ids": position_ids,
            "inference_params": inference_params,
        }
        if (
            getattr(self.base_config, "enable_cross_attn", False)
            and hasattr(self, "transformer")
            and hasattr(self.transformer, "enable_cross_attn")
            and getattr(self.transformer, "enable_cross_attn", False)
        ):
            transformer_kwargs.update({
                "cond_tokens": cond_tokens,
                "cond_attention_mask": cond_attention_mask,
            })
        hidden_states = self.transformer(input_ids, **transformer_kwargs)
        if inference_params is not None:
            assert hidden_states.ndim == 3, "sequence_parallel is not supported in generation mode"
        ### optional extract the last num_last_tokens tokens from the hidden states
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        ### Optional linear projection layer
        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)
        ### Optional output scaling, now the size is still (batch, seqlen, hidden_size)
        if self.output_scale != 1.0:
            hidden_states = hidden_states * self.output_scale
        ### Convert hidden states to vocabulary logits (batch, seqlen, vocab_size) for calculating loss.
        if not self.norm_head:
            lm_logits = self.lm_head(hidden_states)
        else:
            lm_head_weight = F.normalize(self.lm_head.weight)
            lm_logits = F.linear(hidden_states, lm_head_weight, bias=self.lm_head.bias)

        # Compute NLL (language modeling loss) if labels provided
        ### lm_logits: (batch, seqlen, vocab_size), labels: (batch, seqlen): Content: input_ids shifted left by 1
        ### position, with padding set to -100. Generated by tokenizer.
        ### Although calculate attention on padding, for loss calculation they are ignored. Thus the efficicy degrade.
        nll_loss = None
        if labels is not None:
            nll_loss = self.loss_function(
            logits=lm_logits,
                labels=labels,
                vocab_size=self.base_config.vocab_size,
                **loss_kwargs,
            )

        # Compute KL(q_phi || N(0, I)) from ligand encoder if condition features are provided
        kl_loss = None
        if ifp is not None and ligand_vec is not None and hasattr(self, 'ligand_encoder'):
            # Use KL from the ligand encoder's posterior
            mu, sigma, logvar = self.ligand_encoder(ifp, ligand_vec, return_logvar=True)
            kl_per_dim= 0.5 * (logvar.exp() + mu.pow(2) - 1.0 - logvar)
            
            # fb = getattr(self, "free_bits", 0.5)
            # if fb and fb > 0:
            #     kl_per_dim = torch.clamp(kl_per_dim, min=fb)
            
            kl_loss = kl_per_dim.sum(dim=-1).mean()  # Average per sample (sum over latent_dim)
            
            # Normalize KL loss to per-token scale (to match NLL loss normalization)
            # This ensures KL loss and NLL loss have similar scales
            if labels is not None:
                # Calculate average sequence length (excluding padding tokens with label -100)
                seq_lengths = (labels != -100).sum(dim=1).float()  # [batch_size]
                avg_seq_len = seq_lengths.mean()  # Average sequence length
                if avg_seq_len > 0:
                    kl_loss = kl_loss / avg_seq_len
                    # Now KL loss is normalized to "per token" scale, matching NLL loss
        else:
            kl_loss = torch.tensor(0.0, dtype=lm_logits.dtype, device=lm_logits.device)

        # Compute beta_kl with annealing if configured
        beta = self._compute_beta_kl()
        beta_tensor = torch.tensor(beta, dtype=lm_logits.dtype, device=lm_logits.device)

        # Total loss: nll + beta * kl (if nll is None, just return beta*kl to avoid returning None)
        loss = None
        if nll_loss is not None:
            loss = nll_loss + beta_tensor * kl_loss
        else:
            loss = beta_tensor * kl_loss
        
        # Store losses for monitoring (detached to avoid gradient issues)
        # These will be accessed by LossMonitoringCallback
        if self.training:
            self._last_nll_loss = nll_loss.item() if nll_loss is not None else 0.0
            self._last_kl_loss = (beta_tensor * kl_loss).item() if kl_loss is not None else 0.0
            self._last_kl_loss_unscaled = kl_loss.item() if kl_loss is not None else 0.0
            self._last_beta = beta
        
        ### Return standard HF output format.
        return CausalLMOutput(loss=loss, logits=lm_logits, hidden_states=hidden_states)

    @property
    def loss_function(self):
        if getattr(self.base_config, "loss_type", None) is not None:
            loss_type = self.base_config.loss_type
        else:
            loss_type = self.__class__.__name__
            if loss_type not in LOSS_MAPPING:
                loss_groups = f"({'|'.join(LOSS_MAPPING)})"
                loss_type = re.findall(loss_groups, self.__class__.__name__)
                if len(loss_type) > 0:
                    loss_type = loss_type[0]
                else:
                    loss_type = None
        if loss_type is None or loss_type not in LOSS_MAPPING and getattr(self.base_config, "loss_type",
                                                                          None) is not None:
            print(
                f"`loss_type={loss_type}` was set in the base_config but it is unrecognised."
                f"Using the default loss: `ForCausalLMLoss`."
            )
            loss_type = "ForCausalLM"
        return LOSS_MAPPING[loss_type]

    def save_pretrained(
            self,
            save_directory: Union[str, os.PathLike],
            is_main_process: bool = True,
            state_dict: Optional[dict] = None,
            safe_serialization: bool = False,
            **kwargs,
    ):

        if safe_serialization:
            raise ImportError("`safe_serialization` is not implemented yet`.")

        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return
        os.makedirs(save_directory, exist_ok=True)
        # Save the config
        if is_main_process:
            self.base_config.save_pretrained(save_directory)

        # Save the model
        if state_dict is None:
            # Only save the model itself if we are using distributed training
            model_to_save = unwrap_model(self)
            state_dict = model_to_save.state_dict()

        weights_name = SAFE_WEIGHTS_NAME if safe_serialization else WEIGHTS_NAME
        torch.save(state_dict, os.path.join(save_directory, weights_name))

        # find the file where NovoMolGen is defined
        src = inspect.getsourcefile(type(self))
        if src:
            dst = os.path.join(save_directory, os.path.basename(src))
            shutil.copy(src, dst)

    @classmethod
    def from_pretrained(
        cls, 
        pretrained_model_name_or_path, 
        checkpoint_path: str = "",
        config: Optional[Union[NovoMolGenConfig, str, os.PathLike]] = None,
        **kwargs,
        ):
        if config is None:
            config = NovoMolGenConfig.from_pretrained(pretrained_model_name_or_path, checkpoint_path=checkpoint_path, **kwargs)
        model = cls(config)

        if os.path.exists(pretrained_model_name_or_path):
                state_dict = torch.load(os.path.join(pretrained_model_name_or_path, checkpoint_path, WEIGHTS_NAME))
        else:
            state_dict = state_dict_from_pretrained(pretrained_model_name_or_path, checkpoint_path=checkpoint_path, **kwargs)

        model.load_state_dict(state_dict, strict=False)

        return model

    def sample(
            self,
            tokenizer,
            batch_size: int = 4,
            max_length: int = 64,
            temperature: float = 1.0,
            top_k: int = 50,
            top_p: float = 0.95,
            device: torch.device = torch.device("cuda"),
    ):
        """
        Generate a batch of sequences from the model.

        Returns a dictionary with up to three keys:
        {
            "<mol_type>": <list of raw sequences in that moltype>,
            "sequences": <torch.LongTensor of valid token IDs>
        }
        """
        input_ids = tokenizer.encode("", return_tensors="pt").to(device)
        # Repeat the prompt for the desired batch size
        input_ids = input_ids.repeat_interleave(batch_size, dim=0)
        # If the tokenizer includes an EOS token for an empty prompt, we remove it.
        if input_ids.shape[1] > 1:
            input_ids = input_ids[:, :-1]

        generation_output = self.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
        )

        sequences = self._filter_tokens_after_eos(
            generation_output.sequences, eos_id=tokenizer.eos_token_id
        )

        decoded_strings = tokenizer.batch_decode(sequences, skip_special_tokens=True)
        decoded_strings = [s.replace(" ", "") for s in decoded_strings]

        result = {
            self.mol_type: decoded_strings,
            "sequences": sequences,
        }
        return result

    @staticmethod
    def _filter_tokens_after_eos(sequences, eos_id):
        output = copy.deepcopy(sequences)
        for i in range(sequences.size(0)):
            row = sequences[i]
            eos_position = (row == eos_id).nonzero()
            if eos_position.numel() > 0:
                eos_position = eos_position[0, 0].item()  # Get the index of the first occurrence
                output[i, eos_position + 1:] = eos_id
        return output

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        # HF's GenerationMixin would normally do more, but for a basic LM this usually suffices:
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        
        # Only pass cross-attention conditions if cross-attn is enabled
        if getattr(self.base_config, "enable_cross_attn", False): # Logic: If "enable_cross_attn" exists, return true. Else false.
            if "cond_tokens" in kwargs:
                model_inputs["cond_tokens"] = kwargs["cond_tokens"]
            if "cond_attention_mask" in kwargs:
                model_inputs["cond_attention_mask"] = kwargs["cond_attention_mask"]
        
        return model_inputs

    @torch.inference_mode()
    def generate_with_condition(
        self,
        *,
        input_ids: Optional[torch.LongTensor] = None,
        cond_tokens: Optional[torch.Tensor] = None,
        cond_attention_mask: Optional[torch.Tensor] = None,
        sample_posterior: bool = False,
        append: bool = True,
        prefix_safe: Optional[str] = None,
        tokenizer = None,
        # Condition feature inputs
        pocket_vec: Optional[torch.Tensor] = None,
        evo_vec: Optional[torch.Tensor] = None,
        ifp: Optional[torch.Tensor] = None,
        ligand_vec: Optional[torch.Tensor] = None,
        **gen_kwargs,
    ):
        """
        Helper for conditional generation using the new encoder architecture.

        Args:
            input_ids: Optional prompt ids [B, T0] (required if prefix_safe not provided)
            cond_tokens: Optional pre-built condition tokens [B, Lc0, d] to be appended to
            cond_attention_mask: Optional cond mask [B, Lc0] (True=padding)
            sample_posterior: If True, sample z from posterior; else use mu
            append: If True, append new condition tokens to provided cond_tokens; else replace
            prefix_safe: Optional SAFE string prefix to tokenize and use as initial input_ids
            tokenizer: Required if prefix_safe is provided, used to tokenize the prefix
            pocket_vec: Protein pocket embedding [B, 512]
            evo_vec: ESM-2 evolutionary embedding [B, 1280]
            ifp: Interaction fingerprint [B, 16384]
            ligand_vec: Ligand molecular representation [B, 1536]
            **gen_kwargs: Passed through to `generate()`

        Returns:
            If prefix_safe is provided: dict with keys 'full_safe', 'generated_span', 'sequences'
            Otherwise: torch.LongTensor with generated sequences
        """
        if not getattr(self.base_config, "enable_cross_attn", False):
            raise ValueError("Cross-attention is disabled in config; enable_cross_attn must be True for conditioned generation.")

        # Handle prefix_safe input
        if prefix_safe is not None:
            if tokenizer is None:
                raise ValueError("tokenizer must be provided when prefix_safe is specified.")
            if input_ids is not None:
                raise ValueError("Cannot provide both input_ids and prefix_safe.")
            
            # Tokenize the prefix
            prefix_tokens = tokenizer.encode(prefix_safe, return_tensors="pt")
            input_ids = prefix_tokens
        elif input_ids is None:
            raise ValueError("Either input_ids or prefix_safe must be provided for generation.")

        device = input_ids.device
        bsz = input_ids.size(0)

        # Process condition features using the new encoder architecture
        # Get model dtype to ensure type compatibility with cross-attention
        model_dtype = next(self.parameters()).dtype
        cond_latent_dim = self.base_config.cond_tokens_len
        
        if pocket_vec is not None and evo_vec is not None and ifp is not None and ligand_vec is not None:
            # Use new encoder architecture
            # Ensure input features are on correct device and dtype
            pocket_vec = pocket_vec.to(device).to(model_dtype)
            evo_vec = evo_vec.to(device).to(model_dtype)
            ifp = ifp.to(device).to(model_dtype)
            ligand_vec = ligand_vec.to(device).to(model_dtype)
            
            protein_condition = self.protein_encoder(pocket_vec, evo_vec)
            mu, sigma, logvar = self.ligand_encoder(ifp, ligand_vec, return_logvar=True)
            
            # Sample z from posterior
            if sample_posterior:
                z = torch.randn(bsz, cond_latent_dim, dtype=model_dtype, device=device)
            else:
                # eps = torch.randn_like(mu)
                # z = mu + (0.5 * logvar).exp() * eps
                z = mu
            
            # Fuse conditions using sampled z
            fused_condition = self.condition_fusion(protein_condition, z)
            
            # Build condition tokens
            # cond_vector = torch.cat([z, fused_condition], dim=-1)  # [B, 2*dz]
            cond_vector = fused_condition
            new_cond_tokens = self.cond_proj(cond_vector.unsqueeze(-1))  # [B, 2*dz, d]
            # Ensure cond_tokens have correct dtype for cross-attention (critical for flash_attn)
            new_cond_tokens = new_cond_tokens.to(model_dtype)
            new_cond_mask = torch.zeros(bsz, new_cond_tokens.size(1), dtype=torch.bool, device=device)
        
        elif pocket_vec is not None and evo_vec is not None:
            # Ensure input features are on correct device and dtype
            pocket_vec = pocket_vec.to(device).to(model_dtype)
            evo_vec = evo_vec.to(device).to(model_dtype)
            protein_condition = self.protein_encoder(pocket_vec, evo_vec)
            cond_latent_dim = self.base_config.cond_tokens_len
            z = torch.randn(bsz, cond_latent_dim, dtype=model_dtype, device=device)
            fused_condition = self.condition_fusion(protein_condition, z)
            new_cond_tokens = self.cond_proj(fused_condition.unsqueeze(-1))  # [B, dz, d]
            # Ensure cond_tokens have correct dtype for cross-attention
            new_cond_tokens = new_cond_tokens.to(model_dtype)
            new_cond_mask = torch.zeros(bsz, new_cond_tokens.size(1), dtype=torch.bool, device=device)
        else:
            # Fallback: use dummy conditions if not all features provided
            print("no condition features provided")
            cond_latent_dim = self.base_config.cond_tokens_len
            # Use the same dtype as the model parameters
            model_dtype = next(self.parameters()).dtype
            z = torch.randn(bsz, cond_latent_dim, dtype=model_dtype, device=device)
            protein_condition = torch.zeros(bsz, cond_latent_dim, dtype=model_dtype, device=device)
            fused_condition = self.condition_fusion(protein_condition, z)
            
            # Build condition tokens
            # cond_vector = torch.cat([protein_condition, fused_condition], dim=-1)  # [B, 2*dz]
            cond_vector = fused_condition
            new_cond_tokens = self.cond_proj(cond_vector.unsqueeze(-1))  # [B, 2*dz, d]
            new_cond_mask = torch.zeros(bsz, new_cond_tokens.size(1), dtype=torch.bool, device=device)

        # Merge with existing condition tokens if provided
        if cond_tokens is not None and append:
            assert cond_tokens.dim() == 3 and cond_tokens.size(0) == bsz, "cond_tokens must be [B, Lc0, d]"
            cond_tokens = torch.cat([cond_tokens.to(device).to(model_dtype), new_cond_tokens], dim=1)
            if cond_attention_mask is None:
                cond_attention_mask = torch.zeros(bsz, cond_tokens.size(1), dtype=torch.bool, device=device)
            else:
                assert cond_attention_mask.dim() == 2 and cond_attention_mask.size(0) == bsz, "cond_attention_mask must be [B, Lc0]"
                cond_attention_mask = torch.cat([cond_attention_mask.to(device), new_cond_mask], dim=1)
        else:
            cond_tokens = new_cond_tokens
            cond_attention_mask = new_cond_mask

        # For now, we'll use a simple approach: generate without conditions
        # TODO: Implement proper conditional generation that threads cond_tokens through the generation process
        # This requires overriding the generation loop to pass cond_tokens to each forward call
        
        # Store condition tokens for use in forward calls during generation
        if hasattr(self, 'transformer') and hasattr(self.transformer, '_current_cond_tokens'):
            self.transformer._current_cond_tokens = cond_tokens
            self.transformer._current_cond_attention_mask = cond_attention_mask
        
        # Don't pass cond_tokens/cond_attention_mask to base generate() - flash-attn doesn't support them
        generated_sequences = self.generate(
            input_ids=input_ids,
            **gen_kwargs,
        )
        
        # Clear condition tokens after generation
        if hasattr(self, 'transformer') and hasattr(self.transformer, '_current_cond_tokens'):
            self.transformer._current_cond_tokens = None
            self.transformer._current_cond_attention_mask = None
        
        # Handle prefix_safe case: return full SAFE string and generated span
        if prefix_safe is not None:
            # Decode the full generated sequences
            full_safe_strings = tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)
            
            # Extract only the newly generated part (after the prefix)
            prefix_length = input_ids.size(1)
            generated_span_sequences = generated_sequences[:, prefix_length:]
            generated_span_strings = tokenizer.batch_decode(generated_span_sequences, skip_special_tokens=True)
            
            return {
                'full_safe': full_safe_strings,
                'generated_span': generated_span_strings,
                'sequences': generated_sequences
            }
        else:
            return generated_sequences