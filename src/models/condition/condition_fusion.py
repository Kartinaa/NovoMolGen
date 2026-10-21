"""Condition fusion module for combining protein and ligand conditions.

This module provides a clean interface to combine protein condition vectors
and ligand posterior parameters into a unified condition representation
for the NovoMolGen model.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionFusion(nn.Module):
    """Fuse protein and ligand conditions using cross-attention.
    
    Takes protein condition vector [B, protein_dim] and sampled ligand latent
    vector z [B, ligand_dim], and produces a unified condition representation [B, out_dim].
    
    The ligand z vector should be sampled from the posterior q_φ(z|x) when ligand
    information is available, or from the prior N(0,I) when no ligand information
    is provided.
    
    Args:
        protein_dim: int, dimension of protein condition vector
        ligand_dim: int, dimension of ligand latent space (z)
        out_dim: int, output dimension for fused condition
        common_dim: int, shared projection dimension (default: max(protein_dim, ligand_dim))
        dropout: float, dropout probability
        use_cross_attn: bool, whether to use cross-attention fusion (default: True)
    """
    
    def __init__(
        self,
        protein_dim: int,
        ligand_dim: int,
        out_dim: int,
        common_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_cross_attn: bool = True,
    ) -> None:
        super().__init__()
        self.protein_dim = protein_dim
        self.ligand_dim = ligand_dim
        self.out_dim = out_dim
        self.common_dim = common_dim or max(protein_dim, ligand_dim)
        self.use_cross_attn = use_cross_attn
        
        # Projections to common space
        self.protein_proj = nn.Linear(protein_dim, self.common_dim)
        self.ligand_proj = nn.Linear(ligand_dim, self.common_dim)
        
        # Normalization and dropout
        self.ln_protein = nn.LayerNorm(self.common_dim)
        self.ln_ligand = nn.LayerNorm(self.common_dim)
        self.dropout = nn.Dropout(dropout)
        
        if self.use_cross_attn:
            # Cross-attention fusion
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=self.common_dim,
                num_heads=1,
                dropout=dropout,
                batch_first=True
            )
            
            # Final projection to output dimension
            self.final_proj = nn.Sequential(
                nn.Linear(self.common_dim, self.common_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.common_dim * 2, self.out_dim)
            )
        else:
            # Simple concatenation fusion
            self.fusion_mlp = nn.Sequential(
                nn.Linear(self.common_dim * 2, self.common_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.common_dim * 2, self.out_dim)
            )
        
        # Initialize weights
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        protein_condition: torch.Tensor,
        ligand_z: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass to fuse protein and ligand conditions.
        
        Args:
            protein_condition: [B, protein_dim] protein condition vector
            ligand_z: [B, ligand_dim] sampled ligand latent vector (from posterior or prior)
            
        Returns:
            fused_condition: [B, out_dim] unified condition representation
        """
        # Validate input shapes
        assert protein_condition.dim() == 2, f"protein_condition must be 2D, got {protein_condition.dim()}D"
        assert ligand_z.dim() == 2, f"ligand_z must be 2D, got {ligand_z.dim()}D"
        
        B = protein_condition.size(0)
        assert ligand_z.size(0) == B, "Batch size mismatch between protein and ligand"
        
        # Project to common space
        protein_proj = self.dropout(self.ln_protein(self.protein_proj(protein_condition)))
        ligand_proj = self.dropout(self.ln_ligand(self.ligand_proj(ligand_z)))
        
        if self.use_cross_attn:
            # Cross-attention fusion: protein as query, ligand as key/value
            protein_seq = protein_proj.unsqueeze(1)  # [B, 1, common_dim]
            ligand_seq = ligand_proj.unsqueeze(1)    # [B, 1, common_dim]
            
            # Cross-attention: protein attends to ligand
            attn_out, _ = self.cross_attn(
                query=protein_seq,
                key=ligand_seq,
                value=ligand_seq,
                need_weights=False
            )
            
            # Combine protein and attention output
            fused = protein_proj + attn_out.squeeze(1)  # [B, common_dim]
            output = self.final_proj(fused)
        else:
            # Simple concatenation fusion
            concat_features = torch.cat([protein_proj, ligand_proj], dim=-1)
            output = self.fusion_mlp(concat_features)
        
        return output


def create_condition_fusion(
    protein_dim: int,
    ligand_dim: int,
    out_dim: int,
    **kwargs
) -> ConditionFusion:
    """Factory function to create a condition fusion module."""
    return ConditionFusion(
        protein_dim=protein_dim,
        ligand_dim=ligand_dim,
        out_dim=out_dim,
        **kwargs
    )
