"""Pocket + Evolutionary info fusion module for protein conditioning.

Implements a clean, production-ready fusion block that combines two vectors:
  - x_pocket: [B, 512] (pocket-general info from Uni-Mol)
  - x_evo:    [B, 1280] (evolutionary info from ESM-2)

The fusion projects both inputs to a shared space, applies LayerNorm + Dropout,
builds early-fusion features concat([a, b, a*b, |a-b|]), and passes them through
an MLP to produce an output embedding. Optionally, a gated residual branch injects
signal from the difference (a - b).
"""

from typing import Optional

import torch
import torch.nn as nn


class ProteinConditionEncoder(nn.Module):
    """Fuse pocket and evolutionary vectors into a protein condition embedding.

    Fusion pipeline (deterministic, no randomness in forward):
      1) Linear projections to a shared dimension (common_dim)
      2) LayerNorm + Dropout
      3) Early-fusion feature: concat([a, b, a*b, |a-b|])
      4) MLP → out_dim (default same as common_dim)
      5) Optional gated residual: gate(sigmoid) from concat([a,b]) multiplied by
         a projected difference (a - b) injected to the output

    Args:
      pocket_dim: int, dimension of x_pocket (default 512)
      evo_dim: int, dimension of x_evo (default 1280)
      common_dim: int, shared projection size for both inputs (default 512)
      out_dim: Optional[int], output embedding size. If None, equals common_dim
      dropout: float, dropout probability
      use_gated_residual: bool, whether to enable gated residual branch
    """

    def __init__(
        self,
        pocket_dim: int = 512,
        evo_dim: int = 1280,
        common_dim: int = 512,
        out_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_gated_residual: bool = True,
    ) -> None:
        super().__init__()
        self.pocket_dim = pocket_dim
        self.evo_dim = evo_dim
        self.common_dim = common_dim
        self.out_dim = out_dim or common_dim
        self.use_gated_residual = use_gated_residual

        # Projections to a shared space
        self.proj_pocket = nn.Linear(pocket_dim, common_dim)
        self.proj_evo = nn.Linear(evo_dim, common_dim)

        # Normalization + regularization
        self.ln_p = nn.LayerNorm(common_dim)
        self.ln_e = nn.LayerNorm(common_dim)
        self.drop = nn.Dropout(dropout)

        # Fusion MLP
        fuse_in = 4 * common_dim
        hidden = max(512, 2 * self.out_dim)
        self.fuse_mlp = nn.Sequential(
            nn.Linear(fuse_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.out_dim),
        )

        # Gated residual over the difference (a - b)
        if self.use_gated_residual:
            self.diff_proj = nn.Linear(common_dim, self.out_dim)
            self.gate_net = nn.Sequential(
                nn.Linear(2 * common_dim, common_dim),
                nn.GELU(),
                nn.Linear(common_dim, self.out_dim),
            )

        # Initialize linears (Xavier uniform, zero biases)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x_pocket: torch.Tensor, x_evo: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
          x_pocket: [B, 512] pocket-general vector
          x_evo:    [B, 1280] evolutionary vector

        Returns:
          z: [B, out_dim] fused condition embedding
        """
        assert x_pocket.ndim == 2 and x_evo.ndim == 2, "Inputs must be [B, D]."
        assert x_pocket.size(0) == x_evo.size(0), "Batch sizes must match."
        assert x_pocket.size(1) == 512, f"Expected pocket dim 512, got {x_pocket.size(1)}."
        assert x_evo.size(1) == 1280, f"Expected evo dim 1280, got {x_evo.size(1)}."

        # Project to shared space
        a = self.proj_pocket(x_pocket)  # [B, C]
        b = self.proj_evo(x_evo)        # [B, C]

        # Normalize + Dropout
        a = self.drop(self.ln_p(a))
        b = self.drop(self.ln_e(b))

        # Early fusion features
        fused = torch.cat([a, b, a * b, torch.abs(a - b)], dim=-1)  # [B, 4C]
        z = self.fuse_mlp(fused)                                     # [B, out_dim]

        # Optional gated residual from (a - b)
        if self.use_gated_residual:
            gate = torch.sigmoid(self.gate_net(torch.cat([a, b], dim=-1)))  # [B, out_dim]
            z = z + gate * self.diff_proj(a - b)

        return z


def create_protein_encoder(
    latent_dim: int,
    protein_input_dim: int,  # This parameter is kept for backward compatibility but ignored
    hidden_dim: int = 128,
    dropout: float = 0.1,
    pocket_dim: int = 512,
    evo_dim: int = 1280,
    common_dim: int = 512,
    out_dim: Optional[int] = None,
    use_gated_residual: bool = True,
) -> ProteinConditionEncoder:
    """Factory function to create a protein condition encoder.
    
    Args:
        latent_dim: Target latent dimension (used as out_dim if out_dim is None)
        protein_input_dim: Ignored for backward compatibility
        hidden_dim: Hidden dimension for MLP (ignored, kept for compatibility)
        dropout: Dropout probability
        pocket_dim: Dimension of pocket embedding
        evo_dim: Dimension of evolutionary embedding
        common_dim: Shared projection dimension
        out_dim: Output dimension (defaults to latent_dim)
        use_gated_residual: Whether to use gated residual connection
        
    Returns:
        ProteinConditionEncoder instance
    """
    return ProteinConditionEncoder(
        pocket_dim=pocket_dim,
        evo_dim=evo_dim,
        common_dim=common_dim,
        out_dim=out_dim or latent_dim,
        dropout=dropout,
        use_gated_residual=use_gated_residual,
    )


if __name__ == "__main__":
    # Minimal usage example
    torch.use_deterministic_algorithms(True)
    B = 4
    x_pocket = torch.randn(B, 512)
    x_evo = torch.randn(B, 1280)
    model = ProteinConditionEncoder(common_dim=512, out_dim=512, dropout=0.1, use_gated_residual=True)
    z = model(x_pocket, x_evo)
    print("z shape:", z.shape)  # expect [4, 512]

    # Lightweight tests (pytest-style) — can be split into a test file later
    def test_shapes():
        m = ProteinConditionEncoder()
        z = m(torch.randn(2, 512), torch.randn(2, 1280))
        assert z.shape == (2, m.out_dim)

    def test_grad_flow():
        m = ProteinConditionEncoder(use_gated_residual=True)
        z = m(torch.randn(3, 512), torch.randn(3, 1280))
        loss = z.pow(2).mean()
        loss.backward()
        grads = [p.grad for p in m.parameters() if p.requires_grad]
        assert all(g is not None for g in grads)

    def test_toggle_gate():
        m1 = ProteinConditionEncoder(use_gated_residual=False)
        m2 = ProteinConditionEncoder(use_gated_residual=True)
        x1, x2 = torch.randn(5, 512), torch.randn(5, 1280)
        z1 = m1(x1, x2)
        z2 = m2(x1, x2)
        assert z1.shape == z2.shape

    # Run tests quickly
    test_shapes()
    test_grad_flow()
    test_toggle_gate()
