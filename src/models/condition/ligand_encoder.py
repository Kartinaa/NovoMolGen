"""Two-tower ligand encoder with early fusion for VAE posterior parameters.

Implements a self-contained PyTorch module that takes two inputs:
  - ifp: [B, 16384]
  - mol: [B, 1536]
and outputs Gaussian parameters (mu, sigma) for a VAE posterior. Optionally
returns logvar for KL computation compatibility.
"""

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LigandConditionEncoder(nn.Module):
    """Two-tower encoder + early-fusion head producing VAE posterior parameters.

    Architecture:
      - IFP tower (aggressive): 16384 -> 4096 -> 1024 -> d_embed
      - MOL tower (light):      1536  -> 1024 -> d_embed
      - Early fusion with concat([a, b, a*b, |a-b|]) then small MLP to d_embed
        plus a gated residual skip from a linear projection of the fused vector.
      - Heads produce mu and rho; sigma = softplus(rho) + eps_sigma.

    Args:
      d_ifp: int, dimension of IFP input (default 16384)
      d_mol: int, dimension of molecule input (default 1536)
      d_embed: int, shared embedding size after towers
      z_dim: int, latent dimension for VAE parameters
      dropout: float, dropout probability
      use_layernorm: bool, whether to apply LayerNorm after each block
      hidden_fuse: int, hidden size in fusion MLP
      eps_sigma: float, epsilon added to softplus for positivity
    """

    def __init__(
        self,
        d_ifp: int = 16384,
        d_mol: int = 1536,
        d_embed: int = 512,
        z_dim: int = 512,
        dropout: float = 0.1,
        use_layernorm: bool = True,
        hidden_fuse: int = 1024,
        eps_sigma: float = 1e-5,
    ) -> None:
        super().__init__()
        self.d_ifp = d_ifp
        self.d_mol = d_mol
        self.d_embed = d_embed
        self.z_dim = z_dim
        self.eps_sigma = eps_sigma
        self.use_layernorm = use_layernorm

        # IFP tower: 16384 -> 4096 -> 1024 -> d_embed
        self.ifp_fc1 = nn.Linear(d_ifp, 4096)
        self.ifp_fc2 = nn.Linear(4096, 1024)
        self.ifp_fc3 = nn.Linear(1024, d_embed)
        self.ifp_drop1 = nn.Dropout(dropout)
        self.ifp_drop2 = nn.Dropout(dropout)
        self.ifp_drop3 = nn.Dropout(dropout)
        if use_layernorm:
            self.ifp_ln1 = nn.LayerNorm(4096)
            self.ifp_ln2 = nn.LayerNorm(1024)
            self.ifp_ln3 = nn.LayerNorm(d_embed)

        # MOL tower: 1536 -> 1024 -> d_embed
        self.mol_fc1 = nn.Linear(d_mol, 1024)
        self.mol_fc2 = nn.Linear(1024, d_embed)
        self.mol_drop1 = nn.Dropout(dropout)
        self.mol_drop2 = nn.Dropout(dropout)
        if use_layernorm:
            self.mol_ln1 = nn.LayerNorm(1024)
            self.mol_ln2 = nn.LayerNorm(d_embed)

        # Early fusion: concat(a, b, a*b, |a-b|) => 4*d_embed
        self.fuse_in = 4 * d_embed
        self.fuse_fc1 = nn.Linear(self.fuse_in, hidden_fuse)
        self.fuse_fc2 = nn.Linear(hidden_fuse, d_embed)
        self.fuse_drop = nn.Dropout(dropout)

        # Cheap skip projection and learnable gate
        self.fuse_skip = nn.Linear(self.fuse_in, d_embed)
        self.fuse_gate = nn.Parameter(torch.zeros(1))

        # Output heads
        self.mu_head = nn.Linear(d_embed, z_dim)
        self.rho_head = nn.Linear(d_embed, z_dim)

        # Initialize all linear layers
        for layer in [self.ifp_fc1, self.ifp_fc2, self.ifp_fc3, 
                      self.mol_fc1, self.mol_fc2,
                      self.fuse_fc1, self.fuse_fc2, self.fuse_skip,
                      self.mu_head, self.rho_head]:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def _encode_ifp_tower(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        h = self.ifp_fc1(h)
        h = F.gelu(h)
        h = self.ifp_drop1(h)
        if self.use_layernorm:
            h = self.ifp_ln1(h)
        
        h = self.ifp_fc2(h)
        h = F.gelu(h)
        h = self.ifp_drop2(h)
        if self.use_layernorm:
            h = self.ifp_ln2(h)
        
        h = self.ifp_fc3(h)
        h = F.gelu(h)
        h = self.ifp_drop3(h)
        if self.use_layernorm:
            h = self.ifp_ln3(h)
        
        # Numerical stability: replace NaNs/Infs
        h = torch.nan_to_num(h)
        return h

    def _encode_mol_tower(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        h = self.mol_fc1(h)
        h = F.gelu(h)
        h = self.mol_drop1(h)
        if self.use_layernorm:
            h = self.mol_ln1(h)
        
        h = self.mol_fc2(h)
        h = F.gelu(h)
        h = self.mol_drop2(h)
        if self.use_layernorm:
            h = self.mol_ln2(h)
        
        # Numerical stability: replace NaNs/Infs
        h = torch.nan_to_num(h)
        return h

    def forward(self, ifp: torch.Tensor, mol: torch.Tensor, return_logvar: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
          ifp: [B, d_ifp] float tensor (interaction fingerprint)
          mol: [B, d_mol] float tensor (molecule representation)
          return_logvar: whether to also return log-variance

        Returns:
          mu:    [B, z_dim]
          sigma: [B, z_dim] strictly positive
          (optional) logvar: [B, z_dim]
        """
        assert ifp.dim() == 2 and ifp.size(1) == self.d_ifp, f"ifp must be [B, {self.d_ifp}]"
        assert mol.dim() == 2 and mol.size(1) == self.d_mol, f"mol must be [B, {self.d_mol}]"

        # Encode towers
        a = self._encode_ifp_tower(ifp)  # [B, d_embed]
        b = self._encode_mol_tower(mol)  # [B, d_embed]

        # Early fusion features
        f = torch.cat([a, b, a * b, torch.abs(a - b)], dim=-1)  # [B, 4*d_embed]

        # Skip path
        h0 = self.fuse_skip(f)  # [B, d_embed]

        # Fusion MLP
        h = self.fuse_fc1(f)
        h = F.gelu(h)
        h = self.fuse_drop(h)
        h = self.fuse_fc2(h)
        h = F.gelu(h)

        # Gated residual
        gate = torch.sigmoid(self.fuse_gate).to(h.dtype)
        h = h0 + gate * h

        # Heads
        mu = self.mu_head(h)
        rho = self.rho_head(h)
        sigma = F.softplus(rho) + self.eps_sigma

        if return_logvar:
            logvar = 2.0 * torch.log(sigma)
            return mu, sigma, logvar
        return mu, sigma, None

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z via reparameterization trick: z = mu + eps * exp(0.5 * logvar)."""
        eps = torch.randn_like(mu)
        std = torch.exp(0.5 * logvar)
        return mu + eps * std

    @staticmethod
    def kl_to_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute batch-mean KL(q||N(0,I)) for diagonal Gaussians.

        KL = 0.5 * sum(exp(logvar) + mu^2 - 1 - logvar)
        """
        kl = 0.5 * (torch.exp(logvar) + mu.pow(2) - 1.0 - logvar)
        return kl.sum(dim=-1).mean()


def create_ligand_encoder(
    latent_dim: Optional[int] = None,
    ligand_input_dim: Optional[int] = None,
    hidden_dim: int = 256,
    dropout: float = 0.1,
    d_ifp: int = 16384,
    d_mol: int = 1536,
    d_embed: int = 512,
    z_dim: Optional[int] = None,
    **kwargs,
) -> LigandConditionEncoder:
    """Factory with backward-compat args.

    - If latent_dim provided, map to z_dim.
    - ligand_input_dim is ignored in two-tower setup but accepted for compatibility.
    - hidden_dim maps to hidden_fuse; towers are fixed as specified.
    """
    if z_dim is None and latent_dim is not None:
        z_dim = latent_dim
    if z_dim is None:
        z_dim = 512
    return LigandConditionEncoder(
        d_ifp=d_ifp,
        d_mol=d_mol,
        d_embed=d_embed,
        z_dim=z_dim,
        dropout=dropout,
        use_layernorm=True,
        hidden_fuse=hidden_dim,
    )


if __name__ == "__main__":
    # Minimal usage snippet
    torch.manual_seed(0)
    enc = LigandConditionEncoder()
    ifp = torch.randn(2, 16384)
    mol = torch.randn(2, 1536)
    mu, sigma, _ = enc(ifp, mol, return_logvar=False)
    print("mu shape:", mu.shape, "sigma shape:", sigma.shape)
    mu2, sigma2, logvar = enc(ifp, mol, return_logvar=True)
    kl = LigandConditionEncoder.kl_to_standard_normal(mu2, logvar)
    print("KL:", float(kl))
