import torch

from src.condition.ligand_encoder import LigandConditionEncoder


def test_shapes():
    torch.manual_seed(42)
    enc = LigandConditionEncoder()
    bsz = 3
    ifp = torch.randn(bsz, 16384)
    mol = torch.randn(bsz, 1536)
    mu, sigma, logvar = enc(ifp, mol, return_logvar=True)

    assert mu.shape == (bsz, enc.z_dim)
    assert sigma.shape == (bsz, enc.z_dim)
    assert logvar is not None and logvar.shape == (bsz, enc.z_dim)
    assert torch.all(sigma > 0), "sigma must be strictly positive"


def test_backward():
    torch.manual_seed(0)
    enc = LigandConditionEncoder()
    bsz = 2
    ifp = torch.randn(bsz, 16384, requires_grad=True)
    mol = torch.randn(bsz, 1536, requires_grad=True)
    mu, sigma, logvar = enc(ifp, mol, return_logvar=True)

    # Dummy VAE loss: KL + L2 on mu
    kl = LigandConditionEncoder.kl_to_standard_normal(mu, logvar)
    l2 = (mu ** 2).mean()
    loss = kl + 0.01 * l2
    loss.backward()

    assert ifp.grad is not None
    assert mol.grad is not None


def test_determinism_seed():
    torch.manual_seed(123)
    enc1 = LigandConditionEncoder()
    ifp = torch.randn(2, 16384)
    mol = torch.randn(2, 1536)
    mu1, sigma1, _ = enc1(ifp, mol, return_logvar=False)

    torch.manual_seed(123)
    enc2 = LigandConditionEncoder()
    mu2, sigma2, _ = enc2(ifp, mol, return_logvar=False)

    assert torch.allclose(mu1, mu2, atol=1e-6)
    assert torch.allclose(sigma1, sigma2, atol=1e-6)


