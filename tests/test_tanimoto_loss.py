"""
Tests for Tanimoto loss implementation.

Run with: pytest tests/test_tanimoto_loss.py -v
"""

import pytest
import torch


class TestTanimotoLoss:
    """Test suite for Tanimoto loss functions."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Check if RDKit is available."""
        try:
            from rdkit import Chem
            self.rdkit_available = True
        except ImportError:
            self.rdkit_available = False

    def test_compute_morgan_fingerprint_valid(self):
        """Test Morgan fingerprint computation for valid SMILES."""
        if not self.rdkit_available:
            pytest.skip("RDKit not available")

        from src.models.modeling_novomolgen_tanimoto import compute_morgan_fingerprint

        # Valid SMILES
        smiles = "CCO"  # Ethanol
        fp = compute_morgan_fingerprint(smiles, radius=2, n_bits=2048)
        assert fp is not None
        assert len(fp) == 2048

    def test_compute_morgan_fingerprint_invalid(self):
        """Test Morgan fingerprint returns None for invalid SMILES."""
        if not self.rdkit_available:
            pytest.skip("RDKit not available")

        from src.models.modeling_novomolgen_tanimoto import compute_morgan_fingerprint

        # Invalid SMILES
        smiles = "invalid_smiles_xyz"
        fp = compute_morgan_fingerprint(smiles, radius=2, n_bits=2048)
        assert fp is None

    def test_compute_tanimoto_matrix(self):
        """Test Tanimoto similarity matrix computation."""
        if not self.rdkit_available:
            pytest.skip("RDKit not available")

        from src.models.modeling_novomolgen_tanimoto import compute_tanimoto_matrix

        smiles_list = ["CCO", "CCN", "c1ccccc1", "CC(=O)O"]
        T, valid_mask = compute_tanimoto_matrix(smiles_list)

        # Check shape
        assert T.shape == (4, 4)
        assert valid_mask.shape == (4,)

        # Check diagonal is 1.0 (self-similarity)
        for i in range(4):
            assert abs(T[i, i].item() - 1.0) < 1e-5

        # Check symmetry
        for i in range(4):
            for j in range(4):
                assert abs(T[i, j].item() - T[j, i].item()) < 1e-5

        # Check all valid
        assert valid_mask.all()

    def test_compute_tanimoto_matrix_with_invalid(self):
        """Test Tanimoto matrix with some invalid SMILES."""
        if not self.rdkit_available:
            pytest.skip("RDKit not available")

        from src.models.modeling_novomolgen_tanimoto import compute_tanimoto_matrix

        smiles_list = ["CCO", "invalid", "c1ccccc1"]
        T, valid_mask = compute_tanimoto_matrix(smiles_list)

        # Check shape
        assert T.shape == (3, 3)
        assert valid_mask.shape == (3,)

        # Check valid mask
        assert valid_mask[0].item() == True
        assert valid_mask[1].item() == False
        assert valid_mask[2].item() == True

    def test_compute_euclidean_distance_matrix(self):
        """Test Euclidean distance matrix computation."""
        from src.models.modeling_novomolgen_tanimoto import compute_euclidean_distance_matrix

        # Create simple z vectors
        z = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])

        D = compute_euclidean_distance_matrix(z)

        # Check shape
        assert D.shape == (3, 3)

        # Check diagonal is 0
        for i in range(3):
            assert abs(D[i, i].item()) < 1e-5

        # Check symmetry
        for i in range(3):
            for j in range(3):
                assert abs(D[i, j].item() - D[j, i].item()) < 1e-5

        # Check known distances
        # d(z0, z1) = sqrt((1-0)^2 + (0-1)^2) = sqrt(2)
        assert abs(D[0, 1].item() - 1.4142135) < 1e-5

    def test_compute_tanimoto_loss_basic(self):
        """Test Tanimoto loss computation."""
        if not self.rdkit_available:
            pytest.skip("RDKit not available")

        from src.models.modeling_novomolgen_tanimoto import compute_tanimoto_loss

        # Create z vectors
        z = torch.randn(4, 128, requires_grad=True)

        # SMILES list
        smiles_list = ["CCO", "CCN", "c1ccccc1", "CC(=O)O"]

        loss = compute_tanimoto_loss(z, smiles_list)

        # Check loss is scalar
        assert loss.dim() == 0

        # Check loss is in valid range [0, 2]
        # (1 + rho where rho in [-1, 1])
        assert loss.item() >= 0.0
        assert loss.item() <= 2.0

        # Check gradient can be computed
        loss.backward()
        assert z.grad is not None

    def test_compute_tanimoto_loss_batch_size_1(self):
        """Test Tanimoto loss returns 0 for batch size 1."""
        if not self.rdkit_available:
            pytest.skip("RDKit not available")

        from src.models.modeling_novomolgen_tanimoto import compute_tanimoto_loss

        z = torch.randn(1, 128, requires_grad=True)
        smiles_list = ["CCO"]

        loss = compute_tanimoto_loss(z, smiles_list)

        # Should return 0 for batch size 1 (no pairs)
        assert abs(loss.item()) < 1e-5

    def test_compute_tanimoto_loss_all_invalid(self):
        """Test Tanimoto loss returns 0 when all SMILES are invalid."""
        if not self.rdkit_available:
            pytest.skip("RDKit not available")

        from src.models.modeling_novomolgen_tanimoto import compute_tanimoto_loss

        z = torch.randn(4, 128, requires_grad=True)
        smiles_list = ["invalid1", "invalid2", "invalid3", "invalid4"]

        loss = compute_tanimoto_loss(z, smiles_list)

        # Should return 0 when all SMILES are invalid
        assert abs(loss.item()) < 1e-5

    def test_compute_tanimoto_loss_identical_molecules(self):
        """Test Tanimoto loss with identical molecules and z vectors."""
        if not self.rdkit_available:
            pytest.skip("RDKit not available")

        from src.models.modeling_novomolgen_tanimoto import compute_tanimoto_loss

        # Identical z vectors should have 0 distance
        z_base = torch.randn(1, 128)
        z = z_base.repeat(4, 1).requires_grad_(True)

        # Identical SMILES should have 1.0 similarity
        smiles_list = ["CCO", "CCO", "CCO", "CCO"]

        loss = compute_tanimoto_loss(z, smiles_list)

        # With identical z and identical molecules:
        # T = 1.0 for all pairs, D = 0.0 for all pairs
        # This is a degenerate case where std = 0
        # The loss should still be valid
        assert loss.item() >= 0.0
        assert loss.item() <= 2.0


class TestDataPipelineWithSmiles:
    """Test that data pipeline preserves SMILES strings."""

    def test_tokenize_function_preserves_mol_string(self):
        """Test that tokenize_function preserves mol_string."""
        import sys
        from pathlib import Path

        # Add src to path
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

        from data_loader.molecule_data_module import MolDataModule

        # Create mock element
        element = {
            "SAFE": "c1ccccc1.CCO",
            "pocket_vec": [0.1] * 512,
            "evo_vec": [0.1] * 1280,
            "ifp": [0.1] * 16384,
            "ligand_vec": [0.1] * 1536,
        }

        # Create mock tokenizer
        from transformers import AutoTokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        except Exception:
            pytest.skip("Could not load tokenizer")

        result = MolDataModule.tokenize_function(
            element=element,
            max_length=64,
            mol_type="SAFE",
            tokenizer=tokenizer,
        )

        # Check that mol_string is preserved
        assert "mol_string" in result
        assert result["mol_string"] == "c1ccccc1.CCO"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
