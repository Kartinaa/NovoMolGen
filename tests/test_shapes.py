"""Minimal unit tests for feature extraction modules.

Tests shape consistency and basic functionality of IFP and molecular representation extraction.
"""

import unittest
import tempfile
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from features.ifp import get_ifp_vector, get_available_ifp_backends
from features.mol_repr import get_mol_vector, get_available_mol_methods
from io_utils.structures import get_available_backends


class TestFeatureShapes(unittest.TestCase):
    """Test shape consistency of feature extraction."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary test files
        self.temp_dir = tempfile.mkdtemp()
        self.ligand_path = Path(self.temp_dir) / "test_ligand.sdf"
        self.protein_path = Path(self.temp_dir) / "test_protein.pdb"
        
        # Create dummy ligand SDF file
        self._create_dummy_ligand()
        self._create_dummy_protein()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def _create_dummy_ligand(self):
        """Create a dummy SDF file for testing."""
        sdf_content = """
  Mrv2014 01012024

  3  2  0  0  0  0            999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  2  3  1  0  0  0  0
M  END
$$$$
"""
        with open(self.ligand_path, 'w') as f:
            f.write(sdf_content)
    
    def _create_dummy_protein(self):
        """Create a dummy PDB file for testing."""
        pdb_content = """HEADER    TEST PROTEIN                           01-JAN-24   TEST
ATOM      1  N   ALA A   1      20.154  16.967  23.862  1.00 11.18           N
ATOM      2  CA  ALA A   1      19.030  16.140  23.362  1.00 11.18           C
ATOM      3  C   ALA A   1      17.782  16.889  23.362  1.00 11.18           C
ATOM      4  O   ALA A   1      17.782  18.089  23.362  1.00 11.18           O
END
"""
        with open(self.protein_path, 'w') as f:
            f.write(pdb_content)
    
    def test_ifp_vector_shapes(self):
        """Test IFP vector shape consistency."""
        available_backends = get_available_ifp_backends()
        self.assertIn("dummy", available_backends, "Dummy backend should always be available")
        
        for backend in available_backends:
            with self.subTest(backend=backend):
                ifp_vector = get_ifp_vector(self.ligand_path, self.protein_path, backend=backend)
                
                # Check shape
                self.assertEqual(ifp_vector.shape, (16384,), 
                               f"IFP vector should have shape (16384,) for backend {backend}")
                
                # Check dtype
                self.assertEqual(ifp_vector.dtype, np.float32,
                               f"IFP vector should be float32 for backend {backend}")
                
                # Check value range (should be 0 or 1 for binary features)
                self.assertTrue(np.all((ifp_vector == 0) | (ifp_vector == 1)),
                              f"IFP vector should contain only 0s and 1s for backend {backend}")
    
    def test_mol_vector_shapes(self):
        """Test molecular representation vector shape consistency."""
        available_methods = get_available_mol_methods()
        self.assertIn("user_encoder", available_methods, "User encoder should always be available")
        
        for method in available_methods:
            with self.subTest(method=method):
                mol_vector = get_mol_vector(self.ligand_path, method=method)
                
                # Check shape
                self.assertEqual(mol_vector.shape, (1536,),
                               f"Mol vector should have shape (1536,) for method {method}")
                
                # Check dtype
                self.assertEqual(mol_vector.dtype, np.float32,
                               f"Mol vector should be float32 for method {method}")
                
                # Check that it's not all zeros (should have some features)
                self.assertGreater(np.count_nonzero(mol_vector), 0,
                                 f"Mol vector should have non-zero elements for method {method}")
    
    def test_deterministic_output(self):
        """Test that outputs are deterministic for same inputs."""
        # Test IFP determinism
        ifp1 = get_ifp_vector(self.ligand_path, self.protein_path, backend="dummy")
        ifp2 = get_ifp_vector(self.ligand_path, self.protein_path, backend="dummy")
        np.testing.assert_array_equal(ifp1, ifp2, "IFP output should be deterministic")
        
        # Test mol vector determinism (if fallback is available)
        if "fallback" in get_available_mol_methods():
            mol1 = get_mol_vector(self.ligand_path, method="fallback")
            mol2 = get_mol_vector(self.ligand_path, method="fallback")
            np.testing.assert_array_equal(mol1, mol2, "Mol vector output should be deterministic")
    
    def test_file_not_found_errors(self):
        """Test proper error handling for missing files."""
        fake_ligand = Path(self.temp_dir) / "fake_ligand.sdf"
        fake_protein = Path(self.temp_dir) / "fake_protein.pdb"
        
        # Test IFP with missing files
        with self.assertRaises(FileNotFoundError):
            get_ifp_vector(fake_ligand, self.protein_path, backend="dummy")
        
        with self.assertRaises(FileNotFoundError):
            get_ifp_vector(self.ligand_path, fake_protein, backend="dummy")
        
        # Test mol vector with missing files
        with self.assertRaises(FileNotFoundError):
            get_mol_vector(fake_ligand, method="user_encoder")
    
    def test_backend_availability(self):
        """Test backend availability reporting."""
        ifp_backends = get_available_ifp_backends()
        mol_methods = get_available_mol_methods()
        io_backends = get_available_backends()
        
        # Check that we get lists
        self.assertIsInstance(ifp_backends, list)
        self.assertIsInstance(mol_methods, list)
        self.assertIsInstance(io_backends, dict)
        
        # Check that dummy/user_encoder are always available
        self.assertIn("dummy", ifp_backends)
        self.assertIn("user_encoder", mol_methods)


if __name__ == "__main__":
    unittest.main()
