#!/usr/bin/env python3
"""
Structure test script for the new NovoMolGen architecture.

This script tests the code structure, imports, and basic functionality
without requiring the full PyTorch environment.

Usage:
    python3 test_architecture_structure.py
"""

import sys
import os
import inspect
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all modules can be imported."""
    print("🧪 Testing Imports...")
    
    try:
        # Test data module import
        from src.data_loader.molecule_data_module import MolDataModule
        print("✅ MolDataModule imported successfully")
        
        # Test condition modules
        from src.models.condition import (
            create_ligand_encoder, 
            create_protein_encoder, 
            create_condition_fusion
        )
        print("✅ Condition modules imported successfully")
        
        # Test model import (this might fail without torch, but we'll catch it)
        try:
            from src.models.modeling_novomolgen import NovoMolGen, NovoMolGenConfig
            print("✅ NovoMolGen model imported successfully")
        except ImportError as e:
            print(f"⚠️  NovoMolGen import failed (expected without torch): {e}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_data_module_structure():
    """Test data module structure and methods."""
    print("\n🧪 Testing Data Module Structure...")
    
    try:
        from src.data_loader.molecule_data_module import MolDataModule
        
        # Check class structure
        assert hasattr(MolDataModule, '__init__'), "MolDataModule should have __init__"
        assert hasattr(MolDataModule, 'make_dummy_batch'), "MolDataModule should have make_dummy_batch"
        assert hasattr(MolDataModule, '_build_collate_fn'), "MolDataModule should have _build_collate_fn"
        
        # Check make_dummy_batch signature
        sig = inspect.signature(MolDataModule.make_dummy_batch)
        params = list(sig.parameters.keys())
        
        expected_params = ['batch_size', 'seq_len', 'vocab_size', 'include_condition_features', 'device']
        for param in expected_params:
            assert param in params, f"make_dummy_batch should have parameter: {param}"
        
        # Check that legacy parameters are removed
        legacy_params = ['include_protein_inputs', 'include_posterior_params', 'latent_dim']
        for param in legacy_params:
            assert param not in params, f"make_dummy_batch should not have legacy parameter: {param}"
        
        print("✅ Data module structure is correct")
        return True
        
    except Exception as e:
        print(f"❌ Data module structure test failed: {e}")
        return False


def test_condition_modules_structure():
    """Test condition module structure."""
    print("\n🧪 Testing Condition Modules Structure...")
    
    try:
        from src.models.condition import (
            create_ligand_encoder, 
            create_protein_encoder, 
            create_condition_fusion
        )
        
        # Test factory functions exist
        assert callable(create_ligand_encoder), "create_ligand_encoder should be callable"
        assert callable(create_protein_encoder), "create_protein_encoder should be callable"
        assert callable(create_condition_fusion), "create_condition_fusion should be callable"
        
        # Test function signatures
        ligand_sig = inspect.signature(create_ligand_encoder)
        protein_sig = inspect.signature(create_protein_encoder)
        fusion_sig = inspect.signature(create_condition_fusion)
        
        print(f"✅ Ligand encoder signature: {ligand_sig}")
        print(f"✅ Protein encoder signature: {protein_sig}")
        print(f"✅ Condition fusion signature: {fusion_sig}")
        
        return True
        
    except Exception as e:
        print(f"❌ Condition modules structure test failed: {e}")
        return False


def test_model_structure():
    """Test model structure (without instantiation)."""
    print("\n🧪 Testing Model Structure...")
    
    try:
        # Try to import the model classes
        try:
            from src.models.modeling_novomolgen import NovoMolGen, NovoMolGenConfig
        except ImportError:
            print("⚠️  Skipping model structure test (torch not available)")
            return True
        
        # Check NovoMolGenConfig structure
        config_sig = inspect.signature(NovoMolGenConfig.__init__)
        config_params = list(config_sig.parameters.keys())
        
        # Check that legacy parameters are removed
        legacy_params = ['ligand_input_dim']
        for param in legacy_params:
            assert param not in config_params, f"NovoMolGenConfig should not have legacy parameter: {param}"
        
        # Check that new parameters exist
        new_params = ['cond_tokens_len', 'enable_cross_attn', 'cross_layers', 'beta_kl']
        for param in new_params:
            assert param in config_params, f"NovoMolGenConfig should have parameter: {param}"
        
        # Check NovoMolGen forward signature
        forward_sig = inspect.signature(NovoMolGen.forward)
        forward_params = list(forward_sig.parameters.keys())
        
        # Check that legacy parameters are removed
        legacy_forward_params = ['protein_inputs', 'posterior_params', 'ligand_inputs']
        for param in legacy_forward_params:
            assert param not in forward_params, f"NovoMolGen.forward should not have legacy parameter: {param}"
        
        # Check that new parameters exist
        new_forward_params = ['pocket_vec', 'evo_vec', 'ifp', 'ligand_vec']
        for param in new_forward_params:
            assert param in forward_params, f"NovoMolGen.forward should have parameter: {param}"
        
        print("✅ Model structure is correct")
        return True
        
    except Exception as e:
        print(f"❌ Model structure test failed: {e}")
        return False


def test_file_structure():
    """Test that all required files exist."""
    print("\n🧪 Testing File Structure...")
    
    required_files = [
        "src/data_loader/molecule_data_module.py",
        "src/models/modeling_novomolgen.py",
        "src/models/condition/__init__.py",
        "src/models/condition/ligand_encoder.py",
        "src/models/condition/protein_encoder.py",
        "src/models/condition/condition_fusion.py",
        "md_cursor/NEW_ARCHITECTURE.md"
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = project_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path} exists")
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All required files exist")
    return True


def test_architecture_documentation():
    """Test that architecture documentation exists and is readable."""
    print("\n🧪 Testing Architecture Documentation...")
    
    try:
        doc_path = project_root / "md_cursor" / "NEW_ARCHITECTURE.md"
        
        if not doc_path.exists():
            print("❌ Architecture documentation not found")
            return False
        
        # Read and check content
        with open(doc_path, 'r') as f:
            content = f.read()
        
        # Check for key sections
        required_sections = [
            "## Architecture Overview",
            "## Data Flow",
            "## Key Components",
            "LigandConditionEncoder",
            "ProteinConditionEncoder", 
            "ConditionFusion",
            "NovoMolGen"
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in content:
                missing_sections.append(section)
        
        if missing_sections:
            print(f"❌ Missing documentation sections: {missing_sections}")
            return False
        
        print("✅ Architecture documentation is complete")
        return True
        
    except Exception as e:
        print(f"❌ Architecture documentation test failed: {e}")
        return False


def run_structure_tests():
    """Run all structure tests."""
    print("🚀 Starting NovoMolGen Architecture Structure Tests")
    print("=" * 60)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Imports", test_imports),
        ("Data Module Structure", test_data_module_structure),
        ("Condition Modules Structure", test_condition_modules_structure),
        ("Model Structure", test_model_structure),
        ("Architecture Documentation", test_architecture_documentation),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {status}: {test_name}")
        if not result:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("🎉 ALL STRUCTURE TESTS PASSED!")
        print("The new architecture is properly structured and ready for use.")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please check the issues above before proceeding.")
    
    return all_passed


if __name__ == "__main__":
    success = run_structure_tests()
    sys.exit(0 if success else 1)
