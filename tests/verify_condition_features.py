#!/usr/bin/env python3
"""
Script to verify that your dataset has the correct format for condition features
"""

import numpy as np
from datasets import Dataset

def verify_condition_features(dataset):
    """Verify that dataset has correct condition features format"""
    print("🔍 Verifying condition features...")
    print(f"📊 Dataset size: {len(dataset)}")
    print(f"📋 Dataset features: {list(dataset.features.keys())}")
    
    required_keys = ["pocket_vec", "evo_vec", "ifp", "ligand_vec"]
    expected_shapes = [512, 1280, 16384, 1536]
    
    # Check if all required keys exist
    missing_keys = [key for key in required_keys if key not in dataset.features]
    if missing_keys:
        print(f"❌ Missing keys in dataset: {missing_keys}")
        print(f"💡 Your dataset should include these columns: {required_keys}")
        return False
    
    # Check first sample
    sample = dataset[0]
    print(f"\n📦 Checking first sample...")
    
    for key, expected_shape in zip(required_keys, expected_shapes):
        if key not in sample:
            print(f"❌ Missing key in sample: {key}")
            return False
        
        if len(sample[key]) != expected_shape:
            print(f"❌ Wrong shape for {key}: {len(sample[key])} (expected: {expected_shape})")
            return False
        
        if not isinstance(sample[key], list):
            print(f"❌ {key} should be a list, got: {type(sample[key])}")
            return False
        
        if not all(isinstance(x, (int, float)) for x in sample[key]):
            print(f"❌ {key} should contain only numbers")
            return False
        
        print(f"✅ {key}: {len(sample[key])} floats")
    
    # Check a few more samples to ensure consistency
    print(f"\n🔍 Checking sample consistency...")
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        for key, expected_shape in zip(required_keys, expected_shapes):
            if len(sample[key]) != expected_shape:
                print(f"❌ Sample {i}: {key} has wrong shape: {len(sample[key])}")
                return False
    
    print(f"✅ All condition features verified!")
    print(f"✅ Dataset is ready for use with MolDataModule(include_condition_features=True)")
    return True

def create_test_dataset():
    """Create a test dataset with correct format"""
    print("🧪 Creating test dataset with correct format...")
    
    # Create test data
    data = {
        "SMILES": ["CCO", "CC(=O)O", "c1ccccc1", "CCN(CC)CC", "CC(C)O"],
        "pocket_vec": [np.random.randn(512).tolist() for _ in range(5)],
        "evo_vec": [np.random.randn(1280).tolist() for _ in range(5)],
        "ifp": [np.random.randn(16384).tolist() for _ in range(5)],
        "ligand_vec": [np.random.randn(1536).tolist() for _ in range(5)]
    }
    
    dataset = Dataset.from_dict(data)
    return dataset

def test_mol_data_module():
    """Test MolDataModule with condition features"""
    print("\n🧪 Testing MolDataModule with condition features...")
    
    try:
        from src.data_loader.molecule_data_module import MolDataModule
        
        # Create test dataset
        dataset = create_test_dataset()
        
        # Verify the dataset
        if not verify_condition_features(dataset):
            return False
        
        # Test make_dummy_batch with real condition data
        print(f"\n🔧 Testing make_dummy_batch with real condition data...")
        
        # Extract condition data for first 2 samples
        condition_data = {
            "pocket_vec": [dataset[0]["pocket_vec"], dataset[1]["pocket_vec"]],
            "evo_vec": [dataset[0]["evo_vec"], dataset[1]["evo_vec"]],
            "ifp": [dataset[0]["ifp"], dataset[1]["ifp"]],
            "ligand_vec": [dataset[0]["ligand_vec"], dataset[1]["ligand_vec"]]
        }
        
        batch = MolDataModule.make_dummy_batch(
            batch_size=2,
            seq_len=8,
            vocab_size=100,
            include_condition_features=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
            condition_data=condition_data
        )
        
        print(f"✅ make_dummy_batch successful")
        print(f"📦 Batch keys: {list(batch.keys())}")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape} ({value.dtype})")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import torch
    
    print("🚀 Condition Features Verification Script")
    print("=" * 50)
    
    # Test 1: Create and verify test dataset
    print("Test 1: Creating and verifying test dataset")
    test_dataset = create_test_dataset()
    verify_condition_features(test_dataset)
    
    # Test 2: Test MolDataModule integration
    print("\nTest 2: Testing MolDataModule integration")
    test_mol_data_module()
    
    print(f"\n🎯 Summary:")
    print(f"✅ Your dataset should have these columns:")
    print(f"   - SMILES: molecular representation")
    print(f"   - pocket_vec: list of 512 floats")
    print(f"   - evo_vec: list of 1280 floats")
    print(f"   - ifp: list of 16384 floats")
    print(f"   - ligand_vec: list of 1536 floats")
    print(f"\n📚 See CONDITION_FEATURES_GUIDE.md for complete usage instructions")
