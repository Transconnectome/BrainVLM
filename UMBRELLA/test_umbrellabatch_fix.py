#!/usr/bin/env python3
"""
Quick Verification Script for UMBRELLABatch HuggingFace Trainer Compatibility Fix

Tests that the __len__() method and dict-like access work correctly.

Usage:
    python test_umbrellabatch_fix.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

import torch
from project.dataset.umbrella_collator import UMBRELLACollator, UMBRELLABatch


def test_len_method():
    """Test that __len__() method works correctly."""
    print("\n" + "="*60)
    print("TEST 1: __len__() Method")
    print("="*60)

    collator = UMBRELLACollator(tokenizer=None, img_size=128, max_seq_length=2048)

    # Create test batch with 2 samples
    test_batch = [
        {
            'pixel_values': torch.randn(1, 1, 128, 128, 128),
            'input_ids': torch.randint(0, 1000, (100,)),
            'attention_mask': torch.ones(100),
            'labels': torch.randint(0, 1000, (100,)),
            'task_type': 'T1',
            'sample_index': 0,
            'num_images': 1,
            'metadata': {}
        },
        {
            'pixel_values': torch.randn(2, 1, 128, 128, 128),
            'input_ids': torch.randint(0, 1000, (150,)),
            'attention_mask': torch.ones(150),
            'labels': torch.randint(0, 1000, (150,)),
            'task_type': 'T2',
            'sample_index': 1,
            'num_images': 2,
            'metadata': {}
        }
    ]

    batch = collator(test_batch)

    # Test __len__()
    batch_len = len(batch)
    batch_size = batch.batch_size

    print(f"len(batch) = {batch_len}")
    print(f"batch.batch_size = {batch_size}")

    assert batch_len == 2, f"❌ FAILED: Expected len(batch) == 2, got {batch_len}"
    assert batch_size == 2, f"❌ FAILED: Expected batch_size == 2, got {batch_size}"
    assert batch_len == batch_size, f"❌ FAILED: len(batch) != batch.batch_size"

    print("✅ PASSED: __len__() returns correct batch size")
    return True


def test_empty_batch():
    """Test that empty batch has len() == 0."""
    print("\n" + "="*60)
    print("TEST 2: Empty Batch Handling")
    print("="*60)

    collator = UMBRELLACollator(tokenizer=None, img_size=128, max_seq_length=2048)

    empty = collator([])

    empty_len = len(empty)
    empty_size = empty.batch_size

    print(f"len(empty_batch) = {empty_len}")
    print(f"empty_batch.batch_size = {empty_size}")

    assert empty_len == 0, f"❌ FAILED: Expected len(empty) == 0, got {empty_len}"
    assert empty_size == 0, f"❌ FAILED: Expected batch_size == 0, got {empty_size}"

    print("✅ PASSED: Empty batch has len() == 0")
    return True


def test_dict_access():
    """Test dict-like access methods."""
    print("\n" + "="*60)
    print("TEST 3: Dict-like Access (__getitem__, keys, items)")
    print("="*60)

    collator = UMBRELLACollator(tokenizer=None, img_size=128, max_seq_length=2048)

    test_batch = [{
        'pixel_values': torch.randn(1, 1, 128, 128, 128),
        'input_ids': torch.randint(0, 1000, (100,)),
        'attention_mask': torch.ones(100),
        'labels': torch.randint(0, 1000, (100,)),
        'task_type': 'T1',
        'sample_index': 0,
        'num_images': 1,
        'metadata': {}
    }]

    batch = collator(test_batch)

    # Test __getitem__
    print("Testing batch['input_ids']...")
    input_ids = batch['input_ids']
    assert input_ids.shape == (1, 2048), f"❌ FAILED: Expected shape (1, 2048), got {input_ids.shape}"
    print(f"  ✓ batch['input_ids'].shape = {input_ids.shape}")

    print("Testing batch['pixel_values']...")
    pixel_values = batch['pixel_values']
    assert pixel_values.shape[0] == 1, f"❌ FAILED: Expected batch dimension 1, got {pixel_values.shape[0]}"
    print(f"  ✓ batch['pixel_values'].shape = {pixel_values.shape}")

    # Test keys()
    print("Testing batch.keys()...")
    keys = list(batch.keys())
    assert 'input_ids' in keys, "❌ FAILED: 'input_ids' not in keys()"
    assert 'pixel_values' in keys, "❌ FAILED: 'pixel_values' not in keys()"
    print(f"  ✓ keys() = {keys[:3]}... (showing first 3)")

    # Test items()
    print("Testing batch.items()...")
    items_dict = dict(batch.items())
    assert 'input_ids' in items_dict, "❌ FAILED: 'input_ids' not in items()"
    print(f"  ✓ items() contains {len(items_dict)} key-value pairs")

    # Test values()
    print("Testing batch.values()...")
    values = list(batch.values())
    assert len(values) == len(keys), "❌ FAILED: len(values) != len(keys)"
    print(f"  ✓ values() contains {len(values)} values")

    print("✅ PASSED: Dict-like access works correctly")
    return True


def test_iteration():
    """Test __iter__ method."""
    print("\n" + "="*60)
    print("TEST 4: Iteration (__iter__)")
    print("="*60)

    collator = UMBRELLACollator(tokenizer=None, img_size=128, max_seq_length=2048)

    test_batch = [{
        'pixel_values': torch.randn(1, 1, 128, 128, 128),
        'input_ids': torch.randint(0, 1000, (100,)),
        'attention_mask': torch.ones(100),
        'labels': torch.randint(0, 1000, (100,)),
        'task_type': 'T1',
        'sample_index': 0,
        'num_images': 1,
        'metadata': {}
    }]

    batch = collator(test_batch)

    # Test iteration
    print("Testing iteration over batch field names...")
    field_names = list(batch)
    expected_fields = [
        'pixel_values', 'input_ids', 'attention_mask', 'labels',
        'image_mask', 'num_images_per_sample', 'task_types',
        'task_ids', 'sample_indices', 'metadata'
    ]

    print(f"  Iterated fields: {field_names}")
    assert field_names == expected_fields, f"❌ FAILED: Field names don't match expected"

    print("✅ PASSED: Iteration over batch works correctly")
    return True


def test_trainer_compatibility():
    """Test HuggingFace Trainer _prepare_inputs() compatibility."""
    print("\n" + "="*60)
    print("TEST 5: HuggingFace Trainer Compatibility")
    print("="*60)

    collator = UMBRELLACollator(tokenizer=None, img_size=128, max_seq_length=2048)

    test_batch = [{
        'pixel_values': torch.randn(1, 1, 128, 128, 128),
        'input_ids': torch.randint(0, 1000, (100,)),
        'attention_mask': torch.ones(100),
        'labels': torch.randint(0, 1000, (100,)),
        'task_type': 'T1',
        'sample_index': 0,
        'num_images': 1,
        'metadata': {}
    }]

    batch = collator(test_batch)

    # Simulate what Trainer._prepare_inputs() does
    print("Simulating HuggingFace Trainer._prepare_inputs()...")
    print("  Checking: if len(inputs) == 0: ...")

    try:
        if len(batch) == 0:  # This was the failing line
            prepared = batch
        else:
            prepared = batch

        assert prepared is not None, "❌ FAILED: prepared batch is None"
        print("  ✓ len(batch) check passed without TypeError")
        print(f"  ✓ Batch has {len(batch)} samples")

        print("✅ PASSED: HuggingFace Trainer compatibility verified")
        return True

    except TypeError as e:
        print(f"❌ FAILED: TypeError occurred: {e}")
        return False


def run_all_tests():
    """Run all verification tests."""
    print("\n" + "="*70)
    print(" UMBRELLABatch HuggingFace Trainer Compatibility Fix Verification")
    print("="*70)

    tests = [
        ("__len__() Method", test_len_method),
        ("Empty Batch Handling", test_empty_batch),
        ("Dict-like Access", test_dict_access),
        ("Iteration", test_iteration),
        ("Trainer Compatibility", test_trainer_compatibility)
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ TEST FAILED: {test_name}")
            print(f"   Error: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "="*70)
    print(" TEST SUMMARY")
    print("="*70)

    all_passed = all(results.values())

    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")

    print("="*70)

    if all_passed:
        print("\n🎉 ALL TESTS PASSED! Training can proceed.")
        print("\nThe UMBRELLABatch is now fully compatible with HuggingFace Trainer.")
        return 0
    else:
        print("\n⚠️  SOME TESTS FAILED! Review the errors above.")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
