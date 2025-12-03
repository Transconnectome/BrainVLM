#!/usr/bin/env python3
"""
Standalone test to verify the image_mask fix.

This tests the UMBRELLABatch class in isolation without importing the full project.

CRITICAL FIX (2025-12-03):
- UMBRELLABatch now filters out None fields when unpacked with **
- This prevents TypeError: unexpected keyword argument 'image_mask'
"""

import torch
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class UMBRELLABatch:
    """
    Test version of UMBRELLABatch with the fix applied.

    CRITICAL FIX: keys(), values(), items(), __iter__() filter out None fields.
    """

    pixel_values: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    image_mask: torch.Tensor
    num_images_per_sample: List[int]
    task_types: List[str]
    task_ids: torch.Tensor
    sample_indices: List[int]
    metadata: List[Dict[str, Any]]

    def __len__(self):
        return self.input_ids.shape[0]

    def __iter__(self):
        """Filter out None fields (popped fields)."""
        all_fields = [
            'pixel_values', 'input_ids', 'attention_mask', 'labels',
            'image_mask', 'num_images_per_sample', 'task_types',
            'task_ids', 'sample_indices', 'metadata'
        ]
        for field in all_fields:
            if hasattr(self, field) and getattr(self, field) is not None:
                yield field

    def __getitem__(self, key: str):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError(f"UMBRELLABatch has no field '{key}'")

    def get(self, key: str, default=None):
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def pop(self, key: str, default=None):
        """Pop a field by setting it to None."""
        try:
            value = self.__getitem__(key)
            if hasattr(self, key):
                object.__setattr__(self, key, None)
            return value
        except KeyError:
            return default

    def keys(self):
        """Return only non-None field names."""
        all_fields = [
            'pixel_values', 'input_ids', 'attention_mask', 'labels',
            'image_mask', 'num_images_per_sample', 'task_types',
            'task_ids', 'sample_indices', 'metadata'
        ]
        return [field for field in all_fields
                if hasattr(self, field) and getattr(self, field) is not None]

    def values(self):
        """Return only non-None field values."""
        return [getattr(self, key) for key in self.keys()]

    def items(self):
        """Return only non-None (key, value) pairs."""
        return [(key, getattr(self, key)) for key in self.keys()]


def test_pop_removes_from_keys():
    """Test that pop() removes fields from keys()."""
    print("=" * 70)
    print("TEST 1: pop() removes fields from keys()")
    print("=" * 70)

    batch = UMBRELLABatch(
        pixel_values=torch.zeros(2, 1, 1, 224, 224, 8),
        input_ids=torch.zeros(2, 10, dtype=torch.long),
        attention_mask=torch.ones(2, 10, dtype=torch.long),
        labels=torch.zeros(2, 10, dtype=torch.long),
        image_mask=torch.ones(2, 1, dtype=torch.long),
        num_images_per_sample=[1, 1],
        task_types=['T1', 'T1'],
        task_ids=torch.zeros(2, dtype=torch.long),
        sample_indices=[0, 1],
        metadata=[{}, {}]
    )

    print("\nBefore pop():")
    print(f"  Keys: {batch.keys()}")
    print(f"  'image_mask' in keys: {'image_mask' in batch.keys()}")

    # Pop image_mask
    popped = batch.pop('image_mask', None)

    print(f"\nAfter pop('image_mask'):")
    print(f"  Keys: {batch.keys()}")
    print(f"  'image_mask' in keys: {'image_mask' in batch.keys()}")

    assert 'image_mask' not in batch.keys(), "FAIL: image_mask still in keys!"
    print("✅ PASS")


def test_model_call_simulation():
    """Simulate model(**inputs) call."""
    print("\n" + "=" * 70)
    print("TEST 2: Simulate model(**inputs)")
    print("=" * 70)

    batch = UMBRELLABatch(
        pixel_values=torch.zeros(2, 1, 1, 224, 224, 8),
        input_ids=torch.zeros(2, 10, dtype=torch.long),
        attention_mask=torch.ones(2, 10, dtype=torch.long),
        labels=torch.zeros(2, 10, dtype=torch.long),
        image_mask=torch.ones(2, 1, dtype=torch.long),
        num_images_per_sample=[1, 1],
        task_types=['T1', 'T1'],
        task_ids=torch.zeros(2, dtype=torch.long),
        sample_indices=[0, 1],
        metadata=[{}, {}]
    )

    # Simulate compute_loss() extracting metadata
    labels = batch.pop('labels', None)
    task_types = batch.pop('task_types', [])
    task_ids = batch.pop('task_ids', None)
    image_mask = batch.pop('image_mask', None)
    num_images_per_sample = batch.pop('num_images_per_sample', None)
    sample_indices = batch.pop('sample_indices', None)
    metadata = batch.pop('metadata', None)

    print(f"\nRemaining keys: {batch.keys()}")

    # Mock model that only accepts standard inputs
    def mock_model(**kwargs):
        allowed = {'pixel_values', 'input_ids', 'attention_mask'}
        received = set(kwargs.keys())
        unexpected = received - allowed

        if unexpected:
            raise TypeError(f"Unexpected keyword arguments: {unexpected}")

        return {"logits": torch.zeros(2, 10, 32000)}

    try:
        outputs = mock_model(**batch)
        print("✅ PASS: Model call succeeded")
    except TypeError as e:
        print(f"❌ FAIL: {e}")
        raise


def test_dict_unpacking():
    """Test ** unpacking behavior."""
    print("\n" + "=" * 70)
    print("TEST 3: Dict unpacking with **")
    print("=" * 70)

    batch = UMBRELLABatch(
        pixel_values=torch.zeros(2, 1, 1, 224, 224, 8),
        input_ids=torch.zeros(2, 10, dtype=torch.long),
        attention_mask=torch.ones(2, 10, dtype=torch.long),
        labels=torch.zeros(2, 10, dtype=torch.long),
        image_mask=torch.ones(2, 1, dtype=torch.long),
        num_images_per_sample=[1, 1],
        task_types=['T1', 'T1'],
        task_ids=torch.zeros(2, dtype=torch.long),
        sample_indices=[0, 1],
        metadata=[{}, {}]
    )

    # Pop all metadata
    for key in ['labels', 'image_mask', 'task_types', 'task_ids',
                'num_images_per_sample', 'sample_indices', 'metadata']:
        batch.pop(key)

    def capture(**kwargs):
        return set(kwargs.keys())

    received_keys = capture(**batch)
    expected_keys = {'pixel_values', 'input_ids', 'attention_mask'}

    print(f"\nExpected: {expected_keys}")
    print(f"Received: {received_keys}")

    assert received_keys == expected_keys, f"Mismatch!"
    print("✅ PASS")


def test_iter_filters_none():
    """Test __iter__ filters None fields."""
    print("\n" + "=" * 70)
    print("TEST 4: __iter__ filters None fields")
    print("=" * 70)

    batch = UMBRELLABatch(
        pixel_values=torch.zeros(2, 1, 1, 224, 224, 8),
        input_ids=torch.zeros(2, 10, dtype=torch.long),
        attention_mask=torch.ones(2, 10, dtype=torch.long),
        labels=torch.zeros(2, 10, dtype=torch.long),
        image_mask=torch.ones(2, 1, dtype=torch.long),
        num_images_per_sample=[1, 1],
        task_types=['T1', 'T1'],
        task_ids=torch.zeros(2, dtype=torch.long),
        sample_indices=[0, 1],
        metadata=[{}, {}]
    )

    print(f"\nBefore pop: {list(batch)}")

    batch.pop('labels')
    batch.pop('image_mask')
    batch.pop('metadata')

    after = list(batch)
    print(f"After pop:  {after}")

    assert 'labels' not in after
    assert 'image_mask' not in after
    assert 'metadata' not in after
    print("✅ PASS")


def test_items_filters_none():
    """Test items() filters None fields."""
    print("\n" + "=" * 70)
    print("TEST 5: items() filters None fields")
    print("=" * 70)

    batch = UMBRELLABatch(
        pixel_values=torch.zeros(2, 1, 1, 224, 224, 8),
        input_ids=torch.zeros(2, 10, dtype=torch.long),
        attention_mask=torch.ones(2, 10, dtype=torch.long),
        labels=torch.zeros(2, 10, dtype=torch.long),
        image_mask=torch.ones(2, 1, dtype=torch.long),
        num_images_per_sample=[1, 1],
        task_types=['T1', 'T1'],
        task_ids=torch.zeros(2, dtype=torch.long),
        sample_indices=[0, 1],
        metadata=[{}, {}]
    )

    batch.pop('image_mask')
    batch.pop('num_images_per_sample')

    items = list(batch.items())
    item_keys = [k for k, v in items]

    print(f"\nitems() keys: {item_keys}")

    assert 'image_mask' not in item_keys
    assert 'num_images_per_sample' not in item_keys

    # Verify no None values
    for k, v in items:
        assert v is not None, f"{k} should not be None"

    print("✅ PASS")


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("UMBRELLA BATCH IMAGE_MASK FIX VERIFICATION")
    print("=" * 70)

    try:
        test_pop_removes_from_keys()
        test_model_call_simulation()
        test_dict_unpacking()
        test_iter_filters_none()
        test_items_filters_none()

        print("\n" + "=" * 70)
        print("ALL TESTS PASSED ✅")
        print("=" * 70)
        print("\nThe fix successfully prevents image_mask from being passed to model.")
        print("Popped fields are correctly filtered from dict unpacking.")

    except Exception as e:
        print("\n" + "=" * 70)
        print("TEST FAILED ❌")
        print("=" * 70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
