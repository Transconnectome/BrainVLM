#!/usr/bin/env python3
"""
Verification Script for UMBRELLA Configuration Consolidation

This script verifies that the configuration consolidation is working correctly
by checking:
1. UMBRELLATrainingConfig has all required attributes
2. to_training_args() factory method exists and works
3. UMBRELLATrainingArgs instances have all required attributes
4. No AttributeError occurs during trainer initialization

Run this script to verify the consolidation was successful.
"""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent / "project"
sys.path.insert(0, str(project_root))

def verify_config_class():
    """Verify UMBRELLATrainingConfig has all required attributes."""
    print("\n" + "=" * 80)
    print("STEP 1: Verifying UMBRELLATrainingConfig Class")
    print("=" * 80)

    from training.main_umbrella_training_fixed import UMBRELLATrainingConfig

    required_attributes = [
        # Original attributes
        'model_name', 'tokenizer_name',
        'train_json_path', 'eval_json_path', 'task_filter',
        'modality', 'img_size', 'patch_size',
        'T1_img_size', 'T1_patch_size',
        'rsfMRI_img_size', 'rsfMRI_patch_size',
        'batch_size', 'gradient_accumulation_steps', 'learning_rate',
        'num_epochs', 'max_seq_length', 'max_images_per_sample',
        'enable_memory_aware_batching', 'memory_budget_gb',
        'gradient_checkpointing', 'mixed_precision',
        'output_dir', 'logging_steps', 'save_steps', 'eval_steps',
        'save_total_limit', 'warmup_steps',
        'use_wandb', 'wandb_project', 'wandb_api_key',

        # NEW: UMBRELLA-specific attributes
        'mask_human_turns', 'mask_padding_tokens',
        'enable_task_aware_loss', 'task_type_weights',
        'enable_dummy_loss', 'dummy_loss_weight',
        'log_turn_distribution', 'log_image_statistics', 'log_memory_usage',
        'normalize_gradients_by_batch_size', 'base_batch_size',
    ]

    config = UMBRELLATrainingConfig()

    missing = []
    for attr in required_attributes:
        if not hasattr(config, attr):
            missing.append(attr)
            print(f"  ✗ Missing: {attr}")
        else:
            print(f"  ✓ Has: {attr}")

    if missing:
        print(f"\n❌ FAILED: Missing {len(missing)} attributes: {missing}")
        return False
    else:
        print(f"\n✓ SUCCESS: All {len(required_attributes)} attributes present")
        return True


def verify_factory_method():
    """Verify to_training_args() factory method exists and works."""
    print("\n" + "=" * 80)
    print("STEP 2: Verifying to_training_args() Factory Method")
    print("=" * 80)

    from training.main_umbrella_training_fixed import UMBRELLATrainingConfig
    from training.umbrella_trainer import UMBRELLATrainingArgs

    # Check method exists
    config = UMBRELLATrainingConfig()
    if not hasattr(config, 'to_training_args'):
        print("❌ FAILED: to_training_args() method not found")
        return False

    print("✓ to_training_args() method exists")

    # Check method works
    try:
        training_args = config.to_training_args(eval_dataset_available=False)
        print("✓ to_training_args() executes without error")
    except Exception as e:
        print(f"❌ FAILED: to_training_args() raised exception: {e}")
        return False

    # Check return type
    if not isinstance(training_args, UMBRELLATrainingArgs):
        print(f"❌ FAILED: Wrong return type: {type(training_args)}")
        return False

    print(f"✓ Returns correct type: {type(training_args).__name__}")

    print("\n✓ SUCCESS: Factory method works correctly")
    return True


def verify_training_args_attributes():
    """Verify UMBRELLATrainingArgs instances have all required attributes."""
    print("\n" + "=" * 80)
    print("STEP 3: Verifying UMBRELLATrainingArgs Attributes")
    print("=" * 80)

    from training.main_umbrella_training_fixed import UMBRELLATrainingConfig

    config = UMBRELLATrainingConfig()
    training_args = config.to_training_args()

    umbrella_specific_attributes = [
        'mask_human_turns',
        'mask_padding_tokens',
        'enable_task_aware_loss',
        'task_type_weights',
        'enable_memory_aware_batching',
        'memory_budget_gb',
        'enable_dummy_loss',
        'dummy_loss_weight',
        'log_turn_distribution',
        'log_image_statistics',
        'log_memory_usage',
        'normalize_gradients_by_batch_size',
        'base_batch_size',
    ]

    missing = []
    for attr in umbrella_specific_attributes:
        if not hasattr(training_args, attr):
            missing.append(attr)
            print(f"  ✗ Missing: {attr}")
        else:
            value = getattr(training_args, attr)
            print(f"  ✓ Has: {attr} = {value}")

    if missing:
        print(f"\n❌ FAILED: Missing {len(missing)} attributes: {missing}")
        return False
    else:
        print(f"\n✓ SUCCESS: All {len(umbrella_specific_attributes)} UMBRELLA-specific attributes present")
        return True


def verify_attribute_values():
    """Verify attribute values are correctly mapped from config to args."""
    print("\n" + "=" * 80)
    print("STEP 4: Verifying Attribute Value Mapping")
    print("=" * 80)

    from training.main_umbrella_training_fixed import UMBRELLATrainingConfig

    config = UMBRELLATrainingConfig(
        batch_size=4,
        learning_rate=1e-4,
        num_epochs=10,
        mask_human_turns=False,
        enable_task_aware_loss=False,
        task_type_weights={"test": 1.5},
        enable_dummy_loss=False,
        dummy_loss_weight=0.2,
    )

    training_args = config.to_training_args()

    checks = [
        ('per_device_train_batch_size', 'batch_size', 4),
        ('learning_rate', 'learning_rate', 1e-4),
        ('num_train_epochs', 'num_epochs', 10),
        ('mask_human_turns', 'mask_human_turns', False),
        ('enable_task_aware_loss', 'enable_task_aware_loss', False),
        ('task_type_weights', 'task_type_weights', {"test": 1.5}),
        ('enable_dummy_loss', 'enable_dummy_loss', False),
        ('dummy_loss_weight', 'dummy_loss_weight', 0.2),
    ]

    failures = []
    for args_attr, config_attr, expected_value in checks:
        actual_value = getattr(training_args, args_attr)
        if actual_value != expected_value:
            failures.append((args_attr, expected_value, actual_value))
            print(f"  ✗ {args_attr}: expected {expected_value}, got {actual_value}")
        else:
            print(f"  ✓ {args_attr} = {actual_value}")

    if failures:
        print(f"\n❌ FAILED: {len(failures)} attribute value mismatches")
        return False
    else:
        print(f"\n✓ SUCCESS: All attribute values correctly mapped")
        return True


def verify_no_attribute_error():
    """Verify that accessing trainer-required attributes doesn't raise AttributeError."""
    print("\n" + "=" * 80)
    print("STEP 5: Verifying No AttributeError in Trainer Usage Pattern")
    print("=" * 80)

    from training.main_umbrella_training_fixed import UMBRELLATrainingConfig

    config = UMBRELLATrainingConfig()
    training_args = config.to_training_args()

    # These are the attributes that were causing AttributeError before
    critical_attributes = [
        'task_type_weights',
        'enable_task_aware_loss',
        'enable_dummy_loss',
        'dummy_loss_weight',
        'mask_human_turns',
        'log_turn_distribution',
        'log_image_statistics',
        'normalize_gradients_by_batch_size',
        'base_batch_size',
    ]

    try:
        for attr in critical_attributes:
            value = getattr(training_args, attr)
            print(f"  ✓ Accessing {attr}: {value}")

        print("\n✓ SUCCESS: No AttributeError when accessing critical attributes")
        return True

    except AttributeError as e:
        print(f"\n❌ FAILED: AttributeError: {e}")
        return False


def main():
    """Run all verification checks."""
    print("\n" + "=" * 80)
    print("UMBRELLA Configuration Consolidation Verification")
    print("=" * 80)

    results = []

    # Run all verification steps
    results.append(("Config Class Attributes", verify_config_class()))
    results.append(("Factory Method", verify_factory_method()))
    results.append(("Training Args Attributes", verify_training_args_attributes()))
    results.append(("Attribute Value Mapping", verify_attribute_values()))
    results.append(("No AttributeError", verify_no_attribute_error()))

    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(result[1] for result in results)

    print("\n" + "=" * 80)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("=" * 80)
        print("\nConfiguration consolidation is working correctly!")
        print("You can now use:")
        print("  config = UMBRELLATrainingConfig.from_yaml('config.yaml')")
        print("  training_args = config.to_training_args()")
        print("  trainer = UMBRELLATrainer(args=training_args, ...)")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("=" * 80)
        print("\nConfiguration consolidation has issues.")
        print("Please review the failed tests above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
