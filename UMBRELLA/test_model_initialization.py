#!/usr/bin/env python3
"""
Test Script for LlavaForConditionalGeneration Model Initialization

This script validates that the custom patch embedding and freezing strategy
are correctly implemented.

Usage:
    python test_model_initialization.py --config project/config/umbrella_llava_train.yaml
"""

import sys
import torch
import argparse
import logging
from pathlib import Path

# Add project to path
sys.path.append(str(Path(__file__).parent))

from project.training.main_umbrella_training_fixed import (
    UMBRELLATrainingConfig,
    create_llava_model_with_custom_patch_embed
)

logger = logging.getLogger(__name__)


def test_model_initialization(config_path: str):
    """Test model initialization with custom patch embedding."""

    print("\n" + "=" * 80)
    print("MODEL INITIALIZATION VALIDATION TEST")
    print("=" * 80)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load config
    logger.info(f"Loading config from {config_path}")
    config = UMBRELLATrainingConfig.from_yaml(config_path)

    # Test 1: Model initialization
    print("\n[TEST 1] Model Initialization")
    print("-" * 80)
    try:
        model = create_llava_model_with_custom_patch_embed(config)
        print("✅ PASS: Model initialized successfully")
    except Exception as e:
        print(f"❌ FAIL: Model initialization failed: {e}")
        return False

    # Test 2: Check model type
    print("\n[TEST 2] Model Type Verification")
    print("-" * 80)
    from transformers import LlavaForConditionalGeneration
    if isinstance(model, LlavaForConditionalGeneration):
        print("✅ PASS: Model is LlavaForConditionalGeneration")
    else:
        print(f"❌ FAIL: Model is {type(model)}, expected LlavaForConditionalGeneration")
        return False

    # Test 3: Verify custom patch embedding
    print("\n[TEST 3] Custom Patch Embedding Verification")
    print("-" * 80)
    from project.model.patch_embed import PatchEmbed
    if isinstance(model.vision_tower.vision_model.embeddings, PatchEmbed):
        print("✅ PASS: Custom PatchEmbed is integrated")
    else:
        print(f"❌ FAIL: Embeddings is {type(model.vision_tower.vision_model.embeddings)}")
        return False

    # Test 4: Verify freezing strategy
    print("\n[TEST 4] Freezing Strategy Verification")
    print("-" * 80)

    trainable_params = []
    frozen_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
        else:
            frozen_params.append(name)

    total_params = len(trainable_params) + len(frozen_params)
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())

    print(f"Total parameters: {total_count:,}")
    print(f"Trainable parameters: {trainable_count:,} ({100 * trainable_count / total_count:.2f}%)")
    print(f"Frozen parameters: {total_count - trainable_count:,} ({100 * (total_count - trainable_count) / total_count:.2f}%)")

    # Check that only embeddings are trainable
    embeddings_only = all('embeddings' in name for name in trainable_params)
    if embeddings_only:
        print("✅ PASS: Only custom embeddings are trainable")
    else:
        print("❌ FAIL: Non-embedding parameters are trainable:")
        for name in trainable_params:
            if 'embeddings' not in name:
                print(f"  - {name}")
        return False

    # Check that vision encoder is frozen
    vision_encoder_frozen = all(
        not param.requires_grad
        for name, param in model.vision_tower.vision_model.named_parameters()
        if 'encoder' in name or 'layernorm' in name
    )
    if vision_encoder_frozen:
        print("✅ PASS: Vision encoder is frozen")
    else:
        print("❌ FAIL: Vision encoder has trainable parameters")
        return False

    # Check that language model is frozen
    lm_frozen = all(
        not param.requires_grad
        for name, param in model.named_parameters()
        if 'language_model' in name or 'lm_head' in name
    )
    if lm_frozen:
        print("✅ PASS: Language model is frozen")
    else:
        print("❌ FAIL: Language model has trainable parameters")
        return False

    # Check that multi-modal projector is frozen
    projector_frozen = all(
        not param.requires_grad
        for name, param in model.named_parameters()
        if 'multi_modal_projector' in name
    )
    if projector_frozen:
        print("✅ PASS: Multi-modal projector is frozen")
    else:
        print("❌ FAIL: Multi-modal projector has trainable parameters")
        return False

    # Test 5: Test forward pass with dummy data
    print("\n[TEST 5] Forward Pass Test (3D T1 Input)")
    print("-" * 80)
    try:
        # Create dummy 3D T1 input (batch_size=2)
        dummy_t1 = torch.randn(2, 1, 96, 96, 96)

        # Forward pass through vision tower embeddings
        embeddings_output = model.vision_tower.vision_model.embeddings(dummy_t1)

        print(f"Input shape: {dummy_t1.shape}")
        print(f"Embeddings output shape: {embeddings_output.shape}")
        print("✅ PASS: Forward pass succeeded for 3D T1 input")
    except Exception as e:
        print(f"❌ FAIL: Forward pass failed: {e}")
        return False

    # Test 6: Test forward pass with 4D rsfMRI data
    print("\n[TEST 6] Forward Pass Test (4D rsfMRI Input)")
    print("-" * 80)
    try:
        # Create dummy 4D rsfMRI input (batch_size=2)
        dummy_rsfmri = torch.randn(2, 1, 96, 96, 96, 24)

        # Forward pass through vision tower embeddings
        embeddings_output = model.vision_tower.vision_model.embeddings(dummy_rsfmri)

        print(f"Input shape: {dummy_rsfmri.shape}")
        print(f"Embeddings output shape: {embeddings_output.shape}")
        print("✅ PASS: Forward pass succeeded for 4D rsfMRI input")
    except Exception as e:
        print(f"❌ FAIL: Forward pass failed: {e}")
        return False

    # Test 7: Gradient checkpointing
    print("\n[TEST 7] Gradient Checkpointing Verification")
    print("-" * 80)
    if hasattr(model, 'gradient_checkpointing') and model.gradient_checkpointing:
        print("✅ PASS: Gradient checkpointing is enabled")
    elif hasattr(model, 'is_gradient_checkpointing') and model.is_gradient_checkpointing:
        print("✅ PASS: Gradient checkpointing is enabled")
    else:
        print("⚠️  WARNING: Could not verify gradient checkpointing status")

    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print("✅ All tests PASSED")
    print(f"\nTrainable Parameters: {trainable_count:,} / {total_count:,} ({100 * trainable_count / total_count:.2f}%)")
    print("\nTrainable Components:")
    for name in trainable_params[:10]:  # Show first 10
        print(f"  - {name}")
    if len(trainable_params) > 10:
        print(f"  ... and {len(trainable_params) - 10} more")
    print("\n✅ Model is ready for training!")
    print("=" * 80 + "\n")

    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test LlavaForConditionalGeneration model initialization"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='project/config/umbrella_llava_train.yaml',
        help='Path to config file'
    )

    args = parser.parse_args()

    # Validate config exists
    if not Path(args.config).exists():
        print(f"❌ Config file not found: {args.config}")
        sys.exit(1)

    # Run tests
    success = test_model_initialization(args.config)

    if success:
        print("\n✅ ALL TESTS PASSED - Model initialization is correct!")
        sys.exit(0)
    else:
        print("\n❌ TESTS FAILED - Check error messages above")
        sys.exit(1)


if __name__ == "__main__":
    main()
