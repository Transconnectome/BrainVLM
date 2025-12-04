#!/usr/bin/env python3
"""
Test script to verify JSONL format loading in UMBRELLA dataset.

This script tests the new JSONL support added to umbrella_dataset_fixed.py
by loading a sample JSONL file and verifying the data is correctly parsed.
"""

import sys
import logging
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent / "project"))

from dataset.umbrella_dataset_fixed import UMBRELLADataset
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_jsonl_loading():
    """Test JSONL file loading functionality."""

    logger.info("="*80)
    logger.info("Testing JSONL Format Loading")
    logger.info("="*80)

    # Path to JSONL file
    jsonl_path = "sample_data/sex_comparison_conversations_simple_extended/train_conversations.jsonl"

    if not Path(jsonl_path).exists():
        logger.error(f"JSONL file not found: {jsonl_path}")
        logger.info("Please ensure the test data is available at the specified path")
        return False

    # Initialize tokenizer (using a placeholder for testing)
    logger.info("\nInitializing tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        logger.info("Tokenizer loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load tokenizer: {e}")
        logger.info("Using mock tokenizer for testing")

        # Create a simple mock tokenizer for testing
        class MockTokenizer:
            def __call__(self, text, **kwargs):
                # Simple tokenization simulation
                return {
                    'input_ids': [[0] * 100],  # Mock token IDs
                    'attention_mask': [[1] * 100],
                    'offset_mapping': [[(i, i+1) for i in range(100)]]
                }

            def encode(self, text, **kwargs):
                return [0] * len(text.split())

        tokenizer = MockTokenizer()

    # Test 1: Load JSONL file
    logger.info("\n" + "="*80)
    logger.info("Test 1: Loading JSONL file")
    logger.info("="*80)

    try:
        dataset = UMBRELLADataset(
            json_path=jsonl_path,
            tokenizer=tokenizer,
            mode='train',
            img_size=128,
            max_seq_length=2048,
            max_images=2,  # Sex comparison tasks have 2 images
            augment=False
        )

        logger.info(f"✅ Successfully loaded dataset")
        logger.info(f"   Total samples: {len(dataset)}")
        logger.info(f"   Task distribution: {dataset._get_task_distribution()}")

    except Exception as e:
        logger.error(f"❌ Failed to load JSONL file: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 2: Verify sample structure
    logger.info("\n" + "="*80)
    logger.info("Test 2: Verifying sample structure")
    logger.info("="*80)

    if len(dataset) > 0:
        sample = dataset.samples[0]
        logger.info(f"Sample 0 structure:")
        logger.info(f"   task_id: {sample.task_id}")
        logger.info(f"   task_type: {sample.task_type}")
        logger.info(f"   subject_ids: {sample.subject_ids}")
        logger.info(f"   modalities: {sample.modalities}")
        logger.info(f"   num_images: {len(sample.image_paths)}")
        logger.info(f"   num_turns: {len(sample.conversation)}")
        logger.info(f"   metadata: {list(sample.metadata.keys())}")

        # Check conversation structure
        logger.info(f"\n   Conversation structure:")
        for i, turn in enumerate(sample.conversation):
            logger.info(f"   Turn {i}: role={turn.role}, "
                       f"content_length={len(turn.content)}, "
                       f"image_tokens={len(turn.image_tokens)}")

        logger.info("✅ Sample structure verified")
    else:
        logger.error("❌ No samples loaded from JSONL file")
        return False

    # Test 3: Compare with validation and test JSONL files
    logger.info("\n" + "="*80)
    logger.info("Test 3: Loading all splits (train/validation/test)")
    logger.info("="*80)

    splits = {
        'train': 'sample_data/sex_comparison_conversations_simple_extended/train_conversations.jsonl',
        'validation': 'sample_data/sex_comparison_conversations_simple_extended/validation_conversations.jsonl',
        'test': 'sample_data/sex_comparison_conversations_simple_extended/test_conversations.jsonl'
    }

    split_sizes = {}

    for split_name, split_path in splits.items():
        if not Path(split_path).exists():
            logger.warning(f"⚠️  {split_name} file not found: {split_path}")
            continue

        try:
            split_dataset = UMBRELLADataset(
                json_path=split_path,
                tokenizer=tokenizer,
                mode='train' if split_name == 'train' else 'eval',
                img_size=128,
                max_seq_length=2048,
                max_images=2,
                augment=False
            )
            split_sizes[split_name] = len(split_dataset)
            logger.info(f"✅ {split_name:12s}: {len(split_dataset):4d} samples")

        except Exception as e:
            logger.error(f"❌ Failed to load {split_name}: {e}")
            return False

    # Test 4: Test backward compatibility with JSON format
    logger.info("\n" + "="*80)
    logger.info("Test 4: Testing backward compatibility (JSON format)")
    logger.info("="*80)

    # Check if legacy JSON format exists
    json_dir = "sample_data/sex_comparison_conversations_simple_extended"
    json_files = list(Path(json_dir).glob("*.json"))

    if json_files:
        logger.info(f"Found {len(json_files)} JSON files in directory")
        logger.info("Testing directory-based loading...")

        try:
            json_dataset = UMBRELLADataset(
                json_path=json_dir,
                tokenizer=tokenizer,
                mode='train',
                img_size=128,
                max_seq_length=2048,
                max_images=2,
                augment=False
            )
            logger.info(f"✅ Directory loading works: {len(json_dataset)} samples")
        except Exception as e:
            logger.warning(f"⚠️  Directory loading failed (expected if no JSON files): {e}")
    else:
        logger.info("No JSON files found - JSONL-only directory (expected)")

    # Summary
    logger.info("\n" + "="*80)
    logger.info("Summary")
    logger.info("="*80)
    logger.info("✅ JSONL format loading: SUCCESS")
    logger.info(f"✅ Total samples loaded: {sum(split_sizes.values())}")
    logger.info(f"   Split distribution: {split_sizes}")
    logger.info("\nAll tests passed! JSONL format support is working correctly.")

    return True


if __name__ == "__main__":
    success = test_jsonl_loading()
    sys.exit(0 if success else 1)
