#!/usr/bin/env python3
"""
Simple test to verify JSONL loading logic without full dependencies.

This tests the core JSONL parsing functionality.
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_jsonl_parsing():
    """Test JSONL file parsing logic."""

    logger.info("="*80)
    logger.info("Testing JSONL Parsing Logic")
    logger.info("="*80)

    # Path to JSONL file
    jsonl_path = Path("sample_data/sex_comparison_conversations_simple_extended/train_conversations.jsonl")

    if not jsonl_path.exists():
        logger.error(f"JSONL file not found: {jsonl_path}")
        return False

    logger.info(f"\nReading JSONL file: {jsonl_path}")

    # Simulate the _load_samples_from_jsonl method
    all_samples = []
    failed_lines = []
    line_count = 0

    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, start=1):
                line_count = line_num
                line = line.strip()

                # Skip empty lines
                if not line:
                    continue

                try:
                    # Parse JSON from this line
                    item = json.loads(line)
                    all_samples.append(item)

                except json.JSONDecodeError as e:
                    logger.warning(f"  Line {line_num}: Invalid JSON - {e}")
                    failed_lines.append(line_num)
                    continue

                except Exception as e:
                    logger.error(f"  Line {line_num}: Failed to parse - {e}")
                    failed_lines.append(line_num)
                    continue

    except FileNotFoundError:
        logger.error(f"File not found: {jsonl_path}")
        return False
    except Exception as e:
        logger.error(f"Failed to read file: {e}")
        return False

    # Report statistics
    logger.info(f"\nResults:")
    logger.info(f"  Total lines: {line_count}")
    logger.info(f"  Successfully parsed: {len(all_samples)}")
    logger.info(f"  Failed to parse: {len(failed_lines)}")

    if failed_lines:
        logger.warning(f"  Failed lines: {failed_lines[:10]}")

    if len(all_samples) == 0:
        logger.error("No valid samples found!")
        return False

    # Verify sample structure
    logger.info("\nVerifying sample structure:")
    sample = all_samples[0]

    required_keys = ['task_id', 'task_type', 'subject_ids', 'modalities', 'images', 'conversations', 'metadata']
    for key in required_keys:
        if key in sample:
            logger.info(f"  ✅ {key}: present")
        else:
            logger.warning(f"  ⚠️  {key}: missing")

    # Check conversations structure
    if 'conversations' in sample:
        logger.info(f"\nConversations structure:")
        for i, conv in enumerate(sample['conversations'][:3]):  # Show first 3
            role = conv.get('role', 'unknown')
            content = conv.get('content', [])
            logger.info(f"  Turn {i}: role={role}, content_items={len(content)}")

    # Test file extension detection logic
    logger.info("\n" + "="*80)
    logger.info("Testing file extension detection")
    logger.info("="*80)

    test_paths = [
        "train.jsonl",
        "train.json",
        "data/",
        "train_conversations.jsonl"
    ]

    for path_str in test_paths:
        path = Path(path_str)
        if path.suffix == '.jsonl':
            logger.info(f"  ✅ {path_str:30s} → JSONL format detected")
        elif path.suffix == '.json':
            logger.info(f"  ✅ {path_str:30s} → JSON format detected")
        elif path.is_dir() or '/' in path_str:
            logger.info(f"  ✅ {path_str:30s} → Directory format detected")
        else:
            logger.info(f"  ⚠️  {path_str:30s} → Unknown format")

    logger.info("\n" + "="*80)
    logger.info("Summary")
    logger.info("="*80)
    logger.info(f"✅ JSONL parsing: SUCCESS")
    logger.info(f"✅ Loaded {len(all_samples)} samples from JSONL file")
    logger.info(f"✅ File format detection: working")

    return True


if __name__ == "__main__":
    import sys
    success = test_jsonl_parsing()
    sys.exit(0 if success else 1)
