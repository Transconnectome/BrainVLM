"""
Test script for UMBRELLA tokenization and dataset functionality.

Validates:
1. JSON v2 format parsing
2. LLaVA-Next format generation
3. Label masking correctness
4. Image token handling
5. Config-based image size support
"""

import sys
import json
import torch
import logging
from pathlib import Path
from transformers import AutoTokenizer

# Add project path
sys.path.append(str(Path(__file__).parent))

from project.dataset.umbrella_dataset_fixed import UMBRELLADataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_json_parsing():
    """Test 1: Verify JSON v2 format parsing."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 1: JSON v2 Format Parsing")
    logger.info("=" * 80)

    # Load sample JSON
    sample_path = Path("./sample_data/sex_comparison_conversations/test")
    json_files = list(sample_path.glob("*.json"))

    if not json_files:
        logger.error(f"No JSON files found in {sample_path}")
        return False

    sample_file = json_files[0]
    logger.info(f"Testing with: {sample_file}")

    with open(sample_file, 'r') as f:
        data = json.load(f)

    # Validate JSON structure
    assert 'conversations' in data, "Missing 'conversations' key"
    assert isinstance(data['conversations'], list), "conversations must be list"

    for turn in data['conversations']:
        assert 'role' in turn, "Turn missing 'role'"
        assert turn['role'] in ['user', 'assistant'], f"Invalid role: {turn['role']}"
        assert 'content' in turn, "Turn missing 'content'"
        assert isinstance(turn['content'], list), "content must be list"

        for item in turn['content']:
            assert 'type' in item, "Content item missing 'type'"
            assert item['type'] in ['text', 'image'], f"Invalid type: {item['type']}"

    logger.info("✅ JSON v2 format validation passed")
    return True


def test_llava_format_generation():
    """Test 2: Verify LLaVA-Next format generation."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: LLaVA-Next Format Generation")
    logger.info("=" * 80)

    # Create temporary tokenizer
    tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-interleave-qwen-0.5b-hf")

    # Add special tokens
    special_tokens = {
        'additional_special_tokens': ['<|im_start|>', '<|im_end|>', '<image>']
    }
    tokenizer.add_special_tokens(special_tokens)

    # Create dataset
    sample_path = Path("./sample_data/sex_comparison_conversations/test")
    json_files = list(sample_path.glob("*.json"))

    if not json_files:
        logger.error("No JSON files found")
        return False

    # Merge JSON files
    samples = []
    for json_file in json_files[:3]:  # Test with first 3 samples
        with open(json_file, 'r') as f:
            samples.append(json.load(f))

    test_json = "./test_merged.json"
    with open(test_json, 'w') as f:
        json.dump(samples, f)

    dataset = UMBRELLADataset(
        json_path=test_json,
        tokenizer=tokenizer,
        mode='eval',
        img_size=[120, 120, 120],
        modality='T1'
    )

    # Get sample
    sample = dataset[0]
    decoded = tokenizer.decode(sample['input_ids'], skip_special_tokens=False)

    logger.info(f"\nDecoded conversation:\n{decoded}")

    # Validate format
    assert '<|im_start|>user' in decoded, "Missing '<|im_start|>user'"
    assert '<|im_end|>' in decoded, "Missing '<|im_end|>'"
    assert '<|im_start|>assistant' in decoded, "Missing '<|im_start|>assistant'"

    # Should NOT contain old format
    assert 'USER:' not in decoded, "Found old format 'USER:'"
    assert 'ASSISTANT:' not in decoded, "Found old format 'ASSISTANT:'"
    assert 'human' not in decoded.lower() or '<|im_start|>user' in decoded, "Found 'human' role"

    logger.info("✅ LLaVA-Next format generation passed")

    # Cleanup
    Path(test_json).unlink()

    return True


def test_label_masking():
    """Test 3: Verify label masking correctness."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Label Masking")
    logger.info("=" * 80)

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-interleave-qwen-0.5b-hf")
    special_tokens = {
        'additional_special_tokens': ['<|im_start|>', '<|im_end|>', '<image>']
    }
    tokenizer.add_special_tokens(special_tokens)

    # Create dataset
    sample_path = Path("./sample_data/sex_comparison_conversations/test")
    json_files = list(sample_path.glob("*.json"))

    if not json_files:
        logger.error("No JSON files found")
        return False

    samples = []
    for json_file in json_files[:1]:
        with open(json_file, 'r') as f:
            samples.append(json.load(f))

    test_json = "./test_merged.json"
    with open(test_json, 'w') as f:
        json.dump(samples, f)

    dataset = UMBRELLADataset(
        json_path=test_json,
        tokenizer=tokenizer,
        mode='eval',
        img_size=[120, 120, 120],
        modality='T1'
    )

    # Get sample
    sample = dataset[0]
    input_ids = sample['input_ids']
    labels = sample['labels']
    attention_mask = sample['attention_mask']

    # Decode for inspection
    decoded_input = tokenizer.decode(input_ids, skip_special_tokens=False)
    logger.info(f"\nInput text:\n{decoded_input}")

    # Check masking
    num_active = (labels != -100).sum().item()
    num_masked = (labels == -100).sum().item()
    num_padding = (attention_mask == 0).sum().item()

    logger.info(f"\nMasking statistics:")
    logger.info(f"  Active tokens: {num_active}")
    logger.info(f"  Masked tokens: {num_masked}")
    logger.info(f"  Padding tokens: {num_padding}")

    # Validate
    assert num_active > 0, "No active tokens (all masked)"
    assert num_masked > num_active, "More active than masked (incorrect masking)"

    # All padding should be masked
    padding_labels = labels[attention_mask == 0]
    assert (padding_labels == -100).all(), "Padding not properly masked"

    logger.info("✅ Label masking validation passed")

    # Cleanup
    Path(test_json).unlink()

    return True


def test_image_token_handling():
    """Test 4: Verify image token handling."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: Image Token Handling")
    logger.info("=" * 80)

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-interleave-qwen-0.5b-hf")
    special_tokens = {
        'additional_special_tokens': ['<|im_start|>', '<|im_end|>', '<image>']
    }
    tokenizer.add_special_tokens(special_tokens)

    # Create dataset
    sample_path = Path("./sample_data/sex_comparison_conversations/test")
    json_files = list(sample_path.glob("*.json"))

    if not json_files:
        logger.error("No JSON files found")
        return False

    samples = []
    for json_file in json_files[:1]:
        with open(json_file, 'r') as f:
            samples.append(json.load(f))

    test_json = "./test_merged.json"
    with open(test_json, 'w') as f:
        json.dump(samples, f)

    dataset = UMBRELLADataset(
        json_path=test_json,
        tokenizer=tokenizer,
        mode='eval',
        img_size=[120, 120, 120],
        modality='T1'
    )

    # Get sample
    sample = dataset[0]
    original_sample = dataset.samples[0]
    decoded = tokenizer.decode(sample['input_ids'], skip_special_tokens=False)

    # Count image tokens
    num_image_tokens = decoded.count('<image>')
    expected_images = len(original_sample.image_paths)

    logger.info(f"\nImage token analysis:")
    logger.info(f"  Expected images: {expected_images}")
    logger.info(f"  <image> tokens found: {num_image_tokens}")

    assert num_image_tokens == expected_images, \
        f"Image token count mismatch: expected {expected_images}, found {num_image_tokens}"

    # Check no modality-specific tokens
    assert '<image_sMRI>' not in decoded, "Found modality-specific token <image_sMRI>"
    assert '<image_fMRI>' not in decoded, "Found modality-specific token <image_fMRI>"
    assert '<sub1-image>' not in decoded or num_image_tokens > 0, "Found subject token"

    logger.info("✅ Image token handling passed")

    # Cleanup
    Path(test_json).unlink()

    return True


def test_config_image_sizes():
    """Test 5: Verify config-based image size support."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 5: Config-based Image Sizes")
    logger.info("=" * 80)

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-interleave-qwen-0.5b-hf")
    special_tokens = {
        'additional_special_tokens': ['<|im_start|>', '<|im_end|>', '<image>']
    }
    tokenizer.add_special_tokens(special_tokens)

    # Test different image size formats
    test_cases = [
        {
            'name': '3D sMRI (list)',
            'img_size': [120, 120, 120],
            'expected_shape': [120, 120, 120],
            'expected_4d': False
        },
        {
            'name': '4D fMRI (list)',
            'img_size': [96, 96, 96, 24],
            'expected_shape': [96, 96, 96, 24],
            'expected_4d': True
        },
        {
            'name': 'Single int (legacy)',
            'img_size': 128,
            'expected_shape': [128, 128, 128],
            'expected_4d': False
        }
    ]

    # Create minimal test JSON
    test_sample = {
        "task_id": "test_001",
        "conversations": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Test"}]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Response"}]
            }
        ],
        "images": []
    }

    test_json = "./test_config.json"
    with open(test_json, 'w') as f:
        json.dump([test_sample], f)

    for test_case in test_cases:
        logger.info(f"\nTesting: {test_case['name']}")

        dataset = UMBRELLADataset(
            json_path=test_json,
            tokenizer=tokenizer,
            mode='eval',
            img_size=test_case['img_size'],
            modality='T1'
        )

        # Validate
        assert dataset.img_size == test_case['expected_shape'], \
            f"Image size mismatch: expected {test_case['expected_shape']}, got {dataset.img_size}"

        assert dataset.is_4d == test_case['expected_4d'], \
            f"4D flag mismatch: expected {test_case['expected_4d']}, got {dataset.is_4d}"

        logger.info(f"  ✅ {test_case['name']}: img_size={dataset.img_size}, is_4d={dataset.is_4d}")

    # Cleanup
    Path(test_json).unlink()

    logger.info("\n✅ Config-based image sizes validation passed")
    return True


def run_all_tests():
    """Run all validation tests."""
    logger.info("\n" + "=" * 80)
    logger.info("UMBRELLA TOKENIZATION VALIDATION SUITE")
    logger.info("=" * 80)

    tests = [
        ("JSON v2 Parsing", test_json_parsing),
        ("LLaVA-Next Format", test_llava_format_generation),
        ("Label Masking", test_label_masking),
        ("Image Token Handling", test_image_token_handling),
        ("Config Image Sizes", test_config_image_sizes),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"\n❌ {test_name} FAILED: {e}", exc_info=True)
            results.append((test_name, False))

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {test_name}")

    total = len(results)
    passed = sum(1 for _, r in results if r)

    logger.info(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        logger.info("\n🎉 ALL TESTS PASSED!")
        return True
    else:
        logger.error(f"\n⚠️ {total - passed} TESTS FAILED")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
