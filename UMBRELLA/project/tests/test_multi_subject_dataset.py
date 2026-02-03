"""
Unit tests for multi-subject sequential comparison in T1JSONDataset.

Tests the new multi-subject functionality while ensuring backward compatibility
with single-subject samples.

Test Coverage:
- Single-subject loading (backward compatibility)
- Multi-subject format validation
- Image placeholder conversion
- Multi-turn conversation formatting
- Tokenization with multiple images
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, List, Any
import unittest
from unittest.mock import Mock, patch, MagicMock

import torch
import numpy as np

# Assuming the dataset is in parent directory
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.t1_json_dataset import T1JSONDataset


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __call__(self, text, **kwargs):
        """Mock tokenization."""
        return {
            'input_ids': torch.tensor([1, 2, 3, 4, 5]),
            'attention_mask': torch.tensor([1, 1, 1, 1, 1])
        }

    def encode(self, text, **kwargs):
        """Mock encoding."""
        return [1, 2, 3, 4, 5]


def create_dummy_nifti(shape=(128, 128, 128)):
    """Create a dummy NIfTI file for testing."""
    try:
        import nibabel as nib
        data = np.random.randn(*shape).astype(np.float32)
        img = nib.Nifti1Image(data, np.eye(4))
        return img
    except ImportError:
        # If nibabel not available, return None (will skip NIfTI tests)
        return None


class TestSingleSubjectBackwardCompatibility(unittest.TestCase):
    """Test that single-subject samples still work (backward compatibility)."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_root = Path(self.temp_dir.name)

        # Create sample image files
        self.image_path_1 = self.data_root / "sub-001_T1.nii.gz"
        self.image_path_1.touch()

    def tearDown(self):
        """Clean up."""
        self.temp_dir.cleanup()

    def test_single_subject_string_format(self):
        """Test that single subject_id as string is recognized."""
        json_data = {
            "samples": [
                {
                    "subject_id": "sub-001",  # String, not list
                    "task_id": "diagnosis",
                    "modality_paths": {
                        "image_sMRI": str(self.image_path_1)
                    },
                    "conversations": [
                        {"from": "human", "value": "<image_sMRI>\nWhat do you see?"},
                        {"from": "gpt", "value": "This is a normal brain."}
                    ],
                    "metadata": {"age": 25, "sex": 1}
                }
            ]
        }

        json_path = self.data_root / "data.json"
        with open(json_path, 'w') as f:
            json.dump(json_data, f)

        # Create mock dataset (avoid actual image loading)
        with patch('dataset.t1_json_dataset.resolve_path', return_value=str(self.image_path_1)):
            with patch.object(T1JSONDataset, '_load_and_process_image',
                            return_value=torch.zeros((1, 128, 128, 128))):
                dataset = T1JSONDataset(
                    json_file=str(json_path),
                    data_root=str(self.data_root),
                    tokenizer=MockTokenizer(),
                    img_size=128
                )

                # Should use _get_single_item
                sample = dataset[0]

                # Verify structure
                self.assertEqual(sample['subject_id'], 'sub-001')
                self.assertEqual(sample['task_id'], 'diagnosis')
                self.assertIn('pixel_values', sample)
                self.assertIn('input_ids', sample)


class TestMultiSubjectFormat(unittest.TestCase):
    """Test multi-subject format recognition and validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_root = Path(self.temp_dir.name)

        # Create sample image files
        self.image_path_1 = self.data_root / "sub-001_T1.nii.gz"
        self.image_path_2 = self.data_root / "sub-002_T1.nii.gz"
        self.image_path_1.touch()
        self.image_path_2.touch()

    def tearDown(self):
        """Clean up."""
        self.temp_dir.cleanup()

    def test_multi_subject_list_format(self):
        """Test that multi-subject format with list is recognized."""
        json_data = {
            "samples": [
                {
                    "subject_id": ["sub-001", "sub-002"],  # List format
                    "task_id": "comparison",
                    "modality_paths": {
                        "image_sMRI": [
                            str(self.image_path_1),
                            str(self.image_path_2)
                        ]
                    },
                    "conversations": [
                        {"from": "human", "value": "See the reference subject:\n<sub1-image>"},
                        {"from": "gpt", "value": "Noted."},
                        {"from": "human", "value": "Now see the target:\n<sub2-image>"},
                        {"from": "gpt", "value": "Comparison noted."}
                    ],
                    "metadata": {}
                }
            ]
        }

        json_path = self.data_root / "data.json"
        with open(json_path, 'w') as f:
            json.dump(json_data, f)

        with patch('dataset.t1_json_dataset.resolve_path',
                  side_effect=lambda p, _: p):
            with patch.object(T1JSONDataset, '_load_and_process_image',
                            return_value=torch.zeros((1, 128, 128, 128))):
                dataset = T1JSONDataset(
                    json_file=str(json_path),
                    data_root=str(self.data_root),
                    tokenizer=MockTokenizer(),
                    img_size=128
                )

                # Should use _get_multi_subject_sequential
                sample = dataset[0]

                # Verify structure
                self.assertEqual(sample['subject_ids'], ['sub-001', 'sub-002'])
                self.assertEqual(sample['num_images'], 2)
                # pixel_values should contain list of images
                self.assertIn('T1', sample['pixel_values'])
                self.assertIsInstance(sample['pixel_values']['T1'], list)
                self.assertEqual(len(sample['pixel_values']['T1']), 2)

    def test_mismatched_paths_and_subjects_raises_error(self):
        """Test that mismatched number of paths and subjects raises error."""
        json_data = {
            "samples": [
                {
                    "subject_id": ["sub-001", "sub-002"],
                    "task_id": "comparison",
                    "modality_paths": {
                        "image_sMRI": [str(self.image_path_1)]  # Only 1 path!
                    },
                    "conversations": [],
                    "metadata": {}
                }
            ]
        }

        json_path = self.data_root / "data.json"
        with open(json_path, 'w') as f:
            json.dump(json_data, f)

        with patch('dataset.t1_json_dataset.resolve_path',
                  side_effect=lambda p, _: p):
            dataset = T1JSONDataset(
                json_file=str(json_path),
                data_root=str(self.data_root),
                tokenizer=MockTokenizer(),
                img_size=128
            )

            # Should raise ValueError
            with self.assertRaises(ValueError) as context:
                dataset[0]

            self.assertIn("Number of paths", str(context.exception))


class TestImagePlaceholderConversion(unittest.TestCase):
    """Test conversion of subject-specific image placeholders."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_root = Path(self.temp_dir.name)

        # Create sample image files
        self.image_path_1 = self.data_root / "sub-001_T1.nii.gz"
        self.image_path_2 = self.data_root / "sub-002_T1.nii.gz"
        self.image_path_1.touch()
        self.image_path_2.touch()

    def tearDown(self):
        """Clean up."""
        self.temp_dir.cleanup()

    def test_placeholder_conversion(self):
        """Test that <sub1-image> and <sub2-image> are converted to <image_sMRI>."""
        json_data = {
            "samples": [
                {
                    "subject_id": ["sub-001", "sub-002"],
                    "task_id": "comparison",
                    "modality_paths": {
                        "image_sMRI": [
                            str(self.image_path_1),
                            str(self.image_path_2)
                        ]
                    },
                    "conversations": [
                        {"from": "human", "value": "Reference:\n<sub1-image>"},
                        {"from": "gpt", "value": "Seen."},
                        {"from": "human", "value": "Target:\n<sub2-image>"},
                        {"from": "gpt", "value": "Comparison made."}
                    ],
                    "metadata": {}
                }
            ]
        }

        json_path = self.data_root / "data.json"
        with open(json_path, 'w') as f:
            json.dump(json_data, f)

        with patch('dataset.t1_json_dataset.resolve_path',
                  side_effect=lambda p, _: p):
            with patch.object(T1JSONDataset, '_load_and_process_image',
                            return_value=torch.zeros((1, 128, 128, 128))):
                dataset = T1JSONDataset(
                    json_file=str(json_path),
                    data_root=str(self.data_root),
                    tokenizer=MockTokenizer(),
                    img_size=128
                )

                sample = dataset[0]

                # Check that input_ids were created (placeholder conversion happened)
                self.assertIn('T1', sample['input_ids'])
                self.assertIsInstance(sample['input_ids']['T1'], torch.Tensor)


class TestMultiTurnConversationFormatting(unittest.TestCase):
    """Test multi-turn conversation formatting."""

    def test_format_multi_image_conversation(self):
        """Test _format_multi_image_conversation method."""
        temp_dir = tempfile.TemporaryDirectory()
        data_root = Path(temp_dir.name)

        image_path = data_root / "test.nii.gz"
        image_path.touch()

        with patch('dataset.t1_json_dataset.resolve_path',
                  side_effect=lambda p, _: p):
            with patch.object(T1JSONDataset, '_load_and_process_image',
                            return_value=torch.zeros((1, 128, 128, 128))):
                dataset = T1JSONDataset(
                    json_file=None,
                    data_root=str(data_root),
                    tokenizer=MockTokenizer(),
                    img_size=128
                )

                # Create sample conversations
                conversations = [
                    {"from": "human", "value": "Look at <sub1-image>"},
                    {"from": "gpt", "value": "I see subject 1"},
                    {"from": "human", "value": "Now look at <sub2-image>"},
                    {"from": "gpt", "value": "I see subject 2"}
                ]

                inst, answer = dataset._format_multi_image_conversation(conversations)

                # Verify placeholders were converted
                self.assertNotIn("<sub1-image>", inst)
                self.assertNotIn("<sub2-image>", inst)
                self.assertIn("<image_sMRI>", inst)

                # Last answer should be the last gpt response
                self.assertEqual(answer, "I see subject 2")

        temp_dir.cleanup()


class TestEndToEndMultiSubject(unittest.TestCase):
    """End-to-end test of multi-subject comparison workflow."""

    def test_complete_multi_subject_workflow(self):
        """Test complete workflow from JSON to model input."""
        temp_dir = tempfile.TemporaryDirectory()
        data_root = Path(temp_dir.name)

        # Create image files
        image_path_1 = data_root / "sub-001.nii.gz"
        image_path_2 = data_root / "sub-002.nii.gz"
        image_path_1.touch()
        image_path_2.touch()

        # Create JSON with multi-subject format (Korean example)
        json_data = {
            "samples": [
                {
                    "subject_id": ["sub-001", "sub-002"],
                    "task_id": "neurodegenerative_screening",
                    "modality_paths": {
                        "image_sMRI": [
                            str(image_path_1),
                            str(image_path_2)
                        ]
                    },
                    "conversations": [
                        {
                            "from": "human",
                            "value": "여기 건강한 대조군입니다.\n<sub1-image>\n해부학적 특징을 기억해주세요."
                        },
                        {
                            "from": "gpt",
                            "value": "네, 기준 영상을 확인했습니다."
                        },
                        {
                            "from": "human",
                            "value": "이제 분석 대상 피험자입니다.\n<sub2-image>\n비교해주세요."
                        },
                        {
                            "from": "gpt",
                            "value": "기준 피험자와 비교할 때 구조적 차이를 발견했습니다."
                        }
                    ],
                    "metadata": {"age": 65, "sex": 1}
                }
            ]
        }

        json_path = data_root / "multi_subject.json"
        with open(json_path, 'w') as f:
            json.dump(json_data, f)

        # Mock the image loading
        with patch('dataset.t1_json_dataset.resolve_path',
                  side_effect=lambda p, _: p):
            with patch.object(T1JSONDataset, '_load_and_process_image',
                            return_value=torch.randn((1, 128, 128, 128))):
                dataset = T1JSONDataset(
                    json_file=str(json_path),
                    data_root=str(data_root),
                    tokenizer=MockTokenizer(),
                    img_size=128
                )

                # Get sample
                sample = dataset[0]

                # Verify structure for model input
                self.assertEqual(sample['subject_ids'], ['sub-001', 'sub-002'])
                self.assertEqual(sample['num_images'], 2)
                self.assertEqual(sample['task_id'], 'neurodegenerative_screening')

                # Verify pixel_values
                self.assertIn('T1', sample['pixel_values'])
                images = sample['pixel_values']['T1']
                self.assertEqual(len(images), 2)
                self.assertEqual(images[0].shape, (1, 128, 128, 128))
                self.assertEqual(images[1].shape, (1, 128, 128, 128))

                # Verify tokenization
                self.assertIn('T1', sample['input_ids'])
                self.assertIn('T1', sample['attention_mask'])

                # Metadata should be preserved
                self.assertEqual(sample['metadata']['age'], 65)

        temp_dir.cleanup()


if __name__ == '__main__':
    unittest.main()
