"""
UMBRELLA Dataset: Unified Multi-turn Brain Imaging Language Dataset (FIXED VERSION)

CRITICAL UPDATES:
1. **JSONL format support**: Supports JSON, JSONL (newline-delimited JSON), and directories
2. **Directory-based loading**: Supports both single JSON file AND directories with multiple JSON files
3. Proper LLaVA-Next tokenization format with <|im_start|> and <|im_end|>
4. Correct role handling (user/assistant - no intermediate conversion)
5. Variable image size support from config (lists)
6. 3D/4D image handling for sMRI/fMRI
7. Task filtering support (same_sex_comparison, different_sex_comparison, etc.)

Key Features:
- JSON v2 format support (role: user/assistant, content: array)
- JSONL format support (newline-delimited JSON for large datasets)
- LLaVA-Next compatible prompt format
- Multi-turn conversation with proper masking
- Variable-size MRI image support
- Directory, file, and JSONL-based data loading
"""

import os
import json
import re
import numpy as np
import torch
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from torch.utils.data import Dataset
from monai.transforms import (
    LoadImage, Compose, AddChannel, Resize,
    NormalizeIntensity, RandAxisFlip, RandRotate90
)
from monai.data import NibabelReader
from monai.utils import get_seed

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """Single turn in a multi-turn conversation."""
    role: str  # 'user' or 'assistant' (JSON v2 format)
    content: str  # Text content (may contain image tokens)
    image_tokens: List[str] = field(default_factory=list)  # e.g., ['<image>', '<image>']


@dataclass
class UMBRELLASample:
    """Complete sample for UMBRELLA training."""
    task_id: str  # Unique identifier (primary key)
    task_type: str  # 'T1', 'T2', 'T3', etc.
    subject_ids: List[str]  # Subject identifiers
    modalities: List[str]  # ['sMRI', 'fMRI', 'dMRI']
    image_paths: List[str]  # Paths to images
    conversation: List[ConversationTurn]  # Multi-turn conversation
    metadata: Dict[str, Any] = field(default_factory=dict)


class UMBRELLADataset(Dataset):
    """
    Unified Multi-turn Brain Imaging Language Dataset.

    Supports flexible multi-turn conversations with interleaved images and text.
    Each sample can have variable number of images, subjects, and modalities.

    **NEW**: Supports JSON, JSONL (newline-delimited JSON), and directory-based loading.

    JSON Format (v2 - CURRENT STANDARD):
    {
        "task_id": "T1_001",
        "task_type": "T1",
        "subject_ids": ["sub-001"],
        "modalities": ["sMRI"],
        "images": [
            {"path": "/path/to/image.nii.gz", "token": "<image>", "modality": "sMRI"}
        ],
        "conversations": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this brain scan."},
                    {"type": "image", "modality": "sMRI", "image_path": "/path/to/image.nii.gz"}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "This is a T1-weighted MRI showing..."}
                ]
            }
        ],
        "metadata": {"source": "ABCD", "age": 12, "sex": "male"}
    }

    JSONL Format (Newline-Delimited JSON):
    Each line is a complete JSON object (same structure as above).
    Files must have .jsonl extension.

    LLaVA-Next Output Format:
    <|im_start|>user <image>
    Analyze this brain scan.<|im_end|><|im_start|>assistant
    This is a T1-weighted MRI showing...<|im_end|>

    Directory Structure Support:
    Option 1: Single JSON file
        data/train.json

    Option 2: Single JSONL file (recommended for large datasets)
        data/train_conversations.jsonl

    Option 3: Directory with multiple JSON files
        data/train/
            subject1_task1.json
            subject1_task2.json
            subject2_task1.json
            ...

    Option 4: Nested directories with task types
        data/train/
            same_sex_comparison/
                subject1.json
                subject2.json
            different_sex_comparison/
                subject1.json
                subject2.json
    """

    # Standard image token (generic for all modalities)
    IMAGE_TOKEN = '<image>'

    def __init__(self,
                 json_path: str,
                 tokenizer,
                 mode: str = 'train',
                 img_size: Union[int, List[int]] = 128,
                 max_seq_length: int = 2048,
                 max_images: int = 10,
                 image_base_dir: Optional[str] = None,
                 augment: bool = True,
                 use_dummy_loss: bool = False,
                 modality: str = 'sMRI',
                 task_filter: Optional[str] = None):
        """
        Initialize UMBRELLA dataset.

        Args:
            json_path: Path to JSON/JSONL file OR directory containing JSON files
            tokenizer: Text tokenizer (must support <|im_start|> and <|im_end|>)
            mode: 'train' or 'eval'
            img_size: Image spatial dimension (int or list [H, W, D] or [H, W, D, T])
            max_seq_length: Maximum token sequence length
            max_images: Maximum images per sample
            image_base_dir: Base directory for relative image paths
            augment: Apply data augmentation (train only)
            use_dummy_loss: Support dummy loss for placeholder answers
            modality: Primary modality ('sMRI' for sMRI, 'rsfMRI' for fMRI)
            task_filter: Optional task type filter (e.g., 'same_sex_comparison')
        """
        self.json_path = json_path
        self.tokenizer = tokenizer
        self.mode = mode
        self.modality = modality
        self.task_filter = task_filter

        # Handle img_size as int or list
        if isinstance(img_size, int):
            self.img_size = [img_size, img_size, img_size]
        elif isinstance(img_size, list):
            self.img_size = img_size
        else:
            raise ValueError(f"img_size must be int or list, got {type(img_size)}")

        self.max_seq_length = max_seq_length
        self.max_images = max_images
        self.image_base_dir = image_base_dir
        self.augment = augment and mode == 'train'
        self.use_dummy_loss = use_dummy_loss

        # Determine if 4D (fMRI with temporal dimension)
        self.is_4d = len(self.img_size) == 4

        # Load samples (supports JSON, JSONL, and directories)
        self.samples = self._load_samples_smart(json_path)

        # Image transforms
        self.image_transform = self._define_image_transforms()
        self.image_loader = LoadImage(reader=None, image_only=True, dtype=np.float32)

        # Precompute metadata for efficient batching
        self._precompute_metadata()

        logger.info(f"Loaded {len(self.samples)} samples from {json_path}")
        logger.info(f"Image size: {self.img_size} ({'4D fMRI' if self.is_4d else '3D sMRI/dMRI'})")
        logger.info(f"Task distribution: {self._get_task_distribution()}")
        if self.task_filter:
            logger.info(f"Task filter applied: {self.task_filter}")

    def _load_samples_smart(self, path: str) -> List[UMBRELLASample]:
        """
        Smart loading: automatically detect if path is file (JSON/JSONL) or directory.

        Supported formats:
        1. Single JSON file: Contains array of samples
        2. Single JSONL file: Newline-delimited JSON (one sample per line)
        3. Directory: Multiple JSON files (flat or nested)

        Args:
            path: Path to JSON/JSONL file or directory

        Returns:
            List of UMBRELLASample
        """
        path_obj = Path(path)

        if not path_obj.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        if path_obj.is_file():
            # Check file extension
            if path_obj.suffix == '.jsonl':
                # JSONL file (newline-delimited JSON)
                logger.info(f"Loading from JSONL file: {path}")
                return self._load_samples_from_jsonl(path)
            elif path_obj.suffix == '.json':
                # Single JSON file
                logger.info(f"Loading from single JSON file: {path}")
                return self._load_samples_from_file(path)
            else:
                raise ValueError(f"Unsupported file format: {path_obj.suffix} (expected .json or .jsonl)")

        elif path_obj.is_dir():
            # Directory with JSON files
            logger.info(f"Loading from directory: {path}")
            return self._load_samples_from_directory(path)

        else:
            raise ValueError(f"Path is neither file nor directory: {path}")

    def _load_samples_from_file(self, json_path: str) -> List[UMBRELLASample]:
        """Load samples from single JSON file containing array of samples."""
        with open(json_path, 'r') as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"JSON file must contain array of samples, got {type(data)}")

        logger.info(f"  Found {len(data)} samples in file")
        return self._parse_samples(data)

    def _load_samples_from_jsonl(self, jsonl_path: str) -> List[UMBRELLASample]:
        """
        Load samples from JSONL file (newline-delimited JSON).

        Each line in the file is a complete JSON object representing one sample.

        Args:
            jsonl_path: Path to .jsonl file

        Returns:
            List of UMBRELLASample

        Error Handling:
        - Skips malformed lines with warning
        - Logs line numbers of failed parses
        - Continues processing valid lines
        """
        all_samples = []
        failed_lines = []
        line_count = 0

        logger.info(f"  Reading JSONL file (newline-delimited JSON)...")

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

                        # Parse into UMBRELLASample
                        samples = self._parse_samples([item])
                        all_samples.extend(samples)

                    except json.JSONDecodeError as e:
                        logger.warning(f"  Line {line_num}: Invalid JSON - {e}")
                        failed_lines.append(line_num)
                        continue

                    except Exception as e:
                        logger.error(f"  Line {line_num}: Failed to parse sample - {e}")
                        failed_lines.append(line_num)
                        continue

        except FileNotFoundError:
            raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to read JSONL file {jsonl_path}: {e}")

        # Report statistics
        logger.info(f"  Found {line_count} lines in JSONL file")
        logger.info(f"  Successfully loaded {len(all_samples)} samples")

        if failed_lines:
            logger.warning(f"  Failed to parse {len(failed_lines)} lines: {failed_lines[:10]}...")

        if len(all_samples) == 0:
            raise ValueError(f"No valid samples found in JSONL file: {jsonl_path}")

        return all_samples

    def _load_samples_from_directory(self, dir_path: str) -> List[UMBRELLASample]:
        """
        Load samples from directory containing multiple JSON files.

        Supports:
        1. Flat directory: dir_path/*.json
        2. Nested directories: dir_path/*/*.json (with task type subdirs)

        Args:
            dir_path: Path to directory

        Returns:
            List of UMBRELLASample
        """
        dir_path_obj = Path(dir_path)

        # Find all JSON files (both flat and nested)
        json_files = list(dir_path_obj.glob("*.json")) + list(dir_path_obj.glob("*/*.json"))

        if not json_files:
            raise FileNotFoundError(f"No JSON files found in directory: {dir_path}")

        logger.info(f"  Found {len(json_files)} JSON files in directory")

        all_samples = []
        failed_files = []

        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    item = json.load(f)

                # Each file contains ONE sample (not an array)
                if isinstance(item, dict):
                    samples = self._parse_samples([item])
                    all_samples.extend(samples)
                elif isinstance(item, list):
                    # Fallback: file contains array of samples
                    samples = self._parse_samples(item)
                    all_samples.extend(samples)
                else:
                    logger.warning(f"Skipping {json_file}: unexpected format {type(item)}")
                    failed_files.append(str(json_file))

            except Exception as e:
                logger.error(f"Failed to load {json_file}: {e}")
                failed_files.append(str(json_file))

        if failed_files:
            logger.warning(f"Failed to load {len(failed_files)} files: {failed_files[:5]}...")

        logger.info(f"  Successfully loaded {len(all_samples)} samples from {len(json_files) - len(failed_files)} files")

        return all_samples

    def _parse_samples(self, data: List[Dict[str, Any]]) -> List[UMBRELLASample]:
        """
        Parse raw JSON data into UMBRELLASample objects.

        Args:
            data: List of sample dictionaries

        Returns:
            List of UMBRELLASample
        """
        samples = []

        for item in data:
            # Parse conversation - only support 'conversations' key (v2 format)
            conversations_raw = item.get('conversations', [])
            if not conversations_raw:
                logger.warning(f"Sample {item.get('task_id')} has no conversations, skipping")
                continue

            conversation = []

            for turn in conversations_raw:
                # Get role (must be 'user' or 'assistant')
                role = turn.get('role', 'user')
                if role not in ['user', 'assistant']:
                    logger.warning(f"Invalid role '{role}', expected 'user' or 'assistant'")
                    role = 'user'  # Fallback

                # Parse content (must be array format)
                content, image_tokens = self._parse_content(turn.get('content', []))

                conv_turn = ConversationTurn(
                    role=role,
                    content=content,
                    image_tokens=image_tokens
                )
                conversation.append(conv_turn)

            # Get task_id (handle different naming patterns)
            task_id = item.get('task_id', item.get('id', f"unknown_{len(samples)}"))

            # Extract task type from task_id if not explicitly provided
            task_type = item.get('task_type')
            if not task_type:
                # Try to infer from task_id
                if 'same_sex' in task_id:
                    task_type = 'T3_same_sex'
                elif 'different_sex' in task_id:
                    task_type = 'T3_different_sex'
                else:
                    task_type = 'T1'  # Default fallback

            # Apply task filter if specified
            if self.task_filter:
                if self.task_filter not in task_id and self.task_filter not in task_type:
                    continue  # Skip samples that don't match filter

            # Create sample
            sample = UMBRELLASample(
                task_id=task_id,
                task_type=task_type,
                subject_ids=item.get('subject_ids', []),
                modalities=item.get('modalities', []),
                image_paths=[img['path'] for img in item.get('images', [])],
                conversation=conversation,
                metadata=item.get('metadata', {})
            )

            # Store image token mapping for later use
            sample.metadata['image_token_mapping'] = {
                img['token']: img['path'] for img in item.get('images', [])
            }

            samples.append(sample)

        return samples

    def _parse_content(self, content_raw: Union[str, List[Dict]]) -> Tuple[str, List[str]]:
        """
        Parse content from JSON v2 array format.

        Args:
            content_raw: List of content items (v2 format) or string (legacy fallback)

        Returns:
            (text_content, image_tokens) tuple

        Example:
            Input: [
                {"type": "text", "text": "Analyze this scan."},
                {"type": "image", "modality": "sMRI", "image_path": "..."}
            ]
            Output: ("Analyze this scan. <image>", ["<image>"])
        """
        if isinstance(content_raw, str):
            # Legacy format fallback: content is string
            return content_raw, self._extract_image_tokens(content_raw)

        elif isinstance(content_raw, list):
            # JSON v2 format: content is array of {"type": "text/image", ...}
            text_parts = []
            image_tokens = []

            for item in content_raw:
                item_type = item.get('type', '')

                if item_type == 'text':
                    text = item.get('text', '')
                    if text:
                        text_parts.append(text)

                elif item_type == 'image':
                    # Insert generic <image> token
                    image_tokens.append(self.IMAGE_TOKEN)
                    text_parts.append(self.IMAGE_TOKEN)

            # Join with spaces (preserve natural text flow)
            text_content = ' '.join(text_parts)
            return text_content, image_tokens

        else:
            # Fallback for unexpected formats
            logger.warning(f"Unexpected content type: {type(content_raw)}")
            return str(content_raw), []

    def _extract_image_tokens(self, text: str) -> List[str]:
        """Extract all image tokens from text (legacy support)."""
        tokens = []

        # Find modality tokens: <image_sMRI>, <image_fMRI>, etc.
        tokens.extend(re.findall(r'<image_\w+>', text))

        # Find subject tokens: <sub1-image>, <sub2-image>, etc.
        tokens.extend(re.findall(r'<sub\d+-image>', text))

        # Find generic tokens: <image>
        tokens.extend(re.findall(r'<image>', text))

        return tokens

    def _define_image_transforms(self):
        """Define image augmentation transforms."""
        img_size = tuple(self.img_size)

        if self.augment:
            transform = Compose([
                AddChannel(),
                Resize(img_size),
                RandAxisFlip(prob=0.5),
                RandRotate90(prob=0.5, spatial_axes=(0, 1)),
                NormalizeIntensity()
            ])
        else:
            transform = Compose([
                AddChannel(),
                Resize(img_size),
                NormalizeIntensity()
            ])

        return transform

    def _precompute_metadata(self):
        """Precompute sample metadata for efficient batch construction."""
        self.sample_metadata = []

        for idx, sample in enumerate(self.samples):
            metadata = {
                'task_type': sample.task_type,
                'num_images': len(sample.image_paths),
                'num_turns': len(sample.conversation),
                'num_subjects': len(sample.subject_ids),
                'modalities': sample.modalities,
                'task_id': sample.task_id,
                'index': idx
            }
            self.sample_metadata.append(metadata)

    def _get_task_distribution(self) -> Dict[str, int]:
        """Get distribution of task types."""
        dist = {}
        for meta in self.sample_metadata:
            task_type = meta['task_type']
            dist[task_type] = dist.get(task_type, 0) + 1
        return dist

    def get_sample_metadata(self, idx: int) -> Dict[str, Any]:
        """
        Get metadata for a sample (used by MemoryAwareBatchSampler).

        Args:
            idx: Sample index

        Returns:
            Dict with task_type, num_images, etc.
        """
        return self.sample_metadata[idx]

    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load and transform a single image."""
        # Handle relative paths
        if self.image_base_dir and not os.path.isabs(image_path):
            image_path = os.path.join(self.image_base_dir, image_path)

        # Check file exists
        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}")
            # Return zeros for missing images (graceful handling)
            if self.is_4d:
                return torch.zeros(1, *self.img_size)
            else:
                return torch.zeros(1, *self.img_size)

        # Load image
        try:
            image = self.image_loader(image_path)
            if self.image_transform:
                image = self.image_transform(image)
            image = torch.from_numpy(np.array(image)).float()
            return image
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            if self.is_4d:
                return torch.zeros(1, *self.img_size)
            else:
                return torch.zeros(1, *self.img_size)

    def _tokenize_conversation(self, sample: UMBRELLASample) -> Dict[str, torch.Tensor]:
        """
        Tokenize multi-turn conversation with LLaVA-Next format.

        LLaVA-Next Format:
        <|im_start|>user <image><image>
        Compare these scans.<|im_end|><|im_start|>assistant
        Based on comparison...<|im_end|><|im_start|>user
        What about...?<|im_end|><|im_start|>assistant

        Masking Strategy:
        - User turns: Masked with -100 (not in loss)
        - Assistant turns: Active in loss computation
        - Image tokens: Placeholder during tokenization

        Returns:
            Dict with input_ids, attention_mask, labels
        """
        # Build full conversation in LLaVA-Next format
        conversation_parts = []
        turn_boundaries = []  # Track (start_idx, end_idx, role) for masking

        for turn in sample.conversation:
            # Start of turn
            turn_start = len(conversation_parts)

            # Add turn start token
            conversation_parts.append(f"<|im_start|>{turn.role}")

            # Add content
            conversation_parts.append(turn.content)

            # Add turn end token
            conversation_parts.append("<|im_end|>")

            turn_end = len(conversation_parts)
            turn_boundaries.append((turn_start, turn_end, turn.role))

        # Join all parts into full text
        full_text = ''.join(conversation_parts)

        # Tokenize full text
        encoding = self.tokenizer(
            full_text,
            add_special_tokens=True,
            padding='max_length',
            max_length=self.max_seq_length,
            truncation=True,
            return_tensors='pt',
            return_offsets_mapping=True
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        # Initialize labels
        labels = input_ids.clone()

        # Mask padding tokens
        labels[attention_mask == 0] = -100

        # Mask user turns (only assistant turns contribute to loss)
        if 'offset_mapping' in encoding:
            offset_mapping = encoding['offset_mapping'].squeeze(0)

            # Convert character-level turn boundaries to token-level
            for part_start, part_end, role in turn_boundaries:
                # Calculate character positions
                char_start = len(''.join(conversation_parts[:part_start]))
                char_end = len(''.join(conversation_parts[:part_end]))

                # Mask tokens in this character range if role is 'user'
                if role == 'user':
                    for token_idx, (token_start, token_end) in enumerate(offset_mapping):
                        if token_start >= char_start and token_end <= char_end:
                            labels[token_idx] = -100
                        elif token_start < char_end and token_end > char_start:
                            # Partial overlap - mask conservatively
                            labels[token_idx] = -100
        else:
            # Fallback: mask based on <|im_start|>assistant pattern
            self._mask_user_turns_fallback(input_ids, labels)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

    def _mask_user_turns_fallback(self, input_ids: torch.Tensor, labels: torch.Tensor):
        """
        Fallback masking using <|im_start|>assistant pattern.

        Masks everything except text between <|im_start|>assistant and <|im_end|>.
        """
        # Find all occurrences of "<|im_start|>assistant"
        assistant_start_text = "<|im_start|>assistant"
        assistant_tokens = self.tokenizer.encode(assistant_start_text, add_special_tokens=False)

        if not assistant_tokens:
            logger.warning("Could not tokenize '<|im_start|>assistant', masking may be incorrect")
            return

        # Find <|im_end|> tokens
        end_token_text = "<|im_end|>"
        end_tokens = self.tokenizer.encode(end_token_text, add_special_tokens=False)

        # Simple approach: mask everything, then unmask assistant turns
        labels[:] = -100

        # Find assistant turn positions
        seq_len = len(input_ids)
        i = 0
        while i < seq_len:
            # Look for assistant start pattern
            if i + len(assistant_tokens) <= seq_len:
                if torch.equal(input_ids[i:i+len(assistant_tokens)], torch.tensor(assistant_tokens)):
                    # Found assistant start, find corresponding end
                    j = i + len(assistant_tokens)
                    while j < seq_len and j < i + self.max_seq_length:
                        if j + len(end_tokens) <= seq_len:
                            if torch.equal(input_ids[j:j+len(end_tokens)], torch.tensor(end_tokens)):
                                # Unmask assistant turn (excluding special tokens)
                                labels[i+len(assistant_tokens):j] = input_ids[i+len(assistant_tokens):j]
                                i = j + len(end_tokens)
                                break
                        j += 1
            i += 1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get a single sample.

        Returns:
            Dict with:
                - pixel_values: Dict mapping modality to tensor
                - input_ids: Token IDs
                - attention_mask: Attention mask
                - labels: Labels for loss computation
                - task_type: Task type string
                - sample_index: Original index
                - metadata: Additional metadata
        """
        sample = self.samples[index]

        assert len(sample.image_paths) == self.max_images, NotImplementedError("The toal number of images should be the same per each conversation")

        # Load images
        images = []
        for img_path in sample.image_paths[:self.max_images]:
            img = self._load_image(img_path)
            images.append(img)

        # Stack images
        if images:
            pixel_values = torch.stack(images)
        else:
            if self.is_4d:
                pixel_values = torch.zeros(self.max_images, 1, *self.img_size)
            else:
                pixel_values = torch.zeros(self.max_images, 1, *self.img_size)

        # Tokenize conversation
        text_encoding = self._tokenize_conversation(sample)

        # Build output
        output = {
            'pixel_values': {self.modality: pixel_values},  # Nested for compatibility
            'input_ids': text_encoding['input_ids'],
            'attention_mask': text_encoding['attention_mask'],
            'labels': text_encoding['labels'],
            'task_type': sample.task_type,
            'sample_index': index,
            'num_images': len(sample.image_paths),
            'metadata': {
                'task_id': sample.task_id,
                'subject_ids': sample.subject_ids,
                'modalities': sample.modalities,
                **sample.metadata
            }
        }

        return output


def create_umbrella_dataset_from_config(
    config: Dict[str, Any],
    json_path: str,
    tokenizer,
    mode: str = 'train',
    modality: str = 'sMRI',
    task_filter: Optional[str] = None
) -> UMBRELLADataset:
    """
    Factory function to create UMBRELLA dataset from config dict.

    Args:
        config: Configuration dict from YAML
        json_path: Path to JSON/JSONL file OR directory
        tokenizer: Text tokenizer
        mode: 'train' or 'eval'
        modality: Modality name (sMRI, rsfMRI, etc.)
        task_filter: Optional task type filter

    Returns:
        UMBRELLADataset instance
    """
    # Extract modality-specific config
    modality_config = config['dataset'].get(modality, {})
    img_size = modality_config.get('img_size', 128)

    return UMBRELLADataset(
        json_path=json_path,
        tokenizer=tokenizer,
        mode=mode,
        img_size=img_size,
        modality=modality,
        max_seq_length=config.get('trainer', {}).get('max_seq_length', 2048),
        max_images=10,
        augment=(mode == 'train'),
        task_filter=task_filter
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test with sample config
    test_config = {
        'dataset': {
            'sMRI': {'img_size': [120, 120, 120]},
            'rsfMRI': {'img_size': [96, 96, 96, 24]}
        }
    }

    print("Config test:")
    print(f"  sMRI img_size: {test_config['dataset']['sMRI']['img_size']}")
    print(f"  rsfMRI img_size: {test_config['dataset']['rsfMRI']['img_size']}")
    print("\nDataset supports:")
    print("  - JSON format (single file with array)")
    print("  - JSONL format (newline-delimited JSON)")
    print("  - Directory loading (multiple JSON files)")
