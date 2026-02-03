"""
UMBRELLA Dataset: Unified Multi-turn Brain Imaging Language Dataset

Supports in-context learning with interleaved image-text sequences for:
- T1: Single subject, single image
- T2: Single subject, multiple images (multi-modal)
- T3: Multiple subjects, single image per subject (comparison)
- Future variants: Any combination

Core Design:
- LLaVA-style multi-turn conversation format
- Variable conversation lengths (2, 4, 6, 8+ turns)
- Dynamic image token replacement
- Task-agnostic architecture
- UPDATED: Supports both legacy and new LLaVA JSON formats
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
    role: str  # 'human' or 'gpt' (internal format)
    content: str  # Text content (may contain image tokens)
    image_tokens: List[str] = field(default_factory=list)  # e.g., ['<image_sMRI>', '<sub1-image>']


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

    JSON Format (Legacy):
    {
        "task_id": "T1_001",
        "task_type": "T1",
        "subject_ids": ["sub-001"],
        "modalities": ["sMRI"],
        "images": [
            {"path": "/path/to/image.nii.gz", "token": "<image_sMRI>", "modality": "sMRI"}
        ],
        "conversation": [
            {"role": "human", "content": "<image_sMRI>\\nAnalyze this brain scan."},
            {"role": "gpt", "content": "This is a T1-weighted MRI showing..."}
        ],
        "metadata": {"source": "ABCD", "age": 12, "sex": "male"}
    }

    JSON Format (New LLaVA Standard):
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
    """

    # Standard image tokens by modality and subject
    IMAGE_TOKEN_PATTERNS = {
        'modality': r'<image_(\w+)>',  # <image_sMRI>, <image_fMRI>, <image_dMRI>
        'subject': r'<sub(\d+)-image>',  # <sub1-image>, <sub2-image>
        'generic': r'<image>',  # Simple <image> token
    }

    # Role normalization mapping for LLaVA compatibility
    ROLE_MAPPING = {
        'user': 'human',      # New format -> internal
        'assistant': 'gpt',   # New format -> internal
        'human': 'human',     # Backward compatibility
        'gpt': 'gpt'         # Backward compatibility
    }

    def __init__(self,
                 json_path: str,
                 tokenizer,
                 mode: str = 'train',
                 img_size: int = 128,
                 max_seq_length: int = 2048,
                 max_images: int = 10,
                 image_base_dir: Optional[str] = None,
                 augment: bool = True,
                 use_dummy_loss: bool = False):
        """
        Initialize UMBRELLA dataset.

        Args:
            json_path: Path to JSON file with samples
            tokenizer: Text tokenizer
            mode: 'train' or 'eval'
            img_size: Image spatial dimension
            max_seq_length: Maximum token sequence length
            max_images: Maximum images per sample
            image_base_dir: Base directory for relative image paths
            augment: Apply data augmentation (train only)
            use_dummy_loss: Support dummy loss for placeholder answers
        """
        self.json_path = json_path
        self.tokenizer = tokenizer
        self.mode = mode
        self.img_size = img_size
        self.max_seq_length = max_seq_length
        self.max_images = max_images
        self.image_base_dir = image_base_dir
        self.augment = augment and mode == 'train'
        self.use_dummy_loss = use_dummy_loss

        # Load samples
        self.samples = self._load_samples(json_path)

        # Image transforms
        self.image_transform = self._define_image_transforms()
        self.image_loader = LoadImage(reader=None, image_only=True, dtype=np.float32)

        # Precompute metadata for efficient batching
        self._precompute_metadata()

        logger.info(f"Loaded {len(self.samples)} samples from {json_path}")
        logger.info(f"Task distribution: {self._get_task_distribution()}")

    def _load_samples(self, json_path: str) -> List[UMBRELLASample]:
        """Load samples from JSON file with support for both legacy and new formats."""
        with open(json_path, 'r') as f:
            data = json.load(f)

        samples = []
        for item in data:
            # Parse conversation - support both 'conversation' and 'conversations' keys
            conversations_raw = item.get('conversations', item.get('conversation', []))
            conversation = []

            for turn in conversations_raw:
                # Normalize role
                raw_role = turn.get('role', 'human')
                role = self.ROLE_MAPPING.get(raw_role, raw_role)

                # Parse content (handles both string and array formats)
                content, image_tokens = self._parse_content(turn.get('content', ''))

                conv_turn = ConversationTurn(
                    role=role,
                    content=content,
                    image_tokens=image_tokens
                )
                conversation.append(conv_turn)

            # Create sample
            sample = UMBRELLASample(
                task_id=item['task_id'],
                task_type=item.get('task_type', 'T1'),
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
        Parse content from either legacy string or new LLaVA array format.

        Args:
            content_raw: Either string (legacy) or list of content items (new)

        Returns:
            (text_content, image_tokens) tuple

        Examples:
            Legacy: "Analyze this scan." -> ("Analyze this scan.", [])
            New: [{"type": "text", "text": "Analyze"}, {"type": "image"}]
                 -> ("Analyze\n<image>", ["<image>"])
        """
        if isinstance(content_raw, str):
            # Legacy format: content is already a string
            return content_raw, self._extract_image_tokens(content_raw)

        elif isinstance(content_raw, list):
            # New LLaVA format: content is array of {"type": "text/image", ...}
            text_parts = []
            image_tokens = []

            for item in content_raw:
                item_type = item.get('type', '')

                if item_type == 'text':
                    text_parts.append(item.get('text', ''))

                elif item_type == 'image':
                    # Insert generic <image> token
                    image_token = '<image>'
                    image_tokens.append(image_token)
                    text_parts.append(image_token)  # Embed in text for tokenization

            text_content = '\n'.join(text_parts)
            return text_content, image_tokens

        else:
            # Fallback for unexpected formats
            logger.warning(f"Unexpected content type: {type(content_raw)}")
            return str(content_raw), []

    def _extract_image_tokens(self, text: str) -> List[str]:
        """Extract all image tokens from text."""
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
        img_size = (self.img_size, self.img_size, self.img_size)

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
            return torch.zeros(1, self.img_size, self.img_size, self.img_size)

        # Load image
        try:
            image = self.image_loader(image_path)
            if self.image_transform:
                image = self.image_transform(image)
            image = torch.from_numpy(np.array(image)).float()
            return image
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return torch.zeros(1, self.img_size, self.img_size, self.img_size)

    def _tokenize_conversation(self, sample: UMBRELLASample) -> Dict[str, torch.Tensor]:
        """
        Tokenize multi-turn conversation with proper masking for LLaVA-Next format.

        CRITICAL: This method converts JSON v2 format conversations to LLaVA-Next token format.
        
        Input Format (JSON v2):
            {
                "role": "user" or "assistant",
                "content": [
                    {"type": "text", "text": "..."},
                    {"type": "image", ...}
                ]
            }

        Output Format (LLaVA-Next):
            "<|im_start|>user <image>text<|im_end|><|im_start|>assistant text<|im_end|>"

        Masking Strategy (LLaVA-style):
        - User turns: Masked with -100 (not in loss, only for context)
        - Assistant turns: Active in loss computation (model learns to predict)
        - Image tokens: Generic <image> tokens at turn start
        - Role detection: MUST use lowercase "user" and "assistant" ONLY

        Returns:
            Dict with input_ids, attention_mask, labels
        """
        # Build full conversation text with proper LLaVA-Next format
        full_text = ""
        turn_boundaries = []  # Track (start_char, end_char, role) for each turn

        for turn in sample.conversation:
            # CRITICAL: Use lowercase role detection ONLY
            # Internal format stores as 'human' and 'gpt', but we output as 'user' and 'assistant'
            if turn.role == 'human':
                role_token = 'user'
            elif turn.role == 'gpt':
                role_token = 'assistant'
            else:
                role_token = turn.role.lower()

            start_char = len(full_text)

            # Build turn text in LLaVA-Next format: <|im_start|>{role}\n{content}<|im_end|>
            turn_text = f"<|im_start|>{role_token}\n{turn.content}<|im_end|>"

            full_text += turn_text
            end_char = len(full_text)
            turn_boundaries.append((start_char, end_char, turn.role))

        # Tokenize full text with special tokens
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

        # Initialize labels from input_ids
        labels = input_ids.clone()

        # Mask padding tokens
        labels[attention_mask == 0] = -100

        # Get offset mapping for character-to-token alignment
        if 'offset_mapping' in encoding:
            offset_mapping = encoding['offset_mapping'].squeeze(0)

            # Mask user turns (they are context, not targets for training)
            for start_char, end_char, role in turn_boundaries:
                if role == 'human':  # Maps to 'user' in output
                    # Find token indices for this character range
                    for token_idx, (token_start, token_end) in enumerate(offset_mapping):
                        # Mask tokens that overlap with this turn
                        if token_start >= start_char and token_end <= end_char:
                            labels[token_idx] = -100
                        elif token_start < end_char and token_end > start_char:
                            # Partial overlap - mask conservatively to exclude user content
                            labels[token_idx] = -100
        else:
            # Fallback: mask tokens between <|im_start|>user and <|im_end|> patterns
            # This is for tokenizers that don't support offset mapping
            self._mask_user_turns_fallback(input_ids, labels)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

    def _mask_user_turns_fallback(self, input_ids: torch.Tensor, labels: torch.Tensor):
        """
        Fallback masking for LLaVA-Next format when offset mapping is unavailable.
        
        Masks tokens from <|im_start|>user to <|im_end|> to exclude user context from loss.
        """
        # Find all occurrences of <|im_start|>user and <|im_end|> patterns
        user_start_tokens = self.tokenizer.encode("<|im_start|>user", add_special_tokens=False)
        user_end_tokens = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        
        if not user_start_tokens or not user_end_tokens:
            logger.warning("Could not find user turn markers in tokenizer vocabulary")
            return
        
        user_start_tensor = torch.tensor(user_start_tokens)
        user_end_tensor = torch.tensor(user_end_tokens)
        
        seq_len = len(input_ids)
        start_pattern_len = len(user_start_tokens)
        end_pattern_len = len(user_end_tokens)
        
        # Find all user turn regions and mask them
        i = 0
        while i < seq_len - start_pattern_len + 1:
            if torch.equal(input_ids[i:i+start_pattern_len], user_start_tensor):
                # Found start of user turn, find its end
                j = i + start_pattern_len
                while j < seq_len - end_pattern_len + 1:
                    if torch.equal(input_ids[j:j+end_pattern_len], user_end_tensor):
                        # Found end of user turn, mask everything from start to end (inclusive)
                        labels[i:j+end_pattern_len] = -100
                        i = j + end_pattern_len
                        break
                    j += 1
                else:
                    # No matching end found, move to next position
                    i += 1
            else:
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

        # Load images
        images = []
        for img_path in sample.image_paths[:self.max_images]:
            img = self._load_image(img_path)
            images.append(img)

        # Pad to max_images if needed (for consistent batching)
        while len(images) < self.max_images:
            images.append(torch.zeros_like(images[0]) if images else
                         torch.zeros(1, self.img_size, self.img_size, self.img_size))

        # Stack images
        pixel_values = torch.stack(images) if images else torch.zeros(
            self.max_images, 1, self.img_size, self.img_size, self.img_size
        )

        # Tokenize conversation
        text_encoding = self._tokenize_conversation(sample)

        # Build output
        output = {
            'pixel_values': {'T1': pixel_values},  # Nested for compatibility
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


class UMBRELLAMultiSubjectDataset(UMBRELLADataset):
    """
    Extended dataset with enhanced multi-subject comparison support.

    Specifically designed for T3 tasks where multiple subjects are compared
    in a single conversation with sequential image loading.
    """

    def __init__(self, *args, comparison_mode: str = 'sequential', **kwargs):
        """
        Initialize multi-subject dataset.

        Args:
            comparison_mode: How to present subjects
                - 'sequential': One after another in conversation
                - 'parallel': All images in first turn
        """
        super().__init__(*args, **kwargs)
        self.comparison_mode = comparison_mode

    def _format_comparison_conversation(self, sample: UMBRELLASample) -> str:
        """Format conversation for subject comparison with clear ordering."""
        formatted_turns = []

        for turn in sample.conversation:
            content = turn.content

            # Replace generic subject tokens with modality-specific ones
            for i, subject_id in enumerate(sample.subject_ids, 1):
                # <sub1-image> -> actual image reference
                pattern = f'<sub{i}-image>'
                if pattern in content:
                    modality = sample.modalities[min(i-1, len(sample.modalities)-1)]
                    content = content.replace(pattern, f'[Subject {i} ({subject_id}) {modality}]')

            formatted_turns.append({
                'role': turn.role,
                'content': content
            })

        return formatted_turns


def create_umbrella_dataset_from_json(
    json_path: str,
    tokenizer,
    mode: str = 'train',
    **kwargs
) -> UMBRELLADataset:
    """
    Factory function to create UMBRELLA dataset from JSON.

    Args:
        json_path: Path to JSON file
        tokenizer: Text tokenizer
        mode: 'train' or 'eval'
        **kwargs: Additional arguments to UMBRELLADataset

    Returns:
        UMBRELLADataset instance
    """
    return UMBRELLADataset(
        json_path=json_path,
        tokenizer=tokenizer,
        mode=mode,
        **kwargs
    )


def create_example_json(output_path: str):
    """
    Create example JSON file demonstrating all task types.

    Args:
        output_path: Path to write example JSON
    """
    examples = [
        # T1: Single subject, single image
        {
            "task_id": "T1_001",
            "task_type": "T1",
            "subject_ids": ["sub-001"],
            "modalities": ["sMRI"],
            "images": [
                {"path": "data/sub-001/T1w.nii.gz", "token": "<image_sMRI>", "modality": "sMRI"}
            ],
            "conversation": [
                {
                    "role": "human",
                    "content": "<image_sMRI>\nYou are a neurologist analyzing this T1-weighted MRI. What is the estimated sex of this subject?"
                },
                {
                    "role": "gpt",
                    "content": "Based on the structural features visible in this T1-weighted MRI, including overall brain volume and corpus callosum morphology, I estimate this subject is male."
                }
            ],
            "metadata": {"source": "ABCD", "ground_truth": "male"}
        },

        # T2: Single subject, multiple modalities
        {
            "task_id": "T2_001",
            "task_type": "T2",
            "subject_ids": ["sub-002"],
            "modalities": ["sMRI", "dMRI"],
            "images": [
                {"path": "data/sub-002/T1w.nii.gz", "token": "<image_sMRI>", "modality": "sMRI"},
                {"path": "data/sub-002/dwi_FA.nii.gz", "token": "<image_dMRI>", "modality": "dMRI"}
            ],
            "conversation": [
                {
                    "role": "human",
                    "content": "<image_sMRI>\nHere is a T1-weighted structural MRI of a subject."
                },
                {
                    "role": "gpt",
                    "content": "I can see the T1-weighted structural MRI. The gray matter and white matter contrast is clear."
                },
                {
                    "role": "human",
                    "content": "<image_dMRI>\nNow here is the diffusion MRI (FA map) from the same subject. Describe any patterns you observe across both modalities."
                },
                {
                    "role": "gpt",
                    "content": "Comparing the structural MRI with the FA map, I observe consistent white matter integrity. The corpus callosum shows high FA values corresponding to well-defined white matter tracts in the T1 image. The overall pattern suggests normal myelination and structural organization."
                }
            ],
            "metadata": {"source": "UKB", "age": 45}
        },

        # T3: Multiple subjects comparison
        {
            "task_id": "T3_001",
            "task_type": "T3",
            "subject_ids": ["sub-003", "sub-004"],
            "modalities": ["sMRI", "sMRI"],
            "images": [
                {"path": "data/sub-003/T1w.nii.gz", "token": "<sub1-image>", "modality": "sMRI"},
                {"path": "data/sub-004/T1w.nii.gz", "token": "<sub2-image>", "modality": "sMRI"}
            ],
            "conversation": [
                {
                    "role": "human",
                    "content": "<sub1-image>\nThis is a T1-weighted MRI from Subject 1 (reference subject). Please analyze this scan."
                },
                {
                    "role": "gpt",
                    "content": "I have analyzed Subject 1's T1-weighted MRI. The brain appears to have normal structural organization with well-defined gray and white matter boundaries."
                },
                {
                    "role": "human",
                    "content": "<sub2-image>\nNow here is Subject 2's T1-weighted MRI. Compare this with Subject 1 and describe the key differences."
                },
                {
                    "role": "gpt",
                    "content": "Comparing Subject 2 with Subject 1, I observe several differences: 1) Subject 2 shows slightly larger ventricles, suggesting potential age-related or pathological changes. 2) The cortical folding patterns differ, which is normal inter-individual variation. 3) Overall brain volume appears similar. The most notable difference is the ventricular enlargement in Subject 2."
                }
            ],
            "metadata": {"comparison_type": "normal_vs_atrophy"}
        },

        # T1 with follow-up (multi-turn reasoning)
        {
            "task_id": "T1_002",
            "task_type": "T1",
            "subject_ids": ["sub-005"],
            "modalities": ["sMRI"],
            "images": [
                {"path": "data/sub-005/T1w.nii.gz", "token": "<image_sMRI>", "modality": "sMRI"}
            ],
            "conversation": [
                {
                    "role": "human",
                    "content": "<image_sMRI>\nAnalyze this T1-weighted brain MRI and estimate the subject's age."
                },
                {
                    "role": "gpt",
                    "content": "Based on the structural features in this T1-weighted MRI, including cortical thickness, ventricular size, and overall brain volume, I estimate this subject is approximately 35-40 years old."
                },
                {
                    "role": "human",
                    "content": "What specific features led you to that age estimate?"
                },
                {
                    "role": "gpt",
                    "content": "Several features informed my age estimate: 1) The ventricles show minimal enlargement, consistent with a younger adult. 2) Cortical thickness appears preserved without significant atrophy. 3) White matter shows good contrast without extensive hyperintensities. 4) The hippocampi appear normal in size. These features collectively suggest a middle-aged adult without significant age-related brain changes."
                }
            ],
            "metadata": {"ground_truth_age": 38}
        }
    ]

    with open(output_path, 'w') as f:
        json.dump(examples, f, indent=2)

    logger.info(f"Created example JSON with {len(examples)} samples at {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create example JSON
    example_path = "/tmp/umbrella_examples.json"
    create_example_json(example_path)

    print(f"\nExample JSON created at: {example_path}")
    print("This demonstrates T1, T2, T3, and multi-turn reasoning tasks.")
