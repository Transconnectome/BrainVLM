"""
T1 JSON-Based Dataset for BrainVLM

Single class for structural MRI (sMRI/T1) data loading from JSON prompts.
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Any

from monai.transforms import (
    Compose, LoadImage, AddChannel, Resize, RandAxisFlip,
    NormalizeIntensity, Randomizable, apply_transform
)
from monai.utils import MAX_SEED, get_seed

from .dataset_utils import (
    load_json,
    load_nifti,
    format_conversation,
    tokenize_conversation,
    resolve_path
)


def to_3tuple(x):
    """Convert value to 3-tuple."""
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, x, x)


class T1JSONDataset(Dataset, Randomizable):
    """
    JSON-based T1/sMRI Dataset for BrainVLM.

    This class loads structural MRI data from NIfTI files using JSON sample
    definitions that contain file paths and conversation prompts.

    Attributes:
        samples (List[Dict]): Loaded sample definitions from JSON.
        data_root (str): Root directory for data files.
        modality (str): Modality type ('sMRI' or 'T1').
        tokenizer: HuggingFace tokenizer for text processing.
        image_processor: MONAI transforms for image processing.
        max_seq_length (int): Maximum token sequence length.
        img_size (Tuple[int, int, int]): Target image size.
        mode (str): 'train' or 'eval' mode.
    """

    def __init__(
        self,
        json_file: str,
        data_root: str,
        modality: str = 'sMRI',
        tokenizer=None,
        image_processor=None,
        max_seq_length: int = 128,
        img_size: int = 128,
        mode: str = 'train',
        add_context: bool = False,
        **kwargs
    ):
        """
        Initialize the T1 JSON dataset.

        Args:
            json_file: Path to JSON file with sample definitions.
                Expected format:
                {
                    "task_id": "A",
                    "subject_id": "sub-XXXX",
                    "modality_paths": {
                        "image_sMRI": "/path/to/T1.nii.gz"
                    },
                    "conversations": [
                        {"from": "human", "value": "..."},
                        {"from": "gpt", "value": "..."}
                    ]
                }
            data_root: Root directory for data files.
            modality: Modality type (default: 'sMRI').
            tokenizer: HuggingFace tokenizer for text processing.
            image_processor: Optional custom image processor. If None, uses MONAI.
            max_seq_length: Maximum token sequence length.
            img_size: Target image size (single value or 3-tuple).
            mode: 'train' or 'eval' mode for augmentation.
            add_context: Whether to add demographic context to prompts.
            **kwargs: Additional arguments stored as attributes.
        """
        super().__init__()

        self.json_file = json_file
        self.data_root = data_root
        self.modality = modality
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.img_size = to_3tuple(img_size)
        self.mode = mode
        self.add_context = add_context

        # Store additional kwargs
        for name, value in kwargs.items():
            setattr(self, name, value)

        # Load samples from JSON
        self.samples = load_json(json_file)

        # Set up image transforms
        if image_processor is not None:
            self.image_transform = image_processor
        else:
            self.image_transform = self._define_image_augmentation(mode)

        # MONAI image loader
        self.image_loader = LoadImage(
            reader=None,
            image_only=True,
            dtype=np.float32
        )

        # Set random state for reproducibility
        self.set_random_state(seed=get_seed())
        self._seed = 0

    def _define_image_augmentation(self, mode: str = 'train') -> Compose:
        """
        Define MONAI image augmentation pipeline.

        Args:
            mode: 'train' for training augmentations, 'eval' for evaluation.

        Returns:
            MONAI Compose transform pipeline.
        """
        if mode == 'train':
            transform = Compose([
                AddChannel(),
                Resize(self.img_size),
                RandAxisFlip(prob=0.5),
                NormalizeIntensity()
            ])
        else:
            transform = Compose([
                AddChannel(),
                Resize(self.img_size),
                NormalizeIntensity()
            ])
        return transform

    def randomize(self, data=None) -> None:
        """Set random seed for augmentation reproducibility."""
        self._seed = self.R.randint(MAX_SEED, dtype='uint32')

    def _load_and_process_image(self, image_file: str) -> torch.Tensor:
        """
        Load and process a T1 image.

        Args:
            image_file: Path to NIfTI file.

        Returns:
            Processed image tensor of shape (1, H, W, D).
        """
        # Load image using MONAI
        image = self.image_loader(image_file)

        # Apply transforms
        if self.image_transform is not None:
            if isinstance(self.image_transform, Randomizable):
                self.image_transform.set_random_state(seed=self._seed)
            image = apply_transform(
                self.image_transform,
                image,
                map_items=False
            )

        # Convert to tensor
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image)

        return image

    def _process_text(
        self,
        conversations: List[Dict[str, str]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str]:
        """
        Process conversations into instruction and answer.

        Args:
            conversations: List of conversation turns.
            metadata: Optional metadata for context.

        Returns:
            Tuple of (instruction, answer) strings.
        """
        instruction, answer = format_conversation(
            conversations,
            include_image_token=True,
            modality=self.modality
        )

        # Add context if requested
        if self.add_context and metadata:
            context_parts = []
            if 'sex' in metadata:
                sex_val = metadata['sex']
                if isinstance(sex_val, (int, float)):
                    sex_str = 'male' if int(sex_val) == 1 else 'female'
                else:
                    sex_str = str(sex_val)
                context_parts.append(f"Sex: {sex_str}")

            if 'age' in metadata:
                context_parts.append(f"Age: {metadata['age']}")

            if context_parts:
                context_str = " ".join(context_parts)
                instruction = instruction.replace(
                    "ASSISTANT:",
                    f"{context_str} ASSISTANT:"
                )

        return instruction, answer

    def __preprocess_as_hf__(
        self,
        image: torch.Tensor,
        inst: str,
        answer: str
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Preprocess data for HuggingFace model input.

        Args:
            image: T1 image tensor.
            inst: Instruction text.
            answer: Answer text.

        Returns:
            Dict with pixel_values, input_ids, attention_mask, labels.
        """
        inputs = {
            'pixel_values': {},
            'input_ids': {},
            'attention_mask': {},
            'labels': {}
        }

        # Set image
        modality_key = 'T1' if 'smri' in self.modality.lower() or 't1' in self.modality.lower() else self.modality
        inputs['pixel_values'][modality_key] = image

        # Tokenize text
        if self.tokenizer is not None:
            token_dict = tokenize_conversation(
                inst, answer, self.tokenizer, self.max_seq_length
            )
            inputs['input_ids'][modality_key] = token_dict['input_ids']
            inputs['attention_mask'][modality_key] = token_dict['attention_mask']
            inputs['labels'][modality_key] = token_dict['labels']

        return inputs

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def _get_single_item(self, index: int) -> Dict[str, Any]:
        """
        Get a single-subject sample (original behavior).

        Args:
            index: Sample index.

        Returns:
            Dict with pixel_values, input_ids, attention_mask, labels, subject_id.
        """
        # Randomize for augmentation
        self.randomize()

        sample = self.samples[index]

        # Extract sample information
        subject_id = sample.get('subject_id', f'subject_{index}')
        task_id = sample.get('task_id', '')
        conversations = sample.get('conversations', [])
        metadata = sample.get('metadata', {})

        # Get image path
        modality_paths = sample.get('modality_paths', {})
        image_path = None
        for key in modality_paths:
            if 'smri' in key.lower() or 't1' in key.lower():
                image_path = modality_paths[key]
                break

        if image_path is None:
            raise ValueError(f"No sMRI path found in sample {index}: {modality_paths}")

        # Resolve path
        image_path = resolve_path(image_path, self.data_root)

        # Load and process image
        image = self._load_and_process_image(image_path)

        # Process text
        inst, answer = self._process_text(conversations, metadata)

        # Format for HuggingFace
        inputs = self.__preprocess_as_hf__(image, inst, answer)

        # Add metadata
        inputs['subject_id'] = subject_id
        inputs['task_id'] = task_id
        inputs['metadata'] = metadata

        return inputs

    def _get_multi_subject_sequential(self, index: int) -> Dict[str, Any]:
        """
        Process multi-subject comparison as sequential multi-turn conversation.
        
        Each subject appears as a separate image in its corresponding turn.
        The model uses attention to compare subjects across turns.

        Args:
            index: Sample index.

        Returns:
            Dict with pixel_values list, tokenized multi-turn conversation.
        """
        self.randomize()
        sample = self.samples[index]
        
        # Extract data
        subject_ids = sample.get('subject_id', [])
        modality_paths = sample.get('modality_paths', {})
        conversations = sample.get('conversations', [])
        metadata = sample.get('metadata', {})
        task_id = sample.get('task_id', '')
        
        # Get sMRI paths for all subjects
        smri_paths = None
        for key in modality_paths:
            if 'smri' in key.lower() or 't1' in key.lower():
                smri_paths = modality_paths[key]
                break
        
        if smri_paths is None:
            raise ValueError(f"No sMRI paths found in sample {index}")
        
        if not isinstance(smri_paths, list):
            raise ValueError(
                f"Expected list of sMRI paths for multi-subject, got {type(smri_paths)}"
            )
        
        if len(smri_paths) != len(subject_ids):
            raise ValueError(
                f"Number of paths ({len(smri_paths)}) != subjects ({len(subject_ids)})"
            )
        
        # Load all subject images
        images = []
        for i, path in enumerate(smri_paths):
            try:
                resolved_path = resolve_path(path, self.data_root)
                img = self._load_and_process_image(resolved_path)
                images.append(img)
            except Exception as e:
                raise RuntimeError(f"Failed to load image for subject {subject_ids[i]}: {e}")
        
        # Format conversation with multi-image placeholders
        formatted_inst, formatted_answer = self._format_multi_image_conversation(conversations)
        
        # Prepare output
        modality_key = 'T1'
        inputs = {
            'pixel_values': {modality_key: images},  # List of (1, H, W, D)
            'input_ids': {},
            'attention_mask': {},
            'labels': {},
            'num_images': len(images),
            'subject_ids': subject_ids
        }
        
        # Tokenize
        if self.tokenizer is not None:
            token_dict = tokenize_conversation(
                formatted_inst, formatted_answer, self.tokenizer, self.max_seq_length
            )
            inputs['input_ids'][modality_key] = token_dict['input_ids']
            inputs['attention_mask'][modality_key] = token_dict['attention_mask']
            inputs['labels'][modality_key] = token_dict['labels']
        
        inputs['task_id'] = task_id
        inputs['metadata'] = metadata
        
        return inputs

    def _format_multi_image_conversation(
        self,
        conversations: List[Dict[str, str]]
    ) -> Tuple[str, str]:
        """
        Format multi-turn conversation, converting subject-specific image
        placeholders to standard tokens.
        
        Converts: <sub1-image>, <sub2-image>, etc. → <image_sMRI>
        
        Args:
            conversations: List of conversation turns.
            
        Returns:
            Tuple of (instruction, answer) strings with standardized image tokens.
        """
        import re
        
        instruction_parts = []
        answer = ""
        
        for turn in conversations:
            role = turn.get('from', '').lower()
            value = turn.get('value', '')
            
            # Replace subject-specific placeholders with standard image token
            value = re.sub(r'<sub\d+-image>', '<image_sMRI>', value)
            
            if role == 'human':
                instruction_parts.append(value)
            elif role in ['gpt', 'assistant']:
                answer = value
        
        instruction = " ".join(instruction_parts)
        return instruction, answer

    def _extract_inst_answer_multi_turn(
        self,
        formatted_text: str
    ) -> Tuple[str, str]:
        """
        Extract instruction and answer from multi-turn formatted text.
        
        For multi-turn, everything up to last response is "instruction"
        and the last response is "answer".
        
        Args:
            formatted_text: Combined formatted text from all turns.
            
        Returns:
            Tuple of (instruction, answer) strings.
        """
        last_gpt_idx = formatted_text.rfind("gpt:")
        if last_gpt_idx == -1:
            last_gpt_idx = formatted_text.rfind("assistant:")
        
        if last_gpt_idx == -1:
            return formatted_text, ""
        
        instruction = formatted_text[:last_gpt_idx + 4]
        answer = formatted_text[last_gpt_idx + 4:].strip()
        
        return instruction, answer

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get a sample by index.
        
        Routes to either single-subject or multi-subject handling based on
        whether subject_id is a string or list.

        Args:
            index: Sample index.

        Returns:
            Dict with pixel_values, input_ids, attention_mask, labels, subject_id.
        """
        sample = self.samples[index]
        subject_id = sample.get('subject_id')
        
        # Route to appropriate handler
        if isinstance(subject_id, list):
            return self._get_multi_subject_sequential(index)
        else:
            return self._get_single_item(index)


class T1JSONDatasetRaw(T1JSONDataset):
    """
    T1 JSON Dataset with raw output format.

    Returns dict with:
        - image: T1 image tensor
        - subject_id: Subject identifier
        - task_id: Task identifier
        - instruction: Formatted instruction
        - answer: Answer text
        - metadata: Additional metadata
    """

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get a sample by index in raw format.

        Args:
            index: Sample index.

        Returns:
            Dict with image, subject_id, task_id, instruction, answer, metadata.
        """
        # Randomize for augmentation
        self.randomize()

        sample = self.samples[index]

        # Extract sample information
        subject_id = sample.get('subject_id', f'subject_{index}')
        task_id = sample.get('task_id', '')
        conversations = sample.get('conversations', [])
        metadata = sample.get('metadata', {})

        # Get image path
        modality_paths = sample.get('modality_paths', {})
        image_path = None
        for key in modality_paths:
            if 'smri' in key.lower() or 't1' in key.lower():
                image_path = modality_paths[key]
                break

        if image_path is None:
            raise ValueError(f"No sMRI path found in sample {index}")

        # Resolve path
        image_path = resolve_path(image_path, self.data_root)

        # Load and process image
        image = self._load_and_process_image(image_path)

        # Process text
        inst, answer = self._process_text(conversations, metadata)

        return {
            'image': image,
            'subject_id': subject_id,
            'task_id': task_id,
            'instruction': inst,
            'answer': answer,
            'metadata': metadata
        }


# Factory function
def create_t1_dataset(
    json_file: str,
    data_root: str,
    output_format: str = 'hf',
    **kwargs
) -> T1JSONDataset:
    """
    Factory function to create a T1 dataset.

    Args:
        json_file: Path to JSON file with sample definitions.
        data_root: Root directory for data files.
        output_format: Output format ('hf' for HuggingFace, 'raw' for raw).
        **kwargs: Additional arguments passed to dataset class.

    Returns:
        Instantiated T1 dataset object.
    """
    if output_format == 'raw':
        return T1JSONDatasetRaw(
            json_file=json_file,
            data_root=data_root,
            **kwargs
        )
    else:
        return T1JSONDataset(
            json_file=json_file,
            data_root=data_root,
            **kwargs
        )


# Export
__all__ = [
    'T1JSONDataset',
    'T1JSONDatasetRaw',
    'create_t1_dataset',
]
