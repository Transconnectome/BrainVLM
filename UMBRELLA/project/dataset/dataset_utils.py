"""
Dataset Utilities for BrainVLM

Common utility functions for loading and processing neuroimaging data.
"""

import os
import json
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

import nibabel as nib


def load_json(json_file: str) -> List[Dict[str, Any]]:
    """
    Load a JSON file containing dataset samples.

    Args:
        json_file: Path to JSON file with sample definitions.

    Returns:
        List of sample dictionaries with keys:
            - task_id: Task identifier
            - subject_id: Subject identifier
            - modality_paths: Dict of modality to file paths
            - conversations: List of conversation turns

    Raises:
        FileNotFoundError: If JSON file does not exist.
        json.JSONDecodeError: If JSON is malformed.
    """
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"JSON file not found: {json_file}")

    with open(json_file, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    # Handle both list and single sample formats
    if isinstance(samples, dict):
        samples = [samples]

    return samples


def load_nifti(file_path: str, dtype: np.dtype = np.float32) -> np.ndarray:
    """
    Load a NIfTI file (.nii or .nii.gz).

    Args:
        file_path: Path to NIfTI file.
        dtype: Data type for output array.

    Returns:
        NumPy array of image data.

    Raises:
        FileNotFoundError: If file does not exist.
        nibabel.filebasedimages.ImageFileError: If file is not valid NIfTI.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"NIfTI file not found: {file_path}")

    img = nib.load(file_path)
    data = img.get_fdata(dtype=dtype)

    return data


def load_pt_frames(
    subject_path: str,
    start_frame: int,
    num_frames: int,
    stride: int = 1,
    shuffle: bool = False
) -> torch.Tensor:
    """
    Load fMRI frames from .pt files.

    Args:
        subject_path: Directory containing frame_*.pt files.
        start_frame: Starting frame index.
        num_frames: Number of frames to load.
        stride: Frame skip factor.
        shuffle: If True, randomly shuffle frame order.

    Returns:
        Tensor of shape (1, H, W, D, T) where T is temporal dimension.

    Raises:
        FileNotFoundError: If subject directory or frames don't exist.
    """
    import random

    if not os.path.exists(subject_path):
        raise FileNotFoundError(f"Subject directory not found: {subject_path}")

    # Get total available frames
    available_frames = len([f for f in os.listdir(subject_path)
                           if f.startswith('frame_') and f.endswith('.pt')])

    if shuffle:
        # Randomly sample frames from all available
        frame_indices = random.sample(
            list(range(available_frames)),
            min(num_frames // stride, available_frames)
        )
    else:
        # Sequential loading with stride
        frame_indices = list(range(start_frame, start_frame + num_frames, stride))

    # Load frames
    frames = []
    for frame_idx in frame_indices:
        frame_path = os.path.join(subject_path, f'frame_{frame_idx}.pt')
        if not os.path.exists(frame_path):
            raise FileNotFoundError(f"Frame not found: {frame_path}")

        frame = torch.load(frame_path, weights_only=False)
        if frame.dim() == 3:
            frame = frame.unsqueeze(0)  # Add channel dimension
        frames.append(frame)

    # Concatenate along temporal dimension
    y = torch.cat(frames, dim=-1)

    # Ensure shape is (1, H, W, D, T)
    if y.dim() == 4:
        y = y.unsqueeze(0)

    return y


def normalize_fmri(
    data: torch.Tensor,
    stats_path: str,
    method: str = 'znorm_zeroback'
) -> torch.Tensor:
    """
    Apply normalization to fMRI data using global statistics.

    Args:
        data: fMRI tensor to normalize.
        stats_path: Path to global_stats.pt file.
        method: Normalization method:
            - 'none': No normalization
            - 'minmax': Normalize to [0, 1] using global max
            - 'znorm_zeroback': Z-normalization with zero background
            - 'znorm_minback': Z-normalization (min as background)

    Returns:
        Normalized tensor.

    Raises:
        FileNotFoundError: If stats file doesn't exist.
        ValueError: If unknown normalization method.
    """
    if method == 'none':
        return data

    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Stats file not found: {stats_path}")

    stats_dict = torch.load(stats_path, weights_only=False)

    if method == 'minmax':
        data = data / stats_dict['global_max']

    elif method == 'znorm_zeroback':
        background = data == 0
        data = (data - stats_dict['global_mean']) / stats_dict['global_std']
        data[background] = 0

    elif method == 'znorm_minback':
        background = data == 0
        data = (data - stats_dict['global_mean']) / stats_dict['global_std']

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return data


def pad_tensor(
    tensor: torch.Tensor,
    padding: Tuple[int, ...],
    value: Optional[float] = None
) -> torch.Tensor:
    """
    Apply padding to a tensor with automatic background value detection.

    Args:
        tensor: Input tensor, expected shape (1, H, W, D, T) or (1, T, H, W, D).
        padding: Padding values for torch.nn.functional.pad (left, right, ...).
        value: Padding value. If None, uses first element of flattened tensor.

    Returns:
        Padded tensor.
    """
    if value is None:
        value = tensor.flatten()[0].item()

    # Permute to (1, T, H, W, D) for padding
    if tensor.shape[-1] < tensor.shape[1]:  # Temporal dim is last
        tensor = tensor.permute(0, 4, 1, 2, 3)

    # Apply padding
    tensor = torch.nn.functional.pad(tensor, padding, value=value)

    # Permute back to (1, H, W, D, T)
    tensor = tensor.permute(0, 2, 3, 4, 1)

    return tensor


def format_conversation(
    conversations: List[Dict[str, str]],
    include_image_token: bool = True,
    image_token: str = "<image>",
    modality: str = "fMRI"
) -> Tuple[str, str]:
    """
    Format a conversation list into instruction and answer strings.

    Args:
        conversations: List of dicts with 'from' and 'value' keys.
        include_image_token: Whether to include image token placeholder.
        image_token: Token to use for image placeholder.
        modality: Modality type for token naming.

    Returns:
        Tuple of (instruction, answer) strings.

    Example:
        conversations = [
            {"from": "human", "value": "What is the age?"},
            {"from": "gpt", "value": "25 years old"}
        ]
        -> ("USER: <image>\nWhat is the age? ASSISTANT: ", "25 years old")
    """
    instruction_parts = []
    answer = ""

    # Determine image token based on modality
    if modality.lower() in ['fmri', 'rsfmri']:
        modality_token = "<image_fMRI>"
    elif modality.lower() in ['smri', 't1']:
        modality_token = "<image_sMRI>"
    elif modality.lower() == 'dmri':
        modality_token = "<image_dMRI>"
    else:
        modality_token = image_token

    for i, turn in enumerate(conversations):
        role = turn.get('from', '').lower()
        value = turn.get('value', '')

        if role == 'human':
            if i == 0 and include_image_token:
                instruction_parts.append(f"USER: {modality_token}\n{value}")
            else:
                instruction_parts.append(f"USER: {value}")
        elif role in ['gpt', 'assistant']:
            if i == len(conversations) - 1:
                # Last turn is the answer
                answer = value
                instruction_parts.append("ASSISTANT: ")
            else:
                # Multi-turn: include previous assistant responses
                instruction_parts.append(f"ASSISTANT: {value}")

    instruction = " ".join(instruction_parts)

    return instruction, answer


def tokenize_conversation(
    instruction: str,
    answer: str,
    tokenizer,
    max_length: int = 128,
    mask_instruction: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Tokenize instruction-answer pair with optional instruction masking.

    Args:
        instruction: Instruction/prompt text.
        answer: Answer/response text.
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum sequence length.
        mask_instruction: If True, mask instruction tokens in labels.

    Returns:
        Dict with input_ids, attention_mask, and labels tensors.
    """
    full_text = instruction + answer

    encoding = tokenizer(
        full_text,
        add_special_tokens=True,
        padding='max_length',
        max_length=max_length,
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].squeeze(0)
    attention_mask = encoding['attention_mask'].squeeze(0)

    # Initialize labels
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    if mask_instruction:
        # Mask instruction tokens
        for variant in [" ASSISTANT:", "ASSISTANT:"]:
            try:
                assistant_tokens = tokenizer.encode(variant, add_special_tokens=False)
                assistant_tensor = torch.tensor(assistant_tokens)

                for i in range(len(input_ids) - len(assistant_tokens) + 1):
                    if torch.equal(input_ids[i:i+len(assistant_tokens)], assistant_tensor):
                        labels[:i+len(assistant_tokens)] = -100
                        break
                else:
                    continue
                break
            except Exception:
                continue

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


def get_num_frames(subject_path: str) -> int:
    """
    Get the number of available frames in a subject directory.

    Args:
        subject_path: Path to subject directory with frame_*.pt files.

    Returns:
        Number of frame files (excluding global_stats.pt and other files).
    """
    if not os.path.exists(subject_path):
        return 0

    frame_files = [f for f in os.listdir(subject_path)
                  if f.startswith('frame_') and f.endswith('.pt')]

    return len(frame_files)


def resolve_path(
    relative_path: str,
    data_root: str
) -> str:
    """
    Resolve a relative path against a data root.

    Args:
        relative_path: Relative or absolute path.
        data_root: Root directory for data.

    Returns:
        Absolute path.
    """
    if os.path.isabs(relative_path):
        return relative_path

    return os.path.join(data_root, relative_path)


class DatasetConfig:
    """Configuration container for dataset parameters."""

    def __init__(
        self,
        padding: Tuple[int, ...],
        tr: float,
        downsample_ratio: float = 1.0,
        background_method: str = 'first_element',
        input_scaling_method: str = 'znorm_zeroback',
        image_shape: Optional[Tuple[int, ...]] = None
    ):
        """
        Initialize dataset configuration.

        Args:
            padding: Padding tuple for torch.nn.functional.pad.
            tr: Repetition time in seconds.
            downsample_ratio: Temporal downsampling ratio.
            background_method: Method for determining background value.
            input_scaling_method: Normalization method.
            image_shape: Expected image shape (H, W, D).
        """
        self.padding = padding
        self.tr = tr
        self.downsample_ratio = downsample_ratio
        self.background_method = background_method
        self.input_scaling_method = input_scaling_method
        self.image_shape = image_shape

    def get_background_value(self, tensor: torch.Tensor) -> float:
        """Get background value for padding."""
        if self.background_method == 'first_element':
            return tensor.flatten()[0].item()
        elif self.background_method == 'zero':
            return 0.0
        else:
            return tensor.flatten()[0].item()


def detect_conversation_format(sample: Dict[str, Any]) -> str:
    """
    Detect whether a sample uses old (v1) or new (v2) conversation format.

    Args:
        sample: Sample dictionary from JSON file

    Returns:
        "old" if uses v1 format (conversation key, from/value structure)
        "new" if uses v2 format (conversations key, role/content structure)
        "unknown" if format unclear
    """
    # Check for old format (singular "conversation" key with "from"/"value")
    if "conversation" in sample:
        conv = sample["conversation"]
        if isinstance(conv, list) and len(conv) > 0:
            first_item = conv[0]
            if isinstance(first_item, dict) and "from" in first_item and "value" in first_item:
                return "old"

    # Check for new format (plural "conversations" key with "role"/"content")
    if "conversations" in sample:
        convs = sample["conversations"]
        if isinstance(convs, list) and len(convs) > 0:
            first_item = convs[0]
            if isinstance(first_item, dict) and "role" in first_item and "content" in first_item:
                return "new"

    return "unknown"


def normalize_conversation_roles(conversation: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize conversation roles to standard format.

    Converts old format roles (human/gpt) to new format (user/assistant).
    Handles both old structure (from/value) and new structure (role/content).

    Args:
        conversation: List of conversation turns with potential role mismatches

    Returns:
        Normalized conversation list with standardized roles
    """
    normalized = []

    for turn in conversation:
        normalized_turn = turn.copy()

        # Normalize the role field
        role = turn.get("role") or turn.get("from")
        if role:
            role_lower = role.lower().strip()
            if role_lower in ("human",):
                normalized_turn["role"] = "user"
            elif role_lower in ("gpt",):
                normalized_turn["role"] = "assistant"
            else:
                normalized_turn["role"] = role_lower

        normalized.append(normalized_turn)

    return normalized


def get_conversation_list(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract conversation list from sample, handling both v1 and v2 formats.

    Args:
        sample: Sample dictionary from JSON file

    Returns:
        List of conversation turns (normalized to new format)

    Raises:
        KeyError: If neither "conversation" nor "conversations" key exists
    """
    # Try new format first
    if "conversations" in sample:
        conversations = sample["conversations"]
    # Fall back to old format
    elif "conversation" in sample:
        conversations = sample["conversation"]
    else:
        raise KeyError(
            f"Sample missing 'conversations' or 'conversation' key. "
            f"Available keys: {list(sample.keys())}"
        )

    if not isinstance(conversations, list):
        raise ValueError(f"Conversations must be a list, got {type(conversations)}")

    return normalize_conversation_roles(conversations)


# Dataset-specific configurations
DATASET_CONFIGS = {
    'ABCD': DatasetConfig(
        padding=(0, 1, 0, 0, 0, 0),
        tr=0.8,
        downsample_ratio=2.5,
        background_method='first_element',
        image_shape=(96, 96, 95)
    ),
    'UKB': DatasetConfig(
        padding=(3, 9, 0, 0, 10, 8),
        tr=0.735,  # Varies by acquisition
        downsample_ratio=1.0,
        background_method='first_element',
        image_shape=(88, 88, 64)
    ),
    'HCP': DatasetConfig(
        padding=(3, 9, 0, 0, 10, 8),
        tr=0.72,
        downsample_ratio=1.0,
        background_method='first_element',
        image_shape=(91, 109, 91)
    ),
    'HBN_rest': DatasetConfig(
        padding=(7, 8, 1, 0, 7, 8),
        tr=0.8,  # Varies by site
        downsample_ratio=1.0,
        background_method='first_element',
        image_shape=(81, 95, 81)
    ),
    'HBN_task': DatasetConfig(
        padding=(0, 1, 0, 0, 0, 0),
        tr=0.8,
        downsample_ratio=1.0,
        background_method='first_element',
        image_shape=(96, 96, 95)
    ),
    'ABIDE': DatasetConfig(
        padding=(0, -1, -10, -9, -1, 0),
        tr=2.0,  # Varies by site
        downsample_ratio=1.0,
        background_method='first_element',
        image_shape=(97, 115, 97)
    )
}
