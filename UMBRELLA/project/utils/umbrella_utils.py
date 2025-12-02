"""
UMBRELLA Utilities: Helper Functions for Training Pipeline

Provides:
- Image loading and preprocessing
- Conversation formatting
- Token extraction and replacement
- Data validation
- Memory estimation
"""

import os
import re
import json
import torch
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

logger = logging.getLogger(__name__)


# ==============================================================================
# Image Loading Utilities
# ==============================================================================

def load_nifti_image(image_path: str, normalize: bool = True) -> np.ndarray:
    """
    Load NIfTI image from disk.

    Args:
        image_path: Path to .nii or .nii.gz file
        normalize: Apply intensity normalization

    Returns:
        3D numpy array
    """
    try:
        import nibabel as nib
        img = nib.load(image_path)
        data = img.get_fdata().astype(np.float32)

        if normalize:
            # Z-score normalization
            mean = np.mean(data)
            std = np.std(data)
            if std > 0:
                data = (data - mean) / std

        return data
    except Exception as e:
        logger.error(f"Failed to load {image_path}: {e}")
        raise


def resize_image(image: np.ndarray, target_size: Tuple[int, int, int]) -> np.ndarray:
    """
    Resize 3D image to target size.

    Args:
        image: 3D numpy array
        target_size: (H, W, D) tuple

    Returns:
        Resized image
    """
    from scipy.ndimage import zoom

    current_size = image.shape
    zoom_factors = [t / c for t, c in zip(target_size, current_size)]

    return zoom(image, zoom_factors, order=1)


def preprocess_brain_image(
    image_path: str,
    target_size: int = 128,
    normalize: bool = True,
    add_channel: bool = True
) -> torch.Tensor:
    """
    Full preprocessing pipeline for brain images.

    Args:
        image_path: Path to image
        target_size: Target spatial dimension
        normalize: Apply normalization
        add_channel: Add channel dimension

    Returns:
        Preprocessed tensor
    """
    # Load
    image = load_nifti_image(image_path, normalize=normalize)

    # Resize
    image = resize_image(image, (target_size, target_size, target_size))

    # Convert to tensor
    tensor = torch.from_numpy(image).float()

    # Add channel dimension
    if add_channel:
        tensor = tensor.unsqueeze(0)  # (1, H, W, D)

    return tensor


# ==============================================================================
# Conversation Formatting
# ==============================================================================

def format_llava_conversation(
    turns: List[Dict[str, str]],
    system_prompt: Optional[str] = None
) -> str:
    """
    Format conversation in LLaVA style.

    Args:
        turns: List of {'role': 'human'/'gpt', 'content': str}
        system_prompt: Optional system prompt

    Returns:
        Formatted conversation string
    """
    formatted = ""

    # Add system prompt if provided
    if system_prompt:
        formatted += f"SYSTEM: {system_prompt}\n\n"

    # Format each turn
    for turn in turns:
        role = turn['role']
        content = turn['content']

        if role == 'human':
            formatted += f"USER: {content}\n"
        elif role == 'gpt':
            formatted += f"ASSISTANT: {content}\n"
        else:
            logger.warning(f"Unknown role: {role}")
            formatted += f"{role.upper()}: {content}\n"

    return formatted


def create_single_turn_conversation(
    image_token: str,
    question: str,
    answer: str,
    system_context: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Create simple single-turn conversation.

    Args:
        image_token: Image token (e.g., '<image_sMRI>')
        question: Question text
        answer: Answer text
        system_context: Optional context for question

    Returns:
        List of conversation turns
    """
    if system_context:
        human_content = f"{image_token}\n{system_context}\n{question}"
    else:
        human_content = f"{image_token}\n{question}"

    return [
        {'role': 'human', 'content': human_content},
        {'role': 'gpt', 'content': answer}
    ]


def create_comparison_conversation(
    subject1_token: str,
    subject2_token: str,
    reference_analysis: str,
    comparison_question: str,
    comparison_answer: str
) -> List[Dict[str, str]]:
    """
    Create multi-subject comparison conversation.

    Args:
        subject1_token: First subject image token
        subject2_token: Second subject image token
        reference_analysis: Analysis of first subject
        comparison_question: Question for comparison
        comparison_answer: Comparison answer

    Returns:
        List of conversation turns
    """
    return [
        {
            'role': 'human',
            'content': f"{subject1_token}\nAnalyze this brain scan (Subject 1, reference)."
        },
        {
            'role': 'gpt',
            'content': reference_analysis
        },
        {
            'role': 'human',
            'content': f"{subject2_token}\n{comparison_question}"
        },
        {
            'role': 'gpt',
            'content': comparison_answer
        }
    ]


# ==============================================================================
# Token Extraction and Replacement
# ==============================================================================

def extract_image_tokens(text: str) -> List[str]:
    """
    Extract all image tokens from text.

    Supported patterns:
    - <image_sMRI>, <image_fMRI>, <image_dMRI>
    - <sub1-image>, <sub2-image>, ...
    - <image>

    Args:
        text: Input text

    Returns:
        List of found tokens
    """
    tokens = []

    # Modality tokens
    tokens.extend(re.findall(r'<image_\w+>', text))

    # Subject tokens
    tokens.extend(re.findall(r'<sub\d+-image>', text))

    # Generic tokens
    tokens.extend(re.findall(r'<image>', text))

    return tokens


def replace_image_tokens_with_placeholder(
    text: str,
    placeholder: str = "[IMAGE]"
) -> str:
    """
    Replace all image tokens with a uniform placeholder.

    Useful for tokenization where all image positions should be identical.

    Args:
        text: Input text with image tokens
        placeholder: Replacement string

    Returns:
        Text with replaced tokens
    """
    # Replace all patterns
    text = re.sub(r'<image_\w+>', placeholder, text)
    text = re.sub(r'<sub\d+-image>', placeholder, text)
    text = re.sub(r'<image>', placeholder, text)

    return text


def count_image_tokens(text: str) -> int:
    """
    Count total number of image tokens in text.

    Args:
        text: Input text

    Returns:
        Number of image tokens
    """
    return len(extract_image_tokens(text))


def validate_image_token_mapping(
    conversation: List[Dict[str, str]],
    image_paths: List[str]
) -> bool:
    """
    Validate that image tokens in conversation match provided images.

    Args:
        conversation: List of conversation turns
        image_paths: List of image paths

    Returns:
        True if valid, False otherwise
    """
    # Count tokens in conversation
    all_text = " ".join(turn['content'] for turn in conversation)
    num_tokens = count_image_tokens(all_text)

    # Check match
    if num_tokens != len(image_paths):
        logger.warning(
            f"Mismatch: {num_tokens} image tokens but {len(image_paths)} image paths"
        )
        return False

    return True


# ==============================================================================
# Memory Estimation
# ==============================================================================

def estimate_sample_memory_gb(
    num_images: int,
    img_size: int = 128,
    seq_length: int = 2048,
    is_training: bool = True
) -> float:
    """
    Estimate GPU memory for a single sample.

    Args:
        num_images: Number of images
        img_size: Image dimension
        seq_length: Sequence length
        is_training: Include gradients

    Returns:
        Estimated memory in GB
    """
    # Image memory: (num_images, 1, H, W, D) * 4 bytes
    image_bytes = num_images * 1 * img_size ** 3 * 4

    # Text memory: sequence * hidden_dim * 4 bytes
    # Assuming hidden_dim ~ 4096 (Llama-like)
    text_bytes = seq_length * 4096 * 4

    # Gradients multiply by ~2.5
    multiplier = 2.5 if is_training else 1.0

    total_bytes = (image_bytes + text_bytes) * multiplier

    # Add overhead
    overhead_bytes = 0.1 * 1e9  # 100MB

    return (total_bytes + overhead_bytes) / 1e9


def estimate_batch_memory_gb(
    task_types: List[str],
    img_size: int = 128,
    seq_length: int = 2048,
    is_training: bool = True
) -> float:
    """
    Estimate GPU memory for a batch.

    Args:
        task_types: List of task types in batch
        img_size: Image dimension
        seq_length: Sequence length
        is_training: Include gradients

    Returns:
        Estimated memory in GB
    """
    # Images per task type
    images_per_task = {'T1': 1, 'T2': 2, 'T3': 2}

    total_memory = 0.0
    for task_type in task_types:
        num_images = images_per_task.get(task_type, 1)
        total_memory += estimate_sample_memory_gb(
            num_images, img_size, seq_length, is_training
        )

    # Batch overhead
    total_memory += 0.15  # 150MB

    return total_memory


# ==============================================================================
# Data Validation
# ==============================================================================

def validate_umbrella_sample(sample: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate a single UMBRELLA sample.

    Args:
        sample: Sample dict

    Returns:
        (is_valid, list of errors)
    """
    errors = []

    # Required fields
    required = ['task_id', 'task_type', 'images', 'conversation']
    for field in required:
        if field not in sample:
            errors.append(f"Missing required field: {field}")

    # Validate task type
    if 'task_type' in sample:
        if sample['task_type'] not in ['T1', 'T2', 'T3']:
            errors.append(f"Invalid task_type: {sample['task_type']}")

    # Validate images
    if 'images' in sample:
        for i, img in enumerate(sample['images']):
            if 'path' not in img:
                errors.append(f"Image {i} missing 'path'")
            if 'token' not in img:
                errors.append(f"Image {i} missing 'token'")

    # Validate conversation
    if 'conversation' in sample:
        for i, turn in enumerate(sample['conversation']):
            if 'role' not in turn:
                errors.append(f"Turn {i} missing 'role'")
            if 'content' not in turn:
                errors.append(f"Turn {i} missing 'content'")
            if turn.get('role') not in ['human', 'gpt']:
                errors.append(f"Turn {i} invalid role: {turn.get('role')}")

        # Check alternating pattern
        roles = [t['role'] for t in sample['conversation']]
        for i in range(len(roles) - 1):
            if roles[i] == roles[i + 1]:
                errors.append(f"Turns {i} and {i+1} have same role")

    return len(errors) == 0, errors


def validate_umbrella_json(json_path: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate entire UMBRELLA JSON file.

    Args:
        json_path: Path to JSON file

    Returns:
        (is_valid, validation_report)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    report = {
        'total_samples': len(data),
        'valid_samples': 0,
        'invalid_samples': 0,
        'errors': [],
        'task_distribution': {'T1': 0, 'T2': 0, 'T3': 0}
    }

    for i, sample in enumerate(data):
        is_valid, errors = validate_umbrella_sample(sample)

        if is_valid:
            report['valid_samples'] += 1
            task_type = sample.get('task_type', 'T1')
            report['task_distribution'][task_type] += 1
        else:
            report['invalid_samples'] += 1
            report['errors'].append({
                'index': i,
                'task_id': sample.get('task_id', 'unknown'),
                'errors': errors
            })

    is_all_valid = report['invalid_samples'] == 0
    return is_all_valid, report


# ==============================================================================
# JSON Creation Utilities
# ==============================================================================

def create_t1_sample(
    task_id: str,
    subject_id: str,
    image_path: str,
    question: str,
    answer: str,
    modality: str = 'sMRI',
    metadata: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Create a T1 (single subject, single image) sample.

    Args:
        task_id: Unique task identifier
        subject_id: Subject identifier
        image_path: Path to image
        question: Question text
        answer: Answer text
        modality: Image modality
        metadata: Additional metadata

    Returns:
        Sample dict
    """
    token = f'<image_{modality}>'

    return {
        'task_id': task_id,
        'task_type': 'T1',
        'subject_ids': [subject_id],
        'modalities': [modality],
        'images': [
            {'path': image_path, 'token': token, 'modality': modality}
        ],
        'conversation': create_single_turn_conversation(
            image_token=token,
            question=question,
            answer=answer
        ),
        'metadata': metadata or {}
    }


def create_t2_sample(
    task_id: str,
    subject_id: str,
    image_paths: List[str],
    modalities: List[str],
    conversation: List[Dict[str, str]],
    metadata: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Create a T2 (single subject, multiple modalities) sample.

    Args:
        task_id: Unique task identifier
        subject_id: Subject identifier
        image_paths: List of image paths
        modalities: List of modalities
        conversation: Multi-turn conversation
        metadata: Additional metadata

    Returns:
        Sample dict
    """
    images = []
    for path, mod in zip(image_paths, modalities):
        images.append({
            'path': path,
            'token': f'<image_{mod}>',
            'modality': mod
        })

    return {
        'task_id': task_id,
        'task_type': 'T2',
        'subject_ids': [subject_id],
        'modalities': modalities,
        'images': images,
        'conversation': conversation,
        'metadata': metadata or {}
    }


def create_t3_sample(
    task_id: str,
    subject_ids: List[str],
    image_paths: List[str],
    conversation: List[Dict[str, str]],
    modality: str = 'sMRI',
    metadata: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Create a T3 (multiple subjects comparison) sample.

    Args:
        task_id: Unique task identifier
        subject_ids: List of subject identifiers
        image_paths: List of image paths (one per subject)
        conversation: Multi-turn comparison conversation
        modality: Image modality (same for all)
        metadata: Additional metadata

    Returns:
        Sample dict
    """
    images = []
    for i, path in enumerate(image_paths, 1):
        images.append({
            'path': path,
            'token': f'<sub{i}-image>',
            'modality': modality
        })

    return {
        'task_id': task_id,
        'task_type': 'T3',
        'subject_ids': subject_ids,
        'modalities': [modality] * len(subject_ids),
        'images': images,
        'conversation': conversation,
        'metadata': metadata or {}
    }


def save_umbrella_json(samples: List[Dict[str, Any]], output_path: str):
    """
    Save samples to UMBRELLA JSON format.

    Args:
        samples: List of sample dicts
        output_path: Output file path
    """
    # Validate before saving
    errors = []
    for i, sample in enumerate(samples):
        is_valid, sample_errors = validate_umbrella_sample(sample)
        if not is_valid:
            errors.extend([f"Sample {i}: {e}" for e in sample_errors])

    if errors:
        logger.warning(f"Validation errors:\n" + "\n".join(errors))

    # Save
    with open(output_path, 'w') as f:
        json.dump(samples, f, indent=2)

    logger.info(f"Saved {len(samples)} samples to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example: Create samples
    t1_sample = create_t1_sample(
        task_id="T1_001",
        subject_id="sub-001",
        image_path="data/sub-001/T1w.nii.gz",
        question="Estimate the sex of this subject.",
        answer="Based on the structural features, I estimate this subject is male.",
        metadata={'source': 'ABCD'}
    )

    print("T1 Sample created:")
    print(json.dumps(t1_sample, indent=2))

    # Validate
    is_valid, errors = validate_umbrella_sample(t1_sample)
    print(f"\nValidation: {'PASS' if is_valid else 'FAIL'}")

    # Memory estimation
    memory = estimate_sample_memory_gb(num_images=1)
    print(f"\nEstimated memory: {memory:.3f} GB")
