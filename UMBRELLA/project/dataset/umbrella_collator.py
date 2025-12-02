"""
UMBRELLA Collator: Flexible Batch Collation for Heterogeneous Multi-Image Samples

Handles:
- Variable number of images per sample
- Different task types (T1, T2, T3) in same batch
- Proper padding and masking
- Image token position tracking
- Dynamic batch composition
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class UMBRELLABatch:
    """
    Batch container for UMBRELLA training.

    Supports heterogeneous batches with variable images per sample.

    CRITICAL: Implements __len__() and __iter__() for HuggingFace Trainer compatibility.
    The Trainer's _prepare_inputs() method expects inputs to support len() check.
    """

    # Core tensors
    pixel_values: torch.Tensor  # (batch, max_images, C, H, W, D) or (batch, max_images, C, H, W, D, T) for 4D
    input_ids: torch.Tensor     # (batch, max_seq_len)
    attention_mask: torch.Tensor  # (batch, max_seq_len)
    labels: torch.Tensor        # (batch, max_seq_len) - -100 for masked tokens, >=0 for active

    # Image metadata
    image_mask: torch.Tensor    # (batch, max_images) - 1 if valid, 0 if padded
    num_images_per_sample: List[int]  # Actual image count per sample

    # Task metadata
    task_types: List[str]       # Task type strings
    task_ids: torch.Tensor      # Task type as integer (0=T1, 1=T2, 2=T3)
    sample_indices: List[int]   # Original dataset indices

    # Additional metadata
    metadata: List[Dict[str, Any]] = field(default_factory=list)

    def to(self, device: torch.device) -> 'UMBRELLABatch':
        """Move all tensors to device."""
        return UMBRELLABatch(
            pixel_values=self.pixel_values.to(device),
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            labels=self.labels.to(device),
            image_mask=self.image_mask.to(device),
            num_images_per_sample=self.num_images_per_sample,
            task_types=self.task_types,
            task_ids=self.task_ids.to(device),
            sample_indices=self.sample_indices,
            metadata=self.metadata
        )

    def pin_memory(self) -> 'UMBRELLABatch':
        """Pin memory for faster GPU transfer."""
        return UMBRELLABatch(
            pixel_values=self.pixel_values.pin_memory(),
            input_ids=self.input_ids.pin_memory(),
            attention_mask=self.attention_mask.pin_memory(),
            labels=self.labels.pin_memory(),
            image_mask=self.image_mask.pin_memory(),
            num_images_per_sample=self.num_images_per_sample,
            task_types=self.task_types,
            task_ids=self.task_ids.pin_memory(),
            sample_indices=self.sample_indices,
            metadata=self.metadata
        )

    @property
    def batch_size(self) -> int:
        return self.input_ids.size(0)

    @property
    def total_images(self) -> int:
        return sum(self.num_images_per_sample)

    def get_task_distribution(self) -> Dict[str, int]:
        """Get distribution of task types in batch."""
        dist = {}
        for task_type in self.task_types:
            dist[task_type] = dist.get(task_type, 0) + 1
        return dist

    # CRITICAL FIX: HuggingFace Trainer compatibility
    def __len__(self) -> int:
        """
        Return batch size for HuggingFace Trainer compatibility.

        The Trainer's _prepare_inputs() method calls len(inputs) to check
        if the batch is empty. Without this, we get:
        TypeError: object of type 'UMBRELLABatch' has no len()

        Returns:
            Batch size (number of samples in batch)
        """
        return self.batch_size

    def __iter__(self):
        """
        Make batch iterable for dict-like access patterns.

        Some HuggingFace utilities may iterate over inputs expecting dict behavior.
        This provides compatibility by yielding field names.

        Yields:
            Field names that can be used for attribute access
        """
        # Return iterator over field names (dict-like keys)
        yield from [
            'pixel_values',
            'input_ids',
            'attention_mask',
            'labels',
            'image_mask',
            'num_images_per_sample',
            'task_types',
            'task_ids',
            'sample_indices',
            'metadata'
        ]

    def __getitem__(self, key: str):
        """
        Enable dict-like access for HuggingFace compatibility.

        Args:
            key: Field name

        Returns:
            Field value

        Raises:
            KeyError: If field doesn't exist
        """
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError(f"UMBRELLABatch has no field '{key}'")

    def keys(self):
        """Return field names like a dict."""
        return [
            'pixel_values',
            'input_ids',
            'attention_mask',
            'labels',
            'image_mask',
            'num_images_per_sample',
            'task_types',
            'task_ids',
            'sample_indices',
            'metadata'
        ]

    def values(self):
        """Return field values like a dict."""
        return [
            self.pixel_values,
            self.input_ids,
            self.attention_mask,
            self.labels,
            self.image_mask,
            self.num_images_per_sample,
            self.task_types,
            self.task_ids,
            self.sample_indices,
            self.metadata
        ]

    def items(self):
        """Return (key, value) pairs like a dict."""
        return zip(self.keys(), self.values())


class UMBRELLACollator:
    """
    Collate function for UMBRELLA batches with variable images per sample.

    Handles:
    - Padding images to max in batch
    - Creating image masks for valid positions
    - Tokenizing and padding text
    - Task type encoding
    """

    # Task type to integer mapping
    TASK_TYPE_MAP = {'T1': 0, 'T2': 1, 'T3': 2}

    def __init__(self,
                 tokenizer=None,
                 img_size=None,
                 max_seq_length: int = 2048,
                 max_images: int = 10,
                 pad_to_max_images: bool = False):
        """
        Initialize collator.

        Args:
            tokenizer: Text tokenizer
            img_size: Image spatial dimension(s) - can be:
                - int: single value for 3D isotropic (e.g., 128 → [128, 128, 128])
                - list/tuple: explicit shape (e.g., [120, 120, 120] or [96, 96, 96, 24])
            max_seq_length: Maximum sequence length
            max_images: Maximum images per sample
            pad_to_max_images: Pad all batches to max_images (vs max in batch)
        """
        self.tokenizer = tokenizer

        # Handle both int and list-type image sizes
        if isinstance(img_size, (list, tuple)):
            self.img_size = img_size
        elif isinstance(img_size, int):
            self.img_size = [img_size, img_size, img_size]  # Default to 3D isotropic
        elif img_size is None:
            self.img_size = [128, 128, 128]  # Default fallback
        else:
            self.img_size = [128, 128, 128]  # Default fallback

        self.max_seq_length = max_seq_length
        self.max_images = max_images
        self.pad_to_max_images = pad_to_max_images

    def __call__(self, batch: List[Dict[str, Any]]) -> UMBRELLABatch:
        """
        Collate a batch of samples.

        Args:
            batch: List of sample dicts from UMBRELLADataset

        Returns:
            UMBRELLABatch with padded tensors
        """
        if not batch:
            return self._empty_batch()

        batch_size = len(batch)

        # Extract metadata
        task_types = [item.get('task_type', 'T1') for item in batch]
        sample_indices = [item.get('sample_index', -1) for item in batch]
        num_images_per_sample = [item.get('num_images', 1) for item in batch]

        # Determine image padding size
        if self.pad_to_max_images:
            max_imgs_in_batch = self.max_images
        else:
            max_imgs_in_batch = max(num_images_per_sample)

        # Prepare pixel values with proper padding
        pixel_values_list = []
        image_mask_list = []

        for i, item in enumerate(batch):
            # Get pixel values
            pv = item.get('pixel_values', {})
            if isinstance(pv, dict) and 'T1' in pv:
                images = pv['T1']
            elif isinstance(pv, torch.Tensor):
                images = pv
            else:
                # Fallback to zeros with shape from img_size
                # img_size can be [H, W, D] (3D) or [H, W, D, T] (4D)
                fallback_shape = tuple([1, 1] + self.img_size)
                images = torch.zeros(fallback_shape)

            # Ensure minimum 5D: (num_images, C, H, W, ...)
            # Handle both 3D and 4D images
            if images.dim() == 3:
                # 3D image (H, W, D) → add channel and batch dims
                images = images.unsqueeze(0).unsqueeze(0)
            elif images.dim() == 4:
                # Either (C, H, W, D) or (num_images, H, W, D)
                # Assume (num_images, H, W, D) and add channel dimension
                images = images.unsqueeze(1)
            elif images.dim() == 5:
                # Already (num_images, C, H, W, D) - OK
                pass
            elif images.dim() == 6:
                # 4D images (num_images, C, H, W, D, T) - OK for 4D fMRI
                pass

            actual_num_imgs = min(images.size(0), num_images_per_sample[i])

            # Create padded tensor for this sample
            # Shape: (max_imgs_in_batch, C, H, W, ...) where ... matches img_size dimensions
            padded_shape = [max_imgs_in_batch, images.size(1)] + self.img_size
            padded = torch.zeros(padded_shape, dtype=images.dtype)

            # Copy actual images (handle dimension mismatches)
            if actual_num_imgs > 0:
                # Get actual spatial dimensions from images
                actual_spatial_dims = images.shape[2:]
                expected_spatial_dims = tuple(self.img_size)

                if actual_spatial_dims == expected_spatial_dims:
                    # Dimensions match, copy directly
                    padded[:actual_num_imgs] = images[:actual_num_imgs]
                else:
                    # Dimensions differ - use actual image spatial dims, don't resize
                    # This preserves variable-sized images
                    logger.warning(
                        f"Image dimension mismatch: expected {expected_spatial_dims}, "
                        f"got {actual_spatial_dims}. Using actual image dimensions."
                    )
                    # For now, copy what fits
                    min_dims = tuple(min(a, e) for a, e in zip(actual_spatial_dims, expected_spatial_dims))
                    # Create slicing tuple
                    slices = (slice(None), slice(None)) + tuple(slice(0, d) for d in min_dims)
                    padded[tuple(list(range(actual_num_imgs)) + [slice(None)] * len(padded.shape))] = images[:actual_num_imgs][slices]

            pixel_values_list.append(padded)

            # Create mask
            mask = torch.zeros(max_imgs_in_batch, dtype=torch.bool)
            mask[:actual_num_imgs] = True
            image_mask_list.append(mask)

        # Stack pixel values and masks
        pixel_values = torch.stack(pixel_values_list)
        image_mask = torch.stack(image_mask_list)

        # Prepare text data
        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for item in batch:
            ids = item.get('input_ids')
            mask = item.get('attention_mask')
            label = item.get('labels')

            # Convert to tensor if needed
            if not isinstance(ids, torch.Tensor):
                ids = torch.tensor(ids, dtype=torch.long)
            if not isinstance(mask, torch.Tensor):
                mask = torch.tensor(mask, dtype=torch.long)
            if not isinstance(label, torch.Tensor):
                label = torch.tensor(label, dtype=torch.long)

            # Pad or truncate to max_seq_length
            ids = self._pad_or_truncate(ids, self.max_seq_length, pad_value=0)
            mask = self._pad_or_truncate(mask, self.max_seq_length, pad_value=0)
            label = self._pad_or_truncate(label, self.max_seq_length, pad_value=-100)

            input_ids_list.append(ids)
            attention_mask_list.append(mask)
            labels_list.append(label)

        # Stack text tensors
        input_ids = torch.stack(input_ids_list)
        attention_mask = torch.stack(attention_mask_list)
        labels = torch.stack(labels_list)

        # Encode task types
        task_ids = torch.tensor(
            [self.TASK_TYPE_MAP.get(t, 0) for t in task_types],
            dtype=torch.long
        )

        # Collect metadata
        metadata = [item.get('metadata', {}) for item in batch]

        return UMBRELLABatch(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            image_mask=image_mask,
            num_images_per_sample=num_images_per_sample,
            task_types=task_types,
            task_ids=task_ids,
            sample_indices=sample_indices,
            metadata=metadata
        )

    def _pad_or_truncate(self, tensor: torch.Tensor, target_length: int,
                         pad_value: int = 0) -> torch.Tensor:
        """Pad or truncate tensor to target length."""
        current_length = tensor.size(0)

        if current_length < target_length:
            padding = torch.full(
                (target_length - current_length,),
                pad_value,
                dtype=tensor.dtype
            )
            return torch.cat([tensor, padding])
        else:
            return tensor[:target_length]

    def _empty_batch(self) -> UMBRELLABatch:
        """Create an empty batch."""
        # Create pixel_values shape: (0, max_images, C, H, W, ...) where ... matches img_size
        pixel_values_shape = [0, self.max_images, 1] + self.img_size
        return UMBRELLABatch(
            pixel_values=torch.zeros(pixel_values_shape),
            input_ids=torch.zeros(0, self.max_seq_length, dtype=torch.long),
            attention_mask=torch.zeros(0, self.max_seq_length, dtype=torch.long),
            labels=torch.zeros(0, self.max_seq_length, dtype=torch.long),
            image_mask=torch.zeros(0, self.max_images, dtype=torch.bool),
            num_images_per_sample=[],
            task_types=[],
            task_ids=torch.zeros(0, dtype=torch.long),
            sample_indices=[],
            metadata=[]
        )


class MemoryAwareUMBRELLACollator(UMBRELLACollator):
    """
    Extended collator with memory-aware batching support.

    Works with MemoryAwareBatchSampler to construct memory-efficient batches.
    """

    def __init__(self, *args, memory_budget_gb: float = 30.0, **kwargs):
        """
        Initialize memory-aware collator.

        Args:
            memory_budget_gb: GPU memory budget in GB
            *args, **kwargs: Passed to parent
        """
        super().__init__(*args, **kwargs)
        self.memory_budget_gb = memory_budget_gb

        # Memory estimation per image (GB)
        self.per_image_memory = 0.28
        self.base_overhead = 0.27  # Model + tokenizer

    def estimate_batch_memory(self, batch: List[Dict[str, Any]]) -> float:
        """
        Estimate GPU memory for a batch.

        Args:
            batch: List of sample dicts

        Returns:
            Estimated memory in GB
        """
        if not batch:
            return 0.0

        total_images = sum(item.get('num_images', 1) for item in batch)
        return self.base_overhead + total_images * self.per_image_memory

    def __call__(self, batch: List[Dict[str, Any]]) -> UMBRELLABatch:
        """
        Collate with memory validation.

        Logs warning if batch exceeds memory budget.
        """
        estimated_memory = self.estimate_batch_memory(batch)

        if estimated_memory > self.memory_budget_gb:
            logger.warning(
                f"Batch memory ({estimated_memory:.2f}GB) exceeds budget "
                f"({self.memory_budget_gb:.2f}GB). Consider reducing batch size."
            )

        return super().__call__(batch)


class ImageTokenReplacer:
    """
    Utility for replacing image tokens with actual embeddings during forward pass.

    Used to substitute <image_sMRI>, <sub1-image>, etc. with encoded image features.
    """

    # Standard token patterns
    TOKEN_PATTERNS = [
        r'<image_(\w+)>',   # <image_sMRI>, <image_fMRI>
        r'<sub(\d+)-image>',  # <sub1-image>, <sub2-image>
        r'<image>',         # Generic <image>
    ]

    def __init__(self, tokenizer, image_token_id: Optional[int] = None):
        """
        Initialize token replacer.

        Args:
            tokenizer: Text tokenizer
            image_token_id: Token ID for image placeholder (if known)
        """
        self.tokenizer = tokenizer

        # Try to find image token ID
        if image_token_id is not None:
            self.image_token_id = image_token_id
        else:
            # Try common patterns
            for token in ['<image>', '[IMG]', '<img>']:
                tokens = tokenizer.encode(token, add_special_tokens=False)
                if tokens:
                    self.image_token_id = tokens[0]
                    break
            else:
                self.image_token_id = None

    def get_image_token_positions(self, input_ids: torch.Tensor) -> List[int]:
        """
        Find positions of image tokens in sequence.

        Args:
            input_ids: Token IDs (1D tensor)

        Returns:
            List of token positions
        """
        if self.image_token_id is None:
            return []

        positions = (input_ids == self.image_token_id).nonzero(as_tuple=True)[0]
        return positions.tolist()

    def create_image_position_mask(self, batch: UMBRELLABatch) -> torch.Tensor:
        """
        Create mask indicating image token positions for entire batch.

        Args:
            batch: UMBRELLABatch

        Returns:
            Boolean tensor (batch_size, max_seq_len) with True at image positions
        """
        batch_size, seq_len = batch.input_ids.shape
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)

        if self.image_token_id is not None:
            mask = batch.input_ids == self.image_token_id

        return mask


def collate_for_generation(batch: List[Dict[str, Any]],
                          tokenizer,
                          img_size: int = 128,
                          max_seq_length: int = 2048) -> UMBRELLABatch:
    """
    Collate function for inference/generation mode.

    Removes labels and optimizes for autoregressive generation.

    Args:
        batch: List of sample dicts
        tokenizer: Text tokenizer
        img_size: Image dimension
        max_seq_length: Max sequence length

    Returns:
        UMBRELLABatch optimized for generation
    """
    collator = UMBRELLACollator(
        tokenizer=tokenizer,
        img_size=img_size,
        max_seq_length=max_seq_length
    )

    collated = collator(batch)

    # For generation, we don't need labels
    # Set all labels to -100 (ignored)
    collated.labels = torch.full_like(collated.labels, -100)

    return collated


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    collator = UMBRELLACollator(
        tokenizer=None,  # Would be actual tokenizer
        img_size=128,
        max_seq_length=2048
    )

    # Create dummy batch
    dummy_batch = [
        {
            'pixel_values': {'T1': torch.randn(2, 1, 128, 128, 128)},
            'input_ids': torch.randint(0, 1000, (500,)),
            'attention_mask': torch.ones(500),
            'labels': torch.randint(0, 1000, (500,)),
            'task_type': 'T2',
            'sample_index': 0,
            'num_images': 2,
            'metadata': {'task_id': 'T2_001'}
        },
        {
            'pixel_values': {'T1': torch.randn(1, 1, 128, 128, 128)},
            'input_ids': torch.randint(0, 1000, (300,)),
            'attention_mask': torch.ones(300),
            'labels': torch.randint(0, 1000, (300,)),
            'task_type': 'T1',
            'sample_index': 1,
            'num_images': 1,
            'metadata': {'task_id': 'T1_001'}
        }
    ]

    # Collate
    batch = collator(dummy_batch)

    print(f"Batch size: {batch.batch_size}")
    print(f"Batch length (via __len__): {len(batch)}")  # Test __len__ method
    print(f"Pixel values shape: {batch.pixel_values.shape}")
    print(f"Input IDs shape: {batch.input_ids.shape}")
    print(f"Image mask: {batch.image_mask}")
    print(f"Task distribution: {batch.get_task_distribution()}")
